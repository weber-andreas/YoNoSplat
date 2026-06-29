import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn
from tqdm import tqdm

from .ply_export import export_ply, export_occamlgs_ply
from .types import Gaussians
from ..dataset import DatasetCfgWrapper
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, get_overlap_tag, subsample_point_cloud_views, \
    clone_batch
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_context: bool
    save_debug_info: bool
    render_chunk_size: int

    post_opt_gs: bool
    post_opt_gs_iter: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_time_skip_steps: int

    train_ignore_large_loss: float  # ignore training samples with loss larger than this value, <=0 to disable
    train_ignore_large_loss_after_steps: int  # only start filtering after this many steps
    train_ignore_large_loss_mse: float  # mse loss threshold for filtering, <=0 to disable
    train_ignore_large_loss_pose: float  # pose loss threshold for filtering, <=0 to disable


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, "t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[list[DatasetCfgWrapper] | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[list[DatasetCfgWrapper] | None] = None,
        gaussian_downsample_ratio=1.,
        gaussians_per_axis=14,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.eval_cnt = 0

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.test_scores: dict[str, dict] = {}

        self.gaussian_downsample_ratio = gaussian_downsample_ratio
        self.gaussians_per_axis = gaussians_per_axis

        self.low_opa_ratio = []

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        _, v_tgt, _, h, w = batch["target"]["image"].shape

        # Run the model.
        visualization_dump = {}
        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)

        tgt_extrinsics= batch["target"]["extrinsics"]

        output = self.decoder.forward(
            gaussians,
            tgt_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        skip_sample = False
        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                output, batch, gaussians, self.global_step,
                extra_info=visualization_dump,
            )

            # filter out large loss
            ignore_after = self.train_cfg.train_ignore_large_loss_after_steps
            if self.global_step > ignore_after and loss_fn.name == 'mse' and self.train_cfg.train_ignore_large_loss_mse > 0:
                if loss > self.train_cfg.train_ignore_large_loss_mse:
                    loss = 0.
                    skip_sample = True
                    logger.warning(f"skip large mse loss: {loss}")

            if self.global_step > ignore_after and loss_fn.name == 'pose' and self.train_cfg.train_ignore_large_loss_pose > 0:
                if hasattr(loss_fn, 'last_unweighted_loss') and loss_fn.last_unweighted_loss is not None and loss_fn.last_unweighted_loss > self.train_cfg.train_ignore_large_loss_pose:
                    loss = 0.
                    skip_sample = True
                    logger.warning(f"skip large pose loss: {loss}")

            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

            # Log sub-metrics (e.g., trans_loss and rot_loss from pose loss)
            if hasattr(loss_fn, 'last_metrics') and loss_fn.last_metrics:
                for k, v in loss_fn.last_metrics.items():
                    self.log(f"loss/{k}", v)

        # log ratio of gaussians with opacity < 0.01
        opcities = gaussians.opacities.flatten()
        ratio_opacity = (opcities < 0.01).float().mean()
        self.log(f"info/ratio_opacity<0.01", ratio_opacity)

        self.log_gaussian_status(batch["context"]["image"], gaussians, visualization_dump)

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            logger.info(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        # ignore bad samples
        if (self.train_cfg.train_ignore_large_loss > 0 and self.global_step > self.train_cfg.train_ignore_large_loss_after_steps and total_loss > self.train_cfg.train_ignore_large_loss) or skip_sample:
            logger.warning(f"Large loss detected, skip this iteration")
            return 0.00000001 * total_loss

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            logger.info(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
            )

        low_opa_ratio = (gaussians.opacities < self.decoder.prune_opacity_threshold).float().mean()
        self.low_opa_ratio.append(low_opa_ratio.item())
        print(f'All: low opacity ratio: {np.mean(self.low_opa_ratio):.4f}')
        print(f'Current: low opacity ratio: {low_opa_ratio.item():.4f}')

        if self.test_cfg.post_opt_gs:
            extrinsic = visualization_dump['c2w'].clone()
            gaussians_opt, extrinsic_opt = self.opt_gaussian_pose(batch, gaussians, extrinsic, visualization_dump['scales'], visualization_dump['rotations'])
            gaussians = gaussians_opt

        # align the target pose
        if self.test_cfg.align_pose:
            if v_tgt < self.test_cfg.render_chunk_size:
                output_align = self.test_step_align(batch, gaussians, visualization_dump["c2w"])
            else:
                output_align_img = []
                output_align_depth = []
                batch_chunk = clone_batch(batch)
                for frames_start_idx in range(0, v_tgt, self.test_cfg.render_chunk_size):
                    frames_end_idx = min(frames_start_idx + self.test_cfg.render_chunk_size, v_tgt)
                    batch_chunk["target"]["image"] = batch["target"]["image"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["extrinsics"] = batch["target"]["extrinsics"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["intrinsics"] = batch["target"]["intrinsics"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["near"] = batch["target"]["near"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["far"] = batch["target"]["far"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["index"] = batch["target"]["index"][:, frames_start_idx:frames_end_idx]

                    output_align_chunk = self.test_step_align(batch_chunk, gaussians, visualization_dump["c2w"])
                    output_align_img.append(output_align_chunk.color)
                    output_align_depth.append(output_align_chunk.depth)

                    # Clear memory
                    torch.cuda.empty_cache()
                output_align = type(output_align_chunk)  # DecoderOutput
                output_align.color = torch.cat(output_align_img, dim=1)
                output_align.depth = torch.cat(output_align_depth, dim=1)

            output = output_align
        else:
            # chunk inferencing
            output_img = []
            output_depth = []
            for frames_start_idx in range(0, v_tgt, self.test_cfg.render_chunk_size):
                frames_end_idx = min(frames_start_idx + self.test_cfg.render_chunk_size, v_tgt)
                num_calls = frames_end_idx - frames_start_idx

                with self.benchmarker.time("decoder", num_calls=num_calls):
                    output = self.decoder.forward(
                        gaussians,
                        batch["target"]["extrinsics"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["intrinsics"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["near"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["far"][:, frames_start_idx:frames_end_idx],
                        (h, w),
                    )
                output_img.append(output.color)
                output_depth.append(output.depth)

                # Clear memory
                torch.cuda.empty_cache()

            output.color = torch.cat(output_img, dim=1)
            output.depth = torch.cat(output_depth, dim=1)

        # compute scores
        if self.test_cfg.compute_scores:
            # overlap = batch["context"]["overlap"][0]
            # overlap_tag = get_overlap_tag(overlap)
            overlap_tag = None  # disable overlap tag

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            # if self.test_cfg.align_pose:
            #     rgb_pred_align = output_align.color[0]
            #     all_metrics[f"lpips_align"] = compute_lpips(rgb_gt, rgb_pred_align).mean()
            #     all_metrics[f"ssim_align"] = compute_ssim(rgb_gt, rgb_pred_align).mean()
            #     all_metrics[f"psnr_align"] = compute_psnr(rgb_gt, rgb_pred_align).mean()
            #     methods.append('align')

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

            (scene,) = batch["scene"]
            self.test_scores[scene] = {k: v.item() for k, v in all_metrics.items()}

        # # if align pose, save the aligned output
        # if self.test_cfg.align_pose:
        #     output = output_align

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_context:
            for index, color in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                save_image(color, path / scene / f"context/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            frame_str = frame_str[:80]  # avoid too long file name
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

        if self.test_cfg.save_debug_info:
            # save Gaussians, point cloud, input images with corresponding predicted depth (both local and global)
            # rnedered images and depth maps
            rgb_gt = batch["target"]["image"][0]
            rgb_pred = output.color[0]

            save_path = path / scene / f"debug_info"

            # direct depth from gaussian means (used for visualization only)
            global_depth = visualization_dump["depth"][0].squeeze()
            global_depth = vis_depth_map(global_depth.contiguous())
            local_depth = visualization_dump["local_pts"][0][..., -1].squeeze()
            local_depth = vis_depth_map(local_depth.contiguous())

            context_img = batch["context"]["image"][0]

            context_vis = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*global_depth), "Global Depth"),
                add_label(vcat(*local_depth), "Local Depth"),
            )

            target_depth = vis_depth_map(output.depth[0])
            target_vis = hcat(
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*target_depth), "Target (Depth)"),
            )

            save_image(context_vis, save_path / f"{scene}_context.png")
            save_image(target_vis, save_path / f"{scene}_target.png")

            # save depth maps
            for index, depth in zip(batch["target"]["index"][0], target_depth):
                save_image(depth, save_path / f"depth/{index:0>6}.png")

            # save Gaussians
            # Create a mask to filter the Gaussians. First, throw away Gaussians at the
            # borders, since they're generally of lower quality.
            means = rearrange(
                gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", spp=1, h=h, w=w
            )
            opacities = rearrange(
                gaussians.opacities, "() (v h w spp) -> h w spp v", spp=1, h=h, w=w
            )
            GAUSSIAN_TRIM = 8
            mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
            mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

            # filter out Gaussians with too low opacity
            mask = mask & (opacities > 0.01)

            def trim(element):
                element = rearrange(
                    element, "() (v h w spp) ... -> h w spp v ...", spp=1, h=h, w=w
                )
                return element[mask][None]

            output_Gaussian_path = save_path / f"{scene}_gaussians.ply"
            export_ply(
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                trim(visualization_dump["rotations"])[0],
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                output_Gaussian_path,
                save_sh_dc_only=True,
            )

            # save OccamLGS-compatible ply file
            occamlgs_path = save_path / f"{scene}_gaussians_occamlgs.ply"
            export_occamlgs_ply(
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                trim(visualization_dump["rotations"])[0],
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                occamlgs_path,
            )

            # save camera poses for visualization
            np.savez(
                save_path / f"{scene}_cameras.npz",
                target_extrinsics=batch["target"]["extrinsics"][0].cpu().numpy(),
                target_intrinsics=batch["target"]["intrinsics"][0].cpu().numpy(),
                target_indices=batch["target"]["index"][0].cpu().numpy(),
                context_extrinsics=batch["context"]["extrinsics"][0].cpu().numpy(),
                context_intrinsics=batch["context"]["intrinsics"][0].cpu().numpy(),
            )

            # save point cloud
            output_pc_path = save_path / f"{scene}_point_cloud.ply"
            gaussian_cts = visualization_dump["means"].squeeze(-2).cpu().numpy().reshape(-1, 3)
            colors = rearrange(gaussians.harmonics, "b (v h w) d3 d_sh -> b v h w d3 d_sh", h=h, w=w)[..., 0]
            colors = (colors + 1) / 2.
            colors = (colors * 255).cpu().numpy().astype(np.uint8).reshape(-1, 3)
            colors = np.concatenate([colors.reshape(-1, 3), (torch.ones_like(gaussians.opacities)).cpu().numpy().astype(np.uint8).reshape(-1, 1)], axis=-1)

            import trimesh
            pc = trimesh.PointCloud(vertices=gaussian_cts)
            # Set colors explicitly with the right attribute name
            pc.colors = colors

            # Export using trimesh's exporter
            result = pc.export(file_type='ply')

            # Write the bytes to a file
            with open(output_pc_path, 'wb') as f:
                f.write(result)

    def test_step_align(self, batch, gaussians, pred_camera_poses):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()

            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def opt_gaussian_pose(self, batch, gaussians, pred_poses, scales, rotations):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        with torch.set_grad_enabled(True):
            gaussians_opt = Gaussians(
                nn.Parameter(gaussians.means.clone(), requires_grad=True),
                nn.Parameter(gaussians.covariances.clone(), requires_grad=True),
                nn.Parameter(gaussians.harmonics.clone(), requires_grad=True),
                nn.Parameter(gaussians.opacities.clone(), requires_grad=True),
                gaussians.rotations,
                gaussians.scales,
            )

            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            # pose parameters
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            # gaussian parameters
            opt_params.append(
                {
                    "params": [gaussians_opt.means],
                    "lr": 0.0016,
                }
            )
            opt_params.append(
                {
                    "params": [gaussians_opt.harmonics],
                    "lr": 0.0025,
                }
            )

            post_optimizer = torch.optim.Adam(opt_params)

            extrinsics = pred_poses
            with self.benchmarker.time("optimize_gs"):
                for i in range(self.test_cfg.post_opt_gs_iter):
                    post_optimizer.zero_grad(set_to_none=True)

                    output = self.decoder.forward(
                        gaussians_opt,
                        extrinsics,
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step, use_context=True)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        post_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        return gaussians_opt, extrinsics

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        base = self.test_cfg.output_path / name
        self.benchmarker.dump(base / "benchmark.json")
        self.benchmarker.dump_memory(base / "peak_memory.json")
        self.benchmarker.summarize()

        if self.test_scores:
            for scene, scores in self.test_scores.items():
                scene_path = base / scene / "metrics.json"
                scene_path.parent.mkdir(exist_ok=True, parents=True)
                with scene_path.open("w") as f:
                    json.dump(scores, f, indent=2)

            keys = list(next(iter(self.test_scores.values())).keys())
            aggregate = {
                k: float(np.mean([s[k] for s in self.test_scores.values()]))
                for k in keys
            }
            agg_path = base / "metrics.json"
            agg_path.parent.mkdir(exist_ok=True, parents=True)
            with agg_path.open("w") as f:
                json.dump({"scenes": self.test_scores, "mean": aggregate}, f, indent=2)

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            logger.info(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )

        tgt_extrinsics= batch["target"]["extrinsics"]

        output = self.decoder.forward(
            gaussians,
            tgt_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = batch["context"]["image"][0]
        context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0]
            )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    def _log_pts3d(self, pts3d, name):
        # log the 3D points [v, h, w, d] to wandb
        pts3d = pts3d.cpu().numpy()
        pts3d_subsampled = subsample_point_cloud_views(pts3d)
        pts3d_subsampled = rearrange(pts3d_subsampled, "v h w d -> (v h w) d")
        try:
            wandb.log({f"point_cloud/{name}": wandb.Object3D(pts3d_subsampled)})
        except:
            pass

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            logger.debug(self.encoder)  # log the model to wandb log files

        if self.eval_data_cfg is not None:
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        test_datasets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )

        test_datasets = [test_datasets] if not isinstance(test_datasets, list) else test_datasets

        for test_dataset in test_datasets:
            self.benchmarker.clear_history()
            scores_dict = {}

            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("no_opt",):
                    scores_dict[score_tag][method_tag] = []

            dataset_name = test_dataset.dataset.name
            time_skip_first_n_steps = min(
                self.train_cfg.eval_time_skip_steps, test_dataset.dataset.test_len()
            )
            time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            for batch_idx, batch in tqdm(
                enumerate(test_dataset),
                total=min(test_dataset.dataset.test_len(), self.train_cfg.eval_data_length),
            ):
                if batch_idx >= self.train_cfg.eval_data_length:
                    break

                batch = self.data_shim(batch)
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)

                # Render Gaussians.
                b, v, _, h, w = batch["target"]["image"].shape
                assert b == 1
                if batch_idx < time_skip_first_n_steps:
                    time_skip_steps_dict["encoder"] += 1
                    time_skip_steps_dict["decoder"] += v

                # Render Gaussians.
                with self.benchmarker.time("encoder"):
                    gaussians = self.encoder(
                        batch["context"],
                        self.global_step,
                    )

                with self.benchmarker.time("decoder", num_calls=v):
                    output = self.decoder.forward(
                        gaussians,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                rgbs = [output.color[0]]
                tags = ["no_opt"]

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

            # summarise scores and log to logger
            for score_tag, methods in scores_dict.items():
                for method_tag, cur_scores in methods.items():
                    if len(cur_scores) > 0:
                        cur_mean = sum(cur_scores) / len(cur_scores)
                        self.log(f"test/{dataset_name}_{method_tag}_{score_tag}", cur_mean)
            # summarise run time
            logger.info(f"Evaluation Dataset: {dataset_name}")
            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(time_skip_steps_dict[tag]) :]
                logger.info(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.log(f"test/{dataset_name}_runtime_avg_{tag}", np.mean(times))
            self.benchmarker.clear_history()

            overall_eval_time = time.time() - start_t
            logger.info(f"Eval total time cost: {overall_eval_time:.3f}s")
            self.log("test/runtime_all", overall_eval_time)

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        v, _, h, w = context_images.shape
        h, w = h // 14 * self.gaussians_per_axis, w // 14 * self.gaussians_per_axis
        rb = 0
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)
        colors = colors * 0.5 + 0.5

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

    def log_gaussian_status(self, context_images, gaussians, visualization_dump):
        def log_gaussian_params(params, name):
            # params: (n, 3) or (n, 1)
            if name == "depth" and params.shape[-1] == 3:
                params = params[..., 0].unsqueeze(-1)
            if name == 'opacities' and params.shape[-1] == 3:
                params = params[..., 0].unsqueeze(-1)

            max_val = params.max(dim=0)[0]
            min_val = params.min(dim=0)[0]
            median_val = params.median(dim=0)[0]

            if params.shape[-1] == 1:
                self.log(f"gaussian/max_{name}", max_val)
                self.log(f"gaussian/min_{name}", min_val)
                self.log(f"gaussian/median_{name}", median_val)
            else:
                self.log(f"gaussian/max_x_{name}", max_val[0])
                self.log(f"gaussian/max_y_{name}", max_val[1])
                self.log(f"gaussian/max_z_{name}", max_val[2])
                self.log(f"gaussian/min_x_{name}", min_val[0])
                self.log(f"gaussian/min_y_{name}", min_val[1])
                self.log(f"gaussian/min_z_{name}", min_val[2])
                self.log(f"gaussian/median_x_{name}", median_val[0])
                self.log(f"gaussian/median_y_{name}", median_val[1])
                self.log(f"gaussian/median_z_{name}", median_val[2])

        b, v, _, h, w = context_images.shape
        gaussian_ctrs = gaussians.means
        log_gaussian_params(gaussian_ctrs.flatten(end_dim=-2), "ctrs_all")
        log_gaussian_params(gaussian_ctrs.flatten(end_dim=-2).norm(dim=-1, keepdim=True), "ctrs_all_norm")

        gaussian_ctrs_per_view = rearrange(gaussian_ctrs, "b (v hw) xyz -> b v hw xyz", v=v)
        for i in range(v):
            log_gaussian_params(gaussian_ctrs_per_view[:, i].flatten(end_dim=-2), f"ctrs_view{i}")
            log_gaussian_params(gaussian_ctrs_per_view[:, i].flatten(end_dim=-2).norm(dim=-1, keepdim=True), f"ctrs_view{i}_norm")

        log_gaussian_params(visualization_dump["scales"].flatten(end_dim=-2), "scales")
        log_gaussian_params(visualization_dump["opacities"].flatten(end_dim=-2), "opacities")

        del visualization_dump
        torch.cuda.empty_cache()

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            pass

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian" in name or "rgb_embed" in name or "intrinsics_embed" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
