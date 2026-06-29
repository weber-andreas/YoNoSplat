import argparse
import glob
import io
import os
import re

import matplotlib.pyplot as plt
import torch
from PIL import Image

CONTEXT_VIEWS = [32, 64, 128]
POSE_LABELS = {"GTPose": "GT Poses", "PredPose": "Pred Poses"}
INTRIN_LABELS = {"GTIntrin": "GT Intrinsics", "PredIntrin": "Pred Intrinsics"}

CELL_SIZE = 2.8  # inches per grid cell (square, matching 224x224 output)


def crop_to_square(img: Image.Image) -> Image.Image:
    """Center-crop to square so GT matches the square output images."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def discover_combinations(outputs_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (pose_key, intrin_key) pairs that have all three context-view runs."""
    combos = set()
    for name in os.listdir(outputs_dir):
        for pose in POSE_LABELS:
            for intrin in INTRIN_LABELS:
                if pose in name and intrin in name:
                    combos.add((pose, intrin))
    # Keep only combos where all three context-view runs exist
    valid = []
    for pose, intrin in sorted(combos):
        runs = [_run_name(c, pose, intrin) for c in CONTEXT_VIEWS]
        if all(os.path.isdir(os.path.join(outputs_dir, r)) for r in runs):
            valid.append((pose, intrin))
    return valid


def _run_name(n_ctx: int, pose: str, intrin: str) -> str:
    return f"scannetpp_c{n_ctx}_t16_{pose}_{intrin}"


def build_scene_to_torch(dataset_dir: str) -> dict[str, str]:
    scene_map = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.torch"))):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            key = data[0]["key"]
            scene_map[key] = path
        except Exception:
            pass
    return scene_map


def load_gt_image(torch_path: str, frame_idx: int) -> Image.Image:
    data = torch.load(torch_path, map_location="cpu", weights_only=False)
    d = data[0]
    timestamps = d["timestamps"].tolist()
    idx = timestamps.index(frame_idx)
    raw = d["images"][idx].numpy().tobytes()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def find_common_frame(runs: list[str], scenes: list[str], outputs_dir: str) -> str:
    candidate_sets = []
    for scene in scenes:
        frames_per_run = []
        for run in runs:
            color_dir = os.path.join(outputs_dir, run, scene, "color")
            frames_per_run.append(
                set(os.listdir(color_dir)) if os.path.isdir(color_dir) else set()
            )
        common = frames_per_run[0].intersection(*frames_per_run[1:])
        candidate_sets.append(common)
    global_common = candidate_sets[0].intersection(*candidate_sets[1:])
    if not global_common:
        raise RuntimeError("No common frame found across all scenes and runs.")
    return sorted(global_common)[len(global_common) // 2]


def render_grid(
    runs: list[str],
    col_labels: list[str],
    scenes: list[str],
    frame_name: str,
    scene_to_torch: dict,
    outputs_dir: str,
    title: str,
    output_path: str,
) -> None:
    frame_idx = int(os.path.splitext(frame_name)[0])
    n_cols = 1 + len(runs)
    n_rows = len(scenes)
    all_col_labels = ["Ground Truth"] + col_labels

    left_margin = 0.01
    right_margin = 0.84
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(CELL_SIZE * n_cols, CELL_SIZE * n_rows), squeeze=False
    )
    grid_top = 0.86
    fig.subplots_adjust(
        top=grid_top,
        bottom=0.01,
        left=left_margin,
        right=right_margin,
        hspace=0.03,
        wspace=0.03,
    )

    fig.text(
        left_margin, 0.95, title, fontsize=14, fontweight="bold", va="top", ha="left"
    )

    col_header_y = grid_top + 0.005
    col_width = (right_margin - left_margin) / n_cols
    for col_i, label in enumerate(all_col_labels):
        col_center_x = left_margin + (col_i + 0.5) * col_width
        fig.text(
            col_center_x,
            col_header_y,
            label,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="center",
        )

    for row_i, scene in enumerate(scenes):
        torch_path = scene_to_torch.get(scene)
        gt_img = load_gt_image(torch_path, frame_idx) if torch_path else None
        if torch_path is None:
            print(f"  Warning: no torch file for {scene}")

        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_i == 0:
                if gt_img is not None:
                    ax.imshow(crop_to_square(gt_img))
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                img_path = os.path.join(
                    outputs_dir, runs[col_i - 1], scene, "color", frame_name
                )
                if os.path.isfile(img_path):
                    ax.imshow(Image.open(img_path).convert("RGB"))
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

        axes[row_i, -1].annotate(
            f"{scene}\n{os.path.splitext(frame_name)[0]}",
            xy=(1.03, 0.5),
            xycoords="axes fraction",
            fontsize=9,
            va="center",
            ha="left",
        )

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-combination ScanNet++ result grids."
    )
    p.add_argument("--outputs_dir", default="outputs/test")
    p.add_argument("--dataset_dir", default="datasets/scannetpp/test")
    p.add_argument("--n_scenes", type=int, default=5)
    p.add_argument("--frame", default=None, help="Frame filename, e.g. 000120.png")
    p.add_argument("--output_dir", default=".", help="Directory to write output PNGs")
    p.add_argument("--resolution", type=int, default=224, help="Image resolution")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combos = discover_combinations(args.outputs_dir)
    if not combos:
        raise RuntimeError(
            f"No valid pose/intrin combinations found in {args.outputs_dir}"
        )
    print(f"Found combinations: {combos}")

    print("Building scene-to-torch mapping...")
    scene_to_torch = build_scene_to_torch(args.dataset_dir)

    for pose, intrin in combos:
        runs = [_run_name(c, pose, intrin) for c in CONTEXT_VIEWS]
        col_labels = [f"{c} Views" for c in CONTEXT_VIEWS]

        # Common scenes across all three context-view runs
        scene_sets = [set(os.listdir(os.path.join(args.outputs_dir, r))) for r in runs]
        common_scenes = sorted(scene_sets[0].intersection(*scene_sets[1:]))[
            : args.n_scenes
        ]
        if not common_scenes:
            print(f"  Skipping {pose}_{intrin}: no common scenes")
            continue

        frame_name = args.frame or find_common_frame(
            runs, common_scenes, args.outputs_dir
        )
        title = (
            f"Trained on DL3DV, Evaluated on ScanNet++, Resolution: {args.resolution}\n"
            f"{POSE_LABELS[pose]}, {INTRIN_LABELS[intrin]}"
        )
        output_path = os.path.join(
            args.output_dir, f"scannetpp_comparison_{pose}_{intrin}.png"
        )

        print(f"\n[{pose}_{intrin}] scenes={len(common_scenes)} frame={frame_name}")
        render_grid(
            runs,
            col_labels,
            common_scenes,
            frame_name,
            scene_to_torch,
            args.outputs_dir,
            title,
            output_path,
        )


if __name__ == "__main__":
    main()
