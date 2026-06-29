#!/bin/bash
#SBATCH --job-name=yonosplat-scannetpp
#SBATCH --nodes=1
#SBATCH --constraint=GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER|GPU_GEN:BLACKWELL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/storage/user/%u/slurm/logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --comment=Testing

## Useage:
# sbatch launch_slurm.sbatch
# watch -n 2 squeue -u $USER
# scontrol show job <ID>
# sacct
# tail -n 50 -f /storage/user/weand/slurm/logs/yonosplat-scannetpp-1602918.out
# scancel <ID>

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

# ensure log directory exists (expanded at runtime)
mkdir -p /storage/user/$USER/slurm/logs


# pose_free=false : use GT pose
# use_pred_intrinsics_for_embed=false : use GT intrinsics for embedding
# save_debug_info: store gaussian scene
# trainer.limit_test_batches=5 : only process 5 scenes 
conda activate yonosplat

python -m src.main \
  +experiment=yono_scannetpp \
  mode=test \
  wandb.name=scannetpp_c128_t16_GTPoseGTIntrin_v3 \
  dataset/view_sampler@dataset.scannetpp.view_sampler=evaluation \
  dataset.scannetpp.view_sampler.index_path=assets/scannetpp_ctx_128v_tgt_16v.json \
  dataset.scannetpp.view_sampler.num_context_views=128 \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  test.align_pose=false \
  model.encoder.pose_free=false \
  model.encoder.backbone.use_pred_intrinsics_for_embed=false \
  model.decoder.prune_opacity_threshold=0.005 \
  test.save_debug_info=true #\
  # +dataset.scannetpp.test_scene_ids=[6ebe30292e_iphone]
