#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=GPU_GEN:AMPERE|GPU_GEN:ADA|GPU_GEN:HOPPER|GPU_GEN:BLACKWELL
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --output=/storage/user/%u/slurm/logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --comment=Testing

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

# ensure log directory exists (expanded at runtime)
mkdir -p /storage/user/$USER/slurm/logs

python -m src.main \
  +experiment=yono_scannetpp \
  mode=test \
  wandb.name=${WANDB_NAME} \
  dataset/view_sampler@dataset.scannetpp.view_sampler=evaluation \
  dataset.scannetpp.view_sampler.index_path=${INDEX_PATH} \
  dataset.scannetpp.view_sampler.num_context_views=${CTX_VIEWS} \
  checkpointing.load=pretrained_weights/dl3dv_224x224_ctx2to32.ckpt \
  test.save_image=true \
  test.align_pose=false \
  model.encoder.pose_free=${POSE_FREE} \
  model.encoder.backbone.use_pred_intrinsics_for_embed=${PRED_INTRINSICS} \
  model.decoder.prune_opacity_threshold=0.005