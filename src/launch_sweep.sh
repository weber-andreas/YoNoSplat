#!/bin/bash

# Define your grid parameters
CONTEXT_VIEWS=(32 64 128)   # The "Columns"
TEST_VIEWS=(16)          # The "Rows"
GT_POSES=("yes" "no")
GT_INTRINSICS=("yes" "no")

# Loop through the grid
for ctx in "${CONTEXT_VIEWS[@]}"; do
  for tgt in "${TEST_VIEWS[@]}"; do
    for gt_pose in "${GT_POSES[@]}"; do
      for gt_intr in "${GT_INTRINSICS[@]}"; do

        # 1. Map GT Pose boolean
        if [ "$gt_pose" == "yes" ]; then
          pose_free="false"
          pose_tag="GTPose"
        else
          pose_free="true"
          pose_tag="PredPose" # Using 'Pred' instead of 'No' for clarity
        fi

        # 2. Map GT Intrinsics boolean
        if [ "$gt_intr" == "yes" ]; then
          pred_intr="false"
          intr_tag="GTIntrin"
        else
          pred_intr="true"
          intr_tag="PredIntrin"
        fi

        # 3. Define dynamic paths and names
        index_path="assets/scannetpp_ctx_${ctx}v_tgt_${tgt}v.json" 
        wandb_name="scannetpp_c${ctx}_t${tgt}_${pose_tag}_${intr_tag}"
        job_name="yono_${ctx}x${tgt}_${pose_tag}_${intr_tag}"

        # 4. Check if the index JSON actually exists before submitting
        if [ ! -f "$index_path" ]; then
            echo "Warning: File $index_path not found. Skipping $wandb_name."
            continue
        fi

        # 5. Submit to SLURM, passing variables via --export
        sbatchm \
          --job-name="$job_name" \
          --export=ALL,WANDB_NAME="$wandb_name",INDEX_PATH="$index_path",CTX_VIEWS="$ctx",POSE_FREE="$pose_free",PRED_INTRINSICS="$pred_intr" \
          src/job_template.sbatch

        # Small delay to prevent hammering the SLURM scheduler
        sleep 0.5 
        echo "Submitted: $wandb_name"

      done
    done
  done
done