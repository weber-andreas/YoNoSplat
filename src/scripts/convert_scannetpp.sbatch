#!/bin/bash
#SBATCH --job-name=scannetpp-preprocess-yonosplat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/user/%u/slurm/logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --comment=Preprocessing

## Usage:
#   sbatch src/scripts/convert_scannetpp.sbatch
#   tail -f /storage/user/$USER/slurm/logs/scannetpp-preprocess-yonosplat-<JOB_ID>.out

# --- configure before submitting ---
SCANNETPP_DIR="/storage/group/dataset_mirrors/01_incoming/scannetpp"
OUTPUT_DIR="datasets/scannetpp-preprocessed"
RESOLUTION="920x690"   # WxH; use 1920x1440 for native resolution
MIN_DIST="0.05"        # min camera translation in metres between kept frames
NUM_WORKERS=8          # should match --cpus-per-task above
MAX_SCENES=""          # leave empty to process all; set e.g. 5 for a dry run
# ------------------------------------

pwd; hostname; date
mkdir -p /storage/user/$USER/slurm/logs

python -m src.scripts.convert_scannetpp \
    --input-dir   "$SCANNETPP_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --stages test \
    --resolution  "$RESOLUTION" \
    --min-dist    "$MIN_DIST" \
    --num-workers "$NUM_WORKERS" \
    ${MAX_SCENES:+--max-scenes "$MAX_SCENES"}
