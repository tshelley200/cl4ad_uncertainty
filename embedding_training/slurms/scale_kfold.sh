#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH -p shared
#SBATCH -J scaling_kfold
#SBATCH -o /n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/logs/scaling_kfold/scaling_kfold_%j.out
#SBATCH -e /n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/logs/scaling_kfold/scaling_kfold_%j.err
#SBATCH --open-mode=append

set -euo pipefail

module load python/3.10.13-fasrc01
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate cl4ad_new

echo "Scaling files at $(date)"

INPUT_NPZ="/n/holystore01/LABS/iaifi_lab/Lab/CLorca/CLORCA_data/datasets_-1.npz"
BKG_IDS="/n/holystore01/LABS/iaifi_lab/Lab/CLorca/CLORCA_data/background_IDs_-1.npz"
SCALED_NAME="scaled_datasets.npz"
OUTDIR_SCALED="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/scaled_files"
OUTDIR_HDF5="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2"

python scale_inputfiles.py \
  "$INPUT_NPZ" \
  0.0125 0.025 0.0375 -0.0125 -0.025 -0.0375 \
  "$SCALED_NAME" \
  "$OUTDIR_SCALED"

# Bash array
nv_lst=(0.0125 0.025 0.0375 -0.0125 -0.025 -0.0375)

for nv in "${nv_lst[@]}"; do
  echo "Starting nv=${nv} at $(date)"

  SCALED_FILE="${OUTDIR_SCALED}/scaled_datasets/nv_${nv}-scaled_datasets.npz"
  OUT_PREFIX="${OUTDIR_HDF5}/ADC_Delphes_scaled_${nv}"

  python create_kfold_traintestfile.py \
    "$SCALED_FILE" \
    --ids "$BKG_IDS" \
    --output-filename "$OUT_PREFIX"
done

echo "All jobs completed at $(date)"