#!/bin/bash
#SBATCH --account=iaifi_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 6:00:00
#SBATCH --mem=48G
#SBATCH -p shared
#SBATCH -J scaling_kfold_sig
#SBATCH -o /n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/logs/scaling_kfold/scaling_kfold_sig_%j.out
#SBATCH -e /n/home07/tshelley2002/Uncertainty_Project/cl4ad_new/logs/scaling_kfold/scaling_kfold_sig_%j.err
#SBATCH --open-mode=append

set -euo pipefail

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate cl4ad_new

echo "Scaling files at $(date)"

INPUT_NPZ="/n/holystore01/LABS/iaifi_lab/Lab/CLorca/CLORCA_data/bsm_datasets_-1.npz"
#BKG_IDS="/n/holystore01/LABS/iaifi_lab/Lab/CLorca/CLORCA_data/background_IDs_-1.npz"
SCALED_NAME="scaled_datasets_signal.npz"
OUTDIR_SCALED="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/scaled_files"
OUTDIR_HDF5="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/hdf5_files-v2"

python scale_inputfiles.py \
  "$INPUT_NPZ" \
  0.025 -0.025 \
  "$SCALED_NAME" \
  "$OUTDIR_SCALED" \
  --printing_input

# Bash array
nv_lst=(0.025 -0.025)

for nv in "${nv_lst[@]}"; do
  echo "Starting nv=${nv} at $(date)"

  SCALED_FILE="${OUTDIR_SCALED}/scaled_datasets/nv_${nv}-scaled_datasets_signal.npz"
  OUT_PREFIX="${OUTDIR_HDF5}/ADC_Delphes_scaled_${nv}_signal"

  python create_kfold_traintestfile.py \
    "$SCALED_FILE" \
    --signal \
    --output-filename "$OUT_PREFIX"
done

echo "All jobs completed at $(date)"