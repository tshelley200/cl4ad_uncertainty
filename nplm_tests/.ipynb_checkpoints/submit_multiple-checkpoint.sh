#!/bin/bash

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate cl4ad_new

EMB="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/NPLM_embeddings/Run10.7/nv0.25_0.5_l0.5_lr2e-06_r1e-06/val_embeddings_fold0_trimmed.npz"
MODEL="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/Run10.7/nv0.25_0.5_l0.5_lr2e-06_r1e-06/final_models/linear1_10_final.pth"

python submit_toys_slurm_R0_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.85 \
    --n-ref 156000 \
    --n-bkg 30000 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --toys 200 \
    --local 0 \
    --firstseed 0

python submit_toys_slurm_R0_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.87 \
    --n-ref 156000 \
    --n-bkg 30000 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --toys 200 \
    --local 0 \
    --firstseed 0

python submit_toys_slurm_R0_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.96 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --toys 100 \
    --local 0 \
    --firstseed 100

python submit_toys_slurm_D_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.85 \
    --n-ref 156000 \
    --n-bkg 30000 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --shape-nuisance-val -0.025 \
    --scaled-data-name gx_neg0.25 \
    --toys 200 \
    --local 0 \
    --firstseed 0

python submit_toys_slurm_D_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.85 \
    --n-ref 156000 \
    --n-bkg 30000 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --shape-nuisance-val 0.025 \
    --scaled-data-name gx_pos0.25 \
    --toys 200 \
    --local 0 \
    --firstseed 0

python submit_toys_slurm_D_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.96 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --shape-nuisance-val 0.025 \
    --scaled-data-name gx_pos0.25 \
    --toys 100 \
    --local 0 \
    --firstseed 100

python submit_toys_slurm_D_Bkg.py \
    --test-num 9 \
    --embeddings-file "$EMB" \
    --shape-model "$MODEL" \
    --weight-clipping 1.96 \
    --nv-values -0.5 -0.25 0.25 0.5 \
    --shape-nuisance-val -0.025 \
    --scaled-data-name gx_neg0.25 \
    --toys 100 \
    --local 0 \
    --firstseed 100