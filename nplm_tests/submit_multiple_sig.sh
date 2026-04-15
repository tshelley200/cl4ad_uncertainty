#!/bin/bash

source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate cl4ad_new

BASE_EMB_DIR="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/NPLM_embeddings/Run10.7/nv0.25_0.5_l0.5_lr2e-06_r1e-06"
BASE_MODEL_DIR="/n/holystore01/LABS/iaifi_lab/Users/tshelley2002/Uncertainty_Project/Run10.7/nv0.25_0.5_l0.5_lr2e-06_r1e-06/final_models"

EMB="${BASE_EMB_DIR}/val_embeddings_fold0_trimmed.npz"
MODEL="${BASE_MODEL_DIR}/linear1_10_final.pth"

WEIGHT_CLIPPING=1.96
TEST_NUM=9
TOYS=200
LOCAL=0
FIRSTSEED=0
NV_VALUES=(-0.5 -0.25 0.25 0.5)

# Signal classes and corresponding embedding files
SIGNAL_CLASSES=(
    "hChToTauNu"
    "ato4l"
    "leptoquark" 
    "hToTauTau"
)

SIGNAL_EMB_FILES=(
    "${BASE_EMB_DIR}/hChToTauNu.npz"
    "${BASE_EMB_DIR}/ato4l.npz"
    "${BASE_EMB_DIR}/leptoquark.npz" 
    "${BASE_EMB_DIR}/hToTauTau.npz"
)


for i in "${!SIGNAL_CLASSES[@]}"; do
    SIGNAL_CLASS="${SIGNAL_CLASSES[$i]}"
    SIGNAL_EMB="${SIGNAL_EMB_FILES[$i]}"

    echo "=================================================="
    echo "Running jobs for signal class: ${SIGNAL_CLASS}"
    echo "Signal embedding file: ${SIGNAL_EMB}"
    echo "=================================================="

    # # R0 signal, 100 injected signal count
    # python submit_toys_slurm_R0_Signal.py \
    #     --test-num "$TEST_NUM" \
    #     --signal-class "$SIGNAL_CLASS" \
    #     --embeddings-file "$EMB" \
    #     --signal-embeddings-file "$SIGNAL_EMB" \
    #     --shape-model "$MODEL" \
    #     --weight-clipping "$WEIGHT_CLIPPING" \
    #     --nv-values "${NV_VALUES[@]}" \
    #     --toys "$TOYS" \
    #     --local "$LOCAL" \
    #     --firstseed "$FIRSTSEED"

    # R0 signal, n-sig = 50
    python submit_toys_slurm_R0_Signal.py \
        --test-num "$TEST_NUM" \
        --signal-class "$SIGNAL_CLASS" \
        --embeddings-file "$EMB" \
        --signal-embeddings-file "$SIGNAL_EMB" \
        --shape-model "$MODEL" \
        --weight-clipping "$WEIGHT_CLIPPING" \
        --nv-values "${NV_VALUES[@]}" \
        --n-sig 200 \
        --toys "$TOYS" \
        --local "$LOCAL" \
        --firstseed "$FIRSTSEED"

    # # D signal, nuisance = -0.025
    # python submit_toys_slurm_D_Signal.py \
    #     --test-num "$TEST_NUM" \
    #     --signal-class "$SIGNAL_CLASS" \
    #     --embeddings-file "$EMB" \
    #     --shape-model "$MODEL" \
    #     --signal-embeddings-file "$SIGNAL_EMB" \
    #     --weight-clipping "$WEIGHT_CLIPPING" \
    #     --nv-values "${NV_VALUES[@]}" \
    #     --shape-nuisance-val -0.025 \
    #     --scaled-data-name gx_neg0.25 \
    #     --signal-name gx_neg0.25 \
    #     --toys "$TOYS" \
    #     --local "$LOCAL" \
    #     --firstseed "$FIRSTSEED"

    # # D signal, nuisance = +0.025
    # python submit_toys_slurm_D_Signal.py \
    #     --test-num "$TEST_NUM" \
    #     --signal-class "$SIGNAL_CLASS" \
    #     --embeddings-file "$EMB" \
    #     --shape-model "$MODEL" \
    #     --signal-embeddings-file "$SIGNAL_EMB" \
    #     --weight-clipping "$WEIGHT_CLIPPING" \
    #     --nv-values "${NV_VALUES[@]}" \
    #     --shape-nuisance-val 0.025 \
    #     --scaled-data-name gx_pos0.25 \
    #     --signal-name gx_pos0.25 \
    #     --toys "$TOYS" \
    #     --local "$LOCAL" \
    #     --firstseed "$FIRSTSEED"

    # D signal, nuisance = -0.025, n-sig = 50
    python submit_toys_slurm_D_Signal.py \
        --test-num "$TEST_NUM" \
        --signal-class "$SIGNAL_CLASS" \
        --embeddings-file "$EMB" \
        --shape-model "$MODEL" \
        --signal-embeddings-file "$SIGNAL_EMB" \
        --weight-clipping "$WEIGHT_CLIPPING" \
        --nv-values "${NV_VALUES[@]}" \
        --n-sig 200 \
        --shape-nuisance-val -0.025 \
        --scaled-data-name gx_neg0.25 \
        --signal-name gx_neg0.25 \
        --toys "$TOYS" \
        --local "$LOCAL" \
        --firstseed "$FIRSTSEED"

    # D signal, nuisance = +0.025, n-sig = 50
    python submit_toys_slurm_D_Signal.py \
        --test-num "$TEST_NUM" \
        --signal-class "$SIGNAL_CLASS" \
        --embeddings-file "$EMB" \
        --shape-model "$MODEL" \
        --signal-embeddings-file "$SIGNAL_EMB" \
        --weight-clipping "$WEIGHT_CLIPPING" \
        --nv-values "${NV_VALUES[@]}" \
        --n-sig 200 \
        --shape-nuisance-val 0.025 \
        --scaled-data-name gx_pos0.25 \
        --signal-name gx_pos0.25 \
        --toys "$TOYS" \
        --local "$LOCAL" \
        --firstseed "$FIRSTSEED"

done

# python submit_toys_slurm_R0_Bkg.py \
#     --test-num 9 \
#     --embeddings-file "$EMB" \
#     --shape-model "$MODEL" \
#     --weight-clipping 1.9 \
#     --n-ref 104000 \
#     --n-bkg 20000 \
#     --nv-values -0.5 -0.25 0.25 0.5 \
#     --toys 200 \
#     --local 0 \
#     --firstseed 0

# python submit_toys_slurm_D_Bkg.py \
#     --test-num 9 \
#     --embeddings-file "$EMB" \
#     --shape-model "$MODEL" \
#     --weight-clipping 1.9 \
#     --n-ref 104000 \
#     --n-bkg 20000 \
#     --nv-values -0.5 -0.25 0.25 0.5 \
#     --shape-nuisance-val -0.025 \
#     --scaled-data-name gx_neg0.25 \
#     --toys 200 \
#     --local 0 \
#     --firstseed 0

# python submit_toys_slurm_D_Bkg.py \
#     --test-num 9 \
#     --embeddings-file "$EMB" \
#     --shape-model "$MODEL" \
#     --weight-clipping 1.9 \
#     --n-ref 104000 \
#     --n-bkg 20000 \
#     --nv-values -0.5 -0.25 0.25 0.5 \
#     --shape-nuisance-val 0.025 \
#     --scaled-data-name gx_pos0.25 \
#     --toys 200 \
#     --local 0 \
#     --firstseed 0