#!/bin/bash
#
# Runs translation for a single language pair using a specific X-ALMA model.
#

# --- Configuration ---
# Project root, assuming this script is in a subdirectory.
PROJECT_ROOT=$(dirname "$0")/..

# Cache for Hugging Face assets.
HF_CACHE_DIR=${HF_CACHE_DIR:-"${PROJECT_ROOT}/cache"}

# Where to save generated translations.
OUTPUT_DIR="${PROJECT_ROOT}/translations/x-alma-13b"

# Model and data paths.
MODEL_NAME="haoranxu/X-ALMA-13B-Group1"
DATA_PATH="${PROJECT_ROOT}/data/wmt24"
ACCELERATE_CONFIG="${PROJECT_ROOT}/configs/deepspeed_eval_config_bf16.yaml"
RUN_SCRIPT="${PROJECT_ROOT}/run_llmmt.py"

# Language pair for translation.
LANGUAGE_PAIRS="en-de"

# --- Environment Setup ---
export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"

mkdir -p "${HF_CACHE_DIR}/hub"
mkdir -p "${HF_CACHE_DIR}/datasets"
mkdir -p "${OUTPUT_DIR}"

# --- Main Execution ---
echo "Starting translation..."
echo "  Model: ${MODEL_NAME}"
echo "  Language Pairs: ${LANGUAGE_PAIRS}"
echo "  Output Directory: ${OUTPUT_DIR}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" "${RUN_SCRIPT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --cache_dir "${HF_CACHE_DIR}" \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs "${LANGUAGE_PAIRS}" \
    --mmt_data_path "${DATA_PATH}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT_DIR}" \
    --predict_with_generate \
    --max_new_tokens 512 \
    --max_source_length 512 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --chat_style # Required for X-ALMA models

echo "Translation finished. Check output in ${OUTPUT_DIR}"