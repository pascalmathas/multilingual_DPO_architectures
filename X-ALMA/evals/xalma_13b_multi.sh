#!/bin/bash
#
# Runs translation for multiple language pairs, automatically selecting the correct
# X-ALMA model for each pair.
#

# --- Configuration ---
PROJECT_ROOT=$(dirname "$0")/..

HF_CACHE_DIR=${HF_CACHE_DIR:-"${PROJECT_ROOT}/cache"}
OUTPUT_DIR="${PROJECT_ROOT}/translations/x-alma-13b-multi"
DATA_PATH="${PROJECT_ROOT}/data/wmt24pp_alma_json"
ACCELERATE_CONFIG="${PROJECT_ROOT}/configs/deepspeed_eval_config_bf16.yaml"
RUN_SCRIPT="${PROJECT_ROOT}/run_llmmt.py"

# Add language pairs to this array, e.g., ("en-de" "en-zh")
TEST_PAIRS_ARRAY=("en-ar" "en-de" "en-nl" "en-ru" "en-zh" "en-ja" "en-is" "en-es")

MODEL_NAME_PREFIX="haoranxu/X-ALMA-13B-Group"

# --- Language to Model Group Mapping ---
declare -A LANG_TO_GROUP
LANG_TO_GROUP=(
    ["da"]=1 ["nl"]=1 ["de"]=1 ["is"]=1 ["no"]=1 ["sv"]=1 ["af"]=1
    ["ca"]=2 ["ro"]=2 ["gl"]=2 ["it"]=2 ["pt"]=2 ["es"]=2
    ["bg"]=3 ["mk"]=3 ["sr"]=3 ["uk"]=3 ["ru"]=3
    ["id"]=4 ["ms"]=4 ["th"]=4 ["vi"]=4 ["mg"]=4 ["fr"]=4
    ["hu"]=5 ["el"]=5 ["cs"]=5 ["pl"]=5 ["lt"]=5 ["lv"]=5
    ["ka"]=6 ["zh"]=6 ["ja"]=6 ["ko"]=6 ["fi"]=6 ["et"]=6
    ["gu"]=7 ["hi"]=7 ["mr"]=7 ["ne"]=7 ["ur"]=7
    ["az"]=8 ["kk"]=8 ["ky"]=8 ["tr"]=8 ["uz"]=8 ["ar"]=8 ["he"]=8 ["fa"]=8
)

# --- Environment Setup ---
export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"

mkdir -p "${HF_CACHE_DIR}/hub"
mkdir -p "${HF_CACHE_DIR}/datasets"
mkdir -p "${OUTPUT_DIR}"

# --- Group language pairs by model ---
declare -A MODEL_GROUP_TO_PAIRS_MAP
for PAIR in "${TEST_PAIRS_ARRAY[@]}"; do
    TGT_LANG=$(echo "$PAIR" | cut -d'-' -f2)
    MODEL_GROUP_NUM=${LANG_TO_GROUP[$TGT_LANG]}

    if [ -z "$MODEL_GROUP_NUM" ]; then
        echo "Warning: No model group for target language '$TGT_LANG' in pair '$PAIR'. Skipping."
        continue
    fi

    if [ -z "${MODEL_GROUP_TO_PAIRS_MAP[$MODEL_GROUP_NUM]}" ]; then
        MODEL_GROUP_TO_PAIRS_MAP[$MODEL_GROUP_NUM]="$PAIR"
    else
        MODEL_GROUP_TO_PAIRS_MAP[$MODEL_GROUP_NUM]="${MODEL_GROUP_TO_PAIRS_MAP[$MODEL_GROUP_NUM]},$PAIR"
    fi
done

# --- Main Execution ---
echo "Starting multilingual translations..."
echo "Output Directory: ${OUTPUT_DIR}"

for MODEL_GROUP_NUM in "${!MODEL_GROUP_TO_PAIRS_MAP[@]}"; do
    CURRENT_MODEL_NAME="${MODEL_NAME_PREFIX}${MODEL_GROUP_NUM}"
    LANGUAGE_PAIRS="${MODEL_GROUP_TO_PAIRS_MAP[$MODEL_GROUP_NUM]}"

    echo "-----------------------------------------------------"
    echo "Processing Model: ${CURRENT_MODEL_NAME}"
    echo "  Language Pairs: ${LANGUAGE_PAIRS}"

    accelerate launch --config_file "${ACCELERATE_CONFIG}" "${RUN_SCRIPT}" \
        --model_name_or_path "${CURRENT_MODEL_NAME}" \
        --cache_dir "${HF_CACHE_DIR}" \
        --do_predict \
        --low_cpu_mem_usage \
        --language_pairs "${LANGUAGE_PAIRS}" \
        --mmt_data_path "${DATA_PATH}" \
        --per_device_eval_batch_size 2 \
        --output_dir "${OUTPUT_DIR}" \
        --predict_with_generate \
        --max_new_tokens 512 \
        --max_source_length 512 \
        --bf16 \
        --seed 42 \
        --num_beams 5 \
        --overwrite_cache \
        --overwrite_output_dir \
        --chat_style

    if [ $? -ne 0 ]; then
        echo "Error during translation for model ${CURRENT_MODEL_NAME}."
    fi
done

echo "-----------------------------------------------------"
echo "All translations attempted."