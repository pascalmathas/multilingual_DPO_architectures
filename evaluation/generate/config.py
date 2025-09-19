
from evaluation.utils.helpers import (
    build_prompt_aya,
    build_prompt_gemma,
    build_prompt_llama3,
    build_prompt_mistral
)

# --- Model and Path Configuration ---
AYA_MODEL_NAME_FOR_PRINT = "Aya-23-8B"
AYA_MODEL_REPO = "CohereForAI/aya-23-8B"

GEMMA_MODEL_NAME_FOR_PRINT = "google/gemma-1.1-7b-it"
GEMMA_MODEL_PATH_OR_REPO_ID = "google/gemma-1.1-7b-it"

LLAMA3_MODEL_NAME_FOR_PRINT = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA3_MODEL_PATH_OR_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

MISTRAL_MODEL_NAME_FOR_PRINT = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_MODEL_PATH_OR_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


MODEL_CONFIGS = {
    "aya": {
        "model_name_for_print": AYA_MODEL_NAME_FOR_PRINT,
        "model_repo_id": AYA_MODEL_REPO,
        "prompt_builder": build_prompt_aya,
        "stop_tokens": ["<|END_OF_TURN_TOKEN|>", "<eos>"],
    },
    "gemma": {
        "model_name_for_print": GEMMA_MODEL_NAME_FOR_PRINT,
        "model_repo_id": GEMMA_MODEL_PATH_OR_REPO_ID,
        "prompt_builder": build_prompt_gemma,
        "stop_tokens": ["<end_of_turn>", "<eos>"],
    },
    "llama3": {
        "model_name_for_print": LLAMA3_MODEL_NAME_FOR_PRINT,
        "model_repo_id": LLAMA3_MODEL_PATH_OR_REPO_ID,
        "prompt_builder": lambda p: build_prompt_llama3(p, system_prompt="You are a helpful and respectful assistant. Please always respond in the same language as the user's prompt."),
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "mistral": {
        "model_name_for_print": MISTRAL_MODEL_NAME_FOR_PRINT,
        "model_repo_id": MISTRAL_MODEL_PATH_OR_REPO_ID,
        "prompt_builder": build_prompt_mistral,
        "stop_tokens": ["</s>", "[INST]"],
    },
}
