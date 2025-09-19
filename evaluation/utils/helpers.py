import pandas as pd
from typing import List, Dict, Any
import os
import gc
import torch
import json
from vllm import LLM

def load_jsonl_to_dict(filepath: str, key_cols: List[str] = ['original_id', 'language']):
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: File not found at {filepath}")
        return None
    print(f"Loading data from: {filepath}")
    
    loaded_count, skipped_count = 0, 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                record = None
                try:
                    record = json.loads(line)
                    key_values = [record.get(col) for col in key_cols]
                    if any(v is None for v in key_values) or not all(col in record for col in key_cols):
                        skipped_count += 1
                        continue
                    
                    final_key = tuple(str(k) if isinstance(k, (list, dict)) else k for k in key_values)  # noqa
                    
                    data_dict[final_key] = record
                    loaded_count += 1
                except json.JSONDecodeError:
                    skipped_count += 1
                except Exception as e_rec:
                    skipped_count += 1
    except Exception as e_file:
        print(f"Error reading file {filepath}: {e_file}")
        return None
    
    print(f"Successfully loaded {loaded_count} records from {filepath}. Skipped {skipped_count} records.")
    return data_dict if loaded_count > 0 else None

def unload_model(model):
    """Unloads the model and clears CUDA cache."""
    print("\nUnloading model from memory...")
    if model is not None:
        if hasattr(model, 'llm_engine') and hasattr(model.llm_engine, 'model_executor'):
            if hasattr(model.llm_engine.model_executor, 'driver_worker'):
                try:
                    del model.llm_engine.model_executor.driver_worker
                except Exception as e:
                    print(f"Note: Could not delete driver_worker: {e}")
            if hasattr(model.llm_engine.model_executor, 'shutdown'):
                try:
                    model.llm_engine.model_executor.shutdown()
                except Exception as e:
                    print(f"Note: Could not explicitly shutdown model_executor: {e}")
        del model
    
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    print("Model unloaded and CUDA cache cleared.")

def initialize_model(model_repo_id: str, model_name_for_print: str = None):
    """
    Initializes the vLLM model from a local path or Hugging Face Hub ID.
    """
    if model_name_for_print is None:
        model_name_for_print = model_repo_id
    
    os.environ['VLLM_USE_V1'] = '0'
    
    llm_kwargs = {
        "trust_remote_code": True,
        "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
        "swap_space": 4,
        "enforce_eager": False
    }
    
    print(f"\nLoading {model_name_for_print} (from: {model_repo_id}) with VLLM_USE_V1='{os.getenv('VLLM_USE_V1')}' ...\n")
    model = LLM(model=model_repo_id, **llm_kwargs)
    print(f"Model {model_name_for_print} loaded successfully from {model_repo_id}.")
    return model

def build_prompt_aya(prompt_text: str) -> str:
    """Builds a prompt string in ChatML format for Aya models."""
    return (
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt_text.strip()}<|END_OF_TURN_TOKEN|>\n"
        f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    )

def build_prompt_llama3(prompt_text: str, system_prompt: str) -> str:
    """
    Builds a prompt string in Llama-3's instruction format, including a system prompt
    to enforce responding in the same language as the user.
    """
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_text.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )