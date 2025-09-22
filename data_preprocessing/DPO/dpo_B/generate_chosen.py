"""
This script generates text using a vLLM model based on a given dataset.
"""

import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import os
import gc
import torch
import json
import argparse
from vllm import LLM, SamplingParams
from huggingface_hub import login

def initialize_model(model_path_or_repo_id: str):
    """Initializes the vLLM model."""
    llm_kwargs = {
        "model": model_path_or_repo_id,
        "trust_remote_code": True,
        "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "gpu_memory_utilization": 0.95,
        "swap_space": 4,
        "enforce_eager": False,
        "max_model_len": 8192,
    }
    print(f"\nLoading {model_path_or_repo_id} ...\n")
    model = LLM(**llm_kwargs)
    print(f"Model {model_path_or_repo_id} loaded successfully.")
    return model

def unload_model(model):
    """Unloads the model and clears CUDA cache."""
    print("\nUnloading model from memory...")
    if model is not None:
        # vLLM specific cleanup
        if hasattr(model, 'llm_engine') and model.llm_engine is not None:
            if hasattr(model.llm_engine, 'model_executor'):
                if hasattr(model.llm_engine.model_executor, 'destroy_model_parallel_rank'):
                    try:
                        model.llm_engine.model_executor.destroy_model_parallel_rank()
                    except Exception as e:
                        print(f"Note: Could not call destroy_model_parallel_rank: {e}")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded and CUDA cache cleared.")

def build_prompt(prompt_text: str) -> str:
    """Builds a prompt string in Command-R chat format."""
    system_message = "You are Command-R, a highly capable large language model developed by Cohere. Please provide a helpful and accurate response to the user's request."
    prompt_elements = [
        "<s>",
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        system_message,
        "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        prompt_text.strip(),
        "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    ]
    return "".join(prompt_elements)

def generate_batch(llm, prompts_batch: List[str], max_tokens: int, temperature: float) -> list:
    """Generates responses for a batch of prompts."""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=["<|END_OF_TURN_TOKEN|>"])
    outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def main():
    """Main function to generate text from a dataset."""
    parser = argparse.ArgumentParser(description="Generate text using a vLLM model.")
    parser.add_argument("--model_path_or_repo_id", type=str, required=True, help="Path or repo ID of the model.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file.")
    parser.add_argument("--output_filename", type=str, default="prompt_responses_chosen.json", help="Name of the output file.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Name of the prompt column in the input data.")
    parser.add_argument("--lang_column", type=str, default="language", help="Name of the language column in the input data.")
    parser.add_argument("--task_column", type=str, default="task", help="Name of the task column in the input data.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for login.")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    llm = initialize_model(args.model_path_or_repo_id)

    try:
        with open(args.input_data_path, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]
        df = pd.DataFrame(data_list)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_data_path}")
        return

    all_generations = []
    for i in tqdm(range(0, len(df), args.batch_size), desc=f"Generating responses with {args.model_path_or_repo_id}"):
        batch_df = df.iloc[i:i+args.batch_size]
        prompts = [build_prompt(text) for text in batch_df[args.prompt_column].tolist()]
        responses = generate_batch(llm, prompts, args.max_tokens, args.temperature)
        
        for idx, response in enumerate(responses):
            record = {
                'original_prompt': batch_df.iloc[idx][args.prompt_column],
                'generated_response': response,
                'language': batch_df.iloc[idx].get(args.lang_column, "UNKNOWN"),
                'task': batch_df.iloc[idx].get(args.task_column, "UNKNOWN"),
                'model': args.model_path_or_repo_id,
            }
            all_generations.append(record)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=4)

    print(f"\nSaved all results to {output_path}")
    unload_model(llm)

if __name__ == "__main__":
    main()
