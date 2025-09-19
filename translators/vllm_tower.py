"""
This script translates a column of text in a DataFrame using a vLLM-hosted Tower model.
"""

import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import os
import argparse
import gc
import torch
from vllm import LLM, SamplingParams

def initialize_model(model_name_for_print, model_repo_id):
    """Initializes the vLLM model."""
    os.environ['VLLM_USE_V1'] = '0'

    llm_kwargs = {
        "trust_remote_code": True,
        "dtype": "float16",
        "gpu_memory_utilization": 0.95,
        "max_model_len": 2048,
        "swap_space": 4
    }

    print(f"\nLoading {model_name_for_print} ({model_repo_id}) ...\n")
    model = LLM(model=model_repo_id, **llm_kwargs)
    return model

def unload_model(model):
    """Unloads the model from memory."""
    print("\nUnloading model from memory...")
    del model
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()

def build_prompt_tower(src_lang_name: str, tgt_lang_name: str, text: str) -> str:
    """Builds a prompt for the Tower model."""
    return (
        "<|im_start|>user\n"
        f"Translate the following text from {src_lang_name} into {tgt_lang_name}.\n"
        f"{src_lang_name}: {text}\n"
        f"{tgt_lang_name}:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def translate_batch(llm, prompts_batch: List[str], max_tokens=768, temperature=0.0) -> list:
    """Translates a batch of prompts."""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def stream_batch_translate_tower(df_path: str,
                                 rows: int,
                                 output_path: str,
                                 column: str,
                                 src_lang_name: str,
                                 languages: Dict[str, str],
                                 model_name_arg: str,
                                 model_repo_arg: str,
                                 batch_size: int = 32,
                                 export: bool = True):
    """Translates a DataFrame column using a Tower model."""
    print(f"--- Tower Model Translation Job ---")
    print(f"Model Name: {model_name_arg}")
    print(f"Model Repository: {model_repo_arg}")
    print(f"DataFrame path: {df_path}")
    print(f"Rows to translate: {rows if rows != 0 else 'All'}")
    print(f"Column to translate: {column}")
    print(f"Source language: {src_lang_name}")
    target_lang_names = [name for name in languages.keys() if name != src_lang_name]
    print(f"Target language(s): {', '.join(target_lang_names)}")
    print(f"Batch size: {batch_size}")
    print(f"Export: {export}")
    print(f"-----------------------------------")

    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {df_path}")
        return pd.DataFrame()

    if rows != 0:
        print(f'Original df size: {df.shape}')
        df = df.head(rows)
        print(f'Using first {rows} rows. New df size: {df.shape}')

    if not target_lang_names:
        print("Warning: No target languages specified (excluding source language).")
        return pd.DataFrame()

    all_translations = []

    llm = None
    try:
        print(f"\nLoading model: {model_name_arg} ({model_repo_arg})")
        llm = initialize_model(model_name_arg, model_repo_arg)

        for i in tqdm(range(0, len(df), batch_size), desc=f"Translating with {model_name_arg}"):
            batch_df = df.iloc[i:i+batch_size]
            batch_source_texts = batch_df[column].tolist()

            for target_language_name in target_lang_names:
                messages_batch = [build_prompt_tower(src_lang_name, target_language_name, text)
                                for text in batch_source_texts]

                if messages_batch:
                    try:
                        batch_translations = translate_batch(llm, messages_batch)
                        for idx, translation in enumerate(batch_translations):
                            original_text = batch_source_texts[idx]
                            all_translations.append({
                                'org_prompt': original_text,
                                'trs_prompt': translation,
                                'org_lang': src_lang_name,
                                'tar_lang': target_language_name,
                                'model': model_name_arg,
                            })
                    except Exception as batch_error:
                         print(f"\nError during translation batch for lang {target_language_name}: {batch_error}")
                         for original_text in batch_source_texts:
                             all_translations.append({
                                 'org_prompt': original_text,
                                 'trs_prompt': f"ERROR: {batch_error}",
                                 'org_lang': src_lang_name,
                                 'tar_lang': target_language_name,
                                 'model': model_name_arg,
                             })
    except Exception as e:
        print(f"ERROR: Critical error during model initialization or processing: {e}")
    finally:
        if llm:
            unload_model(llm)
            print(f"\nFinished processing with {model_name_arg}. Model unloaded.")

    if not all_translations:
        print("\nNo translations were generated.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_translations)
    print(f"\nTotal translations collected: {len(final_df)}")

    if export:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)

            safe_model_name = model_name_arg.replace("/", "_")
            output_filename = f"translated_prompts_{safe_model_name}_{date_str}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            final_df.to_pickle(output_filepath)
            print(f"\nSaved all results to {output_filepath}")
        except Exception as e:
            print(f"\nError saving results to pickle file: {e}")

    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using a vLLM-hosted Tower model.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--column", type=str, default='source', help="Name of the column to translate.")
    parser.add_argument("--src_lang_name", type=str, default='English', help="Source language name.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation.")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    parser.add_argument("--model_name", type=str, default="tower-13b", help="Name of the model.")
    parser.add_argument("--model_repo", type=str, default="Unbabel/TowerInstruct-13B-v0.1", help="Repo of the model.")
    args = parser.parse_args()

    LANGUAGES = {
        "English": "en",
        "German": "de",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Japanese": "ja",
        "Icelandic": "is",
        "Arabic": "ar",
        "Spanish (Mexico)": "es"
    }

    translations_df = stream_batch_translate_tower(
        df_path=args.df_path,
        rows=args.rows,
        output_path=args.output_path,
        column=args.column,
        src_lang_name=args.src_lang_name,
        languages=LANGUAGES,
        model_name_arg=args.model_name,
        model_repo_arg=args.model_repo,
        batch_size=args.batch_size,
        export=args.export
    )

    if not translations_df.empty:
        print("\n--- Translation Summary ---")
        print(f"Generated {len(translations_df)} translations.")
        print("Sample translations:")
        print(translations_df.head(5))
        print("\nTarget language distribution:")
        print(translations_df['tar_lang'].value_counts())
        print("\nModel used:")
        print(translations_df['model'].value_counts())
    else:
        print("\nTranslation process completed, but no data was generated or returned.")

if __name__ == "__main__":
    main()