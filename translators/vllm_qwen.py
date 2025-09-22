"""
This script translates a column of text in a DataFrame using a vLLM-hosted Qwen model.
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

def initialize_model(model_name_for_print: str, model_repo_id: str):
    """Initializes the vLLM model."""
    llm_kwargs = {
        "trust_remote_code": True,
        "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "swap_space": 4,
        "enforce_eager": False,
    }

    print(f"\nLoading {model_name_for_print} ({model_repo_id}) ...\n")
    model = LLM(model=model_repo_id, **llm_kwargs)
    return model

def unload_model(model):
    """Unloads the model from memory."""
    print("\nUnloading model from memory...")
    if model is not None:
        if hasattr(model, 'llm_engine') and \
           hasattr(model.llm_engine, 'model_executor') and \
           hasattr(model.llm_engine.model_executor, 'driver_worker'):
            try:
                del model.llm_engine.model_executor.driver_worker
            except Exception as e:
                print(f"Note: Could not delete driver_worker: {e}")
        del model
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()

def build_prompt_qwen2_instruct(src_lang_name: str, tgt_lang_name: str, text: str) -> str:
    """Builds a prompt for the Qwen2 Instruct model."""
    system_prompt = (
        f"You are an expert multilingual translator. Your task is to accurately translate the "
        f"provided text from {src_lang_name} into {tgt_lang_name}. "
        f"Respond with only the translated text itself. Do not include any extra "
        f"dialogue, explanations, introductory phrases, or language tags in your output."
    )
    user_message = (
        f"Please translate the following text from {src_lang_name} to {tgt_lang_name}:\n"
        f"{src_lang_name}: {text}\n"
        f"{tgt_lang_name}:"
    )

    prompt = f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message.strip()}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt

def translate_batch(llm, prompts_batch: List[str], max_tokens=768, temperature=0.0) -> list:
    """Translates a batch of prompts."""
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop_tokens)
    outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def stream_batch_translate(df_path: str,
                           rows: int,
                           output_path: str,
                           column: str,
                           src_lang_name: str,
                           languages: Dict[str, str],
                           model_print_name: str,
                           model_repo_id: str,
                           batch_size: int = 4,
                           export: bool = True
                           ) -> pd.DataFrame:
    """Translates a DataFrame column using a Qwen model."""
    print(f"--- {model_print_name} Model Translation Job ---")
    print(f"Model Name: {model_print_name}")
    print(f"Model Repository: {model_repo_id}")
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
    except Exception as e:
        print(f"Error reading pickle file {df_path}: {e}")
        return pd.DataFrame()

    if rows != 0:
        print(f'Original df size: {df.shape}')
        df_to_process = df.head(rows).copy()
        print(f'Using first {rows} rows. New df size for this run: {df_to_process.shape}')
    else:
        df_to_process = df.copy()

    if not target_lang_names:
        print("Warning: No target languages specified (excluding source language).")
        return pd.DataFrame()

    if column not in df_to_process.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        return pd.DataFrame()

    all_translations = []
    llm = None
    try:
        llm = initialize_model(model_print_name, model_repo_id)

        for i in tqdm(range(0, len(df_to_process), batch_size), desc=f"Translating with {model_print_name}"):
            batch_df = df_to_process.iloc[i:i+batch_size]
            batch_source_texts = batch_df[column].astype(str).fillna("").tolist()

            for target_language_name in target_lang_names:
                current_batch_texts_to_process = [text for text in batch_source_texts if text.strip()]

                if not current_batch_texts_to_process:
                    for original_text in batch_source_texts:
                         all_translations.append({
                            'org_prompt': original_text,
                            'trs_prompt': "" if not original_text.strip() else "SKIPPED: Empty input",
                            'org_lang': src_lang_name,
                            'tar_lang': target_language_name,
                            'model': model_print_name,
                        })
                    continue

                messages_batch = [build_prompt_qwen2_instruct(src_lang_name, target_language_name, text)
                                  for text in current_batch_texts_to_process]

                try:
                    batch_translations = translate_batch(llm, messages_batch)

                    translated_idx = 0
                    for original_text in batch_source_texts:
                        if original_text.strip():
                            translation = batch_translations[translated_idx]
                            translated_idx += 1
                        else:
                            translation = ""

                        all_translations.append({
                            'org_prompt': original_text,
                            'trs_prompt': translation,
                            'org_lang': src_lang_name,
                            'tar_lang': target_language_name,
                            'model': model_print_name,
                        })
                except Exception as batch_error:
                     print(f"\nError during translation batch for lang {target_language_name} with {model_print_name}: {batch_error}")
                     for original_text in current_batch_texts_to_process:
                         all_translations.append({
                             'org_prompt': original_text,
                             'trs_prompt': f"ERROR: {batch_error}",
                             'org_lang': src_lang_name,
                             'tar_lang': target_language_name,
                             'model': model_print_name,
                         })
                     for original_text in batch_source_texts:
                         if not original_text.strip():
                             all_translations.append({
                                 'org_prompt': original_text,
                                 'trs_prompt': "",
                                 'org_lang': src_lang_name,
                                 'tar_lang': target_language_name,
                                 'model': model_print_name,
                             })
    except Exception as e:
        print(f"ERROR: Critical error during model ({model_print_name}) initialization or processing: {e}")
    finally:
        if llm:
            unload_model(llm)
            print(f"\nFinished processing with {model_print_name}. Model unloaded.")
        else:
            print(f"\nProcessing finished for {model_print_name}. Model was not loaded or already handled.")

    if not all_translations:
        print(f"\nNo translations were generated by {model_print_name}.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_translations)
    print(f"\nTotal translation records collected: {len(final_df)}")

    if export:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(output_path, exist_ok=True)

            safe_model_name = model_print_name.replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_").lower()
            output_filename = f"translated_prompts_{safe_model_name}_{date_str}.pkl"
            output_filepath = os.path.join(output_path, output_filename)

            final_df.to_pickle(output_filepath)
            print(f"\nSaved all results to {output_filepath}")
        except Exception as e:
            print(f"\nError saving results to pickle file: {e}")

    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using a vLLM-hosted Qwen model.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--column", type=str, default='source', help="Name of the column to translate.")
    parser.add_argument("--src_lang_name", type=str, default='English', help="Source language name.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for translation.")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    parser.add_argument("--model_name", type=str, default="Qwen2.5 14B Instruct", help="Name of the model.")
    parser.add_argument("--model_repo", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Repo of the model.")
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

    if not os.path.exists(args.df_path):
        print(f"Warning: {args.df_path} not found. Creating a dummy DataFrame for testing.")
        data = {args.column: [
            "Hello, how are you today? This is a test for Qwen2.5.",
            "Machine translation is a fascinating field of artificial intelligence.",
            "The weather is pleasant and sunny.",
            "This is a sample sentence in English for translation by a large language model.",
            "Trying out the Qwen2.5 14B Instruct model for multilingual tasks.",
            "One more sentence for the road to ensure everything works smoothly."
        ]}
        dummy_df = pd.DataFrame(data)
        dummy_df.to_pickle(args.df_path)
        print(f"Dummy DataFrame saved to {args.df_path}")

    translations_df = stream_batch_translate(
        df_path=args.df_path,
        rows=args.rows,
        output_path=args.output_path,
        column=args.column,
        src_lang_name=args.src_lang_name,
        languages=LANGUAGES,
        model_print_name=args.model_name,
        model_repo_id=args.model_repo,
        batch_size=args.batch_size,
        export=args.export
    )

    if not translations_df.empty:
        print("\n--- Overall Translation Summary ---")
        print(f"Generated {len(translations_df)} translation records in total.")
        print("Sample translations (first 5):")
        print(translations_df.head(5))
        if len(translations_df) > 5:
            print("\nSample translations (last 5):")
            print(translations_df.tail(5))
        print("\nTarget language distribution:")
        print(translations_df['tar_lang'].value_counts(dropna=False))
        print("\nModel used distribution:")
        print(translations_df['model'].value_counts(dropna=False))

        error_df = translations_df[translations_df['trs_prompt'].str.startswith("ERROR:", na=False)]
        if not error_df.empty:
            print("\nTranslations with errors:")
            print(error_df)
        else:
            print("\nNo errors recorded in translations.")

        skipped_df = translations_df[
            (translations_df['org_prompt'] == "") |
            (translations_df['trs_prompt'] == "SKIPPED: Empty input")
        ]
        if not skipped_df.empty:
            print("\nSkipped/Empty original prompts:")
            print(skipped_df)
        else:
            print("\nNo skipped or originally empty prompts found.")
    else:
        print("\nTranslation process completed, but no data was generated or returned overall.")

if __name__ == "__main__":
    main()
