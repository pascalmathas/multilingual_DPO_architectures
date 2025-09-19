"""
This script translates a column of text in a DataFrame using a vLLM-hosted model.

It supports using different models for different languages, specifically for Arabic.
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from datetime import datetime
import os
import argparse
import gc
import torch
from vllm import LLM, SamplingParams

def initialize_model(model_name_for_print: str, model_repo_id: str, max_len: int = 4096) -> LLM:
    """Initializes the vLLM model."""
    os.environ['VLLM_USE_V1'] = '0'

    llm_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "float16",
        "gpu_memory_utilization": 0.95,
        "max_model_len": max_len,
        "swap_space": 4
    }

    print(f"\nLoading {model_name_for_print} ({model_repo_id}) ...\n")
    if torch.backends.mps.is_available():
        print("MPS backend detected. Forcing dtype to float32 for compatibility, though performance may vary.")
        llm_kwargs["dtype"] = "float32"

    model = LLM(model=model_repo_id, **llm_kwargs)
    return model

def unload_model(model: Optional[LLM]):
    """Unloads the model from memory."""
    if model is None:
        return
    print("\nUnloading model from memory...")
    del model
    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            pass

def build_prompt_cohere(src_lang_name: str, tgt_lang_name: str, text: str) -> str:
    """Builds a prompt for the Cohere model."""
    user_message_content = (
        f"Translate the following text from {src_lang_name} into {tgt_lang_name}.\n"
        f"{src_lang_name}: {text}\n"
        f"{tgt_lang_name}:"
    )

    prompt = (
        "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        f"{user_message_content}"
        "<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    )
    return prompt

def translate_batch(llm: LLM, prompts_batch: List[str], max_tokens: int = 768, temperature: float = 0.0) -> list:
    """Translates a batch of prompts."""
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(prompts=prompts_batch, sampling_params=sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def stream_batch_translate_cohere_multimodel(
    df_path: str,
    rows: int,
    output_path: str,
    column: str,
    src_lang_name: str,
    languages: Dict[str, str],
    general_model_name: str,
    general_model_repo: str,
    arabic_model_name: str,
    arabic_model_repo: str,
    batch_size: int = 32,
    export: bool = True,
    max_model_len_config: int = 4096,
    max_output_tokens: int = 768,
    temperature_setting: float = 0.0
):
    """Translates a DataFrame column using multiple Cohere models."""
    print(f"--- Cohere Multimodel Translation Job ---")
    print(f"General Model: {general_model_name} ({general_model_repo})")
    print(f"Arabic Model: {arabic_model_name} ({arabic_model_repo})")
    print(f"DataFrame path: {df_path}")
    print(f"Rows to translate: {rows if rows != 0 else 'All'}")
    print(f"Column to translate: {column}")
    print(f"Source language: {src_lang_name}")
    target_lang_names = [name for name in languages.keys() if name != src_lang_name]
    print(f"Target language(s): {', '.join(target_lang_names)}")
    print(f"Batch size: {batch_size}")
    print(f"Max model length: {max_model_len_config}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Temperature: {temperature_setting}")
    print(f"Export: {export}")
    print(f"-----------------------------------")

    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {df_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading DataFrame: {e}")
        return pd.DataFrame()

    if rows != 0:
        print(f'Original df size: {df.shape}')
        df = df.head(rows)
        print(f'Using first {rows} rows. New df size: {df.shape}')

    if not target_lang_names:
        print("Warning: No target languages specified (excluding source language).")
        return pd.DataFrame()

    all_translations: List[Dict[str, Any]] = []

    loaded_model_details: Dict[str, Any] = {"repo_id": None, "llm_instance": None, "name_for_print": None}

    try:
        arabic_target_name = "Arabic"

        langs_general_model = [lang for lang in target_lang_names if lang != arabic_target_name]
        langs_arabic_model = [lang for lang in target_lang_names if lang == arabic_target_name]

        ordered_target_langs = langs_general_model + langs_arabic_model

        for target_language_name in ordered_target_langs:
            current_model_repo_to_use: str
            current_model_name_to_use: str

            if target_language_name == arabic_target_name:
                current_model_repo_to_use = arabic_model_repo
                current_model_name_to_use = arabic_model_name
            else:
                current_model_repo_to_use = general_model_repo
                current_model_name_to_use = general_model_name

            if loaded_model_details["repo_id"] != current_model_repo_to_use:
                if loaded_model_details["llm_instance"] is not None:
                    unload_model(loaded_model_details["llm_instance"])

                llm = initialize_model(current_model_name_to_use, current_model_repo_to_use, max_len=max_model_len_config)
                loaded_model_details["llm_instance"] = llm
                loaded_model_details["repo_id"] = current_model_repo_to_use
                loaded_model_details["name_for_print"] = current_model_name_to_use

            current_llm = loaded_model_details["llm_instance"]
            if current_llm is None:
                print(f"ERROR: LLM instance is None for language {target_language_name}. Skipping.")
                continue

            model_name_for_output = loaded_model_details["name_for_print"]

            for i in tqdm(range(0, len(df), batch_size), desc=f"Translating to {target_language_name} with {model_name_for_output}"):
                batch_df = df.iloc[i:i+batch_size]
                batch_source_texts = batch_df[column].tolist()

                prompts_batch = [build_prompt_cohere(src_lang_name, target_language_name, str(text))
                                 for text in batch_source_texts]

                if prompts_batch:
                    try:
                        batch_translations = translate_batch(current_llm, prompts_batch, max_tokens=max_output_tokens, temperature=temperature_setting)
                        for original_idx, translation in enumerate(batch_translations):
                            original_text = batch_source_texts[original_idx]
                            all_translations.append({
                                'org_prompt': original_text,
                                'trs_prompt': translation,
                                'org_lang': src_lang_name,
                                'tar_lang': target_language_name,
                                'tar_lang_code': languages[target_language_name],
                                'model': model_name_for_output,
                            })
                    except Exception as batch_error:
                         print(f"\nError during translation batch for lang {target_language_name} with model {model_name_for_output}: {batch_error}")
                         for original_text in batch_source_texts:
                             all_translations.append({
                                 'org_prompt': original_text,
                                 'trs_prompt': f"ERROR: {str(batch_error)}",
                                 'org_lang': src_lang_name,
                                 'tar_lang': target_language_name,
                                 'tar_lang_code': languages[target_language_name],
                                 'model': model_name_for_output,
                             })

    except Exception as e:
        print(f"ERROR: Critical error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if loaded_model_details["llm_instance"]:
            unload_model(loaded_model_details["llm_instance"])
            print(f"\nFinished processing. Last model ({loaded_model_details['name_for_print']}) unloaded.")

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

            output_filename = f"translated_prompts_cohere_multilang_{date_str}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            final_df.to_pickle(output_filepath)
            print(f"\nSaved all results to {output_filepath}")
        except Exception as e:
            print(f"\nError saving results to pickle file: {e}")

    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using vLLM-hosted Cohere models.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--column", type=str, default='source', help="Name of the column to translate.")
    parser.add_argument("--src_lang_name", type=str, default='English', help="Source language name.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation.")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length.")
    parser.add_argument("--max_output_tokens", type=int, default=768, help="Maximum output tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--general_model_name", type=str, default="c4ai-command-r7b-gen", help="Name of the general model.")
    parser.add_argument("--general_model_repo", type=str, default="CohereLabs/c4ai-command-r7b-12-2024", help="Repo of the general model.")
    parser.add_argument("--arabic_model_name", type=str, default="c4ai-command-r7b-ar", help="Name of the Arabic model.")
    parser.add_argument("--arabic_model_repo", type=str, default="CohereLabs/c4ai-command-r7b-arabic-02-2025", help="Repo of the Arabic model.")
    args = parser.parse_args()

    LANGUAGES = {
        "English": "en",
        "German": "de",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese": "zh",
        "Spanish (Mexico)": "es",
        "Japanese": "ja",
        "Icelandic": "is",
        "Arabic": "ar"
    }

    if not os.path.exists(args.df_path):
        print(f"Warning: Input file {args.df_path} not found. Creating a dummy DataFrame for testing.")
        dummy_data = {
            args.column: [
                "Hello, how are you today?",
                "This is a test sentence.",
                "Machine translation is a fascinating field of study.",
                "Let's see how this model performs on various languages.",
                "The quick brown fox jumps over the lazy dog."
            ]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_pickle(args.df_path)
        print(f"Dummy DataFrame saved to {args.df_path}")
        if args.rows == 0:
            args.rows = len(dummy_df)

    translations_df = stream_batch_translate_cohere_multimodel(
        df_path=args.df_path,
        rows=args.rows,
        output_path=args.output_path,
        column=args.column,
        src_lang_name=args.src_lang_name,
        languages=LANGUAGES,
        general_model_name=args.general_model_name,
        general_model_repo=args.general_model_repo,
        arabic_model_name=args.arabic_model_name,
        arabic_model_repo=args.arabic_model_repo,
        batch_size=args.batch_size,
        export=args.export,
        max_model_len_config=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
        temperature_setting=args.temperature
    )

    if not translations_df.empty:
        print("\n--- Translation Summary ---")
        print(f"Generated {len(translations_df)} translations.")

        successful_translations_df = translations_df[~translations_df['trs_prompt'].str.startswith("ERROR:")]
        error_translations_df = translations_df[translations_df['trs_prompt'].str.startswith("ERROR:")]

        print(f"Successfully generated {len(successful_translations_df)} translations.")
        if len(error_translations_df) > 0:
            print(f"Encountered {len(error_translations_df)} errors during translation.")
            print("Sample errors:")
            print(error_translations_df.head(3))

        if not successful_translations_df.empty:
            print("\nSample successful translations:")
            print(successful_translations_df.head(5))
            print("\nTarget language distribution (successful):")
            print(successful_translations_df['tar_lang'].value_counts())
            print("\nModel used (successful):")
            print(successful_translations_df['model'].value_counts())
        else:
            print("\nNo successful translations were generated.")

    else:
        print("\nTranslation process completed, but no data was generated or returned.")

if __name__ == "__main__":
    main()
