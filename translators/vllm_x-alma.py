"""
This script translates a column of text in a DataFrame using a vLLM-hosted X-ALMA model.

X-ALMA models are grouped by language, so this script will iterate through the groups
and translate the text to the languages in each group.
"""

import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import os
from collections import defaultdict
import argparse
import gc
import torch
from vllm import LLM, SamplingParams

def initialize_model(model_name, model_repo):
    """Initializes the vLLM model."""
    os.environ['VLLM_USE_V1'] = '0'

    llm_kwargs = {
        "trust_remote_code": True,
        "dtype": "float16",
        "gpu_memory_utilization": 0.95,
        "max_model_len": 4096,
        "swap_space": 4
    }

    print(f"\nLoading {model_name} ({model_repo}) ...\n")
    model = LLM(model=model_repo, **llm_kwargs)
    return model

def unload_model(model):
    """Unloads the model from memory."""
    print("\nUnloading model from memory...")
    del model
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()

def build_prompt_x_alma(src, tgt, text) -> List[dict]:
    """Builds a prompt for the X-ALMA model."""
    return [
        {
            "role": "user",
            "content": f"Translate from {src} to {tgt}:\n{src}: {text}\n{tgt}:"
        }
    ]

def translate_batch(llm, messages_batch, max_tokens=4096, temperature=0.6) -> list:
    """Translates a batch of prompts."""
    outputs = llm.chat(messages=messages_batch,
                       sampling_params=SamplingParams(max_tokens=max_tokens,
                                                       temperature=temperature),
                       )
    return [output.outputs[0].text.strip() for output in outputs]

def stream_batch_translate_xalma(df_path, rows, output_path, column, src_lang, languages, xalma_base_repo, batch_size=32, export=True):
    """Translates a DataFrame column using X-ALMA models."""
    print(f"--- X-ALMA Translation Job ---")
    print(f"DataFrame path: {df_path}")
    print(f"Rows to translate: {rows}")
    print(f"Column to translate: {column}")
    print(f"Source language: {src_lang}")
    print(f"Target language(s): {', '.join(languages.keys())}")
    print(f"X-ALMA Base Repo: {xalma_base_repo}")
    print(f"Batch size: {batch_size}")
    print(f"Export: {export}")
    print(f"------------------------------")

    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {df_path}")
        return pd.DataFrame()

    if rows != 0:
        print(f'Original df size: {df.shape}')
        df = df.head(rows)
        print(f'Using first {rows} rows. New df size: {df.shape}')

    target_languages_map = {name: code for name, code in languages.items() if name != src_lang}
    if not target_languages_map:
        print("Warning: No target languages specified (excluding source language).")
        return pd.DataFrame()

    GROUP2LANG = {
        1: ["da", "nl", "de", "is", "no", "sv", "af"],
        2: ["ca", "ro", "gl", "it", "pt", "es"],
        3: ["bg", "mk", "sr", "uk", "ru"],
        4: ["id", "ms", "th", "vi", "mg", "fr"],
        5: ["hu", "el", "cs", "pl", "lt", "lv"],
        6: ["ka", "zh", "ja", "ko", "fi", "et"],
        7: ["gu", "hi", "mr", "ne", "ur"],
        8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
    }
    lang_to_group = {code: group for group, codes in GROUP2LANG.items() for code in codes}

    xalma_target_langs_by_group = defaultdict(list)
    unsupported_target_langs = []
    for lang_name, lang_code in target_languages_map.items():
        group = lang_to_group.get(lang_code)
        if group:
            xalma_target_langs_by_group[group].append(lang_name)
        else:
            unsupported_target_langs.append(lang_name)
            print(f"Warning: Target language '{lang_name}' ({lang_code}) is not listed in the known X-ALMA groups. It will be skipped.")

    if not xalma_target_langs_by_group:
         print("Error: None of the specified target languages belong to a known X-ALMA group.")
         return pd.DataFrame()

    all_xalma_translations = []

    for group_num, group_target_langs in xalma_target_langs_by_group.items():
        if not group_target_langs:
             continue

        xalma_model_repo = f"{xalma_base_repo}-Group{group_num}"
        model_name = f"x-alma-group{group_num}"
        print(f"\nProcessing X-ALMA Group {group_num} ({', '.join(group_target_langs)}) using {xalma_model_repo}")

        llm = None
        try:
            llm = initialize_model(model_name, xalma_model_repo)
        except Exception as e:
            print(f"ERROR: Could not load X-ALMA model {xalma_model_repo}. Skipping group {group_num}. Error: {e}")
            if llm:
                unload_model(llm)
            continue

        group_translations = []
        try:
            for i in tqdm(range(0, len(df), batch_size), desc=f"X-ALMA Group {group_num} Batches"):
                batch_df = df.iloc[i:i+batch_size]
                batch_source_texts = batch_df[column].tolist()

                for target_language_name in group_target_langs:
                    messages_batch = [build_prompt_x_alma(src_lang, target_language_name, text) for text in batch_source_texts]

                    if messages_batch:
                        try:
                            batch_translations = translate_batch(llm, messages_batch)

                            for idx, translation in enumerate(batch_translations):
                                original_text = batch_source_texts[idx]
                                group_translations.append({
                                    'org_prompt': original_text,
                                    'trs_prompt': translation,
                                    'org_lang': src_lang,
                                    'tar_lang': target_language_name,
                                    'model': "x-alma",
                                })
                        except Exception as batch_error:
                             print(f"\nError during translation batch for group {group_num}, lang {target_language_name}: {batch_error}")
                             for idx in range(len(batch_source_texts)):
                                 original_text = batch_source_texts[idx]
                                 group_translations.append({
                                     'org_prompt': original_text,
                                     'trs_prompt': f"ERROR: {batch_error}",
                                     'org_lang': src_lang,
                                     'tar_lang': target_language_name,
                                     'model': "x-alma",
                                 })

        finally:
             if llm:
                 unload_model(llm)

        all_xalma_translations.extend(group_translations)
        print(f"\nFinished processing X-ALMA Group {group_num}. Collected {len(group_translations)} translations for this group.")

    if not all_xalma_translations:
        print("\nNo translations were generated.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_xalma_translations)
    print(f"\nTotal translations collected across all groups: {len(final_df)}")

    if export:
        try:
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"translated_prompts_x-alma_{{date_str}}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            final_df.to_pickle(output_filepath)
            print(f"\nSaved all X-ALMA results to {output_filepath}")
        except Exception as e:
            print(f"\nError saving results to pickle file: {e}")

    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using vLLM-hosted X-ALMA models.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--columns", type=str, nargs='+', default=['prompt', 'chosen', 'rejected'], help="Name of the column(s) to translate.")
    parser.add_argument("--src_lang", type=str, default='English', help="Source language.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation.")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    parser.add_argument("--xalma_base_repo", type=str, default="haoranxu/X-ALMA-13B", help="Base repo for the X-ALMA models.")
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
        "Spanish": "es",
    }

    all_translations = {}

    for column in args.columns:
        print(f"\n{'='*50}")
        print(f"Processing column: {column}")
        print(f"{'='*50}")

        translations_df = stream_batch_translate_xalma(
            df_path=args.df_path,
            rows=args.rows,
            output_path=f"{args.output_path}/{column}_translations",
            column=column,
            src_lang=args.src_lang,
            languages=LANGUAGES,
            xalma_base_repo=args.xalma_base_repo,
            batch_size=args.batch_size,
            export=args.export
        )

        all_translations[column] = translations_df

        if not translations_df.empty:
            print(f"\n--- Translation Summary for '{column}' ---")
            print(f"Generated {len(translations_df)} translations.")
            print("Sample translations:")
            print(translations_df.head(3))
            print(f"\nTarget language distribution for '{column}':")
            print(translations_df['tar_lang'].value_counts())
            print(f"\nModel used for '{column}':")
            print(translations_df['model'].value_counts())
        else:
            print(f"\nTranslation process for '{column}' completed, but no data was generated or returned.")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY - All Columns Processed")
    print(f"{'='*60}")

    total_translations = 0
    for column, df in all_translations.items():
        count = len(df) if not df.empty else 0
        total_translations += count
        print(f"{column}: {count} translations")

    print(f"\nTotal translations across all columns: {total_translations}")

    return all_translations

if __name__ == "__main__":
    results = main()