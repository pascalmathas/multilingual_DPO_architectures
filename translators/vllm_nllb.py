"""
This script translates a column of text in a DataFrame using a Hugging Face NLLB model.

It can be configured to translate to a specific set of languages, and can be used to
translate multiple columns in a DataFrame.
"""

import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import os
import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse

NLLB_LANG_CODES = {
    "English": "eng_Latn",
    "German": "deu_Latn",
    "Dutch": "nld_Latn",
    "Russian": "rus_Cyrl",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Icelandic": "isl_Latn",
    "Arabic": "arb_Arab",
    "Spanish": "spa_Latn",
}

_hf_model_cache = None
_hf_tokenizer_cache = None

def load_huggingface_model_and_tokenizer(model_repo_id: str, src_lang_code_for_tokenizer: str):
    """Loads the Hugging Face model and tokenizer."""
    global _hf_model_cache, _hf_tokenizer_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if _hf_tokenizer_cache is None:
        print(f"\nLoading Hugging Face tokenizer: {model_repo_id} with src_lang={src_lang_code_for_tokenizer} ...\n")
        _hf_tokenizer_cache = AutoTokenizer.from_pretrained(model_repo_id, src_lang=src_lang_code_for_tokenizer)

    if _hf_model_cache is None:
        print(f"\nLoading Hugging Face model: {model_repo_id} ...\n")
        _hf_model_cache = AutoModelForSeq2SeqLM.from_pretrained(model_repo_id)
        _hf_model_cache.to(device)
        _hf_model_cache.eval()

    return _hf_model_cache, _hf_tokenizer_cache, device

def unload_huggingface_model_and_tokenizer():
    """Unloads the Hugging Face model and tokenizer."""
    global _hf_model_cache, _hf_tokenizer_cache
    print("\nUnloading Hugging Face model and tokenizer from memory...")
    del _hf_model_cache
    del _hf_tokenizer_cache
    _hf_model_cache = None
    _hf_tokenizer_cache = None
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def translate_batch_huggingface_nllb(
    model,
    tokenizer,
    device: str,
    texts_batch: List[str],
    src_lang_code: str,
    tgt_lang_code: str,
    max_length_generation: int = 2300,
    max_length_input: int = 2300
) -> List[str]:
    """Translates a batch of texts using the NLLB model."""
    if tokenizer.src_lang != src_lang_code:
        tokenizer.src_lang = src_lang_code

    tgt_lang_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    if tgt_lang_token_id == tokenizer.unk_token_id:
        print(f"Warning: Target language code {tgt_lang_code} converted to UNK token ID. This might be an error.")

    with torch.no_grad():
        inputs = tokenizer(
            texts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length_input
        ).to(device)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token_id,
            max_new_tokens=max_length_generation,
        )

        translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations

def stream_batch_translate_huggingface_nllb(
    df_path: str,
    rows: int,
    output_path: str,
    column: str,
    src_lang_name: str,
    target_languages_map: Dict[str, str],
    model_name_arg: str,
    model_repo_arg: str,
    batch_size: int = 8,
    export: bool = True,
    max_generation_tokens: int = 256,
    max_input_tokens: int = 512
):
    """Translates a DataFrame column using the NLLB model."""
    print(f"--- Hugging Face NLLB Model Translation Job ---")
    print(f"Model Name: {model_name_arg}")
    print(f"Model Repository: {model_repo_arg}")
    print(f"DataFrame path: {df_path}")
    print(f"Rows to translate: {rows if rows != 0 else 'All'}")
    print(f"Column to translate: {column}")
    print(f"Source language: {src_lang_name}")

    if src_lang_name not in NLLB_LANG_CODES:
        print(f"Error: Source language '{src_lang_name}' not found in NLLB_LANG_CODES mapping.")
        return pd.DataFrame()
    src_lang_code = NLLB_LANG_CODES[src_lang_name]
    print(f"Source language code: {src_lang_code}")

    valid_target_langs = {}
    for name, code in target_languages_map.items():
        if name == src_lang_name:
            print(f"Skipping target language '{name}' as it's the source language.")
            continue
        if name not in NLLB_LANG_CODES or NLLB_LANG_CODES[name] != code:
            print(f"Warning: Target language '{name}' with code '{code}' mismatch or not in NLLB_LANG_CODES. Verifying...")
            if name in NLLB_LANG_CODES:
                print(f"Using NLLB_LANG_CODES defined code for '{name}': {NLLB_LANG_CODES[name]}")
                valid_target_langs[name] = NLLB_LANG_CODES[name]
            else:
                print(f"Skipping target language '{name}' as it's not in master NLLB_LANG_CODES list.")
                continue
        else:
            valid_target_langs[name] = code

    if not valid_target_langs:
        print("Warning: No valid target languages specified or found in NLLB_LANG_CODES (excluding source language).")
        return pd.DataFrame()

    target_lang_names_for_print = [name for name in valid_target_langs.keys()]
    print(f"Target language(s): {', '.join(target_lang_names_for_print)}")
    print(f"Batch size (for HF processing): {batch_size}")
    print(f"Max generation tokens: {max_generation_tokens}")
    print(f"Max input tokens: {max_input_tokens}")
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

    all_translations = []
    model, tokenizer, device = None, None, None

    try:
        model, tokenizer, device = load_huggingface_model_and_tokenizer(model_repo_arg, src_lang_code)

        for i in tqdm(range(0, len(df), batch_size), desc=f"Translating with {model_name_arg} (HF)"):
            batch_df = df.iloc[i:i+batch_size]
            batch_source_texts = batch_df[column].tolist()

            valid_texts_with_indices = [
                (idx, text) for idx, text in enumerate(batch_source_texts) if isinstance(text, str) and text.strip()
            ]

            if not valid_texts_with_indices:
                for original_idx, original_text in enumerate(batch_source_texts):
                     for target_language_name, target_language_code in valid_target_langs.items():
                        all_translations.append({
                            'org_prompt': original_text if isinstance(original_text, str) else "INVALID_INPUT",
                            'trs_prompt': "SKIPPED: Empty or invalid input text",
                            'org_lang': src_lang_name,
                            'org_lang_code': src_lang_code,
                            'tar_lang': target_language_name,
                            'tar_lang_code': target_language_code,
                            'model': model_name_arg,
                        })
                continue

            current_batch_texts = [text for _, text in valid_texts_with_indices]

            for target_language_name, target_language_code in valid_target_langs.items():
                try:
                    batch_translations_results = translate_batch_huggingface_nllb(
                        model, tokenizer, device, current_batch_texts,
                        src_lang_code, target_language_code,
                        max_length_generation=max_generation_tokens,
                        max_length_input=max_input_tokens
                    )

                    result_idx = 0
                    for original_idx, original_text in enumerate(batch_source_texts):
                        if isinstance(original_text, str) and original_text.strip() and any(original_idx == vt_idx for vt_idx, _ in valid_texts_with_indices if vt_idx == original_idx):
                            translation = batch_translations_results[result_idx]
                            result_idx +=1
                        else:
                            translation = "SKIPPED: Empty or invalid input text"

                        all_translations.append({
                            'org_prompt': original_text if isinstance(original_text, str) else "INVALID_INPUT",
                            'trs_prompt': translation,
                            'org_lang': src_lang_name,
                            'org_lang_code': src_lang_code,
                            'tar_lang': target_language_name,
                            'tar_lang_code': target_language_code,
                            'model': model_name_arg,
                        })

                except Exception as batch_error:
                    print(f"\nError during HF translation batch for lang {target_language_name} ({target_language_code}): {batch_error}")
                    import traceback
                    traceback.print_exc()
                    for original_text in batch_source_texts:
                        all_translations.append({
                            'org_prompt': original_text if isinstance(original_text, str) else "INVALID_INPUT",
                            'trs_prompt': f"ERROR: {batch_error}",
                            'org_lang': src_lang_name,
                            'org_lang_code': src_lang_code,
                            'tar_lang': target_language_name,
                            'tar_lang_code': target_language_code,
                            'model': model_name_arg,
                        })
    except Exception as e:
        print(f"ERROR: Critical error during model loading or processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if model or tokenizer:
            unload_huggingface_model_and_tokenizer()
            print(f"\nFinished processing with {model_name_arg}. HF Model unloaded.")

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
            safe_model_name = model_name_arg.replace("/", "_").replace("-", "_")
            output_filename = f"translated_prompts_{safe_model_name}_HF_{date_str}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            final_df.to_pickle(output_filepath)
            print(f"\nSaved all results to {output_filepath}")
        except Exception as e:
            print(f"\nError saving results to pickle file: {e}")

    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using a Hugging Face NLLB model.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--columns", type=str, nargs='+', default=['prompt', 'chosen', 'rejected'], help="Name of the column(s) to translate.")
    parser.add_argument("--src_lang_name", type=str, default='English', help="Source language name.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation.")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    parser.add_argument("--max_generation_tokens", type=int, default=2300, help="Maximum generation tokens.")
    parser.add_argument("--max_input_tokens", type=int, default=2300, help="Maximum input tokens.")
    parser.add_argument("--model_name", type=str, default="nllb-3.3b", help="Name of the model.")
    parser.add_argument("--model_repo", type=str, default="facebook/nllb-200-3.3B", help="Repo of the model.")
    parser.add_argument("--target_languages", type=str, nargs='+', default=["German", "Dutch", "Russian", "Chinese", "Japanese", "Icelandic", "Arabic", "Spanish"], help="Target languages.")
    args = parser.parse_args()

    TARGET_LANGUAGES_TO_TRANSLATE = {lang: NLLB_LANG_CODES[lang] for lang in args.target_languages}

    for column_name in args.columns:
        translations_df = stream_batch_translate_huggingface_nllb(
            df_path=args.df_path,
            rows=args.rows,
            output_path=f"{args.output_path}/{column_name}/",
            column=column_name,
            src_lang_name=args.src_lang_name,
            target_languages_map=TARGET_LANGUAGES_TO_TRANSLATE,
            model_name_arg=args.model_name,
            model_repo_arg=args.model_repo,
            batch_size=args.batch_size,
            export=args.export,
            max_generation_tokens=args.max_generation_tokens,
            max_input_tokens=args.max_input_tokens
        )

        if not translations_df.empty:
            print(f"\n--- Translation Summary for column '{column_name}' ---")
            print(f"Generated {len(translations_df)} translations.")
            print("Sample translations (first 10):")
            print(translations_df.head(10))
            print("\nTarget language distribution:")
            print(translations_df['tar_lang'].value_counts())
            print("\nModel used:")
            print(translations_df['model'].value_counts())
            print("\nTranslations for 'INVALID_INPUT' or 'SKIPPED':")
            print(translations_df[translations_df['trs_prompt'].str.contains("SKIPPED|INVALID_INPUT", na=False)])
        else:
            print(f"\nTranslation process for column '{column_name}' completed, but no data was generated or returned.")

if __name__ == "__main__":
    main()