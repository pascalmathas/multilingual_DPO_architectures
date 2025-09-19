"""
This script creates a multilingual DPO dataset by merging an English DPO dataset with its translations.
"""

import pandas as pd
import json
import os
import argparse

def load_translations(file_path, file_format='jsonl'):
    """Loads translations from a file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}. Returning empty list.")
        return []
    if file_format == 'jsonl':
        translations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    translations.append(data['translation'])
                except (json.JSONDecodeError, KeyError):
                    continue
        return translations
    elif file_format == 'pickle':
        try:
            df = pd.read_pickle(file_path)
            return df
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"Unsupported file format: {file_format}")
        return []

def main():
    """Main function to create the multilingual DPO dataset."""
    parser = argparse.ArgumentParser(description="Create a multilingual DPO dataset.")
    parser.add_argument("--eng_data_path", type=str, required=True, help="Path to the English DPO dataset.")
    parser.add_argument("--translations_base_path", type=str, required=True, help="Base path to the translations.")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Path to save the output files.")
    parser.add_argument("--translation_format", type=str, default='jsonl', choices=['jsonl', 'pickle'], help="Format of the translation files.")
    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    eng_df = pd.read_pickle(args.eng_data_path)
    eng_df['language'] = 'English'

    language_pairs = ["en-ar", "en-de", "en-nl", "en-ru", "en-zh", "en-ja", "en-is", "en-es"]
    language_name_map = {
        "en": "English", "ar": "Arabic", "de": "German", "nl": "Dutch", "ru": "Russian",
        "zh": "Chinese", "ja": "Japanese", "is": "Icelandic", "es": "Spanish"
    }

    all_translations = {}
    if args.translation_format == 'jsonl':
        for lang_pair in language_pairs:
            target_lang_code = lang_pair.split('-')[1]
            all_translations[target_lang_code] = {}
            for part in ['prompt', 'chosen', 'rejected']:
                file_path = os.path.join(args.translations_base_path, part, f"test-{lang_pair}.jsonl")
                all_translations[target_lang_code][part] = load_translations(file_path, file_format='jsonl')
    elif args.translation_format == 'pickle':
        prompt_trs_df = load_translations(os.path.join(args.translations_base_path, 'prompt.pkl'), file_format='pickle')
        chosen_trs_df = load_translations(os.path.join(args.translations_base_path, 'chosen.pkl'), file_format='pickle')
        rejected_trs_df = load_translations(os.path.join(args.translations_base_path, 'rejected.pkl'), file_format='pickle')
        for lang_code in language_name_map:
            if lang_code != 'en':
                all_translations[lang_code] = {
                    'prompt': prompt_trs_df[prompt_trs_df['tar_lang'] == language_name_map[lang_code]]['trs_prompt'].tolist(),
                    'chosen': chosen_trs_df[chosen_trs_df['tar_lang'] == language_name_map[lang_code]]['trs_prompt'].tolist(),
                    'rejected': rejected_trs_df[rejected_trs_df['tar_lang'] == language_name_map[lang_code]]['trs_prompt'].tolist()
                }


    final_dfs = [eng_df]
    for lang_pair in language_pairs:
        target_lang_code = lang_pair.split('-')[1]
        target_lang_name = language_name_map.get(target_lang_code, target_lang_code)
        
        prompts = all_translations.get(target_lang_code, {}).get('prompt', [])
        chosens = all_translations.get(target_lang_code, {}).get('chosen', [])
        rejecteds = all_translations.get(target_lang_code, {}).get('rejected', [])

        min_len = min(len(prompts), len(chosens), len(rejecteds))

        translated_df = pd.DataFrame({
            'prompt': prompts[:min_len],
            'chosen': chosens[:min_len],
            'rejected': rejecteds[:min_len],
            'task': eng_df['task'][:min_len],
            'language': target_lang_name
        })
        final_dfs.append(translated_df)

    combined_df = pd.concat(final_dfs, ignore_index=True)

    output_pkl_path = os.path.join(args.output_base_dir, "DPO_multilingual_dataset.pkl")
    combined_df.to_pickle(output_pkl_path)
    print(f"Combined DataFrame saved to: {output_pkl_path}")

    json_output_data = []
    for _, row in combined_df.iterrows():
        json_record = {
            "chosen": [{"content": row['prompt'], "role": "user"}, {"content": row['chosen'], "role": "assistant"}],
            "rejected": [{"content": row['prompt'], "role": "user"}, {"content": row['rejected'], "role": "assistant"}]
        }
        json_output_data.append(json_record)

    output_json_path = os.path.join(args.output_base_dir, "DPO_multilingual_dataset.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output_data, f, ensure_ascii=False, indent=2)
    print(f"Data saved in JSON format to: {output_json_path}")

if __name__ == "__main__":
    main()
