"""
This script processes the google/wmt24pp dataset.

It performs the following steps:
1.  Downloads the dataset for specified language pairs.
2.  Combines the datasets for all language pairs into a single DataFrame.
3.  Cleans the data by removing bad sources.
4.  Exports the cleaned data to Pickle files.
"""

from collections import defaultdict
import os
import pandas as pd
from datasets import load_dataset, load_from_disk
from wmt24pp_constants import LANGUAGE_BY_CODE

def get_dataset(languages, base_dir):
    """Downloads and loads the WMT24pp dataset for the given languages."""
    lang_to_codes = defaultdict(list)
    for code, name in LANGUAGE_BY_CODE.items():
        lang_to_codes[name].append(code)

    preferred_codes = {name: sorted(codes)[0] for name, codes in lang_to_codes.items()}

    language_pairs = {}
    for name, simple_code in languages.items():
        lang_name = name
        if name == "Standard Arabic":
            lang_name = "Arabic"
        elif name == "Chinese":
            lang_name = "Mandarin"

        full_code = preferred_codes.get(lang_name)
        if full_code:
            language_pairs[name] = f"en-{full_code}"
        else:
            language_pairs[name] = f"en-{simple_code}"

    os.makedirs(base_dir, exist_ok=True)

    for lang_name, dataset_code in language_pairs.items():
        save_path = os.path.join(base_dir, dataset_code)
        if os.path.exists(save_path):
            print(f"Dataset for {lang_name} ({dataset_code}) already exists. Skipping download.")
            continue

        print(f"Loading dataset for {lang_name} ({dataset_code})...")
        ds = load_dataset("google/wmt24pp", dataset_code)
        print(f"Saving to {save_path}...")
        ds.save_to_disk(save_path)

    all_dfs = []
    lang_pair_dirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    for lp in lang_pair_dirs:
        dataset_path = os.path.join(base_dir, lp)
        try:
            ds = load_from_disk(dataset_path)
            ds_train = ds.get("train", ds)
            df = ds_train.to_pandas()
            df["lp"] = lp
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to load {lp}: {e}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[[
        "lp", "domain", "document_id", "segment_id",
        "is_bad_source", "source", "target", "original_target"
    ]]
    return final_df, language_pairs

def main():
    """Main function to process the WMT24pp dataset."""
    # languages = {
    #     "Arabic": "ar", 
    #     "German": "de",
    #     "Spanish": "es",
    #     "Russian": "ru",
    #     "Chinese": "zh",
    #     "Japanese": "ja",
    #     "Icelandic": "is",
    # }

    languages = {
        "Spanish": "es",
    }

    df, _ = get_dataset(languages, base_dir="../data/wmt24pp")

    df = df.drop(columns=["original_target"])
    df_clean = df[df["is_bad_source"] == False].reset_index(drop=True)

    print("Bad sources have been removed.")
    print(f"Remaining rows: {df_clean.shape[0]}")

    output_dir = "../data/preprocessed/wmt24pp/"
    os.makedirs(output_dir, exist_ok=True)

    # Export full dataset
    output_file = os.path.join(output_dir, "wmt24pp.pkl")
    df_clean.to_pickle(output_file)
    print(f"Exported full dataset to: {output_file}")

    # Export small dataset for translation
    df_translate = df_clean.head(300).reset_index(drop=True)
    translate_file = os.path.join(output_dir, "wmt24pp_translate.pkl")
    df_translate.to_pickle(translate_file)
    print(f"Exported translation input to: {translate_file}")

if __name__ == "__main__":
    main()
