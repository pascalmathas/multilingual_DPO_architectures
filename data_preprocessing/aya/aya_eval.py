"""
This script processes the Cohere/aya_evaluation_suite dataset to create a filtered and sampled version for evaluation.

The script performs the following steps:
1.  Loads the 'dolly_human_edited' and 'dolly_machine_translated' datasets.
2.  Combines the two datasets, prioritizing human-edited entries for specified languages.
3.  Filters the combined dataset to include only a specific set of target languages.
4.  Removes unnecessary columns ('targets', 'source_id').
5.  Samples a fixed number of entries for each target language.
6.  Saves the final processed dataset in Arrow, Pickle, and JSON Lines formats.
"""

import os
import pickle
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

# --- Configuration ---
SAVE_DIR_RAW = Path("../data/aya_evaluation_suite")
SAVE_DIR_PROCESSED = Path("../data/preprocessed/aya_evaluation_suite")
FILTER_LANGUAGES_ISO = ["arb", "deu", "spa", "rus", "zho", "jpn", "isl", "nld", "eng"]
SAMPLES_PER_LANGUAGE = 150
HUMAN_EDITED_PRIORITY_LANGS_ISO = ["arb", "fra", "hin", "rus", "spa", "srp"]

def ensure_hf_login():
    """Checks for Hugging Face authentication token."""
    if os.getenv("HF_TOKEN") is None:
        try:
            from huggingface_hub import HfFolder
            if HfFolder.get_token() is None:
                raise ValueError("No token found")
            print("Hugging Face token found.")
        except (ImportError, ValueError):
            print("Hugging Face token not found.")
            print("Please log in using `huggingface-cli login` or set the HF_TOKEN environment variable.")
            print("You can get a token from https://huggingface.co/settings/tokens")

def load_and_prepare_datasets(split_name='test'):
    """Loads and prepares the initial datasets from Hugging Face."""
    print("Loading datasets...")
    try:
        ds_dict_dolly_human_edited = load_dataset("CohereLabs/aya_evaluation_suite", "dolly_human_edited")
        ds_dict_dolly_machine_translated = load_dataset("CohereLabs/aya_evaluation_suite", "dolly_machine_translated")

        ds_dolly_human_edited = ds_dict_dolly_human_edited[split_name]
        ds_dolly_machine_translated = ds_dict_dolly_machine_translated[split_name]

        print(f"Relevant datasets loaded and '{split_name}' splits extracted successfully.")
        return ds_dolly_human_edited, ds_dolly_machine_translated
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

def combine_datasets(ds_dolly_human_edited, ds_dolly_machine_translated):
    """Combines human-edited and machine-translated datasets."""
    print("\nCombining dolly_machine_translated and dolly_human_edited...")
    human_edited_lookup = {
        (row['id'], row['language']): row
        for row in ds_dolly_human_edited
        if row['language'] in HUMAN_EDITED_PRIORITY_LANGS_ISO
    }

    combined_data_list = []
    for mt_row in ds_dolly_machine_translated:
        key = (mt_row['id'], mt_row['language'])
        if mt_row['language'] in HUMAN_EDITED_PRIORITY_LANGS_ISO and key in human_edited_lookup:
            combined_data_list.append(human_edited_lookup[key])
        else:
            combined_data_list.append(mt_row)

    combined_ds = Dataset.from_list(combined_data_list, features=ds_dolly_machine_translated.features)
    print(f"Combined dataset created with {len(combined_ds)} entries.")
    return combined_ds

def filter_and_clean_dataset(dataset):
    """Filters by language and removes unnecessary columns."""
    print(f"\nFiltering combined dataset for target languages: {', '.join(FILTER_LANGUAGES_ISO)}...")
    filtered_ds = dataset.filter(lambda example: example['language'] in FILTER_LANGUAGES_ISO)
    print(f"Dataset filtered for target languages now has {len(filtered_ds)} entries.")

    print("\nRemoving 'targets' and 'source_id' columns...")
    columns_to_remove = [col for col in ['targets', 'source_id'] if col in filtered_ds.column_names]
    cleaned_ds = filtered_ds.remove_columns(columns_to_remove)
    print(f"Columns {', '.join(columns_to_remove)} removed.")
    return cleaned_ds

def sample_dataset(dataset):
    """Samples a fixed number of entries for each language."""
    print(f"\nSampling up to {SAMPLES_PER_LANGUAGE} entries for each of the {len(FILTER_LANGUAGES_ISO)} target languages...")
    sampled_datasets_list = []
    for lang_code in FILTER_LANGUAGES_ISO:
        lang_specific_ds = dataset.filter(lambda x: x['language'] == lang_code)
        num_to_sample = min(len(lang_specific_ds), SAMPLES_PER_LANGUAGE)
        if num_to_sample > 0:
            sampled_datasets_list.append(lang_specific_ds.select(range(num_to_sample)))
            print(f"  Selected {num_to_sample} samples for language: {lang_code}")

    if not sampled_datasets_list:
        print("\nNo data available after sampling. Exiting.")
        return None

    final_sampled_ds = concatenate_datasets(sampled_datasets_list)
    print(f"\nFinal sampled dataset created with {len(final_sampled_ds)} entries.")
    return final_sampled_ds

def save_processed_dataset(dataset, base_file_name="aya_filtered_sampled_open_ended"):
    """Saves the processed dataset in multiple formats."""
    if dataset is None or len(dataset) == 0:
        print("\nFinal processed dataset is empty, not saving any files.")
        return

    SAVE_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Save as Arrow
    arrow_path = SAVE_DIR_PROCESSED / base_file_name
    dataset.save_to_disk(str(arrow_path))
    print(f"\nFinal processed dataset (Arrow format) saved to {arrow_path}")

    # Save as Pickle
    pickle_path = SAVE_DIR_PROCESSED / f"{base_file_name}.pkl"
    with open(pickle_path, 'wb') as f_pkl:
        pickle.dump(dataset.to_list(), f_pkl)
    print(f"Final processed dataset (Pickle format) saved to {pickle_path}")

    # Save as JSONL
    jsonl_path = SAVE_DIR_PROCESSED / f"{base_file_name}.jsonl"
    dataset.to_json(jsonl_path, lines=True, force_ascii=False)
    print(f"Final processed dataset (JSON Lines format) saved to {jsonl_path}")


def main():
    """Main function to run the data processing pipeline."""
    ensure_hf_login()

    ds_human, ds_machine = load_and_prepare_datasets()
    if ds_human is None or ds_machine is None:
        return

    combined_ds = combine_datasets(ds_human, ds_machine)
    cleaned_ds = filter_and_clean_dataset(combined_ds)
    final_ds = sample_dataset(cleaned_ds)
    save_processed_dataset(final_ds)

    print("\nScript finished.")

if __name__ == "__main__":
    main()
