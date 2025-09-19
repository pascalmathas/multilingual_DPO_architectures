"""
This script converts a DataFrame from a pickle file into a series of JSON files
formatted for the X-ALMA model.

The script performs the following steps:
1.  Loads a DataFrame from a specified pickle file.
2.  Processes each row of the DataFrame to extract language pair, source text, and target text.
3.  Groups the data by language pair.
4.  Writes the data for each language pair to a separate JSON file.
"""

import pandas as pd
import json
import os
from collections import defaultdict
from tqdm import tqdm

INPUT_PICKLE_FILE = '../data/preprocessed/wmt24pp/wmt24pp_small.pkl'
OUTPUT_BASE_DIR = '../data/preprocessed/wmt24pp_jspn/wmt24pp_alma_json'
LP_COLUMN = 'lp'
SOURCE_COLUMN = 'source'
TARGET_COLUMN = 'target'
JSON_INDENT = 2

def load_dataframe(file_path):
    """Loads a DataFrame from a pickle file."""
    print(f"Loading DataFrame from {file_path}...")
    if not os.path.exists(file_path):
        print(f"\nError: Input pickle file not found at {file_path}")
        return None
    try:
        df = pd.read_pickle(file_path)
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"\nAn unexpected error occurred during DataFrame loading: {e}")
        return None

def process_data(df):
    """Processes the DataFrame and collects data for JSON conversion."""
    output_data_lists = defaultdict(list)
    required_columns = [LP_COLUMN, SOURCE_COLUMN, TARGET_COLUMN]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"\nError: Missing required columns in DataFrame: {missing_cols}")
        return None

    print("\nProcessing rows and collecting data...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Collecting data"):
        try:
            lp_full = str(row[LP_COLUMN])
            source_text = str(row[SOURCE_COLUMN])
            target_text = str(row[TARGET_COLUMN])

            lang_pair_part = lp_full.split('_')[0]
            src_lang, tgt_lang = lang_pair_part.split('-', 1)

            first_lang = src_lang if src_lang != "en" else tgt_lang
            pair_dir_name = first_lang + "en"

            output_dir = os.path.join(OUTPUT_BASE_DIR, pair_dir_name)
            output_filename = f"test.{src_lang}-{tgt_lang}.json"
            output_path = os.path.join(output_dir, output_filename)

            json_data = {
                "translation": {
                    src_lang: source_text,
                    tgt_lang: target_text
                }
            }
            output_data_lists[output_path].append(json_data)

        except Exception as e:
            print(f"\nWarning: Skipping row index {index} due to unexpected error: {e}")

    return output_data_lists

def write_json_files(output_data_lists):
    """Writes the collected data to JSON files."""
    if not output_data_lists:
        print("No data to write.")
        return

    print("\nWriting collected data to JSON files...")
    files_written = 0
    total_lines_written = 0
    for output_path, data_list in tqdm(output_data_lists.items(), desc="Writing files"):
        if not data_list:
            continue
        try:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(data_list, outfile, ensure_ascii=False, indent=JSON_INDENT)
                files_written += 1
                total_lines_written += len(data_list)
        except Exception as e:
            print(f"\nError writing file {output_path}: {e}")

    print("\n--- Summary ---")
    print(f"Created/updated {files_written} JSON files in '{OUTPUT_BASE_DIR}'.")
    print(f"Total translation pairs written across all files: {total_lines_written}")

def main():
    """Main function to run the conversion process."""
    df = load_dataframe(INPUT_PICKLE_FILE)
    if df is not None:
        output_data = process_data(df)
        if output_data:
            write_json_files(output_data)
    print("\nConversion finished.")

if __name__ == "__main__":
    main()
