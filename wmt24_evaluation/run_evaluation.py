import pandas as pd
import os
import numpy as np
import nltk
from collections import defaultdict
import argparse
from utils import (
    evaluate_translations,
    print_results,
    analyze_results_by_model,
    identify_best_model,
    save_results_to_csv,
)
from plotting import plot_score_heatmaps

# Download necessary NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')

def add_paper_results(results):
    paper_scores_gpt4o = {
        'Arabic': {'BLEU': 11.7, 'CHRF': 38.4},
        'German': {'BLEU': 33.7, 'CHRF': 60.8},
        'Dutch': {'BLEU': 36.1, 'CHRF': 61.3},
        'Russian': {'BLEU': 22.9, 'CHRF': 50.3},
        'Chinese': {'BLEU': 42.8, 'CHRF': 41.0},
        'Japanese': {'BLEU': 25.7, 'CHRF': 36.1},
        'Icelandic': {'BLEU': 25.3, 'CHRF': 49.9}
    }

    for language, scores in paper_scores_gpt4o.items():
        if language not in results:
            results[language] = {}

        results[language]['GPT-4o Deutsch paper'] = {
            'BLEU': scores['BLEU'],
            'CHRF': scores['CHRF'],
            'count': -1
        }

    paper_scores_unbabel = {
        'Arabic': {'BLEU': -1, 'CHRF': -1},
        'German': {'BLEU': 32.0, 'CHRF': 59.4},
        'Dutch': {'BLEU': 34.6, 'CHRF': 60.3},
        'Russian': {'BLEU': 22.3, 'CHRF': 48.4},
        'Chinese': {'BLEU': 40.6, 'CHRF': 38.4},
        'Japanese': {'BLEU': 23.5, 'CHRF': 33.1},
        'Icelandic': {'BLEU': 23.7, 'CHRF': 47.9}
    }

    for language, scores in paper_scores_unbabel.items():
        if language not in results:
            results[language] = {}

        results[language]['Unbabel Tower Deutsch paper'] = {
            'BLEU': scores['BLEU'],
            'CHRF': scores['CHRF'],
            'count': -1
        }
    return results

def main(args):
    """Main function to run the evaluation pipeline."""

    # Rename files
    translations = "../data/translations/alma-13b-wmt24pp-translations/"
    for filename in os.listdir(translations):
        full_path = os.path.join(translations, filename)
        if os.path.isfile(full_path) and '.' not in filename:
            new_filename = filename + '.txt'
            new_full_path = os.path.join(translations, new_filename)
            os.rename(full_path, new_full_path)
            print(f"Renamed: {filename} -> {new_filename}")

    # Language mapping
    lang2lp = {
        'Arabic'            : 'en-ar_EG',
        'German'            : 'en-de_DE',
        'Spanish'           : 'en-es_MX',
        'Dutch'             : 'en-nl_NL',
        'Russian'           : 'en-ru_RU',
        'Chinese'           : 'en-zh_CN',
        'Japanese'          : 'en-ja_JP',
        'Icelandic'         : 'en-is_IS',
    }

    # Load WMT data
    try:
        df_wmt = pd.read_pickle(args.wmt_pickle_file)
    except FileNotFoundError:
        print(f"Error: Main WMT pickle file '{args.wmt_pickle_file}' not found.")
        exit()

    all_new_translations = []

    for target_lang_full, lp_code in lang2lp.items():
        source_lang_short, target_lang_short = lp_code.split('-')
        
        translation_txt_file = f"../data/translations/alma-13b-wmt24pp-translations/test-en-{target_lang_short.split('_')[0]}.txt"
        
        if not os.path.exists(translation_txt_file):
            print(f"Info: Translation file '{translation_txt_file}' for {target_lang_full} not found. Skipping.")
            continue
            
        print(f"Processing {target_lang_full} from '{translation_txt_file}'...")
        
        try:
            with open(translation_txt_file, 'r', encoding='utf-8') as f:
                translated_segments_from_txt = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading '{translation_txt_file}': {e}. Skipping.")
            continue

        if not translated_segments_from_txt:
            print(f"Warning: No segments found in '{translation_txt_file}'. Skipping.")
            continue
        
        df_wmt_filtered = df_wmt[df_wmt['lp'] == lp_code]
        
        original_english_segments = df_wmt_filtered['source'].tolist()
        
        if len(original_english_segments) != len(translated_segments_from_txt):
            print(f"Warning: Mismatch in segment count for {target_lang_full} ({lp_code}).")
            print(f"  Original English segments in WMT data: {len(original_english_segments)}")
            print(f"  Translated segments from '{translation_txt_file}': {len(translated_segments_from_txt)}")
            print(f"  Taking the minimum number of segments to proceed. This might indicate a data issue.")
            min_len = min(len(original_english_segments), len(translated_segments_from_txt))
            if min_len == 0:
                print(f"  Zero common segments. Skipping {target_lang_full}.")
                continue
            original_english_segments = original_english_segments[:min_len]
            translated_segments_from_txt = translated_segments_from_txt[:min_len]

        for org_text, trs_text in zip(original_english_segments, translated_segments_from_txt):
            all_new_translations.append({
                "org_prompt": org_text,
                "trs_prompt": trs_text,
                "org_lang": "English",
                "tar_lang": target_lang_full,
                "model": args.model_name
            })

    if all_new_translations:
        output_df = pd.DataFrame(all_new_translations)
        output_df.to_pickle(args.output_pickle_file)
        print(f"\nSuccessfully generated '{args.output_pickle_file}' with {len(output_df)} records.")
        print("First few rows of the generated DataFrame:")
        print(output_df.head())
        print("\nRecords per target language:")
        print(output_df['tar_lang'].value_counts())
    else:
        print("\nNo translation data processed. Output file not generated.")

    # Combine and process dataframes
    translations_folder = '../data/translations/wmt24pp'
    translations = [f for f in os.listdir(translations_folder) if f.endswith('.pkl')]
    dfs = [pd.read_pickle(os.path.join(translations_folder, file)) for file in translations]

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=["org_prompt", "model"]).reset_index(drop=True)
    combined_df['tar_lang'] = combined_df['tar_lang'].replace('Spanish (Mexico)', 'Spanish')

    wmt = (
        pd.read_pickle('../data/preprocessed/wmt24pp/wmt24pp_small.pkl')
          .loc[:, ['lp', 'source', 'target']]
    )

    combined_df = combined_df.copy()
    combined_df['lp'] = combined_df['tar_lang'].map(lang2lp)

    combined_df = combined_df.merge(
        wmt,
        how='left',
        left_on = ['lp', 'org_prompt'],
        right_on = ['lp', 'source'],
        validate='m:1'
    )

    combined_df = (
        combined_df
          .rename(columns={'target': 'reference'})
          .drop(columns=['source', 'lp'])
    )

    # Filter models
    combined_df = combined_df[~((combined_df['model'] == 'tower-13b') & (combined_df['tar_lang'].isin(['Arabic', 'Icelandic'])))]
    combined_df = combined_df[~((combined_df['model'] == 'Qwen2.5 14B Instruct') & (combined_df['tar_lang'].isin(['Arabic', 'Icelandic'])))]

    # Run evaluation
    results = evaluate_translations(combined_df)
    results = add_paper_results(results)
    print_results(results)
    analyze_results_by_model(results) 
    identify_best_model(results)
    save_results_to_csv(results, 'data/preprocessed/experiment/translation_evaluation_results.csv')
    
    # Plot results
    MODEL_PRIORITY_MAP = {
        'TowerInstruct': ['tower-13b', 'Tower-13b'],
        'X-ALMA': ['x-alma'],
        'X-ALMA-BS': ["x-alma-git"],
        'GPT-3.5 Turbo (API)': ['gpt-3.5-turbo-0125'],
        'GPT-4o (API)': ["gpt-4o-2024-08-06"],
        'Gemma-3': ['Gemma-3-12B-IT'],
        'Qwen 2.5': ['Qwen2.5 14B Instruct'],
        "NLLB-200": ['nllb-3.3b'],
    }
    plot_score_heatmaps(results, MODEL_PRIORITY_MAP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run WMT24 evaluation pipeline.')
    parser.add_argument('--wmt_pickle_file', type=str, default='../data/preprocessed/wmt24pp/wmt24pp_small.pkl', help='Path to the WMT pickle file.')
    parser.add_argument('--model_name', type=str, default='x-alma-git', help='Name of the model being evaluated.')
    parser.add_argument('--output_pickle_file', type=str, default='../data/translations/wmt24pp/x_alma_git_translations.pkl', help='Path to save the output pickle file.')
    args = parser.parse_args()
    main(args)
