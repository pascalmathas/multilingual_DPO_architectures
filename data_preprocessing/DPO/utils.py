"""
Utility functions for DPO data processing.
"""

import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()


def get_tokenizer(model_name='gpt2'):
    """Initializes and returns a tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Successfully loaded tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer {model_name}: {e}")
        return None

def clean_text_columns(df, df_name, columns_to_clean=['prompt', 'chosen', 'rejected']):
    """Cleans specified text columns in a DataFrame."""
    print(f"\nCleaning text columns in DataFrame '{df_name}'...")
    df_cleaned = df.copy()
    for col_name in columns_to_clean:
        if col_name in df_cleaned.columns:
            df_cleaned[col_name] = df_cleaned[col_name].astype(str).str.replace('\n', ' ', regex=False)
            df_cleaned[col_name] = df_cleaned[col_name].str.replace(r'\\"', '"', regex=True)
            df_cleaned[col_name] = df_cleaned[col_name].str.replace('\\', ' ', regex=False)
            df_cleaned[col_name] = df_cleaned[col_name].str.strip()
            df_cleaned[col_name] = df_cleaned[col_name].str.replace(r'\s+', ' ', regex=True)
    return df_cleaned

def filter_by_max_token_length(df, df_name, tokenizer, max_tokens, columns_to_check=['prompt', 'chosen', 'rejected']):
    """Filters a DataFrame by the maximum token length of specified columns."""
    if tokenizer is None:
        print(f"Skipping token length filtering for '{df_name}' (tokenizer unavailable).")
        return df

    print(f"\nFiltering '{df_name}' by max token length ({max_tokens} tokens)...") 
    original_row_count = len(df)
    mask = pd.Series(True, index=df.index)
    for col_name in columns_to_check:
        if col_name in df.columns:
            token_lengths = df[col_name].astype(str).progress_apply(lambda x: len(tokenizer.encode(x)))
            mask &= token_lengths <= max_tokens
    df_filtered = df[mask]
    print(f"  '{df_name}': {len(df_filtered)} rows remaining after token length filtering.")
    return df_filtered

def filter_non_empty_essentials(df, df_name, essential_columns=['prompt', 'chosen', 'rejected']):
    """Filters out rows with empty essential columns."""
    print(f"\nFiltering '{df_name}' for non-empty essential columns...")
    original_rows = len(df)
    df_filtered = df.dropna(subset=essential_columns)
    rows_removed = original_rows - len(df_filtered)
    print(f"  '{df_name}': Removed {rows_removed} rows with empty essential columns.")
    return df_filtered

def combine_dpo_datasets(datasets_config, shuffle_final=True, random_state=None):
    """Combines multiple DPO datasets into a single DataFrame."""
    print("\n--- Starting dataset combination ---")
    selected_dfs = []
    for df, num_to_take, df_name in datasets_config:
        if df is None or df.empty:
            print(f"  Skipping '{df_name}' as it is empty or None.")
            continue
        
        num_to_take = len(df) if num_to_take == -1 else min(num_to_take, len(df))
        print(f"  Taking {num_to_take} entries from '{df_name}'.")
        selected_dfs.append(df.head(num_to_take))

    if not selected_dfs:
        return pd.DataFrame(columns=['chosen', 'rejected', 'prompt', 'task'])

    combined_df = pd.concat(selected_dfs, ignore_index=True)
    if shuffle_final:
        combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"--- Dataset combination complete. Final DataFrame has {len(combined_df)} rows. ---")
    return combined_df

def convert_to_python_list(raw_input_val):
    """Safely converts a value to a list of dictionaries."""
    if isinstance(raw_input_val, list):
        return raw_input_val
    if isinstance(raw_input_val, (np.ndarray, str)):
        try:
            return ast.literal_eval(raw_input_val if isinstance(raw_input_val, str) else raw_input_val.tolist())
        except (ValueError, SyntaxError):
            return None
    return None


def is_single_user_assistant_exchange(conversation):
    """Checks if a conversation is a single user-assistant exchange."""
    if not isinstance(conversation, list) or len(conversation) != 2:
        return False
    
    first_turn, second_turn = conversation[0], conversation[1]
    return (
        isinstance(first_turn, dict) and first_turn.get('role') == 'user' and
        isinstance(second_turn, dict) and second_turn.get('role') == 'assistant'
    )
