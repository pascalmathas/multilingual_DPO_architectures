"""
This script downloads, processes, and combines multiple DPO datasets.
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset
from utils import (
    clean_text_columns,
    filter_by_max_token_length,
    filter_non_empty_essentials,
    combine_dpo_datasets,
    get_tokenizer,
    is_single_user_assistant_exchange,
    convert_to_python_list
)


def load_tasksource_dpo(cache_dir="../data/tasksource"):
    """Loads and formats the TaskSource DPO dataset."""
    print("\n--- Loading TaskSource DPO ---")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tasksource/tasksource_dpo_pairs", split="test", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    df["task"] = "discriminative_tasks"
    return df[['chosen', 'rejected', 'prompt', 'task']]

def load_chatbot_arena_dpo(cache_dir="../data/chatbot_arena"):
    """Loads and formats the Chatbot Arena DPO dataset."""
    print("\n--- Loading Chatbot Arena DPO ---")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    df = df[df["language"].str.lower().isin({"english", "en"})].copy()
    
    df['conversation_a'] = df['conversation_a'].apply(convert_to_python_list)
    df['conversation_b'] = df['conversation_b'].apply(convert_to_python_list)

    mask_a = df['conversation_a'].apply(is_single_user_assistant_exchange)
    mask_b = df['conversation_b'].apply(is_single_user_assistant_exchange)
    df = df[mask_a & mask_b]

    df = df[df['winner'] != 'tie']

    df['prompt'] = df.apply(lambda row: row['conversation_a'][0]['content'], axis=1)
    df['chosen'] = df.apply(lambda row: row['conversation_a'][1]['content'] if row['winner'] == 'model_a' else row['conversation_b'][1]['content'], axis=1)
    df['rejected'] = df.apply(lambda row: row['conversation_b'][1]['content'] if row['winner'] == 'model_a' else row['conversation_a'][1]['content'], axis=1)
    df["task"] = "question_answering"
    return df[['chosen', 'rejected', 'prompt', 'task']]

def load_gutenberg_dpo(cache_dir="../data/gutenberg_dpo"):
    """Loads and formats the Gutenberg DPO dataset."""
    print("\n--- Loading Gutenberg DPO ---")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("jondurbin/gutenberg-dpo-v0.1", split="train", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    df["task"] = "generative_writing_tasks"
    return df[['chosen', 'rejected', 'prompt', 'task']]

def load_math_step_dpo(cache_dir="../data/math_step_dpo"):
    """Loads and formats the Math Step DPO dataset."""
    print("\n--- Loading Math Step DPO ---")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("xinlai/Math-Step-DPO-10K", split="train", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    df["task"] = "mathematical_reasoning"
    return df[['chosen', 'rejected', 'prompt', 'task']]

def load_helpsteer_dpo(cache_dir="../data/helpsteer2"):
    """Loads and formats the HelpSteer2 DPO dataset."""
    print("\n--- Loading HelpSteer2 DPO ---")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nvidia/HelpSteer2", split="train", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    
    score_cols = ['helpfulness', 'correctness', 'coherence']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['composite_score'] = df[score_cols].sum(axis=1)
    df.dropna(subset=['composite_score'], inplace=True)

    dpo_data = []
    for _, group in df.groupby('prompt'):
        if len(group) == 2:
            r1, r2 = group.iloc[0], group.iloc[1]
            s1, s2 = r1['composite_score'], r2['composite_score']
            if s1 > s2:
                chosen, rejected = r1['response'], r2['response']
            elif s2 > s1:
                chosen, rejected = r2['response'], r1['response']
            else:
                continue
            dpo_data.append({'prompt': r1['prompt'], 'chosen': chosen, 'rejected': rejected, 'task': 'helpfulness_preference'})
    
    return pd.DataFrame(dpo_data)

def main():
    """Main function to download, process, and save DPO datasets."""
    tokenizer = get_tokenizer()
    max_tokens = 2048

    datasets = {
        "TaskSource DPO": load_tasksource_dpo(),
        "Chatbot Arena DPO": load_chatbot_arena_dpo(),
        "Gutenberg DPO": load_gutenberg_dpo(),
        "Math Step DPO": load_math_step_dpo(),
        "Helpfulness DPO": load_helpsteer_dpo(),
    }

    processed_dfs = {}
    for name, df in datasets.items():
        df_cleaned = clean_text_columns(df, name)
        df_token_filtered = filter_by_max_token_length(df_cleaned, name, tokenizer, max_tokens)
        processed_dfs[name] = filter_non_empty_essentials(df_token_filtered, name)

    # Configuration for combining datasets
    # Format: (DataFrame, num_entries_to_take, name)
    # -1 means take all entries
    combination_config = [
        (processed_dfs["TaskSource DPO"], -1, "TaskSource DPO"),
        (processed_dfs["Chatbot Arena DPO"], -1, "Chatbot Arena DPO"),
        (processed_dfs["Gutenberg DPO"], -1, "Gutenberg DPO"),
        (processed_dfs["Math Step DPO"], -1, "Math Step DPO"),
        (processed_dfs["Helpfulness DPO"], -1, "Helpfulness DPO"),
    ]

    final_df = combine_dpo_datasets(combination_config, shuffle_final=True, random_state=42)

    output_dir = Path("../data/preprocessed/DPO")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dpo_dataset_cleaned.pkl"
    final_df.to_pickle(output_path)
    print(f"\nFinal combined dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
