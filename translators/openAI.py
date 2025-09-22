"""
This script translates a column of text in a DataFrame using the OpenAI API.

It takes a DataFrame, a column name, source and target languages, and a list of models
to use for translation. It then translates the text in the specified column and
returns a DataFrame with the original text, translated text, and other metadata.

The script also calculates the cost of the translations and saves a report to a
text file.
"""

import pandas as pd
import argparse
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import os
from openai import OpenAI

def initialize_client():
    """Initializes the OpenAI client, ensuring the API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)

def translate(client: OpenAI, text: str, target_lang: str, source_lang: str, model: str, temperature: float = 0.0):
    """
    Translates a single text string using the OpenAI API.

    Args:
        client: The OpenAI client.
        text: The text to translate.
        target_lang: The target language.
        source_lang: The source language.
        model: The model to use for translation.
        temperature: The temperature for the translation.

    Returns:
        A tuple containing the translated text and the API usage statistics.
    """
    system_prompt = (
        f"You are a professional {source_lang} to {target_lang} translator, tasked with providing translations suitable for use in\n"
        f"a {target_lang}-speaking region. Your goal is to accurately convey the meaning and nuances of the original {source_lang}\n"
        f"text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Please translate the following {source_lang} text into {target_lang}:\n"
        f"{text}\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    translation = response.choices[0].message.content.strip()
    usage = response.usage

    return translation, usage

def list_translate(
    client: OpenAI,
    df_path: str,
    rows: int,
    output_path: str,
    column: str,
    src_lang: str,
    languages: Dict[str, str],
    models: List[str],
    export: bool = True,
):
    """
    Translates a list of texts in a DataFrame.

    Args:
        client: The OpenAI client.
        df_path: The path to the DataFrame.
        rows: The number of rows to translate.
        output_path: The path to save the translated DataFrame.
        column: The column to translate.
        src_lang: The source language.
        languages: A dictionary of target languages.
        models: A list of models to use for translation.
        export: Whether to export the translated DataFrame.

    Returns:
        A DataFrame with the translations.
    """
    print(f"DataFrame path: {df_path}")
    print(f"Rows to translate: {rows}")
    print(f"Column to translate: {column}")
    print(f"Source language: {src_lang}")
    print(f"Target language:\n{', '.join(languages.keys())}")
    print(f"Translation models: {models}")
    print(f"Export: {export}")

    df = pd.read_pickle(df_path)

    if rows != 0:
        print(f'df size: {df.shape}')
        df = df.head(rows)
        print(f'df small size: {df.shape}')

    all_translations = []
    total_stats = {}

    cost_per_million = {
        'gpt-3.5-turbo-0125': (0.50, 1.50),
        'gpt-4o-2024-08-06': (2.50, 10.00),
    }

    target_languages = [lang for lang in languages.keys() if lang != "English"]

    for model_name in models:
        input_cost_per_million, output_cost_per_million = cost_per_million.get(model_name, (0.0, 0.0))
        model_translations = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        print(f"\nTranslating using model: {model_name}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing with {model_name}"):
            input_text = row[column]

            for target_language in target_languages:
                translated_input, usage = translate(
                    client=client,
                    text=input_text,
                    target_lang=target_language,
                    source_lang=src_lang,
                    model=model_name,
                    temperature=0.0
                )

                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
                output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
                cost = input_cost + output_cost

                total_input_tokens += prompt_tokens
                total_output_tokens += completion_tokens
                total_cost += cost

                model_translations.append({
                    'org_prompt': input_text,
                    'trs_prompt': translated_input,
                    'org_lang': 'English',
                    'tar_lang': target_language,
                    'model': model_name
                })

        total_stats[model_name] = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_cost': total_cost,
        }

        print(f"Model {model_name} total cost: ${total_cost:.6f}")

        all_translations.extend(model_translations)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if export:
            model_data = pd.DataFrame(model_translations)
            output_dir = output_path
            os.makedirs(output_dir, exist_ok=True)
            output_pkl = f"{output_dir}translated_prompts_{model_name}_{date_str}.pkl"
            model_data.to_pickle(output_pkl)

    output_txt = "translation_cost.txt"
    with open(output_txt, "a") as f:
        f.write(f"\n=== Translation Cost Report {date_str} ===\n")
        for model_name, stats in total_stats.items():
            input_tokens = stats['input_tokens']
            output_tokens = stats['output_tokens']
            cost = stats['total_cost']
            f.write(f"Model: {model_name}\n")
            f.write(f"  Total input tokens: {input_tokens}\n")
            f.write(f"  Total output tokens: {output_tokens}\n")
            f.write(f"  Total cost (USD): ${cost:.6f}\n")
            f.write("\n")

    print(f"\nSaved cost report to {output_txt}")

    if not all_translations:
        return pd.DataFrame()

    final_df = pd.DataFrame(all_translations)
    return final_df

def main():
    """Main function to run the translation script."""
    parser = argparse.ArgumentParser(description="Translate a DataFrame column using the OpenAI API.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the input DataFrame pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output translations.")
    parser.add_argument("--column", type=str, default='source', help="Name of the column to translate.")
    parser.add_argument("--src_lang", type=str, default='English', help="Source language.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to translate (0 for all).")
    parser.add_argument("--export", type=bool, default=True, help="Whether to export the results.")
    args = parser.parse_args()

    client = initialize_client()

    LANGUAGES = {
        "Arabic": "arb_Arab",
        "English": "eng_Latn",
        "German": "deu_Latn",
        "Spanish (Mexico)": "spa_Latn",
        "Dutch": "nld_Latn",
        "Russian": "rus_Cyrl",
        "Chinese": "zho_Hans",
        "Japanese": "jpn_Jpan",
        "Icelandic": "isl_Latn",
    }

    MODELS = [
        "gpt-3.5-turbo-0125",
        "gpt-4o-2024-08-06",
    ]

    translations_df = list_translate(
        client=client,
        df_path=args.df_path,
        rows=args.rows,
        output_path=args.output_path,
        column=args.column,
        src_lang=args.src_lang,
        languages=LANGUAGES,
        models=MODELS,
        export=args.export,
    )

    print(translations_df.shape)
    print(translations_df.head(140))

if __name__ == "__main__":
    main()