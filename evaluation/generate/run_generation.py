import argparse
import pandas as pd
from evaluation.generate.config import MODEL_CONFIGS
from evaluation.generate.generator import ModelGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate text using a specified model.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="The name of the model to use for generation.")
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input data file (JSONL format).")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save the generated text.")
    parser.add_argument("--prompt_column", type=str, default="inputs", help="The name of the column containing the prompts.")
    parser.add_argument("--lang_column", type=str, default="language", help="The name of the column containing the language information.")
    parser.add_argument("--rows_to_process", type=int, default=0, help="The number of rows to process from the input file (0 for all).")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for generation.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.3, help="The temperature for sampling.")
    parser.add_argument("--export", action="store_true", help="Export the generated text to files.")

    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model_name]

    generator = ModelGenerator(
        model_config=model_config,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    generations_df = generator.stream_batch_generate(
        df_path=args.input_file,
        rows_to_process=args.rows_to_process,
        output_dir=args.output_dir,
        prompt_column=args.prompt_column,
        lang_column=args.lang_column,
        export=args.export,
    )

    if not generations_df.empty:
        print("\n--- Generation Summary ---")
        print(f"Generated {len(generations_df)} records.")
        print("Sample generations (first 5):")
        with pd.option_context('display.max_colwidth', 100, 'display.width', 1000):
            print(generations_df.head(5))

if __name__ == "__main__":
    main()
