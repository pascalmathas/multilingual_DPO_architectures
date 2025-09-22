"""
This script extracts prompts from a DPO dataset and saves them to a JSONL file.
"""

import json
import os
import argparse

def main():
    """Main function to extract prompts from a DPO dataset."""
    parser = argparse.ArgumentParser(description="Extract prompts from a DPO dataset.")
    parser.add_argument("--input_dpo_json_path", type=str, required=True, help="Path to the input DPO JSON file.")
    parser.add_argument("--output_prompt_only_json_path", type=str, required=True, help="Path to save the output prompt-only JSONL file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dpo_json_path):
        print(f"Error: Input file not found at {args.input_dpo_json_path}")
        return

    with open(args.input_dpo_json_path, 'r', encoding='utf-8') as f:
        try:
            dpo_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {args.input_dpo_json_path}")
            return

    prompts = []
    for record in dpo_data:
        if 'chosen' in record and len(record['chosen']) > 0 and 'content' in record['chosen'][0]:
            prompt = record['chosen'][0]['content']
            prompts.append({"prompt": prompt})

    print(f"Extracted {len(prompts)} prompts.")

    output_dir = os.path.dirname(args.output_prompt_only_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_prompt_only_json_path, 'w', encoding='utf-8') as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Prompt-only dataset saved to: {args.output_prompt_only_json_path}")

if __name__ == "__main__":
    main()
