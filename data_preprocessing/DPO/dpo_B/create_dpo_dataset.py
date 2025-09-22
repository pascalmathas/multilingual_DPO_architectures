"""
This script builds a DPO dataset from chosen and rejected responses in JSON format.
"""

import json
import os
import argparse

def load_json_responses(file_path):
    """Loads responses from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {{}}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return {{}}
    
    responses = {{}}
    for item in data:
        prompt = item.get("original_prompt")
        response = item.get("generated_response")
        if prompt and response:
            responses[prompt.strip()] = response.strip()
    return responses

def main():
    """Main function to build the DPO dataset."""
    parser = argparse.ArgumentParser(description="Build a DPO dataset from chosen and rejected responses.")
    parser.add_argument("--chosen_responses_path", type=str, required=True, help="Path to the JSON file with chosen responses.")
    parser.add_argument("--rejected_responses_path", type=str, required=True, help="Path to the JSON file with rejected responses.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output DPO dataset.")
    args = parser.parse_args()

    print("--- Loading Chosen Responses ---")
    chosen_responses = load_json_responses(args.chosen_responses_path)
    print(f"Loaded {len(chosen_responses)} chosen responses.")

    print("\n--- Loading Rejected Responses ---")
    rejected_responses = load_json_responses(args.rejected_responses_path)
    print(f"Loaded {len(rejected_responses)} rejected responses.")

    dpo_records = []
    for prompt, chosen_response in chosen_responses.items():
        if prompt in rejected_responses:
            rejected_response = rejected_responses[prompt]
            dpo_record = {{
                "chosen": [
                    {{"content": prompt, "role": "user"}},
                    {{"content": chosen_response, "role": "assistant"}}
                ],
                "rejected": [
                    {{"content": prompt, "role": "user"}},
                    {{"content": rejected_response, "role": "assistant"}}
                ]
            }}
            dpo_records.append(dpo_record)

    print(f"\n--- Matched {len(dpo_records)} DPO records ---")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_records, f, ensure_ascii=False, indent=2)
    
    print(f"\nDPO dataset saved to: {args.output_path}")

if __name__ == "__main__":
    main()
