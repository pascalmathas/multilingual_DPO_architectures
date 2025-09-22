"""
This script updates an existing DPO dataset with new chosen responses.
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

def load_existing_dpo_dataset(file_path):
    """Loads an existing DPO dataset from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {{}}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return {{}}
    
    dpo_records = {{}}
    for record in data:
        if 'chosen' in record and len(record['chosen']) > 0 and 'content' in record['chosen'][0]:
            prompt = record['chosen'][0]['content'].strip()
            dpo_records[prompt] = record
    return dpo_records

def main():
    """Main function to update the DPO dataset."""
    parser = argparse.ArgumentParser(description="Update a DPO dataset with new chosen responses.")
    parser.add_argument("--new_chosen_responses_path", type=str, required=True, help="Path to the JSON file with new chosen responses.")
    parser.add_argument("--existing_dpo_dataset_path", type=str, required=True, help="Path to the existing DPO dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the updated DPO dataset.")
    args = parser.parse_args()

    print("--- Loading New Chosen Responses ---")
    new_chosen_responses = load_json_responses(args.new_chosen_responses_path)
    print(f"Loaded {len(new_chosen_responses)} new chosen responses.")

    print("\n--- Loading Existing DPO Dataset ---")
    existing_dpo_records = load_existing_dpo_dataset(args.existing_dpo_dataset_path)
    print(f"Loaded {len(existing_dpo_records)} existing DPO records.")

    updated_dpo_records = []
    for prompt, original_record in existing_dpo_records.items():
        if prompt in new_chosen_responses:
            new_chosen_response = new_chosen_responses[prompt]
            updated_record = {{
                "chosen": [
                    {{"content": prompt, "role": "user"}},
                    {{"content": new_chosen_response, "role": "assistant"}}
                ],
                "rejected": original_record["rejected"]
            }}
            updated_dpo_records.append(updated_record)
        else:
            updated_dpo_records.append(original_record)

    print(f"\n--- Updated {len(updated_dpo_records)} DPO records ---")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dpo_records, f, ensure_ascii=False, indent=2)
    
    print(f"\nUpdated DPO dataset saved to: {args.output_path}")

if __name__ == "__main__":
    main()
