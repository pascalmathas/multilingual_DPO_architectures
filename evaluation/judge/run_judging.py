
import argparse
import pandas as pd
import random
import os
import json
from datetime import datetime
from tqdm import tqdm
from evaluation.utils.helpers import (
    load_jsonl_to_dict,
    build_judge_prompt_openai,
    build_judge_prompt_qwen,
)
from evaluation.judge.judges import OpenAIJudge, VLLMJudge

LANGUAGE_MAP = {
    "arb": "Arabic", "deu": "German", "spa": "Spanish", "rus": "Russian",
    "zho": "Chinese", "jpn": "Japanese", "isl": "Icelandic", "nld": "Dutch",
    "eng": "English",
}

JUDGE_MODEL_COSTS = {
    "gpt-4o-2024-08-06": (5.00, 15.00),
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-3.5-turbo": (0.50, 1.50),
}

def run_evaluation(args):
    print("--- LLM Response Evaluation Job ---")
    print(f"Model A (Challenger): {args.model_a_name} (File: {args.file_path_model_a})")
    print(f"Model B (Baseline): {args.model_b_name} (File: {args.file_path_model_b})")
    print(f"Judge Model: {args.judge_model}")
    print(f"Output Directory: {args.output_dir}")
    print("-----------------------------------")

    data_model_a = load_jsonl_to_dict(args.file_path_model_a)
    data_model_b = load_jsonl_to_dict(args.file_path_model_b)

    if data_model_a is None or data_model_b is None:
        print("Failed to load data from one or both model files. Aborting evaluation.")
        return

    if args.shorten_data:
        # Shorten data logic here if needed
        pass

    evaluated_results = []
    total_judge_cost_usd = 0.0
    total_prompt_tokens_judge, total_completion_tokens_judge = 0, 0
    cost_per_million_input, cost_per_million_output = JUDGE_MODEL_COSTS.get(args.judge_model, (0.0, 0.0))

    if args.judge_type == 'openai':
        judge = OpenAIJudge(args.judge_model)
        build_judge_prompt = build_judge_prompt_openai
    elif args.judge_type == 'vllm':
        judge = VLLMJudge(args.judge_model)
        build_judge_prompt = build_judge_prompt_qwen
    else:
        raise ValueError(f"Invalid judge type: {args.judge_type}")

    print(f"\nStarting evaluation. Number of items to compare: {len(data_model_a)}")

    for key, record_a in tqdm(data_model_a.items(), desc="Evaluating responses"):
        if not (isinstance(key, tuple) and len(key) == 2): continue
        original_id, lang_code = key

        if key not in data_model_b:
            # Handle missing record in model B
            continue

        record_b = data_model_b[key]
        original_prompt = record_a.get("original_prompt")
        response_a_content = str(record_a.get("generated_response", ""))
        response_b_content = str(record_b.get("generated_response", ""))

        winner_model_name = "ERROR_UNKNOWN"
        judge_api_usage = None
        judge_choice_raw = None

        if original_prompt is None:
            winner_model_name = "ERROR_DATA_MISSING_PROMPT"
        else:
            language_name = LANGUAGE_MAP.get(lang_code, lang_code)
            presented_A_is_model_A = random.choice([True, False])

            if presented_A_is_model_A:
                prompt_arg_for_A, prompt_arg_for_B = response_a_content, response_b_content
            else:
                prompt_arg_for_A, prompt_arg_for_B = response_b_content, response_a_content

            judge_messages = build_judge_prompt(language_name, original_prompt, prompt_arg_for_A, prompt_arg_for_B)
            
            if args.judge_type == 'openai':
                judge_choice_raw, judge_api_usage = judge.get_preference(judge_messages)
            else:
                judge_choice_raw = judge.get_preference(judge_messages)

            if judge_choice_raw == "A":
                winner_model_name = args.model_a_name if presented_A_is_model_A else args.model_b_name
            elif judge_choice_raw == "B":
                winner_model_name = args.model_b_name if presented_A_is_model_A else args.model_a_name
            elif judge_choice_raw == "TIE":
                winner_model_name = "TIE"
            else:
                winner_model_name = judge_choice_raw

        if judge_api_usage and args.judge_type == 'openai':
            prompt_tokens, completion_tokens = judge_api_usage.prompt_tokens, judge_api_usage.completion_tokens
            total_prompt_tokens_judge += prompt_tokens
            total_completion_tokens_judge += completion_tokens
            input_cost = (prompt_tokens / 1_000_000) * cost_per_million_input
            output_cost = (completion_tokens / 1_000_000) * cost_per_million_output
            total_judge_cost_usd += (input_cost + output_cost)

        evaluated_results.append({
            "original_id": original_id, "language_code": lang_code,
            "language_name": LANGUAGE_MAP.get(lang_code, lang_code),
            "original_prompt": original_prompt,
            "generated_response_A_content": response_a_content, "model_A_name": args.model_a_name,
            "generated_response_B_content": response_b_content, "model_B_name": args.model_b_name,
            "winner": winner_model_name,
            "judge_model": args.judge_model,
            "judge_saw_A_as_model_A": presented_A_is_model_A,
            "judge_raw_preference": judge_choice_raw
        })

    print(f"\nEvaluation completed.")
    os.makedirs(args.output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if evaluated_results:
        safe_model_a_name = args.model_a_name.replace("/", "_").replace("-", "_").replace(".", "_")
        safe_model_b_name = args.model_b_name.replace("/", "_").replace("-", "_").replace(".", "_")
        safe_judge_name = args.judge_model.replace("/", "_").replace("-", "_").replace(".", "_")

        output_filename_json = f"evaluation_results_{safe_model_a_name}_vs_{safe_model_b_name}_by_{safe_judge_name}_{date_str}.json"
        output_filepath_json = os.path.join(args.output_dir, output_filename_json)

        try:
            with open(output_filepath_json, 'w', encoding='utf-8') as f_out:
                json.dump(evaluated_results, f_out, ensure_ascii=False, indent=4)
            print(f"Saved evaluated results to: {output_filepath_json}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

        df_results = pd.DataFrame(evaluated_results)
        if not df_results.empty:
            print("\n--- Overall Winner Distribution (Raw Counts) ---")
            winner_counts = df_results['winner'].value_counts(dropna=False)
            print(winner_counts.to_frame())

    if args.judge_type == 'openai':
        cost_report_path = os.path.join(args.output_dir, "judge_cost_report.txt")
        with open(cost_report_path, "a", encoding='utf-8') as f_cost:
            f_cost.write(f"\n=== Judge LLM Cost Report: {args.model_a_name} vs {args.model_b_name} (Judged by {args.judge_model}) - {date_str} ===\n")
            f_cost.write(f"Judge Model: {args.judge_model}\n")
            f_cost.write(f"  Total Prompt Tokens: {total_prompt_tokens_judge}, Total Completion Tokens: {total_completion_tokens_judge}\n")
            f_cost.write(f"  Estimated Total Cost (USD): ${total_judge_cost_usd:.6f}\n")
            f_cost.write("-" * 50 + "\n")
        print(f"Appended cost report to: {cost_report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--model_a_name", type=str, required=True, help="Name of the challenger model.")
    parser.add_argument("--file_path_model_a", type=str, required=True, help="Path to the challenger model's generation file.")
    parser.add_argument("--model_b_name", type=str, required=True, help="Name of the baseline model.")
    parser.add_argument("--file_path_model_b", type=str, required=True, help="Path to the baseline model's generation file.")
    parser.add_argument("--judge_model", type=str, required=True, help="Name of the judge model.")
    parser.add_argument("--judge_type", type=str, required=True, choices=['openai', 'vllm'], help="Type of judge to use.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("--shorten_data", action="store_true", help="Shorten the data for testing purposes.")

    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
