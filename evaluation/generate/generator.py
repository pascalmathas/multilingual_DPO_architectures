
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import os
import json
from vllm import SamplingParams
from evaluation.utils.helpers import initialize_model, unload_model

class ModelGenerator:
    def __init__(self, model_config: dict, batch_size: int, max_tokens: int, temperature: float):
        self.model_config = model_config
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = None

    def generate_batch(self, prompts_batch: List[str]) -> list:
        """Generates responses for a batch of prompts."""
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.model_config["stop_tokens"],
        )
        outputs = self.llm.generate(prompts=prompts_batch, sampling_params=sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def stream_batch_generate(
        self,
        df_path: str,
        rows_to_process: int,
        output_dir: str,
        prompt_column: str,
        lang_column: str,
        export: bool = True,
    ) -> pd.DataFrame:
        print(f"--- {self.model_config['model_name_for_print']} Model Generation Job ---")
        print(f"Model Name: {self.model_config['model_name_for_print']}")
        print(f"Model Repository: {self.model_config['model_repo_id']}")
        print(f"Input data path: {df_path}")
        print(f"Rows to process: {rows_to_process if rows_to_process != 0 else 'All'}")
        print(f"Prompt column: {prompt_column}")
        print(f"Language column: {lang_column}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max tokens per generation: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        print(f"Export results: {export}")
        print(f"-----------------------------------")

        try:
            print(f"Attempting to read JSONL file from: {df_path}")
            with open(df_path, 'r', encoding='utf-8') as f:
                data_list = [json.loads(line) for line in f]
            df = pd.DataFrame(data_list)
            print(f"Successfully loaded {len(df)} records from {df_path}.")
        except FileNotFoundError:
            print(f"Error: Input file not found at {df_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading JSONL file {df_path}: {e}")
            return pd.DataFrame()

        if df.empty:
            print("Error: Input DataFrame is empty after loading.")
            return pd.DataFrame()

        if rows_to_process != 0 and rows_to_process < len(df):
            print(f'Original df size: {len(df)}')
            df = df.head(rows_to_process)
            print(f'Processing first {rows_to_process} rows. New df size: {len(df)}')
        elif rows_to_process == 0:
            print(f'Processing all {len(df)} rows from the input file.')
        else:
            print(f'Processing {len(df)} rows (all available or specified number exceeds available).')

        if prompt_column not in df.columns:
            print(f"Error: Prompt column '{prompt_column}' not found in the DataFrame.")
            return pd.DataFrame()
        if lang_column not in df.columns:
            print(f"Warning: Language column '{lang_column}' not found. Language info will be missing in output.")

        all_generations: List[Dict[str, Any]] = []
        try:
            self.llm = initialize_model(self.model_config["model_repo_id"], self.model_config["model_name_for_print"])

            num_batches = (len(df) + self.batch_size - 1) // self.batch_size

            for i in tqdm(range(0, len(df), self.batch_size), total=num_batches, desc=f"Generating with {self.model_config['model_name_for_print']}"):
                batch_df = df.iloc[i:i+self.batch_size]

                batch_original_prompts = batch_df[prompt_column].astype(str).fillna("").tolist()
                batch_languages = batch_df[lang_column].astype(str).fillna("UNKNOWN").tolist() if lang_column in df.columns else ["UNKNOWN"] * len(batch_df)

                prompts_to_build_for_model = []
                indices_to_process = []

                for idx, original_prompt_text in enumerate(batch_original_prompts):
                    if original_prompt_text.strip():
                        prompts_to_build_for_model.append(original_prompt_text)
                        indices_to_process.append(idx)

                current_batch_model_prompts = [self.model_config["prompt_builder"](text) for text in prompts_to_build_for_model]
                batch_generated_responses = [""] * len(batch_original_prompts)

                if current_batch_model_prompts:
                    try:
                        generated_texts = self.generate_batch(current_batch_model_prompts)
                        for i_processed, gen_text in zip(indices_to_process, generated_texts):
                            batch_generated_responses[i_processed] = gen_text
                    except Exception as batch_error:
                        print(f"\nError during generation batch (indices {i+min(indices_to_process) if indices_to_process else i} to {i+max(indices_to_process) if indices_to_process else i+len(batch_original_prompts)-1}): {batch_error}")
                        for i_processed in indices_to_process:
                             batch_generated_responses[i_processed] = f"ERROR: {batch_error}"

                for idx_in_batch_df in range(len(batch_df)):
                    original_prompt = batch_original_prompts[idx_in_batch_df]
                    generated_response = batch_generated_responses[idx_in_batch_df]
                    language = batch_languages[idx_in_batch_df]
                    original_id = batch_df.iloc[idx_in_batch_df].get('id', None)

                    if not original_prompt.strip():
                        generated_response = "SKIPPED: Empty input prompt"

                    record = {
                        'original_prompt': original_prompt,
                        'generated_response': generated_response,
                        'language': language,
                        'model': self.model_config['model_name_for_print'],
                    }
                    if original_id is not None:
                        record['original_id'] = original_id
                    all_generations.append(record)

        except Exception as e:
            print(f"ERROR: Critical error during model initialization or processing loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.llm:
                unload_model(self.llm)
                print(f"\nFinished processing with {self.model_config['model_name_for_print']}. Model unloaded.")
            else:
                print(f"\nProcessing finished for {self.model_config['model_name_for_print']}. Model was not loaded or load failed.")

        if not all_generations:
            print("\nNo generations were created.")
            return pd.DataFrame()

        final_df = pd.DataFrame(all_generations)
        print(f"\nTotal generation records collected: {len(final_df)}")

        if export and not final_df.empty:
            try:
                date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.makedirs(output_dir, exist_ok=True)

                safe_model_name = self.model_config['model_name_for_print'].replace("/", "_").replace("-", "_").replace(".", "_")
                output_filename_pkl = f"generated_responses_{safe_model_name}_{date_str}.pkl"
                output_filepath_pkl = os.path.join(output_dir, output_filename_pkl)
                final_df.to_pickle(output_filepath_pkl)
                print(f"\nSaved all results to {output_filepath_pkl}")

                output_filename_jsonl = f"generated_responses_{safe_model_name}_{date_str}.jsonl"
                output_filepath_jsonl = os.path.join(output_dir, output_filename_jsonl)
                final_df.to_json(output_filepath_jsonl, orient='records', lines=True, force_ascii=False)
                print(f"Saved all results to {output_filepath_jsonl}")

            except Exception as e:
                print(f"\nError saving results: {e}")

        return final_df
