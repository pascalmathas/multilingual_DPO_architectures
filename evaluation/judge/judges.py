import time
from openai import OpenAI
import os
from typing import List, Dict
from vllm import LLM, SamplingParams

class Judge:
    def __init__(self, judge_model_name: str):
        self.judge_model_name = judge_model_name
    
    def get_preference(self, messages, **kwargs):
        raise NotImplementedError

class OpenAIJudge(Judge):
    def __init__(self, judge_model_name: str, api_key: str = None):
        super().__init__(judge_model_name)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def get_preference(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_retries: int = 3, retry_delay: int = 5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model_name,
                    temperature=temperature,
                    messages=messages,
                    max_tokens=20
                )
                preference_text = response.choices[0].message.content.strip()
                usage = response.usage
                cleaned_text = preference_text.upper()
                choice = ""
                if "PREFERRED:" in cleaned_text:
                    choice = cleaned_text.split("PREFERRED:")[-1].strip().strip("'\"")
                else:
                    choice = cleaned_text.strip().strip("'\"")
                
                if choice == "A": 
                    return "A", usage
                if choice == "B": 
                    return "B", usage
                if choice == "TIE": 
                    return "TIE", usage
                
                print(f"Warning: Unexpected judge response format: '{preference_text}'. Defaulting to TIE_UNPARSED.")
                return "TIE_UNPARSED", usage
                
            except Exception as e:
                print(f"Error calling OpenAI API (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return "ERROR_API_CALL", None
        
        return "ERROR_MAX_RETRIES", None

class VLLMJudge(Judge):
    def __init__(self, judge_model_name: str):
        super().__init__(judge_model_name)
        self.llm = LLM(
            model=judge_model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="float16"
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            stop=["<|im_end|>", "\n\n", "<|endoftext|>"],
        )
    
    def get_preference(self, prompt: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                outputs = self.llm.generate([prompt], self.sampling_params)
                preference_text = outputs[0].outputs[0].text.strip()
                cleaned_text = preference_text.upper()
                choice = ""
                if "PREFERRED:" in cleaned_text:
                    choice = cleaned_text.split("PREFERRED:")[-1].strip().strip("'\"")
                else:
                    choice = cleaned_text.strip().strip("'\"")
                
                if choice == "A": 
                    return "A"
                if choice == "B": 
                    return "B"
                if choice == "TIE": 
                    return "TIE"
                
                print(f"Warning: Unexpected judge response format: '{preference_text}'. Defaulting to TIE_UNPARSED.")
                return "TIE_UNPARSED"
                
            except Exception as e:
                print(f"Error during vLLM generation (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return "ERROR_VLLM_GENERATION"
        
        return "ERROR_MAX_RETRIES"