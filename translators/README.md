# Translators

Contains Python scripts for various translation models and APIs.

## Purpose

Standardized interface for translation services (OpenAI, vLLM models), abstracting API specifics.

## Contents

- `openAI.py`: Interface for OpenAI API translations.  
- `vllm_command.py`: Translator using vLLM Command model.  
- `vllm_gemma.py`: Translator using vLLM Gemma model.  
- `vllm_nllb.py`: Translator using vLLM NLLB model.  
- `vllm_qwen.py`: Translator using vLLM Qwen model.  
- `vllm_tower.py`: Translator using vLLM Tower model.  
- `vllm_x-alma.py`: Translator using vLLM X-ALMA model.  

## How to Use

Instantiate a translator class and call its `translate_batch` method. Example with OpenAI:

```python
from translators.openAI import OpenAITranslator

translator = OpenAITranslator(model_name="gpt-4o")  # Example model

texts = ["Hello, how are you?", "What is the weather like today?"]
target_language = "fr"

translated_texts = translator.translate_batch(texts, target_language)

for original, translated in zip(texts, translated_texts):
    print(f"Original: {original} -> Translated ({target_language}): {translated}")
```

## How it Fits into the Bigger Picture

Used in data preprocessing and evaluation to expand the English DPO dataset into multiple languages and to support multilingual evaluation.