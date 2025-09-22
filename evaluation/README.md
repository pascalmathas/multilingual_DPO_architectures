# Evaluation

Contains scripts for model evaluation.

## Purpose

Framework for generating and judging model responses.

## Contents

- **`generate/`**: Scripts for generating model responses.  
  - `config.py`: Generation configuration, including model settings.  
  - `generator.py`: Main generation script.  
  - `run_generation.py`: Runs the generation process.  

- **`judge/`**: Scripts for judging generated responses.  
  - `judges.py`: Logic for different judges.  
  - `run_judging.py`: Runs the judging process.  

- **`utils/`**: Helper functions.  
  - `helpers.py`: Utility functions for evaluation.  

## How to Use

### 1. Generation

Generate responses with:

```bash
python evaluation/generate/run_generation.py --model_name <model_name> --input_file <input_file> --output_dir <output_dir>
```

### 2. Judging

Judge generated responses with:

```bash
python evaluation/judge/run_judging.py --input_file <generated_responses> --output_dir <output_dir>
```

## How it Fits into the Bigger Picture

Evaluates models trained with `data_preprocessing` datasets. Results are used to compare architectures and assess performance.