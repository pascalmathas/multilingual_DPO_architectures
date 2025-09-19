# Data Preprocessing

Contains scripts and utilities for preprocessing datasets.

## Purpose

Prepares raw data for Direct Preference Optimization (DPO) training. Downloads, cleans, filters, and structures data for the three architectures (A, B, C) described in the main README.

## Contents

- **DPO/**: Core scripts for creating DPO datasets.  
  - `download_clean_data.py`: Downloads and cleans the initial English DPO datasets.  
  - `dpo_A.py`: Architecture A — translates prompt, chosen, and rejected responses.  
  - `dpo_B/`: Architecture B — translates prompt and generates new chosen/rejected responses.  
  - `dpo_C.py`: Architecture C — translates prompt, generates new chosen response, translates rejected response.  
  - `dpo_prompt.py`: Handles prompts.  
  - `utils.py`: Utility functions.  

- **aya/**: Scripts for evaluating the Aya model.  
  - `aya_eval.py`: Evaluates Aya model performance.  

- **wmt24pp/**: Scripts for WMT24 parallel data.  
  - `wmt24pp_constants.py`: Constants for WMT24 scripts.  
  - `wmt24pp-xalma.py`: Processes WMT24 data with X-ALMA.  
  - `wmt24pp.py`: Processes WMT24 data.  

## How to Use

Download and clean the initial English DPO dataset:

```bash
python data_preprocessing/DPO/download_clean_data.py
```

This creates `dpo_dataset_cleaned.pkl` in `data/preprocessed/DPO/`, which is the starting point for the three architectures. Use the DPO scripts to generate datasets for each architecture.

## How it Fits into the Bigger Picture

These scripts are the first step in the pipeline. They output DPO datasets for each architecture, which are then used to fine-tune the Aya-23-8B model as described in the main README.
