# WMT24 Evaluation

Contains scripts for evaluating translation quality using WMT24 data and metrics.

## Purpose

Pipeline for assessing translation model performance on WMT24 benchmarks using BLEU, CHRF, and visualizations.

## Contents

- `plotting.py`: Functions to generate evaluation visualizations, e.g., score heatmaps.  
- `run_evaluation.py`: Orchestrates the WMT24 evaluation from data loading to metrics and plots.  
- `utils.py`: Utility functions supporting evaluation, like `evaluate_translations` and `analyze_results_by_model`.  

## How to Use

Run the WMT24 evaluation:

```bash
python wmt24_evaluation/run_evaluation.py \
    --wmt_pickle_file ../data/preprocessed/wmt24pp/wmt24pp_small.pkl \
    --model_name x-alma-git \
    --output_pickle_file ../data/translations/wmt24pp/x_alma_git_translations.pkl
```

This will:
1. Load WMT data.
2. Process translations for the specified model.
3. Compute BLEU and CHRF scores.
4. Print results.
5. Save detailed results to CSV.
6. Generate score heatmaps.

## How it Fits into the Bigger Picture

Quantitatively evaluates translation quality for models across architectures.