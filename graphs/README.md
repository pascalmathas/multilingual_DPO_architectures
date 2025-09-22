# Graphs

Contains graphs and scripts for generating them.

## Purpose

Stores visualizations of experimental results for the paper.

## Contents

- `pipeline_overview.png`: Overview of the experimental setup.  

- **`analysis/`**: Graphs and scripts for result analysis.  
  - `language_improvement_chart_wide.png`: Bar chart of model performance improvements across languages.  
  - `per-language-char.py`: Generates `language_improvement_chart_wide.png`.  

- **`evaluation/`**: Graphs for model evaluation.  
  - `bleu_sample_size.png`: BLEU score vs. sample size.  
  - `bleu_scores_heatmap.png`: BLEU scores heatmap for all target languages and models.  
  - `chrf_sample_size.png`: CHRF++ score vs. sample size.  
  - `chrf_scores_heatmap.png`: CHRF++ scores heatmap for all target languages and models.  

## How to Use

Generate the language improvement chart with:

```bash
python graphs/analysis/per-language-char.py
```

This outputs `language_improvement_chart_wide.png` in `graphs/analysis/`.

## How it Fits into the Bigger Picture

These graphs visualize evaluation results and support the paper’s conclusions.