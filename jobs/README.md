# Jobs

Contains job submission scripts for experiments.

## Purpose

Scripts for running experiments on a Slurm cluster, including DPO training, evaluation, and generation.

## Contents

- **`dpo/`**: DPO training jobs.  
  - `DPO_aya.job`: Runs DPO training on the Aya model.  

- **`env/`**: Environment setup jobs.  
  - `env_vllm.job`: Sets up environment for `vllm`.  
  - `env_xalma.job`: Sets up environment for `xalma`.  

- **`evaluation/`**: Evaluation jobs.  
  - `judge.job`: Runs the judging process.  

- **`generation/`**: Generation jobs.  
  - `vllm_generate.job`: Runs generation with `vllm`.  

- **`xalma/`**: X-ALMA jobs.  
  - `run_xalma_dpo_chosen.job`: Runs DPO chosen job.  
  - `run_xalma_dpo_prompt.job`: Runs DPO prompt job.  
  - `run_xalma_dpo_rejected.job`: Runs DPO rejected job.  
  - `run_xalma_eval.job`: Runs X-ALMA evaluation job.  

## How to Use

Submit a job with `sbatch`. Example for DPO training on Aya:

```bash
sbatch jobs/dpo/DPO_aya.job
```

Ensure environment variables are set in the script, including Hugging Face token, WandB token, project and cache directories, and Singularity image path.

## How it Fits into the Bigger Picture

Automates experiments on a Slurm cluster. Outputs are used for evaluation and graph generation to assess architecture performance.