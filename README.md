<h1 align="left">Synthetic Data for Reinforcement Learning from Human Feedback</h1>
<h3 align="left">A Comparison of Architectures for Creating Synthetic Multilingual Preference Datasets</h3>


This repository contains the code and findings for the paper "A Comparison of Architectures for Creating Synthetic Multilingual Preference Datasets." 

Read the full paper by clicking the icon below.

[![View PDF](https://img.shields.io/badge/Paper-PDF-red)](https://pascalmathas.github.io/multilingual_DPO_architectures/paper/Mathas%2C%20Pascal.%20(2025)%20Synthetic%20Data%20for%20Reinforcement%20Learning%20from%20Human%20Feedback%3A%20A%20Comparison%20of%20Architectures%20for%20Creating%20Synthetic%20Multilingual%20Preference%20Datasets.pdf)

<div align="center" style="margin: 20px 0;">
    <figure style="margin: 0; padding: 0;">
        <img src="graphs/pipeline_overview.png" 
             alt="Experimental pipeline overview showing the process from English DPO dataset through synthetic expansion to 9 languages, following three architectural approaches, training different Aya model variants, and assessing performance against baselines" 
             style="width: 100%; max-width: 100%; height: auto; max-height: 40vh; object-fit: contain; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <figcaption style="margin-top: 10px; font-weight: bold; text-align: center; max-width: 100%; line-height: 1.4;">
            An overview of the experimental setup used in this paper. All architectures begin with the same custom English DPO dataset. We then synthetically expand this dataset to 9 languages, following three different architectural approaches. Next, we train an LLM with each dataset, resulting in the different Aya model variants. Finally, we assess the architectures' performance against three baselines. The red outline highlights the important differences in the architectures.
        </figcaption>
    </figure>
</div>

## Abstract

In this paper, we compare three architectures for creating a synthetic multilingual preference dataset. All architectures are based on a custom-created English DPO dataset. In Architecture A, we translate everything (prompt, chosen, and rejected responses) into the target language using the same model. In Architecture B, we translate only the prompt and generate the chosen and rejected responses using models of differing quality (high-quality for chosen, low-quality for rejected). In Architecture C, we translate only the prompt and generate only the chosen responses, while the rejected responses are translated from English using a lower-quality model to intentionally introduce translation artifacts. We find that Architectures B and C, even with very little data, improve the multilingual performance of the Aya-23-8B model, whereas Architecture A does not. The primary reason is that B and C provide the model with more natural, in-target-language-generated chosen responses, thereby avoiding English biases.

## Introduction

Preference optimization through Reinforcement Learning from Human Feedback (RLHF) has proven beneficial for training large language models (LLMs). However, this has primarily been applied to common languages such as English and Chinese, since most preference optimization datasets are available only in these languages. To improve the alignment of LLMs with humans in more diverse and less common languages, these models should be fine-tuned through RLHF using multilingual data.

A key issue is the high computational cost of obtaining human-annotated preference data, particularly in multiple languages. One solution is to generate synthetic preference data. In this paper, we explore which architecture is best to use. We examine three architectures, all of which can be used to create a preference dataset.

*   **(A) P-trans and R-trans.** In this architecture, we translate our English preference optimization dataset into *x* target languages. This means using the same translation model to translate the prompt, the chosen response, and the rejected response.
*   **(B) P-trans and R-gen.** In this architecture, we translate only the prompts from our English preference optimization dataset into *x* target languages. We then generate the chosen and rejected responses in the target languages. To better differentiate the two responses, we use a higher-quality model for the chosen responses and a lower-quality model for the rejected responses.
*   **(C) R-gen and R-trans.** In this architecture, we again translate only the prompts from our English preference optimization dataset into *x* target languages. We then generate only the chosen responses in the target languages. The English rejected responses from the original dataset are translated into the target languages. This architecture differentiates the chosen and rejected responses by introducing translation artifacts into the rejected responses.

The main research question of this paper is therefore as follows: "How do the proposed architectures (P-trans/R-trans, P-trans/R-gen, and R-gen/R-trans) compare in their effectiveness for creating multilingual synthetic preference datasets when used to fine-tune an LLM via Direct Preference Optimization?"

## Method

### Datasets
We construct a base English preference dataset that serves as the foundation for the three architectures. To ensure that the dataset includes a variety of tasks, we combine five existing preference datasets: Chatbot Arena, Nvidia/HelpSteer2, Tasksource/Dpo-pairs, Gutenberg-DPO, and Math-Step-DPO. The final English preference dataset contains 1,250 entries.

### Target Languages
We selected eight target languages for both the assessment of translation quality and the creation of the synthetic multilingual preference dataset. These languages are German, Dutch, Icelandic, Spanish, Arabic, Chinese, Japanese, and Russian.

<div align="center">
<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse;">
<thead>
<tr style="border-top: 3px solid black; border-bottom: 1px solid black;">
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Code</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Lang2LP</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Language</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Script</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Family</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Subgrouping</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;">ar</td>
<td style="padding: 8px;">en-ar_EG</td>
<td style="padding: 8px;">Arabic</td>
<td style="padding: 8px;">Arabic</td>
<td style="padding: 8px;">Afro-Asiatic</td>
<td style="padding: 8px;">Semitic</td>
</tr>
<tr>
<td style="padding: 8px;">zh</td>
<td style="padding: 8px;">en-zh_CN</td>
<td style="padding: 8px;">Chinese</td>
<td style="padding: 8px;">Han & Hant</td>
<td style="padding: 8px;">Sino-Tibetan</td>
<td style="padding: 8px;">Sinitic</td>
</tr>
<tr>
<td style="padding: 8px;">de</td>
<td style="padding: 8px;">en-de_DE</td>
<td style="padding: 8px;">German</td>
<td style="padding: 8px;">Latin</td>
<td style="padding: 8px;">Indo-European</td>
<td style="padding: 8px;">Germanic</td>
</tr>
<tr>
<td style="padding: 8px;">is</td>
<td style="padding: 8px;">en-is_IS</td>
<td style="padding: 8px;">Icelandic</td>
<td style="padding: 8px;">Latin</td>
<td style="padding: 8px;">Indo-European</td>
<td style="padding: 8px;">Germanic</td>
</tr>
<tr>
<td style="padding: 8px;">ja</td>
<td style="padding: 8px;">en-ja_JP</td>
<td style="padding: 8px;">Japanese</td>
<td style="padding: 8px;">Japanese</td>
<td style="padding: 8px;">Japonic</td>
<td style="padding: 8px;">Japanesic</td>
</tr>
<tr>
<td style="padding: 8px;">nl</td>
<td style="padding: 8px;">en-nl_NL</td>
<td style="padding: 8px;">Dutch</td>
<td style="padding: 8px;">Latin</td>
<td style="padding: 8px;">Indo-European</td>
<td style="padding: 8px;">Germanic</td>
</tr>
<tr>
<td style="padding: 8px;">ru</td>
<td style="padding: 8px;">en-ru_RU</td>
<td style="padding: 8px;">Russian</td>
<td style="padding: 8px;">Cyrillic</td>
<td style="padding: 8px;">Indo-European</td>
<td style="padding: 8px;">Balto-Slavic</td>
</tr>
<tr style="border-bottom: 3px solid black;">
<td style="padding: 8px;">es</td>
<td style="padding: 8px;">en-es_MX</td>
<td style="padding: 8px;">Spanish</td>
<td style="padding: 8px;">Latin</td>
<td style="padding: 8px;">Indo-European</td>
<td style="padding: 8px;">Italic</td>
</tr>
</tbody>
<caption style="caption-side: bottom; margin-top: 10px; font-weight: bold; text-align: center;">
Selected target languages for the experiments of this paper. <strong>code</strong>: language identifying code. <strong>lang2lp</strong>: English to target language translation code. Table based on Dang et al. (2024).
</caption>
</table>
</div>

### Architectures
*   **Architecture A:** We translate the prompts, chosen responses, and rejected responses into the eight target languages using the X-ALMA model.
*   **Architecture B:** We use translated prompts from architecture A. For the chosen responses, we use the multilingual-optimized Cohere Command-R 35B model. For the rejected responses, we use the smaller Qwen2.5 0.5B Instruct model.
*   **Architecture C:** We use the translated prompts from architecture A and the generated chosen responses from architecture B. For the rejected responses, we translate the English rejected responses using the worst-performing model, Facebook's NLLB-200 3.3B model.

### Models

<div align="center">
<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse;">
<thead>
<tr style="border-top: 3px solid black; border-bottom: 1px solid black;">
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Canonical Name</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Model Identifier</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Size</strong></th>
<th style="font-weight: bold; text-align: left; padding: 8px;"><strong>Repo</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;">Gemma-3</td>
<td style="padding: 8px;">gemma-3-12b-it</td>
<td style="padding: 8px;">12B</td>
<td style="padding: 8px;"><a href="https://huggingface.co/google/gemma-3-12b-it" target="_blank">HuggingFace</a></td>
</tr>
<tr>
<td style="padding: 8px;">GPT-3.5 Turbo</td>
<td style="padding: 8px;">gpt-3.5-turbo-0125</td>
<td style="padding: 8px;">≈ 175B</td>
<td style="padding: 8px;"><a href="https://platform.openai.com/docs/models/gpt-3.5-turbo" target="_blank">OpenAI</a></td>
</tr>
<tr>
<td style="padding: 8px;">GPT-4o</td>
<td style="padding: 8px;">gpt-4o-2024-08-06</td>
<td style="padding: 8px;">≈ 1.8T</td>
<td style="padding: 8px;"><a href="https://platform.openai.com/docs/models/gpt-4o" target="_blank">OpenAI</a></td>
</tr>
<tr>
<td style="padding: 8px;">NLLB-200</td>
<td style="padding: 8px;">nllb-200-3.3B</td>
<td style="padding: 8px;">3.3B</td>
<td style="padding: 8px;"><a href="https://huggingface.co/facebook/nllb-200-3.3B" target="_blank">HuggingFace</a></td>
</tr>
<tr>
<td style="padding: 8px;">Qwen 2.5</td>
<td style="padding: 8px;">Qwen2.5-14B-Instruct</td>
<td style="padding: 8px;">14B</td>
<td style="padding: 8px;"><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct" target="_blank">HuggingFace</a></td>
</tr>
<tr>
<td style="padding: 8px;">TowerInstruct</td>
<td style="padding: 8px;">TowerInstruct-13B-v0.1</td>
<td style="padding: 8px;">13B</td>
<td style="padding: 8px;"><a href="https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1" target="_blank">HuggingFace</a></td>
</tr>
<tr>
<td style="padding: 8px;">X-ALMA</td>
<td style="padding: 8px;">X-ALMA</td>
<td style="padding: 8px;">29B</td>
<td style="padding: 8px;"><a href="https://huggingface.co/haoranxu/X-ALMA" target="_blank">HuggingFace</a></td>
</tr>
<tr style="border-bottom: 3px solid black;">
<td style="padding: 8px;">X-ALMA-BS</td>
<td style="padding: 8px;">X-ALMA w/ Beam Search</td>
<td style="padding: 8px;">29B</td>
<td style="padding: 8px;"><a href="https://github.com/fe1ixxu/ALMA" target="_blank">GitHub</a></td>
</tr>
</tbody>
<caption style="caption-side: bottom; margin-top: 10px; font-weight: bold; text-align: center;">
The models used in this paper with identifiers, parameter size, and source links.
</caption>
</table>
</div>

### Direct Preference Optimization
After creating datasets A, B, and C, we fine-tune the multilingual-optimized Aya-23-8B model using the OpenRLHF Python library.

### LLM as a Judge
To evaluate each of the Aya variants, we assess their win rates against three state-of-the-art models: (a) Meta Llama 3 8B Instruct, (b) Gemma 1.1 7B Instruct, and (c) Mistral 7B Instruct v0.3. We use the Aya Evaluation Suite as the dataset for generating responses from the Aya model variants. For each supported language, we replace the machine-translated data with the corresponding human-edited entries. We then sample 150 entries for each of our preference dataset languages, resulting in an evaluation dataset of 1,350 entries. To assess the win rates of the Aya-Base and DPO variants, we use the Qwen2.5 32B Instruct model. To ensure stability, we report the average win rates over five runs. To mitigate positional bias when presenting the responses of models A and B, we randomize their order. The judge model is instructed to select its preferred response: A, B, or a Tie.

## Results

### Assessing Translation Quality
We observe that, for both BLEU and CHRF++ metrics, the models generally follow the same performance trend. The best-performing model is GPT-4o. The best-performing open-source model is X-ALMA with beam search enabled. Significantly smaller models, such as the NLLB-200 3.3B, perform the worst.

**BLEU** **scores for all target languages and tested models.**
![BLEU Scores Heatmap](graphs/evaluation/bleu_scores_heatmap.png)

**CHRF++ scores for all target languages and tested models.**
![CHRF++ Scores Heatmap](graphs/evaluation/chrf_scores_heatmap.png)

### Direct Preference Optimization
We find that architectures B and C, even with very little data, manage to improve the multilingual performance of the Aya-23-8B model, whereas architecture A does not. Aya-B improves performance over Aya-Base across all comparisons, with a gain of 5.69% against Llama-3, 3.43% against Gemma, and 6.14% against Mistral. Aya-C shows similar improvements.

<div align="center">
<table border="1" cellspacing="0" cellpadding="5">
<caption style="margin-bottom: 10px; font-weight: bold; text-align: center;">
Win-loss for the Aya-Base model and DPO-optimized variants against the Meta Llama 3 8B Instruct, Gemma 1.1 7B Instruct, and Mistral 7B Instruct v0.3 models. Averages across 9 languages. Scores reported over 5 runs. Table based on Dang et al. (2024).
</caption>
<thead>
<tr>
<th></th>
<th>Model</th>
<th>Win%</th>
<th>Loss%</th>
<th>ΔW-L%</th>
</tr>
</thead>
<tbody>
<!-- Base -->
<tr>
<td rowspan="3"><b>Base</b></td>
<td>Llama-3</td>
<td>64.27%</td>
<td>35.27%</td>
<td style="background-color:#228B22; color:white;">28.99%</td>
</tr>
<tr>
<td>Gemma-1.1</td>
<td>69.56%</td>
<td>29.97%</td>
<td style="background-color:#006400; color:white;">39.59%</td>
</tr>
<tr>
<td>Mistral-v0.3</td>
<td>69.10%</td>
<td>29.69%</td>
<td style="background-color:#006400; color:white;">39.41%</td>
</tr>
<!-- DPO A -->
<tr>
<td rowspan="3"><b>DPO A</b></td>
<td>Llama-3</td>
<td>62.16%</td>
<td>37.50%</td>
<td style="background-color:#32CD32; color:white;">24.67%</td>
</tr>
<tr>
<td>Gemma-1.1</td>
<td>67.44%</td>
<td>32.00%</td>
<td style="background-color:#228B22; color:white;">35.44%</td>
</tr>
<tr>
<td>Mistral-v0.3</td>
<td>68.13%</td>
<td>30.99%</td>
<td style="background-color:#2E8B57; color:white;">37.14%</td>
</tr>
<!-- DPO B -->
<tr>
<td rowspan="3"><b>DPO B</b></td>
<td>Llama-3</td>
<td>67.08%</td>
<td>32.40%</td>
<td style="background-color:#228B22; color:white;">34.68%</td>
</tr>
<tr>
<td>Gemma-1.1</td>
<td>71.29%</td>
<td>28.27%</td>
<td style="background-color:#004d00; color:white;">43.02%</td>
</tr>
<tr>
<td>Mistral-v0.3</td>
<td>72.32%</td>
<td>26.77%</td>
<td style="background-color:#003300; color:white;">45.55%</td>
</tr>
<!-- DPO C -->
<tr>
<td rowspan="3"><b>DPO C</b></td>
<td>Llama-3</td>
<td>66.15%</td>
<td>33.54%</td>
<td style="background-color:#228B22; color:white;">32.61%</td>
</tr>
<tr>
<td>Gemma-1.1</td>
<td>71.84%</td>
<td>27.70%</td>
<td style="background-color:#003300; color:white;">44.13%</td>
</tr>
<tr>
<td>Mistral-v0.3</td>
<td>70.00%</td>
<td>29.01%</td>
<td style="background-color:#006400; color:white;">40.99%</td>
</tr>
</tbody>
</table>
</div>

**Performance gains of Aya-B over Aya-Base against the Meta Llama 3 8B Instruct, Gemma 1.1 7B Instruct, and Mistral 7B Instruct v0.3 models.**
![Language Improvement Chart](graphs/analysis/language_improvement_chart_wide.png)

## Discussion

Aya-B and Aya-C result in slight performance improvements over the base Aya variant. Interestingly, Aya-A does not show this improvement. The main difference between the three architectures is that B and C rely on lower-quality negatives, rejected responses where the language quality is inferior to that of the chosen response. For both architectures B and C, this helps the model explicitly learn to avoid poor examples, thereby improving its performance. This is not the case for architecture A, which may instead steer the model toward other optimizations without resulting in better multilingual language output.

## Conclusion

In this paper, we examined three architectures for creating synthetic multilingual preference data. Our main finding is that architectures which generate the chosen responses in the target languages lead to improved multilingual performance. This is because such responses are more natural in the target languages and avoid English biases. When trained with Architecture B or C, Aya shows improved multilingual capabilities across both high-resource and low-resource languages, while also enhancing its performance in English. 

Read the full paper by clicking the icon below.

[![View PDF](https://img.shields.io/badge/Paper-PDF-red)](https://pascalmathas.github.io/multilingual_DPO_architectures/paper/Mathas%2C%20Pascal.%20(2025)%20Synthetic%20Data%20for%20Reinforcement%20Learning%20from%20Human%20Feedback%3A%20A%20Comparison%20of%20Architectures%20for%20Creating%20Synthetic%20Multilingual%20Preference%20Datasets.pdf)

## Repository Contents

This repository contains the code and findings for the paper "A Comparison of Architectures for Creating Synthetic Multilingual Preference Datasets." The structure is organized to reflect the pipeline, from data preprocessing to model evaluation and result visualization.

Each major subdirectory includes its own `README.md` file, providing detailed information on its purpose, contents, usage examples.

```
.
├── data/                                 # Raw and preprocessed data files (see data/README.md)
├── data_preprocessing/                   # Scripts for preparing DPO datasets (see data_preprocessing/README.md)
├── evaluation/                           # Scripts for model generation and judging (see evaluation/README.md)
├── graphs/                               # Scripts and output for visualizing results (see graphs/README.md)
├── jobs/                                 # Slurm job submission scripts for experiments (see jobs/README.md)
├── translators/                          # Various translation scripts vllm/API (see translators/README.md)
├── wmt24_evaluation/                     # Scripts for WMT24 translation quality evaluation (see wmt24_evaluation/README.md)
└── X-ALMA/                               # External X-ALMA repository, w/ used scripts (see X-ALMA/README.md)
```

### How to Use the Pipeline

Below are a couple of examples demonstrating how to run certain parts of the pipeline. For more detailed instructions, please refer to the `README.md` file within each subdirectory.

#### 1. Preprocessing the English DPO Dataset

```bash
python data_preprocessing/DPO/download_clean_data.py
```

This command will create a `dpo_dataset_cleaned.pkl` file in the `data/preprocessed/DPO/` directory. This file is then used by other scripts in `data_preprocessing/DPO/` to generate datasets for Architectures A, B, and C.

#### 2. Running Model Generation for Evaluation

```bash
python evaluation/generate/run_generation.py \
    --model_name <model_name> \
    --input_file <path_to_input_data.jsonl> \
    --output_dir <path_to_output_directory>
```

Replace `<model_name>`, `<path_to_input_data.jsonl>`, and `<path_to_output_directory>` with your specific values. The generated responses can then be fed into the `evaluation/judge` scripts for quality assessment.

#### 3. Run the WMT24++ evaluation

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