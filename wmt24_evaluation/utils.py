import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sacrebleu.metrics import BLEU, CHRF
import inspect
import os
from collections import defaultdict

def _bleu_tokenizer_for(lang: str) -> str:
    l = lang.lower()
    if l.startswith("japanese"):
        return "ja-mecab"
    if "chinese" in l:
        return "zh"
    return "13a"

def _build_bleu(tokenizer_name: str):
    sig = inspect.signature(BLEU.__init__).parameters
    kwargs = {}
    if "tokenize" in sig:
        kwargs["tokenize"] = tokenizer_name
    if "effective_order" in sig:
        kwargs["effective_order"] = True
    return BLEU(**kwargs)

def evaluate_translations(df: pd.DataFrame):
    results = {}

    for language, language_df in df.groupby('tar_lang'):
        bleu = _build_bleu(_bleu_tokenizer_for(language))
        chrf = CHRF(word_order=2)

        language_results = defaultdict(list)

        for _, row in language_df.iterrows():
            reference = row['reference']
            if pd.isna(reference) or reference == "":
                continue

            hypothesis = row['trs_prompt']
            model      = row['model']

            bleu_score = bleu.sentence_score(hypothesis, [reference]).score
            chrf_score = chrf.sentence_score(hypothesis, [reference]).score

            language_results[model].append({
                'BLEU'      : bleu_score,
                'CHRF'      : chrf_score,
                'source'    : row['org_prompt'],
                'hypothesis': hypothesis,
                'reference' : reference
            })

        model_averages = {}
        for model, scores in language_results.items():
            if not scores:
                continue
            model_averages[model] = {
                'BLEU'  : np.mean([s['BLEU'] for s in scores]),
                'CHRF'  : np.mean([s['CHRF'] for s in scores]),
                'count' : len(scores)
            }

        results[language] = model_averages

    return results


def print_results(results):
    print("\n===== TRANSLATION EVALUATION RESULTS =====\n")

    all_models = sorted({m for lang in results.values() for m in lang})

    header = f"{ 'Language':<30} { 'Model':<30} { 'BLEU':<12} { 'CHRF':<12} { 'Count':<10}"
    print(header)
    print("-" * len(header))

    for language, lang_results in sorted(results.items()):
        for model in all_models:
            if model not in lang_results:
                continue
            scores = lang_results[model]
            print(f"{language:<30} {model:<30} {scores['BLEU']:<12.2f} "
                  f"{scores['CHRF']:<12.2f} {scores['count']:<10}")
        print("-" * len(header))

def analyze_results_by_model(results):
    print("\n===== MODEL PERFORMANCE SUMMARY (with Imputation for Missing Languages) =====")

    all_models_in_results = set()
    all_languages_in_results = set()
    for lang, lang_data in results.items():
        all_languages_in_results.add(lang)
        for model_name in lang_data.keys():
            all_models_in_results.add(model_name)
    
    sorted_languages = sorted(list(all_languages_in_results))

    language_global_averages = {}
    for lang_name in sorted_languages:
        bleu_scores_for_lang = []
        chrf_scores_for_lang = []
        if lang_name in results:
            for model_name, scores in results[lang_name].items():
                if scores.get('count', 0) > 0 and 'paper' not in model_name.lower():
                    bleu_scores_for_lang.append(scores['BLEU'])
                    chrf_scores_for_lang.append(scores['CHRF'])
        
        language_global_averages[lang_name] = {
            'BLEU': np.mean(bleu_scores_for_lang) if bleu_scores_for_lang else 0,
            'CHRF': np.mean(chrf_scores_for_lang) if chrf_scores_for_lang else 0
        }

    model_overall_metrics = {}
    models_to_summarize = sorted([m for m in all_models_in_results if 'paper' not in m.lower()])

    for model_name in models_to_summarize:
        model_bleu_scores_for_avg = []
        model_chrf_scores_for_avg = []
        actual_langs_covered = 0
        total_samples = 0

        for lang_name in sorted_languages:
            lang_data = results.get(lang_name, {})
            model_scores_for_lang = lang_data.get(model_name)

            if model_scores_for_lang and model_scores_for_lang.get('count', 0) > 0:
                model_bleu_scores_for_avg.append(model_scores_for_lang['BLEU'])
                model_chrf_scores_for_avg.append(model_scores_for_lang['CHRF'])
                actual_langs_covered += 1
                total_samples += model_scores_for_lang['count']
            else:
                model_bleu_scores_for_avg.append(language_global_averages[lang_name]['BLEU'])
                model_chrf_scores_for_avg.append(language_global_averages[lang_name]['CHRF'])

        if model_bleu_scores_for_avg:
            model_overall_metrics[model_name] = {
                'BLEU_avg_imputed': np.mean(model_bleu_scores_for_avg),
                'CHRF_avg_imputed': np.mean(model_chrf_scores_for_avg),
                'actual_languages_covered': actual_langs_covered,
                'total_languages_considered': len(sorted_languages),
                'total_samples': total_samples
            }

    header = f"{ 'Model':<30} { 'BLEU (Avg)':<12} { 'CHRF (Avg)':<12} { 'Langs Cov.':<12} { 'Samples':<10}"
    print(header)
    print("-" * len(header))

    sorted_model_names = sorted(
        model_overall_metrics.keys(),
        key=lambda m: model_overall_metrics[m]['BLEU_avg_imputed'],
        reverse=True
    )

    for model_name in sorted_model_names:
        scores = model_overall_metrics[model_name]
        lang_coverage_str = f"{scores['actual_languages_covered']}/{scores['total_languages_considered']}"
        print(f"{model_name:<30} {scores['BLEU_avg_imputed']:<12.2f} "
              f"{scores['CHRF_avg_imputed']:<12.2f} "
              f"{lang_coverage_str:<12} {scores['total_samples']:<10}")
    print("-" * len(header))
    print(f"* Averages are calculated over all {len(sorted_languages)} languages, with imputation for missing scores.")
    print("* Imputation uses the average score of other evaluated models for that specific language.")

def identify_best_model(results):
    excluded_models = {'GPT-4o Deutsch paper', 'Unbabel Tower Deutsch paper'}
    best_models = {}

    for language, lang_results in results.items():
        if not lang_results:
            continue

        best_model = None
        best_avg_score = -1

        for model, scores in lang_results.items():
            if model in excluded_models:
                continue

            avg_score = (scores['BLEU'] / 100 + scores['CHRF'] / 100) / 2

            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_model = model

        best_models[language] = {
            'model': best_model,
            'scores': lang_results[best_model] if best_model else None
        }

    print("\n===== BEST MODEL PER LANGUAGE =====")
    header = f"{ 'Language':<30} { 'Best Model':<30} { 'BLEU':<12} { 'CHRF':<12}"
    print(header)
    print("-" * len(header))

    for language, details in sorted(best_models.items()):
        if details['model']:
            scores = details['scores']
            print(f"{language:<30} {details['model']:<30} {scores['BLEU']:<12.2f} "
                  f"{scores['CHRF']:<12.2f}")


def save_results_to_csv(results, filename):

    rows = []

    for language, lang_results in results.items():
        for model, scores in lang_results.items():
            row = {
                'Language': language,
                'Model': model,
                'BLEU': scores['BLEU'],
                'CHRF': scores['CHRF'],
                'Sample Count': scores['count']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)

    df = df.sort_values(by=['Language', 'Model']).reset_index(drop=True)

    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")