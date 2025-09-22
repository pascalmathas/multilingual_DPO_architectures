import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os

def plot_score_heatmaps(results: dict, model_priority_map: dict):
    if not results:
        print("Warning: 'results' dictionary is empty. Cannot generate heatmaps.")
        return

    all_models_present_in_results = set()
    for lang_data in results.values():
        if isinstance(lang_data, dict):
            all_models_present_in_results.update(lang_data.keys())

    initial_plot_model_display_names = []
    initial_plot_model_actual_names = []
    for display_name, potential_actual_names in model_priority_map.items():
        found_actual_name = None
        potential_actual_names_list = [potential_actual_names] if isinstance(potential_actual_names, str) else potential_actual_names
        for actual_name_option in potential_actual_names_list:
            if actual_name_option in all_models_present_in_results:
                found_actual_name = actual_name_option
                break
        if found_actual_name:
            initial_plot_model_display_names.append(display_name)
            initial_plot_model_actual_names.append(found_actual_name)
        else:
            print(f"Warning: Model for display name '{display_name}' not found in results. Searched for: {potential_actual_names_list}")

    if not initial_plot_model_actual_names:
        print("\nNo models selected for plotting based on model_priority_map and available data.")
        return

    all_languages_for_plotting = sorted([lang for lang in results.keys() if isinstance(results.get(lang), dict) and results[lang]])
    if not all_languages_for_plotting:
        print("\nNo language data found for plotting.")
        return

    num_plot_models = len(initial_plot_model_actual_names)
    num_languages = len(all_languages_for_plotting)

    plotted_model_is_paper = {}
    for model_actual_name in initial_plot_model_actual_names:
        is_paper = any(
            lang_name_check in results and isinstance(results[lang_name_check], dict) and
            model_actual_name in results[lang_name_check] and
            results[lang_name_check][model_actual_name].get('count') == -1
            for lang_name_check in all_languages_for_plotting
        )
        plotted_model_is_paper[model_actual_name] = is_paper

    language_global_averages = {}
    all_unique_languages_in_dataset = sorted(list(results.keys()))
    for lang_name_global_avg in all_unique_languages_in_dataset:
        bleu_scores_for_lang_avg = []
        chrf_scores_for_lang_avg = []
        if lang_name_global_avg in results and isinstance(results[lang_name_global_avg], dict):
            for scores_for_avg in results[lang_name_global_avg].values():
                if scores_for_avg.get('count', 0) > 0:
                    bleu_scores_for_lang_avg.append(scores_for_avg.get('BLEU', np.nan))
                    chrf_scores_for_lang_avg.append(scores_for_avg.get('CHRF', np.nan))
        language_global_averages[lang_name_global_avg] = {
            'BLEU': np.nanmean(bleu_scores_for_lang_avg) if bleu_scores_for_lang_avg else np.nan,
            'CHRF': np.nanmean(chrf_scores_for_lang_avg) if chrf_scores_for_lang_avg else np.nan
        }

    def populate_matrix_for_sorting(metric_name: str):
        matrix = np.full((num_plot_models, num_languages), np.nan)
        for i, actual_model_name in enumerate(initial_plot_model_actual_names):
            is_this_model_paper = plotted_model_is_paper.get(actual_model_name, False)
            for j, language_name in enumerate(all_languages_for_plotting):
                current_metric_val_for_sort = np.nan
                score_data_exists = (language_name in results and isinstance(results[language_name], dict) and
                                     actual_model_name in results[language_name])
                if score_data_exists:
                    scores_data = results[language_name][actual_model_name]
                    model_count = scores_data.get('count', 0)
                    if model_count > 0 or model_count == -1:
                        current_metric_val_for_sort = scores_data.get(metric_name, np.nan)
                    elif not is_this_model_paper:
                        current_metric_val_for_sort = language_global_averages.get(language_name, {}).get(metric_name, np.nan)
                elif not is_this_model_paper:
                    current_metric_val_for_sort = language_global_averages.get(language_name, {}).get(metric_name, np.nan)
                matrix[i, j] = current_metric_val_for_sort
        return matrix

    bleu_scores_matrix_for_bleu_sorting = populate_matrix_for_sorting('BLEU')
    avg_bleu_for_sort_calc = np.nanmean(bleu_scores_matrix_for_bleu_sorting, axis=1)
    avg_bleu_for_sort_calc = np.nan_to_num(avg_bleu_for_sort_calc, nan=-1.0)
    sorted_indices_bleu = np.argsort(avg_bleu_for_sort_calc)[::-1]

    plot_model_display_names_sorted_bleu = [initial_plot_model_display_names[k] for k in sorted_indices_bleu]
    plot_model_actual_names_sorted_bleu = [initial_plot_model_actual_names[k] for k in sorted_indices_bleu]

    print("\nModels ordered by average BLEU score for BLEU plot (descending; avg calculation includes imputed scores for non-paper models):")
    final_avg_bleu_sorted_for_print = avg_bleu_for_sort_calc[sorted_indices_bleu]
    for i, display_name in enumerate(plot_model_display_names_sorted_bleu):
        avg_score_val = final_avg_bleu_sorted_for_print[i]
        avg_score_str = f"{avg_score_val:.2f}" if avg_score_val != -1.0 else "N/A"
        print(f"- {display_name} (Avg BLEU for sorting: {avg_score_str})")

    chrf_scores_matrix_for_chrf_sorting = populate_matrix_for_sorting('CHRF')
    avg_chrf_for_sort_calc = np.nanmean(chrf_scores_matrix_for_chrf_sorting, axis=1)
    avg_chrf_for_sort_calc = np.nan_to_num(avg_chrf_for_sort_calc, nan=-1.0)
    sorted_indices_chrf = np.argsort(avg_chrf_for_sort_calc)[::-1]

    plot_model_display_names_sorted_chrf = [initial_plot_model_display_names[k] for k in sorted_indices_chrf]
    plot_model_actual_names_sorted_chrf = [initial_plot_model_actual_names[k] for k in sorted_indices_chrf]

    print("\nModels ordered by average CHRF score for CHRF plot (descending; avg calculation includes imputed scores for non-paper models):")
    final_avg_chrf_sorted_for_print = avg_chrf_for_sort_calc[sorted_indices_chrf]
    for i, display_name in enumerate(plot_model_display_names_sorted_chrf):
        avg_score_val = final_avg_chrf_sorted_for_print[i]
        avg_score_str = f"{avg_score_val:.2f}" if avg_score_val != -1.0 else "N/A"
        print(f"- {display_name} (Avg CHRF for sorting: {avg_score_str})")

    def populate_matrices_for_plot(metric_name: str, sorted_actual_model_names: list):
        scores_matrix = np.full((num_plot_models, num_languages), np.nan)
        text_annotations = np.full((num_plot_models, num_languages), "", dtype=object)
        for i, actual_model_name in enumerate(sorted_actual_model_names):
            for j, language_name in enumerate(all_languages_for_plotting):
                current_metric_val = np.nan
                score_data_exists = (language_name in results and isinstance(results[language_name], dict) and
                                     actual_model_name in results[language_name])
                if score_data_exists:
                    scores_data = results[language_name][actual_model_name]
                    model_count = scores_data.get('count', 0)
                    if model_count > 0 or model_count == -1:
                        current_metric_val = scores_data.get(metric_name, np.nan)
                
                scores_matrix[i, j] = current_metric_val
                text_annotations[i,j] = f"{current_metric_val:.2f}" if not np.isnan(current_metric_val) else ""
        return scores_matrix, text_annotations

    bleu_scores_matrix_for_bleu_plot, text_annotations_bleu_for_bleu_plot =         populate_matrices_for_plot('BLEU', plot_model_actual_names_sorted_bleu)

    chrf_scores_matrix_for_chrf_plot, text_annotations_chrf_for_chrf_plot =         populate_matrices_for_plot('CHRF', plot_model_actual_names_sorted_chrf)

    def apply_blanking(scores_matrix, text_annotations, sorted_display_names):
        models_to_blank = {
            'TowerInstruct': ["Arabic", "Icelandic"],
            'Command-R': ["Icelandic"]
        }
        for model_display_to_blank, langs_to_blank in models_to_blank.items():
            if model_display_to_blank in sorted_display_names:
                model_idx = sorted_display_names.index(model_display_to_blank)
                for lang_name_to_blank in langs_to_blank:
                    if lang_name_to_blank in all_languages_for_plotting:
                        lang_idx = all_languages_for_plotting.index(lang_name_to_blank)
                        scores_matrix[model_idx, lang_idx] = np.nan
                        text_annotations[model_idx, lang_idx] = ""
        return scores_matrix, text_annotations
    
    bleu_scores_matrix_for_bleu_plot, text_annotations_bleu_for_bleu_plot =         apply_blanking(bleu_scores_matrix_for_bleu_plot, text_annotations_bleu_for_bleu_plot, plot_model_display_names_sorted_bleu)

    chrf_scores_matrix_for_chrf_plot, text_annotations_chrf_for_chrf_plot =         apply_blanking(chrf_scores_matrix_for_chrf_plot, text_annotations_chrf_for_chrf_plot, plot_model_display_names_sorted_chrf)

    common_heatmap_params = {
        'x': all_languages_for_plotting,
        'texttemplate': "%{text}",
        'hoverongaps': False,
        'xgap': 1, 'ygap': 1,
        'showscale': False
    }
    common_layout_params_base = {
        'xaxis_title_text': 'Language',
        'yaxis_autorange': 'reversed',
        'plot_bgcolor': 'white',
        'font': dict(color='black'),
        'margin': dict(t=50, b=50, l=200, r=50)
    }

    fig_bleu = go.Figure(data=go.Heatmap(
        y=plot_model_display_names_sorted_bleu,
        z=bleu_scores_matrix_for_bleu_plot,
        text=text_annotations_bleu_for_bleu_plot,
        colorscale='Blues',
        zmin=0, zmax=45,
        textfont=dict(size=18),
        **common_heatmap_params
    ))
    bleu_layout_params = common_layout_params_base.copy()
    bleu_layout_params['yaxis_title_text'] = 'Model (ordered by Avg. BLEU)'
    fig_bleu.update_layout(**bleu_layout_params)
    fig_bleu.update_layout(
        font=dict(color='black'),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    fig_chrf = go.Figure(data=go.Heatmap(
        y=plot_model_display_names_sorted_chrf,
        z=chrf_scores_matrix_for_chrf_plot,
        text=text_annotations_chrf_for_chrf_plot,
        colorscale='Blues',
        zmin=0, zmax=65,
        textfont=dict(size=18),
        **common_heatmap_params
    ))
    chrf_layout_params = common_layout_params_base.copy()
    chrf_layout_params['yaxis_title_text'] = 'Model (ordered by Avg. CHRF)'
    fig_chrf.update_layout(**chrf_layout_params)
    fig_chrf.update_layout(
        font=dict(color='black'),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    print("\nGenerating and displaying BLEU and CHRF score heatmaps...")
    fig_bleu.show()
    fig_chrf.show()

    export_dir = "../graphs/evaluation"
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
            print(f"Created directory: {export_dir}")
        except OSError as e:
            print(f"Error creating directory {export_dir}: {e}. Plots may not be saved.")
            export_dir = "." 

    try:

        pio.write_image(fig_bleu, os.path.join(export_dir, "bleu_scores_heatmap.png"), 
                        scale=6, width=1161, height=650)
        pio.write_image(fig_chrf, os.path.join(export_dir, "chrf_scores_heatmap.png"), 
                        scale=6, width=1161, height=650)
        print(f"Heatmaps saved to '{os.path.abspath(export_dir)}' directory.")
    except ValueError as ve: 
        print(f"Error saving heatmap images: {ve}")
        print("Please ensure you have 'kaleido' installed for static image export: pip install -U kaleido")
    except Exception as e:
        print(f"An unexpected error occurred while saving heatmap images: {e}")
