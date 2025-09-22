# -*- coding: utf-8 -*-
"""Generates a bar chart showing the performance improvement of different models across various languages.

This script uses Plotly to create a grouped bar chart comparing the performance
improvement of Gemma, Llama-3, and Mistral models over a baseline.
The chart is then displayed and exported as a high-resolution PNG image.
"""

import plotly.graph_objects as go
import plotly.io as pio

# Data for the chart
LANGUAGES = ["Arabic", "Chinese", "Dutch", "English", "German", "Icelandic", "Japanese", "Russian", "Spanish"]
IMPROVEMENT_DATA = {
    "Gemma": [1.6, 4.67, 12.0, 8.4, 5.87, -9.6, -0.26, 8.53, -0.27],
    "Llama-3": [0.26, 3.2, 18.13, 15.74, 0.4, 6.53, 4.0, 3.2, -0.27],
    "Mistral": [2.4, 8.27, 12.53, 13.87, 1.33, 8.27, 3.97, 4.8, -0.13],
}

# Colors for the bars
MODEL_COLORS = ['#2E86AB', '#A23B72', '#F18F01']


def create_chart():
    """Creates and configures the bar chart."""
    figure = go.Figure()

    models = ["Gemma", "Llama-3", "Mistral"]
    model_display_names = ['Gemma 1.1 7B', 'Llama 3 8B', 'Mistral v0.3']

    for i, (model, display_name) in enumerate(zip(models, model_display_names)):
        figure.add_trace(go.Bar(
            x=LANGUAGES,
            y=IMPROVEMENT_DATA[model],
            name=display_name,
            marker_color=MODEL_COLORS[i]
        ))

    figure.update_layout(
        barmode='group',
        yaxis_title='ΔW-L% Improvement (Aya-B - Aya-Base)',
        xaxis_tickangle=-45,
        legend=dict(
            title='Model',
            x=0.87,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        plot_bgcolor='white',
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='gray'),
        title_text=None,
        font=dict(size=24, color='black'),
        margin=dict(l=60, r=60, t=60, b=100)
    )

    return figure


def export_chart(figure, filename, width=1200, height=800, scale=6):
    """Exports the chart to a PNG file."""
    figure.update_layout(width=width, height=height)
    pio.write_image(figure, f"{filename}.png", width=width, height=height, scale=scale)
    print(f"Chart exported as {filename}.png ({width}x{height} at scale {scale})")


def main():
    """Main function to create and export the chart."""
    chart = create_chart()
    chart.show()
    export_chart(chart, "language_improvement_chart_wide", width=2000, height=800, scale=6)


if __name__ == "__main__":
    main()