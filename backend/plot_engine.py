import math
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .stats_engine import perform_statistical_analysis


def _stratified_downsample(values_series, replicate_data, y_col, frac=None, n_samples=None, random_state=42, max_retries=3):
    """Downsample replicate data using stratified bin-based sampling to preserve distribution shape."""
    n = len(replicate_data)

    # Determine target sample size
    if frac is not None:
        target_n = max(1, int(round(n * frac)))
    elif n_samples is not None:
        target_n = max(1, n_samples)
    else:
        return replicate_data, None, None, False

    # No downsampling needed
    if target_n >= n:
        return replicate_data, None, None, False

    original_values = values_series.values

    # For very small datasets, fall back to random sampling
    if n < 20:
        for attempt in range(1 + max_retries):
            seed = random_state + attempt
            sampled = replicate_data.sample(n=target_n, random_state=seed)
            stat, p_value = ks_2samp(original_values, sampled[y_col].values)
            if p_value >= 0.05:
                return sampled, stat, p_value, False
        return sampled, stat, p_value, True

    # Sturges' rule for bin count
    k = max(2, math.ceil(1 + math.log2(n)))
    bins = pd.cut(values_series, bins=k)
    replicate_data = replicate_data.copy()
    replicate_data['_bin'] = bins.values

    for attempt in range(1 + max_retries):
        seed = random_state + attempt
        sampled_parts = []
        rng = np.random.RandomState(seed)
        for bin_label, group in replicate_data.groupby('_bin', observed=True):
            bin_size = len(group)
            bin_target = max(1, round(target_n * bin_size / n))
            bin_target = min(bin_target, bin_size)
            sampled_parts.append(group.sample(n=bin_target, random_state=rng))

        sampled = pd.concat(sampled_parts).drop(columns=['_bin'])
        if len(sampled) > target_n:
            sampled = sampled.sample(n=target_n, random_state=rng)
        stat, p_value = ks_2samp(original_values, sampled[y_col].values)
        if p_value >= 0.05:
            return sampled, stat, p_value, False

    return sampled, stat, p_value, True


def create_interactive_superplot(data, x_col, y_col, replicate_col, paired=False, width=900, height=720, font_size=18, color_map='Plotly', template='plotly', marker_size=6, downsample_mode='all', downsample_value=None, log_scale=False, show_stars=True, show_axes=True):
    """
    Create an interactive SuperPlot visualization with all customization options.
    """
    stats_results = perform_statistical_analysis(data, x_col, y_col, replicate_col, paired=paired)

    # When log scale is requested, transform data to log10 and plot on a linear axis
    plot_y_col = y_col
    if log_scale:
        plot_y_col = f'_log10_{y_col}'
        data = data.copy()
        positive_mask = data[y_col] > 0
        data = data[positive_mask]
        data[plot_y_col] = np.log10(data[y_col])

    if hasattr(px.colors.qualitative, color_map):
        replicate_colors = getattr(px.colors.qualitative, color_map)
    else:
        replicate_colors = px.colors.qualitative.Plotly

    unique_replicates = sorted(data[replicate_col].unique())
    if len(replicate_colors) < len(unique_replicates):
        replicate_colors = [replicate_colors[i % len(replicate_colors)] for i in range(len(unique_replicates))]
    replicate_to_color = dict(zip(unique_replicates, replicate_colors))

    replicate_means = data.groupby([x_col, replicate_col])[plot_y_col].mean().reset_index()
    treatment_stats = replicate_means.groupby(x_col)[plot_y_col].agg(['mean', 'sem']).reset_index()

    y_axis_label = f"log\u2081\u2080({y_col})" if log_scale else y_col

    fig = go.Figure()

    ks_results = []
    for replicate in unique_replicates:
        replicate_data = data[data[replicate_col] == replicate]

        data_to_plot = replicate_data
        if downsample_value is not None:
            try:
                if downsample_mode == 'percentage':
                    frac = max(0.01, min(1.0, float(downsample_value) / 100.0))
                    data_to_plot, ks_stat, ks_p, ks_failed = _stratified_downsample(
                        replicate_data[plot_y_col], replicate_data, plot_y_col, frac=frac)
                    if ks_stat is not None:
                        ks_results.append({'replicate': replicate, 'ks_stat': ks_stat, 'ks_p': ks_p, 'failed': ks_failed})
                elif downsample_mode == 'number':
                    n_samples = max(1, int(downsample_value))
                    data_to_plot, ks_stat, ks_p, ks_failed = _stratified_downsample(
                        replicate_data[plot_y_col], replicate_data, plot_y_col, n_samples=n_samples)
                    if ks_stat is not None:
                        ks_results.append({'replicate': replicate, 'ks_stat': ks_stat, 'ks_p': ks_p, 'failed': ks_failed})
            except (ValueError, TypeError):
                data_to_plot = replicate_data

        x_jittered = []
        for x in data_to_plot[x_col]:
            try:
                x_idx = list(treatment_stats[x_col]).index(x)
                jitter = np.random.uniform(-0.2, 0.2)
                x_jittered.append(x_idx + jitter)
            except ValueError:
                continue

        if not x_jittered:
            continue

        fig.add_trace(go.Scatter(
            x=x_jittered, y=data_to_plot[plot_y_col], mode='markers',
            name=f'{replicate}',
            marker=dict(size=marker_size, color=replicate_to_color[replicate], opacity=0.6),
            legendgroup=f'replicate_{replicate}',
            hovertemplate=(f"<b>Replicate {replicate}</b><br>Treatment: %{{text}}<br>{y_axis_label}: %{{y:.2f}}<br><extra></extra>"),
            text=data_to_plot[x_col]
        ))

        replicate_mean_data = replicate_means[replicate_means[replicate_col] == replicate]
        x_values, y_values, text_values = [], [], []
        for _, row in replicate_mean_data.iterrows():
            try:
                x_values.append(list(treatment_stats[x_col]).index(row[x_col]))
                y_values.append(row[plot_y_col])
                text_values.append(row[x_col])
            except ValueError:
                continue

        if x_values:
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='markers', name=f'Mean (Replicate {replicate})',
                marker=dict(size=marker_size * 1.8, color=replicate_to_color[replicate], symbol='circle', line=dict(
                color='gray',
                width=0.5
            )),
                legendgroup=f'replicate_{replicate}', showlegend=False,
                hovertemplate=(f"<b>Replicate {replicate} Mean</b><br>Treatment: %{{text}}<br>{y_axis_label}: %{{y:.2f}}<br><extra></extra>"),
                text=text_values
            ))

    fig.add_trace(go.Scatter(
        x=list(range(len(treatment_stats))), y=treatment_stats['mean'], mode='markers', name='Treatment Mean +/- SEM',
        error_y=dict(type='data', array=treatment_stats['sem'], visible=True, color='gray'),
        marker=dict(color='gray', size=marker_size * 2.3, symbol='circle-open', line=dict(color='gray', width=2)),
        hovertemplate=("<b>Treatment Mean</b><br>Treatment: %{text}<br>" + f"{y_axis_label}: %{{y:.2f}} ± %{{error_y.array:.2f}}<br><extra></extra>"),
        text=treatment_stats[x_col]
    ))

    if len(data) > 0:
        max_y = data[plot_y_col].max()
        y_range = data[plot_y_col].max() - data[plot_y_col].min()
        if y_range == 0: y_range = max_y * 0.1 if max_y != 0 else 1.0

        for idx, result in enumerate(stats_results['pairwise_results']):
            if show_stars and result['p_value'] < 0.05:
                try:
                    g1_idx = list(treatment_stats[x_col]).index(result['group1'])
                    g2_idx = list(treatment_stats[x_col]).index(result['group2'])
                    bar_y = max_y + (idx + 1) * (y_range * 0.1)
                    fig.add_shape(type='line', x0=g1_idx, x1=g2_idx, y0=bar_y, y1=bar_y, line=dict(color='black', width=1))
                    stars = '*' * sum([result['p_value'] < cutoff for cutoff in [0.05, 0.01, 0.001]])
                    fig.add_annotation(x=(g1_idx + g2_idx) / 2, y=bar_y, text=stars, showarrow=False, yshift=5, font=dict(size=16, color="black"))
                except (ValueError, IndexError):
                    pass

    fig.update_layout(
        xaxis=dict(
            tickmode='array', ticktext=treatment_stats[x_col],
            tickvals=list(range(len(treatment_stats))),
            tickfont=dict(size=font_size * 0.9), title_font=dict(size=font_size),
            visible=show_axes,
        ),
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(size=font_size)),
            tickfont=dict(size=font_size * 0.9),
            range=[
                data[plot_y_col].min() - y_range * 0.1 if len(data) > 0 else 0,
                (max_y + y_range * (0.1 * (len(stats_results['pairwise_results']) + 2)) if show_stars else max_y + y_range * 0.15) if len(data) > 0 else 1,
            ],
            visible=show_axes,
        ),
        width=width, height=height, hovermode='closest', template=template,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05, font=dict(size=font_size * 0.8)),
        font=dict(size=font_size), margin=dict(r=150, t=100)
    )

    return fig, stats_results, {'data': data, 'x_col': x_col, 'y_col': y_col, 'replicate_col': replicate_col, 'paired': paired}, ks_results
