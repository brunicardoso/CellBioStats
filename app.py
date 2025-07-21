
import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm, AnovaRM
import statsmodels.formula.api as sm
from itertools import combinations
import plotly.express as px
import webbrowser
import openpyxl

def calculate_hierarchical_stats(data, x_col, y_col, replicate_col):
    """
    Calculate descriptive statistics based on replicate means rather than individual values
    data: A pandas DataFrame containing your experimental data
    x_col: The column name representing treatment groups or independent variables
    y_col: The column name containing the measurements or dependent variables
    replicate_col: The column name identifying replication groups within treatments
    """
    # First calculate means for each replicate within each treatment
    replicate_means = data.groupby([x_col, replicate_col])[y_col].mean().reset_index()

    # Calculate overall statistics for each treatment based on replicate means
    treatment_stats = {}
    for treatment in replicate_means[x_col].unique():
        treatment_data = replicate_means[replicate_means[x_col] == treatment]
        treatment_stats[treatment] = {
            'mean': treatment_data[y_col].mean(),
            'sem': stats.sem(treatment_data[y_col]),
            'n_replicates': len(treatment_data),
            'n_total': len(data[data[x_col] == treatment])
        }

    return replicate_means, treatment_stats


def perform_statistical_analysis(data, x_col, y_col, replicate_col, paired=False):
    """
    Perform statistical analysis using replicate means with option for paired tests

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables to analyze
    x_col : str
        Column name for independent variable (groups)
    y_col : str
        Column name for dependent variable (measurement)
    replicate_col : str
        Column name identifying replicates
    paired : bool, default=False
        Whether to perform paired (repeated measures) tests

    Returns:
    --------
    dict
        Dictionary containing all test results
    """
    # Get replicate means for statistical testing
    replicate_means = data.groupby([x_col, replicate_col])[y_col].mean().reset_index()

    groups = sorted(list(replicate_means[x_col].unique()))
    group_data = [replicate_means[replicate_means[x_col] == group][y_col] for group in groups]

    # Test for normality on replicate means
    shapiro_results = []
    for group, gd in zip(groups, group_data):
        if len(gd) < 3:  # Shapiro-Wilk test requires at least 3 samples
            shapiro_results.append({
                'group': group, 'statistic': None, 'p_value': None,
                'normal': False, 'note': "Insufficient samples for normality test"
            })
        else:
            stat, p_value = shapiro(gd)
            shapiro_results.append({
                'group': group, 'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05
            })

    # Test for homoscedasticity
    all_normal = all(result.get('normal', False) for result in shapiro_results)
    levene_stat, levene_p, homoscedastic = None, None, False
    valid_groups = [gd for gd in group_data if len(gd) >= 2]
    if len(valid_groups) > 1:
        try:
            levene_stat, levene_p = levene(*valid_groups)
            homoscedastic = levene_p > 0.05
        except Exception:
            pass

    # Initialize variables
    test_used, stat, p_value, pairwise_results, test_note = None, None, None, [], None

    # Handle paired tests if requested and possible
    if paired and len(groups) > 1:
        common_replicates = set.intersection(*[set(replicate_means[replicate_means[x_col] == g][replicate_col]) for g in groups])

        if not common_replicates:
            paired = False
            test_note = "Paired test requested but no common replicates found. Using unpaired test instead."
        else:
            filtered_means = replicate_means[replicate_means[replicate_col].isin(common_replicates)]
            
            # For 2 groups: use paired t-test or Wilcoxon
            if len(groups) == 2:
                pivoted_data = filtered_means.pivot(index=replicate_col, columns=x_col, values=y_col).dropna()
                if len(pivoted_data) < 3:
                    paired = False
                    test_note = "Paired test requested but insufficient complete pairs (<3). Using unpaired test instead."
                else:
                    g1_data, g2_data = pivoted_data[groups[0]], pivoted_data[groups[1]]
                    if all_normal and homoscedastic:
                        stat, p_value = stats.ttest_rel(g1_data, g2_data)
                        test_used = "Paired t-test"
                    else:
                        stat, p_value = stats.wilcoxon(g1_data, g2_data)
                        test_used = "Wilcoxon signed-rank test"
                    pairwise_results = [{'group1': groups[0], 'group2': groups[1], 'p_value': p_value, 'test': test_used}]
            
            # For >2 groups: use repeated measures ANOVA or Friedman test
            else:
                try:
                    pivoted_data = filtered_means.pivot(index=replicate_col, columns=x_col, values=y_col)
                    complete_cases = pivoted_data.dropna()

                    if len(complete_cases) < 3:
                        paired = False
                        test_note = "Paired test requested but insufficient complete replicates (<3). Using unpaired test instead."
                    else:
                        group_arrays = [complete_cases[group] for group in groups]
                        if all_normal and homoscedastic:
                            test_used = "Repeated measures ANOVA (AnovaRM)"
                            rm_df = complete_cases.reset_index().melt(id_vars=replicate_col, value_vars=groups, var_name='group', value_name='value')
                            rm_df.rename(columns={replicate_col: 'subject'}, inplace=True)
                            rm_df['subject'] = rm_df['subject'].astype(str)
                            rm_df['group'] = rm_df['group'].astype(str)

                            rm_anova = AnovaRM(rm_df, 'value', 'subject', within=['group'])
                            res = rm_anova.fit()
                            stat = res.anova_table.loc['group', 'F Value']
                            p_value = res.anova_table.loc['group', 'Pr > F']

                            n_comparisons = len(list(combinations(groups, 2)))
                            for g1, g2 in combinations(groups, 2):
                                t_stat, p = stats.ttest_rel(complete_cases[g1], complete_cases[g2])
                                pairwise_results.append({'group1': g1, 'group2': g2, 'p_value': min(p * n_comparisons, 1.0), 'test': "Paired t-test (Bonferroni)"})
                        else:
                            test_used = "Friedman test"
                            stat, p_value = stats.friedmanchisquare(*group_arrays)
                            n_comparisons = len(list(combinations(groups, 2)))
                            for g1, g2 in combinations(groups, 2):
                                w_stat, p = stats.wilcoxon(complete_cases[g1], complete_cases[g2])
                                pairwise_results.append({'group1': g1, 'group2': g2, 'p_value': min(p * n_comparisons, 1.0), 'test': "Wilcoxon (Bonferroni)"})
                except Exception as e:
                    paired = False
                    test_note = f"Paired test for >2 groups failed: {str(e)}. Using unpaired test instead."

    # Choose unpaired statistical test if not using paired tests
    if not paired:
        if len(groups) == 2:
            if all_normal and homoscedastic:
                stat, p_value = stats.ttest_ind(*group_data)
                test_used = "Student's t-test (unpaired)"
            else:
                stat, p_value = mannwhitneyu(*group_data)
                test_used = "Mann-Whitney U test"
            pairwise_results = [{'group1': groups[0], 'group2': groups[1], 'p_value': p_value, 'test': test_used}]
        
        elif len(groups) > 2:
            if all_normal and homoscedastic:
                test_used = "One-way ANOVA"
                stat, p_value = stats.f_oneway(*group_data)
                try:
                    flat_values = [val for sublist in group_data for val in sublist]
                    flat_groups = [group for i, group in enumerate(groups) for _ in group_data[i]]
                    tukey = pairwise_tukeyhsd(endog=flat_values, groups=flat_groups, alpha=0.05)
                    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                    for _, row in tukey_df.iterrows():
                        pairwise_results.append({'group1': row['group1'], 'group2': row['group2'], 'p_value': float(row['p-adj']), 'test': "Tukey HSD"})
                except Exception as e:
                    test_note = f"Tukey HSD post-hoc test failed: {str(e)}"
            else:
                test_used = "Kruskal-Wallis H-test"
                stat, p_value = kruskal(*group_data)
                n_comparisons = len(list(combinations(groups, 2)))
                for g1, g2 in combinations(groups, 2):
                    g1_data, g2_data = group_data[groups.index(g1)], group_data[groups.index(g2)]
                    if len(g1_data) > 0 and len(g2_data) > 0:
                        try:
                            m_stat, p = mannwhitneyu(g1_data, g2_data)
                            pairwise_results.append({'group1': g1, 'group2': g2, 'p_value': min(p * n_comparisons, 1.0), 'test': "Mann-Whitney U (Bonferroni)"})
                        except Exception:
                            pass
    
    return {
        'shapiro_results': shapiro_results,
        'levene_result': {'statistic': levene_stat, 'p_value': levene_p} if levene_stat is not None else None,
        'main_test': {'name': test_used, 'statistic': stat, 'p_value': p_value, 'note': test_note},
        'pairwise_results': pairwise_results,
        'paired_test_used': paired
    }
  
def create_stats_table(stats_results, data=None, x_col=None, y_col=None, replicate_col=None):
    """Create a formatted string table with statistical results and descriptive statistics"""
    table = "Statistical Analysis Summary\n"
    table += "=" * 50 + "\n\n"

    # Note if paired tests were used
    if stats_results.get('paired_test_used', False):
        table += "Note: Paired/repeated statistical test was used.\n\n"
    else:
        table += "Note: Unpaired statistical test was used.\n\n"
    # Calculate and display descriptive statistics if data is provided
    if all(v is not None for v in [data, x_col, y_col, replicate_col]):
        _, treatment_stats = calculate_hierarchical_stats(data, x_col, y_col, replicate_col)

        table += "Descriptive Statistics (based on replicate means):\n"
        table += "-" * 50 + "\n"
        for treatment, stats_dict in treatment_stats.items():
            table += f"Group {treatment}:\n"
            table += f"  Mean ± SEM: {stats_dict['mean']:.2f} ± {stats_dict['sem']:.2f}\n"
            table += f"  N (replicates) = {stats_dict['n_replicates']}\n"
            table += f"  N (total observations) = {stats_dict['n_total']}\n"
        table += "\n"

    # Normality tests
    table += "Normality Test (Shapiro-Wilk):\n"
    table += "-" * 50 + "\n"
    for result in stats_results['shapiro_results']:
        table += f"Group {result['group']}:\n"
        if result.get('p_value') is not None:
            table += f"  p-value: {result['p_value']:.4f}"
            table += f" ({'Normal' if result.get('normal', False) else 'Non-normal'})\n"
        else:
            table += f"  {result.get('note', 'Insufficient data for test')}\n"
    table += "\n"

    # Homoscedasticity test
    if stats_results['levene_result']:
        table += "Homoscedasticity Test (Levene):\n"
        table += "-" * 50 + "\n"
        table += f"p-value: {stats_results['levene_result']['p_value']:.4f}\n"
        table += f"({'Homoscedastic' if stats_results['levene_result']['p_value'] > 0.05 else 'Heteroscedastic'})\n\n"

    # Main statistical test
    if stats_results['main_test']['name']:
        table += f"Main Test ({stats_results['main_test']['name']}):\n"
        table += "-" * 50 + "\n"
        if stats_results['main_test']['p_value'] is not None:
            table += f"p-value: {stats_results['main_test']['p_value']:.4f}\n"
        else:
            table += "No p-value available\n"
        table += "\n"

    # Pairwise comparisons
    if stats_results['pairwise_results']:
        table += "Pairwise Comparisons:\n"
        table += "-" * 50 + "\n"
        for result in stats_results['pairwise_results']:
            table += f"{result['group1']} vs {result['group2']}: "
            table += f"p = {result['p_value']:.4f}"
            table += f" ({'Significant' if result['p_value'] < 0.05 else 'Not significant'})"
            if 'test' in result:
                table += f" [{result['test']}]"
            table += "\n"

    return table

def create_interactive_superplot(data, x_col, y_col, replicate_col, paired=False, width=900, height=720, font_size=18, color_map='Plotly', template='plotly', marker_size=6, downsample_mode='all', downsample_value=None):
    """
    Create an interactive SuperPlot visualization with all customization options.
    """
    stats_results = perform_statistical_analysis(data, x_col, y_col, replicate_col, paired=paired)

    if hasattr(px.colors.qualitative, color_map):
        replicate_colors = getattr(px.colors.qualitative, color_map)
    else:
        replicate_colors = px.colors.qualitative.Plotly

    unique_replicates = sorted(data[replicate_col].unique())
    if len(replicate_colors) < len(unique_replicates):
        replicate_colors = [replicate_colors[i % len(replicate_colors)] for i in range(len(unique_replicates))]
    replicate_to_color = dict(zip(unique_replicates, replicate_colors))

    replicate_means = data.groupby([x_col, replicate_col])[y_col].mean().reset_index()
    treatment_stats = replicate_means.groupby(x_col)[y_col].agg(['mean', 'sem']).reset_index()

    fig = go.Figure()

    for replicate in unique_replicates:
        replicate_data = data[data[replicate_col] == replicate]

        data_to_plot = replicate_data
        if downsample_value is not None:
            try:
                if downsample_mode == 'percentage':
                    frac = max(0.01, min(1.0, float(downsample_value) / 100.0))
                    data_to_plot = replicate_data.sample(frac=frac, random_state=42)
                elif downsample_mode == 'number':
                    n_samples = max(1, int(downsample_value))
                    if n_samples < len(replicate_data):
                        data_to_plot = replicate_data.sample(n=n_samples, random_state=42)
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
            x=x_jittered, y=data_to_plot[y_col], mode='markers',
            name=f'{replicate}',
            marker=dict(size=marker_size, color=replicate_to_color[replicate], opacity=0.6),
            legendgroup=f'replicate_{replicate}',
            hovertemplate=(f"<b>Replicate {replicate}</b><br>Treatment: %{{text}}<br>{y_col}: %{{y:.2f}}<br><extra></extra>"),
            text=data_to_plot[x_col]
        ))

        replicate_mean_data = replicate_means[replicate_means[replicate_col] == replicate]
        x_values, y_values, text_values = [], [], []
        for _, row in replicate_mean_data.iterrows():
            try:
                x_values.append(list(treatment_stats[x_col]).index(row[x_col]))
                y_values.append(row[y_col])
                text_values.append(row[x_col])
            except ValueError:
                continue

        if x_values:
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='markers', name=f'Mean (Replicate {replicate})',
                marker=dict(size=marker_size * 1.8, color=replicate_to_color[replicate], symbol='circle', line=dict(
                color='gray',      # Set the outline color here
                width=0.5            # Set the outline width
            )),
                legendgroup=f'replicate_{replicate}', showlegend=False,
                hovertemplate=(f"<b>Replicate {replicate} Mean</b><br>Treatment: %{{text}}<br>{y_col}: %{{y:.2f}}<br><extra></extra>"),
                text=text_values
            ))

    fig.add_trace(go.Scatter(
        x=list(range(len(treatment_stats))), y=treatment_stats['mean'], mode='markers', name='Treatment Mean ± SEM',
        error_y=dict(type='data', array=treatment_stats['sem'], visible=True, color='gray'),
        marker=dict(color='gray', size=marker_size * 2.3, symbol='circle-open', line=dict(color='gray', width=2)),
        hovertemplate=("<b>Treatment Mean</b><br>Treatment: %{text}<br>" + f"{y_col}: %{{y:.2f}} ± %{{error_y.array:.2f}}<br><extra></extra>"),
        text=treatment_stats[x_col]
    ))

    if len(data) > 0:
        max_y = data[y_col].max()
        y_range = data[y_col].max() - data[y_col].min()
        if y_range == 0: y_range = max_y * 0.1 if max_y != 0 else 1.0

        for idx, result in enumerate(stats_results['pairwise_results']):
            if result['p_value'] < 0.05:
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
        xaxis=dict(tickmode='array', ticktext=treatment_stats[x_col], tickvals=list(range(len(treatment_stats))), tickfont=dict(size=font_size * 0.9), title_font=dict(size=font_size)),
        yaxis=dict(title=dict(text=y_col, font=dict(size=font_size)), tickfont=dict(size=font_size * 0.9), range=[data[y_col].min() - y_range * 0.1 if len(data) > 0 else 0, max_y + y_range * (0.1 * (len(stats_results['pairwise_results']) + 2)) if len(data) > 0 else 1]),
        width=width, height=height, hovermode='closest', template=template,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05, font=dict(size=font_size * 0.8)),
        font=dict(size=font_size), margin=dict(r=150, t=100)
    )

    return fig, stats_results, {'data': data, 'x_col': x_col, 'y_col': y_col, 'replicate_col': replicate_col, 'paired': paired}



# ==============================================================================
# Dash App
# ==============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row([
    dbc.Col(html.Img(src=app.get_asset_url('logo.png'), height="300px"))
    # dbc.Col(ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.DARKLY, dbc.themes.BOOTSTRAP]), className="ms-auto")
], align="center", className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            dcc.Upload(id='upload-data', children=dbc.Button("Upload Data", color="primary"), multiple=False, className="mb-3"),
            html.Div(id='upload-feedback', className="text-muted"),
            html.Div([
                html.H5("Map your spreadsheet columns to the required fields", className="mt-3 mb-2"),
                dbc.Row([
                    dbc.Col([html.Label("Treatment/Group Column:"), dcc.Dropdown(id='treatment-col', placeholder="Select column...", className="mb-2")], width=4),
                    dbc.Col([html.Label("Value/Measurement Column:"), dcc.Dropdown(id='value-col', placeholder="Select column...", className="mb-2")], width=4),
                    dbc.Col([html.Label("Replicate Column:"), dcc.Dropdown(id='replicate-col', placeholder="Select column...", className="mb-2")], width=4)
                ]),
                dbc.Checklist(id='paired-test-toggle', options=[{'label': 'Use paired/repeated statistical tests', 'value': 1}], value=[], switch=True, className="mt-2 mb-2"),
                dbc.Button("Generate Analysis/Update Plot", id="generate-analysis", color="success", className="mt-3", disabled=True)
            ], id='column-mapping', style={'display': 'none'})
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader("Plot Customization"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.Label("Font Size:"), dcc.Slider(id='font-size-slider', min=10, max=45, step=1, value=18, marks={i: str(i) for i in range(10, 46, 2)})], width=3),
                dbc.Col([html.Label("Color Map:"), dcc.Dropdown(id='color-map-dropdown', options=[{'label': k, 'value': k} for k in px.colors.qualitative.__dict__.keys() if isinstance(px.colors.qualitative.__dict__[k], list)], value='Plotly')], width=3),
                dbc.Col([html.Label("Plot Style:"), dcc.Dropdown(id='plot-style-dropdown', options=[{'label': s, 'value': s} for s in ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']], value='plotly')], width=3),
                dbc.Col([html.Label("Marker Size:"), dcc.Slider(id='marker-size-slider', min=2, max=20, step=2, value=6, marks={i: str(i) for i in range(2, 21, 2)})], width=3),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([html.Label("Data to Display"), dcc.RadioItems(id='downsample-mode-radio', options=[{'label': 'All ', 'value': 'all'},
                 {'label': 'Percentage ', 'value': 'percentage'}, {'label': 'Fixed Number ', 'value': 'number'}], value='all', inline=True, className="mb-2", labelStyle={'margin-right': '20px'})], width=4),
                dbc.Col([html.Label("Percentage (%) "), dcc.Input(id='downsample-percentage-input', type='number', min=1, max=100, step=1, value=50, disabled=True, className="mb-2")], width=4),
                dbc.Col([html.Label("Fixed Number "), dcc.Input(id='downsample-number-input', type='number', min=1, step=1, value=20, disabled=True, className="mb-2")], width=4),
            ])
        ])
    ], className="mb-4"),


dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.H5("Interactive Plot", className="mb-0"), width="auto"),
                    dbc.Col(
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(
                                id='plot-download-format',
                                options=[
                                    {'label': 'PNG', 'value': 'png'},
                                    {'label': 'JPEG', 'value': 'jpeg'},
                                    {'label': 'SVG', 'value': 'svg'},
                                    {'label': 'PDF', 'value': 'pdf'}
                                ],
                                value='png', clearable=False, style={'width': '120px'}
                            ), width="auto"),
                            dbc.Col(dcc.Dropdown(
                                id='plot-download-resolution',
                                options=[
                                    {'label': 'Standard Res', 'value': 'std'},
                                    {'label': 'High Res (2x)', 'value': 'high'},
                                ],
                                value='std', clearable=False, style={'width': '150px'}
                            ), width="auto"),
                            dbc.Col(dbc.Button("Download Plot", id="btn-download-plot", color="success", disabled=True), width="auto"),
                        ], justify="end", className="g-2")
                    )
                ], align="center", justify="between")
            ),
            dbc.CardBody([
                dcc.Graph(id='plot'),
                dcc.Download(id="download-plot")  # ADDED: dcc.Download component
            ])
        ])
    ], width=12)
], className="mb-4"),

    dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader([dbc.Row([dbc.Col(html.H5("Statistical Analysis Summary", className="mb-0"), width="auto"), dbc.Col([dbc.Button("Download Summary", id="btn-download", color="success", className="float-end", disabled=True), dcc.Download(id="download-stats")], width="auto")], align="center")]),
        dbc.CardBody([html.Div(id='stats-text', style={'whiteSpace': 'pre-line'})])
    ])], width=12)])
], fluid=True)

# --- App Callbacks ---
@app.callback(
    Output('column-mapping', 'style'), Output('treatment-col', 'options'), Output('value-col', 'options'), Output('replicate-col', 'options'), Output('upload-feedback', 'children'),
    Input('upload-data', 'contents'), State('upload-data', 'filename')
)
def update_column_mapping(contents, filename):
    if not contents:
        return {'display': 'none'}, [], [], [], ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return {'display': 'none'}, [], [], [], "Unsupported file type. Please upload a CSV or XLSX file."

        server.df = df
        options = [{'label': col, 'value': col} for col in df.columns]
        return {'display': 'block'}, options, options, options, f"Successfully loaded: {filename} ({len(df)} rows)."
    except Exception as e:
        return {'display': 'none'}, [], [], [], f"Error loading file: {str(e)}"

@app.callback(
    Output('generate-analysis', 'disabled'),
    Input('treatment-col', 'value'), Input('value-col', 'value'), Input('replicate-col', 'value')
)
def update_generate_button(treatment, value, replicate):
    return not all([treatment, value, replicate])

@app.callback(
    Output('downsample-percentage-input', 'disabled'), Output('downsample-number-input', 'disabled'),
    Input('downsample-mode-radio', 'value')
)
def toggle_downsample_inputs(mode):
    if mode == 'percentage': return False, True
    if mode == 'number': return True, False
    return True, True


@app.callback(
    Output('plot', 'figure'),
    Output('stats-text', 'children'),
    Output('btn-download', 'disabled'),
    Output('btn-download-plot', 'disabled'),
    Input('generate-analysis', 'n_clicks'),  # The only trigger
    # Use State for all other parameters so they are only read on button click
    State('treatment-col', 'value'),
    State('value-col', 'value'),
    State('replicate-col', 'value'),
    State('paired-test-toggle', 'value'),
    State('font-size-slider', 'value'),
    State('color-map-dropdown', 'value'),
    State('plot-style-dropdown', 'value'),
    State('marker-size-slider', 'value'),
    State('downsample-mode-radio', 'value'),
    State('downsample-percentage-input', 'value'),
    State('downsample-number-input', 'value')
)
def generate_analysis(n_clicks, treatment_col, value_col, replicate_col, paired_toggle,
                      font_size, color_map, plot_style, marker_size,
                      downsample_mode, downsample_percentage, downsample_number):
    
    # If the button has never been clicked, do nothing.
    if n_clicks is None:
        # Provide an empty plot and initial instructions
        return go.Figure(), "Upload data and click 'Generate Analysis/Update Plot' to begin.", True, True

    # Check if dataframe exists and all required dropdowns are selected
    if not all([treatment_col, value_col, replicate_col]) or not hasattr(server, 'df'):
        return no_update, "Please select all columns to generate the analysis.", True, True

    downsample_value = downsample_percentage if downsample_mode == 'percentage' else downsample_number if downsample_mode == 'number' else None
    use_paired_tests = 1 in paired_toggle

    try:
        fig, stats_results, plot_data = create_interactive_superplot(
            server.df, treatment_col, value_col, replicate_col, paired=use_paired_tests,
            font_size=font_size, color_map=color_map, template=plot_style,
            marker_size=marker_size, downsample_mode=downsample_mode, downsample_value=downsample_value
        )
        stats_table = create_stats_table(stats_results, plot_data['data'], plot_data['x_col'], plot_data['y_col'], plot_data['replicate_col'])
        # Enable both download buttons on success
        return fig, stats_table, False, False
    except Exception as e:
        import traceback
        # On error, return an empty plot and the error message
        return go.Figure(), f"An error occurred: {e}\n{traceback.format_exc()}", True, True


@app.callback(
    Output("download-stats", "data"),
    Input("btn-download", "n_clicks"),
    State("stats-text", "children"),
    prevent_initial_call=True
)
def download_stats(n_clicks, stats_text):
    if not stats_text:
        return None

    filename = f"statistical_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"

    return dict(
        content=stats_text,
        filename=filename,
        type="text/plain"
    )

# --- Add this new callback to the end of your file ---
@app.callback(
    Output("download-plot", "data"),
    Input("btn-download-plot", "n_clicks"),
    State("plot", "figure"),
    State("plot-download-format", "value"),
    State("plot-download-resolution", "value"),
    State("value-col", "value"),
    prevent_initial_call=True,
)
def download_plot(n_clicks, figure, dl_format, dl_resolution, y_col_name):
    if not figure:
        return no_update

    try:
        # Recreate the Plotly figure object from the dictionary stored in dcc.Graph
        fig = go.Figure(figure)

        # Set image scale based on the resolution dropdown
        scale = 2 if dl_resolution == 'high' else 1

        # Generate the image bytes in the chosen format and scale
        img_bytes = fig.to_image(format=dl_format, scale=scale)

        # Create a descriptive filename
        y_col_name_str = y_col_name.replace(" ", "_") if y_col_name else "plot"
        filename = f"superplot_{y_col_name_str}_{pd.Timestamp.now().strftime('%Y%m%d')}.{dl_format}"

        # Use dcc.send_bytes or dcc.send_string for an easy download interface
        if dl_format in ['png', 'jpeg', 'pdf']:
            return dcc.send_bytes(img_bytes, filename)
        elif dl_format == 'svg':
            # SVG is text-based, so it must be decoded before sending
            return dcc.send_string(img_bytes.decode(), filename)

    except Exception as e:
        print(f"Error during plot download: {e}")
        return no_update

if __name__ == '__main__':
    webbrowser.open_new("http://127.0.0.1:8050")
    app.run(debug=False)