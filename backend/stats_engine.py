import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from itertools import combinations


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
            table += f"  Mean +/- SEM: {stats_dict['mean']:.2f} +/- {stats_dict['sem']:.2f}\n"
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
