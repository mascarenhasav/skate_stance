import pandas as pd
from scipy.stats import mannwhitneyu
def mann_whitney(df, X, y):
    # Separando grupos Regular e Goofy para análise
    regular_group = df[df['stance_binary'] == 1]
    goofy_group = df[df['stance_binary'] == 0]

    mannwhitney_results = {}

    # Aplicando testes ANOVA e Mann-Whitney para cada variável
    for column in X.columns:
        mannwhitney = mannwhitneyu(regular_group[column], goofy_group[column])
        mannwhitney_results[column] = mannwhitney.pvalue

    # Resultados em um dataframe para melhor visualização
    stat_results_df = pd.DataFrame({
        'MannWhitney_pvalue': mannwhitney_results
    }).sort_values(by='MannWhitney_pvalue')

    # Ajustar valores para visualização melhor
    stat_results_df['MannWhitney_Significant'] = stat_results_df['MannWhitney_pvalue'] < 0.05

    return stat_results_df