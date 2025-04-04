import pandas as pd
from scipy.stats import f_oneway


def anova(df, X, y):
    # Separando grupos Regular e Goofy para análise
    regular_group = df[df['stance_binary'] == 1]
    goofy_group = df[df['stance_binary'] == 0]

    anova_results = {}

    # Aplicando testes ANOVA e Mann-Whitney para cada variável
    for column in X.columns:
        anova = f_oneway(regular_group[column], goofy_group[column])
        anova_results[column] = anova.pvalue

    # Resultados em um dataframe para melhor visualização
    stat_results_df = pd.DataFrame({
        'ANOVA_pvalue': anova_results,
    }).sort_values(by='ANOVA_pvalue')

    # Ajustar valores para visualização melhor
    stat_results_df['ANOVA_Significant'] = stat_results_df['ANOVA_pvalue'] < 0.05

    return stat_results_df