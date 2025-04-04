from scipy.stats import spearmanr
import itertools
import pandas as pd
import numpy as np


def spearman_correlation(df_temp):  

    numeric_cols = [
        'skate_stance','ollie_foot','bowl_foot','snowboard_foot','surf_foot',
        'foot_sweep','foot_chair','foot_pedal','foot_kick',
        'hand_write','hand_throw','hand_hammer','eye_test1','eye_test2'
    ]
    numeric_cols = [c for c in numeric_cols if c in df_temp.columns]
    df_sub = df_temp[numeric_cols].dropna()

    # 1) Calcular correlação e valor-p de Spearman para cada par
    results = []
    for var1, var2 in itertools.combinations(numeric_cols, 2):
        corr, pval = spearmanr(df_sub[var1], df_sub[var2])
        results.append((var1, var2, corr, pval))

    # 2) DataFrame com resultados
    corr_df = pd.DataFrame(results, columns=['Var1','Var2','Spearman_Corr','p_value'])

    # 3) Correção de Bonferroni (opcional)
    n_tests = len(corr_df)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests
    corr_df['significant_bonferroni'] = corr_df['p_value'] < bonferroni_alpha

    # 4) Ver correlações significativas
    sig_corrs = corr_df[corr_df['significant_bonferroni']]
    print(f"Foram realizados {n_tests} testes. Alpha inicial={alpha}, Bonferroni Alpha={bonferroni_alpha}")
    print("Correlações significativas após correção de Bonferroni:")

    # ---- PARTE IMPORTANTE: Ordenar colunas pelo valor de correlação com skate_stance ----
    # Criar a matriz de correlação Spearman
    corr_matrix = df_sub.corr(method='spearman')

    # Extrair as correlações de cada coluna com 'skate_stance'
    # Se quiser usar o módulo do coeficiente, use abs():
    skate_corr = corr_matrix['skate_stance']

    # Ordenar as colunas com base nesse valor (da maior para a menor correlação com skate_stance)
    ordered_cols = skate_corr.sort_values(ascending=False).index

    # Reordenar a matriz de correlação (linhas e colunas) nessa sequência
    corr_matrix = corr_matrix.loc[ordered_cols, ordered_cols]

    # 5) Criar uma matriz de p-values para poder mascarar correlações não significativas (opcional)
    p_val_matrix = pd.DataFrame(np.ones((len(ordered_cols), len(ordered_cols))),
                                columns=ordered_cols, index=ordered_cols)

    for var1, var2, corr, pval in results:
        # Somente se var1 e var2 estiverem na lista final
        if var1 in ordered_cols and var2 in ordered_cols:
            p_val_matrix.loc[var1, var2] = pval
            p_val_matrix.loc[var2, var1] = pval

    mask_not_significant = p_val_matrix > bonferroni_alpha
    
    return corr_matrix, mask_not_significant