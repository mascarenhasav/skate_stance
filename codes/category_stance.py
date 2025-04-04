import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width

def category_stance(df):
    
    df_cat = pd.DataFrame()

    df_cat['SS'] = (df['skate_stance'] + df['ollie_foot'] + df['bowl_foot'] + df['downhill_foot'])/4
    df_cat['HS'] = (df['hand_write'] + df['hand_throw'] + df['hand_hammer'])/3
    df_cat['FS'] = (df['foot_kick'] + df['foot_pedal'] + df['foot_chair'])/3
    df_cat['ES'] = (df['eye_test1'] + df['eye_test2'])/2

    df_cat.dropna(inplace=True)
    df_cat.reset_index(drop=True, inplace=True)
    # Drop any column with "Unnamed" in its name
    #df_cat = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Alternatively, specifically drop the "Unnamed: x" column
    #df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    #df_cat.to_csv("df_cat.csv")
    
    X = ['SS', 'HS', 'FS', 'ES']  # por exemplo
    plot_continuous_grid(df_cat, X)
    
    plt.show()

def plot_continuous_grid(df_temp, columns, bins=10):
    if len(columns) != 4:
        raise ValueError("A função espera exatamente 4 colunas.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]

        # 1) Calcular bins automaticamente ou usar fixos
        bin_edges = np.histogram_bin_edges(df_temp[col].dropna(), bins=bins)

        # 2) Obter contadores e bordas
        counts, edges = np.histogram(df_temp[col].dropna(), bins=bin_edges)

        # 3) Calcular porcentagens
        pct = 100.0 * counts / counts.sum()

        # 4) Plotar histograma
        sns.histplot(df_temp[col], bins=bin_edges, kde=True, ax=ax)

        ax.set_title(f'Distribuição de {col}', fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        # Limite superior automático baseado nos dados
        ax.set_ylim(0, counts.max() + counts.max() * 0.2)

        # 5) Adicionar porcentagens nas barras
        for j, c in enumerate(counts):
            if c > 0:
                bin_center = (edges[j] + edges[j+1]) / 2
                ax.text(bin_center, c, f'{pct[j]:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

