import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width


def distributions(df, cols=[]):
    df = df[cols]
    
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(cols):
        plt.subplot(5, 3, i+1)
        if col == 'skate_stance':
            color = 'red'
        else:
            color = 'brown'
        sns.histplot(df[col], bins=20, kde=True, color=color)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('../images/distributions_lateralities.png')
    plt.show()