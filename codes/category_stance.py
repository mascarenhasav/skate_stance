import matplotlib.pyplot as plt
from cycler import cycler

plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width

def category_stance(df):
    
    df_cat = df.copy()

    df_cat['SS'] = (df_cat['skate_stance'] + df_cat['ollie_foot'] + df_cat['bowl_foot'] + df_cat['downhill_foot'])/4
    df_cat['HS'] = (df_cat['hand_write'] + df_cat['hand_throw'] + df_cat['hand_hammer'])/3
    df_cat['FS'] = (df_cat['foot_kick'] + df_cat['foot_pedal'] + df_cat['foot_chair'])/3
    df_cat['ES'] = (df_cat['eye_test1'] + df_cat['eye_test2'])/2

    df_cat.dropna(inplace=True)
    df_cat.reset_index(drop=True, inplace=True)
    # Drop any column with "Unnamed" in its name
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Alternatively, specifically drop the "Unnamed: x" column
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    #df_cat.to_csv("df_cat.csv")