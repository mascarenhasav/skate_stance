import pre_processing as pp
import spearman_corr as sc
import anova as an
import mann_whitney as mw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random_forest as rf
import logistic_regression as lr
import predict_stance as ps
import clustering as cl
import category_stance as cs
import distributions as dist
import keyboard

def wait_enter(message="\nPress [Enter] to next analysis..."):
    key = input(message+": ")
    if key == "":
        return 1
    else:
        return 0


if __name__ == "__main__":
    path = '../data/data.csv'
      
    df_init = pp.pre_processing(path)
    
    
    # Remove the columns that is not lateralities to be analyzed
    columns_to_drop = ['friends_share_stance', 'parents_share_stance', 'fav_skater_share_stance',
                    'changed_stance', 'stance_awareness', 'consistent_stance', "expertise",
                    "gender", "race", "age", "ethnicity", "residence_locality", "home_locality",
                    "years_skate", "freq_skate",
                    "downhill_foot",
                    "ollie_foot",
                    "bowl_foot",
                    "snowboard_foot",
                    "surf_foot"]
    df = df_init.copy()
    df.drop(columns=columns_to_drop, inplace=True)
    df.to_csv(f"teste.csv", index=False)


    # filter only the goofy and regular skaters
    df = df[df["skate_stance"].isin([-10, 10])]
    df = df.dropna()
    
    # Converter skate_stance para binária (Goofy=0, Regular=1)
    df['stance_binary'] = df['skate_stance'].map({-10: 0, 10: 1})
    print("[PREPROCESING] dataset after filtering data, only the lateralities")
    #print(df.info())
    
    # select dependent and independent variables ---------------------
    X = df.drop(columns=['skate_stance', 'stance_binary'])
    y = df['stance_binary']
    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #display(sig_corrs.sort_values('Spearman_Corr', ascending=False))
    # ----------------------------------------------------------------
    
    print("----------------------------------------------------------------")    
    country_dist_flag = wait_enter(f"[COUNTRY DISTRIBUTION] Press enter to continue, any other key to skip")    
    if country_dist_flag:
        # country distribution ----------------------------------------
        
        print("[COUNTRY DISTRIBUTION] country distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title("Country Distribution")
        sns.countplot(data=df_init, x='residence_locality', order=df_init['residence_locality'].value_counts().index)
        plt.xticks(rotation=90)
        plt.savefig("../images/country_dist.png")
        plt.show()
        #print(df.head())
        print("----------------------------------------------------------------")
        # -------------------------------------------------------------
    
    print("----------------------------------------------------------------")
    distributions_flag = wait_enter(f"[DISTRIBUTION] Press enter to continue, any other key to skip")    
    if distributions_flag:
        # distributions of skate stance and other columns ----------------
        print("[DISTRIBUTIONS] distributions of the variables")
        variables_to_distribution = [
        'skate_stance',  # Q1 - Variável alvo
        'ollie_foot',    # Q2
        'bowl_foot',     # Q3
        'downhill_foot', # Q4
        "surf_foot",     # Q5
        "snowboard_foot",# Q6
        'hand_write',    # Q15
        'hand_throw',    # Q16
        'hand_hammer',   # Q17
        'foot_kick',     # Q18
        "foot_sweep",    # Q19
        'foot_pedal',    # Q20
        'foot_chair',    # Q21
        'eye_test1',     # Q22
        'eye_test2'      # Q23
        ]
        dist.distributions(df_init, variables_to_distribution)
        print("---------------------------------------------------------------")
        # -----------------------------------------------------------------
        
    print("----------------------------------------------------------------")
    category_stance_flag = wait_enter(f"[CATEGORY STANCE] Press enter to continue, any other key to skip")    
    if category_stance_flag:
        # category stance ----------------------------------------------------------------
        print(f"[CATEGORY STANCE] Applying category stance")
        cs.category_stance(df_init)
        print("----------------------------------------------------------------")
        # ----------------------------------------------------------------
    
    print("----------------------------------------------------------------")
    spearman_corr_flag = wait_enter(f"[SPERMAN CORRELATION] Press enter to continue, any other key to skip")    
    if spearman_corr_flag:
        # Spearman Correlation -------------------------------------------
        path_matrix_corr = "../images/correlation_matrix_esp.png"
        print("[SPEARMAN] Spearman Correlation Matrix")
        spearman_corr_matrix, spearman_corr_matrix_mask = sc.spearman_correlation(df)
        print(spearman_corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            spearman_corr_matrix,
            mask=None,  # se quiser ocultar as cels não significativas, use mask=mask_not_significant
            annot=True, # se quiser ver os valores de correlação na heatmap
            cmap='coolwarm',
            center=0,
            fmt='.1f'
        )
        plt.title("Correlation Matrix (Spearman) Ordered by correlation with skate_stance\n")
        plt.savefig(path_matrix_corr)
        plt.show()
        print(f"[SPEARMAM] figure saved in {path_matrix_corr}")
        print("----------------------------------------------------------------")
        # ----------------------------------------------------------------
    
    
    print("----------------------------------------------------------------")
    variance_analysis_flag = wait_enter(f"[VARIANCE ANALYSIS] Press enter to continue, any other key to skip")    
    if variance_analysis_flag:
        # Variance analysis ----------------------------------------------
        print("[VARIANCE] anova")
        anova_stats = an.anova(df, X, y) # anova stats
        print(f"\n[ANOVA] Results: \n{anova_stats}")
        print("----------------------------------------------------------------")
        print("[VARIANCE] mann-whitney")
        mann_stats = mw.mann_whitney(df, X, y) # mw stats
        print(f"\n[MANN-WHITNEY] Results: \n{mann_stats}")
        print("----------------------------------------------------------------")
        # ----------------------------------------------------------------

    # Classification -------------------------------------------------
    print("----------------------------------------------------------------")
    random_forest_flag = wait_enter(f"[RANDOM FOREST] Press enter to continue, any other key to skip")    
    if random_forest_flag:
        # random forest
        print(f"[MODEL] random forest")
        rf_model, rf_fpr, rf_tpr, rf_thresholds = rf.random_forest(X, y, X_train, y_train, X_test, y_test)
        print("----------------------------------------------------------------")

    print("----------------------------------------------------------------")
    logistic_regression_flag = wait_enter(f"[LOGISCTIC REGRESSION] Press enter to continue, any other key to skip")    
    if logistic_regression_flag:
        # logistic regression
        print(f"[MODEL] logistic regression")
        lr_model, lr_fpr, lr_tpr, lr_thresholds = lr.logistic_regression(X, y, X_train, y_train, X_test, y_test)
        print("----------------------------------------------------------------")
    # ----------------------------------------------------------------
    
    print("----------------------------------------------------------------")
    prediction_flag = wait_enter(f"[PREDICTION] Press enter to continue, any other key to skip")    
    if prediction_flag:
        # prediction of stance -------------------------------------------
        print(f"[PREDICTION] Predicting stance for a new person")
        ex_new_person = {
            "hand_write": 10,
            "hand_throw": 10,
            "hand_hammer": 10,
            "foot_kick": -5,
            "foot_sweep": -5,
            "foot_pedal": 5,
            "foot_chair": -5,
            "eye_test1": 10,
            "eye_test2": -5
        }
        # Exibir probabilidades para o exemplo
        prediction = ps.predict_stance(lr_model, ex_new_person)
        print(prediction)
        print("----------------------------------------------------------------")
    # ----------------------------------------------------------------
    
    
    print("----------------------------------------------------------------")
    clustering_flag = wait_enter(f"[CLUSTERING] Press enter to continue, any other key to skip")    
    if clustering_flag:
        # clustering ----------------------------------------------------------------
        print(f"[CLUSTERING] K-means clustering")
        cl.clustering(df)
        print("----------------------------------------------------------------")
        # ---------------------------------------------------------------------------
        
    
    