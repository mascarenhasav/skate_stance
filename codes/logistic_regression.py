import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from cycler import cycler
plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width

lr_confusion_path = "../images/lr_confusion.png"
lr_coefs_path = "../images/lr_coefs.png"
lr_roc_path = "../images/lr_roc.png"

def logistic_regression(X, y, X_train, y_train, X_test, y_test):
    # Modelo Regressão Logística
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
          
    # Validação cruzada com regressão logística para garantir robustez
    cv_scores = cross_val_score(log_model, X, y, cv=10, scoring='accuracy')
    cv_mean_accuracy = cv_scores.mean()

    # Ajuste fino de hiperparâmetros (GridSearchCV)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X, y)

    # Melhores parâmetros encontrados
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Cross-Validation Accuracy: {cv_mean_accuracy}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    
    # Treinar o modelo otimizado com melhores hiperparâmetros
    opt_log_model = LogisticRegression(C=best_params["C"], penalty=best_params["penalty"], solver=best_params["solver"], max_iter=1000)
    opt_log_model.fit(X, y)
    y_pred_log = opt_log_model.predict(X_test)

    # Extrair coeficientes (importância das features)
    feature_importance = pd.Series(opt_log_model.coef_[0], index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=True)
    
    # reports ----------------------------------------------------------------
    
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_log, target_names=['Goofy (0)', 'Regular (1)']))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred_log),
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=1.5,
        linecolor='black',
        annot_kws={"size": 16}
    )
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(ticks=[0.5,1.5], labels=['Goofy (0)', 'Regular (1)'])
    plt.yticks(ticks=[0.5,1.5], labels=['Goofy (0)', 'Regular (1)'])
    plt.tight_layout()
    plt.savefig(lr_confusion_path)
    plt.show()

    # ----- Importance
    # Obter coeficientes
    coefficients = log_model.coef_[0]
    feature_names = X_test.columns

    # DataFrame com ordenação
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['AbsCoefficient'] = coef_df['Coefficient']
    coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=True)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        y=coef_df['Feature'],
        x=coef_df['AbsCoefficient'],
        palette='coolwarm',
        hue = coef_df['Feature'],
        legend=False,       # Remover legenda
        edgecolor='black',     # Adiciona contorno preto
        linewidth=1.5          # Define espessura do contorno
    )
    #plt.axvline(x=0, color='black', linestyle='--', lw=1)
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression – Feature Coefficients')
    plt.tight_layout()
    plt.savefig(lr_coefs_path)
    plt.show()


    # ------- ROC
    # Obter probabilidades da classe positiva (Regular = 1)
    y_proba_lr = log_model.predict_proba(X_test)[:, 1]

    # Calcular ROC
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_lr:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(lr_roc_path)
    plt.show()
    
    return opt_log_model, fpr_lr, tpr_lr, thresholds_lr