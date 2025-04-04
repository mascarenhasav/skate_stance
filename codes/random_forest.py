from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width


def random_forest(X, y, X_train, y_train, X_test, y_test):
    # Modelo RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Goofy (0)', 'Regular (1)']))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred_rf),
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=1.5,
        linecolor='black',
        annot_kws={"size": 16}
    )

    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(ticks=[0.5,1.5], labels=['Goofy (0)', 'Regular (1)'])
    plt.yticks(ticks=[0.5,1.5], labels=['Goofy (0)', 'Regular (1)'])
    plt.tight_layout()
    plt.show()

    # ----- Importance
    importances = rf_model.feature_importances_
    feature_names = X_test.columns

    # Organizar e ordenar
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        y=feat_df['Feature'],
        x=feat_df['Importance'],
        palette='coolwarm',
        hue = feat_df['Feature'],
        legend=False,
        edgecolor='black',     # Adiciona contorno preto
        linewidth=1.5          # Define espessura do contorno
    )
    plt.xlabel('Feature Importance')
    plt.title('Random Forest – Feature Importance')
    plt.tight_layout()
    plt.show()

    # ------- ROC
    # Probabilidades para classe positiva (classe Regular = 1)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Random Forest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return rf_model, fpr, tpr, thresholds