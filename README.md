# 🛹 Skate Stance and Human Laterality Analysis

This repository presents a complete analysis pipeline investigating **human lateralities** and their relationship to **skateboarding stance** (Goofy vs. Regular). The research combines exploratory data analysis, statistical inference, machine learning classification, and unsupervised clustering techniques to understand whether skate stance can be predicted from typical laterality patterns (hand, foot, eye dominance).

## 🧠 Summary of the Research

Human laterality describes consistent preferences for one side of the body, such as handedness or footedness. This project explores whether **skate stance** is just another expression of laterality or a unique motor behavior. Using questionnaire data scored from `-10` (always left) to `+10` (always right), we performed:

- **Spearman correlation analysis** with Bonferroni correction  
- **ANOVA and Mann-Whitney U tests** to assess significance  
- **Supervised learning** (Logistic Regression and Random Forest)  
- **Unsupervised learning** (K-means Clustering with PCA visualization)

Results suggest that **footedness tasks** (e.g., stepping on a chair) have some predictive power for stance, but **hand and eye lateralities show minimal influence**. Even so, overall predictive performance is moderate, and skate stance may represent an independent motor dimension.

---

## 📁 Project Structure

