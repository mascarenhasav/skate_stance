# 🛹 Skate Stance and Human Laterality Analysis

This repository presents a complete analysis pipeline investigating **human lateralities** and their relationship to **skateboarding stance** (Goofy vs. Regular). The research combines exploratory data analysis, statistical inference, machine learning classification, and unsupervised clustering techniques to understand whether skate stance can be predicted from typical laterality patterns (hand, foot, eye dominance).

You can also check the data in real time on [Skate Stance](https://mascarenhasav.github.io/skate_stance)


## 🧠 Summary of the Research

Human laterality describes consistent preferences for one side of the body, such as handedness or footedness. This project explores whether **skate stance** is just another expression of laterality or a unique motor behavior. Using questionnaire data scored from `-10` (always left) to `+10` (always right), we performed:

- **Spearman correlation analysis** with Bonferroni correction  
- **ANOVA and Mann-Whitney U tests** to assess significance  
- **Supervised learning** (Logistic Regression and Random Forest)  
- **Unsupervised learning** (K-means Clustering with PCA visualization)

Results suggest that **footedness tasks** (e.g., stepping on a chair) have some predictive power for stance, but **hand and eye lateralities show minimal influence**. Even so, overall predictive performance is moderate, and skate stance may represent an independent motor dimension.

---

## 📁 Project Structure

    ├── bibtex/ # Bibliographic references in BibTeX format 
    ├── codes/ # Source code for analyses and models 
      └── main.py # Main script for stance prediction
    ├── data/ # Datasets used in the study 
    ├── images/ # Figures and visualizations 
    ├── LICENSE # License information 
    ├── README.md # Project overview and instructions 
    ├── requirements.txt # Python package requirements 

## ⚙️ Installation & Usage

### 🔧 Requirements (Python) and Installation

Ensure that Python 3.x is installed on your system. Required Python packages are listed in `requirements.txt`.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mascarenhasav/skate_stance.git
   cd skate_stance

2. **Install the required Python packages via:**

    ```bash
    pip install -r requirements.txt
    ```

Main libraries used:

- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy

### 🖥️ Running the Analysis

The analyses are organized within the codes/ directory. To replicate the study's findings:

1. Navigate to the codes/ directory:

    ```bash
    cd codes
    ```

2. Execute the analysis scripts:

    The main.py script provides a simple interface for predicting an individual's skate stance based on their lateral preference data.
    Ensure that the data/ directory contains the necessary datasets before running the scripts.
  
    Run the script:
  
    ```bash
    python main.py
    ```
    The script will prompt you to enter values for various lateral preference measures. After inputting the required data, it will output a prediction of the skate     stance (Regular or Goofy).

Note: The prediction model in main.py is based on the analyses conducted in this study and serves as a practical application of the research findings.


### 📊 Outputs and Results

Key outputs of the analyses include:

- Statistical Test Results: Summaries of ANOVA and Mann-Whitney U tests identifying significant associations.

- Machine Learning Performance Metrics: Classification reports detailing accuracy, precision, recall, and F1-scores for predictive models.
- Visualizations: Graphs and figures illustrating correlations, clustering results, and model performance, available in the images/ directory.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.


## 🙋 Acknowledgments

This research was conducted as part of an exploration into motor behavior and human laterality. Contributions, feedback, and collaborations are welcome.
