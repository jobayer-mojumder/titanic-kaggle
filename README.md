<!-- @format -->

# 🧠 Feature Engineering vs. Model Tuning to Increase Data Mining Model Quality

This repository implements a full experimental pipeline to evaluate the impact of **Feature Engineering (FE)** and **Model Tuning (MT)** on the predictive performance of machine learning models using the Titanic dataset. It supports automated execution of 11 experimental steps across 5 models, with complete statistical summaries and ANOVA-ready exports.

---

## 📁 Project Structure

```
📦src/
├── analysis.py               # Core logic for evaluating best/top/balanced combinations
├── anova.py                  # JASP-compatible ANOVA table generator
├── constant.py               # Stores model keys and baseline accuracy scores
├── data.py                   # Titanic dataset loading and formatting
├── dispatcher.py             # CLI dispatcher routing 11 experiments
├── evaluation.py             # Model evaluation metrics (CV, Kaggle)
├── feature_implementation.py# Modular feature engineering definitions
├── main.py                   # Entry point to run any experiment
├── model_tuning.py           # Hyperparameter search via RandomizedSearchCV
├── plots.py                  # Accuracy, improvement, and ηp² plotting functions
├── preprocessing.py          # Cleans, fills, encodes and prepares data
├── runner.py                 # Executes model training per experiment step
├── stats.py                  # CLI reporting tool for all statistics
├── summary.py                # Logs results per experiment (Kaggle/Local)
```

---

## 🔬 Experimental Design

### 🧪 Feature Engineering Experiments (Untuned)

| No. | Description                                         |
| --- | --------------------------------------------------- |
| 1   | Baseline model (no FE or tuning)                    |
| 2   | Add each of the 12 engineered features individually |
| 3   | Top 10 combinations of features per model           |
| 4   | Best single combination per model                   |
| 5   | All 12 features combined                            |
| 6   | Top 3 features per model (based on importance)      |

### 🧪 Model Tuning Experiments

| No. | Description                                |
| --- | ------------------------------------------ |
| 7   | Tune baseline model (no FE)                |
| 8   | Tune models with each feature individually |
| 9   | Tune top 10 feature combinations           |
| 10  | Tune best combination per model            |
| 11  | Tune all features together                 |

### 📊 Statistics

| No. | Description                                     |
| --- | ----------------------------------------------- |
| 12  | Partial η², CV accuracy, and improvement scores |
|     | ANOVA-ready CSV exports for JASP                |

---

## ⚙️ Models Supported

-   Decision Tree
-   XGBoost
-   Random Forest
-   LightGBM
-   CatBoost

Each model is run with and without tuning across multiple feature configurations.

---

## 📈 Features Used

12 engineered features including:

-   Title
-   FamilySize
-   IsAlone
-   AgeGroup
-   FarePerPerson
-   Deck
-   IsMother
-   TicketPrefix
-   and more...

---

## 🛠 How to Run

```bash
# Launch experiments or view stats interactively
cd src
python main.py
```

Choose an experiment step [1–11] or view stats [12] interactively.

---

## 📊 Output Folders

```
📁results/
├── kaggle/                   # Kaggle accuracy for each run
├── local/                    # Cross-validation scores
📁stats/
├── kaggle/ or /local/       # Merged and interpreted statistical summaries
📁plots/
├── *.png                     # Visualizations of ηp² and accuracy improvements
```

---
