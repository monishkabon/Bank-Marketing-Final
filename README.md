# Bank Marketing Campaign Response Prediction

An end-to-end Machine Learning solution designed to predict customer term deposit subscription likelihood from outbound telemarketing campaigns, enabling cost-effective prospect prioritization and data-driven marketing outreach.

---

## Project Team & Responsibilities

| Contributor | Focus Area & Responsibility | GitHub Profile |
| :--- | :--- | :--- |
| **Monishka Athukorala** | **Data Preprocessing & Feature Engineering Pipeline** (Phase 2)<br>• Custom Scikit-Learn domain transformers<br>• Sentinel mapping & recency/age cohort binning<br>• Leakage-safe ColumnTransformer & train/test partitioning | [@monishkabon](https://github.com/monishkabon) |
| **Sandeera Nawarathna** | **Model Evaluation, Validation & Diagnostics** (Phase 4)<br>• Holdout evaluation & diagnostic reporting<br>• Precision-Recall & ROC-AUC curve analysis<br>• Probability calibration (Brier score & ECE)<br>• Cumulative gains, lift curves & subgroup error analysis | [@Sandeera-2003](https://github.com/Sandeera-2003) |
| **Dilni Nishshanka** | **Model Development & Benchmarking** (Phase 3)<br>• Stratified 5-Fold Cross-Validation setup<br>• Multi-family model exploration & hyperparameter tuning<br>• Class imbalance mitigation & metric tracking<br>• Candidate bundle persistence & benchmark compilation | [@dilni2004](https://github.com/dilni2004) |
| **Sandamal Siribaddana** | **Problem Framing & Exploratory Data Analysis (EDA)** (Phase 1)<br>• Domain problem formulation & business translation<br>• Demographic & macroeconomic feature distribution analysis<br>• Target correlation & historical contact anomaly detection<br>• Critical identification of post-call duration target leakage | [@SandamalSiribaddana](https://github.com/SandamalSiribaddana) |

---

## What the Project is About

Direct marketing campaigns via telephone represent a significant operational expense for retail banking institutions. In the Portuguese Bank Marketing dataset ($N = 41,188$ campaign contacts across 2008–2010), only **11.27%** of prospects ultimately subscribe to long-term deposits, resulting in substantial wasted effort on non-responsive calls.

The core objective of this project is to build an operational propensity model that scores leads **prior to contact**, maximizing campaign yield while controlling call volume.

### Operational Reality & Leakage Safeguard
A fundamental design requirement is the **strict exclusion of the `duration` (call length) attribute**. While call duration is heavily correlated with conversion, its value is unknown before a call takes place and serves only as an unachievable post-contact benchmark. By enforcing the complete removal of `duration` across all preprocessing and evaluation stages, this system provides genuine pre-call predictive capability.

---

## How It Was Done

The workflow is architected into four modular, auditable phases:

```
[Phase 1: Problem Framing & EDA]
        │  • 41,188 records analyzed across 20 attributes
        │  • Target imbalance: 11.27% conversion rate
        │  • Documented macroeconomic driver influence (euribor3m, emp.var.rate)
        ▼
[Phase 2: Data Preprocessing & Pipeline Engineering]
        │  • Stratified 80/20 train-test partition (32,950 train / 8,236 holdout)
        │  • Custom BankFeatureEngineer (scikit-learn TransformerMixin)
        │  • Sentinel handling (pdays=999 binned into recency groups)
        │  • U-shaped non-linear age cohort engineering
        │  • ColumnTransformer with RobustScaler and OneHotEncoder
        ▼
[Phase 3: Model Development & Benchmarking]
        │  • Leakage-free Stratified 5-Fold Cross-Validation on training data
        │  • 7 candidate models spanning linear, bagging, boosting, and neural families
        │  • Primary optimization metric: PR-AUC (Average Precision)
        │  • Serialized candidate bundles saved to models/
        ▼
[Phase 4: Evaluation, Validation & Diagnostics]
        │  • Unbiased evaluation on 8,236 holdout test samples
        │  • Decision threshold calibration & trade-off analysis
        │  • Reliability diagrams & probability calibration assessment
        │  • Cumulative gains & business lift curves
```

### 1. Exploratory Data Analysis & Empirical Insights
* **Macroeconomic Regime Shifts**: Interest rate indicators (`euribor3m`) and employment variation rates (`emp.var.rate`) demonstrate high predictive leverage over conversion probabilities across market cycles.
* **Non-Linear Demographics**: Both younger individuals ($<25$ years) and retirees ($60+$ years) exhibit significantly higher conversion propensity compared to working-age cohorts ($30$–$50$ years), justifying specialized demographic binning.
* **CRM Inconsistencies**: Historical outreach records contain structural artifacts (e.g., sentinel values where `previous > 0` but `pdays == 999`), requiring dedicated transformation logic.

### 2. Feature Engineering & Preprocessing Pipeline
* Built a custom, reusable [`BankFeatureEngineer`](src/transformers.py) transformer integrated cleanly into Scikit-Learn pipelines.
* Formulated intentional dual representations for continuous and categorical age cohorts to satisfy both linear step functions and tree-based decision thresholds.
* Encoded categorical attributes while preventing silent out-of-vocabulary inference failures via robust one-hot encoding schemas.

### 3. Model Development & Stratified Cross-Validation
* Evaluated multiple model families across identical stratified folds:
  * **Linear Classifiers**: Logistic Regression (with class-weight balancing).
  * **Ensemble Methods**: Random Forest (tuned and balanced variants).
  * **Gradient Boosting**: Histogram-based Gradient Boosting (`HistGradientBoostingClassifier`).
  * **Neural Architectures**: Multi-Layer Perceptron (`MLPClassifier`).
* Evaluated models primarily on **PR-AUC (Average Precision)** rather than raw accuracy or ROC-AUC to account for severe class imbalance.

### 4. Diagnostics & Business Translation
* Calibrated decision thresholds against variable operational call capacities.
* Verified probability calibration using Brier score loss and Expected Calibration Error (ECE) to guarantee reliable risk scoring.
* Generated cumulative gains and lift curves to quantify direct marketing ROI.

---

## Project Outcomes & Benchmark Results

### Benchmark Summary (Stratified 5-Fold CV vs. Unseen Holdout Test Set)

| Rank | Model | Model Family | 5-Fold CV PR-AUC | Holdout Test PR-AUC | Holdout ROC-AUC | Balanced Accuracy | Holdout F1-Score | Holdout Brier Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **HistGradientBoosting** | Gradient Boosting | **0.4662** | **0.4906** | **0.8132** | 0.7616 | 0.4910 | 0.1505 |
| 🥈 | **Random Forest (Tuned)** | Bagging Ensemble | **0.4665** | **0.4891** | **0.8140** | 0.7633 | **0.5062** | 0.1421 |
| 🥉 | **HistGradientBoosting (Tuned)** | Gradient Boosting | **0.4680** | **0.4890** | **0.8125** | **0.7667** | 0.4972 | 0.1529 |
| 4 | **Random Forest (Balanced)** | Bagging Ensemble | 0.4663 | 0.4852 | 0.8136 | 0.7620 | 0.5023 | 0.1467 |
| 5 | **MLP Classifier** | Neural Network | 0.4461 | 0.4660 | 0.8035 | 0.5971 | 0.3154 | **0.0778** |
| 6 | **Logistic Regression (Balanced)** | Linear Model | 0.4454 | 0.4472 | 0.7998 | 0.7495 | 0.4613 | 0.1618 |
| — | **Dummy Baseline** | Baseline | 0.1127 | 0.1127 | 0.5000 | 0.5000 | 0.0000 | 0.1000 |

### Key Takeaways
1. **Exceptional Lift over Baseline**: The top-performing **HistGradientBoosting** achieved a holdout PR-AUC of **0.4906**, outperforming the random-guess baseline (**0.1127**) by more than **4.3×**.
2. **Robust Generalization**: The validation-to-holdout score delta across top models remained minimal ($\Delta < 0.025$), confirming zero overfitting and reliable out-of-sample stability.
3. **Actionable Business Impact**: Cumulative gains analysis indicates that targeting the **top 30%** highest-propensity prospects identified by the model captures **~70% of all subscribing customers**, reducing required telemarketing call volume by **70%** while retaining the vast majority of conversions.

---

## Repository Structure

```
├── data/
│   ├── raw/                 # Original Bank Marketing datasets
│   └── processed/           # Leakage-free train/test splits & feature metadata
├── models/
│   ├── candidate_models.joblib    # Serialized candidate model bundle
│   └── cv_benchmark_summary.csv   # Cross-validation performance logs
├── notebooks/
│   ├── 01_Problem_Framing_EDA.ipynb             # Phase 1: EDA & anomaly detection
│   ├── 02_Data_Preprocessing_Pipeline.ipynb     # Phase 2: Pipeline transformation
│   ├── 03_Model_Development.ipynb              # Phase 3: CV benchmarking
│   └── 04_Evaluation_and_Diagnostics.ipynb      # Phase 4: Holdout testing & lift
└── src/
    ├── __init__.py
    ├── transformers.py      # Custom Scikit-Learn BankFeatureEngineer
    ├── modeling.py          # Cross-validation & model benchmarking helpers
    └── evaluation.py        # Diagnostic metrics, calibration & curve utilities
```

---

## Original Development Repository

This exhibit repository is a curated, production-clean showcase of the collaborative project. The original development and experimental workspace containing raw commit histories and intermediate work logs is maintained at:

* **Original Repository**: [https://github.com/dilni2004/Bank-Marketing-Analysis](https://github.com/dilni2004/Bank-Marketing-Analysis.git) (Maintained by [@dilni2004](https://github.com/dilni2004 ), [@monishkabon ](https://github.com/monishkabon ), [@Sandeera-2003 ](https://github.com/Sandeera-2003 ), [@SandamalSiribaddana ](https://github.com/SandamalSiribaddana ))
