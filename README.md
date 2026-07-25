# Credit Card Fraud Detection — Imbalanced Classification & Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006ACC)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](#live-demo)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end fraud detection case study built around the real technical challenge of this problem — extreme class imbalance — using five different modeling approaches, from class-weighted logistic regression to unsupervised autoencoder anomaly detection, plus a live interactive dashboard.

## Live Demo

🔗 **[Try the live dashboard](https://fraud-detection-ywooiuryujcxw7q2ss5hsy.streamlit.app/)** 

## Overview

Payment platforms and card issuers need to flag fraud in real time. The core difficulty isn't modeling accuracy in the usual sense — it's that fraud is **extremely rare**, so a model that predicts "not fraud" every time scores 99%+ accuracy while catching zero fraud. This project is built around techniques for genuinely imbalanced classification, not standard accuracy optimization.

The notebook answers six business questions:

1. How rare is fraud in this data, and why does that break naive accuracy metrics?
2. Does oversampling (SMOTE) or class-weighting handle the imbalance better?
3. Which supervised model catches the most fraud without drowning investigators in false alarms?
4. Can an unsupervised autoencoder catch fraud *without ever seeing a labeled fraud example*?
5. What decision threshold should the business actually use, given asymmetric costs?
6. What would this model realistically save, in dollar terms?

## Repository Structure

```
fraud-detection-imbalanced-classification/
├── app.py                              # Live Streamlit dashboard
├── pipeline.py                         # Standalone script (metrics/figures)
├── Fraud_Detection_Analysis.ipynb      # Full analysis notebook (15 modules, executed)
├── requirements.txt
├── runtime.txt                         # Pinned Python version for clean cloud deploys
├── data/
│   └── creditcard.csv                  # 50,000-row sample (all 492 real fraud cases included)
├── outputs/                            # Saved figures & metrics.json
├── LICENSE
└── README.md
```

## Dataset

Public anonymized credit card transactions — European cardholders, September 2013. Features `V1`-`V28` are PCA-transformed (the original transaction details are confidential — a common real-world constraint in financial data). `Amount` and `Time` are the only non-anonymized columns; `Class` is the target (1 = fraud).

> **Note on sampling:** the original dataset (284,807 transactions, 492 fraud, ~0.17% fraud rate) is ~150MB — over GitHub's 25MB upload limit. This repo ships a **50,000-row sample containing every one of the 492 real fraud cases**, plus a random sample of legitimate transactions (resulting fraud rate: 0.98% — more concentrated than the true population rate, purely a sampling artifact for file size, not a data quality choice). The full dataset is public — see [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — and the same code runs unmodified against it.

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction |
| `V1`-`V28` | PCA-transformed, anonymized features |
| `Amount` | Transaction amount (€) |
| `Class` | 1 = fraud, 0 = legitimate |

## Notebook Structure (15 Modules)

| Module | Contents |
|---|---|
| 1 | Business Understanding |
| 2 | Data Understanding |
| 3 | Data Quality Assessment |
| 4 | The Core Challenge — Class Imbalance |
| 5 | Exploratory Data Analysis |
| 6 | Feature Engineering |
| 7 | Train/Test Split — stratified |
| 8 | Baseline — Logistic Regression (class-weighted) |
| 8b | Logistic Regression + SMOTE |
| 9 | Random Forest (class-weighted) |
| 9b | XGBoost (`scale_pos_weight` tuned) |
| 10 | Autoencoder Anomaly Detection — unsupervised |
| 11 | Model Evaluation — Precision, Recall, F1, ROC-AUC, PR-AUC |
| 12 | Business-Driven Threshold Selection |
| 13 | Feature Importance |
| 14 | Business Impact Summary |
| 15 | Business Recommendations & Limitations |

## Key Results

**Model comparison (30% stratified holdout)**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (class-weighted) | 29.4% | 91.9% | 0.445 | 0.986 | 0.859 |
| Logistic Regression + SMOTE | 27.0% | 91.2% | 0.417 | 0.985 | 0.850 |
| **Random Forest** | **95.9%** | 79.1% | **0.867** | 0.982 | **0.885** |
| XGBoost | 96.8% | 81.8% | 0.886 | 0.985 | 0.884 |
| Autoencoder (unsupervised) | 43.6% | 81.1% | 0.567 | — | — |

> **PR-AUC (Precision-Recall AUC), not accuracy, is the primary metric** — with this level of imbalance, a model predicting "no fraud" for everything scores ~99% accuracy while catching nothing. Random Forest wins on PR-AUC; class-weighted logistic regression alone catches more fraud (92% recall) but at the cost of far more false alarms (only 29% precision) — a real precision/recall tradeoff, not a clear-cut winner, which is exactly why the notebook builds an explicit business-driven threshold rather than defaulting to 0.5.

**Business-tuned operating point:** at a threshold chosen to keep precision ≥ 85%, Random Forest catches **85.1% of fraud at 85.1% precision** — meaning roughly 1 in 7 flagged transactions is a false alarm, and the vast majority of real fraud gets caught.

**Estimated business impact:** on the test set alone, this operating point corresponds to an estimated **€15,398.63** in fraud caught.

**Top fraud indicators:** `V14`, `V10`, `V17`, and `V4` (anonymized PCA components) are the strongest predictors — consistent with published research on this dataset.

**The autoencoder's role:** trained *only* on legitimate transactions, with no fraud labels at all, it still catches 81% of fraud by flagging unusual transaction patterns — useful as a complementary signal for fraud types the supervised models haven't seen before.


## How to Run

### Notebook
```bash
git clone https://github.com/NurMithu/fraud-detection-imbalanced-classification.git
cd fraud-detection-imbalanced-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Fraud_Detection_Analysis.ipynb
```

### Live Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy for Free
1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**.
3. Select this repo, branch `main`, file path `app.py`.
4. Click **Deploy** — you'll get a public link in a few minutes.
5. Add that link to the "Live Demo" section above.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` (SMOTE) · `matplotlib` · `seaborn` · `plotly` · `streamlit` · `Jupyter`

## Business Recommendations (Summary)

- **Deploy the best-performing supervised model at a business-chosen threshold**, not the default 0.5 — the threshold should reflect the real cost ratio between missed fraud and false alarms, which is a business decision, not a modeling one.
- **Run the autoencoder as a parallel, complementary signal** — it catches anomalies without needing labeled fraud examples, useful as fraud patterns evolve.
- **Monitor PR-AUC, not accuracy**, once deployed — accuracy stays high even if the model catches almost no fraud.
- **Revisit the threshold periodically** as fraud patterns and business cost tolerances shift.

Full detail in Module 15 of the notebook.

## Limitations & Future Work

- This build uses a 50,000-row sample for file-size reasons (all 492 real fraud cases included); the full 284,807-row dataset (true ~0.17% fraud rate) is publicly available for a full-scale run with the same code.
- No hyperparameter search was run — parameters used are reasonable defaults, not tuned.
- The autoencoder here is a lightweight MLP-based bottleneck network for dependency-footprint reasons; a deeper autoencoder (TensorFlow/Keras) is a natural upgrade for production.
- A live system needs a feedback loop — confirmed outcomes should periodically retrain the model, since fraud patterns evolve.
- `V1`-`V28` being anonymized limits feature-level business storytelling; a production system on raw features would allow more actionable insight.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built as an end-to-end analytics case study — from a genuinely hard, realistic imbalanced-classification problem to a validated, business-ready fraud detection system with a live interactive demo.*
