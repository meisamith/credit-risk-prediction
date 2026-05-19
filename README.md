# Credit Default Risk Prediction System

A machine-learning system that predicts the probability a loan applicant
will **default**, trained on 1.3M real peer-to-peer loans and evaluated
with the metrics credit-risk teams at banks and fintechs actually use.

**Live demo:** _(Streamlit Cloud URL — added in deployment step)_
**Author:** Amith Choudhary · CSBS, JSS Science & Technology University, Mysuru

---

## 1. Problem

Lenders lose money two ways: by approving borrowers who default, and by
rejecting borrowers who would have repaid. A credit-risk model estimates
**P(default)** for each applicant so the lender can price risk and make
defensible approve/reject decisions. This project builds, compares, and
honestly evaluates such a model end-to-end.

## 2. Data

**Lending Club Loan Data** (2007–2018Q4) — real, anonymized peer-to-peer
loans with final repayment outcomes. Sourced from Kaggle
(`wordsforthewise/lending-club`), accessed via the Kaggle API for
reproducible ingestion.

- Raw file: ~2.26M loan records, 151 columns
- After filtering to **resolved** loans only: **1,348,099 loans**
- Target definition: `default = 1` for Charged Off / Default, `0` for
  Fully Paid
- **Class balance: 20% default rate** (real, meaningful imbalance)

"Current" (still-active) loans were excluded — their outcome is unknown,
so including them would leak future information into training.

## 3. Methodology

### 3.1 Leakage handling (the part that matters most)

The single biggest credit-modeling mistake is training on columns that
only exist *after* a loan plays out. Fields like `total_pymnt`,
`recoveries`, and `out_prncp` are unknown at application time. A model
that uses them scores a near-perfect AUC in the notebook and is
completely useless in production. These post-origination columns were
explicitly dropped before any modeling.

### 3.2 Cleaning & feature engineering

- Parsed `term` ("36 months" → 36) and `emp_length` to numeric
- Dropped 57 columns with >40% missing values (mostly empty
  `hardship_*` and secondary-applicant fields)
- Median imputation for numeric features, mode for categoricals
- Selected 22 credit-domain features: loan amount, grade, FICO range,
  DTI, interest rate, employment length, home ownership, purpose, etc.

### 3.3 Models compared

| Model | Role | ROC-AUC |
|---|---|---|
| Logistic Regression | Interpretable baseline | 0.708 |
| Random Forest | Non-linear ensemble | 0.703 |
| **XGBoost** | **Best performer** | **0.720** |

All models used class-imbalance handling (`class_weight='balanced'` /
`scale_pos_weight`). Data was split 80/20 with stratification so the
20% default rate is preserved in both train and test sets.

## 4. Results

**Best model: XGBoost, ROC-AUC 0.720** on the held-out test set
(269,620 loans the model never saw during training).

At the default decision threshold, on the defaulting class:

- **Recall ≈ 0.67** — the model catches roughly two-thirds of actual
  defaulters
- **Precision ≈ 0.32** — among loans flagged risky, about a third truly
  default

This precision/recall trade-off is deliberate. For a lender, a missed
default (false negative) is far more expensive than a false alarm, so
the operating point favors recall. The threshold is a *business* choice,
not a modeling one — which is exactly how it works at a real risk desk.

Saved artifacts: `models/credit_risk_model.joblib` (full preprocessing
+ XGBoost pipeline). Figures in `reports/figures/`:
`eda_overview.png`, `confusion_matrix.png`, `roc_curves.png`.

### Key EDA findings

- **Loan grade is the strongest single predictor** — default rate rises
  monotonically from ~7% (Grade A) to ~50% (Grade G)
- Lower FICO scores cluster strongly in the defaulting population
- Higher interest rates correlate with default (risk was priced in, but
  still realized)

## 5. Limitations

- **AUC ceiling ~0.72** — defaults have a large irreducible random
  component (job loss, medical events) that no application-time feature
  captures. This is honest and expected for credit risk; a model
  claiming 0.95 here is almost certainly leaking.
- **Temporal validity** — the split is random, not time-based. A
  production model should be back-tested on later vintages to check for
  drift.
- **No macroeconomic features** — unemployment and rate cycles strongly
  affect default and are not modeled here.
- **Imputation is simple** — median/mode is defensible for a first
  iteration but a production system would model missingness explicitly.

## 6. Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn ·
matplotlib/seaborn · Jupyter · Streamlit · joblib · Git/GitHub

## Project structure

```
credit-risk/
├── data/raw/         # Lending Club CSV (gitignored — never committed)
├── notebooks/        # 01_eda.ipynb — full analysis narrative
├── src/              # config, data-loading & cleaning logic
├── models/           # saved .joblib pipeline (gitignored)
├── app/app.py        # Streamlit demo app
├── reports/figures/  # EDA, ROC, confusion-matrix charts
├── requirements.txt  # pinned dependencies (reproducible)
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

## Getting the dataset

```bash
pip install kaggle
# place your Kaggle API token in ~/.kaggle/
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
gunzip -c data/raw/accepted_2007_to_2018Q4.csv.gz > data/raw/loans.csv
```

## License

Educational portfolio project. Dataset subject to its original license.
