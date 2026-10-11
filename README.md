# ADS-505 Applied Data Science for Business — Final Project
## Corporate Bankruptcy Prediction & Financial Distress Early Warning System

**Authors:** Imogen Ronde, MaCoi Hollins, Jordan Blackman  
**Repository:** [https://github.com/jblackman-bit/ADS-505-Final-Project](https://github.com/jblackman-bit/ADS-505-Final-Project)  
**Shared Team Resources:** [Google Drive Project Folder](https://drive.google.com/drive/folders/1stEs9XUg6vsdBv-fbH_vnOC6YxlaRYhK)

---

### Project Overview & Social Mission
Corporate insolvency imposes severe negative externalities across the broader economy, jeopardizing the livelihoods of employees, families, and retirement/pension funds that depend on business continuity.

The primary objective of this project is to develop an interpretable, reproducible machine learning early-warning and ranking pipeline that detects corporate financial distress using balance sheet metrics, cash flow indicators, and operating ratios.

By identifying vulnerable operating companies *before* formal insolvency or liquidation occurs, **non-profit turnaround organizations, government economic development agencies, and specialized financial advisory practices** can proactively deploy consulting, debt-restructuring, and bridge-relief resources within their finite field outreach bandwidth (top 5% **Critical** and next 10% **Elevated** risk cohorts).

---

### Dataset Summary & Macroeconomic Context
- **Source:** Taiwan Economic Journal (1999–2009 corporate filings), available via the [UCI Machine Learning Repository](https://doi.org/10.24432/C5004D).
- **Macroeconomic Stress Window:** The 1999–2009 timeframe captures major global economic shocks—including the **Dot-com bubble contraction (2000–2001)** and the **Global Financial Crisis (2007–2008)**—providing realistic stress-testing conditions for corporate solvency modeling.
- **Dimensions:** 6,819 company observations across 96 variables (95 financial indicators + 1 binary target; 94 active predictors after dropping the constant `Net Income Flag`).
- **Target Variable (`Bankrupt?`):**
  - `0`: Solvent / Operating (6,599 companies, 96.77%)
  - `1`: Bankrupt / Distressed (220 companies, 3.23%)
- **Data Quality:** 0 missing values, 0 duplicate records.

---

### Repository Structure
```text
ADS-505-Final-Project/
│
├── raw_bankruptcy.csv                    # Raw TEJ Bankruptcy Dataset (6,819 x 96)
├── bankruptcy_clean.csv                  # Cleaned Dataset exported by eda.ipynb (6,819 x 95)
├── eda.ipynb                             # Part 1: Exploratory Data Analysis & Feature Auditing
├── modeling.ipynb                        # Part 2: Preprocessing, Model Tuning, Calibration & Risk Tiers
├── figures/                              # Saved publication-quality charts for Brief & Presentation
│   ├── cost_sensitivity.png              # Asymmetric cost curves across outreach thresholds
│   ├── decision_boundary.png             # 2D Profitability vs. Leverage distress boundary
│   ├── test_tiers.png                    # Holdout test bankruptcy capture & tier default rates
│   ├── top_signals.png                   # Top 12 features by permutation importance (PR-AUC drop)
│   └── validation_curves.png             # Validation Precision-Recall and Cumulative Capture curves
├── results/                              # Saved evaluation tables exported by modeling.ipynb
│   ├── model_comparison.csv              # 5-fold CV and Validation metrics across all 4 models
│   ├── permutation_importance.csv        # Test-set permutation importance across all features
│   ├── raw_scale_ablation.csv            # R&D Expense Rate & raw-scale indicator ablation results
│   ├── seed_stability.csv                # Multi-seed (5 random splits) stability evaluation
│   ├── test_calibration_by_tier.csv      # Platt-calibrated vs. observed bankruptcy rates by tier
│   ├── test_confusion.csv                # Confusion matrices for Critical-only & Critical + Elevated
│   ├── test_summary.csv                  # Final holdout test metrics (Gradient Boosting vs. Altman)
│   └── test_tiers.csv                    # Final holdout test cohort breakdown across 3 risk tiers
├── pdf-requirements/                     # Course assignment rubrics and deliverable guidelines that are gitignored
└── README.md                             # Project documentation, setup instructions, and team roles
```

---

### Key Analytical & Modeling Highlights

#### 1. Exploratory Data Analysis (`eda.ipynb`)
1. **Zero-Variance Pruning:** `Net Income Flag` is constant (`1` across all 6,819 rows) and is removed when exporting `bankruptcy_clean.csv`.
2. **Structural Zeros vs. Missing Data:** High-zero features such as `Long-term Liability to Current Assets` (37.67% zeros) and `Research and development expense rate` (20.88% zeros) represent authentic operational states ($0 long-term debt, $0 R&D) and are preserved without imputation.
3. **Dual-Scale Discovery (24 Raw-Scale Columns):** While 70 ratios are normalized to `[0, 1]`, 24 columns contain unnormalized values in the millions or billions (`> 1.0`) caused by raw currency units or near-zero denominators (e.g., `Cash/Current Liability`, where raw-scale firms default at ~10x the base rate).
4. **Skewness & Multicollinearity Framework:** 86.2% of features exhibit severe skewness ($|\text{Skew}| \ge 1.0$), and collinear clusters ($|r| > 0.85$) across ROA, Net Value Per Share, and Leverage are documented in a candidate pruning table (`LINEAR_DROP`) for linear models.

#### 2. Bankruptcy Risk Modeling (`modeling.ipynb`)
1. **Leakage-Free Raw-Scale Pipeline:** Values $> 1.0$ are masked to `NaN` and imputed inside each model's scikit-learn `Pipeline` via `SimpleImputer(strategy='median', add_indicator=True)`, replacing raw values with training-fold medians while adding **24 binary indicator flags** to preserve acute distress signals. An ablation study on `Research and development expense rate` confirmed this pipeline achieves the highest CV and validation PR-AUC.
2. **Model Benchmarking (60 / 20 / 20 Stratified Split):** We compared an unregularized **5-ratio Altman-style baseline** against three class-weighted (`class_weight='balanced'`) tuned models: **L2 Logistic Regression** (with `QuantileTransformer`), **Random Forest**, and **Histogram Gradient Boosting** (`HistGradientBoostingClassifier`).
3. **Primary Recommendation — Gradient Boosting:**
   - **PR-AUC Lift:** Achieves **0.56 Validation PR-AUC** (vs. `0.44` Altman) and **0.47 Holdout Test PR-AUC** (vs. `0.37` Altman)—a **+27% relative lift** and **~14.5x lift** over the 3.23% random baseline.
   - **Outreach Capture:** Captures **61.4% (27 of 44)** of test bankruptcies in the top 5% **Critical** tier (~12x lift) and **90.9% (40 of 44)** in the top 15% **Critical + Elevated** cumulative outreach list (~6.1x lift).
   - **Multi-Seed Stability:** Outperforms the Altman baseline across **5 out of 5** additional random train/validation/test splits (`0.500` vs. `0.384` mean PR-AUC). The 5-ratio Altman baseline is retained as a transparent, lightweight fallback and field diagnostic guide.
4. **Probability Calibration:** Wrapping the fitted model in `CalibratedClassifierCV(method='sigmoid')` on the validation set reduced the test Brier score from `0.0256` to `0.0228` and aligned predicted probabilities with observed test bankruptcy rates (**44.4% predicted vs. 39.7% observed** in Critical; **6.1% vs. 9.8%** in Elevated; **0.5% vs. 0.4%** in Stable).

---

### Setup & How to Reproduce Results

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jblackman-bit/ADS-505-Final-Project.git
   cd ADS-505-Final-Project
   ```
2. **Create the Conda / Python Environment (Python 3.13 recommended for exact bit-for-bit reproducibility):**
   ```bash
   conda create -n ads505 -y python=3.13.5 scikit-learn pandas numpy matplotlib seaborn scipy jupyter ipykernel jinja2
   conda activate ads505
   ```
   *(Alternatively, with `pip`: `pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter jinja2`)*
3. **Run the Notebooks in Order:**
   - **Step 1 (`eda.ipynb`):** Reads `raw_bankruptcy.csv`, runs all exploratory analyses, exports `bankruptcy_clean.csv`, and saves `figures/decision_boundary.png` (~1 minute).
   - **Step 2 (`modeling.ipynb`):** Reads `bankruptcy_clean.csv`, runs 5-fold cross-validation tuning, ablation checks, multi-seed stability, permutation importance, and Platt calibration, and exports all tables to `results/` and charts to `figures/` (~10–15 minutes).

---

### Team Member Contributions
- **Imogen Ronde:** Co-developed the modeling pipeline (`modeling.ipynb`), refactored preprocessing into standard scikit-learn `Pipeline` / `SimpleImputer(add_indicator=True)` steps, implemented the unregularized Altman baseline, `R&D Expense Rate` raw-scale ablation study, dual-policy confusion matrices, and `CalibratedClassifierCV` probability calibration.
- **MaCoi Hollins:** Co-developed the initial modeling workflow and baseline comparisons (`modeling.ipynb`), contributed to risk-tier thresholding, cost sensitivity analysis, permutation importance evaluation, and executive deliverable drafting.
- **Jordan Blackman:** Led the Exploratory Data Analysis (`eda.ipynb`), data hygiene and dual-scale feature auditing, collinearity pruning framework (`LINEAR_DROP`), repository integration, and alignment across the technical notebooks, Executive Brief, and Executive Presentation.

---

### References & Academic Bibliography (APA 7th Edition)
1. **Altman, E. I. (1968).** Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, *23*(4), 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x
2. **Ohlson, J. A. (1980).** Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, *18*(1), 109–131. https://doi.org/10.2307/2490395
3. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
4. **Yeh, C. (2020).** *Taiwanese Bankruptcy Prediction* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5004D
