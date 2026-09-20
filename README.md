<<<<<<< HEAD
# ADS-505-Final-Project

Google Drive: https://drive.google.com/drive/folders/1stEs9XUg6vsdBv-fbH_vnOC6YxlaRYhK
=======
# ADS-505 Applied Data Science for Business - Final Project
## Corporate Bankruptcy Prediction & Financial Distress Early Warning System

### Project Overview & Social Mission
Corporate insolvency imposes severe negative externalities across the broader economy, jeopardizing the livelihoods of employees, families, and retirement/pension funds that depend on business continuity. 

The primary objective of this project is to develop an interpretable, robust machine learning classification pipeline that detects early signals of corporate financial distress using balance sheet metrics, cash flow indicators, and operating ratios.

By identifying vulnerable operating companies *before* formal insolvency or liquidation occurs, **non-profit turnaround organizations, government economic development agencies, and specialized financial advisory practices** can proactively deploy consulting, restructuring, and bridge-relief resources where they are needed most.

- **Shared Team Resources**: [Google Drive Project Folder](https://drive.google.com/drive/folders/1stEs9XUg6vsdBv-fbH_vnOC6YxlaRYhK)

---

### Dataset Summary & Macroeconomic Context
- **Source**: Taiwan Economic Journal (1999–2009 corporate filings), available via the [UCI Machine Learning Repository](https://doi.org/10.24432/C5004D).
- **Macroeconomic Stress Window**: The 1999–2009 timeframe captures major global economic shocks—including the **Dot-com bubble contraction (2000–2001)** and the **Global Financial Crisis (2007–2008)**—providing realistic stress-testing conditions for corporate solvency modeling.
- **Dimensions**: 6,819 company observations across 96 variables (95 continuous financial indicators + 1 binary target).
- **Target Variable**: `Bankrupt?`
  - `0`: Solvent / Operating (6,599 companies, 96.77%)
  - `1`: Bankrupt / Distressed (220 companies, 3.23%)
- **Data Quality**: 0 missing values, 0 duplicate records.

---

### Repository Structure
```
ADS-505-Final-Project/
│
├── raw_bankruptcy.csv          # Raw TEJ Bankruptcy Dataset (6,819 x 96)
├── bankruptcy_clean.csv        # Cleaned Dataset (Whitespace stripped, Net Income Flag dropped, 6,819 x 95)
├── eda.ipynb                   # Comprehensive EDA Notebook (Hygiene, Sparsity, Outliers, Skewness, Bivariate 2D, VIF, Altman, Dictionary)
├── README.md                   # Project documentation, setup instructions, and findings
└── pdf-requirements/           # Course assignment rubrics and technical deliverable guidelines
```

---

### Key EDA Insights & Analytical Sections
1. **Zero-Variance Pruning**: Feature `Net Income Flag` has a constant value of `1` across all observations ($Variance = 0$) and is dropped.
2. **Structural Zeros vs. Missing Data**:
   - Zero values in ratios such as `Long-term Liability to Current Assets` (37.67% zeros), `Tax rate (A)` (37.66% zeros), and `Research and development expense rate` (20.88% zeros) represent **genuine operational realities** ($0 debt, $0 tax, $0 R&D) rather than missing data. They are preserved without imputation.
3. **Severe Target Imbalance (~30:1)**: Only 3.23% of companies experienced bankruptcy. Standard accuracy is deceptive (a naive model predicting all solvent achieves 96.77% accuracy while offering zero utility). Modeling requires **Stratified K-Fold**, **PR-AUC / Recall / F1** optimization, and cost-sensitive loss functions.
4. **Outliers as Authentic Tail Signals (Tukey IQR Analysis)**: Up to 22% of records exceed Tukey IQR fences on leverage and turnover ratios. Because these extreme values represent authentic distressed firms on the verge of collapse, **outliers are preserved without filtering or truncation**, with `RobustScaler` / tree-based models recommended.
5. **Feature Skewness Audit**: 86.2% of features (81/94) exhibit severe skewness ($|\text{Skew}| \ge 1.0$), demonstrating the necessity of `RobustScaler` or tree-based models for non-linear robustness.
6. **2D Bivariate Decision Boundary**: Interactive visualization mapping Profitability (`Net Income to Total Assets`) vs. Leverage (`Debt ratio %`), revealing a concentrated cluster of bankruptcy events in the low-margin, high-debt quadrant.
7. **Multicollinearity & Candidate Pruning Framework**:
   - Severe pairwise collinearity ($|r| > 0.85$) mapped across ROA metrics, Share value metrics, and Debt vs. Net Worth.
   - Candidate pruning recommendations are documented for linear models while preserving the full feature space for tree ensembles.
8. **Composite Financial Health Index (Altman Z-Score Proxy)**: Theoretical framework established for engineering a domain-weighted solvency barometer in the modeling stage.
9. **Comprehensive 96-Variable Data Dictionary (Section 12)**: Full formulas, interpretations, and distress directionalities.

---

### How to Run the EDA Notebook
1. **Prerequisites**: Ensure Python 3.9+ and required packages are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy jupyter
   ```
2. **Launch Jupyter**:
   ```bash
   jupyter notebook eda.ipynb
   ```
3. **Run All Cells**: Execute the notebook top-to-bottom (`Kernel -> Restart & Run All`).

---

### References & Academic Bibliography (APA 7th Edition)
1. **Altman, E. I. (1968)**. Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, *23*(4), 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x
2. **Yeh, C. (2020)**. *Taiwanese Bankruptcy Prediction* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5004D
3. **Ohlson, J. A. (1980)**. Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, *18*(1), 109–131. https://doi.org/10.2307/2490395
>>>>>>> 17a7472 (Add EDA notebook, clean dataset, and updated documentation)
