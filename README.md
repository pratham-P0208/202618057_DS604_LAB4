# Medical Insurance Cost — Statistical Modeling & Interactive Dashboard

M.Sc. Data Science — Semester 1 — Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

## 📁 Project Structure
Everything — EDA, hypothesis testing, OLS regression, diagnostics, AND live prediction — runs
from a **single file, `202618057_LAB4.py`**, as one website. No separate scripts to run.
```
.
├── 202618057_LAB4.py              # Single Streamlit app: EDA + Hypothesis Testing + Regression + Live Prediction & Diagnostics
├── requirements.txt     # Python dependencies
├── insurance.csv    # Dataset
└── README.md
```

## 📊 Dataset Summary
**Medical Insurance Costs** (1,338 records, 7 columns):

| Column     | Type        | Description                              |
|------------|-------------|-------------------------------------------|
| `age`      | numeric     | Age of primary beneficiary                |
| `sex`      | categorical | male / female                             |
| `bmi`      | numeric     | Body Mass Index                           |
| `children` | numeric     | Number of dependents                      |
| `smoker`   | categorical | yes / no                                  |
| `region`   | categorical | northeast / northwest / southeast / southwest |
| `charges`  | numeric     | Individual medical costs billed (target)  |

## ⚙️ How to Run

### Local (VS Code / IDE)
```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app (everything - EDA, tests, regression, prediction - lives here)
streamlit run app.py
# Opens at http://localhost:8501
```

## 🔬 Statistical Findings (Summary)

**Hypothesis Test 1 — Smokers vs. Non-Smokers (Charges)**
- Shapiro-Wilk test showed both groups deviate from normality → **Mann-Whitney U test** used.
- Result: **p ≈ 5.3e-130 → REJECT H0**. Smokers have significantly higher medical charges
  (mean ≈ $32,050 vs ≈ $8,434 for non-smokers).

**Hypothesis Test 2 — One-Way ANOVA (Charges across Regions)**
- Result: **F ≈ 2.97, p ≈ 0.031 → REJECT H0** at α = 0.05. Mean charges differ modestly across
  the four regions (southeast is highest).

**Bonus — Chi-Square (Smoker status vs Region)**
- Result: **χ² ≈ 7.34, p ≈ 0.062 → FAIL TO REJECT H0**. Smoking prevalence is not significantly
  associated with region.

**OLS Regression Model**
- `charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)`
- **R² = 0.751, Adjusted R² = 0.749**
- Strongest predictor: **smoker status** (+$23,850 avg. charges, p < 0.001), followed by `age`
  and `bmi` (both p < 0.001).
- `sex` and most region dummies are not statistically significant at α = 0.05.

**Gauss-Markov Diagnostics**
- Residuals vs Fitted: shows mild funnel/heteroscedasticity pattern (higher variance at higher fitted values).
- Normality: **Jarque-Bera p < 0.001 → residuals are not normally distributed** (right-skewed,
  common for cost/insurance data). A log-transform of `charges` is recommended to improve fit.
- Multicollinearity: **All VIF values ≈ 1.0** for continuous predictors → no multicollinearity concern.

## 🖥️ Dashboard Tabs

1. **Data Exploration** — sidebar-style filters (age/BMI sliders, region/smoker multi-select),
   reactive histograms, count plots, scatter plots with trendlines, and a correlation heatmap.
2. **Hypothesis Testing Lab** — pick a test type (two-group comparison, ANOVA, or Chi-Square),
   select variables via dropdowns, and get live test statistics, p-values, and a plain-language
   Reject/Fail-to-Reject conclusion at α = 0.05.
3. **Live Prediction & Diagnostics** — enter age, BMI, children, sex, smoker status, and region
   to get a real-time predicted charge with 95% confidence and prediction intervals, plus
   residual diagnostic plots (Residuals vs Fitted, Q-Q plot, VIF table).

URL:- [https://202618057ds604lab4-ps6km7aujuwhh9jcsm2ifu.streamlit.app/]