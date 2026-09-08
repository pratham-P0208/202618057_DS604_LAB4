import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
import streamlit as st

# =============================================================================
# PAGE CONFIG & GLOBAL STYLE
# =============================================================================
st.set_page_config(
    page_title="Medical Insurance Cost Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALPHA = 0.05
NUMERIC_COLS = ["age", "bmi", "children", "charges"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    h1, h2, h3 {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(x):
    """Format a number as currency with exactly 2 decimal places."""
    return f"${x:,.2f}"


# =============================================================================
# DATA & MODEL (cached — computed once)
# =============================================================================
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


@st.cache_resource
def fit_model(data):
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)"
    return smf.ols(formula=formula, data=data).fit()


df_full = load_data()
model = fit_model(df_full)

# =============================================================================
# HEADER
# =============================================================================
st.title("🏥 Medical Insurance Cost — Statistical Dashboard")
st.caption("M.Sc. Data Science · Lab-4 · Statistical Modeling with Python")
st.divider()

# =============================================================================
# SIDEBAR — GLOBAL FILTERS (used by Tab 1: Data Exploration)
# =============================================================================
with st.sidebar:
    st.header("🔎 Filters")
    st.caption("These filters apply to the **Data Exploration** tab.")

    age_range = st.slider(
        "Age range",
        int(df_full.age.min()), int(df_full.age.max()),
        (int(df_full.age.min()), int(df_full.age.max())),
    )
    bmi_range = st.slider(
        "BMI range",
        float(df_full.bmi.min()), float(df_full.bmi.max()),
        (float(df_full.bmi.min()), float(df_full.bmi.max())),
    )
    regions_sel = st.multiselect(
        "Region", options=sorted(df_full.region.unique()),
        default=sorted(df_full.region.unique()),
    )
    smoker_sel = st.multiselect(
        "Smoker status", options=sorted(df_full.smoker.unique()),
        default=sorted(df_full.smoker.unique()),
    )

    df = df_full[
        df_full.age.between(*age_range)
        & df_full.bmi.between(*bmi_range)
        & df_full.region.isin(regions_sel)
        & df_full.smoker.isin(smoker_sel)
    ]

    st.divider()
    st.metric("Records matching filters", f"{len(df)} / {len(df_full)}")

    st.divider()
    st.caption(
        "**About this dataset**  \n"
        "1,338 records · 7 fields  \n"
        "age, sex, bmi, children, smoker, region, charges"
    )

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(
    ["📊  Data Exploration", "🔬  Hypothesis Testing", "🎯  Prediction & Diagnostics"]
)

# =============================================================================
# TAB 1 — DATA EXPLORATION
# =============================================================================
with tab1:
    st.subheader("Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average charges", money(df["charges"].mean()))
    k2.metric("Average age", f"{df['age'].mean():.2f}")
    k3.metric("Average BMI", f"{df['bmi'].mean():.2f}")
    k4.metric("Smoker share", f"{(df['smoker'].eq('yes').mean() * 100):.2f}%")

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Distribution")
        metric = st.selectbox("Numeric feature", NUMERIC_COLS, index=3)
        fig_hist = px.histogram(
            df, x=metric, color="smoker", marginal="box", nbins=40,
            opacity=0.75, color_discrete_sequence=["#2E86AB", "#E63946"],
        )
        fig_hist.update_layout(margin=dict(t=20, b=10), legend_title_text="Smoker")
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        st.subheader("Category Counts")
        cat_feature = st.selectbox("Categorical feature", ["sex", "smoker", "region", "children"])
        fig_count = px.histogram(
            df, x=cat_feature, color=cat_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_count.update_layout(margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig_count, use_container_width=True)

    st.divider()

    left2, right2 = st.columns([1, 1], gap="large")

    with left2:
        st.subheader("Relationship Between Variables")
        sc1, sc2, sc3 = st.columns(3)
        x_var = sc1.selectbox("X-axis", NUMERIC_COLS, index=0, key="scatter_x")
        y_var = sc2.selectbox("Y-axis", NUMERIC_COLS, index=3, key="scatter_y")
        color_var = sc3.selectbox("Color", CATEGORICAL_COLS, key="scatter_color")
        fig_scatter = px.scatter(
            df, x=x_var, y=y_var, color=color_var, trendline="ols", opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig_scatter.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right2:
        st.subheader("Correlation Matrix")
        corr = df[NUMERIC_COLS].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_corr.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    with st.expander("📋  Summary statistics table"):
        desc = df[NUMERIC_COLS].describe().T
        desc["IQR"] = df[NUMERIC_COLS].quantile(0.75) - df[NUMERIC_COLS].quantile(0.25)
        desc["skewness"] = df[NUMERIC_COLS].skew()
        desc["kurtosis"] = df[NUMERIC_COLS].kurt()
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    with st.expander("🗂️  Preview filtered raw data"):
        st.dataframe(df, use_container_width=True)

# =============================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =============================================================================
with tab2:
    st.subheader("Choose a Test")
    st.caption(
        "Select a test type and variables. The result — test statistic, p-value, and a plain-language "
        "conclusion at α = 0.05 — updates automatically."
    )

    test_type = st.selectbox(
        "Test type",
        [
            "Two-Group Comparison (t-test / Mann-Whitney U)",
            "One-Way ANOVA (3+ groups)",
            "Chi-Square Test (categorical association)",
        ],
    )

    st.divider()

    # ---------------- Two-group comparison ----------------
    if test_type.startswith("Two-Group"):
        c1, c2 = st.columns(2)
        group_col = c1.selectbox("Grouping variable", CATEGORICAL_COLS, index=1)
        metric_col = c2.selectbox("Numeric metric", NUMERIC_COLS, index=3, key="t2metric")

        levels = df_full[group_col].unique().tolist()
        if len(levels) != 2:
            c3, c4 = st.columns(2)
            g1 = c3.selectbox("Group A", levels, index=0)
            g2 = c4.selectbox("Group B", [lv for lv in levels if lv != g1], index=0)
        else:
            g1, g2 = levels[0], levels[1]

        st.info(f"**H0:** No difference in mean {metric_col} between *{g1}* and *{g2}*.  \n"
                f"**H1:** There is a significant difference.")

        data1 = df_full.loc[df_full[group_col] == g1, metric_col]
        data2 = df_full.loc[df_full[group_col] == g2, metric_col]

        sh1 = stats.shapiro(data1.sample(min(len(data1), 500), random_state=42))
        sh2 = stats.shapiro(data2.sample(min(len(data2), 500), random_state=42))
        lev_stat, lev_p = stats.levene(data1, data2)

        st.markdown("**Assumption checks**")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(f"Shapiro-Wilk p ({g1})", f"{sh1.pvalue:.4f}")
        rc2.metric(f"Shapiro-Wilk p ({g2})", f"{sh2.pvalue:.4f}")
        rc3.metric("Levene's p (equal variance)", f"{lev_p:.4f}")

        is_normal = (sh1.pvalue > ALPHA) and (sh2.pvalue > ALPHA)
        equal_var = lev_p > ALPHA

        if is_normal:
            stat_val, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
            test_name = "Two-Sample t-test"
        else:
            stat_val, p_val = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        st.markdown(f"**Test applied:** {test_name}")
        m1, m2 = st.columns(2)
        m1.metric("Test statistic", f"{stat_val:.4f}")
        m2.metric("p-value", f"{p_val:.4g}")

        if p_val < ALPHA:
            st.success(f"**Conclusion (alpha = 0.05): REJECT H0** — the difference in {metric_col} between "
                       f"{g1} and {g2} is statistically significant.")
        else:
            st.warning(f"**Conclusion (alpha = 0.05): FAIL TO REJECT H0** — not enough evidence of a difference "
                       f"in {metric_col} between {g1} and {g2}.")

        fig = px.box(
            df_full[df_full[group_col].isin([g1, g2])], x=group_col, y=metric_col,
            color=group_col, color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- ANOVA ----------------
    elif test_type.startswith("One-Way ANOVA"):
        c1, c2 = st.columns(2)
        group_col = c1.selectbox("Grouping variable (3+ categories)", ["region"], index=0)
        metric_col = c2.selectbox("Numeric metric", NUMERIC_COLS, index=3, key="anovametric")

        st.info(f"**H0:** Mean {metric_col} is equal across all {group_col} groups.  \n"
                f"**H1:** At least one group differs.")

        groups = [g[metric_col].values for _, g in df_full.groupby(group_col)]
        f_stat, p_val = stats.f_oneway(*groups)

        m1, m2 = st.columns(2)
        m1.metric("F-statistic", f"{f_stat:.4f}")
        m2.metric("p-value", f"{p_val:.4g}")

        if p_val < ALPHA:
            st.success(f"**Conclusion (alpha = 0.05): REJECT H0** — at least one {group_col} group has a "
                       f"significantly different mean {metric_col}.")
        else:
            st.warning(f"**Conclusion (alpha = 0.05): FAIL TO REJECT H0** — no significant difference in "
                       f"{metric_col} across {group_col} groups.")

        fig = px.box(
            df_full, x=group_col, y=metric_col, color=group_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Chi-Square ----------------
    else:
        c1, c2 = st.columns(2)
        cat_a = c1.selectbox("Categorical variable A", CATEGORICAL_COLS, index=1)
        cat_b = c2.selectbox("Categorical variable B", [c for c in CATEGORICAL_COLS if c != cat_a], index=0)

        st.info(f"**H0:** {cat_a} and {cat_b} are independent (not linked).  \n"
                f"**H1:** {cat_a} and {cat_b} are associated (linked).")

        contingency = pd.crosstab(df_full[cat_a], df_full[cat_b])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        st.markdown("**Contingency table**")
        st.dataframe(contingency, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Chi-square statistic", f"{chi2:.4f}")
        m2.metric("Degrees of freedom", dof)
        m3.metric("p-value", f"{p_val:.4g}")

        if p_val < ALPHA:
            st.success(f"**Conclusion (alpha = 0.05): REJECT H0** — {cat_a} and {cat_b} appear to be linked.")
        else:
            st.warning(f"**Conclusion (alpha = 0.05): FAIL TO REJECT H0** — no significant association between "
                       f"{cat_a} and {cat_b}.")

        fig = px.bar(
            contingency.reset_index().melt(id_vars=cat_a), x=cat_a, y="value", color=cat_b,
            barmode="group", color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(margin=dict(t=20, b=10), yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3 — PREDICTION & DIAGNOSTICS
# =============================================================================
with tab3:
    st.subheader("Regression Model")
    st.code("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)", language="text")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("R-squared", f"{model.rsquared:.4f}")
    mcol2.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
    mcol3.metric("Observations", int(model.nobs))

    with st.expander("Coefficient table (estimates, p-values, 95% CI)"):
        conf = model.conf_int()
        coef_df = pd.DataFrame({
            "coefficient": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
            "CI_lower_95": conf[0],
            "CI_upper_95": conf[1],
        })
        st.dataframe(coef_df.style.format("{:.4f}"), use_container_width=True)

    with st.expander("Full statsmodels OLS summary (raw text)"):
        st.text(str(model.summary()))

    st.divider()

    # ------------------ Prediction ------------------
    st.subheader("🎯 Predict Medical Charges")
    st.caption("Enter a profile below to get a real-time predicted cost with 95% intervals.")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=27.50, step=0.10, format="%.2f")
    with pc2:
        in_children = st.slider("Number of children", 0, 5, 0)
        in_sex = st.radio("Sex", ["male", "female"], horizontal=True)
    with pc3:
        in_smoker = st.radio("Smoker", ["yes", "no"], horizontal=True)
        in_region = st.selectbox("Region", sorted(df_full.region.unique()))

    input_df = pd.DataFrame({
        "age": [in_age], "bmi": [in_bmi], "children": [in_children],
        "sex": [in_sex], "smoker": [in_smoker], "region": [in_region],
    })

    pred = model.get_prediction(input_df)
    pred_summary = pred.summary_frame(alpha=0.05)

    point_pred = pred_summary["mean"].iloc[0]
    ci_low, ci_high = pred_summary["mean_ci_lower"].iloc[0], pred_summary["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred_summary["obs_ci_lower"].iloc[0], pred_summary["obs_ci_upper"].iloc[0]

    st.markdown(
        f"""
        <div style="background-color:#f0f7f4; border-left:6px solid #2E8B57;
                    padding:1rem 1.5rem; border-radius:0.5rem; margin-top:0.5rem;">
            <span style="font-size:0.95rem; color:#333;">Predicted Charges</span><br>
            <span style="font-size:2.2rem; font-weight:700; color:#2E8B57;">{money(point_pred)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    p1, p2 = st.columns(2)
    p1.metric("95% CI — mean response", f"{money(ci_low)} to {money(ci_high)}")
    p2.metric("95% PI — individual prediction", f"{money(pi_low)} to {money(pi_high)}")

    st.divider()

    # ------------------ Diagnostics ------------------
    st.subheader("Residual Diagnostics (Gauss-Markov Checks)")

    fitted_vals = model.fittedvalues
    residuals = model.resid

    diag1, diag2 = st.columns(2)
    with diag1:
        st.markdown("**Residuals vs Fitted** — linearity & homoscedasticity")
        fig_resid = px.scatter(
            x=fitted_vals, y=residuals, opacity=0.5,
            labels={"x": "Fitted values", "y": "Residuals"},
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig_resid, use_container_width=True)

    with diag2:
        st.markdown("**Q-Q Plot** — normality of residuals")
        qq = sm.qqplot(residuals, line="s", fit=True)
        theoretical_q = qq.axes[0].lines[0].get_xdata()
        sample_q = qq.axes[0].lines[0].get_ydata()
        line_x = qq.axes[0].lines[1].get_xdata()
        line_y = qq.axes[0].lines[1].get_ydata()
        import matplotlib.pyplot as plt
        plt.close(qq)

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode="markers", name="Residuals", opacity=0.6))
        fig_qq.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Reference line", line=dict(color="red")))
        fig_qq.update_layout(
            xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    jb_stat, jb_p, skew_r, kurt_r = jarque_bera(residuals)
    d1, d2, d3 = st.columns(3)
    d1.metric("Jarque-Bera p-value", f"{jb_p:.4g}")
    d2.metric("Omnibus p-value", f"{sm.stats.omni_normtest(residuals)[1]:.4g}")
    d3.metric("Residual mean", f"{residuals.mean():.4f}")

    if jb_p < ALPHA:
        st.warning("Residuals deviate significantly from normality (Jarque-Bera p < 0.05). "
                   "This is common with right-skewed cost data; a log-transform of `charges` could improve fit.")
    else:
        st.success("Residuals appear approximately normal (Jarque-Bera p >= 0.05).")

    st.markdown("**Multicollinearity** — Variance Inflation Factor (continuous predictors)")
    X_vif = sm.add_constant(df_full[["age", "bmi", "children"]].copy())
    vif_data = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })
    st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True)
    st.caption("Rule of thumb: VIF above 5-10 indicates problematic multicollinearity.")

st.divider()
st.caption("Lab-4 · M.Sc. Data Science — Statistical Modeling with Python · Built with Streamlit + statsmodels")