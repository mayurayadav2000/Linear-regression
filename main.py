import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Linear Regression App", page_icon="📈", layout="wide")

st.title("📈 Linear Regression Predictor")
st.markdown("A complete ML app implementing **Linear Regression** — OLS, Ridge, Lasso, Residuals & More")
st.markdown("---")

# ── Load Dataset ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "linear_data.txt")
    df = pd.read_csv(file_path)
    return df

df_raw = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
model_type = st.sidebar.selectbox("Regression Type", ["Linear", "Ridge", "Lasso"], index=0)
alpha_val  = st.sidebar.select_slider("Alpha (Ridge/Lasso)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
test_size  = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Your Own Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded!")

df = df_raw.copy()

# ── Preprocess ────────────────────────────────────────────────────────────────
feature_cols = df.columns[:-1].tolist()
target_col   = df.columns[-1]

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

# ── Build Model ───────────────────────────────────────────────────────────────
if model_type == "Ridge":
    model = Ridge(alpha=alpha_val, fit_intercept=fit_intercept)
elif model_type == "Lasso":
    model = Lasso(alpha=alpha_val, fit_intercept=fit_intercept, max_iter=10000)
else:
    model = LinearRegression(fit_intercept=fit_intercept)

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

mse   = mean_squared_error(y_test, y_pred_test)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred_test)
r2    = r2_score(y_test, y_pred_test)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(feature_cols) - 1)

residuals = y_test - y_pred_test

COLORS = ["#4C72B0", "#2a9d8f", "#e63946", "#e9c46a"]

# ═══════════════════════════════ TABS ═════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", "📈 Visualizations", "📐 LR Concepts",
    "📉 Model Results", "🔮 Predict"
])

# ══════════════════ TAB 1 — DATASET ══════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",     df.shape[0])
    c2.metric("Total Features", len(feature_cols))
    c3.metric("Target Column",  target_col)
    c4.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Target Variable Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[target_col], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution of '{target_col}'")

    axes[1].boxplot(df[target_col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", color="#4C72B0"),
                    medianprops=dict(color="white", linewidth=2))
    axes[1].set_ylabel(target_col)
    axes[1].set_title(f"Boxplot of '{target_col}'")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature vs Target (Scatter)")
    sel_feat = st.selectbox("Select Feature to Plot vs Target", feature_cols)
    fig_sc, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df[sel_feat], df[target_col], color="#4C72B0",
               alpha=0.6, edgecolors="white", s=60)
    # Add regression line
    m_coef = np.polyfit(df[sel_feat], df[target_col], 1)
    x_line = np.linspace(df[sel_feat].min(), df[sel_feat].max(), 100)
    ax.plot(x_line, np.polyval(m_coef, x_line), color="#e63946",
            linewidth=2, label="Regression Line")
    ax.set_xlabel(sel_feat)
    ax.set_ylabel(target_col)
    ax.set_title(f"{sel_feat} vs {target_col}")
    ax.legend()
    st.pyplot(fig_sc)

# ══════════════════ TAB 2 — VISUALIZATIONS ═══════════════════════════════════
with tab2:
    st.subheader("Feature Distributions")
    n_feats = len(feature_cols)
    ncols   = min(4, n_feats)
    nrows   = (n_feats + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat  = np.array(axes).flatten() if n_feats > 1 else [axes]
    for i, feat in enumerate(feature_cols):
        axes_flat[i].hist(df[feat], bins=20, color=COLORS[i % len(COLORS)], edgecolor="white")
        axes_flat[i].set_title(feat, fontsize=9)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(df.corr(), annot=(len(df.columns) <= 12), fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.4)
    ax.set_title("Feature + Target Correlation Matrix")
    st.pyplot(fig3)

    st.subheader("Scatter Matrix — Top Features vs Target")
    top_feats = feature_cols[:3]
    fig4, axes2 = plt.subplots(1, len(top_feats), figsize=(6 * len(top_feats), 5))
    if len(top_feats) == 1:
        axes2 = [axes2]
    for i, feat in enumerate(top_feats):
        axes2[i].scatter(df[feat], df[target_col], color=COLORS[i],
                         alpha=0.6, edgecolors="white", s=50)
        coef = np.polyfit(df[feat], df[target_col], 1)
        xln  = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes2[i].plot(xln, np.polyval(coef, xln), color="#e63946", linewidth=2)
        axes2[i].set_xlabel(feat)
        axes2[i].set_ylabel(target_col)
        axes2[i].set_title(f"{feat} vs {target_col}")
    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("Actual vs Predicted — Test Set")
    fig5, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred_test, color="#4C72B0", alpha=0.7,
               edgecolors="white", s=60, label="Predictions")
    mn_v = min(y_test.min(), y_pred_test.min())
    mx_v = max(y_test.max(), y_pred_test.max())
    ax.plot([mn_v, mx_v], [mn_v, mx_v], color="#e63946",
            linewidth=2, linestyle="--", label="Perfect Prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig5)

    st.subheader("Residuals Analysis")
    fig6, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    # Residuals vs Predicted
    axes3[0].scatter(y_pred_test, residuals, color="#4C72B0",
                     alpha=0.7, edgecolors="white", s=55)
    axes3[0].axhline(0, color="#e63946", linestyle="--", linewidth=2)
    axes3[0].set_xlabel("Predicted Values")
    axes3[0].set_ylabel("Residuals")
    axes3[0].set_title("Residuals vs Predicted")

    # Residuals histogram
    axes3[1].hist(residuals, bins=25, color="#2a9d8f", edgecolor="white")
    axes3[1].axvline(0, color="#e63946", linestyle="--", linewidth=2)
    axes3[1].set_xlabel("Residual")
    axes3[1].set_ylabel("Frequency")
    axes3[1].set_title("Residuals Distribution")

    # Q-Q plot
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes3[2].scatter(osm, osr, color="#e9c46a", alpha=0.8, edgecolors="white", s=50)
    x_qq = np.array([osm.min(), osm.max()])
    axes3[2].plot(x_qq, slope * x_qq + intercept, color="#e63946", linewidth=2)
    axes3[2].set_xlabel("Theoretical Quantiles")
    axes3[2].set_ylabel("Sample Quantiles")
    axes3[2].set_title("Q-Q Plot (Normality Check)")

    plt.tight_layout()
    st.pyplot(fig6)

# ══════════════════ TAB 3 — LR CONCEPTS ══════════════════════════════════════
with tab3:
    st.subheader("📐 How Linear Regression Works")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
### What is Linear Regression?
**Linear Regression** finds the best-fit straight line (hyperplane) through
data points by minimizing the **sum of squared residuals**.

---

### The Equation
$$\\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n$$

| Term | Meaning |
|------|---------|
| ŷ | Predicted value |
| w₀ | Intercept (bias) |
| w₁...wₙ | Coefficients (slopes) |
| x₁...xₙ | Feature values |

---

### Cost Function — OLS (Ordinary Least Squares)
$$J = \\frac{1}{m} \\sum_{i=1}^{m} (y_i - \\hat{y}_i)^2$$

Minimize **MSE** to find the best-fit line.

---

### Evaluation Metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| **MSE** | Σ(y−ŷ)²/n | Mean Squared Error |
| **RMSE** | √MSE | Root Mean Squared Error |
| **MAE** | Σ\|y−ŷ\|/n | Mean Absolute Error |
| **R²** | 1 − SS_res/SS_tot | Variance explained (0–1) |

---

### Regularization Variants
| Type | Penalty | Effect |
|------|---------|--------|
| **Linear** | None | Basic OLS |
| **Ridge** | L2 = α·Σw² | Shrinks all coefficients |
| **Lasso** | L1 = α·Σ\|w\| | Forces some to zero |
        """)

    with col_r:
        # Best-fit line demo
        np.random.seed(42)
        x_demo = np.linspace(0, 10, 80)
        y_demo = 2.5 * x_demo + 3 + np.random.randn(80) * 3

        fig7, axes4 = plt.subplots(2, 1, figsize=(6, 8))

        # Scatter + regression line
        coef_d = np.polyfit(x_demo, y_demo, 1)
        y_fit  = np.polyval(coef_d, x_demo)
        axes4[0].scatter(x_demo, y_demo, color="#4C72B0", alpha=0.7,
                         edgecolors="white", s=45, label="Data Points")
        axes4[0].plot(x_demo, y_fit, color="#e63946", linewidth=2.5, label="Best-fit Line")
        for xi, yi, yfi in zip(x_demo[::6], y_demo[::6], y_fit[::6]):
            axes4[0].plot([xi, xi], [yi, yfi], color="gray", linewidth=0.8, alpha=0.7)
        axes4[0].set_title("Linear Regression — Best-fit Line & Residuals",
                            fontsize=11, fontweight="bold")
        axes4[0].set_xlabel("x")
        axes4[0].set_ylabel("y")
        axes4[0].legend()

        # Coefficient bar chart
        coefs_vals = model.coef_ if hasattr(model.coef_, '__len__') else [model.coef_]
        coef_df    = pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs_vals})
        coef_df    = coef_df.sort_values("Coefficient", ascending=True)
        bar_colors = ["#e63946" if c < 0 else "#2a9d8f" for c in coef_df["Coefficient"]]
        axes4[1].barh(coef_df["Feature"], coef_df["Coefficient"],
                      color=bar_colors, edgecolor="white")
        axes4[1].axvline(0, color="black", linewidth=0.9)
        axes4[1].set_xlabel("Coefficient Value")
        axes4[1].set_title("Model Coefficients (Feature Weights)",
                            fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig7)

    # ── Coefficient Table ──────────────────────────────────────────────────
    st.subheader("Coefficient Table")
    coef_table = pd.DataFrame({
        "Feature":        feature_cols,
        "Coefficient":    np.round(model.coef_, 6),
        "Abs Importance": np.round(np.abs(model.coef_), 6),
        "Direction":      ["Positive ↑" if c > 0 else "Negative ↓" for c in model.coef_]
    }).sort_values("Abs Importance", ascending=False).reset_index(drop=True)
    coef_table["Intercept"] = round(float(model.intercept_), 6) if fit_intercept else 0
    st.dataframe(coef_table, use_container_width=True)

    # ── Cross-Validation ───────────────────────────────────────────────────
    st.subheader("Cross-Validation Scores (5-Fold) — R² Score")
    cv_model  = LinearRegression(fit_intercept=fit_intercept)
    cv_scores = cross_val_score(cv_model, X_scaled, y, cv=5, scoring="r2")
    cv_df = pd.DataFrame({
        "Fold":    [f"Fold {i+1}" for i in range(5)],
        "R² Score": [f"{s:.4f}" for s in cv_scores]
    })
    st.dataframe(cv_df, use_container_width=True)
    st.success(f"Mean CV R²: **{cv_scores.mean():.4f}** ± {cv_scores.std():.4f}")

    # ── Model parameters ──────────────────────────────────────────────────
    st.subheader("Current Model Parameters")
    params_df = pd.DataFrame({
        "Parameter":   ["Model Type", "Alpha", "Fit Intercept", "Test Size"],
        "Value":       [model_type, alpha_val, fit_intercept, test_size],
        "Description": [
            "Regression variant (Linear / Ridge / Lasso)",
            "Regularization strength for Ridge/Lasso",
            "Whether to calculate the intercept",
            "Fraction of data used for testing"
        ]
    })
    st.dataframe(params_df, use_container_width=True)

# ══════════════════ TAB 4 — MODEL RESULTS ════════════════════════════════════
with tab4:
    st.subheader("Model Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📐 R² Score",  f"{r2:.4f}")
    c2.metric("📐 Adj R²",    f"{adj_r2:.4f}")
    c3.metric("📉 RMSE",      f"{rmse:.4f}")
    c4.metric("📉 MAE",       f"{mae:.4f}")
    st.metric("📉 MSE", f"{mse:.4f}")

    # Train vs Test performance
    st.subheader("Train vs Test Performance")
    train_r2   = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    perf_df = pd.DataFrame({
        "Set":   ["Train", "Test"],
        "R²":    [round(train_r2,   4), round(r2,   4)],
        "RMSE":  [round(train_rmse, 4), round(rmse, 4)],
        "MAE":   [round(mean_absolute_error(y_train, y_pred_train), 4), round(mae, 4)]
    })
    st.dataframe(perf_df, use_container_width=True)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Values (Test Set)")
    pred_df = pd.DataFrame({
        "Actual":    np.round(y_test[:20],      3),
        "Predicted": np.round(y_pred_test[:20], 3),
        "Error":     np.round(residuals[:20],   3)
    })
    st.dataframe(pred_df, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (|Coefficients|)")
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": np.abs(model.coef_)
    }).sort_values("Importance", ascending=False)
    fig8, ax = plt.subplots(figsize=(8, max(3, len(feature_cols) * 0.4)))
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                   color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel("|Coefficient|")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    st.pyplot(fig8)

    # Model comparison
    st.subheader("Model Comparison — Linear vs Ridge vs Lasso")
    variants     = ["Linear", "Ridge", "Lasso"]
    variant_r2   = []
    variant_rmse = []
    for v in variants:
        if v == "Ridge":
            m = Ridge(alpha=alpha_val)
        elif v == "Lasso":
            m = Lasso(alpha=alpha_val, max_iter=10000)
        else:
            m = LinearRegression()
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        variant_r2.append(r2_score(y_test, yp))
        variant_rmse.append(np.sqrt(mean_squared_error(y_test, yp)))

    fig9, axes5 = plt.subplots(1, 2, figsize=(12, 4))
    b1 = axes5[0].bar(variants, variant_r2,
                      color=["#4C72B0", "#2a9d8f", "#e9c46a"], edgecolor="white")
    axes5[0].bar_label(b1, fmt="%.4f", padding=3)
    axes5[0].set_ylabel("R² Score")
    axes5[0].set_title("R² Score Comparison")
    axes5[0].set_ylim(0, max(variant_r2) * 1.15)

    b2 = axes5[1].bar(variants, variant_rmse,
                      color=["#4C72B0", "#2a9d8f", "#e9c46a"], edgecolor="white")
    axes5[1].bar_label(b2, fmt="%.4f", padding=3)
    axes5[1].set_ylabel("RMSE")
    axes5[1].set_title("RMSE Comparison")
    plt.tight_layout()
    st.pyplot(fig9)

    # Accuracy vs alpha for Ridge/Lasso
    st.subheader("R² Score vs Alpha (Ridge & Lasso)")
    alphas   = [0.001, 0.01, 0.1, 1, 10, 100]
    ridge_r2 = []
    lasso_r2 = []
    for a in alphas:
        mr = Ridge(alpha=a);      mr.fit(X_train, y_train)
        ml = Lasso(alpha=a, max_iter=10000); ml.fit(X_train, y_train)
        ridge_r2.append(r2_score(y_test, mr.predict(X_test)))
        lasso_r2.append(r2_score(y_test, ml.predict(X_test)))

    fig10, ax = plt.subplots(figsize=(8, 4))
    ax.plot([str(a) for a in alphas], ridge_r2, marker="o", color="#4C72B0",
            linewidth=2, label="Ridge")
    ax.plot([str(a) for a in alphas], lasso_r2, marker="s", color="#2a9d8f",
            linewidth=2, label="Lasso")
    ax.axvline(x=str(alpha_val), color="red", linestyle="--",
               label=f"Current α={alpha_val}")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score vs Alpha — Ridge & Lasso")
    ax.legend()
    st.pyplot(fig10)

# ══════════════════ TAB 5 — PREDICT ══════════════════════════════════════════
with tab5:
    st.subheader("🔮 Make a Prediction")
    st.markdown("Adjust the feature values below and click **Predict**.")

    input_vals = []
    num_cols   = 3
    row_groups = [feature_cols[i:i+num_cols] for i in range(0, len(feature_cols), num_cols)]

    for row_feats in row_groups:
        cols = st.columns(num_cols)
        for i, feat in enumerate(row_feats):
            min_v  = float(df[feat].min())
            max_v  = float(df[feat].max())
            mean_v = float(df[feat].mean())
            step_v = round((max_v - min_v) / 100, 4) if (max_v - min_v) > 0 else 0.01
            val = cols[i].number_input(feat, min_value=min_v, max_value=max_v,
                                       value=mean_v, step=step_v)
            input_vals.append(val)

    if st.button("🚀 Predict", use_container_width=True):
        inp_arr    = np.array([input_vals])
        inp_scaled = scaler.transform(inp_arr)
        prediction = float(model.predict(inp_scaled)[0])

        st.success(f"### 🎯 Predicted **{target_col}**: `{prediction:.4f}`")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Prediction in Context")
            fig11, ax = plt.subplots(figsize=(6, 4))
            ax.hist(y, bins=30, color="#4C72B0", edgecolor="white",
                    alpha=0.7, label="All Values")
            ax.axvline(prediction, color="#e63946", linewidth=3,
                       linestyle="--", label=f"Prediction = {prediction:.2f}")
            ax.axvline(np.mean(y), color="#2a9d8f", linewidth=2,
                       linestyle=":", label=f"Mean = {np.mean(y):.2f}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frequency")
            ax.set_title("Prediction vs Data Distribution")
            ax.legend()
            st.pyplot(fig11)

        with col_b:
            st.subheader("Prediction Summary")
            pct = (prediction - y.min()) / (y.max() - y.min()) * 100
            st.markdown(f"""
| Term | Value |
|------|-------|
| **Predicted {target_col}** | `{prediction:.4f}` |
| **Dataset Min** | `{y.min():.4f}` |
| **Dataset Max** | `{y.max():.4f}` |
| **Dataset Mean** | `{np.mean(y):.4f}` |
| **Percentile Position** | `{pct:.1f}%` |
| **Model Type** | `{model_type}` |
| **R² Score** | `{r2:.4f}` |
| **RMSE** | `{rmse:.4f}` |
            """)

        # Contribution of each feature
        st.subheader("Feature Contribution to Prediction")
        contributions = inp_scaled[0] * model.coef_
        contrib_df = pd.DataFrame({
            "Feature":      feature_cols,
            "Contribution": contributions
        }).sort_values("Contribution", ascending=True)
        bar_clrs = ["#e63946" if c < 0 else "#2a9d8f" for c in contrib_df["Contribution"]]
        fig12, ax = plt.subplots(figsize=(8, max(3, len(feature_cols) * 0.4)))
        ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                color=bar_clrs, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Contribution to Prediction")
        ax.set_title("How Each Feature Contributed to This Prediction")
        st.pyplot(fig12)

st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit & Scikit-learn | Linear Regression Project</center>",
    unsafe_allow_html=True
)
