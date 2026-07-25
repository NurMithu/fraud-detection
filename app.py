"""
Credit Card Fraud Detection Dashboard
Live companion to Fraud_Detection_Analysis.ipynb

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🕵️", layout="wide")

PRIMARY = "#4F46E5"
DANGER = "#DC2626"
GREEN = "#059669"
GRAY = "#94A3B8"

REQUIRED_COLS_PREFIX = "V"


@st.cache_data
def load_default():
    return pd.read_csv("data/creditcard.csv")


@st.cache_data
def prepare(df: pd.DataFrame):
    df = df.copy()
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    else:
        df["Hour"] = 0
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    return df


def get_features(df):
    return [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Hour"]


@st.cache_resource
def train_models(df: pd.DataFrame):
    features = get_features(df)
    X = df[features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    log_reg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    log_reg.fit(X_train, y_train)
    lr_probs = log_reg.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
    xgb = XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    def eval_at(probs, thr=0.5):
        preds = (probs >= thr).astype(int)
        return {
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probs) if y_test.nunique() > 1 else np.nan,
            "PR_AUC": average_precision_score(y_test, probs),
        }

    all_results = {
        "Logistic Regression": (log_reg, lr_probs, eval_at(lr_probs)),
        "Random Forest": (rf, rf_probs, eval_at(rf_probs)),
        "XGBoost": (xgb, xgb_probs, eval_at(xgb_probs)),
    }
    best_name = max(all_results, key=lambda n: all_results[n][2]["PR_AUC"])
    best_model, best_probs, best_metrics = all_results[best_name]

    return {
        "features": features,
        "X_test": X_test, "y_test": y_test,
        "all_results": all_results,
        "best_name": best_name, "best_model": best_model, "best_probs": best_probs,
        "best_metrics": best_metrics,
    }


# ---------------------------------------------------------------------------
st.sidebar.title("🕵️ Fraud Detection")
st.sidebar.caption("Imbalanced classification — live model")

data_mode = st.sidebar.radio("Data source", ["Demo dataset", "Upload my own"])
if data_mode == "Upload my own":
    uploaded = st.sidebar.file_uploader(
        "Transactions CSV (V1-V28, Amount, Time, Class)", type="csv"
    )
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        if not any(c.startswith("V") for c in raw_df.columns) or "Class" not in raw_df.columns:
            st.sidebar.error("Missing required columns. Showing demo data instead.")
            raw_df = load_default()
        else:
            st.sidebar.success("Your data is loaded ✅")
    else:
        st.sidebar.info("Upload a CSV to use your own data. Showing demo data meanwhile.")
        raw_df = load_default()
else:
    raw_df = load_default()
    st.sidebar.info("Showing the public credit card fraud demo dataset.")

with st.spinner("Training models (Logistic Regression, Random Forest, XGBoost)..."):
    df = prepare(raw_df)
    m = train_models(df)

threshold = st.sidebar.slider(
    "Decision threshold", 0.0, 1.0, 0.5, 0.01,
    help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but miss more fraud.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "[View the full notebook & methodology on GitHub](https://github.com/NurMithu)"
)

# ---------------------------------------------------------------------------
fraud_rate = df["Class"].mean()
preds_at_thr = (m["best_probs"] >= threshold).astype(int)
precision_at_thr = precision_score(m["y_test"], preds_at_thr, zero_division=0)
recall_at_thr = recall_score(m["y_test"], preds_at_thr, zero_division=0)

st.title("Credit Card Fraud Detection Dashboard")
st.subheader(f"Live Model — Best: {m['best_name']}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fraud Rate in Data", f"{fraud_rate:.2%}")
k2.metric("PR-AUC (best model)", f"{m['best_metrics']['PR_AUC']:.3f}")
k3.metric("Precision @ threshold", f"{precision_at_thr:.1%}")
k4.metric("Recall @ threshold", f"{recall_at_thr:.1%}")

st.markdown("---")

tab_overview, tab_models, tab_flagged, tab_reco = st.tabs(
    ["📊 Overview", "🎯 Model Comparison", "🚩 Flagged Transactions", "📋 Recommendations"]
)

# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig = px.bar(x=counts.index, y=counts.values, title="Class Imbalance",
                     color=counts.index, color_discrete_map={"Legitimate": PRIMARY, "Fraud": DANGER})
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Transactions")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            df[df["Amount"] < df["Amount"].quantile(0.99)], x="Amount", color="Class",
            nbins=50, barmode="overlay", title="Transaction Amount Distribution",
            color_discrete_map={0: PRIMARY, 1: DANGER},
        )
        st.plotly_chart(fig, use_container_width=True)

    importance_model = m["best_model"]
    if hasattr(importance_model, "feature_importances_"):
        importance = pd.Series(importance_model.feature_importances_, index=m["features"]).sort_values(ascending=False).head(10)
    else:
        importance = pd.Series(np.abs(importance_model.coef_[0]), index=m["features"]).sort_values(ascending=False).head(10)
    fig = px.bar(importance.sort_values(), orientation="h", title=f"Top 10 Fraud Indicators ({m['best_name']})",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_models:
    st.markdown("#### All models trained live on this data (holdout test set)")
    comp_rows = []
    for name, (_, probs, metrics) in m["all_results"].items():
        row = dict(metrics)
        row["Model"] = name
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows).set_index("Model")[["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]]
    st.dataframe(
        comp_df.style.format("{:.3f}").highlight_max(subset=["PR_AUC"], color="#D1FAE5"),
        use_container_width=True,
    )

    fig = go.Figure()
    for name, (_, probs, _) in m["all_results"].items():
        p, r, _ = precision_recall_curve(m["y_test"], probs)
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=name))
    fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

    cm = confusion_matrix(m["y_test"], preds_at_thr)
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted Legit", "Predicted Fraud"], y=["Actual Legit", "Actual Fraud"],
        title=f"Confusion Matrix — {m['best_name']} @ threshold {threshold:.2f}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_flagged:
    st.markdown(f"#### Transactions flagged at threshold {threshold:.2f}")
    test_view = df.loc[m["y_test"].index].copy()
    test_view["Fraud_Probability"] = m["best_probs"]
    test_view["Flagged"] = preds_at_thr
    flagged = test_view[test_view["Flagged"] == 1].sort_values("Fraud_Probability", ascending=False)

    st.metric("Transactions flagged for review", f"{len(flagged):,}")

    display_cols = ["Amount", "Hour", "Fraud_Probability", "Class"]
    st.dataframe(
        flagged[display_cols].head(25).rename(columns={"Class": "Actually Fraud"})
        .style.format({"Amount": "€{:.2f}", "Fraud_Probability": "{:.1%}"}),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download all flagged transactions (CSV)",
        flagged[display_cols].to_csv(index=True).encode("utf-8"),
        "flagged_transactions.csv", "text/csv",
    )

# ---------------------------------------------------------------------------
with tab_reco:
    avg_fraud_amount = df.loc[df["Class"] == 1, "Amount"].mean()
    n_fraud_test = int(m["y_test"].sum())
    caught = int(round(recall_at_thr * n_fraud_test))
    estimated_savings = caught * avg_fraud_amount

    st.markdown("#### Business recommendations")
    st.markdown(
        f"""
- **Deploy {m['best_name']} at a business-chosen threshold**, not the default
  0.5 — at the current threshold ({threshold:.2f}), the model catches
  **{recall_at_thr:.0%}** of fraud with **{precision_at_thr:.0%}** precision,
  meaning roughly **{'1 in ' + str(round(1/precision_at_thr)) if precision_at_thr > 0 else 'few'}** flagged
  transactions is a false alarm.
- **Estimated fraud caught on this test set: ~€{estimated_savings:,.0f}** —
  use the threshold slider above to see the precision/recall tradeoff and
  choose the operating point that matches your actual cost of a missed fraud
  vs. a false alarm.
- **Monitor PR-AUC, not accuracy**, once deployed — accuracy stays high even
  if the model catches almost no fraud, because legitimate transactions vastly
  outnumber fraud.
- **Revisit the threshold periodically** as fraud patterns and business
  tolerance shift — this isn't a "set once" decision.
"""
    )
    st.link_button("📓 View the full notebook & methodology on GitHub", "https://github.com/NurMithu")

st.markdown("---")
st.caption(
    "Live models trained on the public credit card fraud dataset. "
    "Upload your own transactions CSV in the sidebar (same column schema) to see results on your own data."
)
