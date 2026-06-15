"""MedicaidGuard-XAI investigator dashboard.

Run with:  streamlit run dashboard/app.py

Every page that shows a score also shows the demonstration disclaimer. That is
not decoration: a prioritisation tool whose outputs get read as findings is the
main foreseeable harm from this kind of system, so the framing is part of the
product rather than a footnote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medicaidguard.config import (  # noqa: E402
    ARTIFACTS_DIR,
    RESULTS_DIR,
    SYNTHETIC_DIR,
    TABLES_DIR,
    load_settings,
)
from medicaidguard.evaluation.splits import time_split  # noqa: E402
from medicaidguard.features.pipeline import FeaturePipeline  # noqa: E402
from medicaidguard.impact.simulator import SimulationInputs, simulate  # noqa: E402
from medicaidguard.models.fusion import HybridRiskModel  # noqa: E402
from medicaidguard.models.rules import rule_evidence, rule_score  # noqa: E402

DISCLAIMER = ("Synthetic/public research demonstration. Alerts are review "
              "recommendations, not findings of fraud.")

st.set_page_config(page_title="MedicaidGuard-XAI", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner="Loading dataset…")
def load_tables():
    names = ["claims", "evv_events", "patients", "providers", "caregivers",
             "scenario_labels", "regions"]
    if not (SYNTHETIC_DIR / "claims.parquet").exists():
        return None
    return {n: pd.read_parquet(SYNTHETIC_DIR / f"{n}.parquet") for n in names}


@st.cache_resource(show_spinner="Scoring claims…")
def score_everything():
    tables = load_tables()
    if tables is None:
        return None
    p = ARTIFACTS_DIR / "final_model.joblib"
    if not p.exists():
        return None
    import joblib

    bundle = joblib.load(p)
    settings = load_settings()
    claims = tables["claims"].sort_values(["service_date", "claim_id"]).reset_index(drop=True)
    split = time_split(claims, settings.split)
    pipe = FeaturePipeline()
    pipe.fit(claims, tables["evv_events"], tables["patients"],
             pd.Series((split == "train").to_numpy(), index=claims.index))
    feats = pipe.transform(claims, tables["evv_events"], tables["patients"])
    X = feats.drop(columns=["claim_id"])
    Xm = X[bundle["columns"]]

    prob = bundle["calibrator"].transform(bundle["model"].predict_proba(Xm)[:, 1])
    rs = rule_score(X).to_numpy() / 100.0
    score = np.clip(100 * (0.75 * prob / max(prob.max(), 1e-9) + 0.25 * rs), 0, 100)

    scored = claims.copy()
    scored["risk_score"] = score
    scored["model_probability"] = prob
    scored["review_priority"] = HybridRiskModel.priority_band(score)
    scored["alert_id"] = [f"SYN-{i:06d}" for i in range(len(scored))]
    scored["split"] = split.to_numpy()
    scored["simulated_label"] = (
        tables["scenario_labels"].set_index("claim_id")["is_suspicious"]
        .reindex(scored["claim_id"]).fillna(0).astype(int).to_numpy())
    return {"scored": scored, "features": feats, "tables": tables,
            "bundle": bundle, "split": split}


def banner():
    st.warning(DISCLAIMER, icon="⚠️")


def require(state):
    if state is None:
        st.error("No trained model or dataset found. Run:\n\n"
                 "```\npython scripts/generate_data.py\npython scripts/train.py\n```")
        st.stop()
    return state


# --------------------------------------------------------------------------

state = score_everything()
st.sidebar.title("MedicaidGuard-XAI")
page = st.sidebar.radio("Page", [
    "1 · Executive overview", "2 · Alert explorer", "3 · Alert details",
    "4 · Provider risk", "5 · EVV analytics", "6 · Model evaluation",
    "7 · Scenario simulator", "8 · Reproducibility",
])
st.sidebar.caption(DISCLAIMER)

# --- Page 1 ---------------------------------------------------------------
if page.startswith("1"):
    st.title("Executive overview")
    banner()
    s = require(state)
    d, t = s["scored"], s["tables"]

    c = st.columns(5)
    c[0].metric("Claims", f"{len(d):,}")
    c[1].metric("EVV events", f"{len(t['evv_events']):,}")
    c[2].metric("Caregivers", f"{t['caregivers'].shape[0]:,}")
    high = int((d["review_priority"].isin(["high", "critical"])).sum())
    c[3].metric("High/critical alerts", f"{high:,}")
    c[4].metric("Alert rate", f"{high / len(d):.2%}")

    a, b = st.columns(2)
    counts = d["review_priority"].value_counts().reindex(
        ["low", "medium", "high", "critical"]).fillna(0)
    a.plotly_chart(px.bar(counts, title="Review priority distribution",
                          labels={"value": "claims", "index": "priority"}),
                   use_container_width=True)
    b.plotly_chart(px.histogram(d, x="risk_score", nbins=50,
                                title="Risk score distribution"),
                   use_container_width=True)

    st.subheader("Data quality")
    miss = {n: float(df.isna().mean().mean()) for n, df in t.items()}
    st.dataframe(pd.DataFrame({"table": miss.keys(),
                               "mean missing rate": [round(v, 4) for v in miss.values()]}),
                 use_container_width=True, hide_index=True)

# --- Page 2 ---------------------------------------------------------------
elif page.startswith("2"):
    st.title("Alert explorer")
    banner()
    s = require(state)
    d = s["scored"]

    c = st.columns(4)
    pri = c[0].multiselect("Priority", ["critical", "high", "medium", "low"],
                           default=["critical", "high"])
    prov = c[1].selectbox("Provider", ["(all)"] + sorted(d["provider_id"].unique()))
    lo = c[2].slider("Minimum risk score", 0, 100, 60)
    lim = c[3].number_input("Rows", 10, 2000, 200, step=10)

    dates = st.slider("Service date range",
                      value=(d["service_date"].min().to_pydatetime(),
                             d["service_date"].max().to_pydatetime()))

    view = d[d["review_priority"].isin(pri) & (d["risk_score"] >= lo)]
    view = view[(view["service_date"] >= dates[0]) & (view["service_date"] <= dates[1])]
    if prov != "(all)":
        view = view[view["provider_id"] == prov]

    st.caption(f"{len(view):,} matching claims")
    st.dataframe(
        view.nlargest(int(lim), "risk_score")[
            ["alert_id", "claim_id", "service_date", "provider_id", "caregiver_id",
             "patient_id", "billed_hours", "authorized_hours", "risk_score",
             "model_probability", "review_priority"]],
        use_container_width=True, hide_index=True)

# --- Page 3 ---------------------------------------------------------------
elif page.startswith("3"):
    st.title("Alert details")
    banner()
    s = require(state)
    d, feats = s["scored"], s["features"]

    top = d.nlargest(300, "risk_score")
    pick = st.selectbox("Alert", top["alert_id"].tolist())
    row = d[d["alert_id"] == pick].iloc[0]
    i = d.index[d["alert_id"] == pick][0]

    c = st.columns(4)
    c[0].metric("Risk score", f"{row['risk_score']:.0f}/100")
    c[1].metric("Priority", str(row["review_priority"]).title())
    c[2].metric("Calibrated probability", f"{row['model_probability']:.3f}")
    c[3].metric("Billed vs authorised",
                f"{row['billed_hours']:.2f} / {row['authorized_hours']:.2f} h")
    st.caption("Probability calibrated with isotonic regression fitted on the "
               "validation window.")

    X = feats.drop(columns=["claim_id"])
    st.subheader("Rule evidence")
    ev = rule_evidence(X, X.index[i])
    if ev:
        st.dataframe(pd.DataFrame(ev), use_container_width=True, hide_index=True)
    else:
        st.info("No rule fired above the reporting threshold; this alert rests on "
                "the learned model alone, which is weaker evidence for a reviewer.")

    st.subheader("Model attribution")
    if st.button("Compute SHAP explanation for this alert"):
        from medicaidguard.explainability.shap_explainer import SHAP_LIMITATION, ShapExplainer
        Xm = X[s["bundle"]["columns"]]
        ex = ShapExplainer(s["bundle"]["model"], Xm.sample(min(200, len(Xm)), random_state=0))
        loc = pd.DataFrame(ex.local_explanation(Xm, int(i)))
        st.dataframe(loc, use_container_width=True, hide_index=True)
        st.caption(SHAP_LIMITATION)

    st.subheader("Timeline")
    tl = go.Figure()
    tl.add_trace(go.Bar(x=[row["billed_hours"]], y=["Billed"], orientation="h",
                        name="Billed hours"))
    evv = s["tables"]["evv_events"]
    e = evv[evv["claim_id"] == row["claim_id"]]
    if len(e):
        dur = (e.iloc[0]["check_out_time"] - e.iloc[0]["check_in_time"]).total_seconds() / 3600
        tl.add_trace(go.Bar(x=[dur], y=["EVV verified"], orientation="h",
                            name="EVV hours"))
    else:
        st.info("No EVV record is linked to this claim.")
    tl.update_layout(height=220, xaxis_title="hours")
    st.plotly_chart(tl, use_container_width=True)

    st.text_area("Investigator notes", key=f"notes_{pick}", height=90)
    st.selectbox("Review status", ["pending", "in review", "cleared", "escalated"],
                 key=f"status_{pick}")
    st.caption("Notes are session-local in this demonstration and are not persisted.")

# --- Page 4 ---------------------------------------------------------------
elif page.startswith("4"):
    st.title("Provider risk analysis")
    banner()
    s = require(state)
    d = s["scored"]
    prov = st.selectbox("Provider", sorted(d["provider_id"].unique()))
    sub = d[d["provider_id"] == prov]

    c = st.columns(4)
    c[0].metric("Claims", f"{len(sub):,}")
    c[1].metric("Mean risk", f"{sub['risk_score'].mean():.1f}")
    c[2].metric("Peer mean risk", f"{d['risk_score'].mean():.1f}")
    c[3].metric("High/critical", int(sub["review_priority"].isin(["high", "critical"]).sum()))

    monthly = (sub.set_index("service_date").resample("ME")
               .agg(mean_risk=("risk_score", "mean"), claims=("claim_id", "count"),
                    hours=("billed_hours", "sum")).reset_index())
    st.plotly_chart(px.line(monthly, x="service_date", y="mean_risk",
                            title="Monthly mean risk score"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(monthly, x="service_date", y="hours",
                          title="Monthly billed hours"), use_container_width=True)
    b.plotly_chart(px.box(d.assign(is_this=lambda x: np.where(
        x["provider_id"] == prov, prov, "peers")), x="is_this", y="billed_hours",
        title="Billed hours vs peers"), use_container_width=True)

# --- Page 5 ---------------------------------------------------------------
elif page.startswith("5"):
    st.title("EVV analytics")
    banner()
    s = require(state)
    d, feats, t = s["scored"], s["features"], s["tables"]
    X = feats.drop(columns=["claim_id"])

    c = st.columns(4)
    c[0].metric("Claims without EVV", f"{int((X.get('evv_present', pd.Series(1)) == 0).sum()):,}")
    c[1].metric("Overlapping visits", f"{int(X.get('visit_overlaps_prev', pd.Series(0)).sum()):,}")
    c[2].metric("Implausible travel",
                f"{int(X.get('implausible_travel_flag', pd.Series(0)).sum()):,}")
    c[3].metric("Median location accuracy",
                f"{X.get('location_accuracy_m', pd.Series([np.nan])).median():.0f} m")

    a, b = st.columns(2)
    a.plotly_chart(px.histogram(X, x="evv_billing_duration_diff", nbins=60,
                                title="Billed hours minus EVV-verified hours"),
                   use_container_width=True)
    speeds = X["implied_speed_kmh"].clip(0, 400)
    b.plotly_chart(px.histogram(speeds, nbins=60,
                                title="Implied travel speed between visits (km/h)"),
                   use_container_width=True)
    st.plotly_chart(px.histogram(X, x="checkin_distance_from_home_m", nbins=60,
                                 log_y=True,
                                 title="Check-in distance from registered home (m)"),
                    use_container_width=True)

# --- Page 6 ---------------------------------------------------------------
elif page.startswith("6"):
    st.title("Model evaluation")
    banner()
    p = RESULTS_DIR / "model_comparison.csv"
    if not p.exists():
        st.error("Run scripts/train.py first.")
        st.stop()
    comp = pd.read_csv(p, index_col=0)
    st.subheader("Held-out test window")
    st.dataframe(comp.round(4), use_container_width=True)

    st.info("Selection uses PR-AUC with a Brier calibration floor, not accuracy. "
            "At this prevalence a constant 'not suspicious' predictor reaches "
            f"{comp['accuracy_trivial_baseline'].iloc[0]:.3f} accuracy while "
            "detecting nothing, so accuracy cannot rank these models.")

    st.plotly_chart(px.bar(comp.reset_index(), x="index", y="pr_auc",
                           title="PR-AUC by model"), use_container_width=True)

    s = require(state)
    d = s["scored"]
    te = d["split"] == "test"
    y, sc = d.loc[te, "simulated_label"].to_numpy(), d.loc[te, "model_probability"].to_numpy()
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    pr, rc, _ = precision_recall_curve(y, sc)
    fpr, tpr, _ = roc_curve(y, sc)
    frac, mean_pred = calibration_curve(y, sc, n_bins=10, strategy="quantile")

    a, b, c3 = st.columns(3)
    a.plotly_chart(px.area(x=rc, y=pr, title="Precision-Recall",
                           labels={"x": "recall", "y": "precision"}), use_container_width=True)
    b.plotly_chart(px.area(x=fpr, y=tpr, title="ROC",
                           labels={"x": "FPR", "y": "TPR"}), use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name="model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash")))
    fig.update_layout(title="Calibration", xaxis_title="predicted", yaxis_title="observed")
    c3.plotly_chart(fig, use_container_width=True)

    for name, title in [("exp2_feature_ablation.csv", "Feature ablation"),
                        ("exp5_scenario_generalization.csv", "Scenario generalisation"),
                        ("exp6_prevalence.csv", "Class imbalance"),
                        ("exp1_statistical_comparison.csv", "Statistical comparison")]:
        f = TABLES_DIR / name
        if f.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(f).round(4), use_container_width=True, hide_index=True)

# --- Page 7 ---------------------------------------------------------------
elif page.startswith("7"):
    st.title("Operational scenario simulator")
    st.error("Illustrative scenario-based estimate—not measured real-world savings.",
             icon="🚫")
    s = require(state)
    d = s["scored"]

    c = st.columns(3)
    claims_n = c[0].number_input("Claims per period", 1_000, 5_000_000, 100_000, step=1_000)
    cap = c[1].slider("Review capacity (% of claims)", 0.5, 25.0, 5.0) / 100
    hours = c[2].number_input("Investigator hours per review", 0.25, 20.0, 1.5, step=0.25)
    c = st.columns(3)
    cost = c[0].number_input("Cost per investigator hour", 0.0, 500.0, 65.0)
    rate = c[1].slider("Hypothetical improper-payment rate (%)", 0.1, 20.0, 3.0) / 100
    amount = c[2].number_input("Hypothetical amount per case", 0.0, 100_000.0, 1_200.0)
    realise = st.slider("Recovery realisation rate (%)", 0.0, 100.0, 35.0) / 100

    te = d["split"] == "test"
    y = d.loc[te, "simulated_label"].to_numpy()
    sc = d.loc[te, "model_probability"].to_numpy()
    from medicaidguard.evaluation.metrics import precision_at_k, recall_at_k
    p_at = precision_at_k(y, sc, cap)
    r_at = recall_at_k(y, sc, cap)
    st.caption(f"Measured on the held-out synthetic test window: precision@{cap:.1%} "
               f"= {p_at:.3f}, recall@{cap:.1%} = {r_at:.3f}. Everything below this "
               "line is derived from your assumptions, not measured.")

    out = simulate(SimulationInputs(
        claims_reviewed_period=int(claims_n), review_capacity_pct=float(cap),
        investigator_hours_per_review=float(hours), investigator_cost_per_hour=float(cost),
        precision_at_capacity=float(p_at), recall_at_capacity=float(r_at),
        hypothetical_improper_rate=float(rate), hypothetical_amount_per_case=float(amount),
        recovery_realisation_rate=float(realise)))
    st.dataframe(pd.DataFrame([out]).T.rename(columns={0: "value"}),
                 use_container_width=True)

    f = TABLES_DIR / "exp7_operational_thresholds.csv"
    if f.exists():
        cc = pd.read_csv(f)
        st.plotly_chart(px.line(cc, x="capacity_pct",
                                y=["measured_precision", "measured_recall"],
                                title="Detection performance vs review capacity"),
                        use_container_width=True)

# --- Page 8 ---------------------------------------------------------------
else:
    st.title("Reproducibility")
    banner()
    settings = load_settings()
    st.subheader("Configuration")
    st.json(settings.to_dict())

    for label, path in [("Run summary", RESULTS_DIR / "run_summary.json"),
                        ("Generation metadata", SYNTHETIC_DIR / "generation_metadata.json")]:
        if Path(path).exists():
            st.subheader(label)
            st.json(json.loads(Path(path).read_text()))

    from medicaidguard.data.metadata import run_metadata
    st.subheader("Environment")
    st.json(run_metadata())
