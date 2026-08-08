"""
Palo Alto Networks — Machine Learning-Based Employee Attrition Prediction
and Risk Scoring System.
Unified Mentor Data Analyst Internship Project.
Author: M. Sandeep Reddy
"""

import base64
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict_utils import engineer_features, encode_and_scale, predict_risk

# =====================================================================
# PAGE CONFIG
# =====================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo.png")
MARK_PATH = os.path.join(APP_DIR, "assets", "mark.png")
HERO_PATH = os.path.join(APP_DIR, "assets", "hero_banner.jpg")

st.set_page_config(
    page_title="Palo Alto Networks | Attrition Risk Intelligence",
    page_icon=MARK_PATH if os.path.exists(MARK_PATH) else "🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ORANGE = "#FA582D"
ORANGE_LIGHT = "#FF7A52"
INK = "#141414"
CARD_BG = "#1B1B1B"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"

RISK_COLORS = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}

PLOTLY_TEMPLATE = "plotly_dark"


def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E5E5"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def img_to_b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


LOGO_B64 = img_to_b64(LOGO_PATH)
HERO_B64 = img_to_b64(HERO_PATH)

# =====================================================================
# GLOBAL CSS
# =====================================================================
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0E0E10; color: #EDEDED; }}
    section[data-testid="stSidebar"] {{ background-color: {INK}; border-right: 1px solid #2A2A2A; }}
    section[data-testid="stSidebar"] * {{ color: #EDEDED !important; }}

    .hero {{
        position: relative;
        height: 210px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 24px;
        border: 1px solid #2E2620;
        box-shadow: 0 6px 32px rgba(250,88,45,0.14);
    }}
    .hero-photo {{
        position: absolute; top:0; left:0; right:0; bottom:0; width:100%; height:100%;
        background-image: url('data:image/jpeg;base64,{HERO_B64}');
        background-size: cover; background-position: center 45%; background-repeat: no-repeat;
    }}
    .hero-scrim {{
        position: absolute; top:0; left:0; right:0; bottom:0;
        background: linear-gradient(180deg, rgba(2,2,2,0.05) 0%, rgba(2,2,2,0.15) 55%, rgba(8,7,6,0.92) 100%);
    }}
    .hero-textblock {{
        position: absolute; left: 26px; bottom: 20px; right: 26px; z-index: 3;
    }}
    .hero-textblock h1 {{
        font-size: 25px; margin:0; color: #FFFFFF; font-weight: 800; letter-spacing: 0.3px;
        text-shadow: 0 2px 12px rgba(0,0,0,0.9);
    }}
    .hero-textblock p {{
        margin:8px 0 0 0; color: {ORANGE_LIGHT}; font-size: 12.5px; letter-spacing: 1.6px; text-transform: uppercase;
        font-weight: 600; text-shadow: 0 2px 8px rgba(0,0,0,0.9);
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #2A2A2A;
        border-left: 3px solid {ORANGE};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.2s ease, background-color 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px) scale(1.015);
        box-shadow: 0 10px 22px rgba(250,88,45,0.28);
        border-color: {ORANGE};
        border-left: 3px solid {ORANGE_LIGHT};
        background-color: #212121;
    }}
    div[data-testid="stMetric"]:active {{
        transform: translateY(-1px) scale(0.965);
        box-shadow: 0 3px 8px rgba(250,88,45,0.4) inset;
        border-left: 3px solid #FF9873;
        transition: transform 0.06s ease, box-shadow 0.06s ease;
    }}
    div[data-testid="stMetricValue"] {{ transition: color 0.15s ease; }}
    div[data-testid="stMetric"]:hover div[data-testid="stMetricValue"] {{ color: #FF9873 !important; }}

    .section-card {{
        background-color: {CARD_BG};
        border: 1px solid #2A2A2A;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }}
    .section-card h4 {{ color: {ORANGE_LIGHT}; margin-top:0; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG}; border-radius: 8px 8px 0 0; color: #C9C9C9; padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ORANGE} !important; color: #111 !important; font-weight: 600;
    }}
    .risk-badge {{
        display:inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight:700;
    }}
    .reco-box {{
        background: linear-gradient(135deg, #201810 0%, #1B1B1B 100%);
        border-left: 3px solid {ORANGE};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 14.5px;
        color: #EDEDED;
    }}
    div[data-testid="stPopoverBody"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {ORANGE} !important;
    }}
    button[data-testid="stPopoverButton"] {{
        background-color: transparent !important;
        border: 1px solid #3A3A3A !important;
        color: {ORANGE_LIGHT} !important;
        font-size: 12px !important;
        padding: 2px 10px !important;
        margin-top: -6px !important;
    }}
    button[data-testid="stPopoverButton"]:hover {{
        border-color: {ORANGE} !important; color: #FFF !important;
    }}
    .footer-note {{ color:#7A7A7A; font-size:12px; text-align:center; margin-top: 30px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# DATA + MODEL LOADING
# =====================================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(APP_DIR, "data", "best_model.pkl"))
    scaler = joblib.load(os.path.join(APP_DIR, "data", "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(APP_DIR, "data", "label_encoders.pkl"))
    return model, scaler, label_encoders


@st.cache_data
def load_data():
    scored_path = os.path.join(APP_DIR, "data", "employees_scored.csv")
    raw_path = os.path.join(APP_DIR, "data", "Palo_Alto_Networks.csv")
    model, scaler, label_encoders = load_model_artifacts()

    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path)
    else:
        df = pd.read_csv(raw_path)
        proba, cat = predict_risk(df, model, label_encoders, scaler)
        df["AttritionProbability"] = proba
        df["RiskCategory"] = cat

    if "EmployeeID" not in df.columns:
        df.insert(0, "EmployeeID", [f"PA-{i+1:04d}" for i in range(len(df))])
    return df


@st.cache_data
def load_model_comparison():
    path = os.path.join(APP_DIR, "..", "data", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


model, scaler, label_encoders = load_model_artifacts()
df = load_data()
model_comparison = load_model_comparison()

# =====================================================================
# SIDEBAR — brand + filters (User Capabilities)
# =====================================================================
with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:6px;padding:10px 0;'>"
            f"<img src='data:image/png;base64,{LOGO_B64}' style='width:180px;'/></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<p style='text-align:center;color:{ORANGE_LIGHT};letter-spacing:1px;font-size:12px;"
        f"text-transform:uppercase;margin-top:-6px;'>Attrition Risk Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🧩 User Filters")

    departments = sorted(df["Department"].unique())
    sel_departments = st.multiselect("Department", departments, default=departments)

    roles = sorted(df["JobRole"].unique())
    sel_roles = st.multiselect("Job Role", roles, default=roles)

    st.markdown("### ⚙️ Risk Threshold")
    risk_threshold = st.slider(
        "High-risk cutoff (attrition probability %)", min_value=30, max_value=90, value=60, step=5,
        help="Employees at or above this probability are flagged High Risk throughout the dashboard.",
    )

    st.markdown("### 👤 Employee Lookup")
    employee_options = df["EmployeeID"].tolist()
    sel_employee = st.selectbox("Employee ID", employee_options, key="sidebar_employee_select")

    st.markdown("---")
    st.caption(f"Employees in dataset: **{len(df):,}** total")

# --- Apply filters ---
fdf = df[df["Department"].isin(sel_departments) & df["JobRole"].isin(sel_roles)].copy()

if fdf.empty:
    st.warning("No employees match the selected filters. Please broaden your selection.")
    st.stop()

# Re-derive risk category from the live, user-adjustable threshold
def categorize(p, cutoff):
    cutoff_frac = cutoff / 100
    med_cutoff = cutoff_frac * 0.5
    if p >= cutoff_frac:
        return "High Risk"
    elif p >= med_cutoff:
        return "Medium Risk"
    return "Low Risk"


fdf["RiskCategoryLive"] = fdf["AttritionProbability"].apply(lambda p: categorize(p, risk_threshold))

# =====================================================================
# HERO HEADER
# =====================================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-photo"></div>
        <div class="hero-scrim"></div>
        <div class="hero-textblock">
            <h1>Employee Attrition Prediction & Risk Scoring</h1>
            <p>Palo Alto Networks · Predictive HR Intelligence Dashboard</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# TOP-LINE INTERACTIVE KPI GRID
# =====================================================================
total_employees = len(fdf)
actual_attrition_rate = fdf["Attrition"].mean() * 100
avg_probability = fdf["AttritionProbability"].mean() * 100
high_risk_ct = (fdf["RiskCategoryLive"] == "High Risk").sum()
medium_risk_ct = (fdf["RiskCategoryLive"] == "Medium Risk").sum()
low_risk_ct = (fdf["RiskCategoryLive"] == "Low Risk").sum()
avg_income = fdf["MonthlyIncome"].mean()
overtime_share = (fdf["OverTime"] == "Yes").mean() * 100

base_total = len(df)
base_attrition_rate = df["Attrition"].mean() * 100
base_avg_prob = df["AttritionProbability"].mean() * 100
base_high_risk = (df["AttritionProbability"] >= risk_threshold / 100).sum()


def pct_delta(current, base):
    if base:
        return (current - base) / abs(base) * 100
    return 0.0


def gauge_fig(value, title, max_val=100, zones=((0, 50, "#10B981"), (50, 70, "#F59E0B"), (70, 100, "#EF4444")), suffix="%"):
    def _hex_to_rgba(hex_color, alpha=0.28):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": ORANGE_LIGHT, "size": 30}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#888", "tickfont": {"color": "#999"}},
                "bar": {"color": ORANGE_LIGHT, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [z[0], z[1]], "color": _hex_to_rgba(z[2])} for z in zones],
            },
            title={"text": title, "font": {"size": 13, "color": "#CCCCCC"}},
        )
    )
    fig.update_layout(height=210, margin=dict(l=25, r=25, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#EEE"})
    return fig


kpi_row1 = st.columns(4)
kpi_row2 = st.columns(4)

with kpi_row1[0]:
    st.metric("Total Employees", f"{total_employees:,}", f"{pct_delta(total_employees, base_total):+.1f}% vs full workforce",
               help="Employees matching the current department/role filters.")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.caption("Employee count by department (current filters)")
        bd = fdf["Department"].value_counts().reset_index()
        bd.columns = ["Department", "Employees"]
        fig = px.bar(bd, x="Department", y="Employees", color_discrete_sequence=[ORANGE_LIGHT], text_auto=True)
        st.plotly_chart(style_fig(fig, height=240), width="stretch", key="pop_total")

with kpi_row1[1]:
    st.metric("Historical Attrition Rate", f"{actual_attrition_rate:.1f}%", f"{actual_attrition_rate - base_attrition_rate:+.1f} pp vs full workforce",
               delta_color="inverse", help="Share of employees in this filter who have already left (ground-truth label).")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.plotly_chart(style_fig(gauge_fig(actual_attrition_rate, "Historical Attrition", max_val=50,
                                             zones=((0, 15, "#10B981"), (15, 30, "#F59E0B"), (30, 50, "#EF4444"))), height=230),
                         width="stretch", key="pop_attr_gauge")

with kpi_row1[2]:
    st.metric("Avg. Predicted Risk", f"{avg_probability:.1f}%", f"{avg_probability - base_avg_prob:+.1f} pp vs full workforce",
               delta_color="inverse", help="Mean model-predicted attrition probability across the filtered employees.")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.plotly_chart(style_fig(gauge_fig(avg_probability, "Avg. Predicted Risk", max_val=100), height=230),
                         width="stretch", key="pop_prob_gauge")

with kpi_row1[3]:
    st.metric("High-Risk Employees", f"{high_risk_ct:,}", f"{high_risk_ct - base_high_risk:+,} vs full workforce",
               delta_color="inverse", help=f"Employees at or above the {risk_threshold}% risk threshold set in the sidebar.")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.caption("Top 5 highest-risk employees (current filters)")
        bd = fdf.nlargest(5, "AttritionProbability")[["EmployeeID", "JobRole", "AttritionProbability"]].copy()
        bd["AttritionProbability"] = (bd["AttritionProbability"] * 100).round(1)
        bd.columns = ["Employee ID", "Role", "Risk (%)"]
        st.dataframe(bd, width="stretch", hide_index=True)

with kpi_row2[0]:
    st.metric("Avg. Monthly Income", f"${avg_income:,.0f}", f"{pct_delta(avg_income, df['MonthlyIncome'].mean()):+.1f}% vs full workforce",
               help="Mean monthly income across the filtered employees.")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.caption("Avg. monthly income by department")
        bd = fdf.groupby("Department")["MonthlyIncome"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(bd, x="Department", y="MonthlyIncome", color_discrete_sequence=[ORANGE_LIGHT], text_auto=",.0f")
        st.plotly_chart(style_fig(fig, height=240), width="stretch", key="pop_income")

with kpi_row2[1]:
    st.metric("Overtime Share", f"{overtime_share:.1f}%", f"{overtime_share - (df['OverTime']=='Yes').mean()*100:+.1f} pp vs full workforce",
               delta_color="inverse", help="Share of filtered employees currently working overtime — the single strongest attrition driver identified in this analysis.")
    with st.popover("🔍 Breakdown", width="stretch"):
        st.plotly_chart(style_fig(gauge_fig(overtime_share, "Overtime Share"), height=230), width="stretch", key="pop_ot_gauge")

with kpi_row2[2]:
    medium_pct = medium_risk_ct / total_employees * 100 if total_employees else 0
    st.metric("Medium-Risk Employees", f"{medium_risk_ct:,}", f"{medium_pct:.1f}% of filtered workforce",
               help="Employees between the medium and high risk cutoffs — worth monitoring.")
    with st.popover("🔍 Breakdown", width="stretch"):
        risk_dist = fdf["RiskCategoryLive"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        rd = pd.DataFrame({"Risk": risk_dist.index, "Employees": risk_dist.values})
        fig = px.bar(rd, x="Risk", y="Employees", color="Risk", color_discrete_map=RISK_COLORS, text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=240), width="stretch", key="pop_medium")

with kpi_row2[3]:
    if model_comparison is not None:
        best_row = model_comparison.loc[model_comparison["ROC-AUC"].idxmax()]
        model_name = model_comparison["ROC-AUC"].idxmax()
        auc_val = best_row["ROC-AUC"] * 100
    else:
        model_name, auc_val = "Model", 0
    st.metric("Model ROC-AUC", f"{auc_val:.1f}%", model_name,
               help="Held-out test-set ROC-AUC of the deployed model — see the Model Performance tab for full validation metrics.")
    with st.popover("🔍 Breakdown", width="stretch"):
        if model_comparison is not None:
            mc = model_comparison.reset_index()
            mc.columns = ["Model"] + list(model_comparison.columns)
            fig = px.bar(mc, x="Model", y="ROC-AUC", color_discrete_sequence=[ORANGE_LIGHT], text_auto=".3f")
            st.plotly_chart(style_fig(fig, height=240), width="stretch", key="pop_model")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# CACHED: full-dataset scaled features + SHAP explainer (for explainability tabs)
# =====================================================================
@st.cache_resource
def get_explainer_and_scaled(_model, _scaler, _label_encoders, data_hash):
    import shap
    engineered = engineer_features(df)
    scaled = encode_and_scale(engineered, _label_encoders, _scaler)
    model_type = type(_model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(_model)
    else:
        explainer = shap.LinearExplainer(_model, scaled)
    return explainer, scaled


explainer, X_scaled_full = get_explainer_and_scaled(model, scaler, label_encoders, len(df))
feature_cols = list(scaler.feature_names_in_)


def get_shap_values(scaled_subset):
    sv = explainer(scaled_subset).values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


# =====================================================================
# AUTOMATED BUSINESS RECOMMENDATIONS (data-driven, generated from current filters)
# =====================================================================
def generate_recommendations(fdf, threshold):
    recs = []
    high_risk_pct = (fdf["RiskCategoryLive"] == "High Risk").mean() * 100

    if high_risk_pct > 20:
        recs.append(
            f"🔴 **{high_risk_pct:.0f}% of the filtered workforce is High Risk** (≥{threshold}% predicted "
            f"probability). This exceeds a healthy 20% ceiling — prioritize retention conversations for this "
            f"group immediately."
        )
    elif high_risk_pct > 10:
        recs.append(
            f"🟠 **{high_risk_pct:.0f}% of the filtered workforce is High Risk.** Monitor closely and begin "
            f"proactive check-ins with the highest-probability individuals."
        )
    else:
        recs.append(f"🟢 High-risk share is contained at {high_risk_pct:.0f}% — maintain current retention practices.")

    ot_high_risk = fdf[fdf["OverTime"] == "Yes"]["AttritionProbability"].mean() * 100
    ot_low_risk = fdf[fdf["OverTime"] == "No"]["AttritionProbability"].mean() * 100
    if not np.isnan(ot_high_risk) and not np.isnan(ot_low_risk) and ot_high_risk - ot_low_risk > 10:
        recs.append(
            f"⏱️ Employees working overtime show **{ot_high_risk:.0f}% average predicted risk** vs. "
            f"**{ot_low_risk:.0f}%** for those who don't — a {ot_high_risk - ot_low_risk:.0f} point gap. "
            f"Workload rebalancing is the single most actionable lever available."
        )

    dept_risk = fdf.groupby("Department")["AttritionProbability"].mean().sort_values(ascending=False)
    if len(dept_risk) > 1:
        top_dept = dept_risk.index[0]
        recs.append(
            f"🏢 **{top_dept}** has the highest average predicted risk ({dept_risk.iloc[0]*100:.0f}%) among "
            f"filtered departments — consider targeting retention budget here first."
        )

    low_engagement = fdf[fdf["EngagementScore"] < 2.0]
    if len(low_engagement) > 0:
        recs.append(
            f"💬 **{len(low_engagement)} employees** have an Engagement Score below 2.0/4 — this composite "
            f"(job satisfaction + environment + relationships + work-life balance) is a strong leading "
            f"indicator worth a targeted pulse survey."
        )

    promo_stall = fdf[fdf["PromotionDelay"] >= 5]
    if len(promo_stall) > 0:
        recs.append(
            f"📈 **{len(promo_stall)} employees** have gone 5+ years without a promotion relative to their "
            f"tenure — review promotion cadence for this group."
        )

    return recs

# =====================================================================
# TABS — Dashboard Modules
# =====================================================================
tab_exec, tab_risk, tab_profile, tab_dept, tab_explain, tab_model = st.tabs(
    [
        "📊 Executive Overview",
        "🛡️ Attrition Risk Dashboard",
        "🧬 Employee Risk Profile",
        "🗂️ Department-Level Risk View",
        "🧠 Explainability Panel",
        "🤖 Model Performance",
    ]
)

# ---------------------------------------------------------------------
# TAB 1 — EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------
with tab_exec:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Risk Distribution vs. Historical Attrition")
        comp_df = pd.DataFrame({
            "Metric": ["Historical Attrition Rate", "Avg. Predicted Risk", "High-Risk Share"],
            "Value (%)": [actual_attrition_rate, avg_probability, high_risk_ct / total_employees * 100],
        })
        fig = px.bar(comp_df, x="Metric", y="Value (%)", color="Metric",
                     color_discrete_sequence=[ORANGE, ORANGE_LIGHT, RED], text_auto=".1f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=360), width="stretch")
    with c2:
        st.markdown("#### Risk Category Mix")
        risk_dist = fdf["RiskCategoryLive"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        fig = px.pie(names=risk_dist.index, values=risk_dist.values, hole=0.5,
                     color=risk_dist.index, color_discrete_map=RISK_COLORS)
        st.plotly_chart(style_fig(fig, height=360), width="stretch")

    st.markdown("#### 💡 Automated Business Recommendations")
    st.caption("Generated live from the current filter selection and model outputs — not static analyst commentary.")
    for rec in generate_recommendations(fdf, risk_threshold):
        st.markdown(f"<div class='reco-box'>{rec}</div>", unsafe_allow_html=True)

    st.markdown("#### 📥 Downloadable Report")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_data = fdf[[
            "EmployeeID", "Department", "JobRole", "Age", "MonthlyIncome", "OverTime",
            "AttritionProbability", "RiskCategoryLive"
        ]].rename(columns={"RiskCategoryLive": "RiskCategory"}).copy()
        csv_data["AttritionProbability"] = (csv_data["AttritionProbability"] * 100).round(1)
        st.download_button(
            "⬇️ Download Filtered Employee Risk Report (CSV)",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name="attrition_risk_report.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl2:
        summary_lines = [
            "PALO ALTO NETWORKS — ATTRITION RISK SUMMARY REPORT",
            f"Filtered workforce: {total_employees:,} employees",
            f"Historical attrition rate: {actual_attrition_rate:.1f}%",
            f"Average predicted attrition risk: {avg_probability:.1f}%",
            f"High-risk employees (>={risk_threshold}%): {high_risk_ct:,}",
            f"Medium-risk employees: {medium_risk_ct:,}",
            f"Low-risk employees: {low_risk_ct:,}",
            "",
            "AUTOMATED RECOMMENDATIONS:",
        ] + [f"- {r}" for r in generate_recommendations(fdf, risk_threshold)]
        st.download_button(
            "⬇️ Download Executive Summary (TXT)",
            data="\n".join(summary_lines).encode("utf-8"),
            file_name="executive_summary.txt",
            mime="text/plain",
            width="stretch",
        )

# ---------------------------------------------------------------------
# TAB 2 — ATTRITION RISK DASHBOARD (required module)
# ---------------------------------------------------------------------
with tab_risk:
    st.markdown(
        "<div class='section-card'><h4>Overall Risk Distribution</h4>"
        "<p style='color:#B9B9B9;font-size:13.5px;'>Every employee is scored with an <b>Attrition Probability (0-100%)</b> "
        "by the trained model, then bucketed into a <b>Risk Category</b>: "
        "<span style='color:#10B981;font-weight:600;'>Low Risk (&lt; 30%)</span>, "
        "<span style='color:#F59E0B;font-weight:600;'>Medium Risk (30-60%)</span>, "
        "<span style='color:#EF4444;font-weight:600;'>High Risk (&gt; 60%)</span> — "
        "using the live threshold set in the sidebar.</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Risk Category Counts")
        risk_dist = fdf["RiskCategoryLive"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
        rd = pd.DataFrame({"Risk Category": risk_dist.index, "Employees": risk_dist.values})
        fig = px.bar(rd, x="Risk Category", y="Employees", color="Risk Category",
                     color_discrete_map=RISK_COLORS, text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=380), width="stretch")

    with c2:
        st.markdown("#### Attrition Probability Distribution")
        fig = px.histogram(fdf, x="AttritionProbability", nbins=30, color_discrete_sequence=[ORANGE_LIGHT])
        fig.add_vline(x=risk_threshold / 100, line_dash="dash", line_color=RED, annotation_text="High-risk cutoff")
        fig.add_vline(x=risk_threshold / 200, line_dash="dot", line_color=AMBER, annotation_text="Medium-risk cutoff")
        fig.update_xaxes(tickformat=".0%", title="Predicted Attrition Probability")
        fig.update_yaxes(title="Employees")
        st.plotly_chart(style_fig(fig, height=380), width="stretch")

    st.markdown("#### High-Risk Employee Counts by Segment")
    seg_choice = st.radio("Segment by:", ["Department", "JobRole", "Gender", "MaritalStatus"], horizontal=True)
    high_risk_df = fdf[fdf["RiskCategoryLive"] == "High Risk"]
    if len(high_risk_df) > 0:
        seg_counts = high_risk_df[seg_choice].value_counts().sort_values(ascending=False)
        fig = px.bar(x=seg_counts.index, y=seg_counts.values, color_discrete_sequence=[RED], text_auto=True)
        fig.update_layout(xaxis_title=seg_choice, yaxis_title="High-Risk Employees")
        st.plotly_chart(style_fig(fig, height=360), width="stretch")
    else:
        st.info("No employees in the current filter meet the High Risk threshold.")

    st.markdown("#### Risk vs. Key Drivers")
    driver_choice = st.selectbox("Compare risk against:", ["OverTime", "WorkLifeBalance", "JobSatisfaction", "EnvironmentSatisfaction", "StockOptionLevel"])
    fig = px.box(fdf, x=driver_choice, y="AttritionProbability", color=driver_choice,
                 color_discrete_sequence=px.colors.sequential.Oranges_r)
    fig.update_yaxes(tickformat=".0%", title="Predicted Attrition Probability")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, height=380), width="stretch")

# ---------------------------------------------------------------------
# TAB 3 — EMPLOYEE RISK PROFILE (required module)
# ---------------------------------------------------------------------
with tab_profile:
    st.markdown("#### Individual Employee Risk Profile")
    profile_employee = st.selectbox(
        "Select an Employee ID", fdf["EmployeeID"].tolist(),
        index=fdf["EmployeeID"].tolist().index(sel_employee) if sel_employee in fdf["EmployeeID"].tolist() else 0,
        key="profile_employee_select",
    )
    emp_row = fdf[fdf["EmployeeID"] == profile_employee].iloc[0]
    emp_idx = fdf[fdf["EmployeeID"] == profile_employee].index[0]

    risk_color = RISK_COLORS.get(emp_row["RiskCategoryLive"], ORANGE)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Attrition Probability", f"{emp_row['AttritionProbability']*100:.1f}%")
    p2.markdown(
        f"<div style='padding-top:8px;'><span class='risk-badge' style='background:{risk_color}22;"
        f"color:{risk_color};border:1px solid {risk_color};'>{emp_row['RiskCategoryLive']}</span></div>",
        unsafe_allow_html=True,
    )
    p3.metric("Department", emp_row["Department"])
    p4.metric("Job Role", emp_row["JobRole"])

    st.markdown("---")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Monthly Income", f"${emp_row['MonthlyIncome']:,.0f}")
    d2.metric("Years at Company", f"{emp_row['YearsAtCompany']}")
    d3.metric("Overtime", emp_row["OverTime"])
    d4.metric("Work-Life Balance", f"{emp_row['WorkLifeBalance']}/4")

    st.markdown("#### Key Contributing Factors (Individual Reason Codes)")
    st.caption("SHAP values for this specific employee — positive values push risk up, negative values push risk down.")

    emp_scaled = X_scaled_full.loc[[emp_idx]]
    emp_shap = get_shap_values(emp_scaled)[0]
    contrib = pd.Series(emp_shap, index=feature_cols).sort_values(key=np.abs, ascending=False).head(8)
    contrib_df = pd.DataFrame({
        "Feature": contrib.index,
        "SHAP Value": contrib.values,
        "Employee's Value": [emp_row.get(f, X_scaled_full.loc[emp_idx, f]) if f in emp_row.index else X_scaled_full.loc[emp_idx, f] for f in contrib.index],
    })
    fig = px.bar(
        contrib_df.sort_values("SHAP Value"), x="SHAP Value", y="Feature", orientation="h",
        color="SHAP Value", color_continuous_scale=["#10B981", "#3B3B3B", RED],
        color_continuous_midpoint=0,
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(style_fig(fig, height=380), width="stretch")

    top_factor = contrib.index[0]
    direction = "increases" if contrib.iloc[0] > 0 else "decreases"
    st.info(
        f"For **{profile_employee}**, the strongest individual driver is **{top_factor}**, which "
        f"**{direction}** predicted attrition risk. Use this alongside the What-If tool in the "
        f"Explainability Panel to explore how changing this factor might shift the prediction.",
        icon="🧭",
    )

# ---------------------------------------------------------------------
# TAB 4 — DEPARTMENT-LEVEL RISK VIEW (required module)
# ---------------------------------------------------------------------
with tab_dept:
    st.markdown("#### Aggregated Risk by Department & Role")

    dept_agg = fdf.groupby("Department").agg(
        Employees=("EmployeeID", "count"),
        AvgRisk=("AttritionProbability", "mean"),
        HighRiskCount=("RiskCategoryLive", lambda s: (s == "High Risk").sum()),
        HistoricalAttrition=("Attrition", "mean"),
    ).reset_index()
    dept_agg["AvgRisk"] = (dept_agg["AvgRisk"] * 100).round(1)
    dept_agg["HistoricalAttrition"] = (dept_agg["HistoricalAttrition"] * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(dept_agg.sort_values("AvgRisk", ascending=False), x="Department", y="AvgRisk",
                     color="AvgRisk", color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"], text_auto=".1f")
        fig.update_layout(title="Average Predicted Risk by Department", coloraxis_showscale=False, yaxis_title="Avg. Risk (%)")
        st.plotly_chart(style_fig(fig, height=380), width="stretch")
    with c2:
        fig = px.bar(dept_agg.sort_values("HighRiskCount", ascending=False), x="Department", y="HighRiskCount",
                     color_discrete_sequence=[RED], text_auto=True)
        fig.update_layout(title="High-Risk Employee Count by Department", yaxis_title="High-Risk Employees")
        st.plotly_chart(style_fig(fig, height=380), width="stretch")

    st.markdown("#### Risk Heatmap: Department × Job Role")
    heat_data = fdf.pivot_table(values="AttritionProbability", index="JobRole", columns="Department", aggfunc="mean") * 100
    fig = px.imshow(
        heat_data.round(1), text_auto=".1f", color_continuous_scale=["#141414", "#8a4a2a", ORANGE_LIGHT],
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Risk %"))
    st.plotly_chart(style_fig(fig, height=440), width="stretch")

    st.markdown("#### Department Summary Table")
    display_dept = dept_agg.rename(columns={
        "AvgRisk": "Avg. Predicted Risk (%)", "HighRiskCount": "High-Risk Employees",
        "HistoricalAttrition": "Historical Attrition Rate (%)",
    })
    st.dataframe(
        display_dept, width="stretch", hide_index=True,
        column_config={
            "Avg. Predicted Risk (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Historical Attrition Rate (%)": st.column_config.ProgressColumn(min_value=0, max_value=50, format="%.1f%%"),
        },
    )

# ---------------------------------------------------------------------
# TAB 5 — EXPLAINABILITY PANEL (required module: feature importance + what-if)
# ---------------------------------------------------------------------
with tab_explain:
    st.markdown("#### Global Feature Importance")
    st.caption("What drives the model's predictions across the entire workforce, not just one employee.")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp_label = "Importance"
    else:
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
        imp_label = "|Coefficient|"

    top_imp = importances.head(12).sort_values()
    fig = px.bar(x=top_imp.values, y=top_imp.index, orientation="h", color_discrete_sequence=[ORANGE_LIGHT])
    fig.update_layout(title=f"Top 12 Features — {type(model).__name__}", xaxis_title=imp_label, yaxis_title="")
    st.plotly_chart(style_fig(fig, height=440), width="stretch")

    st.markdown("#### SHAP Impact Distribution")
    st.caption("Sampled from the current filtered workforce. Red points push risk up; green points push risk down.")
    sample_n = min(200, len(fdf))
    sample_idx = fdf.sample(sample_n, random_state=42).index if len(fdf) > sample_n else fdf.index
    sample_scaled = X_scaled_full.loc[sample_idx]
    sample_shap = get_shap_values(sample_scaled)

    mean_abs_shap = pd.Series(np.abs(sample_shap).mean(axis=0), index=feature_cols).sort_values(ascending=False).head(10)
    fig2 = px.bar(x=mean_abs_shap.values, y=mean_abs_shap.index, orientation="h", color_discrete_sequence=[ORANGE])
    fig2.update_layout(title="Mean |SHAP Value| — Current Filtered Sample", xaxis_title="Mean |SHAP Value|", yaxis_title="")
    st.plotly_chart(style_fig(fig2, height=380), width="stretch")

    st.markdown("---")
    st.markdown("#### 🧪 What-If Scenario Exploration")
    st.caption("Adjust an employee's profile below and see the model's predicted attrition risk update live.")

    whatif_employee = st.selectbox("Start from employee:", fdf["EmployeeID"].tolist(), key="whatif_base")
    base_row = fdf[fdf["EmployeeID"] == whatif_employee].iloc[0]

    w1, w2, w3 = st.columns(3)
    with w1:
        wi_overtime = st.selectbox("OverTime", ["No", "Yes"], index=0 if base_row["OverTime"] == "No" else 1)
        wi_income = st.slider("Monthly Income ($)", 1000, 20000, int(base_row["MonthlyIncome"]), step=100)
        wi_wlb = st.slider("Work-Life Balance (1-4)", 1, 4, int(base_row["WorkLifeBalance"]))
    with w2:
        wi_jobsat = st.slider("Job Satisfaction (1-4)", 1, 4, int(base_row["JobSatisfaction"]))
        wi_envsat = st.slider("Environment Satisfaction (1-4)", 1, 4, int(base_row["EnvironmentSatisfaction"]))
        wi_years = st.slider("Years at Company", 0, 40, int(base_row["YearsAtCompany"]))
    with w3:
        wi_promo = st.slider("Years Since Last Promotion", 0, 15, int(base_row["YearsSinceLastPromotion"]))
        wi_stock = st.slider("Stock Option Level (0-3)", 0, 3, int(base_row["StockOptionLevel"]))
        wi_distance = st.slider("Distance From Home (km)", 0, 30, int(base_row["DistanceFromHome"]))

    whatif_row = base_row.to_frame().T.copy()
    whatif_row["OverTime"] = wi_overtime
    whatif_row["MonthlyIncome"] = wi_income
    whatif_row["WorkLifeBalance"] = wi_wlb
    whatif_row["JobSatisfaction"] = wi_jobsat
    whatif_row["EnvironmentSatisfaction"] = wi_envsat
    whatif_row["YearsAtCompany"] = wi_years
    whatif_row["YearsSinceLastPromotion"] = wi_promo
    whatif_row["StockOptionLevel"] = wi_stock
    whatif_row["DistanceFromHome"] = wi_distance

    raw_cols_needed = [c for c in df.columns if c not in ("EmployeeID", "AttritionProbability", "RiskCategory")]
    whatif_proba, whatif_cat = predict_risk(whatif_row[raw_cols_needed], model, label_encoders, scaler)

    original_proba = base_row["AttritionProbability"] * 100
    new_proba = whatif_proba[0] * 100
    delta = new_proba - original_proba

    wr1, wr2, wr3 = st.columns(3)
    wr1.metric("Original Risk", f"{original_proba:.1f}%")
    wr2.metric("Scenario Risk", f"{new_proba:.1f}%", f"{delta:+.1f} pp", delta_color="inverse")
    wr3.markdown(
        f"<div style='padding-top:8px;'><span class='risk-badge' style='background:{RISK_COLORS[whatif_cat[0]]}22;"
        f"color:{RISK_COLORS[whatif_cat[0]]};border:1px solid {RISK_COLORS[whatif_cat[0]]};'>{whatif_cat[0]}</span></div>",
        unsafe_allow_html=True,
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=["Original", "Scenario"], y=[original_proba, new_proba],
                           marker_color=[ORANGE, "#4C6EF5" if delta < 0 else RED], text=[f"{original_proba:.1f}%", f"{new_proba:.1f}%"], textposition="auto"))
    fig3.update_layout(title="Original vs. Scenario Predicted Risk", yaxis_title="Attrition Probability (%)")
    st.plotly_chart(style_fig(fig3, height=320), width="stretch")

    if delta < -5:
        st.success(f"This scenario reduces predicted attrition risk by {abs(delta):.1f} percentage points — a meaningful improvement.", icon="✅")
    elif delta > 5:
        st.warning(f"This scenario increases predicted attrition risk by {delta:.1f} percentage points.", icon="⚠️")
    else:
        st.info("This scenario has a modest effect on predicted risk (within ±5 percentage points).", icon="ℹ️")

# ---------------------------------------------------------------------
# TAB 6 — MODEL PERFORMANCE (validation metrics + clear metric definitions)
# ---------------------------------------------------------------------
with tab_model:
    st.markdown("#### Model Validation — Held-Out Test Set Performance")
    st.caption(
        "All metrics below are computed on a stratified 20% test split that the model never saw during "
        "training or SMOTE resampling — not training-set fit. This directly reflects real-world deployment performance."
    )

    if model_comparison is not None:
        mc_display = model_comparison.copy()
        mc_display = (mc_display * 100).round(1)
        st.dataframe(
            mc_display, width="stretch",
            column_config={c: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%") for c in mc_display.columns},
        )

        mc_melt = model_comparison.reset_index()
        mc_melt.columns = ["Model"] + list(model_comparison.columns)
        fig = px.bar(mc_melt.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                     x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=[ORANGE, "#4C6EF5", "#10B981", "#F59E0B", "#EF4444"])
        fig.update_layout(xaxis_title="Model", yaxis_title="Score (0-1)")
        st.plotly_chart(style_fig(fig, height=400), width="stretch")

        best_name = model_comparison["ROC-AUC"].idxmax()
        st.success(f"**Deployed model: {best_name}** — selected as the best performer by ROC-AUC on the held-out test set.", icon="🏆")

    st.markdown("#### Metric Definitions")
    st.markdown(
        """
        <div class='section-card'>
        <p style='font-size:14px;color:#DDD;line-height:1.7;'>
        <b>Accuracy</b> — overall share of predictions (both "left" and "stayed") the model got right.<br>
        <b>Precision</b> — of the employees the model flagged as likely to leave, what share actually left. High precision means fewer false alarms.<br>
        <b>Recall</b> — of the employees who actually left, what share the model correctly flagged in advance. High recall means fewer missed at-risk employees — usually the priority metric for retention use cases.<br>
        <b>F1-Score</b> — the harmonic mean of Precision and Recall, balancing both concerns into a single number.<br>
        <b>ROC-AUC</b> — the model's overall ability to rank employees by risk correctly, across every possible threshold (1.0 = perfect, 0.5 = random guessing).<br><br>
        <b>Risk Category thresholds</b> — <span style='color:#10B981;'>Low Risk: &lt;30% predicted probability</span>,
        <span style='color:#F59E0B;'>Medium Risk: 30-60%</span>,
        <span style='color:#EF4444;'>High Risk: &gt;60%</span> (the High-Risk cutoff is adjustable live via the sidebar slider,
        which recalculates every chart and KPI on this dashboard).
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Why Precision/Recall Trade-offs Matter Here")
    st.info(
        "For attrition prediction, missing a true leaver (a false negative) is typically more costly than "
        "a false alarm (a false positive) — a missed high performer's resignation is far more expensive than "
        "one unnecessary retention conversation. This is why the deployed model is selected by **ROC-AUC** "
        "rather than Accuracy alone, and why HR teams using this tool should weight **Recall** heavily when "
        "deciding how aggressive to set the risk threshold in the sidebar.",
        icon="🎯",
    )

# =====================================================================
# FOOTER
# =====================================================================
st.markdown(
    """
    <div class="footer-note">
        Palo Alto Networks — Machine Learning-Based Employee Attrition Prediction and Risk Scoring System<br>
        · Prepared by M. Sandeep Reddy
    </div>
    """,
    unsafe_allow_html=True,
)
