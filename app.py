"""
CS6180 Team 26 — Clinical SOAP Note Generation
Streamlit showcase app

Run:
    streamlit run app.py

Expected file layout next to app.py:
    data/clinicalnlp_taskB_test1.csv
    data/clinicalnlp_taskB_test1_metadata.csv
    results/simple_baseline_results.json
    results/pe_results.json
    results/gvr_results.json            (optional)
    results/cd_results.json             (optional)
    results/simple_baseline_results_predictions.json   (live eval — overrides static)
    results/gvr_results_predictions.json               (live eval — overrides static)
    results/pe_results_predictions.json                (live eval — overrides static)
    results/cd_results_predictions.json                (live eval — overrides static)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS6180 Team 26 · SOAP Note Generation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 1.1rem 1.3rem; text-align: center; height: 100%;
  }
  .kpi-label { color:#94a3b8; font-size:0.72rem; font-weight:600;
               letter-spacing:0.07em; text-transform:uppercase; margin-bottom:0.3rem; }
  .kpi-value { color:#f1f5f9; font-size:1.85rem; font-weight:700; line-height:1; }
  .kpi-sub   { color:#64748b; font-size:0.72rem; margin-top:0.2rem; }

  .section-title {
    font-size:0.72rem; font-weight:700; color:#64748b;
    letter-spacing:0.09em; text-transform:uppercase;
    border-bottom:1px solid #1e293b; padding-bottom:0.4rem; margin-bottom:0.9rem;
  }

  .badge { display:inline-block; border-radius:6px;
           padding:0.15rem 0.6rem; font-size:0.7rem; font-weight:700; margin:0.1rem; }
  .badge-gvr    { background:#1e3a5f; color:#60a5fa; }
  .badge-simple { background:#1a3325; color:#34d399; }
  .badge-pe     { background:#2e1a4a; color:#c084fc; }
  .badge-cd     { background:#3b2010; color:#fb923c; }

  .note-panel {
    background:#0a0f1a; border:1px solid #1e293b; border-radius:10px;
    padding:0.9rem 1.1rem; height:430px; overflow-y:auto;
    font-family:'Courier New',monospace; font-size:0.75rem;
    line-height:1.65; color:#cbd5e1; white-space:pre-wrap;
  }
  .panel-header {
    font-size:0.65rem; font-weight:700; color:#64748b;
    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.35rem;
    display:flex; align-items:center; gap:0.5rem;
  }
  .panel-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }

  .pill {
    display:inline-block; background:#1e293b; border:1px solid #334155;
    border-radius:20px; padding:0.15rem 0.65rem;
    font-size:0.72rem; color:#94a3b8; margin:0.1rem;
  }
  .pill-blue   { background:#1e3a5f; border-color:#3b82f6; color:#93c5fd; }
  .pill-green  { background:#1a3325; border-color:#22c55e; color:#86efac; }
  .pill-red    { background:#3b1f1f; border-color:#ef4444; color:#fca5a5; }
  .pill-amber  { background:#2d1f0a; border-color:#f59e0b; color:#fcd34d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
BASE = Path(__file__).parent

TECHNIQUE_META = {
    "simple_baseline": {
        "label": "Simple Baseline",
        "short": "Simple BL",
        "badge": "badge-simple",
        "color": "#34d399",
        "desc":  "Minimal ACI-Bench paper prompt · free-text output",
    },
    "prompt_engineering": {
        "label": "Prompt Engineering",
        "short": "Prompt Eng.",
        "badge": "badge-pe",
        "color": "#c084fc",
        "desc":  "Crafted single-shot prompt → structured JSON via Pydantic",
    },
    "gvr": {
        "label": "GVR (Llama-3.1-8B)",
        "short": "GVR",
        "badge": "badge-gvr",
        "color": "#60a5fa",
        "desc":  "Generate → Validate → Retry · up to 3 LLM calls",
    },
    "constrained_decoding": {
        "label": "Constrained Decoding",
        "short": "Constrained",
        "badge": "badge-cd",
        "color": "#fb923c",
        "desc":  "instructor JSON mode · schema enforced at token level",
    },
}

# Paper baselines (from ACI-Bench paper + Team 26 slide Table 1)
# GPT-4 row is from ACI-Bench Table 6 (wo FT, ACI-Bench prompt) as shown on Team 26 slide
PAPER_BASELINES = {
    "GPT-4 (ACI-Bench)": {
        "rouge1": 0.5176, "rouge2": 0.2258, "rougeL": 0.4597, "umls": 0.5778,
        "division-subjective":          {"rouge1": 0.4120},
        "division-objective_exam":      {"rouge1": 0.5011},
        "division-objective_results":   {"rouge1": 0.3765},
        "division-assessment_and_plan": {"rouge1": 0.3816},
    },
    "ChatGPT":          {"rouge1": 0.4744, "rouge2": 0.1901, "rougeL": 0.4247, "umls": None},
    "text-davinci-003": {"rouge1": 0.4651, "rouge2": 0.1692, "rougeL": 0.4163, "umls": None},
    "text-davinci-002": {"rouge1": 0.4398, "rouge2": 0.1538, "rougeL": 0.3893, "umls": None},
}

# All Team 26 results — hardcoded from presentation Table 1 & Table 2.
# Scores stored as decimals (0–1). Division values: ROUGE-1 from Table 2.
# Full ROUGE-2/L by division available only for GVR (from its eval JSON).
# Subset breakdowns available only for Simple BL and GVR (from their eval JSONs).
# Live JSON eval files override these values if present in results/.
STATIC_EVAL = {
    "simple_baseline": {
        "ALL": {"rouge1": 0.5301, "rouge2": 0.2231, "rougeL": 0.2969, "umls": 0.5752},
        "division-subjective":          {"rouge1": 0.2842, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_exam":      {"rouge1": 0.3582, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_results":   {"rouge1": 0.3410, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-assessment_and_plan": {"rouge1": 0.4251, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-virtassist": {"rouge1": 0.4402, "rouge2": 0.2021, "rougeL": 0.2934, "umls": 0.5600},
        "dataset-virtscribe": {"rouge1": 0.4294, "rouge2": 0.2015, "rougeL": 0.2873, "umls": 0.5691},
        "dataset-aci":        {"rouge1": 0.3859, "rouge2": 0.1624, "rougeL": 0.2485, "umls": 0.5089},
    },
    "prompt_engineering": {
        "ALL": {"rouge1": 0.3896, "rouge2": 0.1732, "rougeL": 0.3615, "umls": 0.5301},
        "division-subjective":          {"rouge1": 0.3649, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_exam":      {"rouge1": 0.5453, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_results":   {"rouge1": 0.4177, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-assessment_and_plan": {"rouge1": 0.2751, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-virtassist": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-virtscribe": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-aci":        {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
    },
    "constrained_decoding": {
        "ALL": {"rouge1": 0.3391, "rouge2": 0.1520, "rougeL": 0.2382, "umls": 0.4824},
        "division-subjective":          {"rouge1": 0.2849, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_exam":      {"rouge1": 0.5199, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-objective_results":   {"rouge1": 0.4766, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "division-assessment_and_plan": {"rouge1": 0.2600, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-virtassist": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-virtscribe": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
        "dataset-aci":        {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "umls": 0.0},
    },
    "gvr": {
        "ALL": {"rouge1": 0.4082, "rouge2": 0.1801, "rougeL": 0.2675, "umls": 0.5337},
        "division-subjective":          {"rouge1": 0.3699, "rouge2": 0.1928, "rougeL": 0.2720, "umls": 0.4132},
        "division-objective_exam":      {"rouge1": 0.5288, "rouge2": 0.3332, "rougeL": 0.4533, "umls": 0.4701},
        "division-objective_results":   {"rouge1": 0.3911, "rouge2": 0.1217, "rougeL": 0.3823, "umls": 0.1199},
        "division-assessment_and_plan": {"rouge1": 0.3049, "rouge2": 0.1280, "rougeL": 0.2273, "umls": 0.3882},
        "dataset-virtassist": {"rouge1": 0.4402, "rouge2": 0.2021, "rougeL": 0.2934, "umls": 0.5600},
        "dataset-virtscribe": {"rouge1": 0.4294, "rouge2": 0.2015, "rougeL": 0.2873, "umls": 0.5691},
        "dataset-aci":        {"rouge1": 0.3859, "rouge2": 0.1624, "rougeL": 0.2485, "umls": 0.5089},
    },
}

DIVISION_KEYS = {
    "division-subjective":          "Subjective (HPI)",
    "division-objective_exam":      "Objective Exam",
    "division-objective_results":   "Objective Results",
    "division-assessment_and_plan": "Assessment & Plan",
}

SUBSET_KEYS = {
    "dataset-virtassist": "VirtAssist",
    "dataset-virtscribe": "VirtScribe",
    "dataset-aci":        "ACI",
}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=11),
    title="",
    title_font_color="#e2e8f0",
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
)

COLOR_MAP = {m["label"]: m["color"] for m in TECHNIQUE_META.values()}
COLOR_MAP["GPT-4 (ACI-Bench)"] = "#facc15"


# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────
def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _index(raw) -> dict:
    if not raw:
        return {}
    return {r["encounter_id"]: r for r in raw}


def _merge_eval(live_path: Path, static_key: str) -> dict:
    """
    Build eval dict by starting with STATIC_EVAL and overriding with
    any keys present in the live JSON file (if it exists).
    """
    result = {k: dict(v) for k, v in STATIC_EVAL.get(static_key, {}).items()}
    live = _load_json(live_path)
    if live:
        result.update(live)
    return result


def _soap_to_text(soap: dict) -> str:
    if not soap:
        return ""
    if "raw" in soap:
        return soap["raw"]
    parts = []
    if soap.get("chief_complaint"):
        parts.append(f"CHIEF COMPLAINT\n\n{soap['chief_complaint']}")
    if soap.get("subjective"):
        parts.append(f"HISTORY OF PRESENT ILLNESS\n\n{soap['subjective']}")
    if soap.get("objective_exam"):
        parts.append(f"PHYSICAL EXAM\n\n{soap['objective_exam']}")
    if (soap.get("objective_results") or "").strip():
        parts.append(f"RESULTS\n\n{soap['objective_results']}")
    if soap.get("assessment_and_plan"):
        parts.append(f"ASSESSMENT AND PLAN\n\n{soap['assessment_and_plan']}")
    return "\n\n".join(parts)


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "clinicalnlp_taskB_test1.csv")
    meta = pd.read_csv(BASE / "data" / "clinicalnlp_taskB_test1_metadata.csv")
    df = df.merge(
        meta[["encounter_id", "patient_gender", "patient_age", "patient_firstname",
              "patient_familyname", "cc", "2nd_complaints", "doctor_name"]],
        on="encounter_id", how="left",
    )
    # Normalize string columns that have trailing whitespace in the raw CSV
    for col in ["patient_gender", "dataset", "cc", "doctor_name",
                "patient_firstname", "patient_familyname"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Per-encounter result dicts (for Encounter Explorer)
    sb_res  = _index(_load_json(BASE / "results" / "simple_baseline_results.json"))
    pe_res  = _index(_load_json(BASE / "results" / "pe_results.json"))
    gvr_res = _index(_load_json(BASE / "results" / "gvr_results.json"))
    cd_res  = _index(_load_json(BASE / "results" / "cd_results.json"))

    # Aggregate eval metrics — static fallback + live JSON override
    sb_eval  = _merge_eval(BASE / "results" / "simple_baseline_results_predictions.json", "simple_baseline")
    pe_eval  = _merge_eval(BASE / "results" / "pe_results_predictions.json",               "prompt_engineering")
    gvr_eval = _merge_eval(BASE / "results" / "gvr_results_predictions.json",              "gvr")
    cd_eval  = _merge_eval(BASE / "results" / "cd_results_predictions.json",               "constrained_decoding")

    return df, sb_res, pe_res, gvr_res, cd_res, sb_eval, pe_eval, gvr_eval, cd_eval


df, sb_res, pe_res, gvr_res, cd_res, sb_eval, pe_eval, gvr_eval, cd_eval = load_data()

PER_ENC = {
    "simple_baseline":      sb_res,
    "prompt_engineering":   pe_res,
    "gvr":                  gvr_res,
    "constrained_decoding": cd_res,
}

EVAL = {
    "simple_baseline":      sb_eval,
    "prompt_engineering":   pe_eval,
    "gvr":                  gvr_eval,
    "constrained_decoding": cd_eval,
}

# All techniques have results (either live JSON or STATIC_EVAL fallback)
EVAL_OK = {k: True for k in TECHNIQUE_META}


def pipeline_stats(res: dict) -> dict:
    if not res:
        return {"available": False, "total": 0, "success": 0, "fail": 0, "failures": []}
    recs = list(res.values())
    return {
        "available": True,
        "total":     len(recs),
        "success":   sum(1 for r in recs if r["success"]),
        "fail":      sum(1 for r in recs if not r["success"]),
        "failures":  [r for r in recs if not r["success"]],
    }


STATS = {k: pipeline_stats(v) for k, v in PER_ENC.items()}


# ─────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────
def kpi_card(label, value, sub=""):
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        + (f"<div class='kpi-sub'>{sub}</div>" if sub else "")
        + "</div>"
    )


def badge_html(tkey):
    m = TECHNIQUE_META[tkey]
    return f"<span class='badge {m['badge']}'>{m['short']}</span>"


def pct(val, decimals=2):
    """Convert 0-1 decimal to percentage string."""
    return f"{val * 100:.{decimals}f}"


def make_fullnote_bar() -> go.Figure:
    """Grouped bar: ROUGE-1, ROUGE-2, ROUGE-L, MEDCON for all 4 techniques."""
    rows = []
    for tkey, tmeta in TECHNIQUE_META.items():
        a = EVAL[tkey].get("ALL", {})
        for raw, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                           ("rougeL", "ROUGE-L"), ("umls", "MEDCON")]:
            rows.append({"Technique": tmeta["label"], "Metric": label,
                         "Score": round(a.get(raw, 0) * 100, 2)})
    fig = px.bar(
        pd.DataFrame(rows), x="Metric", y="Score", color="Technique",
        barmode="group", text_auto=".1f", height=400,
        color_discrete_map=COLOR_MAP, labels={"Score": "Score (%)"},
    )
    fig.update_layout(
        **PLOT_BASE,
        yaxis=dict(gridcolor="#1e293b", range=[0, 72], title="Score (%)"),
        xaxis=dict(showgrid=False),
    )
    fig.update_traces(textfont_color="#e2e8f0", textposition="outside")
    # GPT-4 MEDCON reference line
    fig.add_hline(y=57.78, line_dash="dot", line_color="#facc15", opacity=0.5,
                  annotation_text="GPT-4 MEDCON (57.78)",
                  annotation_font_color="#facc15",
                  annotation_position="bottom right")
    return fig


def make_subset_bar() -> go.Figure:
    """ROUGE-1 by dataset subset — only shows techniques with non-zero subset data."""
    # Filter to techniques that have at least one non-zero subset value
    techniques_with_data = [
        (tkey, tmeta) for tkey, tmeta in TECHNIQUE_META.items()
        if any(EVAL[tkey].get(sk, {}).get("rouge1", 0) > 0 for sk in SUBSET_KEYS)
    ]
    rows = []
    for sk, slabel in SUBSET_KEYS.items():
        for tkey, tmeta in techniques_with_data:
            val = EVAL[tkey].get(sk, {}).get("rouge1", 0)
            rows.append({"Subset": slabel, "Score": round(val * 100, 2),
                         "Technique": tmeta["label"]})
    fig = px.bar(
        pd.DataFrame(rows), x="Subset", y="Score", color="Technique",
        barmode="group", text_auto=".1f", height=370,
        color_discrete_map=COLOR_MAP, labels={"Score": "ROUGE-1 (%)"},
    )
    fig.update_layout(
        **PLOT_BASE,
        yaxis=dict(gridcolor="#1e293b", range=[0, 72], title="ROUGE-1 (%)"),
        xaxis=dict(showgrid=False),
    )
    fig.update_traces(textfont_color="#e2e8f0", textposition="outside")
    return fig


def make_division_bar() -> go.Figure:
    """ROUGE-1 by SOAP division for all 4 techniques + GPT-4 reference."""
    rows = []
    for dk, dlabel in DIVISION_KEYS.items():
        for tkey, tmeta in TECHNIQUE_META.items():
            val = EVAL[tkey].get(dk, {}).get("rouge1", 0)
            rows.append({"Division": dlabel, "Score": round(val * 100, 2),
                         "Technique": tmeta["label"]})
        # GPT-4 reference
        gpt4_val = PAPER_BASELINES["GPT-4 (ACI-Bench)"].get(dk, {}).get("rouge1")
        if gpt4_val:
            rows.append({"Division": dlabel, "Score": round(gpt4_val * 100, 2),
                         "Technique": "GPT-4 (ACI-Bench)"})
    fig = px.bar(
        pd.DataFrame(rows), x="Division", y="Score", color="Technique",
        barmode="group", text_auto=".1f", height=390,
        color_discrete_map=COLOR_MAP, labels={"Score": "ROUGE-1 (%)"},
    )
    fig.update_layout(
        **PLOT_BASE,
        yaxis=dict(gridcolor="#1e293b", range=[0, 72], title="ROUGE-1 (%)"),
        xaxis=dict(showgrid=False),
    )
    fig.update_traces(textfont_color="#e2e8f0", textposition="outside")
    return fig


def make_medcon_bar() -> go.Figure:
    """MEDCON (UMLS) overall for all 4 techniques + GPT-4 reference."""
    names, vals, colors = [], [], []
    names.append("GPT-4 (ACI-Bench)")
    vals.append(round(PAPER_BASELINES["GPT-4 (ACI-Bench)"]["umls"] * 100, 2))
    colors.append("#facc15")
    for tkey, tmeta in TECHNIQUE_META.items():
        names.append(tmeta["label"])
        vals.append(round(EVAL[tkey].get("ALL", {}).get("umls", 0) * 100, 2))
        colors.append(tmeta["color"])
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside", textfont_color="#e2e8f0",
    ))
    fig.update_layout(
        **PLOT_BASE, height=340,
        yaxis=dict(gridcolor="#1e293b", range=[0, 72], title="MEDCON (%)"),
        xaxis=dict(showgrid=False),
    )
    return fig


def make_benchmark_bar(metric_key: str, metric_label: str) -> go.Figure:
    """All techniques + all paper baselines for a chosen metric."""
    names, vals, colors = [], [], []
    for name, s in PAPER_BASELINES.items():
        val = s.get(metric_key)
        if val is None:
            continue
        names.append(name)
        vals.append(round(val * 100, 2))
        colors.append("#475569")
    for tkey, tmeta in TECHNIQUE_META.items():
        val = EVAL[tkey].get("ALL", {}).get(metric_key, 0)
        names.append(f"{tmeta['short']}\n(Llama-8B)")
        vals.append(round(val * 100, 2))
        colors.append(tmeta["color"])

    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.1f}" for v in vals],
        textposition="outside", textfont_color="#e2e8f0",
    ))
    ref_val = PAPER_BASELINES["GPT-4 (ACI-Bench)"].get(metric_key)
    if ref_val:
        fig.add_hline(
            y=round(ref_val * 100, 2), line_dash="dot",
            line_color="#facc15", opacity=0.6,
            annotation_text=f"GPT-4 ({round(ref_val * 100, 2)})",
            annotation_font_color="#facc15",
            annotation_position="bottom right",
        )
    fig.update_layout(
        **PLOT_BASE, height=380,
        yaxis=dict(gridcolor="#1e293b", range=[0, 72], title=f"{metric_label} (%)"),
        xaxis=dict(showgrid=False),
    )
    return fig


# ─────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("""
    <h1 style='color:#f1f5f9;font-size:1.85rem;font-weight:700;margin-bottom:0.05rem;'>
      🏥 Clinical SOAP Note Generation
    </h1>
    <p style='color:#64748b;font-size:0.85rem;margin-top:0;'>
      CS6180 Team 26 &nbsp;·&nbsp; ACI-Bench Test Set 1 (40 encounters) &nbsp;·&nbsp;
      Northeastern University
    </p>""", unsafe_allow_html=True)
with h2:
    st.markdown(
        "<div style='text-align:right;padding-top:0.85rem;'>"
        + "".join(badge_html(t) for t in TECHNIQUE_META)
        + "</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b;margin:0.6rem 0 1.1rem;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab_ov, tab_met, tab_exp, tab_ds = st.tabs([
    "📊  Overview", "📈  Metrics", "🔍  Encounter Explorer", "🗂️  Dataset",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
with tab_ov:

    # ── KPI strip ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    pe_s = STATS["prompt_engineering"]
    with c1:
        st.markdown(kpi_card("Total Encounters", "40", "ACI-Bench Test Set 1"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Best ROUGE-1", "53.01", "Simple BL (Llama-8B)"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Best MEDCON", "57.52", "Simple BL — GPT-4: 57.78"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "PE Pass Rate",
            f"{pe_s['success']}/{pe_s['total']}" if pe_s["available"] else "39/40",
            "2 Pydantic validation failure",
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Techniques", "4", "GVR · PE · CD · Simple BL"),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison table + radar ───────────────────────────
    col_tbl, col_radar = st.columns([1.2, 1])

    with col_tbl:
        st.markdown(
            "<div class='section-title'>Full-Note Results — All Techniques vs Paper Baselines</div>",
            unsafe_allow_html=True)

        rows = []
        # Our techniques first (highlighted)
        for tkey, tmeta in TECHNIQUE_META.items():
            e    = EVAL[tkey].get("ALL", {})
            st_i = STATS[tkey]
            rows.append({
                "Model":   tmeta["label"],
                "ROUGE-1": f"{e.get('rouge1', 0) * 100:.2f}",
                "ROUGE-2": f"{e.get('rouge2', 0) * 100:.2f}",
                "ROUGE-L": f"{e.get('rougeL', 0) * 100:.2f}",
                "MEDCON":  f"{e.get('umls', 0) * 100:.2f}",
                "Success": f"{st_i['success']}/{st_i['total']}"
                           if st_i["available"] else "39/40",
                "_ours": True,
            })
        # Paper baselines
        for name, scores in PAPER_BASELINES.items():
            rows.append({
                "Model":   name,
                "ROUGE-1": f"{scores['rouge1'] * 100:.2f}",
                "ROUGE-2": f"{scores['rouge2'] * 100:.2f}",
                "ROUGE-L": f"{scores['rougeL'] * 100:.2f}",
                "MEDCON":  f"{scores['umls'] * 100:.2f}" if scores.get("umls") is not None else "—",
                "Success": "—",
                "_ours": False,
            })

        tdf = pd.DataFrame(rows)
        ours_mask = tdf["_ours"].tolist()
        display_df = tdf.drop(columns=["_ours"])
        n_cols = len(display_df.columns)

        def _row_style(r):
            is_ours = ours_mask[r.name] if r.name < len(ours_mask) else False
            return (["background-color:#111827;color:#e2e8f0"] if is_ours
                    else ["color:#94a3b8"]) * n_cols

        st.dataframe(
            display_df.style.apply(_row_style, axis=1)
                       .set_properties(**{"font-size": "12.5px"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "All scores shown as percentages (×100). "
            "Team 26 techniques use Llama-3.1-8b-instruct via OpenRouter. "
            "GPT-4 from ACI-Bench. "
        )

    with col_radar:
        st.markdown("<div class='section-title'>Radar — ROUGE & MEDCON</div>",
                    unsafe_allow_html=True)
        cats = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "MEDCON"]
        fig_r = go.Figure()
        # GPT-4 reference
        fig_r.add_trace(go.Scatterpolar(
            r=[51.76, 22.58, 45.97, 57.78, 51.76],
            theta=cats + [cats[0]],
            fill="toself", name="GPT-4 (ACI-Bench)",
            line_color="#facc15", fillcolor="#facc15", opacity=0.12,
        ))
        for tkey, tmeta in TECHNIQUE_META.items():
            e = EVAL[tkey].get("ALL", {})
            vals = [
                round(e.get("rouge1", 0) * 100, 2),
                round(e.get("rouge2", 0) * 100, 2),
                round(e.get("rougeL", 0) * 100, 2),
                round(e.get("umls",   0) * 100, 2),
            ]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=tmeta["label"],
                line_color=tmeta["color"], fillcolor=tmeta["color"], opacity=0.22,
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 65], gridcolor="#1e293b",
                                tickfont_color="#475569", tickfont_size=8),
                angularaxis=dict(gridcolor="#1e293b", tickfont_color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", height=360,
            legend=dict(font_color="#cbd5e1", bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=-0.22),
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

    # ── Pipeline pass-rate row ─────────────────────────────
    st.markdown("<div class='section-title'>Pipeline Stats — All Four Techniques</div>",
                unsafe_allow_html=True)
    stat_cols = st.columns(4)
    for col, (tkey, tmeta) in zip(stat_cols, TECHNIQUE_META.items()):
        st_i = STATS[tkey]
        with col:
            if not st_i["available"]:
                # Fallback: known results from slide
                known = {"simple_baseline": (40,39), "prompt_engineering": (40,39),
                         "gvr": (40,39), "constrained_decoding": (40,40)}
                tot, suc = known.get(tkey, (40, "?"))
                fail_txt = "D2N127 (Pydantic)" if tkey in ("prompt_engineering","gvr") else "none"
                pct_val  = suc / tot if isinstance(suc, int) else 1.0
                st.markdown(f"""
                <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1rem;'>
                  <div style='margin-bottom:0.5rem;'>
                    <span class='badge {tmeta["badge"]}'>{tmeta["short"]}</span></div>
                  <div style='color:{tmeta["color"]};font-size:1.7rem;font-weight:700;'>
                    {suc}/{tot}</div>
                  <div style='color:#64748b;font-size:0.72rem;margin-top:0.2rem;'>
                    {pct_val*100:.1f}% success</div>
                  <div style='color:#475569;font-size:0.7rem;margin-top:0.4rem;'>
                    Failed: {fail_txt}</div>
                </div>""", unsafe_allow_html=True)
                st.progress(pct_val)
            else:
                pct_val = st_i["success"] / st_i["total"]
                fails   = ", ".join(r["encounter_id"] for r in st_i["failures"]) or "none"
                st.markdown(f"""
                <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:1rem;'>
                  <div style='margin-bottom:0.5rem;'>
                    <span class='badge {tmeta["badge"]}'>{tmeta["short"]}</span></div>
                  <div style='color:{tmeta["color"]};font-size:1.7rem;font-weight:700;line-height:1;'>
                    {st_i["success"]}/{st_i["total"]}</div>
                  <div style='color:#64748b;font-size:0.72rem;margin-top:0.2rem;'>
                    {pct_val*100:.1f}% success</div>
                  <div style='color:#475569;font-size:0.7rem;margin-top:0.4rem;'>
                    Failed: {fails}</div>
                </div>""", unsafe_allow_html=True)
                st.progress(pct_val)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Technique description cards ────────────────────────
    st.markdown("<div class='section-title'>Technique Descriptions</div>",
                unsafe_allow_html=True)
    TECH_INFO = {
        "simple_baseline": {
            "icon": "📝",
            "long": "Replicates the ACI-Bench paper's minimal prompt. Free-text output, "
                    "post-processed section headers. Direct apples-to-apples comparison "
                    "with the paper's GPT baselines.",
            "bullets": ["Minimal paper prompt", "Free-text output",
                        "No retry / schema validation", "Post-processed section headers"],
        },
        "prompt_engineering": {
            "icon": "✏️",
            "long": "Carefully engineered single-shot prompt with chain-of-thought reasoning "
                    "and explicit JSON field instructions. Pydantic validates the response.",
            "bullets": ["Single-shot (no retry)", "Chain-of-thought reasoning",
                        "Full schema in prompt", "Pydantic validation"],
        },
        "gvr": {
            "icon": "🔁",
            "long": "Generate → Validate (JSON + Pydantic) → Retry with targeted field-level "
                    "error feedback. Up to 3 LLM calls per encounter.",
            "bullets": ["Up to 3 LLM calls", "JSON + Pydantic retry loop",
                        "Field-level error injection", "39/40 success"],
        },
        "constrained_decoding": {
            "icon": "🔒",
            "long": "Uses the instructor library in JSON mode to enforce the SOAPNote "
                    "Pydantic schema at the token level. Structurally cannot produce invalid output.",
            "bullets": ["Token-level schema enforcement", "instructor + JSON mode",
                        "No retry needed by design", "OpenRouter / llama-3.1-8B"],
        },
    }
    desc_cols = st.columns(4)
    for col, (tkey, tmeta) in zip(desc_cols, TECHNIQUE_META.items()):
        d = TECH_INFO[tkey]
        bullets = "".join(
            f"<li style='color:#94a3b8;font-size:0.75rem;margin-bottom:0.15rem;'>{b}</li>"
            for b in d["bullets"]
        )
        with col:
            st.markdown(f"""
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                        padding:1rem;height:100%;'>
              <div style='font-size:1.3rem;margin-bottom:0.35rem;'>{d['icon']}</div>
              <div style='font-weight:600;color:#e2e8f0;font-size:0.85rem;margin-bottom:0.35rem;'>
                {tmeta["label"]}</div>
              <p style='color:#64748b;font-size:0.75rem;line-height:1.5;margin-bottom:0.55rem;'>
                {d["long"]}</p>
              <ul style='padding-left:1rem;margin:0;'>{bullets}</ul>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — METRICS
# ══════════════════════════════════════════════════════════
with tab_met:
    st.info(
        "📊 All four technique results loaded from Team 26 experiments (Tables 1 & 2). "
        "**Division chart** shows ROUGE-1 for all techniques including GPT-4 reference. "
        "**Subset chart** shows full breakdown for Simple BL & GVR; PE & CD subset scores "
    )

    # ── Full-note ALL — ROUGE + MEDCON ────────────────────
    st.markdown("<div class='section-title'>Full-Note Scores — All 40 Encounters</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_fullnote_bar(), use_container_width=True)

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

    # ── By subset + by division ────────────────────────────
    col_sub, col_div = st.columns(2)
    with col_sub:
        st.markdown("<div class='section-title'>ROUGE-1 by Dataset Subset</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(make_subset_bar(), use_container_width=True)
        st.caption(
            "virtassist = wake-word triggered · virtscribe = dictation · aci = natural conversation. "        )
    with col_div:
        st.markdown(
            "<div class='section-title'>ROUGE-1 by SOAP Division (incl. GPT-4)</div>",
            unsafe_allow_html=True)
        st.plotly_chart(make_division_bar(), use_container_width=True)
        st.caption(
            "PE & GVR strongest on Objective Exam · CD leads on Objective Results · "
            "Simple BL leads on Assessment & Plan"
        )

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

    # ── MEDCON overall ────────────────────────────────────
    st.markdown("<div class='section-title'>MEDCON (UMLS F1) — Overall Score</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_medcon_bar(), use_container_width=True)
    st.caption(
        "MEDCON measures F1 overlap of UMLS medical concepts — captures clinical factual "
        "accuracy beyond surface text similarity. Simple BL (57.52) nearly matches "
        "GPT-4 (57.78)."
    )

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)

    # ── vs Paper baselines — metric toggle ────────────────
    st.markdown("<div class='section-title'>vs ACI-Bench Paper Baselines</div>",
                unsafe_allow_html=True)
    bm_choice = st.radio(
        "Select metric", ["ROUGE-1", "ROUGE-2", "ROUGE-L", "MEDCON"],
        horizontal=True, key="bm_metric",
    )
    bm_key = {"ROUGE-1": "rouge1", "ROUGE-2": "rouge2",
              "ROUGE-L": "rougeL", "MEDCON": "umls"}[bm_choice]
    st.plotly_chart(make_benchmark_bar(bm_key, bm_choice), use_container_width=True)
    st.caption(
        "Dark gray = paper baselines. Colored = Team 26 (Llama-3.1-8b-instruct). "
        "GPT-4 dotted line = ACI-Bench."
    )


# ══════════════════════════════════════════════════════════
# TAB 3 — ENCOUNTER EXPLORER
# ══════════════════════════════════════════════════════════
with tab_exp:
    encounter_ids = df["encounter_id"].tolist()

    # ── Controls ──────────────────────────────────────────
    ctl1, ctl2, ctl3, ctl4 = st.columns([2.2, 0.9, 1.2, 1.2])
    with ctl1:
        sel_id = st.selectbox(
            "Select Encounter", encounter_ids,
            format_func=lambda x: f"{x}  ·  {df[df.encounter_id==x]['cc'].values[0]}",
        )
    with ctl2:
        st.selectbox("Subset", ["All", "virtassist", "virtscribe", "aci"], key="sub_filt")

    explorable = {k: TECHNIQUE_META[k]["label"] for k, v in PER_ENC.items() if v}
    with ctl3:
        tech_a = st.selectbox(
            "Left panel", list(explorable.keys()),
            format_func=lambda k: explorable[k],
            index=list(explorable.keys()).index("simple_baseline")
                  if "simple_baseline" in explorable else 0,
        )
    with ctl4:
        remaining = [k for k in explorable if k != tech_a]
        tech_b = st.selectbox(
            "Right panel", remaining,
            format_func=lambda k: explorable[k],
            index=remaining.index("prompt_engineering")
                  if "prompt_engineering" in remaining else 0,
        )

    # ── Patient info strip ────────────────────────────────
    row    = df[df["encounter_id"] == sel_id].iloc[0]
    age    = row.get("patient_age", "?")
    gender = row.get("patient_gender", "?")
    cc_v   = row.get("cc", "—")
    dset   = row.get("dataset", "—")
    second = str(row.get("2nd_complaints", "")).strip()

    def status_pill(tkey):
        r = PER_ENC.get(tkey, {}).get(sel_id)
        if not r:
            return "<span class='pill'>no data</span>"
        return ("<span class='pill pill-green'>✓ success</span>" if r["success"]
                else f"<span class='pill pill-red'>✗ {r.get('failure_type','fail')}</span>")

    status_row = " &nbsp; ".join(
        f"{badge_html(t)} {status_pill(t)}" for t in TECHNIQUE_META if PER_ENC.get(t)
    )

    st.markdown(f"""
    <div style='background:#0a0f1a;border:1px solid #1e293b;border-radius:10px;
                padding:0.8rem 1.1rem;margin-bottom:0.85rem;'>
      <div style='display:flex;flex-wrap:wrap;align-items:center;gap:0.3rem;margin-bottom:0.45rem;'>
        <strong style='color:#e2e8f0;font-size:0.9rem;margin-right:0.3rem;'>{sel_id}</strong>
        <span class='pill'>{dset}</span>
        <span class='pill'>{gender} · {age} yo</span>
        <span class='pill pill-blue'>CC: {cc_v}</span>
        {'<span class="pill pill-amber">2nd: ' + second + '</span>'
         if second and second != "nan" else ''}
      </div>
      <div style='display:flex;flex-wrap:wrap;gap:0.35rem;align-items:center;'>
        <span style='color:#475569;font-size:0.7rem;'>Pipeline results:</span>
        {status_row}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Three-panel note view ─────────────────────────────
    def get_note(tkey):
        r = PER_ENC.get(tkey, {}).get(sel_id)
        if not r:
            return "⚠️  No per-encounter results file for this technique."
        if not r["success"]:
            return f"⚠️  Failed ({r.get('failure_type', '')})\n\n{r.get('final_error', '')}"
        return _soap_to_text(r.get("soap_note") or {}) or "⚠️  Empty output."

    p_ref, p_a, p_b = st.columns(3)

    with p_ref:
        st.markdown("""<div class='panel-header'>
          <span class='panel-dot' style='background:#475569;'></span>
          Reference Note (Gold Standard)
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='note-panel'>{str(row.get('note', ''))}</div>",
                    unsafe_allow_html=True)

    with p_a:
        tm_a = TECHNIQUE_META[tech_a]
        st.markdown(f"""<div class='panel-header'>
          <span class='panel-dot' style='background:{tm_a["color"]};'></span>
          {tm_a["label"]}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='note-panel'>{get_note(tech_a)}</div>",
                    unsafe_allow_html=True)

    with p_b:
        tm_b = TECHNIQUE_META[tech_b]
        st.markdown(f"""<div class='panel-header'>
          <span class='panel-dot' style='background:{tm_b["color"]};'></span>
          {tm_b["label"]}
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='note-panel'>{get_note(tech_b)}</div>",
                    unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────
    nav_l, _, nav_r = st.columns([1, 12, 1])
    idx = encounter_ids.index(sel_id)
    with nav_l:
        if st.button("⬅ Prev") and idx > 0:
            st.rerun()
    with nav_r:
        if st.button("Next ➡") and idx < len(encounter_ids) - 1:
            st.rerun()

    # ── Structured field extraction expander ──────────────
    TEXT_FIELDS = {
        "chief_complaint", "subjective", "objective_exam", "objective_results",
        "assessment_and_plan", "allergies", "past_medical_history", "current_medications",
        "family_history", "occupation", "associated_symptoms", "pain_quality",
        "pain_location", "symptom_duration", "raw",
    }
    structured = [
        t for t in [tech_a, tech_b]
        if t and t != "simple_baseline"
        and (PER_ENC.get(t, {}).get(sel_id) or {}).get("success")
    ]
    if structured:
        with st.expander("🔬 Structured Field Extraction (typed clinical fields)", expanded=False):
            fcols = st.columns(len(structured))
            for col, tkey in zip(fcols, structured):
                with col:
                    sn = PER_ENC[tkey][sel_id].get("soap_note") or {}
                    typed = {k: v for k, v in sn.items()
                             if k not in TEXT_FIELDS and v is not None and v != ""}
                    st.markdown(f"**{TECHNIQUE_META[tkey]['label']}**")
                    st.json(typed if typed else {"note": "No typed fields extracted."})


# ══════════════════════════════════════════════════════════
# TAB 4 — DATASET
# ══════════════════════════════════════════════════════════
with tab_ds:
    st.markdown("<div class='section-title'>ACI-Bench Test Set 1 — Dataset Overview</div>",
                unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns(3)
    with r1a:
        sc = df["dataset"].value_counts().reset_index()
        sc.columns = ["Subset", "Count"]
        fig_pie = px.pie(
            sc, values="Count", names="Subset", hole=0.44, title="Dataset Subsets",
            color_discrete_map={"virtassist": "#60a5fa", "virtscribe": "#34d399", "aci": "#f59e0b"},
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
                               title_font_color="#e2e8f0", height=270,
                               margin=dict(l=0, r=0, t=40, b=0),
                               legend=dict(font_color="#cbd5e1", bgcolor="rgba(0,0,0,0)"))
        fig_pie.update_traces(textfont_color="#e2e8f0")
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1b:
        gc = df["patient_gender"].value_counts().reset_index()
        gc.columns = ["Gender", "Count"]
        fig_g = px.bar(gc, x="Gender", y="Count", text_auto=True, title="Patient Gender",
                       color="Gender",
                       color_discrete_map={"male": "#60a5fa", "female": "#f472b6"})
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#94a3b8", title_font_color="#e2e8f0",
                              showlegend=False, height=270, margin=dict(l=0, r=0, t=40, b=0),
                              yaxis=dict(gridcolor="#1e293b"), xaxis=dict(showgrid=False))
        fig_g.update_traces(textfont_color="#e2e8f0", textposition="outside")
        st.plotly_chart(fig_g, use_container_width=True)

    with r1c:
        ages = pd.to_numeric(df["patient_age"], errors="coerce").dropna()
        fig_age = px.histogram(ages, nbins=10, title="Patient Age Distribution",
                                color_discrete_sequence=["#818cf8"], labels={"value": "Age"})
        fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#94a3b8", title_font_color="#e2e8f0",
                               height=270, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                               yaxis=dict(gridcolor="#1e293b"), xaxis=dict(showgrid=False))
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("<hr style='border-color:#1e293b;margin:0.5rem 0;'>", unsafe_allow_html=True)

    cc_col, stat_col = st.columns([1.4, 1])

    with cc_col:
        st.markdown("<div class='section-title'>Chief Complaints (Top 20)</div>",
                    unsafe_allow_html=True)
        cc_df = df["cc"].value_counts().head(20).reset_index()
        cc_df.columns = ["CC", "Count"]
        fig_cc = px.bar(
            cc_df.sort_values("Count"), y="CC", x="Count", orientation="h", height=460,
            color="Count", color_continuous_scale=["#1e3a5f", "#3b82f6", "#93c5fd"],
            text_auto=True,
        )
        fig_cc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#94a3b8", coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(showgrid=False, tickfont_size=11),
                              xaxis=dict(gridcolor="#1e293b"))
        fig_cc.update_traces(textfont_color="#e2e8f0")
        st.plotly_chart(fig_cc, use_container_width=True)

    with stat_col:
        st.markdown("<div class='section-title'>Pipeline Pass Rates</div>",
                    unsafe_allow_html=True)
        for tkey, tmeta in TECHNIQUE_META.items():
            st_i = STATS[tkey]
            known_fails = {
                "simple_baseline":    ("none", 40, 40),
                "prompt_engineering": ("D2N127 (Pydantic)", 40, 39),
                "gvr":                ("D2N117 (JSON parse)", 40, 39),
                "constrained_decoding": ("none", 40, 40),
            }
            if not st_i["available"]:
                fail_txt, tot, suc = known_fails[tkey]
                pct_val = suc / tot
            else:
                pct_val  = st_i["success"] / st_i["total"]
                suc, tot = st_i["success"], st_i["total"]
                fail_txt = (", ".join(r["encounter_id"] for r in st_i["failures"])
                            or "none")

            st.markdown(f"""
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                        padding:0.6rem 0.85rem;margin-bottom:0.3rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='badge {tmeta["badge"]}'>{tmeta["short"]}</span>
                <span style='color:{tmeta["color"]};font-weight:600;font-size:0.85rem;'>
                  {suc}/{tot}</span>
              </div>
              <div style='color:#475569;font-size:0.7rem;margin-top:0.25rem;'>
                Failed: {fail_txt}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct_val)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Model &amp; Infra</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                    padding:0.85rem;color:#94a3b8;font-size:0.78rem;line-height:1.9;'>
          🤖 <strong style='color:#e2e8f0;'>meta-llama/llama-3.1-8b-instruct</strong><br>
          🌐 OpenRouter API<br>
          🌡️ Temperature: 0.0 (deterministic)<br>
          📏 Max tokens: 3,000<br>
          🗂️ ACI-Bench test set 1 — 40 encounters<br>
          📊 ROUGE via evaluate · MEDCON via QuickUMLS
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e293b;margin-top:2rem;'>
<div style='text-align:center;color:#334155;font-size:0.72rem;padding-bottom:1rem;'>
  CS6180 Team 26 &nbsp;·&nbsp; Clinical NLP — Structured SOAP Note Generation &nbsp;·&nbsp;
  ACI-Bench (Yim et al., 2023) &nbsp;·&nbsp; Northeastern University
</div>""", unsafe_allow_html=True)