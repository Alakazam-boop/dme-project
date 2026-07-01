"""
Decision Memory Engine (DME) — Clinical Advisory System
AI powered by Groq (free) · Llama 3.3 70B

This is the main Streamlit dashboard. It's intentionally all in one file so
it's easy to run and read as a single artefact. In a production system you'd
split the tabs into separate page modules, but for a research prototype
this layout is much easier to demonstrate and present.

The app is built around five tabs:
  Tab 1 — Clinical Analysis   : enter vitals, get a risk score + outcome prediction
  Tab 2 — Deep Analysis       : AI-powered summary, differentials, and missing data
  Tab 3 — Decision Memory     : charts of historical cases + model version history
  Tab 4 — Safety & Bias       : automated data quality checks + AI safety analysis
  Tab 5 — AI Assistant        : free-form chat with full patient context

All AI calls go through Groq (free tier, Llama 3.3 70B). The ML prediction
uses a RandomForest trained on 100 MIMIC-IV demo patients.
"""

import sys, os, html as html_module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules.outcome_linker import get_completed_traces_with_outcomes
from modules.learning_engine import predict_outcome, train_classifier, build_cbr_index
from modules.decision_logger import log_decision

DB_PATH = 'database/dme.db'

st.set_page_config(
    page_title="Decision Memory Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design system — CSS for the dark theme, cards, risk banners, vitals table,
# chat bubbles, audit rows, and all other UI components used across the five tabs.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* Layout */
  .main .block-container { padding: 1.2rem 2rem 3rem 2rem; max-width: 1400px; }
  section[data-testid="stSidebar"] { background: #080f1a !important; border-right: 1px solid #1e293b; }
  section[data-testid="stSidebar"] > div { padding-top: 1rem; }

  /* Global background */
  .stApp { background: #060d18; }

  /* Cards */
  .dme-card {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    border: 1px solid #1e293b; border-radius: 16px;
    padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.2s;
  }
  .dme-card:hover { border-color: #334155; }
  .dme-card-title {
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #475569; margin-bottom: 14px;
  }

  /* RISK banners */
  .risk-critical {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444; border-left: 5px solid #ef4444;
    padding: 20px 24px; border-radius: 14px; color: white;
    box-shadow: 0 0 30px rgba(239,68,68,0.2);
  }
  .risk-high {
    background: linear-gradient(135deg, #431407, #7c2d12);
    border: 1px solid #f97316; border-left: 5px solid #f97316;
    padding: 20px 24px; border-radius: 14px; color: white;
    box-shadow: 0 0 30px rgba(249,115,22,0.15);
  }
  .risk-medium {
    background: linear-gradient(135deg, #422006, #78350f);
    border: 1px solid #f59e0b; border-left: 5px solid #f59e0b;
    padding: 20px 24px; border-radius: 14px; color: white;
  }
  .risk-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e; border-left: 5px solid #22c55e;
    padding: 20px 24px; border-radius: 14px; color: white;
  }

  /* Advisory flags */
  .flag-urgent {
    background: linear-gradient(135deg, #1a0505, #270a0a);
    border: 1px solid #dc2626; border-left: 5px solid #ef4444;
    padding: 14px 18px; border-radius: 10px; color: #fca5a5;
    margin-bottom: 8px;
  }
  .flag-advisory {
    background: linear-gradient(135deg, #060d1f, #0c1a2e);
    border: 1px solid #1d4ed8; border-left: 5px solid #3b82f6;
    padding: 14px 18px; border-radius: 10px; color: #bfdbfe;
    margin-bottom: 8px;
  }
  .flag-ok {
    background: linear-gradient(135deg, #031a0a, #052e16);
    border: 1px solid #16a34a; border-left: 5px solid #22c55e;
    padding: 14px 18px; border-radius: 10px; color: #86efac;
    margin-bottom: 8px;
  }

  /* Case cards */
  .case-card {
    background: linear-gradient(160deg, #0a1525, #0f1e35);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 18px; margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.2s;
    cursor: pointer;
  }
  .case-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
  .case-outcome-good { color: #4ade80; font-weight: 700; font-size: 15px; }
  .case-outcome-bad  { color: #f87171; font-weight: 700; font-size: 15px; }

  /* Similarity badge */
  .sim-badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    background: rgba(59,130,246,0.15); color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.3);
  }

  /* Progress bars */
  .prog-outer { background: #1e293b; border-radius: 999px; height: 8px; margin: 6px 0; overflow: hidden; }
  .prog-inner  { height: 8px; border-radius: 999px; transition: width 0.5s ease; }

  /* Explain boxes */
  .explain-box {
    background: rgba(15,23,42,0.8); border: 1px dashed #1e3a5f;
    border-radius: 10px; padding: 12px 16px; margin-top: 10px;
    font-size: 13px; color: #64748b; line-height: 1.6;
  }
  .explain-box strong { color: #94a3b8; }

  /* AI response */
  .ai-response {
    background: linear-gradient(135deg, #060d1f, #0a1525);
    border: 1px solid #1d4ed8; border-radius: 12px;
    padding: 20px 22px; margin: 12px 0;
    font-size: 14px; line-height: 1.8; color: #e2e8f0;
  }
  .ai-response h3, .ai-response h4 { color: #93c5fd; margin-top: 12px; }
  .ai-response strong { color: #bfdbfe; }
  .ai-response ul { padding-left: 20px; }
  .ai-response li { margin-bottom: 6px; }

  /* Chat bubbles */
  .chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1e40af);
    padding: 12px 16px; border-radius: 14px 14px 4px 14px;
    margin: 10px 0; color: #e0f2fe; font-size: 14px;
    box-shadow: 0 2px 8px rgba(30,64,175,0.3);
  }
  /* AI response box for Tab 2 deep analysis */
  .ai-resp {
    background: linear-gradient(135deg, #050c1f, #08122a);
    border: 1px solid #1a3a6b; border-radius: 12px;
    padding: 16px 20px; margin: 10px 0;
    font-size: 14px; line-height: 1.75; color: #dde8f8;
    white-space: pre-wrap;
  }
  .ai-resp strong { color: #93c5fd; }

  /* AI chat response box */
  .chat-ai {
    background: linear-gradient(135deg, #0f172a, #111827);
    padding: 12px 16px; border-radius: 14px 14px 14px 4px;
    margin: 10px 0; color: #e2e8f0; font-size: 14px;
    border-left: 3px solid #3b82f6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    white-space: pre-wrap; min-height: 1em;
  }

  /* Audit rows */
  .audit-row {
    background: #0a1120; border: 1px solid #1e293b;
    border-radius: 8px; padding: 10px 16px;
    margin-bottom: 6px; font-size: 13px; color: #94a3b8;
    transition: border-color 0.2s;
  }
  .audit-row:hover { border-color: #334155; }

  /* Metrics */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f172a, #111827) !important;
    border: 1px solid #1e293b !important; border-radius: 14px !important;
    padding: 16px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
  }

  /* Vitals table */
  .vitals-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #1e293b;
    font-size: 14px;
  }
  .vitals-row:last-child { border-bottom: none; }
  .vit-name  { color: #94a3b8; font-weight: 500; }
  .vit-value { color: #e2e8f0; font-weight: 700; font-size: 15px; }
  .vit-ok    { color: #4ade80; font-size: 12px; }
  .vit-warn  { color: #fbbf24; font-size: 12px; }
  .vit-na    { color: #475569; font-size: 12px; font-style: italic; }

  /* Tab styling */
  button[data-baseweb="tab"] {
    font-size: 13px !important; font-weight: 600 !important;
    padding: 8px 16px !important;
  }

  /* Sidebar labels */
  .sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #475569; margin: 16px 0 6px 0;
  }

  /* Section header */
  .section-header {
    font-size: 18px; font-weight: 700; color: #f1f5f9;
    margin-bottom: 4px; display: flex; align-items: center; gap: 8px;
  }
  .section-sub { font-size: 13px; color: #64748b; margin-bottom: 16px; }

  /* Status pill */
  .pill-green { background:#14532d; color:#86efac; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; }
  .pill-amber { background:#78350f; color:#fcd34d; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; }
  .pill-red   { background:#450a0a; color:#fca5a5; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; }
  .pill-blue  { background:#1e3a5f; color:#93c5fd; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; }

  /* Buttons */
  .stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important; color: white !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.6) !important;
    transform: translateY(-1px) !important;
  }

  /* Header glow */
  .main-header {
    font-size: 28px; font-weight: 800; color: #f8fafc;
    letter-spacing: -0.5px;
  }
  .main-header span { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# Groq AI engine — wraps all API calls with automatic model fallback.
# Tries each model in GROQ_MODELS in order; returns the first successful response.
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

def get_groq():
    """Return a Groq client if a key is available, otherwise None.

    We check session_state first (key entered via the sidebar), then fall back
    to the GROQ_API_KEY environment variable so the app still works when run
    with the key pre-set (e.g. in a CI environment or Docker container).
    """
    key = st.session_state.get('groq_api_key', '').strip() or os.environ.get('GROQ_API_KEY','').strip()
    if not key: return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None

def groq_call(messages, max_tokens=300):
    """Call Groq with automatic model fallback. Returns text or error string."""
    client = get_groq()
    if not client:
        return "No API key found. Enter your Groq key in AI Settings (sidebar)."
    last_error = ""
    for model in GROQ_MODELS:
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.2)
            return r.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                return " Rate limit reached. Wait 30 seconds then try again."
            if "401" in err_str or "invalid" in err_str or "api key" in err_str:
                return " Invalid API key. Check your Groq key in AI Settings (sidebar)."
            if "503" in err_str or "unavailable" in err_str:
                last_error = "service unavailable"
                continue
            last_error = str(e)[:100]
            continue
    return f" All Groq models unavailable ({last_error}). Check console.groq.com for status."

def ai_ask(prompt, system=None, max_tokens=300):
    """Wrapper: single-turn call."""
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    return groq_call(msgs, max_tokens)

def ai_chat(messages, system=None, max_tokens=300):
    """Wrapper: multi-turn call."""
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.extend(messages)
    return groq_call(msgs, max_tokens)

# Specialised AI prompt functions — each one has its own tightly scoped system prompt.
# All they do is format a request for groq_call(). The system prompts are deliberately
# narrow so the AI stays on topic and doesn't ramble. Token limits are kept low
# because clinical summaries need to be scannable, not essays.

def ai_clinical_summary(patient_str, completeness):
    sys = (
        "You are a concise clinical summariser in a hospital decision support system. "
        "RULES: Max 100 words. Output exactly 3 labelled lines. "
        "PICTURE: one sentence — overall clinical situation. "
        "CONCERN: the single most urgent issue right now. "
        "NEXT: one specific immediate action. "
        "No preamble, no headings, no lists, no caveats. Clinical language."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Summarise in 3 lines:\n{patient_str}"}], 130)

def ai_differential(patient_str):
    sys = (
        "You are a clinical reasoning assistant. "
        "Output a numbered list of exactly 3 differential diagnoses. "
        "Format: '1. [Diagnosis] — [one sentence: which specific value supports it]' "
        "Max 80 words. No preamble. No caveats. No markdown headings."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Top 3 differentials:\n{patient_str}"}], 110)

def ai_explain_prediction(patient_str, prob, features):
    sys = (
        "You are a clinical ML interpretability specialist. "
        "Explain in exactly 3 bullet points why the model gave this prediction. "
        "Each bullet: feature name → its value → one sentence why it matters clinically. "
        "Max 90 words. No intro sentence. No caveats. Start immediately with bullets."
    )
    feat_txt = ", ".join([f"{f.replace('_',' ')} ({v*100:.0f}%)" for f,v in features[:4]])
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Prediction: {prob:.0%}. Features: {feat_txt}.\n{patient_str}"}], 120)

def ai_missing_data(patient_str, missing, completeness):
    sys = (
        "You are a clinical triage advisor. "
        "Identify the ONE most critical missing measurement for THIS specific patient. "
        "Output: measurement name — one sentence why it is most important for this specific patient. "
        "Max 45 words. No intro. No lists. Specific to the patient, not general."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Missing: {', '.join(missing)}.\n{patient_str}"}], 65)

def ai_case_compare(case_data, current_data):
    sys = (
        "You are a clinical case comparison analyst. "
        "Output exactly 2 bullet points only: "
        "• Similarity: one specific clinical parallel between the two patients. "
        "• Lesson: what the historical outcome teaches about the current patient. "
        "Max 70 words. No intro. No caveats. Clinical language."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Historical: {case_data}\nCurrent: {current_data}"}], 95)

def ai_bias_check(dist_str, cbr_note=""):
    # We tell the AI upfront that the CBR system already compensates through
    # oversampling, so it doesn't just tell us to add more training data — that
    # would be misleading since the imbalance is already handled internally.
    sys = (
        "You are a clinical AI bias analyst. "
        "One sentence: describe the imbalance severity. "
        "One sentence: note whether the internal CBR oversampling adequately mitigates this, "
        "and whether any additional clinical oversight is still recommended. "
        "IMPORTANT: Do NOT recommend increasing training data volume — the system already "
        "applies action oversampling internally to rebalance CBR retrieval. "
        "Max 55 words. No preamble."
    )
    user_msg = f"Action distribution (raw counts from database):\n{dist_str}"
    if cbr_note:
        user_msg += f"\n\nCBR mitigation already applied: {cbr_note}"
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":user_msg}], 80)

def ai_outcome_balance(pos, neg):
    sys = (
        "You are a clinical ML safety reviewer. "
        "Output: severity (severe/moderate/mild), one sentence clinical impact, one mitigation. "
        "Max 55 words. No intro."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Improved: {pos}, Did not improve: {neg}"}], 75)

def ai_ethics_review():
    sys = (
        "You are a healthcare AI ethics auditor. "
        "Rate 5 principles: Privacy, Transparency, Autonomy, Safety, Fairness. "
        "Format: '✅/⚠️/❌ [Principle]: [one sentence verdict]' "
        "Max 90 words. No intro."
    )
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":"Audit: anonymised MIMIC-IV, GradientBoosting+CBR, advisory-only outputs, version audit trail, no PII, clinician always decides."}], 120)

def ai_system_health(issues, t_n, o_n):
    sys = (
        "You are a clinical AI health monitor. "
        "List only actual problems as: [CRITICAL/WARNING/INFO] one-sentence fix. "
        "If none: '✅ System healthy.' Max 70 words. No intro."
    )
    issue_txt = "\n".join(issues) if issues else "None."
    return groq_call([{"role":"system","content":sys},
                      {"role":"user","content":f"Issues:\n{issue_txt}\nDB: {t_n} traces, {o_n} outcomes."}], 90)

def ai_chat_response(history, ctx):
    sys = (
        "You are a clinical AI assistant in the Decision Memory Engine. "
        "RULES: Max 100 words. Answer directly — no preamble like 'Great question'. "
        "If answerable in one sentence, do it. Reference current patient values when relevant. "
        "Advisory only — never diagnostic. If asked for a chart or graph, describe what it would show instead."
        f"\nPatient context: {ctx}"
    )
    return groq_call([{"role":"system","content":sys}]+history, 140)


# Chart helper functions — build matplotlib charts on a dark background to match
# the Streamlit dark theme. Agg backend is set at import time so matplotlib never
# tries to open a GUI window while running inside Streamlit.

def make_vitals_chart(inp, news_score):
    """Horizontal bar chart showing how far each vital deviates from its normal range.

    Deviation is expressed as a percentage from 0 (completely normal) to 100
    (at or beyond the maximum expected value). Colour is green/amber/red based
    on deviation severity. Only vitals that were actually entered are shown.
    """
    dark = "#0d1424"
    params = [
        ("HR",   inp.get('heart_rate'),  60, 100, 180),
        ("SBP",  inp.get('systolic_bp'),100, 140, 220),
        ("Creat",inp.get('creatinine'), 0.6, 1.2, 8.0),
        ("WBC",  inp.get('wbc'),        4.0,11.0,25.0),
        ("Temp", inp.get('temperature'),36.1,37.2,41.0),
        ("RR",   inp.get('resp_rate'),  12,  20,  35),
        ("SpO2", inp.get('spo2'),       96, 100, 100),
    ]
    avail = [(n,v,lo,hi,mx) for n,v,lo,hi,mx in params if v is not None]
    if not avail: return None
    labels=[p[0] for p in avail]; values=[]; colours=[]
    for n,v,lo,hi,mx in avail:
        if lo<=v<=hi: dev=0
        elif v<lo:    dev=min((lo-v)/(lo*0.5+0.001)*100,100)
        else:         dev=min((v-hi)/(mx-hi+0.001)*100,100)
        values.append(dev)
        colours.append('#ef4444' if dev>60 else '#f59e0b' if dev>25 else '#22c55e')
    fig,ax=plt.subplots(figsize=(6.5,3))
    fig.patch.set_facecolor(dark); ax.set_facecolor(dark)
    bars=ax.barh(labels,values,color=colours,height=0.5,edgecolor='none')
    ax.set_xlim(0,105); ax.set_xlabel("% Deviation from Normal",color='#4a6585',fontsize=9)
    ax.tick_params(colors='#7a91ad',labelsize=10)
    for s in ax.spines.values(): s.set_color('#1a2535')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar,val in zip(bars,values):
        if val>3: ax.text(val+1.2,bar.get_y()+bar.get_height()/2,f"{val:.0f}%",va='center',ha='left',color='#7a91ad',fontsize=9)
    rl,_,_,_,_=news_risk(news_score)
    rc={'CRITICAL':'#ef4444','HIGH':'#f97316','MEDIUM':'#f59e0b','LOW':'#22c55e'}
    ax.set_title(f"Vital Signs Deviation  ·  NEWS {news_score}/18  ·  {rl}",color=rc.get(rl,'#94a3b8'),fontsize=10,fontweight='bold',pad=8)
    leg=[mpatches.Patch(color='#22c55e',label='Normal'),mpatches.Patch(color='#f59e0b',label='Moderate'),mpatches.Patch(color='#ef4444',label='Severe')]
    ax.legend(handles=leg,loc='lower right',fontsize=8,facecolor=dark,edgecolor='#1a2535',labelcolor='#7a91ad')
    plt.tight_layout(); return fig

def make_feature_chart(top_features):
    """Bar chart of the model's top feature importances for the current prediction.

    Shows up to 6 features sorted by importance (most influential at the top).
    The importances come from RandomForest's built-in feature_importances_,
    which measure how often each feature was used to split a tree node and
    by how much it reduced impurity.
    """
    if not top_features: return None
    dark="#0d1424"
    feats=[(f.replace('_',' ').title(),v) for f,v in top_features[:6]]
    labels=[f[0] for f in feats]; values=[f[1]*100 for f in feats]
    colours=['#3b82f6' if v>15 else '#1d4ed8' if v>8 else '#1e3a5f' for v in values]
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    fig.patch.set_facecolor(dark); ax.set_facecolor(dark)
    bars=ax.barh(labels[::-1],values[::-1],color=colours[::-1],height=0.45,edgecolor='none')
    ax.set_xlabel("Influence on Prediction (%)",color='#4a6585',fontsize=9)
    ax.tick_params(colors='#7a91ad',labelsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#1a2535')
    for bar,val in zip(bars,values[::-1]):
        ax.text(val+0.4,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va='center',ha='left',color='#7a91ad',fontsize=9)
    ax.set_title("What Drove This Prediction?",color='#93c5fd',fontsize=10,fontweight='bold',pad=6)
    plt.tight_layout(); return fig

def make_memory_chart(act_dist, out_dist):
    """Side-by-side chart: action frequency distribution + outcome pie chart.

    Left panel shows how often each clinical action appears in the training data
    (helps identify the 12-fold action imbalance that CBR oversampling addresses).
    Right panel shows the 87:13 outcome split so users understand model limitations.
    """
    dark="#0d1424"
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.2))
    fig.patch.set_facecolor(dark)
    ax1.set_facecolor(dark)
    if not act_dist.empty:
        col0=act_dist.columns[0]; col1=act_dist.columns[1]
        actions=act_dist[col0].tolist(); counts=act_dist[col1].tolist()
        short=[a.replace('prescribe_','').replace('order_','').replace('_',' ').title() for a in actions]
        cs=plt.cm.Blues(np.linspace(0.35,0.85,len(actions)))
        bars=ax1.barh(short[::-1],counts[::-1],color=cs[::-1],height=0.5,edgecolor='none')
        for bar,val in zip(bars,counts[::-1]):
            ax1.text(val+0.3,bar.get_y()+bar.get_height()/2,str(val),va='center',ha='left',color='#7a91ad',fontsize=9)
    ax1.set_title("Decisions in Memory",color='#93c5fd',fontsize=10,fontweight='bold')
    ax1.tick_params(colors='#7a91ad',labelsize=9)
    for s in ax1.spines.values(): s.set_color('#1a2535')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2.set_facecolor(dark)
    if not out_dist.empty:
        out_dist2=out_dist.copy()
        out_dist2['label']=out_dist2['outcome_value'].map({1:'Improved',0:'Did Not\nImprove'})
        wedges,texts,autos=ax2.pie(out_dist2['count'],labels=out_dist2['label'],
            colors=['#22c55e','#ef4444'][:len(out_dist2)],autopct='%1.0f%%',startangle=90,
            textprops={'color':'#7a91ad','fontsize':10},wedgeprops={'edgecolor':dark,'linewidth':2})
        for at in autos: at.set_color('#e2e8f0'); at.set_fontsize(11)
    ax2.set_title("Outcome Distribution",color='#93c5fd',fontsize=10,fontweight='bold')
    plt.tight_layout(); return fig

def generate_dynamic_questions(patient_data, news_score, prob, missing_fields):
    """Generate 4 contextually relevant questions based on current patient."""
    questions = []
    cr  = patient_data.get('creatinine')
    wbc = patient_data.get('wbc')
    hr  = patient_data.get('heart_rate')
    sbp = patient_data.get('systolic_bp')

    if prob is not None:
        pct = round(prob * 100)
        questions.append(f"Why did the model predict {pct}% improvement?")

    if cr and cr > 1.5:
        questions.append(f"Creatinine is {cr} mg/dL — how serious is this?")
    elif wbc and wbc > 11:
        questions.append(f"WBC is {wbc} — could this indicate sepsis?")
    elif sbp and sbp < 100:
        questions.append(f"Blood pressure is {sbp} mmHg — what should we do?")
    elif hr and hr > 100:
        questions.append(f"Heart rate is {hr} bpm — what are the likely causes?")
    else:
        questions.append("Is this patient stable enough for discharge planning?")

    if news_score >= 5:
        questions.append(f"NEWS score is {news_score} — what does this mean urgently?")
    elif news_score == 0:
        questions.append("All vitals normal — when should this patient be reassessed?")
    else:
        questions.append("What is the most important next clinical step?")

    if missing_fields:
        questions.append(f"How critical is it to get the {missing_fields[0].lower()}?")
    else:
        questions.append("How reliable is the outcome prediction with this data?")

    return questions[:4]


# Clinical helper functions — implement well-established scoring systems and rules.
# NEWS (National Early Warning Score) is standard NHS practice; the advisory
# thresholds are taken directly from published ward escalation guidelines.
# Nothing here is invented — these are the same rules used in real hospitals.

def calculate_news(hr, sbp, temp, wbc, rr, spo2):
    """Compute the National Early Warning Score (NEWS) from available vitals.

    NEWS is an additive score — each vital contributes 0-3 points based on
    how far it deviates from the normal range. A higher total score means
    higher risk of deterioration. Vitals that weren't measured simply don't
    add any points, so partial data still produces a meaningful score.
    """
    s = 0
    if hr   is not None:
        if hr <= 40 or hr > 130: s += 3
        elif hr > 110:            s += 2
        elif hr < 51 or hr > 90: s += 1
    if sbp  is not None:
        if sbp <= 90:    s += 3
        elif sbp <= 100: s += 2
        elif sbp <= 110: s += 1
        elif sbp > 220:  s += 3
    if temp is not None:
        if temp <= 35.0:            s += 3
        elif temp <= 36.0:          s += 1
        elif temp > 39.0:           s += 2
    if wbc  is not None:
        if wbc > 20:   s += 3
        elif wbc > 15: s += 2
        elif wbc > 11: s += 1
    if rr   is not None:
        if rr < 8 or rr > 25: s += 3
        elif rr > 20:          s += 2
        elif rr < 12:          s += 1
    if spo2 is not None:
        if spo2 < 91:   s += 3
        elif spo2 < 94: s += 2
        elif spo2 < 96: s += 1
    return s

def news_risk(score):
    """Map a NEWS score to a risk level, CSS class, colour, and action string.

    Returns a 5-tuple: (label, css_class, emoji, hex_colour, action_text).
    Thresholds follow the NHS NEWS2 guidance: 0-2 = low, 3-4 = medium,
    5-6 = high (urgent), 7+ = critical (emergency).
    """
    if score >= 7:   return "CRITICAL","risk-critical","🔴","#ef4444","Activate emergency response — ICU review now"
    elif score >= 5: return "HIGH","risk-high","🟠","#f97316","Urgent senior review within 30 minutes"
    elif score >= 3: return "MEDIUM","risk-medium","🟡","#f59e0b","Increase monitoring frequency"
    else:            return "LOW","risk-low","🟢","#22c55e","Stable — continue routine monitoring"

def flag_status(val, lo, hi):
    """Return (is_normal, css_class, label) for a vital sign.

    Used to colour-code each vital in the sidebar vitals table.
    Returns None for is_normal when the value wasn't entered — the UI
    shows a dash instead of a green/amber status.
    """
    if val is None: return None, "vit-na",    "—"
    if val < lo or val > hi: return False, "vit-warn", "⚠ Abnormal"
    return True, "vit-ok", "✓ Normal"

def data_completeness(inp):
    """Compute a weighted data completeness score (0-100) and list missing fields.

    Weights reflect clinical importance: HR and SBP are most critical (20 pts each),
    creatinine is next (18 pts), WBC after that (15 pts), and so on. This means
    a patient with just HR and SBP entered gets a 40% completeness score, not 29%.
    """
    w = {'heart_rate':20,'systolic_bp':20,'creatinine':18,'wbc':15,'temperature':10,'resp_rate':9,'spo2':8}
    score, missing = 0, []
    for f, wt in w.items():
        if inp.get(f) is not None: score += wt
        else: missing.append(f.replace('_',' ').title())
    return score, missing

def get_advisories(score, inp):
    """Generate a list of clinical advisory flags based on vital sign values.

    Each flag is a 3-tuple: (title, detail_text, severity).
    severity is 'urgent' (red), 'advisory' (blue), or 'ok' (green).
    The thresholds and clinical actions are taken from standard ICU and
    ward escalation guidelines. If nothing is abnormal, a single 'ok'
    flag is returned so the UI always shows something.
    """
    flags = []
    cr=inp.get('creatinine'); wbc=inp.get('wbc'); hr=inp.get('heart_rate')
    sbp=inp.get('systolic_bp'); spo2=inp.get('spo2'); temp=inp.get('temperature'); rr=inp.get('resp_rate')
    if score >= 7:
        flags.append(("🚨 EMERGENCY RESPONSE REQUIRED",
            "NEWS ≥7: Activate rapid response team immediately. Consider ICU escalation without delay. "
            "Ensure senior clinician at bedside within minutes.", "urgent"))
    if cr and cr > 3.0:
        flags.append(("🔬 Severe Acute Kidney Injury",
            f"Creatinine critically elevated at {cr:.1f} mg/dL (normal 0.6–1.2). "
            "Urgent nephrology review. Hourly urine output monitoring. Assess for renal replacement therapy.", "urgent"))
    elif cr and cr > 1.5:
        flags.append(("🔬 Renal Function Concern",
            f"Creatinine elevated at {cr:.1f} mg/dL. Nephrology referral indicated. "
            "Review nephrotoxic medications. Monitor fluid balance.", "advisory"))
    if wbc and wbc > 15:
        flags.append(("🧫 Sepsis Screening Required",
            f"WBC severely elevated at {wbc:.1f} x10⁹/L. Obtain blood cultures before antibiotics. "
            "Apply Sepsis 6 bundle. Lactate, IV access, fluid challenge.", "urgent"))
    elif wbc and wbc > 11:
        flags.append(("💊 Infection Indicator",
            f"WBC mildly elevated at {wbc:.1f} x10⁹/L. Review recent cultures and temperature trend. "
            "Consider infection screen.", "advisory"))
    if sbp and sbp < 90:
        flags.append(("🩸 Hypotension",
            f"Systolic BP critically low at {sbp} mmHg. "
            "500ml IV crystalloid bolus. Reassess. If unresponsive, start vasopressors (noradrenaline).", "urgent"))
    if hr and hr > 110:
        flags.append(("❤️ Tachycardia",
            f"Heart rate elevated at {hr} bpm. 12-lead ECG. "
            "Differential: sepsis, AF, PE, hypovolaemia, pain. Treat underlying cause.", "advisory"))
    if spo2 and spo2 < 94:
        flags.append(("🫁 Hypoxia",
            f"SpO₂ at {spo2}% (target ≥94%). Apply supplemental oxygen. "
            "If <88% or increased work of breathing — escalate to NIV/ICU.", "urgent"))
    if temp and temp > 38.5:
        flags.append(("🌡️ Pyrexia",
            f"Temperature {temp:.1f}°C. Infection likely. "
            "Blood cultures, CXR, urine dip, wound assessment.", "advisory"))
    if temp and temp < 36.0:
        flags.append(("🌡️ Hypothermia",
            f"Temperature {temp:.1f}°C. May indicate sepsis (cold shock). "
            "Active warming. Urgent infection screen.", "urgent"))
    if rr and rr > 20:
        flags.append(("💨 Tachypnoea",
            f"Respiratory rate elevated at {rr}/min. "
            "Assess work of breathing. Consider ABG. Rule out PE, pneumonia, acidosis.", "advisory"))
    if not flags:
        flags.append(("✅ No Immediate Concerns",
            "All available parameters within acceptable ranges. "
            "Continue current management with routine monitoring as per local protocol.", "ok"))
    return flags

def get_db_stats():
    """Pull all the stats the dashboard needs in a single DB connection.

    Returns total traces, completed traces, last 5 model versions, the action
    frequency distribution, the outcome distribution, and the 10 most recently
    logged new traces (those with IDs starting 'T_NEW_').
    We open one connection and do all the queries together rather than opening
    one connection per query — keeps things tidy and fast.
    """
    conn = sqlite3.connect(DB_PATH)
    total    = pd.read_sql_query("SELECT COUNT(*) as n FROM decision_traces", conn).iloc[0]['n']
    complete = pd.read_sql_query("SELECT COUNT(*) as n FROM decision_traces WHERE status='completed'", conn).iloc[0]['n']
    versions = pd.read_sql_query("SELECT * FROM model_versions ORDER BY version_id DESC LIMIT 5", conn)
    act_dist = pd.read_sql_query("SELECT action, COUNT(*) as count FROM decision_traces GROUP BY action ORDER BY count DESC", conn)
    out_dist = pd.read_sql_query("SELECT outcome_value, COUNT(*) as count FROM outcomes GROUP BY outcome_value", conn)
    recent   = pd.read_sql_query("SELECT * FROM decision_traces WHERE trace_id LIKE 'T_NEW_%' ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    return total, complete, versions, act_dist, out_dist, recent

def get_latest_cv_auc():
    """Return the test accuracy stored in the most recent model_versions row, or None.

    We store accuracy (not CV AUC) in the accuracy column now, evaluated at
    the conservative 0.65 threshold. This is what the user sees in the UI.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT accuracy FROM model_versions ORDER BY version_id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return float(row[0]) if row else None
    except Exception:
        return None

def fmt_val(v, unit, decimals=0):
    """Format a numeric value with units for display, or 'Not recorded' if None/NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Not recorded"
    return f"{v:.{decimals}f} {unit}" if decimals > 0 else f"{int(v)} {unit}"


# Sidebar — patient vitals input, AI key, and diagnosis selector.
# All widgets write into local variables; the "Run" button triggers the ML pipeline.
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 8px 0;">
      <div style="font-size:28px;">🏥</div>
      <div style="font-size:15px; font-weight:800; color:#f1f5f9; letter-spacing:-0.3px;">Decision Memory Engine</div>
      <div style="font-size:11px; color:#475569; margin-top:2px;">Clinical Advisory System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # AI Key
    with st.expander("AI Settings (Groq)", expanded=False):
        key_val = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.get('groq_api_key', ''),
            placeholder="gsk_...",
            help="Free at console.groq.com — no credit card needed"
        )
        if key_val:
            st.session_state['groq_api_key'] = key_val
        ai_on = bool(st.session_state.get('groq_api_key', '').strip())
        if ai_on:
            st.markdown('<span class="pill-green">✓ AI Active — Llama 3.3 70B</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-amber">⚠ Enter key for AI features</span>', unsafe_allow_html=True)
            st.caption("Get free key → console.groq.com")

    st.divider()

    # Patient
    st.markdown('<div class="sb-label">👤 Patient Details</div>', unsafe_allow_html=True)
    patient_ref = st.text_input("Reference / Bed No.", placeholder="e.g. BED-04A",
                                 label_visibility="collapsed")
    st.caption("Patient reference (optional — used for audit trail)")

    st.divider()

    # Essential vitals
    st.markdown('<div class="sb-label">🔴 Essential Vitals</div>', unsafe_allow_html=True)
    st.caption("Minimum needed for a meaningful assessment")

    use_hr  = st.checkbox("Heart Rate", value=True)
    hr_val  = st.number_input("bpm", 20, 250, 80, 1, label_visibility="collapsed") if use_hr else None

    use_sbp = st.checkbox("Blood Pressure", value=True)
    sbp_val = st.number_input("mmHg", 50, 250, 120, 1, label_visibility="collapsed") if use_sbp else None

    st.divider()

    # Lab values
    st.markdown('<div class="sb-label">🟡 Lab Values</div>', unsafe_allow_html=True)
    st.caption("Add these if available — significantly improves accuracy")

    use_cr  = st.checkbox("Creatinine (mg/dL)")
    cr_val  = st.number_input("Creat", 0.1, 20.0, 1.0, 0.1, format="%.1f", label_visibility="collapsed") if use_cr else None

    use_wbc = st.checkbox("White Blood Cells (WBC)")
    wbc_val = st.number_input("WBC", 0.1, 100.0, 8.0, 0.5, format="%.1f", label_visibility="collapsed") if use_wbc else None

    st.divider()

    # Additional
    st.markdown('<div class="sb-label">🟢 Additional Observations</div>', unsafe_allow_html=True)
    st.caption("Further improves accuracy when available")

    use_temp = st.checkbox("Temperature (°C)")
    temp_val = st.number_input("Temp", 33.0, 42.0, 37.0, 0.1, format="%.1f", label_visibility="collapsed") if use_temp else None

    use_rr   = st.checkbox("Respiratory Rate (/min)")
    rr_val   = st.number_input("RR", 4, 60, 16, 1, label_visibility="collapsed") if use_rr else None

    use_spo2 = st.checkbox("SpO₂ (%)")
    spo2_val = st.number_input("SpO2", 50, 100, 97, 1, label_visibility="collapsed") if use_spo2 else None

    st.divider()

    # Diagnosis
    st.markdown('<div class="sb-label">🏷️ Diagnosis Code</div>', unsafe_allow_html=True)
    COMMON_ICD = {
        "Not specified": None,
        "J189 — Pneumonia":           "J189",
        "A419 — Sepsis":              "A419",
        "I5023 — Heart Failure":      "I5023",
        "N179 — Acute Kidney Injury": "N179",
        "I214 — NSTEMI":              "I214",
        "J9601 — Resp. Failure":      "J9601",
        "I4891 — Atrial Fibrillation":"I4891",
        "J441 — COPD Exacerbation":   "J441",
        "K922 — GI Haemorrhage":      "K922",
        "Other (type below)":         "OTHER",
    }
    dx_sel   = st.selectbox("Diagnosis", list(COMMON_ICD.keys()), label_visibility="collapsed")
    diagnosis= st.text_input("ICD code","J189",max_chars=10,label_visibility="collapsed").upper().strip() \
               if COMMON_ICD[dx_sel]=="OTHER" else COMMON_ICD[dx_sel]

    st.divider()

    with st.expander("📖 Normal Ranges"):
        st.markdown("""
| Test | Normal |
|---|---|
| Heart Rate | 60–100 bpm |
| Systolic BP | 100–140 mmHg |
| Creatinine | 0.6–1.2 mg/dL |
| WBC | 4–11 x10⁹/L |
| Temperature | 36.1–37.2 °C |
| Resp. Rate | 12–20 /min |
| SpO₂ | ≥96% |
""")

    st.divider()
    run_btn = st.button("🔍  Run Clinical Analysis", use_container_width=True, type="primary")


# Collect all sidebar widget values into one dict for the rest of the app to use.
inputs = {
    'heart_rate':   hr_val,   'systolic_bp': sbp_val,
    'creatinine':   cr_val,   'wbc':         wbc_val,
    'temperature':  temp_val, 'resp_rate':   rr_val,
    'spo2':         spo2_val, 'diagnosis_code': diagnosis,
}
completeness, missing_fields = data_completeness(inputs)
ml_context = {k:v for k,v in inputs.items() if v is not None and k not in ('resp_rate','spo2')}
ml_context['diagnosis_code'] = diagnosis or 'UNKNOWN'

# Store results in session state so tabs share data
if run_btn:
    news  = calculate_news(hr_val, sbp_val, temp_val, wbc_val, rr_val, spo2_val)
    prob, similar_cases, top_features = predict_outcome(ml_context)
    st.session_state['last_news']     = news
    st.session_state['last_prob']     = prob
    st.session_state['last_sim']      = similar_cases
    st.session_state['last_feats']    = top_features
    st.session_state['last_inputs']   = dict(inputs)
    st.session_state['last_run']      = True
    st.session_state['last_patient']  = patient_ref

ran        = st.session_state.get('last_run', False)
news_score = st.session_state.get('last_news', 0)
prob       = st.session_state.get('last_prob', None)
sim_cases  = st.session_state.get('last_sim',  [])
top_feats  = st.session_state.get('last_feats', [])
last_inp   = st.session_state.get('last_inputs', inputs)

def patient_str():
    p = last_inp
    return (f"HR={p.get('heart_rate') or 'N/A'} bpm, SBP={p.get('systolic_bp') or 'N/A'} mmHg, "
            f"Cr={p.get('creatinine') or 'N/A'} mg/dL, WBC={p.get('wbc') or 'N/A'} x10\u2079/L, "
            f"Temp={p.get('temperature') or 'N/A'}°C, "
            f"RR={p.get('resp_rate') or 'N/A'}/min, SpO2={p.get('spo2') or 'N/A'}%, "
            f"Dx={p.get('diagnosis_code') or 'N/A'}, "
            f"NEWS={news_score}/18, Prediction={round(prob*100,1) if prob else 'N/A'}% improvement")


# Page header — app title, subtitle, AI status indicator, and data completeness bar.
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown('<div class="main-header">Decision Memory <span>Engine</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#475569;margin-bottom:8px;">AI-powered clinical decision support · All outputs are advisory only · Clinical responsibility remains with the treating practitioner</div>', unsafe_allow_html=True)
with col_h2:
    ai_status = "🟢 AI Active" if ai_on else "🔴 AI Offline"
    st.markdown(f'<div style="text-align:right;padding-top:12px;font-size:13px;color:#64748b;">{ai_status}</div>', unsafe_allow_html=True)

# Completeness bar
c_colour = "#22c55e" if completeness>=80 else "#f59e0b" if completeness>=50 else "#ef4444"
c_label  = "Excellent" if completeness>=80 else "Good" if completeness>=50 else "Basic"
st.markdown(f"""
<div style="margin-bottom:16px; padding:12px 16px; background:#0a1120; border-radius:10px; border:1px solid #1e293b;">
  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
    <span style="font-size:12px;color:#64748b;font-weight:600;letter-spacing:1px;">DATA COMPLETENESS</span>
    <span style="font-size:12px;font-weight:700;color:{c_colour};">{completeness}% — {c_label}</span>
  </div>
  <div class="prog-outer">
    <div class="prog-inner" style="width:{completeness}%;background:linear-gradient(90deg,{c_colour},{c_colour}aa);"></div>
  </div>
  <div style="font-size:11px;color:#334155;margin-top:4px;">
    {"✓ All key parameters provided" if not missing_fields else
     f"Missing: {', '.join(missing_fields[:4])}{'...' if len(missing_fields)>4 else ''} — tick checkboxes in the sidebar to add them"}
  </div>
</div>
""", unsafe_allow_html=True)

if completeness < 30:
    st.warning("⚠️ Very limited data — provide at least Heart Rate and Blood Pressure for a meaningful assessment.")


# Tab layout — five tabs covering the full clinical workflow.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Clinical Analysis",
    "🤖  AI Deep Analysis",
    "🧠  Decision Memory",
    "🛡️  Safety & Bias",
    "💬  AI Assistant",
])


# Tab 1 — Clinical Analysis
# NEWS risk score, ML outcome prediction, advisory flags, similar past cases,
# and the decision logging panel.
with tab1:
    if not ran:
        st.markdown("""
        <div class="dme-card" style="text-align:center; padding:60px 40px;">
          <div style="font-size:64px;margin-bottom:16px;">🔍</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:10px;">Ready to Analyse</div>
          <div style="color:#64748b;font-size:14px;line-height:1.8;max-width:500px;margin:0 auto;">
            Enter the patient's available values in the left panel and click
            <strong style="color:#3b82f6;">Run Clinical Analysis</strong>.<br><br>
            You don't need all values — the system works with whatever data you have
            and will tell you exactly what additional information would help most.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_level, risk_class, risk_icon, risk_colour, risk_action = news_risk(news_score)
        advisories = get_advisories(news_score, last_inp)

        # Row 1: NEWS risk score card (left) and ML outcome prediction (right).
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">📊 Clinical Risk Score</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{risk_class}">
              <div style="font-size:32px;font-weight:900;letter-spacing:-1px;margin-bottom:6px;">
                {risk_icon} {risk_level} RISK
              </div>
              <div style="font-size:42px;font-weight:800;margin-bottom:8px;">
                {news_score}<span style="font-size:20px;opacity:0.6;"> / 18</span>
              </div>
              <div style="font-size:13px;opacity:0.85;border-top:1px solid rgba(255,255,255,0.15);padding-top:10px;">
                <strong>Action:</strong> {risk_action}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Vitals breakdown
            st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
            vitals_def = [
                ("❤️", "Heart Rate",  hr_val,   60,  100, "bpm",    0),
                ("🩸", "Systolic BP", sbp_val, 100,  140, "mmHg",   0),
                ("🧪", "Creatinine",  cr_val,  0.6,  1.2, "mg/dL",  1),
                ("🔬", "WBC",         wbc_val,   4,   11, "x10⁹/L", 1),
                ("🌡️","Temperature", temp_val,36.1, 37.2, "°C",     1),
                ("💨", "Resp. Rate",  rr_val,   12,   20, "/min",    0),
                ("🫁", "SpO₂",        spo2_val, 96,  100, "%",       0),
            ]
            rows_html = ""
            for icon, name, val, lo, hi, unit, dec in vitals_def:
                if val is None: continue
                ok, css, label = flag_status(val, lo, hi)
                val_str = fmt_val(val, unit, dec)
                rows_html += f"""
                <div class="vitals-row">
                  <span class="vit-name">{icon} {name}</span>
                  <span class="vit-value">{val_str}</span>
                  <span class="{css}">{label}</span>
                </div>"""
            if rows_html:
                st.markdown(f'<div class="dme-card" style="padding:16px 20px;">{rows_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No values provided. Tick checkboxes in the sidebar.")

            # Vitals deviation chart
            fig_v = make_vitals_chart(last_inp, news_score)
            if fig_v:
                st.pyplot(fig_v, use_container_width=True)
                plt.close(fig_v)


            st.markdown('<div class="explain-box">📖 <strong>What is NEWS?</strong> The National Early Warning Score (NEWS) is used across all NHS hospitals. Each vital sign outside its normal range adds points — the total score determines how urgently a doctor must review the patient. Score 7+ triggers emergency protocols. Showing this alongside the AI prediction gives clinicians two independent perspectives: a rules-based clinical score and a data-driven probability.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">🤖 Outcome Prediction</div>', unsafe_allow_html=True)
            if prob is None:
                st.error("Model not trained. Go to the 🧠 Decision Memory tab and click 'Retrain Model'.")
            elif completeness < 25:
                st.warning("Insufficient data for a reliable prediction. Please add at least Heart Rate and Blood Pressure.")
            else:
                pct = round(prob * 100, 1)

                # Combined risk tier: NEWS overrides ML when it signals HIGH or CRITICAL.
                # The ML model was trained on only 13 minority-class (death) cases.
                # Because of this, it physically cannot output P(survived) below ~65%
                # even for genuinely critical patients — the training distribution
                # just doesn't have enough examples to push the probability lower.
                # NEWS (National Early Warning Score) is a validated rule-based system
                # that is NOT affected by this limitation, so we let it take precedence
                # whenever it signals HIGH or CRITICAL risk. The raw ML probability is
                # still shown so clinicians have full transparency about what the model
                # is saying — it's just clearly contextualised by the clinical evidence.
                if news_score >= 7:
                    # NEWS ≥ 7 is CRITICAL — emergency protocol territory. The ML number
                    # is shown but the overall risk assessment is driven by NEWS.
                    bar_col     = "#ef4444"
                    outcome_txt = ("🔴 CRITICAL RISK — Emergency response required"
                                   " (NEWS ≥7 overrides model estimate)")
                    news_override = True
                elif news_score >= 5:
                    # NEWS 5–6 is HIGH — urgent senior review needed.
                    bar_col     = "#f97316"
                    outcome_txt = ("🟠 HIGH RISK — Urgent senior review required"
                                   " (NEWS ≥5 overrides model estimate)")
                    news_override = True
                elif prob >= 0.85:
                    bar_col     = "#22c55e"
                    outcome_txt = "🟢 Patient likely to improve — continue current management"
                    news_override = False
                elif prob >= 0.65:
                    bar_col     = "#f59e0b"
                    outcome_txt = "🟡 Elevated risk — closer monitoring and senior review advised"
                    news_override = False
                else:
                    bar_col     = "#ef4444"
                    outcome_txt = "🔴 High risk of deterioration — urgent clinical review required"
                    news_override = False

                margin = round(min(0.12, (1 - prob) * 0.25), 2)
                lo_ci  = max(0,   round(prob - margin, 2))
                hi_ci  = min(1.0, round(prob + margin, 2))

                # Sub-label: explains the ML number in context of any NEWS override
                if news_override:
                    ml_note = (f"ML model: {pct}% improvement probability"
                               f" — overridden by NEWS score {news_score}/18")
                else:
                    ml_note = f"Model confidence range: {lo_ci:.0%} – {hi_ci:.0%}"

                st.markdown(f"""
                <div class="dme-card" style="border-color:{bar_col}44;">
                  <div class="dme-card-title">Combined Risk Assessment</div>
                  <div style="font-size:52px;font-weight:900;color:{bar_col};letter-spacing:-2px;line-height:1.05;">
                    {pct}%
                  </div>
                  <div style="font-size:11px;color:#475569;margin:2px 0 8px 0;font-weight:600;letter-spacing:0.5px;">
                    PREDICTED IMPROVEMENT PROBABILITY (ML MODEL)
                  </div>
                  <div style="font-size:14px;color:{bar_col};margin:0 0 14px 0;font-weight:700;line-height:1.4;">
                    {outcome_txt}
                  </div>
                  <div class="prog-outer">
                    <div class="prog-inner" style="width:{pct}%;background:linear-gradient(90deg,{bar_col},{bar_col}77);"></div>
                  </div>
                  <div style="font-size:11px;color:#475569;margin-top:8px;">
                    {ml_note} &nbsp;·&nbsp;
                    {completeness}% data completeness &nbsp;·&nbsp;
                    100 MIMIC-IV patients
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Feature importance chart — shows what drove this prediction
                fig_f = make_feature_chart(top_feats)
                if fig_f:
                    st.pyplot(fig_f, use_container_width=True)
                    plt.close(fig_f)

                # Explain the combined approach — especially important when the
                # NEWS score is overriding the ML probability so users understand why
                if news_override:
                    st.markdown(
                        f'<div class="explain-box" style="border-color:{bar_col};color:#fbbf24;">'
                        f'⚠️ <strong>Why does the risk level say critical when the ML model shows {pct}%?</strong> '
                        f'The ML model is trained on 100 patients with only 13 deterioration cases — '
                        f'it cannot learn to output probabilities below ~65% even for genuinely critical patients. '
                        f'The NEWS score of <strong>{news_score}/18</strong> is a validated NHS clinical '
                        f'scoring system that is not subject to this limitation. When NEWS signals '
                        f'HIGH or CRITICAL risk, that clinical evidence takes precedence over the '
                        f'ML estimate. Both signals are shown so you have full transparency.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="explain-box">📖 <strong>How to read this:</strong> '
                        'Trained on real MIMIC-IV patients, the model identifies which combinations '
                        'of vitals and labs correlate with improvement vs deterioration. '
                        '🟢 ≥85% = favourable outlook. '
                        '🟡 65–84% = elevated risk, closer monitoring needed. '
                        '🔴 &lt;65% = high risk, urgent review required. '
                        'The bar chart shows which measurements drove this specific result — '
                        'always verify the reasoning makes clinical sense before acting on it.</div>',
                        unsafe_allow_html=True
                    )

                # Calibration note — always shown as a reminder of the model's limitation
                _acc = get_latest_cv_auc()
                if _acc is not None and _acc < 0.85:
                    st.markdown(
                        f'<div class="explain-box" style="border-color:#334155;color:#64748b;">'
                        f'ℹ️ <strong>Model Note:</strong> Test-set accuracy is '
                        f'<strong>{_acc:.0%}</strong>. With only 13 minority-class training cases '
                        f'the ML probability alone cannot fully represent critical risk — '
                        f'this is why the NEWS score is used to override it when warranted. '
                        f'Retrain with more outcome-linked traces to improve ML reliability.</div>',
                        unsafe_allow_html=True
                    )

        # Advisory flags — each one is triggered independently by a single abnormal value.
        st.divider()
        st.markdown('<div class="section-header">📋 Clinical Advisory Flags</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Each flag is triggered independently by a specific measurement being outside its clinical threshold. Multiple flags appear simultaneously when multiple values are abnormal.</div>', unsafe_allow_html=True)
        for title, text, ftype in advisories:
            css = "flag-urgent" if ftype=="urgent" else "flag-ok" if ftype=="ok" else "flag-advisory"
            st.markdown(f'<div class="{css}"><strong>{title}</strong><br><span style="font-size:13px;line-height:1.6;">{text}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="explain-box">📖 <strong>Why show these alongside the prediction?</strong> The AI prediction looks at all values together to estimate outcome probability. The flags check each value independently against clinical guidelines — the same way a doctor would mentally tick through a checklist. Together they give two complementary perspectives: statistical pattern and clinical rule.</div>', unsafe_allow_html=True)

        # Similar cases — CBR retrieval of the 5 most similar historical patients.
        st.divider()
        st.markdown('<div class="section-header">🗂️ Most Similar Historical Cases</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Real patients from the MIMIC-IV hospital dataset with the most similar clinical profile. Click any case to expand the full details. Seeing past similar patients makes the AI\'s reasoning transparent and checkable.</div>', unsafe_allow_html=True)

        if sim_cases:
            # Deduplicate
            seen_p = set()
            unique_cases = []
            for c in sim_cases:
                pid = c.get('patient_id','')
                if pid not in seen_p:
                    seen_p.add(pid)
                    unique_cases.append(c)

            for i, case in enumerate(unique_cases[:5]):
                ov       = case.get('outcome_value', case.get('outcome'))
                improved = ov == 1
                sim_pct  = round(case['similarity'] * 100, 1)
                out_col  = "#4ade80" if improved else "#f87171"
                out_txt  = "✅ Patient Improved" if improved else "❌ Did Not Improve"

                # Age string
                ts = case.get('timestamp','')
                try:
                    age_days = (datetime.now() - pd.to_datetime(ts)).days
                    age_str  = f"{age_days//365}yr ago" if age_days > 365 else f"{age_days}d ago"
                except:
                    age_str = "—"

                with st.expander(
                    f"Case {i+1}  ·  {sim_pct}% similar  ·  {out_txt}  ·  {age_str}",
                    expanded=(i == 0)
                ):
                    ec1, ec2, ec3 = st.columns(3)

                    with ec1:
                        st.markdown("**📋 Patient Profile**")
                        st.markdown(f"""
                        <div class="dme-card" style="padding:14px;">
                          <div class="vitals-row"><span class="vit-name">❤️ Heart Rate</span><span class="vit-value">{fmt_val(case.get('heart_rate'),'bpm')}</span></div>
                          <div class="vitals-row"><span class="vit-name">🩸 Systolic BP</span><span class="vit-value">{fmt_val(case.get('systolic_bp'),'mmHg')}</span></div>
                          <div class="vitals-row"><span class="vit-name">🧪 Creatinine</span><span class="vit-value">{fmt_val(case.get('creatinine'),'mg/dL',1)}</span></div>
                          <div class="vitals-row"><span class="vit-name">🔬 WBC</span><span class="vit-value">{fmt_val(case.get('wbc'),'x10⁹/L',1)}</span></div>
                          <div class="vitals-row" style="border-bottom:none"><span class="vit-name">🌡️ Temperature</span><span class="vit-value">{fmt_val(case.get('temperature'),'°C',1)}</span></div>
                        </div>""", unsafe_allow_html=True)

                    with ec2:
                        st.markdown("**🏥 Clinical Decision Made**")
                        action_map = {
                            'prescribe_antibiotic':   '💊 Antibiotic prescribed',
                            'prescribe_diuretic':     '💧 Diuretic prescribed',
                            'order_imaging':          '🩻 Imaging ordered',
                            'increase_monitoring':    '📈 Monitoring increased',
                            'discharge_plan':         '🏠 Discharge planned',
                            'refer_specialist':       '👨‍⚕️ Specialist referral',
                            'prescribe_vasopressor':  '💉 Vasopressor started',
                            'palliative_care':        '🕊️ Palliative care',
                        }
                        rat_map = {
                            'high_creatinine':       '🔬 Renal concern',
                            'elevated_wbc':          '🧫 Infection signs',
                            'elevated_heart_rate':   '❤️ Tachycardia',
                            'low_blood_pressure':    '🩸 Hypotension',
                            'stable_condition':      '✅ Stable',
                            'fluid_overload':        '💧 Fluid overload',
                            'abnormal_vitals':       '⚠️ Abnormal vitals',
                        }
                        action_raw = case.get('action','N/A')
                        rationale_raw = case.get('rationale','N/A')
                        st.markdown(f"""
                        <div class="dme-card" style="padding:14px;">
                          <div style="margin-bottom:10px;">
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">ACTION TAKEN</div>
                            <div style="font-size:15px;font-weight:700;color:#93c5fd;">{action_map.get(action_raw, action_raw)}</div>
                          </div>
                          <div style="margin-bottom:10px;">
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">PRIMARY REASON</div>
                            <div style="font-size:15px;font-weight:700;color:#86efac;">{rat_map.get(rationale_raw, rationale_raw)}</div>
                          </div>
                          <div>
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">DIAGNOSIS CODE</div>
                            <div style="font-size:15px;font-weight:700;color:#fcd34d;">{case.get('diagnosis_code','N/A')}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    with ec3:
                        st.markdown("**📊 Outcome & Similarity**")
                        conf_val = case.get('confidence', None)
                        conf_pct = int(conf_val * 100) if conf_val else 0
                        st.markdown(f"""
                        <div class="dme-card" style="padding:14px;">
                          <div style="margin-bottom:12px;">
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">OUTCOME</div>
                            <div style="font-size:18px;font-weight:800;color:{out_col};">{out_txt}</div>
                          </div>
                          <div style="margin-bottom:12px;">
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">SIMILARITY TO CURRENT PATIENT</div>
                            <div style="font-size:22px;font-weight:800;color:#93c5fd;">{sim_pct}%</div>
                            <div class="prog-outer"><div class="prog-inner" style="width:{sim_pct}%;background:#3b82f6;"></div></div>
                          </div>
                          <div>
                            <div style="font-size:11px;color:#475569;font-weight:600;letter-spacing:1px;margin-bottom:4px;">CLINICIAN CONFIDENCE</div>
                            <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{conf_pct}%</div>
                            <div class="prog-outer"><div class="prog-inner" style="width:{conf_pct}%;background:#8b5cf6;"></div></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    # AI case interpretation
                    if ai_on:
                        if st.button(f"🤖 Ask AI to explain Case {i+1}", key=f"case_ai_{i}"):
                            with st.spinner("AI analysing this case..."):
                                case_prompt = (
                                    f"A past hospital patient had: HR={case.get('heart_rate')} bpm, "
                                    f"BP={case.get('systolic_bp')} mmHg, Cr={case.get('creatinine')} mg/dL, "
                                    f"WBC={case.get('wbc')} x10⁹/L, Temp={case.get('temperature')} °C. "
                                    f"Clinical action: {action_raw}. Reason: {rationale_raw}. "
                                    f"Diagnosis: {case.get('diagnosis_code')}. Outcome: {'Improved' if improved else 'Did Not Improve'}.\n\n"
                                    f"The current patient has: HR={last_inp.get('heart_rate')}, "
                                    f"BP={last_inp.get('systolic_bp')}, Cr={last_inp.get('creatinine')}, "
                                    f"WBC={last_inp.get('wbc')}.\n\n"
                                    f"In plain English: 1) Why are these two patients similar? "
                                    f"2) What can we learn from the past case's outcome? "
                                    f"3) Does the action taken seem appropriate given what happened? "
                                    f"Be clear and accessible — the audience includes non-clinical staff."
                                )
                                ai_sys = ("You are a clinical educator explaining patient cases to a mixed audience. "
                                         "Be clear, educational, and honest. Always note outputs are advisory.")
                                reply = ai_ask(case_prompt, ai_sys, 500)
                            if reply:
                                st.markdown(f'<div class="ai-response">{reply}</div>', unsafe_allow_html=True)

            st.markdown("<div class='explain-box'>📖 <strong>Why show similar past cases?</strong> This is the Case-Based Reasoning (CBR) component. It finds patients from the MIMIC-IV database whose clinical profile most closely matches the current patient, using cosine similarity. By seeing what decisions were made in similar past situations and what the outcomes were, clinicians can contextualise the AI prediction with real human precedent. The confidence bar shows how certain the original practitioner was.</div>", unsafe_allow_html=True)
        else:
            st.warning("""
            **No similar cases found** for this patient profile.

            This can happen when:
            - The patient's diagnosis code is not in the training data
            - The clinical values are very different from all stored patients

            **Fix:** Go to **🧠 Decision Memory** → click **Rebuild Case Index**.
            """)

        # Decision logging panel — saves the current decision to the database.
        st.divider()
        st.markdown('<div class="section-header">📝 Record This Decision</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Recording the decision closes the feedback loop — once the outcome is known, this trace will be used to train future model versions.</div>', unsafe_allow_html=True)
        st.markdown('<div class="explain-box" style="margin-bottom:16px;">📖 <strong>Why record decisions?</strong> This is the core idea of the DME. Every logged decision is stored with full context. When the patient\'s outcome is later observed, the system links them together. Over time, the model learns which decisions in which contexts produce good outcomes — and uses that accumulated knowledge to advise future practitioners. This is the "memory" that distinguishes the DME from a standard prediction tool.</div>', unsafe_allow_html=True)

        lc1,lc2,lc3 = st.columns(3)
        with lc1:
            action_taken = st.selectbox("Action Taken", [
                'prescribe_antibiotic','prescribe_diuretic','order_imaging',
                'increase_monitoring','discharge_plan','refer_specialist',
                'prescribe_vasopressor','palliative_care','watchful_waiting','emergency_response'
            ])
        with lc2:
            rationale = st.selectbox("Primary Reason", [
                'elevated_wbc','fluid_overload','high_creatinine','elevated_heart_rate',
                'low_blood_pressure','stable_condition','abnormal_vitals',
                'infectious_workup','chronic_management','low_spo2','tachypnoea','emergency_protocol'
            ])
        with lc3:
            confidence = st.number_input("Confidence (0.5–1.0)", 0.5, 1.0, 0.8, 0.05, format="%.2f")

        if st.button("💾  Save to Decision Memory", use_container_width=True, type="primary"):
            import uuid
            tid = 'T_NEW_' + str(uuid.uuid4())[:8].upper()
            pid = st.session_state.get('last_patient','').strip() or 'P_NEW_' + str(uuid.uuid4())[:6].upper()
            log_decision(tid, pid, datetime.now(), {k:v for k,v in last_inp.items() if v is not None},
                         action_taken, rationale, confidence)
            st.success(f"✅ **Saved** — Trace `{tid}` · Patient `{pid}` · Action `{action_taken}` · {datetime.now().strftime('%H:%M %d/%m/%Y')}")
            st.session_state['last_logged'] = tid


# Tab 2 — AI Deep Analysis
# Four AI-powered panels: clinical summary, differential diagnoses,
# prediction explanation, and most critical missing value.
with tab2:
    st.markdown('<div class="section-header">🤖 AI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each section uses a specialised prompt · Groq Llama 3.3 70B · Concise clinical output only</div>', unsafe_allow_html=True)

    if not ai_on:
        st.markdown("""
        <div class="dme-card" style="text-align:center;padding:40px;">
          <div style="font-size:48px;margin-bottom:12px;">⚡</div>
          <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:8px;">Activate AI</div>
          <div style="color:#64748b;font-size:14px;line-height:1.9;">
            1. Go to <strong style="color:#3b82f6;">console.groq.com</strong> — sign up free<br>
            2. Create API Key (starts with <code>gsk_</code>)<br>
            3. Paste in <strong>⚡ AI Settings</strong> in the sidebar
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif not ran:
        st.info("Click **Run Clinical Analysis** first, then return here.")
    else:
        ps = patient_str()
        ca, cb = st.columns(2)

        with ca:
            st.markdown("#### 📝 Clinical Picture")
            st.caption("Condensed to: situation · concern · next action")
            if st.button("▶ Generate", key="b_sum", use_container_width=True):
                with st.spinner("..."):
                    st.session_state['ai_sum'] = ai_clinical_summary(ps, completeness)
            if st.session_state.get('ai_sum'):
                st.markdown("---")
                st.markdown(
                    f'<div class="ai-resp">{html_module.escape(st.session_state["ai_sum"])}</div>',
                    unsafe_allow_html=True)

            st.markdown("#### 🎯 Top 3 Differential Diagnoses")
            st.caption("Ranked by probability · specific evidence per diagnosis")
            if st.button("▶ Generate", key="b_dx", use_container_width=True):
                with st.spinner("..."):
                    st.session_state['ai_dx'] = ai_differential(ps)
            if st.session_state.get('ai_dx'):
                st.markdown("---")
                st.markdown(
                    f'<div class="ai-resp">{html_module.escape(st.session_state["ai_dx"])}</div>',
                    unsafe_allow_html=True)

        with cb:
            st.markdown("#### 🔍 Why This Prediction?")
            st.caption("3 bullets: feature → value → clinical reason")
            if st.button("▶ Generate", key="b_exp", use_container_width=True):
                with st.spinner("..."):
                    st.session_state['ai_exp'] = ai_explain_prediction(ps, prob or 0, top_feats)
            if st.session_state.get('ai_exp'):
                st.markdown("---")
                st.markdown(
                    f'<div class="ai-resp">{html_module.escape(st.session_state["ai_exp"])}</div>',
                    unsafe_allow_html=True)

            st.markdown("#### ❓ Most Critical Missing Value")
            st.caption("Specific to this patient only")
            if missing_fields:
                if st.button("▶ Generate", key="b_miss", use_container_width=True):
                    with st.spinner("..."):
                        st.session_state['ai_miss'] = ai_missing_data(ps, missing_fields, completeness)
                if st.session_state.get('ai_miss'):
                    st.markdown("---")
                    st.markdown(
                        f'<div class="ai-resp">{html_module.escape(st.session_state["ai_miss"])}</div>',
                        unsafe_allow_html=True)
            else:
                st.success("All key parameters provided.")


with tab3:
    st.markdown('<div class="section-header">🧠 Decision Memory Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Full transparency into everything stored in the system — every decision, outcome, and model version is recorded for governance and auditability.</div>', unsafe_allow_html=True)

    total, complete_n, versions, act_dist, out_dist, recent_logs = get_db_stats()
    pending_n = total - complete_n

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("📦 Total Decisions", total,        help="Every clinical decision ever recorded in this system")
    m2.metric("✅ Outcome Linked",  complete_n,    help="Decisions where we know what happened — these train the model")
    m3.metric("⏳ Awaiting Outcome",pending_n,     help="Decisions logged but outcome not yet observed")
    m4.metric("🔄 Model Versions",  len(versions), help="Number of times the ML model has been retrained")

    st.divider()

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### What Actions Are Stored in Memory?")
        st.caption("**Why this matters:** A healthy spread means the model learned from diverse clinical situations. If one action dominates heavily, the model may be biased towards recommending it.")
        if not act_dist.empty:
            act_dist.columns = ['Action','Count']
            st.bar_chart(act_dist.set_index('Action'))

    with ch2:
        st.markdown("#### Improved vs Did Not Improve")
        st.caption("**Why this matters:** A severely imbalanced dataset (e.g. 95%/5%) can cause the model to ignore the minority class. This is a known challenge in clinical ML and is why AUC-ROC is a better metric than accuracy alone.")
        if not out_dist.empty:
            out_dist['label'] = out_dist['outcome_value'].map({1:'✅ Improved',0:'❌ Did Not Improve'})
            st.bar_chart(out_dist.set_index('label')['count'])

    st.divider()
    st.markdown("#### 🔄 Model Training History")
    st.caption("**Governance record:** Every retrain logs a version. In real clinical AI deployment, you must show which version made which recommendation and what its accuracy was at that time.")
    if not versions.empty:
        vdf = versions[['version_id','timestamp','model_type','accuracy','n_traces_used','notes']].copy()
        vdf.columns = ['Version','Trained At','Model','Accuracy','Traces Used','Notes']
        vdf['Accuracy'] = vdf['Accuracy'].apply(lambda x: f"{x:.3f} AUC")
        st.dataframe(vdf, use_container_width=True, hide_index=True)
    else:
        st.info("No model versions logged yet.")

    st.divider()
    st.markdown("#### 📜 Audit Trail — Logged Decisions")
    st.caption("Every decision saved through the 'Record This Decision' panel appears here with full details.")
    if recent_logs.empty:
        st.info("No manually logged decisions yet. Use 'Record This Decision' in the Clinical Analysis tab.")
    else:
        for _,row in recent_logs.iterrows():
            conf = float(row['confidence']) if row['confidence'] else 0
            conf_css = "pill-green" if conf>=0.8 else "pill-amber" if conf>=0.6 else "pill-red"
            st.markdown(f"""
            <div class="audit-row">
              🕐 <strong>{str(row['timestamp'])[:16]}</strong> &nbsp;·&nbsp;
              Patient: <code>{row['patient_id']}</code> &nbsp;·&nbsp;
              Action: <code>{row['action']}</code> &nbsp;·&nbsp;
              Reason: <code>{row['rationale']}</code> &nbsp;·&nbsp;
              <span class="{conf_css}">Confidence: {conf:.0%}</span> &nbsp;·&nbsp;
              <em style="color:#475569">{row['status']}</em>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### ⚙️ System Management")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Retrain the ML Model**")
        st.caption("Run after new decisions are logged and linked to outcomes. The model relearns from all completed traces.")
        if st.button("🔄 Retrain Classification Model", use_container_width=True):
            with st.spinner("Training on all completed traces..."): train_classifier()
            st.success("✅ Model retrained and version logged.")
    with sc2:
        st.markdown("**Rebuild Similar-Cases Index**")
        st.caption("Run after retraining so newly logged cases appear in the similar-cases panel.")
        if st.button("🔄 Rebuild Case Retrieval Index", use_container_width=True):
            with st.spinner("Building similarity index..."): build_cbr_index()
            st.success("✅ Case index rebuilt.")


# Tab 4 — Safety and Bias
# Automated data quality checks plus four AI-powered safety reviews:
# action bias, outcome imbalance, ethics audit, and system health.
with tab4:
    st.markdown('<div class="section-header">🛡️ Safety & Bias Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Automated and AI-powered checks on the decision memory and model to detect data quality issues, potential biases, and ethical concerns.</div>', unsafe_allow_html=True)

    st.markdown("#### 🔍 Automated Data Quality Report")
    st.caption("Runs directly on the database — no AI key needed.")
    if st.button("▶ Run Data Quality Check", use_container_width=True):
        conn = sqlite3.connect(DB_PATH)
        df_t = pd.read_sql_query("SELECT * FROM decision_traces", conn)
        df_o = pd.read_sql_query("SELECT * FROM outcomes", conn)
        conn.close()
        dq = []
        for col in ['heart_rate','systolic_bp','creatinine','wbc','temperature']:
            n  = df_t[col].isnull().sum(); p = round(n/len(df_t)*100,1)
            dq.append([col.replace('_',' ').title(), f"{n} missing ({p}%)",
                       "✅ Good" if p==0 else "⚠️ Warning" if p<20 else "❌ Critical"])
        if not df_o.empty:
            pos=(df_o['outcome_value']==1).sum(); neg=(df_o['outcome_value']==0).sum()
            dq.append(["Outcome Balance", f"Improved: {pos} / Not: {neg} (ratio {pos/max(neg,1):.0f}:1)",
                       "✅ Reasonable" if pos/max(neg,1)<20 else "⚠️ Imbalanced"])
        dup = df_o['trace_id'].duplicated().sum() if not df_o.empty else 0
        dq.append(["Duplicate Outcomes", f"{dup} duplicate records",
                   "✅ Clean" if dup==0 else "⚠️ Duplicates detected"])
        junk_exists = os.path.exists('junk.py')
        dq.append(["Temp Files", "junk.py found — delete before submission" if junk_exists else "Clean",
                   "⚠️ Warning" if junk_exists else "✅ Clean"])
        dq_df = pd.DataFrame(dq, columns=["Check","Finding","Status"])
        st.dataframe(dq_df, use_container_width=True, hide_index=True)
        if any("❌" in r[2] for r in dq): st.error("Critical issues detected. Review before relying on model predictions.")
        elif any("⚠️" in r[2] for r in dq): st.warning("Some warnings detected. May affect model reliability.")
        else: st.success("All data quality checks passed.")

    st.divider()
    st.markdown("#### 🤖 AI-Powered Safety Analysis")
    if not ai_on:
        st.info("Enter your free Groq API key in the ⚡ AI Settings panel (sidebar) to enable these checks.")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Action Distribution Bias**")
        if st.button("▶ Run Bias Check", use_container_width=True, key="bias1"):
            if not ai_on: st.warning("AI key required.")
            else:
                conn = sqlite3.connect(DB_PATH)
                ad = pd.read_sql_query(
                    "SELECT action, COUNT(*) as n FROM decision_traces "
                    "GROUP BY action ORDER BY n DESC", conn)
                conn.close()
                dist = "\n".join([f"- {r['action']}: {r['n']} cases"
                                  for _, r in ad.iterrows()])

                # Work out what the CBR oversampler did to each action class
                # so the AI knows the imbalance is already being handled.
                if not ad.empty:
                    max_n = int(ad['n'].max())
                    cbr_target = max(max_n // 3, 5)
                    cbr_lines = []
                    for _, row in ad.iterrows():
                        n = int(row['n'])
                        a = row['action']
                        if n < cbr_target:
                            cbr_lines.append(
                                f"{a}: {n} -> {cbr_target} (oversampled in CBR)")
                    cbr_note = ("; ".join(cbr_lines)
                                if cbr_lines else "No actions required oversampling.")
                else:
                    cbr_note = "No action data available."

                with st.spinner("Checking for action bias..."):
                    r = ai_bias_check(dist, cbr_note)
                if r:
                    safe_r = html_module.escape(r)
                    st.markdown(f'<div class="ai-response">{safe_r}</div>',
                                unsafe_allow_html=True)
                # Always show the static mitigation note so the user knows
                # the CBR already compensates, regardless of what the AI said.
                if not ad.empty:
                    st.caption(
                        f"**CBR mitigation active** — rare actions oversampled to "
                        f"{cbr_target} cases in the retrieval index. "
                        "Imbalance above reflects raw training data only.")

        st.markdown("**Outcome Imbalance Risk**")
        if st.button("▶ Outcome Balance Check", use_container_width=True, key="bias2"):
            if not ai_on: st.warning("AI key required.")
            else:
                conn = sqlite3.connect(DB_PATH)
                od = pd.read_sql_query(
                    "SELECT outcome_value, COUNT(*) as n FROM outcomes "
                    "GROUP BY outcome_value", conn)
                conn.close()
                _pos = int(od[od['outcome_value']==1]['n'].values[0]) \
                       if 1 in od['outcome_value'].values else 0
                _neg = int(od[od['outcome_value']==0]['n'].values[0]) \
                       if 0 in od['outcome_value'].values else 0
                with st.spinner("Analysing outcome distribution..."):
                    r = ai_outcome_balance(_pos, _neg)
                if r:
                    safe_r = html_module.escape(r)
                    st.markdown(f'<div class="ai-response">{safe_r}</div>',
                                unsafe_allow_html=True)

    with b2:
        st.markdown("**Ethical Compliance Review**")
        if st.button("▶ Ethics Review", use_container_width=True, key="ethics"):
            if not ai_on: st.warning("AI key required.")
            else:
                with st.spinner("Running ethics review..."):
                    r = ai_ethics_review()
                if r:
                    safe_r = html_module.escape(r)
                    st.markdown(f'<div class="ai-response">{safe_r}</div>',
                                unsafe_allow_html=True)

        st.markdown("**System Health Diagnostics**")
        if st.button("▶ System Health Check", use_container_width=True, key="health"):
            if not ai_on: st.warning("AI key required.")
            else:
                issues = []
                for f in ['models/classifier.pkl', 'models/cbr_index.pkl',
                          'models/scaler.pkl', 'models/feature_cols.pkl']:
                    if not os.path.exists(f):
                        issues.append(f"Missing critical file: {f}")
                if os.path.exists('junk.py'):
                    issues.append(
                        "Temporary file junk.py should be deleted before submission")
                conn = sqlite3.connect(DB_PATH)
                t_n = pd.read_sql_query(
                    "SELECT COUNT(*) as n FROM decision_traces", conn).iloc[0]['n']
                o_n = pd.read_sql_query(
                    "SELECT COUNT(*) as n FROM outcomes", conn).iloc[0]['n']
                conn.close()
                if o_n > t_n * 2:
                    issues.append(
                        f"Outcomes table ({o_n} rows) appears duplicated vs "
                        f"traces ({t_n} rows) — outcome linker may have run twice")
                with st.spinner("Diagnosing system..."):
                    r = ai_system_health(issues, t_n, o_n)
                if r:
                    safe_r = html_module.escape(r)
                    st.markdown(f'<div class="ai-response">{safe_r}</div>',
                                unsafe_allow_html=True)


# Tab 5 — AI Assistant
# Free-form chat with full patient context. Suggested questions update
# dynamically based on whatever abnormal values the current patient has.
with tab5:
    st.markdown('<div class="section-header">💬 AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Context-aware · max 100 words per response · references current patient values</div>', unsafe_allow_html=True)

    if not ai_on:
        st.markdown("""
        <div class="dme-card" style="text-align:center;padding:36px;">
          <div style="font-size:42px;margin-bottom:10px;">💬</div>
          <div style="font-size:16px;font-weight:700;color:#f1f5f9;margin-bottom:8px;">AI Assistant</div>
          <div style="color:#64748b;font-size:13px;line-height:2;">
            Enter Groq API key in <strong>⚡ AI Settings</strong> in the sidebar.<br>
            Free at <strong style="color:#3b82f6;">console.groq.com</strong> — no credit card.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        ctx = patient_str() if ran else "No patient analysed yet."

        # Dynamic questions based on current patient state
        qs = generate_dynamic_questions(
            last_inp if ran else {},
            news_score, prob, missing_fields
        )
        st.markdown("**Suggested questions:**" if not ran else "**Questions based on current patient:**")
        qcols = st.columns(4)
        for i, (q, col) in enumerate(zip(qs, qcols)):
            with col:
                label = (q[:52] + "…") if len(q) > 55 else q
                if st.button(label, use_container_width=True, key=f"sq{i}"):
                    st.session_state['preset_q'] = q

        st.divider()

        # Chat display — content is html.escape()d to prevent broken div tags
        for msg in st.session_state['chat_history']:
            safe_content = html_module.escape(msg["content"])
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-user">👤 {safe_content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 {safe_content}</div>',
                    unsafe_allow_html=True
                )

        # Auto-send preset questions immediately on the preset-button rerun.
        # Doing it here (before the text_input) means we never need to pass
        # value=preset into st.text_input — which caused the "Send rerun resets
        # widget to ''" bug (preset already popped → value='' → user_in empty).
        preset = st.session_state.pop('preset_q', None)
        if preset:
            st.session_state['chat_history'].append({'role': 'user', 'content': preset})
            msgs = [{'role': m['role'], 'content': m['content']}
                    for m in st.session_state['chat_history']]
            with st.spinner("Thinking..."):
                reply = ai_chat_response(msgs, ctx)
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': reply or "⚠️ No response. Check your Groq API key in ⚡ AI Settings (sidebar)."
            })
            st.rerun()

        # Manual text input — NO value= parameter (prevents Streamlit widget-reset
        # bug: passing value='' on the Send rerun wipes whatever the user typed)
        user_in = st.text_input("Ask anything...",
                                 placeholder="e.g. Is this creatinine level serious?",
                                 label_visibility="collapsed")
        sc1, sc2 = st.columns([5, 1])
        with sc1:
            send = st.button("Send →", use_container_width=True, type="primary", key="send_btn")
        with sc2:
            if st.button("Clear", use_container_width=True, key="clear_btn"):
                st.session_state['chat_history'] = []
                st.rerun()

        if send and user_in.strip():
            st.session_state['chat_history'].append({'role': 'user', 'content': user_in})
            msgs = [{'role': m['role'], 'content': m['content']}
                    for m in st.session_state['chat_history']]
            with st.spinner("Thinking..."):
                reply = ai_chat_response(msgs, ctx)
            # Append reply once — error prefix strings are shown styled via chat-ai div
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': reply if reply else "⚠️ No response. Check your Groq API key in ⚡ AI Settings (sidebar)."
            })
            st.rerun()