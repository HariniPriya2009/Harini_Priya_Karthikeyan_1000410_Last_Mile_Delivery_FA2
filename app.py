"""
============================================================================
 LOGISIGHT ANALYTICS
 Last Mile Delivery Intelligence Dashboard  —  Premium SaaS Redesign (v2)
 ---------------------------------------------------------------------------
 A production-ready Streamlit analytics dashboard for logistics managers to
 analyse last-mile delivery performance and surface operational bottlenecks.

 Redesign highlights
   • Premium gradient KPI metric cards (exact brand gradients)
   • Every chart wrapped in a white "premium card" container
   • Row-based dashboard layout (KPI row -> 5 chart rows -> AI panel)
   • Glassmorphism sidebar with rounded filter containers
   • AI Insights Panel with colourful BI cards
   • Full-dataset processing (no sampling)

 Author : LogiSight Analytics
 Stack  : Streamlit + Pandas + Plotly + Scikit-learn + Statsmodels
============================================================================
"""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
#  PAGE CONFIGURATION  (must be the first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="LogiSight Analytics | Last Mile Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
#  DESIGN TOKENS
# ============================================================================
PRIMARY_1 = "#0066FF"
PRIMARY_2 = "#3B82F6"
PRIMARY_3 = "#38BDF8"
ACCENT_1 = "#FF55C5"
ACCENT_2 = "#55E3FF"
ACCENT_3 = "#8271B7"
BG = "#F5F8FC"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#64748B"

PRIMARY_SEQ = [PRIMARY_1, PRIMARY_2, PRIMARY_3]
ACCENT_SEQ = [ACCENT_1, ACCENT_2, ACCENT_3]
GRADIENT_SEQ = ["#0066FF", "#3B82F6", "#38BDF8", "#55E3FF", "#8271B7", "#FF55C5"]

# ---- Exact KPI gradients (per redesign specification) ----------------------
KPI_GRADIENTS = {
    "deliveries": "linear-gradient(135deg,#2563EB,#3B82F6)",   # Card 1 · blue
    "time":       "linear-gradient(135deg,#7C3AED,#A855F7)",   # Card 2 · purple
    "late":       "linear-gradient(135deg,#F97316,#FB923C)",   # Card 3 · orange
    "rating":     "linear-gradient(135deg,#10B981,#34D399)",   # Card 4 · green
    "vehicle":    "linear-gradient(135deg,#EC4899,#F472B6)",   # Card 5 · pink
    "area":       "linear-gradient(135deg,#F59E0B,#FBBF24)",   # Card 6 · amber
}

# ---- AI Insights Panel gradients -------------------------------------------
AI_GRADIENTS = {
    "vehicle": "linear-gradient(135deg,#2563EB,#3B82F6)",
    "traffic": "linear-gradient(135deg,#F97316,#FB923C)",
    "area":    "linear-gradient(135deg,#10B981,#34D399)",
    "risk":    "linear-gradient(135deg,#EC4899,#F472B6)",
    "summary": "linear-gradient(135deg,#7C3AED,#A855F7)",
}

TRAFFIC_COLORS = {
    "Low": "#38BDF8",
    "Medium": "#3B82F6",
    "High": "#FF55C5",
    "Jam": "#8271B7",
}
AGE_GROUP_COLORS = {
    "Under 25": "#38BDF8",
    "25-40": "#0066FF",
    "40+": "#FF55C5",
}

DATA_CANDIDATES = [
    "Last_mile_Delivery_Data.csv",
    "data/Last_mile_Delivery_Data.csv",
    "sample_data.csv",
    "data/sample_data.csv",
]


# ============================================================================
#  CUSTOM CSS LOADER
# ============================================================================
# ---------------------------------------------------------------------------
#  EMBEDDED STYLESHEET  (self-contained: works even without assets/style.css)
# ---------------------------------------------------------------------------
_EMBEDDED_CSS = r"""/* ============================================================================
   LOGISIGHT ANALYTICS — Premium SaaS Analytics Theme  (v2 Redesign)
   Last Mile Delivery Intelligence Dashboard
   ============================================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ---------------------------------------------------------------------------
   ROOT DESIGN TOKENS
   --------------------------------------------------------------------------- */
:root {
    --primary-1: #0066FF;
    --primary-2: #3B82F6;
    --primary-3: #38BDF8;
    --accent-1: #FF55C5;
    --accent-2: #55E3FF;
    --accent-3: #8271B7;
    --bg: #F5F8FC;
    --card-bg: #FFFFFF;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --text-soft: #94A3B8;
    --radius: 25px;
    --radius-sm: 16px;
    --shadow: 0px 10px 30px rgba(0, 0, 0, 0.15);
    --shadow-hover: 0px 18px 45px rgba(0, 0, 0, 0.20);
    --shadow-soft: 0px 6px 20px rgba(15, 23, 42, 0.08);
    --grad-primary: linear-gradient(135deg, #0066FF 0%, #3B82F6 55%, #38BDF8 100%);
    --grad-nav: linear-gradient(120deg, #0052CC 0%, #0066FF 40%, #3B82F6 75%, #38BDF8 100%);
}

/* ---------------------------------------------------------------------------
   GLOBAL / BASE
   --------------------------------------------------------------------------- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.10) 0%, transparent 40%),
        radial-gradient(circle at 100% 0%, rgba(255, 85, 197, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(130, 113, 183, 0.07) 0%, transparent 45%);
    background-attachment: fixed;
}

#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 1650px;
}

/* ---------------------------------------------------------------------------
   TOP NAVIGATION BAR
   --------------------------------------------------------------------------- */
.ls-navbar {
    background: var(--grad-nav);
    border-radius: var(--radius);
    padding: 18px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.15);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    animation: slideDown 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.ls-navbar::after {
    content: "";
    position: absolute;
    top: -60%;
    right: -5%;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.ls-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.4px;
    z-index: 2;
}
.ls-brand .ls-logo-badge {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.20);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: inset 0 1px 4px rgba(255,255,255,0.4);
}
.ls-brand .ls-sub {
    display: block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    opacity: 0.82;
    margin-top: 2px;
}
.ls-nav-tabs {
    display: flex;
    gap: 6px;
    background: rgba(255, 255, 255, 0.14);
    padding: 6px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    z-index: 2;
}
.ls-nav-tab {
    color: rgba(255, 255, 255, 0.85);
    padding: 9px 20px;
    border-radius: 11px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
}
.ls-nav-tab:hover { background: rgba(255, 255, 255, 0.18); color: #ffffff; }
.ls-nav-tab.active {
    background: #ffffff;
    color: #0066FF;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
}
.ls-nav-right { display: flex; align-items: center; gap: 12px; color: #ffffff; z-index: 2; }
.ls-date-pill {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.30);
    border-radius: 13px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    backdrop-filter: blur(8px);
}
.ls-live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #55E3FF;
    box-shadow: 0 0 0 0 rgba(85, 227, 255, 0.7);
    animation: pulse 2s infinite;
}

/* ---------------------------------------------------------------------------
   PAGE HEADER
   --------------------------------------------------------------------------- */
.ls-page-title {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.6px;
    margin: 6px 0 2px 0;
}
.ls-page-sub {
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 20px;
}

/* ---------------------------------------------------------------------------
   KPI CARDS  (premium metric cards)
   --------------------------------------------------------------------------- */
.kpi-card {
    height: 140px;
    border-radius: var(--radius);
    padding: 22px 24px;
    color: #ffffff;
    position: relative;
    overflow: hidden;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.35s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
    animation: fadeUp 0.6s cubic-bezier(0.22, 1, 0.36, 1) backwards;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 18px 45px rgba(0, 0, 0, 0.22);
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: -45%; right: -25%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(255,255,255,0.30) 0%, transparent 70%);
    border-radius: 50%;
}
.kpi-card::after {
    content: "";
    position: absolute;
    bottom: -35px; left: -25px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, transparent 70%);
    border-radius: 50%;
}
.kpi-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    position: relative;
    z-index: 2;
}
.kpi-icon {
    width: 34px; height: 34px;
    border-radius: 11px;
    background: rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.30);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.kpi-title {
    font-size: 13px;
    font-weight: 600;
    opacity: 0.96;
    letter-spacing: 0.2px;
    line-height: 1.15;
}
.kpi-value {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1.4px;
    line-height: 1;
    position: relative;
    z-index: 2;
    text-shadow: 0 2px 10px rgba(0,0,0,0.12);
}
.kpi-sub {
    font-size: 11px;
    font-weight: 500;
    opacity: 0.85;
    margin-top: 5px;
    position: relative;
    z-index: 2;
    letter-spacing: 0.1px;
}

/* ---------------------------------------------------------------------------
   PREMIUM CHART CARD CONTAINERS  (st.container(key="chart_*"))
   --------------------------------------------------------------------------- */
[class*="st-key-chart_"] {
    background: #FFFFFF !important;
    border-radius: var(--radius) !important;
    padding: 20px 22px 14px 22px !important;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}
[class*="st-key-chart_"]:hover {
    box-shadow: 0px 18px 45px rgba(0, 0, 0, 0.18) !important;
}

/* ---------------------------------------------------------------------------
   GLASS CARDS / SECTION CONTAINERS
   --------------------------------------------------------------------------- */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.85);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: var(--shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 18px;
}
.glass-card:hover { box-shadow: var(--shadow-hover); }

.section-title {
    font-size: 16.5px;
    font-weight: 800;
    color: var(--text-main);
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 3px;
}
.section-title .st-ico {
    width: 32px; height: 32px;
    border-radius: 10px;
    background: var(--grad-primary);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 6px 16px rgba(0, 102, 255, 0.30);
    flex-shrink: 0;
}
.section-desc {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 8px;
    padding-left: 42px;
}

/* ---------------------------------------------------------------------------
   INSIGHT CARDS (AI Insight Engine)
   --------------------------------------------------------------------------- */
.insight-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(245,248,252,0.97) 100%);
    border-radius: 20px;
    padding: 16px 18px;
    border-left: 5px solid var(--primary-1);
    box-shadow: var(--shadow-soft);
    margin-bottom: 12px;
    display: flex;
    gap: 13px;
    align-items: flex-start;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.5s ease backwards;
}
.insight-card:hover {
    transform: translateX(5px);
    box-shadow: 0 10px 28px rgba(0, 102, 255, 0.16);
}
.insight-card.ic-pink { border-left-color: var(--accent-1); }
.insight-card.ic-cyan { border-left-color: var(--accent-2); }
.insight-card.ic-purple { border-left-color: var(--accent-3); }
.insight-card.ic-blue { border-left-color: var(--primary-2); }
.insight-card.ic-green { border-left-color: #10B981; }
.insight-card.ic-orange { border-left-color: #F97316; }
.insight-ico { font-size: 22px; line-height: 1; flex-shrink: 0; margin-top: 2px; }
.insight-text { font-size: 13.5px; color: var(--text-main); font-weight: 500; line-height: 1.55; }
.insight-text b { color: var(--primary-1); font-weight: 800; }
.insight-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--text-soft);
    margin-top: 4px;
}

/* Insight box under charts */
.insight-box {
    background: linear-gradient(120deg, rgba(0,102,255,0.06) 0%, rgba(56,189,248,0.06) 100%);
    border: 1px solid rgba(0, 102, 255, 0.14);
    border-radius: 18px;
    padding: 14px 17px;
    margin-top: 10px;
}
.insight-box .ib-title {
    font-size: 11.5px;
    font-weight: 800;
    color: var(--primary-1);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 7px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.insight-box ul { margin: 0; padding-left: 17px; }
.insight-box li {
    font-size: 12.5px;
    color: var(--text-main);
    font-weight: 500;
    margin-bottom: 5px;
    line-height: 1.5;
}

/* ---------------------------------------------------------------------------
   AI INSIGHTS PANEL  (ROW 6 — colorful BI cards)
   --------------------------------------------------------------------------- */
.ai-panel-card {
    border-radius: 22px;
    padding: 20px;
    color: #ffffff;
    position: relative;
    overflow: hidden;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.15);
    height: 100%;
    min-height: 150px;
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.35s ease;
    animation: fadeUp 0.6s ease backwards;
}
.ai-panel-card:hover { transform: translateY(-5px); box-shadow: 0px 18px 45px rgba(0,0,0,0.22); }
.ai-panel-card::before {
    content: "";
    position: absolute;
    top: -40%; right: -20%;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(255,255,255,0.28) 0%, transparent 70%);
    border-radius: 50%;
}
.ai-ico {
    width: 42px; height: 42px;
    border-radius: 13px;
    background: rgba(255,255,255,0.22);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.30);
    display: flex; align-items: center; justify-content: center;
    font-size: 21px;
    margin-bottom: 12px;
    position: relative; z-index: 2;
}
.ai-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.85;
    position: relative; z-index: 2;
}
.ai-value {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.6px;
    margin: 3px 0 5px 0;
    position: relative; z-index: 2;
    line-height: 1.1;
}
.ai-desc {
    font-size: 12px;
    font-weight: 500;
    opacity: 0.92;
    line-height: 1.45;
    position: relative; z-index: 2;
}

/* ---------------------------------------------------------------------------
   SCORECARD TABLE
   --------------------------------------------------------------------------- */
.scorecard { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-size: 13px; }
.scorecard th {
    text-align: left; color: var(--text-muted); font-weight: 700;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px; padding: 4px 14px;
}
.scorecard td {
    background: #ffffff; padding: 13px 14px; color: var(--text-main);
    font-weight: 600; box-shadow: var(--shadow-soft);
}
.scorecard tr td:first-child { border-radius: 14px 0 0 14px; }
.scorecard tr td:last-child { border-radius: 0 14px 14px 0; }
.scorecard tr:hover td { background: #F5F8FC; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.badge-good { background: rgba(16,185,129,0.14); color: #059669; }
.badge-warn { background: rgba(245,158,11,0.16); color: #D97706; }
.badge-bad  { background: rgba(239,68,68,0.14); color: #DC2626; }

/* ---------------------------------------------------------------------------
   SIDEBAR  (glassmorphism + rounded filter containers)
   --------------------------------------------------------------------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F5F8FC 100%);
    border-right: 1px solid rgba(0, 102, 255, 0.08);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-main); font-weight: 800; }

.sidebar-brand {
    background: var(--grad-primary);
    border-radius: 20px;
    padding: 18px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0px 10px 30px rgba(0, 102, 255, 0.30);
}
.sidebar-brand .sb-title { font-size: 18px; font-weight: 800; }
.sidebar-brand .sb-sub { font-size: 10.5px; opacity: 0.88; letter-spacing: 1px; }

/* Rounded glass containers for filter groups */
[class*="st-key-filter_"] {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(12px);
    border-radius: 20px !important;
    padding: 14px 16px 8px 16px !important;
    box-shadow: 0px 6px 20px rgba(15, 23, 42, 0.08) !important;
    border: 1px solid rgba(0, 102, 255, 0.10) !important;
    margin-bottom: 14px;
}
.sidebar-section-label {
    font-size: 11px;
    font-weight: 800;
    color: var(--primary-1);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* Streamlit widget polish */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 13px !important;
    border-color: rgba(0, 102, 255, 0.18) !important;
    background: #ffffff !important;
}
[data-testid="stSidebar"] .stDateInput input { border-radius: 13px !important; }
[data-testid="stSidebar"] label { font-weight: 600 !important; color: #334155 !important; }

/* Multiselect chips — brand gradient */
[data-testid="stMultiSelectTagsContainer"] span span {
    background: linear-gradient(135deg, #0066FF 0%, #3B82F6 100%) !important;
    border-radius: 9px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stMultiSelectTagsContainer"] span span * { color: #ffffff !important; }
[data-testid="stMultiSelectTagsContainer"] span span svg { fill: #ffffff !important; }
[data-baseweb="tag"] {
    background: linear-gradient(135deg, #0066FF 0%, #3B82F6 100%) !important;
    border-radius: 9px !important; border: none !important;
}
[data-baseweb="tag"] span { color: #ffffff !important; font-weight: 600 !important; }
[data-baseweb="tag"] svg { fill: #ffffff !important; }

/* ---------------------------------------------------------------------------
   BUTTONS
   --------------------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 14px !important;
    border: none !important;
    background: var(--grad-primary) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    padding: 10px 18px !important;
    box-shadow: 0 6px 18px rgba(0, 102, 255, 0.26) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(0, 102, 255, 0.36) !important;
}

/* ---------------------------------------------------------------------------
   TABS
   --------------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.75);
    padding: 7px;
    border-radius: 18px;
    box-shadow: var(--shadow-soft);
    border: 1px solid rgba(255,255,255,0.9);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 9px 20px;
    font-weight: 700;
    font-size: 13.5px;
    color: var(--text-muted);
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: var(--grad-primary) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 16px rgba(0, 102, 255, 0.28);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------------------------------------------------------------------------
   METRICS / MISC
   --------------------------------------------------------------------------- */
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 18px; padding: 16px 18px;
    box-shadow: var(--shadow-soft); border: 1px solid rgba(255,255,255,0.9);
}
[data-testid="stMetricValue"] { color: var(--text-main); font-weight: 800; }
[data-testid="stMetricLabel"] { color: var(--text-muted); font-weight: 600; }
[data-testid="stDataFrame"] { border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-soft); }
.js-plotly-plot .plotly { border-radius: 16px; }

/* ---------------------------------------------------------------------------
   ANIMATIONS
   --------------------------------------------------------------------------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(85, 227, 255, 0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(85, 227, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(85, 227, 255, 0); }
}

/* Responsive */
@media (max-width: 1100px) {
    .ls-navbar { flex-direction: column; gap: 14px; align-items: flex-start; }
    .ls-nav-tabs { flex-wrap: wrap; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
"""


def load_css(path: str = "assets/style.css") -> None:
    """
    Inject the premium glassmorphism stylesheet into the app.

    Priority:
      1. external file (assets/style.css) if present  -> easy live editing
      2. embedded stylesheet (always available)       -> deployment-proof
    """
    css = _EMBEDDED_CSS
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                file_css = fh.read()
            if file_css.strip():
                css = file_css
    except Exception:
        css = _EMBEDDED_CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# ============================================================================
#  DATA LOADING  (cached)
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    return pd.read_csv(path)


def resolve_dataset() -> tuple[pd.DataFrame, str]:
    """
    Find and load the dataset.

    Priority:
      1. full CSV on disk (Last_mile_Delivery_Data.csv)      -> preferred
      2. embedded full dataset (embedded_data.py)            -> deployment-proof
      3. any smaller CSV on disk (sample_data.csv)           -> last resort
    """
    # 1) full dataset on disk
    for candidate in DATA_CANDIDATES:
        if os.path.exists(candidate):
            return load_data(candidate), candidate

    # 2) embedded full dataset (guarantees all 43,739 rows)
    try:
        from embedded_data import get_full_csv_bytes
        raw = pd.read_csv(io.BytesIO(get_full_csv_bytes()))
        return raw, "embedded full dataset (43,739 rows)"
    except Exception:
        pass

    st.error(
        "❌ No dataset found. Please place `Last_mile_Delivery_Data.csv` "
        "in the project root, or include `embedded_data.py`."
    )
    st.stop()


# ============================================================================
#  DATA CLEANING & FEATURE ENGINEERING  (cached)
# ============================================================================
@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      - strip whitespace from text columns
      - remove duplicate rows
      - handle missing values
      - fix out-of-range ratings
      - convert datatypes (dates / times)
      - create Age Group column
      - create Late Delivery flag (Delivery_Time > mean + 1 std)
    """
    df = raw.copy()

    # ---- 1. Strip whitespace & normalise null sentinels -------------------
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    null_tokens = {"nan", "none", "null", "nat", "<na>", "n/a", "na", ""}
    for col in obj_cols:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(null_tokens))

    # ---- 2. Remove duplicate rows ------------------------------------------
    df = df.drop_duplicates().reset_index(drop=True)

    # ---- 3. Handle missing values ------------------------------------------
    if "Weather" in df.columns:
        df["Weather"] = df["Weather"].fillna(df["Weather"].mode().iloc[0])
    if "Traffic" in df.columns:
        df["Traffic"] = df["Traffic"].fillna(df["Traffic"].mode().iloc[0])
    if "Agent_Rating" in df.columns:
        df["Agent_Rating"] = df["Agent_Rating"].fillna(df["Agent_Rating"].median())

    # ---- 4. Normalise known label typos ------------------------------------
    if "Area" in df.columns:
        df["Area"] = df["Area"].replace({"Metropolitian": "Metropolitan"})

    # ---- 5. Fix out-of-range ratings (data quality: 6.0 outliers) ----------
    if "Agent_Rating" in df.columns:
        df["Agent_Rating"] = pd.to_numeric(df["Agent_Rating"], errors="coerce")
        df["Agent_Rating"] = df["Agent_Rating"].clip(lower=1.0, upper=5.0)
        df["Agent_Rating"] = df["Agent_Rating"].fillna(df["Agent_Rating"].median())

    # ---- 6. Convert datatypes ----------------------------------------------
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df = df.dropna(subset=["Order_Date"])

    df["Agent_Age"] = pd.to_numeric(df["Agent_Age"], errors="coerce")
    df["Delivery_Time"] = pd.to_numeric(df["Delivery_Time"], errors="coerce")
    df = df.dropna(subset=["Agent_Age", "Delivery_Time"])
    df["Agent_Age"] = df["Agent_Age"].astype(int)
    df["Delivery_Time"] = df["Delivery_Time"].astype(float)

    for tcol in ["Order_Time", "Pickup_Time"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], format="%H:%M:%S", errors="coerce").dt.time

    # ---- 7. Age Group feature ----------------------------------------------
    def _age_group(age: int) -> str:
        if age < 25:
            return "Under 25"
        if age <= 40:
            return "25-40"
        return "40+"

    df["Age_Group"] = df["Agent_Age"].apply(_age_group)

    # ---- 8. Late Delivery flag (mean + 1 std) ------------------------------
    threshold = df["Delivery_Time"].mean() + df["Delivery_Time"].std()
    df["Late_Delivery"] = df["Delivery_Time"] > threshold
    df.attrs["late_threshold"] = float(threshold)

    # ---- 9. Derived time features ------------------------------------------
    df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b %Y")
    df["Day_Name"] = df["Order_Date"].dt.day_name()
    df["Weekday_Num"] = df["Order_Date"].dt.weekday
    df["Hour"] = pd.to_datetime(
        df["Order_Time"].astype(str), format="%H:%M:%S", errors="coerce"
    ).dt.hour

    # ---- 10. Synthetic Agent ID (dataset has no explicit agent id) ----------
    df["Agent_ID"] = (
        "AGT-"
        + df.groupby(["Agent_Age", "Agent_Rating", "Area"], observed=True)
        .ngroup()
        .add(1)
        .astype(str)
        .str.zfill(4)
    )

    return df.reset_index(drop=True)


# ============================================================================
#  AGGREGATION HELPERS
# ============================================================================
def late_threshold(df: pd.DataFrame) -> float:
    return float(df["Delivery_Time"].mean() + df["Delivery_Time"].std())


def agg_mean(df: pd.DataFrame, by: str | list[str], value: str = "Delivery_Time") -> pd.DataFrame:
    """Grouped mean + count, sorted ascending by mean."""
    out = (
        df.groupby(by, observed=True)[value]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg_Delivery_Time", "count": "Deliveries"})
        .sort_values("Avg_Delivery_Time")
    )
    return out


def agg_late(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Late delivery % by a grouping column."""
    out = (
        df.groupby(by, observed=True)["Late_Delivery"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Late_Pct", "count": "Deliveries"})
    )
    out["Late_Pct"] = (out["Late_Pct"] * 100).round(2)
    return out.sort_values("Late_Pct", ascending=False)


# ============================================================================
#  PLOTLY THEME
# ============================================================================
def style_fig(fig: go.Figure, height: int = 420, showlegend: bool = True) -> go.Figure:
    """Apply the LogiSight light theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, sans-serif", size=12.5, color=TEXT_MAIN),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=10, r=10, t=46, b=10),
        title=dict(font=dict(size=15.5, color=TEXT_MAIN, family="Inter, sans-serif"), x=0.01),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,102,255,0.10)",
            borderwidth=1,
        ),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=PRIMARY_2,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.25)",
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MUTED, size=12),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MUTED, size=12),
    )
    return fig


# ============================================================================
#  UI COMPONENT BUILDERS
# ============================================================================
def render_navbar(date_label: str) -> None:
    """Top gradient navigation bar with brand, tabs and date pill."""
    st.markdown(
        f"""
        <div class="ls-navbar">
            <div class="ls-brand">
                <div class="ls-logo-badge">🚚</div>
                <div>
                    LogiSight Analytics
                    <span class="ls-sub">Last Mile Delivery Intelligence</span>
                </div>
            </div>
            <div class="ls-nav-tabs">
                <div class="ls-nav-tab active">Dashboard</div>
                <div class="ls-nav-tab">Insights</div>
                <div class="ls-nav-tab">Performance</div>
                <div class="ls-nav-tab">Operations</div>
            </div>
            <div class="ls-nav-right">
                <div class="ls-date-pill">📅 {date_label}</div>
                <div class="ls-live-dot"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(icon: str, value: str, label: str, sub: str, gradient: str, delay: float = 0.0) -> str:
    """Return HTML for a single premium gradient KPI metric card."""
    return f"""
        <div class="kpi-card" style="background:{gradient};animation-delay:{delay}s;">
            <div class="kpi-top">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-title">{label}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    """


@contextmanager
def chart_card(key: str, icon: str, title: str, desc: str):
    """
    Context manager that renders a premium white chart container.
    The container receives the Streamlit class `st-key-chart_<key>` which is
    styled by the stylesheet (25px radius, soft shadow, padding).
    """
    with st.container(key=f"chart_{key}"):
        st.markdown(
            f"""
            <div class="section-title"><span class="st-ico">{icon}</span>{title}</div>
            <div class="section-desc">{desc}</div>
            """,
            unsafe_allow_html=True,
        )
        yield


def section_header(icon: str, title: str, desc: str = "") -> None:
    desc_html = f'<div class="section-desc">{desc}</div>' if desc else ""
    st.markdown(
        f"""
        <div class="section-title"><span class="st-ico">{icon}</span>{title}</div>
        {desc_html}
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, bullets: list[str]) -> None:
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="ib-title">✨ {title}</div>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(icon: str, text: str, tag: str, variant: str = "") -> str:
    return f"""
        <div class="insight-card {variant}">
            <div class="insight-ico">{icon}</div>
            <div>
                <div class="insight-text">{text}</div>
                <div class="insight-tag">{tag}</div>
            </div>
        </div>
    """


def ai_panel_card(icon: str, label: str, value: str, desc: str, gradient: str, delay: float = 0.0) -> str:
    """Return HTML for a colourful AI Insights Panel card."""
    return f"""
        <div class="ai-panel-card" style="background:{gradient};animation-delay:{delay}s;">
            <div class="ai-ico">{icon}</div>
            <div class="ai-label">{label}</div>
            <div class="ai-value">{value}</div>
            <div class="ai-desc">{desc}</div>
        </div>
    """


# ============================================================================
#  AI INSIGHT ENGINE
# ============================================================================
def build_insights(df: pd.DataFrame) -> list[dict]:
    """Generate automatic business insights from the filtered data."""
    insights: list[dict] = []

    if df.empty:
        return insights

    # --- Traffic impact -----------------------------------------------------
    traf = agg_mean(df, "Traffic")
    if len(traf) >= 2:
        slowest = traf.iloc[-1]
        fastest = traf.iloc[0]
        pct = (slowest["Avg_Delivery_Time"] - fastest["Avg_Delivery_Time"]) / fastest[
            "Avg_Delivery_Time"
        ] * 100
        insights.append(
            dict(
                icon="🚦",
                text=f"<b>{slowest['Traffic']}</b> traffic increases average delivery "
                f"time by <b>{pct:.0f}%</b> versus <b>{fastest['Traffic']}</b> traffic.",
                tag="Traffic Impact",
                variant="ic-pink",
            )
        )

    # --- Vehicle impact -----------------------------------------------------
    veh = agg_mean(df, "Vehicle")
    if len(veh) >= 2:
        fast_v = veh.iloc[0]
        slow_v = veh.iloc[-1]
        pct = (slow_v["Avg_Delivery_Time"] - fast_v["Avg_Delivery_Time"]) / slow_v[
            "Avg_Delivery_Time"
        ] * 100
        insights.append(
            dict(
                icon="🏍️",
                text=f"<b>{fast_v['Vehicle'].title()}</b> deliveries are <b>{pct:.0f}%</b> "
                f"faster than <b>{slow_v['Vehicle'].title()}</b> on average.",
                tag="Fleet Performance",
                variant="ic-cyan",
            )
        )

    # --- Area impact --------------------------------------------------------
    area = agg_mean(df, "Area")
    if len(area) >= 1:
        worst = area.iloc[-1]
        insights.append(
            dict(
                icon="📍",
                text=f"<b>{worst['Area']}</b> records the highest average delay at "
                f"<b>{worst['Avg_Delivery_Time']:.0f} min</b> per delivery.",
                tag="Geographic Bottleneck",
                variant="ic-purple",
            )
        )

    # --- Weather impact -----------------------------------------------------
    wea = agg_mean(df, "Weather")
    if len(wea) >= 1:
        worst_w = wea.iloc[-1]
        insights.append(
            dict(
                icon="🌦️",
                text=f"<b>{worst_w['Weather']}</b> conditions are the most disruptive, "
                f"averaging <b>{worst_w['Avg_Delivery_Time']:.0f} min</b>.",
                tag="Weather Impact",
                variant="ic-blue",
            )
        )

    # --- Late delivery ------------------------------------------------------
    late_pct = df["Late_Delivery"].mean() * 100
    insights.append(
        dict(
            icon="⏰",
            text=f"<b>{late_pct:.1f}%</b> of all deliveries are flagged as late "
            f"(above mean + 1 std dev).",
            tag="SLA Risk",
            variant="ic-pink",
        )
    )

    # --- Age group / rating -------------------------------------------------
    age = agg_mean(df, "Age_Group")
    if len(age) >= 1:
        best_age = age.iloc[0]
        insights.append(
            dict(
                icon="👥",
                text=f"Agents aged <b>{best_age['Age_Group']}</b> are the fastest cohort, "
                f"averaging <b>{best_age['Avg_Delivery_Time']:.0f} min</b>.",
                tag="Workforce Insight",
                variant="ic-cyan",
            )
        )

    # --- Rating correlation -------------------------------------------------
    if df["Agent_Rating"].nunique() > 2:
        corr = df["Agent_Rating"].corr(df["Delivery_Time"])
        direction = "faster" if corr < 0 else "slower"
        insights.append(
            dict(
                icon="⭐",
                text=f"Higher agent ratings correlate with <b>{direction}</b> deliveries "
                f"(correlation r = <b>{corr:.2f}</b>).",
                tag="Quality vs Speed",
                variant="ic-purple",
            )
        )

    # --- Category -----------------------------------------------------------
    cat = agg_mean(df, "Category")
    if len(cat) >= 1:
        worst_c = cat.iloc[-1]
        insights.append(
            dict(
                icon="📦",
                text=f"<b>{worst_c['Category']}</b> is the slowest product category at "
                f"<b>{worst_c['Avg_Delivery_Time']:.0f} min</b> average.",
                tag="Category Insight",
                variant="ic-blue",
            )
        )

    return insights


def compute_delay_risk(df: pd.DataFrame) -> float:
    """
    Predict the probability of a late delivery under the highest-risk
    operating segment (traffic x weather). Falls back to the overall late rate.
    """
    if df.empty:
        return 0.0
    base = df["Late_Delivery"].mean() * 100
    try:
        combo = (
            df.groupby(["Traffic", "Weather"], observed=True)["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
        )
        combo = combo[combo["count"] >= max(20, int(0.002 * len(df)))]
        if not combo.empty:
            return float(combo["mean"].max() * 100)
    except Exception:
        pass
    return float(base)


def build_ai_panel(df: pd.DataFrame) -> list[dict]:
    """Build the 5 colourful AI Insights Panel cards."""
    cards: list[dict] = []

    veh = agg_mean(df, "Vehicle")
    if len(veh):
        best = veh.iloc[0]
        cards.append(
            dict(
                key="vehicle", icon="🏍️", label="Best Performing Vehicle",
                value=best["Vehicle"].title(),
                desc=f"Fastest fleet at {best['Avg_Delivery_Time']:.0f} min average "
                     f"across {int(best['Deliveries']):,} trips.",
            )
        )

    traf = agg_mean(df, "Traffic")
    if len(traf):
        worst = traf.iloc[-1]
        cards.append(
            dict(
                key="traffic", icon="🚦", label="Worst Traffic Condition",
                value=str(worst["Traffic"]),
                desc=f"Adds the most delay at {worst['Avg_Delivery_Time']:.0f} min "
                     f"average per delivery.",
            )
        )

    area = agg_mean(df, "Area")
    if len(area):
        fast = area.iloc[0]
        cards.append(
            dict(
                key="area", icon="📍", label="Fastest Area",
                value=str(fast["Area"]),
                desc=f"Quickest zone at {fast['Avg_Delivery_Time']:.0f} min average.",
            )
        )

    risk = compute_delay_risk(df)
    cards.append(
        dict(
            key="risk", icon="⚠️", label="Delay Risk Prediction",
            value=f"{risk:.0f}%",
            desc="Predicted late-delivery probability in the highest-risk "
                 "traffic + weather segment.",
        )
    )

    late_pct = df["Late_Delivery"].mean() * 100
    cards.append(
        dict(
            key="summary", icon="🧠", label="Delivery Performance Summary",
            value=f"{late_pct:.1f}% Late",
            desc=f"{len(df):,} deliveries analysed · avg {df['Delivery_Time'].mean():.0f} min "
                 f"· {df['Agent_Rating'].mean():.2f}★ rating.",
        )
    )
    return cards


def build_agent_scatter(df: pd.DataFrame) -> go.Figure:
    """Rating vs Delivery Time scatter (Scattergl) with an OLS trendline.
    Uses the FULL dataset via WebGL rendering for smooth performance."""
    fig = go.Figure()
    for grp, color in AGE_GROUP_COLORS.items():
        sub = df[df["Age_Group"] == grp]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub["Agent_Rating"],
                y=sub["Delivery_Time"],
                mode="markers",
                name=grp,
                marker=dict(color=color, size=7, opacity=0.40, line=dict(width=0)),
                hovertemplate="Rating %{x:.1f}<br>Time %{y:.0f} min"
                f"<extra>{grp}</extra>",
            )
        )
    # Manual OLS trendline (numpy) — avoids statsmodels overhead on full data
    x = df["Agent_Rating"].to_numpy(dtype=float)
    y = df["Delivery_Time"].to_numpy(dtype=float)
    if len(x) > 2 and np.ptp(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(
            go.Scattergl(
                x=xr,
                y=m * xr + b,
                mode="lines",
                name="OLS Trend",
                line=dict(color="#0F172A", width=3, dash="dash"),
                hoverinfo="skip",
            )
        )
    return fig


# ============================================================================
#  EXPORT HELPERS
# ============================================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(df: pd.DataFrame, insights: list[dict]) -> bytes:
    """Build a multi-page PDF report with matplotlib (no extra deps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["font.family"] = "DejaVu Sans"
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ---- Page 1: Cover + KPIs -----------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#F5F8FC")
        fig.text(0.5, 0.93, "LogiSight Analytics", ha="center", fontsize=26,
                 fontweight="bold", color="#0066FF")
        fig.text(0.5, 0.90, "Last Mile Delivery Intelligence Report", ha="center",
                 fontsize=13, color="#64748B")
        fig.text(0.5, 0.865, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 ha="center", fontsize=9, color="#94A3B8")

        kpis = [
            ("Total Deliveries", f"{len(df):,}"),
            ("Avg Delivery Time", f"{df['Delivery_Time'].mean():.1f} min"),
            ("Late Delivery %", f"{df['Late_Delivery'].mean()*100:.1f}%"),
            ("Avg Agent Rating", f"{df['Agent_Rating'].mean():.2f} ★"),
            ("Fastest Vehicle", agg_mean(df, 'Vehicle').iloc[0]['Vehicle'].title()),
            ("Best Area", agg_mean(df, 'Area').iloc[0]['Area']),
        ]
        for i, (label, value) in enumerate(kpis):
            col = i % 2
            row = i // 2
            x = 0.08 + col * 0.47
            y = 0.74 - row * 0.12
            box = plt.Rectangle((x, y), 0.42, 0.10, facecolor="white",
                                edgecolor="#0066FF", linewidth=1.2)
            fig.add_artist(box)
            fig.text(x + 0.02, y + 0.065, label, fontsize=9, color="#64748B")
            fig.text(x + 0.02, y + 0.022, str(value), fontsize=15,
                     fontweight="bold", color="#0F172A")

        fig.text(0.08, 0.30, "Key AI Insights", fontsize=14, fontweight="bold",
                 color="#0F172A")
        y = 0.26
        for ins in insights[:6]:
            txt = ins["text"].replace("<b>", "").replace("</b>", "")
            fig.text(0.08, y, f"• {txt[:95]}", fontsize=9, color="#334155", wrap=True)
            y -= 0.028
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: Performance charts -----------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#F5F8FC")

        veh = agg_mean(df, "Vehicle")
        axes[0, 0].barh(veh["Vehicle"], veh["Avg_Delivery_Time"],
                        color=["#38BDF8", "#3B82F6", "#8271B7", "#FF55C5"][:len(veh)])
        axes[0, 0].set_title("Vehicle Performance", fontweight="bold", color="#0F172A")
        axes[0, 0].set_xlabel("Avg Delivery Time (min)")

        traf = agg_mean(df, "Traffic")
        axes[0, 1].bar(traf["Traffic"], traf["Avg_Delivery_Time"],
                       color=["#38BDF8", "#3B82F6", "#8271B7", "#FF55C5"][:len(traf)])
        axes[0, 1].set_title("Traffic Impact", fontweight="bold", color="#0F172A")
        axes[0, 1].set_ylabel("Avg Delivery Time (min)")

        axes[1, 0].hist(df["Delivery_Time"], bins=30, color="#3B82F6", alpha=0.85)
        axes[1, 0].set_title("Delivery Time Distribution", fontweight="bold", color="#0F172A")
        axes[1, 0].set_xlabel("Delivery Time (min)")

        wea = agg_mean(df, "Weather")
        axes[1, 1].barh(wea["Weather"], wea["Avg_Delivery_Time"], color="#8271B7")
        axes[1, 1].set_title("Weather Impact", fontweight="bold", color="#0F172A")
        axes[1, 1].set_xlabel("Avg Delivery Time (min)")

        for ax in axes.flat:
            ax.set_facecolor("white")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: Scorecard table --------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#F5F8FC")
        fig.text(0.5, 0.95, "Delivery Performance Scorecard", ha="center",
                 fontsize=16, fontweight="bold", color="#0F172A")
        ax = fig.add_axes([0.08, 0.35, 0.84, 0.55])
        ax.axis("off")
        sc = agg_mean(df, "Area").copy()
        sc["Late_%"] = sc["Area"].map(
            agg_late(df, "Area").set_index("Area")["Late_Pct"]
        )
        table_data = [
            [r["Area"], f"{r['Avg_Delivery_Time']:.1f}", f"{r['Deliveries']:,}",
             f"{r['Late_%']:.1f}%"]
            for _, r in sc.iterrows()
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["Area", "Avg Time (min)", "Deliveries", "Late %"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#0066FF")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("white")
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.read()


def build_snapshot_png(df: pd.DataFrame) -> bytes | None:
    """Export a composed dashboard snapshot as PNG using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = "DejaVu Sans"
        fig = plt.figure(figsize=(14, 8), facecolor="#F5F8FC")

        fig.text(0.5, 0.955, "LogiSight Analytics — Dashboard Snapshot",
                 ha="center", fontsize=20, fontweight="bold", color="#0066FF")
        fig.text(0.5, 0.925, f"{datetime.now():%Y-%m-%d %H:%M} · "
                 f"{len(df):,} deliveries · Avg {df['Delivery_Time'].mean():.1f} min · "
                 f"Late {df['Late_Delivery'].mean()*100:.1f}%",
                 ha="center", fontsize=11, color="#64748B")

        ax1 = fig.add_axes([0.06, 0.55, 0.40, 0.30])
        veh = agg_mean(df, "Vehicle")
        ax1.barh(veh["Vehicle"], veh["Avg_Delivery_Time"],
                 color=["#38BDF8", "#3B82F6", "#8271B7", "#FF55C5"][:len(veh)])
        ax1.set_title("Vehicle Performance", fontweight="bold", color="#0F172A")
        ax1.set_xlabel("Avg Delivery Time (min)")

        ax2 = fig.add_axes([0.56, 0.55, 0.40, 0.30])
        traf = agg_mean(df, "Traffic")
        ax2.bar(traf["Traffic"], traf["Avg_Delivery_Time"],
                color=["#38BDF8", "#3B82F6", "#8271B7", "#FF55C5"][:len(traf)])
        ax2.set_title("Traffic Impact", fontweight="bold", color="#0F172A")
        ax2.set_ylabel("Avg Delivery Time (min)")

        ax3 = fig.add_axes([0.06, 0.09, 0.40, 0.30])
        ax3.hist(df["Delivery_Time"], bins=35, color="#3B82F6", alpha=0.85)
        ax3.axvline(df["Delivery_Time"].mean(), color="#FF55C5", linestyle="--")
        ax3.set_title("Delivery Time Distribution", fontweight="bold", color="#0F172A")
        ax3.set_xlabel("Delivery Time (min)")

        ax4 = fig.add_axes([0.56, 0.09, 0.40, 0.30])
        wea = agg_mean(df, "Weather")
        ax4.barh(wea["Weather"], wea["Avg_Delivery_Time"], color="#8271B7")
        ax4.set_title("Weather Impact", fontweight="bold", color="#0F172A")
        ax4.set_xlabel("Avg Delivery Time (min)")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("white")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor="#F5F8FC")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ============================================================================
#  MAIN APPLICATION
# ============================================================================
def main() -> None:
    raw, source_path = resolve_dataset()

    with st.spinner("🚚 Loading LogiSight Analytics engine…"):
        df_all = clean_data(raw)

    # ------------------------------------------------------------------
    #  SIDEBAR FILTERS  (glassmorphism rounded containers)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sb-title">🚚 LogiSight</div>
                <div class="sb-sub">ANALYTICS CONTROL PANEL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### 🎛️ Filters")
        st.caption("All charts update dynamically.")

        with st.container(key="filter_env"):
            st.markdown('<div class="sidebar-section-label">🌦️ Environment</div>',
                        unsafe_allow_html=True)
            weather_sel = st.multiselect(
                "Weather", sorted(df_all["Weather"].dropna().unique()),
                default=sorted(df_all["Weather"].dropna().unique()),
            )
            traffic_sel = st.multiselect(
                "Traffic", sorted(df_all["Traffic"].dropna().unique()),
                default=sorted(df_all["Traffic"].dropna().unique()),
            )

        with st.container(key="filter_fleet"):
            st.markdown('<div class="sidebar-section-label">🚗 Fleet & Geography</div>',
                        unsafe_allow_html=True)
            vehicle_sel = st.multiselect(
                "Vehicle Type", sorted(df_all["Vehicle"].dropna().unique()),
                default=sorted(df_all["Vehicle"].dropna().unique()),
            )
            area_sel = st.multiselect(
                "Area", sorted(df_all["Area"].dropna().unique()),
                default=sorted(df_all["Area"].dropna().unique()),
            )

        with st.container(key="filter_people"):
            st.markdown('<div class="sidebar-section-label">📦 Product & People</div>',
                        unsafe_allow_html=True)
            category_sel = st.multiselect(
                "Category", sorted(df_all["Category"].dropna().unique()),
                default=sorted(df_all["Category"].dropna().unique()),
            )
            age_sel = st.multiselect(
                "Agent Age Group", ["Under 25", "25-40", "40+"],
                default=["Under 25", "25-40", "40+"],
            )

        with st.container(key="filter_time"):
            st.markdown('<div class="sidebar-section-label">📅 Time Period</div>',
                        unsafe_allow_html=True)
            min_d = df_all["Order_Date"].min().date()
            max_d = df_all["Order_Date"].max().date()
            date_range = st.date_input(
                "Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d
            )

        st.markdown("---")
        st.caption(f"📁 Source: `{source_path}`")
        st.caption(f"🗂️ Records loaded: **{len(df_all):,}**")

    # ------------------------------------------------------------------
    #  APPLY FILTERS
    # ------------------------------------------------------------------
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    mask = (
        df_all["Weather"].isin(weather_sel)
        & df_all["Traffic"].isin(traffic_sel)
        & df_all["Vehicle"].isin(vehicle_sel)
        & df_all["Area"].isin(area_sel)
        & df_all["Category"].isin(category_sel)
        & df_all["Age_Group"].isin(age_sel)
        & (df_all["Order_Date"].dt.date >= start_d)
        & (df_all["Order_Date"].dt.date <= end_d)
    )
    df = df_all[mask].copy()

    # ------------------------------------------------------------------
    #  NAVBAR + PAGE HEADER
    # ------------------------------------------------------------------
    date_label = f"{start_d:%d %b %Y} – {end_d:%d %b %Y}"
    render_navbar(date_label)

    st.markdown(
        '<div class="ls-page-title">Last Mile Delivery Intelligence</div>'
        '<div class="ls-page-sub">Real-time operational analytics across fleet, '
        'agents, geography and demand categories.</div>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("⚠️ No records match the current filters. Please widen your selection.")
        st.stop()

    # ------------------------------------------------------------------
    #  KPI COMPUTATION
    # ------------------------------------------------------------------
    total_deliveries = len(df)
    avg_time = df["Delivery_Time"].mean()
    late_pct = df["Late_Delivery"].mean() * 100
    avg_rating = df["Agent_Rating"].mean()

    veh_rank = agg_mean(df, "Vehicle")
    fastest_vehicle = veh_rank.iloc[0]["Vehicle"].title() if len(veh_rank) else "—"
    fastest_veh_time = veh_rank.iloc[0]["Avg_Delivery_Time"] if len(veh_rank) else 0

    area_rank = agg_mean(df, "Area")
    best_area = area_rank.iloc[0]["Area"] if len(area_rank) else "—"
    best_area_time = area_rank.iloc[0]["Avg_Delivery_Time"] if len(area_rank) else 0

    # ==================================================================
    #  ROW 1 — PREMIUM KPI CARDS
    # ==================================================================
    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="medium")
    with k1:
        st.markdown(kpi_card("📦", f"{total_deliveries:,}", "Total Deliveries",
                             "Filtered order volume", KPI_GRADIENTS["deliveries"], 0.05),
                    unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("⏱️", f"{avg_time:.1f}", "Avg Delivery Time",
                             "Minutes across all routes", KPI_GRADIENTS["time"], 0.10),
                    unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("⚠️", f"{late_pct:.1f}%", "Late Delivery %",
                             "Above mean + 1σ threshold", KPI_GRADIENTS["late"], 0.15),
                    unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card("⭐", f"{avg_rating:.2f}", "Avg Agent Rating",
                             "Out of 5.00 stars", KPI_GRADIENTS["rating"], 0.20),
                    unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card("🏍️", fastest_vehicle, "Fastest Vehicle",
                             f"Avg {fastest_veh_time:.0f} min", KPI_GRADIENTS["vehicle"], 0.25),
                    unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card("🏆", best_area, "Best Performing Area",
                             f"Avg {best_area_time:.0f} min", KPI_GRADIENTS["area"], 0.30),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    # ==================================================================
    #  TABS — Dashboard / Insights / Performance / Operations
    # ==================================================================
    tab_dash, tab_insights, tab_perf, tab_ops = st.tabs(
        ["📊 Dashboard", "💡 Insights", "📈 Performance", "⚙️ Operations"]
    )

    # ==================================================================
    #  TAB 1 — DASHBOARD  (ROW 2 → ROW 6)
    # ==================================================================
    with tab_dash:

        # ---------------- ROW 2 : Delay Analyzer | Vehicle Comparison ----
        r2c1, r2c2 = st.columns(2, gap="large")
        with r2c1:
            with chart_card("delay", "🌦️", "Delay Analyzer",
                            "Average delivery time by weather & traffic condition."):
                grp = (
                    df.groupby(["Weather", "Traffic"], observed=True)["Delivery_Time"]
                    .mean().reset_index()
                    .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
                )
                fig1 = px.bar(
                    grp, x="Weather", y="Avg_Delivery_Time", color="Traffic",
                    barmode="group", color_discrete_map=TRAFFIC_COLORS,
                    labels={"Avg_Delivery_Time": "Avg Delivery Time (min)", "Weather": ""},
                )
                fig1.update_traces(marker_line_width=0, opacity=0.95)
                style_fig(fig1, height=330)
                st.plotly_chart(fig1, width="stretch", key="pc_delay")

                traf_sorted = agg_mean(df, "Traffic")
                wea_sorted = agg_mean(df, "Weather")
                bullets = []
                if len(traf_sorted) >= 2:
                    bullets.append(
                        f"<b>{traf_sorted.iloc[-1]['Traffic']}</b> traffic adds "
                        f"{traf_sorted.iloc[-1]['Avg_Delivery_Time'] - traf_sorted.iloc[0]['Avg_Delivery_Time']:.0f} min "
                        f"vs <b>{traf_sorted.iloc[0]['Traffic']}</b>."
                    )
                if len(wea_sorted) >= 1:
                    bullets.append(
                        f"<b>{wea_sorted.iloc[-1]['Weather']}</b> is the most disruptive weather "
                        f"({wea_sorted.iloc[-1]['Avg_Delivery_Time']:.0f} min avg)."
                    )
                insight_box("Auto-Generated Insights", bullets)

        with r2c2:
            with chart_card("vehicle", "🚗", "Vehicle Comparison",
                            "Average delivery time by vehicle type — fastest highlighted."):
                veh = agg_mean(df, "Vehicle")
                fig2 = px.bar(
                    veh, x="Avg_Delivery_Time", y="Vehicle", orientation="h",
                    labels={"Avg_Delivery_Time": "Avg Delivery Time (min)", "Vehicle": ""},
                    color="Avg_Delivery_Time",
                    color_continuous_scale=[PRIMARY_3, PRIMARY_2, PRIMARY_1],
                )
                fig2.update_traces(marker_line_width=0)
                fig2.update_layout(coloraxis_showscale=False)
                style_fig(fig2, height=330, showlegend=False)
                st.plotly_chart(fig2, width="stretch", key="pc_vehicle")

                bullets = []
                if len(veh) >= 2:
                    fast, slow = veh.iloc[0], veh.iloc[-1]
                    pct = (slow["Avg_Delivery_Time"] - fast["Avg_Delivery_Time"]) / slow[
                        "Avg_Delivery_Time"] * 100
                    bullets.append(
                        f"<b>{fast['Vehicle'].title()}</b> is the fastest fleet at "
                        f"{fast['Avg_Delivery_Time']:.0f} min."
                    )
                    bullets.append(
                        f"<b>{slow['Vehicle'].title()}</b> is {pct:.0f}% slower — "
                        f"reallocate high-priority orders."
                    )
                insight_box("Fleet Insights", bullets)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ---------------- ROW 3 : Agent Performance | Area Heatmap -------
        r3c1, r3c2 = st.columns(2, gap="large")
        with r3c1:
            with chart_card("agent", "👤", "Agent Performance",
                            "Rating vs delivery time by age group with OLS trendline."):
                fig3 = build_agent_scatter(df)
                style_fig(fig3, height=360)
                st.plotly_chart(fig3, width="stretch", key="pc_agent")

                corr_r = df["Agent_Rating"].corr(df["Delivery_Time"])
                insight_box("Agent Insights", [
                    f"Rating–speed correlation is <b>r = {corr_r:.2f}</b> "
                    f"({'faster' if corr_r < 0 else 'slower'} deliveries with higher ratings).",
                    f"<b>{len(df):,}</b> deliveries plotted in full (WebGL accelerated).",
                ])

        with r3c2:
            with chart_card("heatmap", "🔥", "Area Heatmap",
                            "Average delivery time across areas and weather conditions."):
                heat = (
                    df.groupby(["Area", "Weather"], observed=True)["Delivery_Time"]
                    .mean().reset_index()
                    .pivot(index="Area", columns="Weather", values="Delivery_Time")
                )
                fig4 = px.imshow(
                    heat,
                    color_continuous_scale=[PRIMARY_3, PRIMARY_2, ACCENT_3, ACCENT_1],
                    aspect="auto", text_auto=".0f",
                    labels=dict(color="Avg Time (min)"),
                )
                fig4.update_traces(textfont=dict(size=11, color="white"))
                style_fig(fig4, height=360, showlegend=False)
                st.plotly_chart(fig4, width="stretch", key="pc_heatmap")

                if not heat.empty:
                    flat = heat.stack()
                    worst_combo = flat.idxmax()
                    insight_box("Geographic Insights", [
                        f"Highest delay: <b>{worst_combo[0]}</b> under "
                        f"<b>{worst_combo[1]}</b> at <b>{flat.max():.0f} min</b>.",
                        "Darker cells mark bottlenecks needing intervention.",
                    ])

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ---------------- ROW 4 : Category Visualizer | Monthly Trends ---
        r4c1, r4c2 = st.columns(2, gap="large")
        with r4c1:
            with chart_card("category", "📦", "Category Visualizer",
                            "Delivery time distribution across product categories."):
                fig5 = px.box(
                    df, x="Category", y="Delivery_Time", color="Category",
                    labels={"Delivery_Time": "Delivery Time (min)", "Category": ""},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    points=False,
                )
                style_fig(fig5, height=360, showlegend=False)
                fig5.update_xaxes(tickangle=-30)
                st.plotly_chart(fig5, width="stretch", key="pc_category")

                cat = agg_mean(df, "Category")
                insight_box("Category Insights", [
                    f"<b>{cat.iloc[-1]['Category']}</b> is the slowest "
                    f"({cat.iloc[-1]['Avg_Delivery_Time']:.0f} min avg).",
                    f"<b>{cat.iloc[0]['Category']}</b> is the fastest "
                    f"({cat.iloc[0]['Avg_Delivery_Time']:.0f} min avg).",
                ])

        with r4c2:
            with chart_card("monthly", "📈", "Monthly Trends",
                            "Average delivery time and volume trend over time."):
                monthly = (
                    df.groupby("Month", observed=True)
                    .agg(Avg_Time=("Delivery_Time", "mean"),
                         Deliveries=("Delivery_Time", "count"))
                    .reset_index()
                )
                fig_m = px.line(
                    monthly, x="Month", y="Avg_Time", markers=True,
                    labels={"Avg_Time": "Avg Delivery Time (min)", "Month": ""},
                    color_discrete_sequence=[PRIMARY_1],
                )
                fig_m.update_traces(line=dict(width=3), marker=dict(size=8))
                style_fig(fig_m, height=360, showlegend=False)
                st.plotly_chart(fig_m, width="stretch", key="pc_monthly")

                if len(monthly) >= 2:
                    best_m = monthly.loc[monthly["Avg_Time"].idxmin()]
                    worst_m = monthly.loc[monthly["Avg_Time"].idxmax()]
                    insight_box("Trend Insights", [
                        f"Fastest month: <b>{best_m['Month']}</b> at {best_m['Avg_Time']:.0f} min.",
                        f"Slowest month: <b>{worst_m['Month']}</b> at {worst_m['Avg_Time']:.0f} min.",
                    ])

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ---------------- ROW 5 : Late Delivery | Traffic Impact ---------
        r5c1, r5c2 = st.columns(2, gap="large")
        with r5c1:
            with chart_card("late", "⏰", "Late Delivery Analysis",
                            "Late delivery percentage by area (above mean + 1σ)."):
                late_area = agg_late(df, "Area")
                fig_l = px.bar(
                    late_area, x="Area", y="Late_Pct", color="Late_Pct",
                    labels={"Late_Pct": "Late Delivery %", "Area": ""},
                    color_continuous_scale=[PRIMARY_3, ACCENT_1], text_auto=".1f",
                )
                fig_l.update_layout(coloraxis_showscale=False)
                style_fig(fig_l, height=340, showlegend=False)
                st.plotly_chart(fig_l, width="stretch", key="pc_late")

                if len(late_area):
                    worst = late_area.iloc[0]
                    insight_box("SLA Risk", [
                        f"<b>{worst['Area']}</b> has the highest late rate at "
                        f"<b>{worst['Late_Pct']:.1f}%</b>.",
                        f"Overall late rate: <b>{df['Late_Delivery'].mean()*100:.1f}%</b> "
                        f"of {len(df):,} deliveries.",
                    ])

        with r5c2:
            with chart_card("traffic", "🚦", "Traffic Impact Analysis",
                            "How traffic conditions drive average delivery time."):
                traf = agg_mean(df, "Traffic")
                fig_t = px.bar(
                    traf, x="Traffic", y="Avg_Delivery_Time", color="Traffic",
                    labels={"Avg_Delivery_Time": "Avg Time (min)", "Traffic": ""},
                    color_discrete_map=TRAFFIC_COLORS, text_auto=".0f",
                )
                style_fig(fig_t, height=340, showlegend=False)
                st.plotly_chart(fig_t, width="stretch", key="pc_traffic")

                if len(traf) >= 2:
                    delta = traf.iloc[-1]["Avg_Delivery_Time"] - traf.iloc[0]["Avg_Delivery_Time"]
                    insight_box("Traffic Insights", [
                        f"<b>{traf.iloc[-1]['Traffic']}</b> traffic is slowest at "
                        f"{traf.iloc[-1]['Avg_Delivery_Time']:.0f} min.",
                        f"Delta vs <b>{traf.iloc[0]['Traffic']}</b> traffic: "
                        f"<b>+{delta:.0f} min</b>.",
                    ])

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ---------------- ROW 6 : AI INSIGHTS PANEL ----------------------
        section_header("🧠", "AI Insights Panel",
                       "Automated intelligence distilled from your filtered dataset.")
        ai_cards = build_ai_panel(df)
        ai_cols = st.columns(len(ai_cards), gap="medium")
        for i, (col, card) in enumerate(zip(ai_cols, ai_cards)):
            with col:
                st.markdown(
                    ai_panel_card(card["icon"], card["label"], card["value"],
                                  card["desc"], AI_GRADIENTS[card["key"]], i * 0.08),
                    unsafe_allow_html=True,
                )

    # ==================================================================
    #  TAB 2 — INSIGHTS (AI INSIGHT ENGINE)
    # ==================================================================
    with tab_insights:
        section_header("🧠", "AI Insight Engine",
                       "Automatic business intelligence derived from your filtered data.")

        insights = build_insights(df)
        left, right = st.columns(2, gap="large")
        for i, ins in enumerate(insights):
            target = left if i % 2 == 0 else right
            with target:
                st.markdown(
                    insight_card(ins["icon"], ins["text"], ins["tag"], ins["variant"]),
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        section_header("📌", "Executive Summary",
                       "Top-line narrative for stakeholders.")
        top_ins = insights[:4]
        summary = " ".join(
            i["text"].replace("<b>", "").replace("</b>", "") for i in top_ins
        )
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="insight-text" style="font-size:14px;line-height:1.75;">
                    {summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==================================================================
    #  TAB 3 — PERFORMANCE (ADVANCED ANALYTICS)
    # ==================================================================
    with tab_perf:
        section_header("⚡", "Advanced Analytics",
                       "Ten premium enhancement visuals beyond the core dashboard.")

        # 1 & 2 ---------------------------------------------------------
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with chart_card("hist", "📊", "1 · Delivery Time Histogram",
                            "Distribution of delivery durations with mean marker."):
                fig_h = px.histogram(
                    df, x="Delivery_Time", nbins=40,
                    labels={"Delivery_Time": "Delivery Time (min)"},
                    color_discrete_sequence=[PRIMARY_2],
                )
                fig_h.add_vline(
                    x=df["Delivery_Time"].mean(), line_dash="dash", line_color=ACCENT_1,
                    annotation_text=f"Mean {df['Delivery_Time'].mean():.0f}",
                )
                style_fig(fig_h, height=340, showlegend=False)
                st.plotly_chart(fig_h, width="stretch", key="pc_hist")
        with c2:
            with chart_card("monthly2", "📈", "2 · Monthly Trend Line",
                            "Average delivery time per month."):
                monthly = (
                    df.groupby("Month", observed=True)["Delivery_Time"].mean().reset_index()
                )
                fig_m = px.line(
                    monthly, x="Month", y="Delivery_Time", markers=True,
                    labels={"Delivery_Time": "Avg Delivery Time (min)", "Month": ""},
                    color_discrete_sequence=[PRIMARY_1],
                )
                fig_m.update_traces(line=dict(width=3), marker=dict(size=8))
                style_fig(fig_m, height=340, showlegend=False)
                st.plotly_chart(fig_m, width="stretch", key="pc_monthly2")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # 3 & 4 ---------------------------------------------------------
        c3, c4 = st.columns(2, gap="large")
        with c3:
            with chart_card("late2", "⏰", "3 · Late Delivery % by Area",
                            "Share of deliveries exceeding the late threshold."):
                late_area = agg_late(df, "Area")
                fig_l = px.bar(
                    late_area, x="Area", y="Late_Pct", color="Late_Pct",
                    labels={"Late_Pct": "Late Delivery %", "Area": ""},
                    color_continuous_scale=[PRIMARY_3, ACCENT_1], text_auto=".1f",
                )
                fig_l.update_layout(coloraxis_showscale=False)
                style_fig(fig_l, height=340, showlegend=False)
                st.plotly_chart(fig_l, width="stretch", key="pc_late2")
        with c4:
            with chart_card("agents", "👥", "4 · Agent Count by Area",
                            "Number of distinct agents operating per area."):
                agent_area = (
                    df.groupby("Area", observed=True)["Agent_ID"].nunique().reset_index()
                    .rename(columns={"Agent_ID": "Agents"})
                )
                fig_a = px.bar(
                    agent_area, x="Area", y="Agents", color="Area",
                    color_discrete_sequence=GRADIENT_SEQ, text_auto=True,
                )
                style_fig(fig_a, height=340, showlegend=False)
                st.plotly_chart(fig_a, width="stretch", key="pc_agents")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # 5 & 6 ---------------------------------------------------------
        c5, c6 = st.columns(2, gap="large")
        with c5:
            with chart_card("traffic2", "🚦", "5 · Traffic Impact Analysis",
                            "Average delivery time by traffic condition."):
                traf = agg_mean(df, "Traffic")
                fig_t = px.bar(
                    traf, x="Traffic", y="Avg_Delivery_Time", color="Traffic",
                    labels={"Avg_Delivery_Time": "Avg Time (min)", "Traffic": ""},
                    color_discrete_map=TRAFFIC_COLORS, text_auto=".0f",
                )
                style_fig(fig_t, height=340, showlegend=False)
                st.plotly_chart(fig_t, width="stretch", key="pc_traffic2")
        with c6:
            with chart_card("weather2", "🌦️", "6 · Weather Impact Analysis",
                            "Average delivery time by weather condition."):
                wea = agg_mean(df, "Weather")
                fig_w = px.bar(
                    wea, x="Weather", y="Avg_Delivery_Time", color="Weather",
                    labels={"Avg_Delivery_Time": "Avg Time (min)", "Weather": ""},
                    color_discrete_sequence=GRADIENT_SEQ, text_auto=".0f",
                )
                style_fig(fig_w, height=340, showlegend=False)
                st.plotly_chart(fig_w, width="stretch", key="pc_weather2")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # 7 -------------------------------------------------------------
        with chart_card("corr", "🔗", "7 · Delivery Time Correlation Matrix",
                        "Relationships between key numeric operational variables."):
            num_cols = ["Delivery_Time", "Agent_Rating", "Agent_Age", "Late_Delivery"]
            corr_df = df[num_cols].copy()
            corr_df["Late_Delivery"] = corr_df["Late_Delivery"].astype(int)
            corr = corr_df.corr()
            fig_c = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale=[ACCENT_1, "#FFFFFF", PRIMARY_1],
                zmin=-1, zmax=1,
            )
            fig_c.update_traces(textfont=dict(size=12))
            style_fig(fig_c, height=400, showlegend=False)
            st.plotly_chart(fig_c, width="stretch", key="pc_corr")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # 8 & 9 ---------------------------------------------------------
        c8, c9 = st.columns(2, gap="large")
        with c8:
            with chart_card("topagents", "🏅", "8 · Top 10 Best Agents",
                            "Highest rated agents with the fastest average times."):
                agents = (
                    df.groupby("Agent_ID", observed=True)
                    .agg(Avg_Rating=("Agent_Rating", "mean"),
                         Avg_Time=("Delivery_Time", "mean"),
                         Deliveries=("Order_ID", "count"))
                    .reset_index()
                )
                agents = agents[agents["Deliveries"] >= max(3, int(agents["Deliveries"].median()))]
                top_agents = agents.sort_values(
                    ["Avg_Rating", "Avg_Time"], ascending=[False, True]
                ).head(10)
                fig_ta = px.bar(
                    top_agents, x="Avg_Rating", y="Agent_ID", orientation="h",
                    labels={"Avg_Rating": "Avg Rating", "Agent_ID": ""},
                    color="Avg_Rating", color_continuous_scale=[PRIMARY_3, PRIMARY_1],
                    hover_data=["Avg_Time", "Deliveries"], text_auto=".2f",
                )
                fig_ta.update_layout(coloraxis_showscale=False)
                fig_ta.update_yaxes(categoryorder="total ascending")
                style_fig(fig_ta, height=380, showlegend=False)
                st.plotly_chart(fig_ta, width="stretch", key="pc_topagents")
        with c9:
            with chart_card("slowareas", "🐌", "9 · Top 10 Slowest Areas",
                            "Areas with the highest average delivery time."):
                slow_areas = agg_mean(df, "Area").sort_values(
                    "Avg_Delivery_Time", ascending=False
                ).head(10)
                fig_sa = px.bar(
                    slow_areas, x="Avg_Delivery_Time", y="Area", orientation="h",
                    labels={"Avg_Delivery_Time": "Avg Time (min)", "Area": ""},
                    color="Avg_Delivery_Time", color_continuous_scale=[PRIMARY_3, ACCENT_1],
                    text_auto=".0f",
                )
                fig_sa.update_layout(coloraxis_showscale=False)
                fig_sa.update_yaxes(categoryorder="total ascending")
                style_fig(fig_sa, height=380, showlegend=False)
                st.plotly_chart(fig_sa, width="stretch", key="pc_slowareas")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # 10 ------------------------------------------------------------
        with chart_card("scorecard", "🏆", "10 · Delivery Performance Scorecard",
                        "Consolidated area-level operational scorecard."):
            sc = agg_mean(df, "Area").rename(columns={"Avg_Delivery_Time": "Avg_Time"})
            late_map = agg_late(df, "Area").set_index("Area")["Late_Pct"]
            sc["Late_%"] = sc["Area"].map(late_map).fillna(0)
            rating_map = df.groupby("Area", observed=True)["Agent_Rating"].mean()
            sc["Rating"] = sc["Area"].map(rating_map).fillna(0)

            def _score(row) -> int:
                s = 0
                s += 1 if row["Avg_Time"] <= sc["Avg_Time"].median() else 0
                s += 1 if row["Late_%"] <= sc["Late_%"].median() else 0
                s += 1 if row["Rating"] >= sc["Rating"].median() else 0
                return s

            def _grade(row) -> str:
                return ["badge-bad", "badge-warn", "badge-warn", "badge-good"][_score(row)]

            def _label(row) -> str:
                return ["Needs Attention", "Below Target", "On Track", "Excellent"][_score(row)]

            rows_html = ""
            for _, r in sc.sort_values("Avg_Time").iterrows():
                rows_html += f"""
                    <tr>
                        <td>{r['Area']}</td>
                        <td>{r['Avg_Time']:.1f} min</td>
                        <td>{int(r['Deliveries']):,}</td>
                        <td>{r['Late_%']:.1f}%</td>
                        <td>{r['Rating']:.2f} ★</td>
                        <td><span class="badge {_grade(r)}">{_label(r)}</span></td>
                    </tr>
                """
            st.markdown(
                f"""
                <table class="scorecard">
                    <thead>
                        <tr>
                            <th>Area</th><th>Avg Time</th><th>Deliveries</th>
                            <th>Late %</th><th>Rating</th><th>Status</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

    # ==================================================================
    #  TAB 4 — OPERATIONS (EXPORTS & RAW DATA)
    # ==================================================================
    with tab_ops:
        section_header("📤", "Export Center",
                       "Download the filtered dataset, a PDF report or a dashboard snapshot.")

        insights = build_insights(df)
        e1, e2, e3 = st.columns(3, gap="large")

        with e1:
            st.download_button(
                "⬇️ Download Filtered Data (CSV)",
                data=df_to_csv_bytes(df),
                file_name=f"logisight_filtered_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                width="stretch",
            )
        with e2:
            if st.button("📄 Generate PDF Report", width="stretch"):
                with st.spinner("Building PDF report…"):
                    pdf_bytes = build_pdf_report(df, insights)
                    st.session_state["pdf_bytes"] = pdf_bytes
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    "⬇️ Download Report PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"logisight_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
        with e3:
            if st.button("🖼️ Export Dashboard Snapshot", width="stretch"):
                with st.spinner("Rendering snapshot…"):
                    png = build_snapshot_png(df)
                    st.session_state["png_bytes"] = png
            if st.session_state.get("png_bytes"):
                st.download_button(
                    "⬇️ Download Snapshot (PNG)",
                    data=st.session_state["png_bytes"],
                    file_name=f"logisight_snapshot_{datetime.now():%Y%m%d_%H%M}.png",
                    mime="image/png",
                    width="stretch",
                )
            elif "png_bytes" in st.session_state and st.session_state["png_bytes"] is None:
                st.caption("⚠️ Snapshot rendering failed. Please retry.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        section_header("🗃️", "Filtered Dataset Preview",
                       f"Showing {min(len(df), 1000):,} of {len(df):,} filtered records.")
        st.dataframe(df.head(1000), width="stretch", height=420)

    # ------------------------------------------------------------------
    #  FOOTER
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align:center;color:#94A3B8;font-size:12px;margin-top:30px;">
            🚚 <b>LogiSight Analytics</b> · Last Mile Delivery Intelligence ·
            Built with Streamlit & Plotly · {datetime.now():%Y}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
