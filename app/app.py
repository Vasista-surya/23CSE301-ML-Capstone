from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration and theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone | Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0b1020; color: #f2f5fb; }
    [data-testid="stHeader"] { background: rgba(11,16,32,.88); }
    [data-testid="stSidebar"] { background: #10172a; border-right: 1px solid #24304a; }
    [data-testid="stSidebar"] * { color: #e7edf8; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    h1 { font-weight: 800 !important; }
    .hero { padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, #16213d 0%, #142b3d 52%, #18384a 100%);
        border: 1px solid #2c4865; margin-bottom: 1.25rem; }
    .eyebrow { color: #79e2d0; font-size: .75rem; text-transform: uppercase;
        letter-spacing: .18em; font-weight: 800; }
    .hero-sub { color: #bdcce0; font-size: 1rem; line-height: 1.65; max-width: 940px; }
    div[data-testid="stMetric"] { background: #121b30; border: 1px solid #273652;
        border-radius: 16px; padding: 15px 17px; }
    div[data-testid="stMetricLabel"] { color: #a8b8d1; }
    div[data-testid="stMetricValue"] { color: #f2f6ff; font-family: 'Manrope',sans-serif; }
    div[data-testid="stPlotlyChart"] { background: #10182b; border: 1px solid #202d47;
        border-radius: 15px; padding: 8px; }
    .pill { display:inline-block; padding:4px 9px; border-radius:30px; margin:2px 4px 2px 0;
        color:#9cf0df; background:#12352f; border:1px solid #256358; font-size:.76rem; }
    .footer { color:#8292ac; font-size:.78rem; border-top:1px solid #26334c;
        padding-top:1rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def find_dataset(filename: str) -> Path | None:
    """Locate a CSV in the expected repo layout or common launch directories."""
    candidates = [
        DATA_DIR / filename,
        ROOT / filename,
        Path.cwd() / "datasets" / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string, low_memory=False)


def optional_csv(filename: str) -> pd.DataFrame | None:
    path = find_dataset(filename)
    if path is None:
        return None
    try:
        return read_csv(str(path))
    except Exception as exc:
        st.warning(f"Couldn't load `{filename}`: {exc}")
        return None


def numeric_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return [
        col for col in frame.select_dtypes(include=np.number).columns
