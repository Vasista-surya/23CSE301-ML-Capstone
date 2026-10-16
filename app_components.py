"""Reusable visual components for the Machine Learning Capstone dashboard.

The dashboard deliberately keeps data access and model-service orchestration in
``streamlit_app.py``. This module contains presentation helpers only so the visual
language stays consistent across the six dashboard workspaces.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # Plotly is optional at import time so syntax checks work without UI deps.
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - handled by a clear UI message at runtime
    px = None
    go = None

try:
    import streamlit as st
except ImportError:  # pragma: no cover - the runtime dependency is in requirements.txt
    st = None


NAVY = "#101827"
INK = "#172033"
MUTED = "#667085"
BLUE = "#2563EB"
TEAL = "#0F9D8A"
VIOLET = "#7457D9"
AMBER = "#D88A1A"
RED = "#D94F70"
PALETTE = [BLUE, TEAL, VIOLET, AMBER, RED, "#4B8FCB", "#8A6FD1", "#5BAE9E"]


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the dashboard dependencies from "
            "requirements.txt before launching the app."
        )


def inject_styles() -> None:
    """Install the dashboard's small, local design system."""

    _require_streamlit()
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #e7ebf2;
            --surface: #ffffff;
            --canvas: #f6f8fb;
            --navy: #101827;
            --blue: #2563eb;
            --teal: #0f9d8a;
            --violet: #7457d9;
        }

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: var(--navy); border-right: 1px solid #1e293b; }
        [data-testid="stSidebar"] * { color: #dbe5f2; }
        [data-testid="stSidebar"] .stRadio label { padding: 0.32rem 0.45rem; border-radius: 8px; }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1b2940; }
        [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { border-color: #71809b; }
        [data-testid="stSidebar"] [aria-checked="true"] > div:first-child { border-color: #73a7ff; background: #73a7ff; }
        .block-container { max-width: 1480px; padding: 2.2rem 3rem 3.5rem; }
        .brand-mark { display: flex; align-items: center; gap: 0.7rem; margin: 0.35rem 0 2.25rem; }
        .brand-icon { width: 2.25rem; height: 2.25rem; border-radius: 10px; background: #1d4ed8; display: grid; place-items: center; font-weight: 700; color: white; letter-spacing: -0.05em; }
        .brand-name { font-family: 'Space Grotesk', sans-serif; color: white; font-weight: 700; font-size: 1.03rem; line-height: 1.05; }
        .brand-caption { color: #91a2bc; font-size: 0.72rem; margin-top: 0.16rem; }
        .sidebar-rule { height: 1px; background: #26344a; margin: 1.15rem 0; }
        .sidebar-meta { color: #91a2bc; font-size: 0.72rem; line-height: 1.65; }
        .sidebar-status { display: flex; align-items: center; gap: 0.45rem; margin: 0.55rem 0; font-size: 0.76rem; color: #c6d3e4; }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; background: #40c9a2; }
        .status-dot.warn { background: #e9ae4a; }
        .page-kicker { color: var(--blue); font-size: 0.73rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.55rem; }
        .page-title { color: var(--ink); font-size: clamp(2rem, 3.3vw, 3.35rem); line-height: 1.02; letter-spacing: -0.055em; margin: 0; }
        .page-subtitle { color: var(--muted); max-width: 780px; font-size: 1rem; line-height: 1.6; margin: 0.9rem 0 0; }
        .page-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.15rem; }
        .pill { display: inline-flex; align-items: center; gap: 0.35rem; border: 1px solid var(--line); background: #fff; color: #4d5b70; border-radius: 999px; padding: 0.38rem 0.72rem; font-size: 0.75rem; font-weight: 600; }
        .pill.blue { color: #1d4ed8; background: #eff5ff; border-color: #d8e6ff; }
        .pill.teal { color: #087b6b; background: #edfbf7; border-color: #cdeee6; }
        .pill.violet { color: #6048c5; background: #f4f1ff; border-color: #e4ddff; }
        .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 2.2rem 0 0.95rem; }
        .section-heading h2 { font-size: 1.25rem; letter-spacing: -0.025em; margin: 0; }
        .section-heading p { color: var(--muted); font-size: 0.82rem; margin: 0.27rem 0 0; }
        .section-heading .section-note { color: var(--muted); font-size: 0.76rem; text-align: right; }
        .metric-card { position: relative; overflow: hidden; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1.05rem 1.1rem 1rem; min-height: 112px; box-shadow: 0 2px 12px rgba(16, 24, 39, 0.025); }
        .metric-card:before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent, var(--blue)); }
        .metric-label { color: var(--muted); font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.04em; margin-top: 0.5rem; line-height: 1.08; }
        .metric-caption { color: #8b95a5; font-size: 0.72rem; line-height: 1.35; margin-top: 0.42rem; }
        .metric-delta { color: var(--teal); font-size: 0.73rem; font-weight: 700; margin-top: 0.35rem; }
        .track-card { background: var(--surface); border: 1px solid var(--line); border-radius: 15px; padding: 1.25rem; height: 100%; }
        .track-index { color: var(--accent, var(--blue)); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; }
        .track-card h3 { font-size: 1.15rem; margin: 0.65rem 0 0.45rem; }
        .track-card p { color: var(--muted); font-size: 0.82rem; line-height: 1.55; min-height: 3.9rem; }
        .track-stat { display: flex; justify-content: space-between; border-top: 1px solid var(--line); padding-top: 0.75rem; margin-top: 0.9rem; color: #778296; font-size: 0.76rem; }
        .track-stat strong { color: var(--ink); }
        .workflow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; padding: 1.05rem 1.2rem; border: 1px solid var(--line); background: #fff; border-radius: 14px; }
        .workflow-step { display: flex; align-items: center; gap: 0.45rem; color: #45536a; font-size: 0.79rem; font-weight: 600; }
        .workflow-number { display: grid; place-items: center; width: 1.45rem; height: 1.45rem; border-radius: 50%; color: white; background: var(--blue); font-size: 0.68rem; }
        .workflow-arrow { color: #aab4c2; margin: 0 0.15rem; }
        .callout { border: 1px solid #dbe6fb; border-left: 3px solid var(--blue); background: #f7faff; color: #44536b; border-radius: 10px; padding: 0.8rem 0.95rem; font-size: 0.82rem; line-height: 1.55; }
        .callout.teal { border-color: #ccece5; border-left-color: var(--teal); background: #f4fcfa; }
        .callout.warn { border-color: #f0dfba; border-left-color: var(--amber); background: #fffaf1; }
        .empty-state { border: 1px dashed #cbd5e1; border-radius: 14px; padding: 1.6rem; background: rgba(255,255,255,0.75); }
        .empty-state h4 { margin: 0 0 0.35rem; font-size: 1rem; }
        .empty-state p { color: var(--muted); margin: 0; font-size: 0.82rem; line-height: 1.55; }
        .source-note { color: #8b95a5; font-size: 0.72rem; margin-top: 0.35rem; }
        .footer-note { color: #8b95a5; font-size: 0.72rem; text-align: center; padding: 2.5rem 0 0.3rem; }
        div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 0.75rem 0.9rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
         .stButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 600; }
         .stButton button[kind="primary"], .stFormSubmitButton button { background: linear-gradient(135deg, #2563eb, #4f46e5); border: 0; box-shadow: 0 8px 18px rgba(37, 99, 235, .18); }
         .stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover { background: linear-gradient(135deg, #1d4ed8, #4338ca); box-shadow: 0 10px 22px rgba(37, 99, 235, .28); }
         [data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid var(--line); }
         [data-testid="stTabs"] button[role="tab"] { color: var(--muted); font-weight: 600; border-radius: 8px 8px 0 0; padding: .55rem .9rem; }
         [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--blue); background: #eff5ff; }
         [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.68); }
         [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
         @media (max-width: 900px) {
             .block-container { padding: 1.35rem 1rem 2.5rem; }
             .page-title { font-size: 2.25rem; }
             .section-heading { align-items: start; flex-direction: column; }
             .section-heading .section-note { text-align: left; }
         }
         </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    _require_streamlit()
    st.markdown(
        """
        <div class="brand-mark">
            <div class="brand-icon">ML</div>
            <div><div class="brand-name">Capstone Lab</div><div class="brand-caption">23CSE301 · 2026–27</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    kicker: str,
    title: str,
    subtitle: str,
    pills: Sequence[tuple[str, str]] | None = None,
) -> None:
    _require_streamlit()
    st.markdown(f'<div class="page-kicker">{escape(str(kicker))}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{escape(str(title))}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{escape(str(subtitle))}</p>', unsafe_allow_html=True)
    if pills:
        html = '<div class="page-meta">'
        for label, tone in pills:
            html += f'<span class="pill {escape(str(tone))}">{escape(str(label))}</span>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, description: str = "", note: str = "") -> None:
    _require_streamlit()
    note_html = f'<div class="section-note">{escape(str(note))}</div>' if note else ""
    st.markdown(
        f'<div class="section-heading"><div><h2>{escape(str(title))}</h2><p>{escape(str(description))}</p></div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: Any,
    caption: str = "",
    delta: str = "",
    accent: str = BLUE,
) -> None:
    _require_streamlit()
    delta_html = f'<div class="metric-delta">{escape(str(delta))}</div>' if delta else ""
    caption_html = f'<div class="metric-caption">{escape(str(caption))}</div>' if caption else ""
    st.markdown(
        f'<div class="metric-card" style="--accent:{escape(str(accent))}"><div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>{delta_html}{caption_html}</div>',
        unsafe_allow_html=True,
    )


def metric_grid(metrics: Iterable[Mapping[str, Any]], columns: int = 4) -> None:
    _require_streamlit()
    metric_list = list(metrics)
    cols = st.columns(min(columns, max(1, len(metric_list))))
    for col, item in zip(cols, metric_list):
        with col:
            metric_card(
                str(item.get("label", "")),
                item.get("value", "—"),
                str(item.get("caption", "")),
                str(item.get("delta", "")),
                str(item.get("accent", BLUE)),
            )


def render_workflow(steps: Sequence[str]) -> None:
    _require_streamlit()
    parts: list[str] = ['<div class="workflow">']
    for index, step in enumerate(steps, start=1):
        if index > 1:
            parts.append('<span class="workflow-arrow">→</span>')
        parts.append(
            f'<span class="workflow-step"><span class="workflow-number">{index}</span>{step}</span>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def callout(message: str, tone: str = "blue") -> None:
    _require_streamlit()
    # Callers may intentionally pass a small amount of trusted markup (for
    # example, ``<strong>`` around a model name). Keep that existing behaviour
    # while ensuring the reusable component does not construct HTML from raw
    # user input.
    st.markdown(f'<div class="callout {escape(str(tone))}">{message}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str) -> None:
    _require_streamlit()
    st.markdown(
        f'<div class="empty-state"><h4>{escape(str(title))}</h4><p>{escape(str(message))}</p></div>',
        unsafe_allow_html=True,
    )


def source_note(message: str) -> None:
    _require_streamlit()
    st.markdown(f'<div class="source-note">{escape(str(message))}</div>', unsafe_allow_html=True)


def show_table(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    column_config: Mapping[str, Any] | None = None,
) -> None:
    _require_streamlit()
    if frame is None or frame.empty:
        empty_state("No tabular result available", "The selected source did not return any rows.")
        return
    options: dict[str, Any] = {
        "width": "stretch",
        "hide_index": True,
    }
    if height is not None:
        options["height"] = height
    if column_config is not None:
        options["column_config"] = column_config
    st.dataframe(frame, **options)


def _plotly_guard() -> bool:
    if px is None or go is None:
        if st is not None:
            callout("Plotly is not installed, so interactive charts are unavailable. Install requirements.txt to enable them.", "warn")
        return False
    return True


def polish_figure(fig: Any, height: int = 360) -> Any:
    """Apply the shared Plotly canvas, typography, and color defaults."""

    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=36, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color=INK, size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=15, color=INK), x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#172033", font_color="#ffffff"),
    )
    fig.update_xaxes(showline=False, gridcolor="#edf0f5", zeroline=False)
    fig.update_yaxes(showline=False, gridcolor="#edf0f5", zeroline=False)
    return fig


def render_figure(fig: Any, *, height: int = 360, key: str | None = None) -> None:
    _require_streamlit()
    if fig is None:
        empty_state("Chart unavailable", "The chart data was not returned by the selected source.")
        return
    polish_figure(fig, height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True}, key=key)


def comparison_bar(
    frame: pd.DataFrame,
    metric: str,
    *,
    label: str = "Algorithm",
    title: str = "",
    color: str = BLUE,
    height: int = 380,
) -> Any:
    if not _plotly_guard() or frame is None or frame.empty or metric not in frame.columns or label not in frame.columns:
        return None
    plot = frame[[label, metric]].dropna().copy().sort_values(metric, ascending=True)
    fig = px.bar(plot, x=metric, y=label, orientation="h", title=title, text=metric, color_discrete_sequence=[color])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    return polish_figure(fig, height=height)


def metric_line(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str = "",
    markers: bool = True,
    color: str = BLUE,
    height: int = 340,
) -> Any:
    if not _plotly_guard() or frame is None or frame.empty or x not in frame.columns or y not in frame.columns:
        return None
    fig = px.line(frame, x=x, y=y, markers=markers, title=title, color_discrete_sequence=[color])
    return polish_figure(fig, height=height)


def scatter_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    hover_data: Sequence[str] | None = None,
    title: str = "",
    color_sequence: Sequence[str] = PALETTE,
    height: int = 390,
) -> Any:
    if not _plotly_guard() or frame is None or frame.empty or x not in frame.columns or y not in frame.columns:
        return None
    fig = px.scatter(
        frame,
        x=x,
        y=y,
        color=color if color in frame.columns else None,
        hover_data=list(hover_data or []),
        title=title,
        color_discrete_sequence=list(color_sequence),
        opacity=0.78,
    )
    return polish_figure(fig, height=height)


def histogram_chart(
    frame: pd.DataFrame,
    column: str,
    *,
    color: str | None = None,
    title: str = "",
    nbins: int = 28,
    height: int = 340,
) -> Any:
    if not _plotly_guard() or frame is None or frame.empty or column not in frame.columns:
        return None
    fig = px.histogram(
        frame,
        x=column,
        color=color if color in frame.columns else None,
        nbins=nbins,
        title=title,
        color_discrete_sequence=PALETTE,
        marginal="box",
    )
    return polish_figure(fig, height=height)


def heatmap_chart(
    frame: pd.DataFrame,
    *,
    title: str = "",
    height: int = 390,
) -> Any:
    if not _plotly_guard() or frame is None or frame.empty:
        return None
    fig = px.imshow(frame, text_auto=".2f", color_continuous_scale="Blues", title=title, aspect="auto")
    return polish_figure(fig, height=height)


def render_notebook_image(image_bytes: bytes | None, caption: str = "") -> None:
    _require_streamlit()
    if not image_bytes:
        empty_state("Notebook figure unavailable", "This figure was not persisted in the supplied notebook output.")
        return
    st.image(image_bytes, width="stretch", caption=caption or None)


def render_service_status(available: bool, message: str = "") -> None:
    _require_streamlit()
    dot_class = "" if available else "warn"
    label = "ML service connected" if available else "Live ML service not detected"
    detail = message or ("Predictions use the configured service layer." if available else "Results remain read-only until ml_services.py is available.")
    st.markdown(
        f'<div class="sidebar-status"><span class="status-dot {dot_class}"></span>{label}</div>'
        f'<div class="sidebar-meta">{detail}</div>',
        unsafe_allow_html=True,
    )
