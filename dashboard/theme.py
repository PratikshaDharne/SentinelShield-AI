"""SentinelShield AI visual system: soft light mesh background + glassmorphism cards, deep-navy text, navy/royal blues
with teal, violet, amber, emerald and rose accents. Every chart value is labelled; text contrast stays > 10:1."""
from __future__ import annotations
import html
from pathlib import Path
import plotly.graph_objects as go

ASSETS = Path(__file__).parent / "assets"

NAVY, INK, ROYAL, BLUE, SKY, ICE, PALE, LINE, MUTED = "#0A1F44", "#0B2A5B", "#2447C4", "#3B6FE0", "#5B9BF0", "#A9C6F5", "#EEF3FF", "#C9D6F2", "#2D4272"
TEAL, EMERALD, AMBER, ROSE, VIOLET, CYAN = "#0E9F9A", "#0F9D58", "#F59E0B", "#E11D48", "#7C3AED", "#0891B2"
DEC = {"ALLOW": EMERALD, "REWRITE": AMBER, "BLOCK": ROSE}               # traffic-light semantics
SHADES = [NAVY, TEAL, VIOLET, AMBER, ROSE, ROYAL, CYAN]
FONT = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
BRAND = "SentinelShield AI"


def logo_html(dark_bg: bool = False, width: int = 230) -> str:
    """Text wordmark only (no logo graphic)."""
    c1, c2 = ("#FFFFFF", "#9FE7E2") if dark_bg else (NAVY, TEAL)
    fs = max(18, int(width / 9.2))
    return (f'<div class="wm" style="font-size:{fs}px;color:{c1}">Sentinel<span style="color:{c2}">Shield</span> AI</div>'
            f'<div class="wm-sub" style="color:{"#C7D7F7" if dark_bg else MUTED}">Agentic LLM Guardrail Firewall</div>')


# ----------------------------------------------------------------------------------------------- css
GLASS = "background: rgba(255,255,255,.58); backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%); border: 1px solid rgba(255,255,255,.85);"
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, li, td, th, div {{ font-family: {FONT}; }}
.stApp {{ color: {NAVY}; background:
    radial-gradient(900px 520px at 8% 4%, rgba(124,58,237,.16), transparent 60%),
    radial-gradient(800px 520px at 96% 8%, rgba(14,159,154,.18), transparent 60%),
    radial-gradient(900px 600px at 70% 100%, rgba(59,111,224,.20), transparent 60%),
    radial-gradient(700px 480px at 0% 95%, rgba(245,158,11,.13), transparent 60%),
    linear-gradient(135deg, #F3F6FF 0%, #F7F3FF 50%, #EAF8F6 100%); background-attachment: fixed; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; height: 2.2rem; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3, h4, h5 {{ color: {NAVY} !important; font-weight: 800 !important; letter-spacing: -0.3px; }}
h1 {{ font-size: 2.1rem !important; }} h2 {{ font-size: 1.5rem !important; }} h3 {{ font-size: 1.2rem !important; }}
p, li, .stMarkdown, label, span {{ color: {NAVY}; }}
a {{ color: {ROYAL} !important; font-weight: 600; }}
.wm {{ font-weight: 800; letter-spacing: -0.6px; line-height: 1.1; }}
.wm-sub {{ font-size: 11.5px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-top: 3px; }}
/* ---------- sidebar (navy glass) ---------- */
section[data-testid="stSidebar"] {{ background: linear-gradient(185deg, rgba(8,24,56,.97) 0%, rgba(14,48,112,.95) 55%, rgba(17,78,120,.93) 100%);
    border-right: 1px solid rgba(255,255,255,.25); box-shadow: 6px 0 30px rgba(10,31,68,.25); }}
section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.22); }}
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{ gap: 6px; width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ width: 100%; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{ background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.18);
    border-radius: 12px; padding: 9px 14px; width: 100%; transition: all .18s; cursor: pointer; backdrop-filter: blur(6px); }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,.20); transform: translateX(3px); }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{ background: linear-gradient(135deg, #FFFFFF, #DDF7F4) !important; border-color: #FFFFFF; box-shadow: 0 4px 16px rgba(14,159,154,.45); }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] * {{ color: {NAVY} !important; font-weight: 800 !important; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{ display: none !important; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{ font-size: 15px; font-weight: 600; margin: 0; }}
section[data-testid="stSidebar"] button[data-testid^="stBaseButton"] {{ background: #FFFFFF; color: {NAVY} !important; border: none; font-weight: 800; border-radius: 12px; }}
section[data-testid="stSidebar"] button[data-testid^="stBaseButton"] * {{ color: {NAVY} !important; }}
/* ---------- page header ---------- */
.sg-title {{ display:flex; align-items:center; gap:14px; margin: 0 0 4px 0; }}
.sg-title h1 {{ margin:0; padding:0; background: linear-gradient(90deg, {NAVY} 0%, {ROYAL} 60%, {TEAL} 100%); -webkit-background-clip: text; background-clip: text;
    color: transparent !important; -webkit-text-fill-color: transparent; }}
.sg-sub {{ color: {MUTED}; font-size: 15.5px; font-weight: 500; margin: 0 0 14px 0; }}
.sg-bar {{ height: 4px; width: 100%; border-radius: 4px; background: linear-gradient(90deg, {NAVY}, {ROYAL} 30%, {TEAL} 62%, {AMBER} 84%, {ROSE}); margin: 8px 0 22px 0; opacity:.9; }}
/* ---------- KPI cards (glass) ---------- */
.kpi {{ {GLASS} border-top: 5px solid {ROYAL}; border-radius: 18px; padding: 15px 18px 13px 18px;
    box-shadow: 0 8px 30px rgba(10,31,68,.10), inset 0 1px 0 rgba(255,255,255,.9); min-height: 118px; transition: transform .18s, box-shadow .18s; }}
.kpi:hover {{ transform: translateY(-3px); box-shadow: 0 14px 38px rgba(10,31,68,.17); }}
.kpi .l {{ font-size: 12.5px; font-weight: 800; letter-spacing: 1.1px; text-transform: uppercase; color: {MUTED}; }}
.kpi .v {{ font-size: 27px; white-space: nowrap; font-weight: 800; color: {NAVY}; line-height: 1.15; margin-top: 4px; }}
.kpi .s {{ font-size: 13px; font-weight: 700; color: {TEAL}; margin-top: 2px; }}
.kpi.allow {{ border-top-color: {EMERALD}; }} .kpi.rewrite {{ border-top-color: {AMBER}; }} .kpi.block {{ border-top-color: {ROSE}; }}
.kpi.teal {{ border-top-color: {TEAL}; }} .kpi.violet {{ border-top-color: {VIOLET}; }} .kpi.navy {{ border-top-color: {NAVY}; }}
/* ---------- panels / tables ---------- */
.panel {{ {GLASS} border-left: 5px solid {TEAL}; border-radius:16px; padding:16px 20px; color:{NAVY}; font-size:15px; font-weight:500; box-shadow:0 6px 24px rgba(10,31,68,.08); }}
.panel b, .panel strong {{ color:{NAVY}; }}
.tbl-wrap {{ overflow-x:auto; {GLASS} border-radius:16px; box-shadow:0 6px 24px rgba(10,31,68,.09); }}
table.sg {{ border-collapse: collapse; width:100%; font-size:14px; }}
table.sg th {{ background:linear-gradient(90deg, {NAVY}, #12397F); color:#FFFFFF; font-weight:800; text-align:left; padding:11px 12px; white-space:nowrap; letter-spacing:.3px; }}
table.sg td {{ white-space:nowrap; color:{NAVY}; padding:9px 12px; border-top:1px solid rgba(36,71,196,.14); font-weight:600; vertical-align:top; }}
table.sg tr:nth-child(even) td {{ background:rgba(36,71,196,.05); }}
table.sg tr:hover td {{ background:rgba(14,159,154,.10); }}
table.sg td.w {{ white-space:normal; min-width:260px; }}
.pill {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:800; font-size:12.5px; letter-spacing:.5px; border:1.5px solid; }}
.pill.ALLOW {{ background:#D7F5E4; color:#065F35; border-color:{EMERALD}; }}
.pill.REWRITE {{ background:#FDE9B8; color:#6B3F00; border-color:{AMBER}; }}
.pill.BLOCK {{ background:{ROSE}; color:#fff; border-color:#B3123A; }}
.pill.PENDING {{ background:#EDE7FD; color:#4C1D95; border-color:{VIOLET}; }}
.pill.ok {{ background:#D7F5E4; color:#065F35; border-color:{EMERALD}; }} .pill.bad {{ background:{ROSE}; color:#fff; border-color:#B3123A; }}
.health {{ {GLASS} border-radius:16px; padding:12px 14px; height:100%; box-shadow:0 6px 20px rgba(10,31,68,.08); }}
.health .n {{ font-weight:800; font-size:13px; letter-spacing:.8px; text-transform:uppercase; color:{MUTED}; }}
.health .t {{ font-weight:800; font-size:20px; color:{NAVY}; margin-top:2px; }}
/* ---------- widgets ---------- */
.stMain button[data-testid^="stBaseButton"] {{ background:linear-gradient(135deg, {ROYAL}, {NAVY}); color:#fff; border:none; border-radius:12px; font-weight:800; padding:.55rem 1.1rem; box-shadow:0 6px 18px rgba(36,71,196,.35); transition: all .18s; }}
.stMain button[data-testid^="stBaseButton"]:hover {{ background:linear-gradient(135deg, {TEAL}, {ROYAL}); color:#fff; transform: translateY(-2px); box-shadow:0 10px 24px rgba(14,159,154,.40); }}
.stMain button[data-testid^="stBaseButton"] p {{ color:#fff !important; font-weight:800; }}
div[data-testid="stForm"], .login-card {{ {GLASS} border-top:6px solid {TEAL}; border-radius:24px; padding:26px 30px; box-shadow:0 24px 60px rgba(10,31,68,.20); }}
.stTextInput input, .stTextArea textarea, .stNumberInput input, div[data-baseweb="select"] > div {{ border:1.5px solid {ROYAL} !important; border-radius:12px !important; color:{NAVY} !important; background:rgba(255,255,255,.85) !important; font-weight:600; }}
.stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label, .stSlider label, .stToggle label {{ color:{NAVY} !important; font-weight:800 !important; font-size:14px !important; }}
button[data-baseweb="tab"] {{ font-weight:800; font-size:15px; color:{MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{NAVY}; }}
div[data-baseweb="tab-highlight"] {{ background:linear-gradient(90deg, {TEAL}, {ROYAL}) !important; height:4px !important; border-radius:4px; }}
div[data-testid="stExpander"] {{ {GLASS} border-radius:16px; }} div[data-testid="stExpander"] summary p {{ font-weight:800; color:{NAVY}; }}
div[data-testid="stAlert"] {{ border-radius:14px; border:1.5px solid {TEAL}; background:rgba(255,255,255,.7); }} div[data-testid="stAlert"] p {{ color:{NAVY} !important; font-weight:600; }}
code {{ color:{INK}; background:rgba(36,71,196,.10); padding:2px 6px; border-radius:6px; font-weight:600; }}
div[data-testid="stPlotlyChart"] {{ {GLASS} border-radius:18px; padding:8px; box-shadow:0 8px 28px rgba(10,31,68,.10); }}
.js-plotly-plot .plotly .main-svg {{ border-radius:14px; }}
</style>
"""


def page_header(title: str, subtitle: str = ""):
    import streamlit as st
    st.markdown(f'<div class="sg-title"><h1>{html.escape(title)}</h1></div><div class="sg-sub">{html.escape(subtitle)}</div><div class="sg-bar"></div>', unsafe_allow_html=True)


def kpi(label: str, value, sub: str = "", kind: str = "") -> str:
    return f'<div class="kpi {kind}"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}</div></div>'


def pill(v: str) -> str:
    v = str(v)
    return f'<span class="pill {html.escape(v)}">{html.escape(v)}</span>' if v in ("ALLOW", "REWRITE", "BLOCK", "PENDING") else html.escape(v)


def html_table(columns: list[str], rows: list[list], pills: tuple[str, ...] = ("decision", "Decision", "status", "Status")) -> str:
    th = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for r in rows:
        tds = []
        for c, v in zip(columns, r):
            txt = '' if v is None else str(v)
            tds.append(f"<td{' class=w' if len(txt) > 40 else ''}>{pill(v) if c in pills else html.escape(txt)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl-wrap"><table class="sg"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# ----------------------------------------------------------------------------------------------- plotly
def style(fig: go.Figure, title: str | None = None, height: int = 380, legend: bool = True, legend_y: float = -0.2, bottom: int = 54) -> go.Figure:
    axis = dict(showline=True, linecolor=NAVY, linewidth=2, gridcolor="rgba(36,71,196,.16)", zerolinecolor="rgba(36,71,196,.45)", ticks="outside", tickcolor=NAVY,
                tickfont=dict(size=14, color=NAVY, family=FONT), title_font=dict(size=14, color=NAVY, family=FONT))
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)", height=height,
        font=dict(family=FONT, size=14, color=NAVY),
        title=dict(text=f"<b>{title}</b>" if title else None, x=0.01, xanchor="left", font=dict(size=18, color=NAVY, family=FONT)),
        margin=dict(l=58, r=24, t=64 if title else 24, b=bottom), showlegend=legend,
        legend=dict(font=dict(size=13, color=NAVY), bgcolor="rgba(255,255,255,.75)", bordercolor=LINE, borderwidth=1, orientation="h", y=legend_y, yanchor="top"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=ROYAL, font=dict(size=14, color=NAVY, family=FONT)),
        colorway=SHADES)
    fig.update_xaxes(**axis); fig.update_yaxes(**axis)
    return fig


def show(fig, key: str | None = None):
    import streamlit as st
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def bar_labels(fig, fmt: str = "%{text}", size: int = 14):
    fig.update_traces(textfont=dict(size=size, color=NAVY, family=FONT), textposition="outside", cliponaxis=False, selector=dict(type="bar"))
    return fig
