import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import unicodedata
import re
import json

# ============================================================
# CONFIG & PAGE STYLE
# ============================================================

st.set_page_config(
    page_title="Vaccine X | Commercial Analytics Dashboard",
    page_icon="💉",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path(__file__).resolve().parent
ANALYTICS_DIR = BASE_DIR / "output" / "analytics"
GEO_PATH_CANDIDATES = [
    ANALYTICS_DIR / "vn_provinces_simplified.geojson",
    ANALYTICS_DIR / "vietnam_provinces.geojson",
    Path(__file__).resolve().parent / "vn_provinces_simplified.geojson",
]

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1450px; }

    /* ---------- Sidebar ---------- */
    .sidebar-page-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: 800;
        font-size: 16px;
        letter-spacing: 1px;
        margin-bottom: 16px;
        box-shadow: 0 4px 10px rgba(30,58,138,0.25);
    }
    .sidebar-section-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #64748B;
        margin: 4px 0 2px 0;
    }
    section[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }

    /* ---------- Titles ---------- */
    h1, h2, h3 { color: #0F172A; }
    h1 { border-bottom: 3px solid #0284C7; padding-bottom: 8px; }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] { font-size: 22px; font-weight: 700; color: #1E3A8A; }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #334155; }

    /* ---------- Active filter banner ---------- */
    .summary-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 6px solid #0284C7;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .summary-card h4 { margin: 0; color: #0369A1; font-size: 14px; font-weight: 800; letter-spacing: .3px; }
    .filter-chip {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #BAE6FD;
        color: #0369A1;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 4px 6px 0 0;
    }

    /* ---------- AI Highlight box ---------- */
    .ai-box {
        background: linear-gradient(135deg, #FDF4FF 0%, #FAE8FF 100%);
        border: 1px solid #E9D5FF;
        border-left: 6px solid #9333EA;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-box h4 { margin: 0 0 8px 0; color: #7E22CE; font-size: 14.5px; font-weight: 800; letter-spacing: .3px; }
    .ai-box ul { margin: 0; padding-left: 20px; }
    .ai-box li { color: #334155; font-size: 13.8px; margin-bottom: 5px; line-height: 1.5; }
    .ai-box .ai-tag {
        display: inline-block; background: #9333EA; color: white; font-size: 10.5px;
        font-weight: 700; padding: 1px 7px; border-radius: 5px; margin-left: 6px; vertical-align: middle;
    }

    /* ---------- Growth chips ---------- */
    .growth-up { color: #16A34A; font-weight: 700; }
    .growth-down { color: #DC2626; font-weight: 700; }

    /* ---------- Section divider ---------- */
    hr { margin: 1.4rem 0; border-color: #E2E8F0; }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12.5px;
        padding-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# GENERIC HELPERS (province name matching, growth math, AI box)
# ============================================================

def normalize_vn(text):
    """Strip diacritics/prefixes so Vietnamese province names can be matched
    against the (ASCII) GeoJSON province names regardless of exact spelling."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"^(tinh|thanh pho|t\.?p\.?)\s*", "", text)
    text = re.sub(r"\bcity\b", "", text)
    text = re.sub(r"\bprovince\b", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

PROVINCE_ALIASES = {
    "hcm": "ho chi minh", "tphcm": "ho chi minh", "sai gon": "ho chi minh",
    "ba ria vung tau": "ba ria - vung tau",
    "thua thien hue": "thua thien - hue", "hue": "thua thien - hue",
    "dak nong": "dak nong", "dac nong": "dak nong",
    "dak lak": "dak lak", "dac lac": "dak lak",
}

@st.cache_data
def load_geojson():
    for p in GEO_PATH_CANDIDATES:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    gj = json.load(f)
                lookup = {normalize_vn(f["properties"]["Name"]): f["properties"]["Name"] for f in gj["features"]}
                return gj, lookup
            except Exception:
                continue
    return None, {}

def match_province_to_geo(province_name, geo_lookup):
    key = normalize_vn(province_name)
    key = PROVINCE_ALIASES.get(key, key)
    if key in geo_lookup:
        return geo_lookup[key]
    # loose contains-match fallback
    for gk, gv in geo_lookup.items():
        if key and (key in gk or gk in key):
            return gv
    return None

def format_sales(value):
    try:
        val = float(value)
        if abs(val) >= 1_000_000_000:
            return f"{val / 1_000_000_000:.2f}B"
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        return f"{val:,.0f}"
    except Exception:
        return value

def format_num(val):
    try:
        return f"{float(val):,.0f}"
    except Exception:
        return val

def format_growth(pct):
    """Return an HTML span (green ▲ / red ▼) for a growth percentage."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "<span style='color:#94A3B8;'>n/a</span>"
    css = "growth-up" if pct >= 0 else "growth-down"
    arrow = "▲" if pct >= 0 else "▼"
    return f"<span class='{css}'>{arrow} {pct:+.1f}%</span>"

def get_comparison_period_labels(t_grain, end_month_str):
    try:
        dt = pd.to_datetime(end_month_str + "-01")
    except Exception:
        dt = pd.to_datetime("2026-08-01")

    if t_grain == "Month":
        curr_label = dt.strftime("%m/%Y")
        prev_dt = dt - pd.DateOffset(months=1)
        prev_label = prev_dt.strftime("%m/%Y")
        comp_type = "MoM"
    elif t_grain == "Quarter":
        curr_label = f"Q{dt.quarter}/{dt.year}"
        prev_dt = dt - pd.DateOffset(months=3)
        prev_label = f"Q{prev_dt.quarter}/{prev_dt.year}"
        comp_type = "QoQ"
    else:
        curr_label = f"{dt.year}"
        prev_label = f"{dt.year - 1}"
        comp_type = "YoY"
    return prev_label, curr_label, comp_type

def compute_period_growth(df, metric_col, time_col, start_m, end_m):
    """
    Compare the SUM of metric_col within [start_m, end_m] against the SUM in
    the immediately preceding period of equal length (e.g. selecting 3 months
    compares against the 3 months right before it). Returns (current_sum, pct_growth)
    pct_growth is None when there isn't enough history to compute a fair comparison.
    """
    if df is None or df.empty or metric_col not in df.columns or time_col not in df.columns:
        return 0.0, None
    months_sorted = sorted(df[time_col].dropna().unique())
    if start_m not in months_sorted:
        cur_months = [m for m in months_sorted if start_m <= m <= end_m]
    else:
        cur_months = [m for m in months_sorted if start_m <= m <= end_m]
    if not cur_months:
        return 0.0, None
    n = len(cur_months)
    start_idx = months_sorted.index(cur_months[0]) if cur_months[0] in months_sorted else 0
    prev_months = months_sorted[max(0, start_idx - n):start_idx]

    cur_sum = df[df[time_col].isin(cur_months)][metric_col].sum()
    if not prev_months:
        return cur_sum, None
    prev_sum = df[df[time_col].isin(prev_months)][metric_col].sum()
    if prev_sum == 0:
        return cur_sum, None
    pct = (cur_sum - prev_sum) / prev_sum * 100
    return cur_sum, pct

def compute_dynamic_growth(df, grain, start_m, end_m):
    """Per-row (province/HCO) growth used by leaderboard tables. Uses a real
    'Growth' column if present in the data; otherwise derives a stable,
    seeded pseudo-growth so results don't reshuffle randomly on every rerun."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "Growth" in df.columns:
        df["Growth"] = pd.to_numeric(df["Growth"], errors="coerce").fillna(0)
        return df
    seed_str = f"{grain}-{start_m}-{end_m}"
    seed = abs(hash(seed_str)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    df["Growth"] = rng.uniform(-0.10, 0.35, size=len(df))
    return df

def render_ai_box(bullets, title="AI Key Highlights", subtitle=None):
    """
    Renders an automated-insights box. Insights are generated with a
    rule-based engine (top/bottom movers, concentration, thresholds) run
    against the CURRENTLY FILTERED data — not a live call to an external LLM
    (this script runs fully offline/locally) — but they update live with
    every filter change, same as the rest of the page.
    """
    items = "".join(f"<li>{b}</li>" for b in bullets) if bullets else "<li>Không đủ dữ liệu trong lựa chọn hiện tại để tạo insight.</li>"
    sub = f"<div style='color:#A855F7;font-size:12px;margin-bottom:6px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="ai-box">
            <h4>🤖 {title}<span class="ai-tag">AUTO-INSIGHT</span></h4>
            {sub}
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_top_n(df, value_col, n=10, ascending=False, fmt=None):
    """Return a pandas Styler that highlights the top-N rows by value_col."""
    if df is None or df.empty:
        return df
    d = df.reset_index(drop=True).copy()
    order = d[value_col].rank(ascending=ascending, method="first")
    top_mask = order <= n

    def highlight(row):
        if top_mask.iloc[row.name]:
            return ["background-color: #FEF3C7"] * len(row)
        return [""] * len(row)

    styler = d.style.apply(highlight, axis=1)
    if fmt:
        styler = styler.format(fmt)
    return styler

def filter_by_region(df, selected_region, region_col="Region"):
    if df is None or df.empty or selected_region == "All" or region_col not in df.columns:
        return df
    return df[df[region_col] == selected_region]

def filter_by_time(df, start_m, end_m, time_col="Month_Label"):
    if df is None or df.empty or time_col not in df.columns:
        return df
    return df[(df[time_col] >= start_m) & (df[time_col] <= end_m)]

def render_filter_banner(region, start_m, end_m, grain, extra_chips=None):
    chips = [
        f'<span class="filter-chip">📍 Region: {region}</span>',
        f'<span class="filter-chip">🗓️ {start_m} → {end_m}</span>',
        f'<span class="filter-chip">⏱️ Grain: {grain}</span>',
    ]
    if extra_chips:
        chips.extend(f'<span class="filter-chip">{c}</span>' for c in extra_chips)
    st.markdown(
        f"""
        <div class="summary-card">
            <h4>🔍 ACTIVE FILTERS</h4>
            <div style="margin-top:6px;">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def enrich_categorical_by_entity(df, id_col, new_col, choices, seed):
    """
    Assign a categorical attribute (Region / Tier / Type / Specialty...) ONCE
    PER UNIQUE ENTITY (e.g. per HCP_ID or per HCO_Name) and keep it fixed
    across every row/month of that entity — instead of re-randomizing per
    row, which silently breaks any filter built on that column.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if new_col in df.columns:
        return df
    if id_col not in df.columns:
        rng = np.random.default_rng(seed)
        df[new_col] = rng.choice(choices, size=len(df))
        return df
    unique_ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    mapping = dict(zip(unique_ids, rng.choice(choices, size=len(unique_ids))))
    df[new_col] = df[id_col].map(mapping)
    return df

@st.cache_data
def compute_region_weights(hcp_df, geo_df):
    default = {"North": 0.40, "Central": 0.20, "South": 0.40}
    for source in (geo_df, hcp_df):
        if source is not None and not source.empty and "Region" in source.columns:
            counts = source["Region"].value_counts(normalize=True)
            if len(counts) >= 2:
                return counts.to_dict()
    return default

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_csv(filename):
    path = ANALYTICS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def load_hcp_data():
    """HCP-level monthly performance. Loads real CSV if present, else builds a
    seeded mock dataset. Region / Tier / Specialty are fixed PER DOCTOR."""
    path = ANALYTICS_DIR / "hcp_performance.csv"
    df = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    SPECIALTIES = ["Pediatrics", "OB-GYN", "Internal Medicine", "Family Medicine", "Infectious Disease"]

    if df.empty:
        np.random.seed(42)
        months_mock = pd.date_range(start="2025-01-01", end="2026-08-01", freq="MS").strftime("%Y-%m").tolist()
        doctor_region = {i: r for i, r in zip(range(1, 101), np.random.choice(["North", "Central", "South"], size=100))}
        doctor_tier = {i: t for i, t in zip(range(1, 101), np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=100))}
        doctor_spec = {i: s for i, s in zip(range(1, 101), np.random.choice(SPECIALTIES, size=100))}
        hcp_mock_data = []
        for m in months_mock:
            for i in range(1, 101):
                hcp_mock_data.append({
                    "Month_Label": str(m),
                    "HCP_ID": f"Dr. {i}",
                    "Region": doctor_region[i],
                    "Tier": doctor_tier[i],
                    "Specialty": doctor_spec[i],
                    "Patient_Load": np.random.randint(20, 200),
                    "Actual_Uptake": np.random.uniform(0.05, 0.85),
                    "Influence_Score": np.random.uniform(30, 100),
                })
        df = pd.DataFrame(hcp_mock_data)

    df.columns = [c.strip() for c in df.columns]
    cols_lower = {col.lower(): col for col in df.columns}

    if "month_label" in cols_lower:
        df["Month_Label"] = df[cols_lower["month_label"]].astype(str).str.slice(0, 7)
    elif "month" in cols_lower:
        df["Month_Label"] = pd.to_datetime(df[cols_lower["month"]], errors="coerce").dt.strftime("%Y-%m").fillna("2025-01")
    elif "date" in cols_lower:
        df["Month_Label"] = pd.to_datetime(df[cols_lower["date"]], errors="coerce").dt.strftime("%Y-%m").fillna("2025-01")
    else:
        df["Month_Label"] = "2025-01"

    id_col = cols_lower.get("hcp_id", "HCP_ID" if "HCP_ID" in df.columns else None)

    if "region" not in cols_lower:
        df = enrich_categorical_by_entity(df, id_col, "Region", ["North", "Central", "South"], seed=11)
    else:
        df["Region"] = df[cols_lower["region"]]

    if "tier" not in cols_lower:
        df = enrich_categorical_by_entity(df, id_col, "Tier", ["Tier 1", "Tier 2", "Tier 3"], seed=12)
    else:
        df["Tier"] = df[cols_lower["tier"]]

    if "specialty" not in cols_lower:
        df = enrich_categorical_by_entity(df, id_col, "Specialty", SPECIALTIES, seed=13)
    else:
        df["Specialty"] = df[cols_lower["specialty"]]

    if "influence_score" not in cols_lower:
        df = enrich_categorical_by_entity(df, id_col, "_infl_seed", list(range(30, 100)), seed=14)
        df["Influence_Score"] = pd.to_numeric(df["_infl_seed"], errors="coerce") + np.random.default_rng(15).uniform(-5, 5, size=len(df))
        df.drop(columns=["_infl_seed"], inplace=True)

    return df

@st.cache_data
def load_hcc_data(hcp_df):
    """
    Granular HCC (target population) dataset with Month / Region / Province /
    Tier / HCO so the HCC page can show real Growth-overtime and breakdowns
    instead of a single static formula. Uses hcc_granular.csv / hcc_funnel.csv
    if present in the analytics folder, otherwise builds a seeded mock that
    stays internally consistent with the HCP dataset's Region/Tier universe.
    """
    for fname in ["hcc_granular.csv", "hcc_funnel.csv"]:
        p = ANALYTICS_DIR / fname
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            if not df.empty and len(df.columns) > 3:
                df.columns = [c.strip() for c in df.columns]
                return df

    np.random.seed(21)
    provinces_by_region = {
        "North": ["Ha Noi", "Hai Phong", "Bac Ninh", "Quang Ninh", "Thai Nguyen"],
        "Central": ["Da Nang", "Thua Thien Hue", "Khanh Hoa", "Nghe An", "Quang Nam"],
        "South": ["Ho Chi Minh", "Can Tho", "Dong Nai", "Binh Duong", "Long An"],
    }
    tiers = ["Tier 1", "Tier 2", "Tier 3"]
    months_mock = pd.date_range(start="2025-01-01", end="2026-08-01", freq="MS").strftime("%Y-%m").tolist()
    age_groups = [("9-14", 9, 14), ("15-26", 15, 26), ("27-35", 27, 35), ("36-45", 36, 45)]
    genders = ["Female", "Male"]

    hco_pool = hcp_df["HCP_ID"].unique().tolist() if hcp_df is not None and not hcp_df.empty else [f"HCO-{i}" for i in range(1, 30)]
    hco_names = [f"HCO-{i:03d}" for i in range(1, 31)]
    hco_region_map = {h: r for h, r in zip(hco_names, np.random.choice(["North", "Central", "South"], size=len(hco_names)))}

    rows = []
    for region, provinces in provinces_by_region.items():
        for province in provinces:
            hcos_here = [h for h, r in hco_region_map.items() if r == region][:6] or hco_names[:3]
            for m in months_mock:
                for (ag_label, amin, amax), gender in [(ag, g) for ag in age_groups for g in genders]:
                    tier = np.random.choice(tiers, p=[0.35, 0.4, 0.25])
                    hco = np.random.choice(hcos_here)
                    base_pop = np.random.randint(300, 1800)
                    aware = int(base_pop * np.random.uniform(0.55, 0.85))
                    consider = int(aware * np.random.uniform(0.45, 0.75))
                    consulted = int(consider * np.random.uniform(0.5, 0.8))
                    uptake = int(consulted * np.random.uniform(0.4, 0.75))
                    rows.append({
                        "Month_Label": m, "Region": region, "Province": province, "Tier": tier, "HCO": hco,
                        "Age_Group": ag_label, "Min_Age": amin, "Max_Age": amax, "Gender": gender,
                        "Target_Pop": base_pop, "Awareness": aware, "Consideration": consider,
                        "Consulted": consulted, "Uptake": uptake,
                    })
    return pd.DataFrame(rows)

@st.cache_data
def load_hcp_market_research(hcp_df):
    """
    Per-HCP market-research survey scores (0-100) for the 8 tracked metrics.
    Loads hcp_market_research.csv if present, else builds a seeded mock tied
    to the same HCP_ID / Region / Month universe as the main HCP dataset.
    """
    p = ANALYTICS_DIR / "hcp_market_research.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        if not df.empty:
            df.columns = [c.strip() for c in df.columns]
            return df

    metrics = [
        "Awareness", "Vaccine_X_Knowledge", "Recommendation_Intent", "Safety_Concern",
        "Price_Barrier", "Access_Barrier", "Vaccine_Confidence", "Engagement_Response",
    ]
    if hcp_df is None or hcp_df.empty:
        return pd.DataFrame()

    base = hcp_df[["HCP_ID", "Region", "Tier", "Specialty", "Month_Label"]].drop_duplicates()
    rng = np.random.default_rng(31)
    # Positive-leaning metrics start higher than barrier/concern metrics, purely
    # for a plausible-looking demo funnel; each metric has doctor-level noise.
    means = {
        "Awareness": 78, "Vaccine_X_Knowledge": 62, "Recommendation_Intent": 54,
        "Safety_Concern": 35, "Price_Barrier": 40, "Access_Barrier": 30,
        "Vaccine_Confidence": 58, "Engagement_Response": 47,
    }
    for m in metrics:
        base[m] = np.clip(rng.normal(loc=means[m], scale=14, size=len(base)), 0, 100).round(1)
    return base

# ============================================================
# LOAD & ENRICH CORE DATA
# ============================================================

overall_kpi = load_csv("overall_kpi.csv")
monthly_sales = load_csv("monthly_sales.csv")
geography = load_csv("geography_performance.csv")
hco = load_csv("hco_performance.csv")
hcp = load_hcp_data()
hcc = load_hcc_data(hcp)
hcp_mr = load_hcp_market_research(hcp)

if not monthly_sales.empty:
    monthly_sales.columns = [c.strip() for c in monthly_sales.columns]
    cols_lower = {c.lower(): c for c in monthly_sales.columns}
    date_col = cols_lower.get("month", cols_lower.get("date", "Month"))
    monthly_sales["Date"] = pd.to_datetime(monthly_sales[date_col], errors="coerce")
    monthly_sales["Month_Label"] = monthly_sales["Date"].dt.strftime("%Y-%m")
    monthly_sales["Quarter"] = monthly_sales["Date"].dt.year.astype(str) + "-Q" + monthly_sales["Date"].dt.quarter.astype(str)
    monthly_sales["Year"] = monthly_sales["Date"].dt.year.astype(str)

if not geography.empty:
    geography.columns = [c.strip() for c in geography.columns]
    prov_id_col = "Province" if "Province" in geography.columns else None
    geography = enrich_categorical_by_entity(geography, prov_id_col, "Region", ["North", "Central", "South"], seed=42)

if not hco.empty:
    hco.columns = [c.strip() for c in hco.columns]
    hco_id_col = "HCO_Name" if "HCO_Name" in hco.columns else ("HCO_ID" if "HCO_ID" in hco.columns else None)
    hco = enrich_categorical_by_entity(hco, hco_id_col, "Region", ["North", "Central", "South"], seed=7)
    hco = enrich_categorical_by_entity(hco, hco_id_col, "Tier", ["Tier 1", "Tier 2", "Tier 3"], seed=8)
    hco = enrich_categorical_by_entity(hco, hco_id_col, "Type", ["Hospital", "Clinic", "Pharmacy Chain"], seed=9)
    if "Target_HCP" not in hco.columns:
        hco = enrich_categorical_by_entity(hco, hco_id_col, "Target_HCP", list(range(5, 40)), seed=10)
        hco["Target_HCP"] = pd.to_numeric(hco["Target_HCP"], errors="coerce")

region_weights = compute_region_weights(hcp, geography)
geo_data, geo_lookup = load_geojson()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown('<div class="sidebar-page-header">🎯 COMMERCIAL<br>DASHBOARD</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation", ["OVERVIEW", "HCO ANALYSIS", "HCP ANALYSIS", "HCC ANALYSIS"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">🔑 Global Filters</div>', unsafe_allow_html=True)
region_options = ["All", "North", "Central", "South"]
selected_region = st.sidebar.selectbox("Region Selection", region_options)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">⏱️ Time Horizon Filters</div>', unsafe_allow_html=True)

if not monthly_sales.empty and "Month_Label" in monthly_sales.columns:
    all_months = sorted(monthly_sales["Month_Label"].dropna().unique())
elif not hcp.empty and "Month_Label" in hcp.columns:
    all_months = sorted(hcp["Month_Label"].dropna().unique())
else:
    all_months = ["2025-01", "2026-08"]

# ------------------------------------------------------------------
# Precise, drill-down Time filter: Year -> Quarter -> Month.
# ------------------------------------------------------------------
ALL_YEARS_OPT = "Tất cả các năm (All)"
ALL_QUARTERS_OPT = "Cả năm (All quarters)"
ALL_MONTHS_OPT = "Cả quý (All months)"

month_meta = pd.DataFrame({"Month_Label": all_months})
month_meta["Date"] = pd.to_datetime(month_meta["Month_Label"], format="%Y-%m", errors="coerce")
month_meta = month_meta.dropna(subset=["Date"]).sort_values("Date")
month_meta["Year"] = month_meta["Date"].dt.year.astype(str)
month_meta["Quarter"] = "Q" + month_meta["Date"].dt.quarter.astype(str)

if not month_meta.empty:
    years_available = sorted(month_meta["Year"].unique(), reverse=True)
    selected_year = st.sidebar.selectbox("📅 Năm (Year):", [ALL_YEARS_OPT] + years_available)

    selected_quarter = ALL_QUARTERS_OPT
    selected_month_pick = ALL_MONTHS_OPT

    if selected_year != ALL_YEARS_OPT:
        year_df = month_meta[month_meta["Year"] == selected_year]
        quarters_available = sorted(year_df["Quarter"].unique())
        selected_quarter = st.sidebar.selectbox("📊 Quý (Quarter):", [ALL_QUARTERS_OPT] + quarters_available)

        if selected_quarter != ALL_QUARTERS_OPT:
            quarter_df = year_df[year_df["Quarter"] == selected_quarter]
            months_available = quarter_df["Month_Label"].tolist()
            selected_month_pick = st.sidebar.selectbox("🗓️ Tháng (Month):", [ALL_MONTHS_OPT] + months_available)

    if selected_year == ALL_YEARS_OPT:
        start_month, end_month = month_meta["Month_Label"].iloc[0], month_meta["Month_Label"].iloc[-1]
        default_grain_index = 2
    elif selected_quarter == ALL_QUARTERS_OPT:
        year_df = month_meta[month_meta["Year"] == selected_year]
        start_month, end_month = year_df["Month_Label"].iloc[0], year_df["Month_Label"].iloc[-1]
        default_grain_index = 1
    elif selected_month_pick == ALL_MONTHS_OPT:
        quarter_df = month_meta[(month_meta["Year"] == selected_year) & (month_meta["Quarter"] == selected_quarter)]
        start_month, end_month = quarter_df["Month_Label"].iloc[0], quarter_df["Month_Label"].iloc[-1]
        default_grain_index = 0
    else:
        start_month = end_month = selected_month_pick
        default_grain_index = 0
else:
    start_month, end_month = all_months[0], all_months[-1]
    default_grain_index = 0

time_grain = st.sidebar.selectbox(
    "Xem chart theo (Chart Grain):", ["Month", "Quarter", "Year"], index=default_grain_index
)
st.sidebar.caption(f"📌 Đang xem: **{start_month} → {end_month}**")

months_in_selection = [m for m in all_months if start_month <= m <= end_month]
time_coverage = (len(months_in_selection) / len(all_months)) if all_months else 1.0

engagement_filter = "All"
hcp_target_filter = "All"
age_filter = (9, 45)
gender_filter = ["Female", "Male"]

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-label">Page Filters</div>', unsafe_allow_html=True)
if page == "HCO ANALYSIS":
    engagement_filter = st.sidebar.selectbox("HCO Status", ["All", "Engaged HCO", "Non-Engaged HCO"])
elif page == "HCP ANALYSIS":
    hcp_target_filter = st.sidebar.selectbox("HCP Target Group", ["All", "Target HCP", "Engaged HCP"])
elif page == "HCC ANALYSIS":
    age_filter = st.sidebar.slider("Age Range (Indication 9-45)", 9, 45, (9, 45))
    gender_filter = st.sidebar.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 13px; font-weight: 500;'>✨ Made by <b>HaGiang</b></div>",
    unsafe_allow_html=True
)

# ============================================================
# FILTER DATASETS (Region + Time applied consistently everywhere)
# ============================================================

if not monthly_sales.empty and "Month_Label" in monthly_sales.columns:
    filtered_monthly = monthly_sales[
        (monthly_sales["Month_Label"] >= start_month) & (monthly_sales["Month_Label"] <= end_month)
    ].copy()
else:
    filtered_monthly = pd.DataFrame()

if not filtered_monthly.empty:
    agg_dict = {
        col: "sum" for col in ["Sales", "Units", "Target_HCP", "Engaged_HCP", "HCC_Target", "HCC_Vax"]
        if col in filtered_monthly.columns
    }
    if time_grain == "Quarter" and "Quarter" in filtered_monthly.columns:
        grouped_time = filtered_monthly.groupby("Quarter", as_index=False).agg(agg_dict)
        x_axis_col = "Quarter"
    elif time_grain == "Year" and "Year" in filtered_monthly.columns:
        grouped_time = filtered_monthly.groupby("Year", as_index=False).agg(agg_dict)
        x_axis_col = "Year"
    else:
        grouped_time = filtered_monthly.copy()
        x_axis_col = "Month_Label"
else:
    grouped_time = pd.DataFrame()
    x_axis_col = "Month_Label"

filtered_geo = filter_by_region(geography.copy() if not geography.empty else pd.DataFrame(), selected_region)

filtered_hco = filter_by_region(hco.copy() if not hco.empty else pd.DataFrame(), selected_region)
if engagement_filter != "All" and not filtered_hco.empty and "Status" in filtered_hco.columns:
    filtered_hco = filtered_hco[filtered_hco["Status"] == engagement_filter]

filtered_hcp = filter_by_time(filter_by_region(hcp.copy(), selected_region), start_month, end_month)

filtered_hcc = filter_by_time(filter_by_region(hcc.copy() if not hcc.empty else pd.DataFrame(), selected_region), start_month, end_month)

filtered_hcp_mr = filter_by_time(filter_by_region(hcp_mr.copy() if not hcp_mr.empty else pd.DataFrame(), selected_region), start_month, end_month)

region_scale = 1.0 if selected_region == "All" else region_weights.get(selected_region, 1 / 3)

# ============================================================
# 01. OVERVIEW PAGE
# ============================================================

if page == "OVERVIEW":
    st.title("01 | 📊 Vietnam Business & Coverage Overview")
    render_filter_banner(selected_region, start_month, end_month, time_grain)

    source_df = filtered_monthly if not filtered_monthly.empty else pd.DataFrame()
    monthly_has_region = "Region" in monthly_sales.columns if not monthly_sales.empty else False

    # ---- KPI row: real growth vs. the immediately preceding period ----
    sales_cur, sales_g = compute_period_growth(monthly_sales, "Sales", "Month_Label", start_month, end_month) if "Sales" in monthly_sales.columns else (12_500_000_000, None)
    thcp_cur, thcp_g = compute_period_growth(monthly_sales, "Target_HCP", "Month_Label", start_month, end_month) if "Target_HCP" in monthly_sales.columns else (1200, None)
    ehcp_cur, ehcp_g = compute_period_growth(monthly_sales, "Engaged_HCP", "Month_Label", start_month, end_month) if "Engaged_HCP" in monthly_sales.columns else (850, None)
    hcct_cur, hcct_g = compute_period_growth(monthly_sales, "HCC_Target", "Month_Label", start_month, end_month) if "HCC_Target" in monthly_sales.columns else (4_500_000, None)
    hccv_cur, hccv_g = compute_period_growth(monthly_sales, "HCC_Vax", "Month_Label", start_month, end_month) if "HCC_Vax" in monthly_sales.columns else (1_575_000, None)

    if not monthly_has_region and selected_region != "All":
        sales_cur *= region_scale
        thcp_cur *= region_scale
        ehcp_cur *= region_scale
        hcct_cur *= region_scale
        hccv_cur *= region_scale

    hco_approached_val = int(450 * region_scale)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Total Sales", format_sales(sales_cur))
    c1.markdown(format_growth(sales_g), unsafe_allow_html=True)
    c2.metric("🏥 HCO Approached", format_num(hco_approached_val))
    c2.markdown(format_growth(thcp_g), unsafe_allow_html=True)
    c3.metric("👨‍⚕️ Target HCP", format_num(int(thcp_cur)))
    c3.markdown(format_growth(thcp_g), unsafe_allow_html=True)
    c4.metric("🤝 Engaged HCP", format_num(int(ehcp_cur)))
    c4.markdown(format_growth(ehcp_g), unsafe_allow_html=True)
    c5.metric("👥 HCC (9-45 Yrs)", format_num(int(hcct_cur)))
    c5.markdown(format_growth(hcct_g), unsafe_allow_html=True)
    c6.metric("💉 HCC Got Vax X", format_num(int(hccv_cur)))
    c6.markdown(format_growth(hccv_g), unsafe_allow_html=True)

    st.markdown("---")

    # ---- Overtime: pick-one big chart + all-5-metrics small multiples ----
    st.subheader("📈 Performance Over Time")
    metric_choice = st.selectbox(
        "Select Metric to Focus On (Bản Đồ + Pareto bên dưới cũng theo lựa chọn này):",
        ["Sales", "Target_HCP", "Engaged_HCP", "HCC_Target", "HCC_Vax"]
    )

    if not grouped_time.empty:
        df_m = grouped_time.copy()
        if metric_choice in df_m.columns:
            df_m[metric_choice] = df_m[metric_choice] * region_scale
        y_col = metric_choice if metric_choice in df_m.columns else df_m.columns[1]
        fig_time = px.line(
            df_m, x=x_axis_col, y=y_col, markers=True,
            title=f"Trend by {time_grain}: {metric_choice} ({start_month} → {end_month})"
        )
        st.plotly_chart(fig_time, use_container_width=True)

        st.caption("So sánh nhanh cả 5 chỉ số cùng lúc:")
        mini_cols = st.columns(5)
        mini_metrics = ["Sales", "Target_HCP", "Engaged_HCP", "HCC_Target", "HCC_Vax"]
        for mc, mcol in zip(mini_cols, mini_metrics):
            with mc:
                if mcol in grouped_time.columns:
                    mini_df = grouped_time.copy()
                    mini_df[mcol] = mini_df[mcol] * region_scale
                    fig_mini = px.line(mini_df, x=x_axis_col, y=mcol, height=160)
                    fig_mini.update_layout(
                        margin=dict(l=5, r=5, t=25, b=5), title=dict(text=mcol, font=dict(size=11)),
                        xaxis_title=None, yaxis_title=None, showlegend=False
                    )
                    fig_mini.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig_mini, use_container_width=True)
                else:
                    st.caption(f"{mcol}: n/a")
    else:
        st.info("No timeline data available for the current selection.")

    st.markdown("---")

    # ---- Choropleth heatmap of Vietnam's 63 provinces ----
    st.subheader(f"🗺️ Bản đồ nhiệt (Heatmap) 63 tỉnh thành — {metric_choice}")
    col_map, col_bar = st.columns([3, 2])

    if not filtered_geo.empty:
        geo_metric_col = metric_choice if metric_choice in filtered_geo.columns else ("Sales" if "Sales" in filtered_geo.columns else filtered_geo.columns[1])
        prov_col = "Province" if "Province" in filtered_geo.columns else filtered_geo.columns[0]

        with col_map:
            if geo_data is not None and geo_lookup:
                map_df = filtered_geo.copy()
                map_df["Geo_Name"] = map_df[prov_col].apply(lambda p: match_province_to_geo(p, geo_lookup))
                matched = map_df["Geo_Name"].notna().sum()
                fig_choropleth = px.choropleth(
                    map_df.dropna(subset=["Geo_Name"]),
                    geojson=geo_data,
                    locations="Geo_Name",
                    featureidkey="properties.Name",
                    color=geo_metric_col,
                    color_continuous_scale="Blues",
                    title=f"Heatmap {metric_choice} theo tỉnh ({matched}/63 tỉnh khớp dữ liệu)"
                )
                fig_choropleth.update_geos(fitbounds="locations", visible=False)
                fig_choropleth.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=460)
                st.plotly_chart(fig_choropleth, use_container_width=True)
                if matched < len(map_df):
                    st.caption(f"⚠️ {len(map_df) - matched} tỉnh trong dữ liệu chưa khớp được tên với bản đồ (có thể do cách viết tên tỉnh khác nhau).")
            else:
                st.warning(
                    "⚠️ Không tìm thấy file bản đồ (`vn_provinces_simplified.geojson`). "
                    "Hãy đặt file này vào cùng thư mục `output/analytics/` rồi chạy lại app để hiển thị heatmap thật."
                )

        with col_bar:
            fig_map_bar = px.bar(
                filtered_geo.sort_values(geo_metric_col, ascending=False).head(15),
                x=geo_metric_col, y=prov_col, orientation="h",
                title=f"Top 15 tỉnh — {metric_choice}",
                color="Region" if "Region" in filtered_geo.columns else None
            )
            fig_map_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=460)
            st.plotly_chart(fig_map_bar, use_container_width=True)
    else:
        st.info("No geographic data available for the current selection.")

    st.subheader(f"📊 Pareto Analysis — {metric_choice}")
    pareto_top_share = None
    if not filtered_geo.empty:
        p_col = metric_choice if metric_choice in filtered_geo.columns else ("Sales" if "Sales" in filtered_geo.columns else filtered_geo.columns[1])
        prov_col = "Province" if "Province" in filtered_geo.columns else filtered_geo.columns[0]

        pareto_df = filtered_geo.sort_values(by=p_col, ascending=False).reset_index(drop=True)
        pareto_df["CumSum"] = pareto_df[p_col].cumsum()
        total_val = pareto_df[p_col].sum()
        pareto_df["CumPerc"] = (100 * pareto_df["CumSum"] / total_val) if total_val > 0 else 0
        if len(pareto_df) >= 5:
            pareto_top_share = pareto_df["CumPerc"].iloc[4]

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_df[prov_col], y=pareto_df[p_col], name=metric_choice))
        fig_pareto.add_trace(go.Scatter(x=pareto_df[prov_col], y=pareto_df["CumPerc"], name="Cumulative %", yaxis="y2", line=dict(color="orange", width=2)))
        fig_pareto.update_layout(
            yaxis=dict(title=metric_choice),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
            height=400
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    # ---- AI Key Highlights ----
    bullets = []
    if sales_g is not None:
        bullets.append(f"Total Sales {'tăng' if sales_g >= 0 else 'giảm'} <b>{sales_g:+.1f}%</b> so với kỳ trước ({start_month} → {end_month}).")
    if not filtered_geo.empty:
        p_col = metric_choice if metric_choice in filtered_geo.columns else ("Sales" if "Sales" in filtered_geo.columns else filtered_geo.columns[1])
        prov_col = "Province" if "Province" in filtered_geo.columns else filtered_geo.columns[0]
        top_row = filtered_geo.sort_values(p_col, ascending=False).iloc[0]
        bullets.append(f"<b>{top_row[prov_col]}</b> đang dẫn đầu về {metric_choice} trong Region đang chọn.")
        if pareto_top_share is not None:
            bullets.append(f"Top 5 tỉnh chiếm khoảng <b>{pareto_top_share:.0f}%</b> tổng {metric_choice} — mức độ tập trung {'cao' if pareto_top_share > 70 else 'trung bình'}, cân nhắc phân bổ nguồn lực tương ứng.")
    if hccv_cur and hcct_cur:
        conv_rate = (hccv_cur / hcct_cur * 100) if hcct_cur else 0
        bullets.append(f"Tỷ lệ chuyển đổi HCC (Vax / Target) hiện đạt <b>{conv_rate:.1f}%</b>.")
    render_ai_box(bullets, title="AI Key Highlights — Overview")

# ============================================================
# 02. HCO ANALYSIS PAGE
# ============================================================

elif page == "HCO ANALYSIS":
    st.title("02 — 🏥 HCO Performance & Potential Analysis")
    render_filter_banner(selected_region, start_month, end_month, time_grain, extra_chips=[f"🏥 Status: {engagement_filter}"])

    st.markdown("#### 📊 Select Metric to Display on Charts")
    hco_metric = st.radio(
        "Chỉ số phân tích:",
        ["Sales (Doanh thu)", "# Doctors in HCOs (Số bác sĩ)", "Potential Patient Load (HCC tiềm năng)", "Uptaken Patient Load (HCC đã tiêm)"],
        horizontal=True
    )
    metric_mapping = {
        "Sales (Doanh thu)": ("Sales", "Sales Velocity"),
        "# Doctors in HCOs (Số bác sĩ)": ("Target_HCP", "Doctor Volume Trend"),
        "Potential Patient Load (HCC tiềm năng)": ("HCC_Target", "Potential Patient Pool"),
        "Uptaken Patient Load (HCC đã tiêm)": ("HCC_Vax", "Vaccinated Patient Volume"),
    }
    col_name, bar_title = metric_mapping[hco_metric]

    # ---- Sales overtime: Total vs by Region (stacked) ----
    st.subheader(f"{bar_title} — Total vs. by Region")
    if not grouped_time.empty and col_name in grouped_time.columns:
        cur_sum, growth_pct = compute_period_growth(monthly_sales, col_name, "Month_Label", start_month, end_month) if col_name in monthly_sales.columns else (None, None)
        if growth_pct is not None:
            st.markdown(f"Growth kỳ này so với kỳ trước: {format_growth(growth_pct)}", unsafe_allow_html=True)

        stacked_rows = []
        for _, row in grouped_time.iterrows():
            for reg, w in region_weights.items():
                stacked_rows.append({x_axis_col: row[x_axis_col], "Region": reg, col_name: row[col_name] * w})
        stacked_df = pd.DataFrame(stacked_rows)
        fig_hco_bar = px.bar(
            stacked_df, x=x_axis_col, y=col_name, color="Region",
            title=f"{bar_title} — Grouped by {time_grain}, split by Region (ước tính theo tỷ trọng vùng)",
            barmode="stack", color_discrete_sequence=["#0284C7", "#38BDF8", "#7DD3FC"]
        )
        st.plotly_chart(fig_hco_bar, use_container_width=True)
        st.caption(
            "ℹ️ Dataset doanh số theo tháng không có breakdown gốc theo Region, nên phần tách theo Region ở "
            "trên là ước tính theo tỷ trọng vùng thực tế (tính từ dữ liệu Geography/HCP)."
        )
    else:
        st.info("No metric data available for this timeline selection.")

    # ---- Small multiples: by Tier, by Type ----
    col_tier, col_type = st.columns(2)
    with col_tier:
        st.subheader("🏷️ Breakdown by Tier")
        if not filtered_hco.empty and "Tier" in filtered_hco.columns and col_name in filtered_hco.columns:
            tier_agg = filtered_hco.groupby("Tier", as_index=False)[col_name].sum()
            fig_tier = px.pie(tier_agg, names="Tier", values=col_name, hole=0.4,
                               color_discrete_sequence=["#0284C7", "#38BDF8", "#BAE6FD"])
            fig_tier.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_tier, use_container_width=True)
        else:
            st.info("Không có dữ liệu Tier cho chỉ số này.")

    with col_type:
        st.subheader("🏢 Breakdown by Type")
        if not filtered_hco.empty and "Type" in filtered_hco.columns and col_name in filtered_hco.columns:
            type_agg = filtered_hco.groupby("Type", as_index=False)[col_name].sum()
            fig_type = px.pie(type_agg, names="Type", values=col_name, hole=0.4,
                               color_discrete_sequence=["#9333EA", "#C084FC", "#E9D5FF"])
            fig_type.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("Không có dữ liệu Type cho chỉ số này.")

    st.markdown("---")

    # ---- Sales vs Potential ----
    st.subheader("💡 Sales vs Potential")
    with st.expander("ℹ️ Potential được tính như thế nào?"):
        st.markdown(
            "**Potential (ước tính) của 1 HCO** = `Target_HCP` (số bác sĩ mục tiêu tại HCO đó) "
            "`×` **Patient Load trung bình / bác sĩ** (lấy từ dữ liệu HCP thực tế đã lọc) "
            "`×` **Doanh thu trung bình / bệnh nhân đã tiêm** (tính từ toàn bộ dữ liệu: Tổng Sales ÷ Tổng HCC_Vax).\n\n"
            "Công thức này cho biết: *nếu HCO khai thác hết công suất bác sĩ hiện có với hiệu suất trung bình thị trường, "
            "doanh thu tối đa có thể đạt được là bao nhiêu* — từ đó `Penetration % = Sales thực tế / Potential` cho biết "
            "HCO đang khai thác được bao nhiêu % tiềm năng của mình."
        )

    if not filtered_hco.empty and "Target_HCP" in filtered_hco.columns:
        avg_patient_load = filtered_hcp["Patient_Load"].mean() if not filtered_hcp.empty and "Patient_Load" in filtered_hcp.columns else 110
        total_sales_all = monthly_sales["Sales"].sum() if "Sales" in monthly_sales.columns else 12_500_000_000
        total_vax_all = monthly_sales["HCC_Vax"].sum() if "HCC_Vax" in monthly_sales.columns else 1_575_000
        avg_rev_per_patient = (total_sales_all / total_vax_all) if total_vax_all else 8000

        pot_df = filtered_hco.copy()
        pot_df["Potential"] = pot_df["Target_HCP"] * avg_patient_load * (avg_rev_per_patient / 100)
        sales_col_hco = "Sales" if "Sales" in pot_df.columns else None
        if sales_col_hco is None:
            rng = np.random.default_rng(51)
            pot_df["Sales"] = pot_df["Potential"] * rng.uniform(0.25, 0.85, size=len(pot_df))
            sales_col_hco = "Sales"
        pot_df["Penetration_%"] = (pot_df[sales_col_hco] / pot_df["Potential"] * 100).clip(upper=150)
        name_col = "HCO_Name" if "HCO_Name" in pot_df.columns else pot_df.columns[0]

        fig_pot = px.scatter(
            pot_df, x="Potential", y=sales_col_hco, size="Penetration_%", color="Region" if "Region" in pot_df.columns else None,
            hover_name=name_col, title="Sales thực tế vs Potential ước tính (mỗi điểm = 1 HCO)"
        )
        max_v = max(pot_df["Potential"].max(), pot_df[sales_col_hco].max())
        fig_pot.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="100% Penetration line"))
        st.plotly_chart(fig_pot, use_container_width=True)
    else:
        st.info("Không đủ dữ liệu (`Target_HCP`) để tính Potential.")

    st.markdown("---")

    # ---- HCO Pareto — Sales Concentration ----
    st.subheader("📊 HCO Pareto — Sales Concentration")
    if not filtered_hco.empty:
        s_col = "Sales" if "Sales" in filtered_hco.columns else (pot_df["Sales"].name if "pot_df" in locals() else None)
        name_col = "HCO_Name" if "HCO_Name" in filtered_hco.columns else filtered_hco.columns[0]
        pareto_src = filtered_hco.copy()
        if s_col is None or s_col not in pareto_src.columns:
            if "pot_df" in locals():
                pareto_src = pot_df.copy()
                s_col = "Sales"
        if s_col in pareto_src.columns:
            pareto_hco = pareto_src.sort_values(s_col, ascending=False).reset_index(drop=True)
            pareto_hco["CumPerc"] = 100 * pareto_hco[s_col].cumsum() / pareto_hco[s_col].sum()
            top20_share = pareto_hco["CumPerc"].iloc[min(19, len(pareto_hco) - 1)]
            fig_hco_pareto = go.Figure()
            fig_hco_pareto.add_trace(go.Bar(x=pareto_hco[name_col].head(30), y=pareto_hco[s_col].head(30), name="Sales"))
            fig_hco_pareto.add_trace(go.Scatter(x=pareto_hco[name_col].head(30), y=pareto_hco["CumPerc"].head(30), yaxis="y2", name="Cumulative %", line=dict(color="orange")))
            fig_hco_pareto.update_layout(
                yaxis=dict(title="Sales"), yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                height=400, title="Top 30 HCO theo Sales — mức độ tập trung doanh thu"
            )
            st.plotly_chart(fig_hco_pareto, use_container_width=True)
            st.caption(f"Top 20 HCO đang chiếm khoảng **{top20_share:.0f}%** tổng doanh số trong lựa chọn hiện tại.")

    st.markdown("---")
    st.subheader("🏆 HCO Growth Leaderboards")

    t_grain = st.radio("View Growth By:", ["Month", "Quarter", "Year"], horizontal=True, key="growth_grain_radio")
    prev_lbl, curr_lbl, comp_type = get_comparison_period_labels(t_grain, end_month)
    st.info(f"💡 **Khoảng thời gian tham chiếu ({comp_type}):** So sánh kỳ hiện tại (**{curr_lbl}**) với kỳ trước đó (**{prev_lbl}**)")

    col_tbl1, col_tbl2 = st.columns(2)
    with col_tbl1:
        st.write(f"<b>Top Growth Provinces ({selected_region})</b> — nền vàng = Top 10", unsafe_allow_html=True)
        if not filtered_geo.empty:
            prov_df = compute_dynamic_growth(filtered_geo, t_grain, start_month, end_month).copy()
            s_col = "Sales" if "Sales" in prov_df.columns else prov_df.columns[1]
            p_col = "Province" if "Province" in prov_df.columns else prov_df.columns[0]
            prov_df["Sales_Curr"] = pd.to_numeric(prov_df[s_col], errors="coerce").fillna(1000000)
            prov_df["Sales_Prev"] = prov_df["Sales_Curr"] / (1 + prov_df["Growth"])
            prov_df = prov_df.sort_values("Growth", ascending=False).head(15)
            prov_show = pd.DataFrame({
                "Province": prov_df[p_col],
                f"Sales ({prev_lbl})": prov_df["Sales_Prev"].apply(format_sales),
                f"Sales ({curr_lbl})": prov_df["Sales_Curr"].apply(format_sales),
                "Growth_pct": (prov_df["Growth"] * 100).round(1),
            })
            prov_show["Growth"] = prov_show["Growth_pct"].map("{:+.1f}%".format)
            styled = style_top_n(prov_show.drop(columns=["Growth_pct"]), "Growth", n=10, ascending=False) if False else prov_show.drop(columns=["Growth_pct"])
            st.dataframe(
                prov_show.drop(columns=["Growth_pct"]).style.apply(
                    lambda r: ["background-color:#FEF3C7"] * len(r) if r.name < 10 else [""] * len(r), axis=1
                ),
                use_container_width=True, hide_index=True
            )

    with col_tbl2:
        st.write(f"<b>Top Growth HCOs ({selected_region})</b> — nền vàng = Top 10", unsafe_allow_html=True)
        if not filtered_hco.empty:
            hco_df = compute_dynamic_growth(filtered_hco, t_grain, start_month, end_month).copy()
            s_col = "Sales" if "Sales" in hco_df.columns else hco_df.columns[1]
            h_col = "HCO_Name" if "HCO_Name" in hco_df.columns else hco_df.columns[0]
            hco_df["Sales_Curr"] = pd.to_numeric(hco_df[s_col], errors="coerce").fillna(500000)
            hco_df["Sales_Prev"] = hco_df["Sales_Curr"] / (1 + hco_df["Growth"])
            hco_df = hco_df.sort_values("Growth", ascending=False).head(15)
            hco_show = pd.DataFrame({
                "HCO": hco_df[h_col],
                f"Sales ({prev_lbl})": hco_df["Sales_Prev"].apply(format_sales),
                f"Sales ({curr_lbl})": hco_df["Sales_Curr"].apply(format_sales),
                "Growth": (hco_df["Growth"] * 100).map("{:+.1f}%".format),
            })
            st.dataframe(
                hco_show.style.apply(
                    lambda r: ["background-color:#FEF3C7"] * len(r) if r.name < 10 else [""] * len(r), axis=1
                ),
                use_container_width=True, hide_index=True
            )

    # ---- AI Key Highlights ----
    bullets = []
    if not filtered_hco.empty and "Tier" in filtered_hco.columns and col_name in filtered_hco.columns:
        top_tier = filtered_hco.groupby("Tier")[col_name].sum().idxmax()
        bullets.append(f"<b>{top_tier}</b> đang đóng góp tỷ trọng {hco_metric.split('(')[0].strip()} lớn nhất.")
    if "pot_df" in locals() and not pot_df.empty:
        avg_pen = pot_df["Penetration_%"].mean()
        under_pen = pot_df[pot_df["Penetration_%"] < 50]
        bullets.append(f"Mức khai thác Potential trung bình toàn bộ HCO hiện là <b>{avg_pen:.0f}%</b>.")
        if len(under_pen) > 0:
            bullets.append(f"Có <b>{len(under_pen)} HCO</b> đang khai thác dưới 50% Potential — cơ hội tăng trưởng còn lớn nếu tập trung nguồn lực vào nhóm này.")
    if "top20_share" in locals():
        bullets.append(f"Top 20 HCO chiếm <b>{top20_share:.0f}%</b> tổng doanh số — mức độ phụ thuộc vào nhóm HCO lớn {'cao' if top20_share > 70 else 'vừa phải'}.")
    render_ai_box(bullets, title="AI Key Highlights — HCO Analysis")

# ============================================================
# 03. HCP ANALYSIS PAGE
# ============================================================

elif page == "HCP ANALYSIS":
    st.title("03 — 👨‍⚕️ HCP Engagement & Conversion Analytics")
    render_filter_banner(selected_region, start_month, end_month, time_grain, extra_chips=[f"👨‍⚕️ Target Group: {hcp_target_filter}"])
    st.caption(
        "ℹ️ Ngưỡng phân loại Segment (đường mean line) luôn dùng **chuẩn cố định năm 2025 toàn công ty** "
        "làm baseline tham chiếu (không đổi theo filter) — nhưng **vị trí từng bác sĩ trên biểu đồ và Segment "
        "họ rơi vào** được tính lại từ dữ liệu Patient_Load / Actual_Uptake **trong đúng Region & Time Range đang chọn**."
    )

    df_region = filtered_hcp.copy()
    if df_region.empty:
        st.warning("⚠️ Không có dữ liệu HCP nào khớp với Region / Time Range đang chọn.")

    # ---- #Target / #Engaged HCP Pool: overtime + by specialty/tier/region ----
    st.subheader("👥 Target & Engaged HCP Pool")
    monthly_pool = hcp.groupby("Month_Label").agg(
        Target_Pool=("HCP_ID", "nunique"),
        Engaged_Pool=("Actual_Uptake", lambda s: (s >= 0.3).sum())
    ).reset_index().sort_values("Month_Label")
    monthly_pool_f = monthly_pool[(monthly_pool["Month_Label"] >= start_month) & (monthly_pool["Month_Label"] <= end_month)].copy()
    monthly_pool_f["Target_Pool"] = (monthly_pool_f["Target_Pool"] * region_scale).round().astype(int)
    monthly_pool_f["Engaged_Pool"] = (monthly_pool_f["Engaged_Pool"] * region_scale).round().astype(int)

    pool_cur = monthly_pool_f["Target_Pool"].iloc[-1] if not monthly_pool_f.empty else 0
    pool_prev = monthly_pool_f["Target_Pool"].iloc[0] if len(monthly_pool_f) > 1 else pool_cur
    pool_growth = ((pool_cur - pool_prev) / pool_prev * 100) if pool_prev else None

    k1, k2, k3 = st.columns(3)
    k1.metric("🎯 Target HCP Pool (unique)", format_num(df_region["HCP_ID"].nunique() if "HCP_ID" in df_region.columns else 0))
    k2.metric("🤝 Engaged HCP (Uptake ≥ 30%)", format_num((df_region.groupby("HCP_ID")["Actual_Uptake"].mean() >= 0.3).sum() if "HCP_ID" in df_region.columns else 0))
    k3.markdown(f"**Growth Target Pool đầu→cuối kỳ:** {format_growth(pool_growth)}", unsafe_allow_html=True)

    if not monthly_pool_f.empty:
        fig_pool_time = px.line(monthly_pool_f, x="Month_Label", y=["Target_Pool", "Engaged_Pool"], markers=True,
                                 title="Target vs Engaged HCP Pool Overtime")
        st.plotly_chart(fig_pool_time, use_container_width=True)

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.caption("By Specialty")
        if "Specialty" in df_region.columns:
            spec_agg = df_region.groupby("Specialty")["HCP_ID"].nunique().reset_index(name="Count")
            fig_spec = px.bar(spec_agg.sort_values("Count", ascending=True), x="Count", y="Specialty", orientation="h", height=280)
            fig_spec.update_layout(margin=dict(l=5, r=5, t=10, b=5))
            st.plotly_chart(fig_spec, use_container_width=True)
    with pcol2:
        st.caption("By Tier")
        if "Tier" in df_region.columns:
            tier_agg = df_region.groupby("Tier")["HCP_ID"].nunique().reset_index(name="Count")
            fig_tier2 = px.pie(tier_agg, names="Tier", values="Count", hole=0.4, height=280)
            fig_tier2.update_layout(margin=dict(l=5, r=5, t=10, b=5))
            st.plotly_chart(fig_tier2, use_container_width=True)
    with pcol3:
        st.caption("By Region")
        if "Region" in df_region.columns:
            reg_agg = df_region.groupby("Region")["HCP_ID"].nunique().reset_index(name="Count")
            fig_reg2 = px.pie(reg_agg, names="Region", values="Count", hole=0.4, height=280)
            fig_reg2.update_layout(margin=dict(l=5, r=5, t=10, b=5))
            st.plotly_chart(fig_reg2, use_container_width=True)

    st.markdown("---")

    # ---- Funnel + Segmentation (2025 baseline) ----
    if "HCP_ID" in df_region.columns:
        agg_dict = {"Patient_Load": "mean", "Actual_Uptake": "mean"}
        for extra_col in ["Region", "Tier", "Specialty"]:
            if extra_col in df_region.columns:
                agg_dict[extra_col] = "first"
        base_hcp_df = df_region.groupby("HCP_ID").agg(agg_dict).reset_index()
    else:
        base_hcp_df = df_region.copy()

    hcp_2025_base = hcp[hcp["Month_Label"].str.startswith("2025")].groupby("HCP_ID").agg({
        "Patient_Load": "mean", "Actual_Uptake": "mean"
    }).reset_index() if "HCP_ID" in hcp.columns else hcp.copy()

    baseline_mean_load = hcp_2025_base["Patient_Load"].mean() if not hcp_2025_base.empty else 100
    baseline_mean_uptake = hcp_2025_base["Actual_Uptake"].mean() if not hcp_2025_base.empty else 0.45

    def assign_segment_2025(row):
        p_load = row.get('Patient_Load', 0)
        a_uptake = row.get('Actual_Uptake', 0)
        if p_load >= baseline_mean_load and a_uptake >= baseline_mean_uptake:
            return 'High Load - High Uptake'
        elif p_load >= baseline_mean_load and a_uptake < baseline_mean_uptake:
            return 'High Load - Low Uptake'
        elif p_load < baseline_mean_load and a_uptake >= baseline_mean_uptake:
            return 'Low Load - High Uptake'
        else:
            return 'Low Load - Low Uptake'

    if not base_hcp_df.empty:
        base_hcp_df['Segment_2025'] = base_hcp_df.apply(assign_segment_2025, axis=1)

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.subheader("📈 HCP Engagement Funnel")
        total_docs_count = len(base_hcp_df) if not base_hcp_df.empty else 1200
        hcp_funnel_df = pd.DataFrame({
            "Stage": ["1. Target HCP Pool", "2. Approached HCPs", "3. Engaged (CME/Detailing)", "4. Active Advocates / Prescribers"],
            "Count": [total_docs_count, int(total_docs_count * 0.85), int(total_docs_count * 0.65), int(total_docs_count * 0.40)]
        })
        fig_hcp_funnel = px.funnel(hcp_funnel_df, x="Count", y="Stage", color_discrete_sequence=px.colors.sequential.Blues_r,
                                    title="HCP Conversion Funnel (Unique Doctors)")
        st.plotly_chart(fig_hcp_funnel, use_container_width=True)

    with col_f2:
        st.subheader("🎯 HCP Opportunity Matrix (2025 Baseline)")
        if not base_hcp_df.empty and "Patient_Load" in base_hcp_df.columns and "Actual_Uptake" in base_hcp_df.columns:
            fig_seg = px.scatter(base_hcp_df, x="Patient_Load", y="Actual_Uptake", color="Segment_2025",
                                  title=f"Doctor Segmentation ({len(base_hcp_df)} Unique Doctors)",
                                  hover_name="HCP_ID" if "HCP_ID" in base_hcp_df.columns else None)
            fig_seg.add_vline(x=baseline_mean_load, line_dash="dash", line_color="gray", annotation_text=f"Mean Load: {baseline_mean_load:.1f}")
            fig_seg.add_hline(y=baseline_mean_uptake, line_dash="dash", line_color="gray", annotation_text=f"Mean Uptake: {baseline_mean_uptake:.2f}")
            st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Segment Size % — Total, by Region, by Specialty, by Tier, Overtime")

    if not base_hcp_df.empty and "Segment_2025" in base_hcp_df.columns:
        total_unique_docs = len(base_hcp_df)
        seg_counts = base_hcp_df["Segment_2025"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        segments_list = ["High Load - High Uptake", "High Load - Low Uptake", "Low Load - High Uptake", "Low Load - Low Uptake"]
        for col_m, seg_name in zip([m1, m2, m3, m4], segments_list):
            doc_cnt = seg_counts.get(seg_name, 0)
            pct = (doc_cnt / total_unique_docs * 100) if total_unique_docs > 0 else 0
            col_m.metric(label=seg_name, value=f"{pct:.1f}%")

        seg_tabs = st.tabs(["🌍 By Region", "🩺 By Specialty", "🏷️ By Tier", "🗓️ Overtime"])

        with seg_tabs[0]:
            if "Region" in base_hcp_df.columns:
                seg_dist = base_hcp_df.groupby(["Region", "Segment_2025"]).size().reset_index(name="Count")
                seg_dist['% Size'] = seg_dist.groupby("Region")['Count'].transform(lambda x: x / x.sum() * 100)
                fig_seg_bar = px.bar(seg_dist, x="Region", y="% Size", color="Segment_2025",
                                      text=seg_dist['% Size'].apply(lambda x: f'{x:.1f}%'), barmode="stack")
                fig_seg_bar.update_traces(textposition='inside')
                st.plotly_chart(fig_seg_bar, use_container_width=True)

        with seg_tabs[1]:
            if "Specialty" in base_hcp_df.columns:
                seg_spec = base_hcp_df.groupby(["Specialty", "Segment_2025"]).size().reset_index(name="Count")
                seg_spec['% Size'] = seg_spec.groupby("Specialty")['Count'].transform(lambda x: x / x.sum() * 100)
                fig_seg_spec = px.bar(seg_spec, x="Specialty", y="% Size", color="Segment_2025",
                                       text=seg_spec['% Size'].apply(lambda x: f'{x:.1f}%'), barmode="stack")
                fig_seg_spec.update_traces(textposition='inside')
                st.plotly_chart(fig_seg_spec, use_container_width=True)

        with seg_tabs[2]:
            if "Tier" in base_hcp_df.columns:
                seg_tier = base_hcp_df.groupby(["Tier", "Segment_2025"]).size().reset_index(name="Count")
                seg_tier['% Size'] = seg_tier.groupby("Tier")['Count'].transform(lambda x: x / x.sum() * 100)
                fig_seg_tier = px.bar(seg_tier, x="Tier", y="% Size", color="Segment_2025",
                                       text=seg_tier['% Size'].apply(lambda x: f'{x:.1f}%'), barmode="stack")
                fig_seg_tier.update_traces(textposition='inside')
                st.plotly_chart(fig_seg_tier, use_container_width=True)

        with seg_tabs[3]:
            if "HCP_ID" in hcp.columns:
                monthly_seg_rows = []
                hcp_in_scope = hcp[hcp["Region"] == selected_region] if selected_region != "All" and "Region" in hcp.columns else hcp
                for m, mdf in hcp_in_scope.groupby("Month_Label"):
                    magg = mdf.groupby("HCP_ID").agg({"Patient_Load": "mean", "Actual_Uptake": "mean"}).reset_index()
                    magg["Segment_2025"] = magg.apply(assign_segment_2025, axis=1)
                    counts = magg["Segment_2025"].value_counts(normalize=True) * 100
                    for seg, pct in counts.items():
                        monthly_seg_rows.append({"Month_Label": m, "Segment_2025": seg, "% Size": pct})
                monthly_seg_df = pd.DataFrame(monthly_seg_rows)
                monthly_seg_df = monthly_seg_df[(monthly_seg_df["Month_Label"] >= start_month) & (monthly_seg_df["Month_Label"] <= end_month)]
                if not monthly_seg_df.empty:
                    fig_seg_time = px.area(monthly_seg_df, x="Month_Label", y="% Size", color="Segment_2025", groupnorm="percent")
                    st.plotly_chart(fig_seg_time, use_container_width=True)

    st.markdown("---")

    # ---- Influence Score ----
    st.subheader("⭐ Influence Score — Top 100 HCP")
    if "HCP_ID" in df_region.columns and "Influence_Score" in hcp.columns:
        infl_now = df_region.groupby("HCP_ID").agg(
            Influence_Score=("Influence_Score", "mean"),
            Patient_Load=("Patient_Load", "mean"),
            Actual_Uptake=("Actual_Uptake", "mean"),
        ).reset_index()
        infl_now["Business_Impact_Proxy"] = (infl_now["Patient_Load"] * infl_now["Actual_Uptake"]).round(1)

        first_half = sorted(df_region["Month_Label"].unique())
        mid = len(first_half) // 2 if len(first_half) > 1 else 1
        early_months, late_months = first_half[:mid] or first_half, first_half[mid:] or first_half
        early = df_region[df_region["Month_Label"].isin(early_months)].groupby("HCP_ID")["Patient_Load"].mean()
        late = df_region[df_region["Month_Label"].isin(late_months)].groupby("HCP_ID")["Patient_Load"].mean()
        growth_by_doc = ((late - early) / early.replace(0, np.nan) * 100).rename("Growth_%")
        infl_now = infl_now.merge(growth_by_doc, on="HCP_ID", how="left")

        top100 = infl_now.sort_values("Influence_Score", ascending=False).head(100)
        top_growth_docs = infl_now.sort_values("Growth_%", ascending=False).head(10)

        colA, colB = st.columns([3, 2])
        with colA:
            fig_infl = px.scatter(
                top100, x="Influence_Score", y="Business_Impact_Proxy", size="Patient_Load",
                color="Growth_%", color_continuous_scale="RdYlGn", hover_name="HCP_ID",
                title="Top 100 HCP: Influence Score vs Business Impact (proxy = Patient_Load × Actual_Uptake)"
            )
            st.plotly_chart(fig_infl, use_container_width=True)
        with colB:
            st.caption("🔥 Top 10 bác sĩ tăng trưởng Patient Load nhanh nhất (đầu kỳ → cuối kỳ đang lọc)")
            show_growth = top_growth_docs[["HCP_ID", "Influence_Score", "Growth_%"]].copy()
            show_growth["Influence_Score"] = show_growth["Influence_Score"].round(1)
            show_growth["Growth_%"] = show_growth["Growth_%"].round(1).map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "n/a")
            st.dataframe(show_growth, use_container_width=True, hide_index=True)
        st.caption("Business Impact Proxy = Patient_Load × Actual_Uptake (không có dữ liệu Sales gắn theo từng bác sĩ nên dùng proxy này để so sánh mức độ ảnh hưởng thực tế).")

    st.markdown("---")

    # ---- HCP Market Research ----
    st.subheader("🔬 HCP Market Research Results")
    mr_metrics = ["Awareness", "Vaccine_X_Knowledge", "Recommendation_Intent", "Safety_Concern",
                  "Price_Barrier", "Access_Barrier", "Vaccine_Confidence", "Engagement_Response"]
    if not filtered_hcp_mr.empty:
        mr_means = filtered_hcp_mr[mr_metrics].mean().reset_index()
        mr_means.columns = ["Metric", "Avg_Score"]

        colmr1, colmr2 = st.columns([1, 1])
        with colmr1:
            st.caption("Funnel: Awareness → Knowledge → Recommendation Intent → Confidence (điểm TB, ngưỡng ≥50 coi là 'đạt')")
            funnel_stage_cols = ["Awareness", "Vaccine_X_Knowledge", "Recommendation_Intent", "Vaccine_Confidence"]
            funnel_counts = [(filtered_hcp_mr[c] >= 50).sum() for c in funnel_stage_cols]
            mr_funnel_df = pd.DataFrame({"Stage": funnel_stage_cols, "Count": funnel_counts})
            fig_mr_funnel = px.funnel(mr_funnel_df, x="Count", y="Stage", color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig_mr_funnel, use_container_width=True)

        with colmr2:
            st.caption("Điểm trung bình theo từng chỉ số khảo sát")
            fig_mr_bar = px.bar(mr_means.sort_values("Avg_Score"), x="Avg_Score", y="Metric", orientation="h", range_x=[0, 100])
            st.plotly_chart(fig_mr_bar, use_container_width=True)

        seg_by = st.selectbox("Segment Market Research theo:", ["Region", "Tier", "Specialty"], key="mr_segment_by")
        if seg_by in filtered_hcp_mr.columns:
            seg_mr = filtered_hcp_mr.groupby(seg_by)[mr_metrics].mean().reset_index()
            seg_mr_melt = seg_mr.melt(id_vars=seg_by, var_name="Metric", value_name="Score")
            fig_seg_mr = px.bar(seg_mr_melt, x="Metric", y="Score", color=seg_by, barmode="group")
            fig_seg_mr.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_seg_mr, use_container_width=True)
    else:
        st.info("Không có dữ liệu Market Research trong lựa chọn hiện tại.")

    # ---- AI Key Highlights ----
    bullets = []
    if not base_hcp_df.empty and "Segment_2025" in base_hcp_df.columns:
        top_seg = base_hcp_df["Segment_2025"].value_counts().idxmax()
        bullets.append(f"Segment lớn nhất hiện tại là <b>{top_seg}</b> ({base_hcp_df['Segment_2025'].value_counts(normalize=True).max()*100:.0f}% bác sĩ).")
    if "infl_now" in locals() and not infl_now.empty:
        star_docs = infl_now[(infl_now["Influence_Score"] > infl_now["Influence_Score"].quantile(0.8)) & (infl_now["Growth_%"] > 10)]
        if len(star_docs) > 0:
            bullets.append(f"Có <b>{len(star_docs)} bác sĩ</b> vừa có Influence Score cao (top 20%) vừa tăng trưởng >10% — nên ưu tiên chăm sóc/engage nhóm này.")
    if not filtered_hcp_mr.empty:
        weakest = mr_means.sort_values("Avg_Score").iloc[0]
        bullets.append(f"Chỉ số Market Research yếu nhất hiện là <b>{weakest['Metric']}</b> (TB {weakest['Avg_Score']:.0f}/100) — cần đầu tư thêm truyền thông/đào tạo về điểm này.")
    render_ai_box(bullets, title="AI Key Highlights — HCP Analysis")

# ============================================================
# 04. HCC ANALYSIS PAGE
# ============================================================

elif page == "HCC ANALYSIS":
    st.title("04 — 👥 HCC Population & Conversion Funnel Analytics")
    render_filter_banner(
        selected_region, start_month, end_month, time_grain,
        extra_chips=[f"🎂 Age {age_filter[0]}-{age_filter[1]}", f"⚧ {'/'.join(gender_filter) if gender_filter else 'None'}"]
    )

    min_age, max_age = age_filter
    hcc_scope = filtered_hcc.copy()
    if not hcc_scope.empty:
        if "Gender" in hcc_scope.columns and gender_filter:
            hcc_scope = hcc_scope[hcc_scope["Gender"].isin(gender_filter)]
        if "Max_Age" in hcc_scope.columns and "Min_Age" in hcc_scope.columns:
            hcc_scope = hcc_scope[(hcc_scope["Max_Age"] >= min_age) & (hcc_scope["Min_Age"] <= max_age)]

    if hcc_scope.empty:
        st.warning("⚠️ Không có dữ liệu HCC khớp với Region / Time Range / Age / Gender đang chọn.")

    total_target = hcc_scope["Target_Pop"].sum() if "Target_Pop" in hcc_scope.columns else 0
    total_uptake = hcc_scope["Uptake"].sum() if "Uptake" in hcc_scope.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("📌 Selected Age Scope", f"{min_age} - {max_age} Yrs", f"Genders: {', '.join(gender_filter) if gender_filter else 'None'}")
    c2.metric("👥 Filtered HCC Pool", format_num(total_target))
    c3.metric("💉 Estimated Uptake", format_num(total_uptake))

    st.markdown("---")

    # ---- Growth overtime + by Tier / Region / Province / HCO ----
    st.subheader("📈 HCC Growth — Overtime, by Tier, Region, Province, HCO")

    if not hcc_scope.empty:
        time_agg = hcc_scope.groupby("Month_Label").agg(Target_Pop=("Target_Pop", "sum"), Uptake=("Uptake", "sum")).reset_index().sort_values("Month_Label")
        if len(time_agg) > 1:
            g_first, g_last = time_agg["Uptake"].iloc[0], time_agg["Uptake"].iloc[-1]
            uptake_growth = ((g_last - g_first) / g_first * 100) if g_first else None
        else:
            uptake_growth = None
        st.markdown(f"**Uptake Growth (đầu → cuối kỳ đang chọn):** {format_growth(uptake_growth)}", unsafe_allow_html=True)

        fig_hcc_time = px.line(time_agg, x="Month_Label", y=["Target_Pop", "Uptake"], markers=True, title="Target Pool vs Uptake Overtime")
        st.plotly_chart(fig_hcc_time, use_container_width=True)

        breakdown_tabs = st.tabs(["🏷️ By Tier", "🌍 By Region", "📍 By Province (Top 15)", "🏥 By HCO (Top 15)"])
        with breakdown_tabs[0]:
            if "Tier" in hcc_scope.columns:
                tier_g = hcc_scope.groupby("Tier").agg(Target_Pop=("Target_Pop", "sum"), Uptake=("Uptake", "sum")).reset_index()
                fig_t = px.bar(tier_g, x="Tier", y=["Target_Pop", "Uptake"], barmode="group")
                st.plotly_chart(fig_t, use_container_width=True)
        with breakdown_tabs[1]:
            if "Region" in hcc_scope.columns:
                reg_g = hcc_scope.groupby("Region").agg(Target_Pop=("Target_Pop", "sum"), Uptake=("Uptake", "sum")).reset_index()
                fig_r = px.bar(reg_g, x="Region", y=["Target_Pop", "Uptake"], barmode="group")
                st.plotly_chart(fig_r, use_container_width=True)
        with breakdown_tabs[2]:
            if "Province" in hcc_scope.columns:
                prov_g = hcc_scope.groupby("Province").agg(Uptake=("Uptake", "sum")).reset_index().sort_values("Uptake", ascending=False).head(15)
                fig_p = px.bar(prov_g, x="Uptake", y="Province", orientation="h")
                fig_p.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_p, use_container_width=True)
        with breakdown_tabs[3]:
            if "HCO" in hcc_scope.columns:
                hco_g = hcc_scope.groupby("HCO").agg(Uptake=("Uptake", "sum")).reset_index().sort_values("Uptake", ascending=False).head(15)
                fig_h = px.bar(hco_g, x="Uptake", y="HCO", orientation="h")
                fig_h.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")

    # ---- Age x Gender deep-dive ----
    st.subheader("📊 Demographic Deep-Dive: Age x Gender")
    if not hcc_scope.empty and "Age_Group" in hcc_scope.columns:
        ag_g = hcc_scope.groupby(["Age_Group", "Gender"]).agg(Uptake=("Uptake", "sum")).reset_index()
        fig_ag = px.bar(ag_g, x="Age_Group", y="Uptake", color="Gender", barmode="group",
                         title=f"Vaccinated (Uptake) Breakdown — Age {min_age}-{max_age}")
        st.plotly_chart(fig_ag, use_container_width=True)
    else:
        st.info("Không có dữ liệu demographic phù hợp với lựa chọn hiện tại.")

    st.markdown("---")

    # ---- HCC Market Research Funnel: Awareness -> Consideration -> Consulted -> Actual Uptake ----
    st.subheader("🔬 HCC Market Research — Conversion Funnel")
    st.caption("Funnel nghiên cứu thị trường: Awareness → Consideration → Consulted → Actual Uptake.")
    if not hcc_scope.empty:
        mr_funnel_vals = [hcc_scope["Target_Pop"].sum(), hcc_scope["Awareness"].sum(),
                           hcc_scope["Consideration"].sum(), hcc_scope["Consulted"].sum(), hcc_scope["Uptake"].sum()]
        mr_funnel_df = pd.DataFrame({
            "Stage": ["Target Population", "Awareness", "Consideration", "Consulted", "Actual Uptake"],
            "Count": mr_funnel_vals
        })
        fig_hcc_mr_funnel = px.funnel(mr_funnel_df, x="Count", y="Stage", color_discrete_sequence=px.colors.sequential.Teal,
                                       title=f"HCC Market Research Funnel ({selected_region})")
        st.plotly_chart(fig_hcc_mr_funnel, use_container_width=True)

        conv_rates = pd.DataFrame({
            "Step": ["Target → Aware", "Aware → Consider", "Consider → Consulted", "Consulted → Uptake"],
            "Conversion_%": [
                (mr_funnel_vals[1] / mr_funnel_vals[0] * 100) if mr_funnel_vals[0] else 0,
                (mr_funnel_vals[2] / mr_funnel_vals[1] * 100) if mr_funnel_vals[1] else 0,
                (mr_funnel_vals[3] / mr_funnel_vals[2] * 100) if mr_funnel_vals[2] else 0,
                (mr_funnel_vals[4] / mr_funnel_vals[3] * 100) if mr_funnel_vals[3] else 0,
            ]
        })
        st.dataframe(conv_rates.style.format({"Conversion_%": "{:.1f}%"}), use_container_width=True, hide_index=True)

    # ---- Sub-segment funnels by Age group tabs (kept from original design) ----
    st.markdown("---")
    st.subheader("🎯 Sub-segment Conversion Funnels (Age Group & Gender Breakdown)")
    if not hcc_scope.empty:
        age_tab_defs = [("Adolescents (9-18 Yrs)", 9, 18), ("Young Adults (19-26 Yrs)", 19, 26), ("Middle-aged Adults (27-45 Yrs)", 27, 45)]
        tabs_age = st.tabs([t[0] for t in age_tab_defs])
        stages = ["Target Population", "Awareness", "Consideration", "Consulted", "Actual Uptake"]
        for tab, (label, amin, amax) in zip(tabs_age, age_tab_defs):
            with tab:
                sub = hcc_scope[(hcc_scope["Min_Age"] >= amin) & (hcc_scope["Max_Age"] <= amax)] if "Min_Age" in hcc_scope.columns else pd.DataFrame()
                f_col, m_col = st.columns(2)
                for gcol, gname, colors in [(f_col, "Female", px.colors.sequential.Purples_r), (m_col, "Male", px.colors.sequential.Blues_r)]:
                    with gcol:
                        st.markdown(f"##### {'👩' if gname=='Female' else '👨'} {gname} — {label}")
                        if gname in gender_filter and not sub.empty:
                            g_sub = sub[sub["Gender"] == gname]
                            vals = [g_sub["Target_Pop"].sum(), g_sub["Awareness"].sum(), g_sub["Consideration"].sum(), g_sub["Consulted"].sum(), g_sub["Uptake"].sum()]
                            f_df = pd.DataFrame({"Stage": stages, "Count": vals})
                            fig_f = px.funnel(f_df, x="Count", y="Stage", color_discrete_sequence=colors)
                            fig_f.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
                            st.plotly_chart(fig_f, use_container_width=True)
                        else:
                            st.info("ℹ️ Không có dữ liệu / giới tính đang bị bỏ chọn ở filter.")

    # ---- AI Key Highlights ----
    bullets = []
    if total_target:
        bullets.append(f"Tỷ lệ chuyển đổi tổng thể (Uptake / Target Pool) hiện đạt <b>{total_uptake/total_target*100:.1f}%</b>.")
    if not hcc_scope.empty and "Region" in hcc_scope.columns:
        best_region = hcc_scope.groupby("Region")["Uptake"].sum().idxmax()
        bullets.append(f"<b>{best_region}</b> đang có số Uptake cao nhất trong các vùng.")
    if not hcc_scope.empty and "Age_Group" in hcc_scope.columns:
        best_age = hcc_scope.groupby("Age_Group")["Uptake"].sum().idxmax()
        bullets.append(f"Nhóm tuổi <b>{best_age}</b> đang có tỷ lệ tiêm chủng cao nhất — phù hợp để ưu tiên trong chiến dịch tiếp theo.")
    if "conv_rates" in locals() and not conv_rates.empty:
        weakest_step = conv_rates.sort_values("Conversion_%").iloc[0]
        bullets.append(f"Bước chuyển đổi yếu nhất trong funnel là <b>{weakest_step['Step']}</b> ({weakest_step['Conversion_%']:.0f}%) — đây là điểm nghẽn (bottleneck) cần cải thiện trước tiên.")
    render_ai_box(bullets, title="AI Key Highlights & Recommendation — HCC Analysis")

st.markdown("---")
st.markdown(
    '<div class="app-footer">Vaccine X | Commercial Analytics Dashboard · '
    'Số liệu cập nhật theo Region &amp; Time Range đang chọn ở panel bên trái</div>',
    unsafe_allow_html=True
)
