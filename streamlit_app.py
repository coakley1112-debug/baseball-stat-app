
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def read_required_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        st.error(
            f"Missing required data file: {filename}. Upload {filename} to the same GitHub repository folder as this app file. "
            "Streamlit Cloud is case-sensitive, so the filename must match exactly."
        )
        st.stop()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.error(f"Could not read {filename}: {e}")
        st.stop()

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# Daniel Cohen Baseball Explorer - Full Updated Version
# ============================================================

team_id_mapping = {
    "SFN": "SFG", "SLN": "STL", "CHN": "CHC", "NYA": "NYY", "NYN": "NYM",
    "FLO": "MIA", "LAN": "LAD", "BRO": "LAD", "SLA": "BAL",
    "WS1": "MIN", "WS2": "TEX", "NY1": "SFG", "ML4": "MIL", "ML1": "ATL",
    "TBA": "TBR", "SDN": "SDP", "CHA": "CWS", "KCA": "KCR",
    "SE1": "MIL", "MON": "WAS", "CAL": "LAA", "ANA": "LAA",
    "PHA": "OAK", "KC1": "OAK", "BSN": "ATL"
}

team_id_to_name = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies", "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners", "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals"
}

team_id_to_historical_name = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox", "CHA": "Chicago White Sox", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals", "KCA": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "CAL": "California Angels", "ANA": "Anaheim Angels",
    "LAD": "Los Angeles Dodgers", "LAN": "Los Angeles Dodgers", "BRO": "Brooklyn Dodgers",
    "MIA": "Miami Marlins", "FLO": "Florida Marlins", "MIL": "Milwaukee Brewers",
    "ML4": "Milwaukee Brewers", "SE1": "Seattle Pilots", "MIN": "Minnesota Twins",
    "WS1": "Washington Senators", "NYM": "New York Mets", "NYN": "New York Mets",
    "NYY": "New York Yankees", "NYA": "New York Yankees", "OAK": "Athletics",
    "PHA": "Philadelphia Athletics", "KC1": "Kansas City Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres",
    "SDN": "San Diego Padres", "SEA": "Seattle Mariners", "SFG": "San Francisco Giants",
    "SFN": "San Francisco Giants", "NY1": "New York Giants", "STL": "St. Louis Cardinals",
    "SLN": "St. Louis Cardinals", "SLA": "St. Louis Browns", "TBR": "Tampa Bay Rays",
    "TBA": "Tampa Bay Devil Rays", "TEX": "Texas Rangers", "WS2": "Washington Senators",
    "TOR": "Toronto Blue Jays", "WAS": "Washington Nationals", "MON": "Montreal Expos",
    "ML1": "Milwaukee Braves", "BSN": "Boston Braves"
}


# Lightweight team/league context features for ML and filters. Historical exceptions are handled by year:
# Houston was NL through 2012 and AL starting 2013; Milwaukee/Seattle was AL through 1997 and NL starting 1998.
AL_TEAMS = {"BAL", "BOS", "NYY", "TBR", "TOR", "CWS", "CLE", "DET", "KCR", "MIN", "HOU", "LAA", "OAK", "SEA", "TEX"}
NL_TEAMS = {"ATL", "MIA", "NYM", "PHI", "WAS", "CHC", "CIN", "MIL", "PIT", "STL", "ARI", "COL", "LAD", "SDP", "SFG"}
TEAM_PARK_FACTOR = {
    "COL": 1.15, "BOS": 1.06, "CIN": 1.05, "NYY": 1.04, "PHI": 1.04, "BAL": 1.03,
    "HOU": 1.02, "TEX": 1.02, "ATL": 1.01, "CHC": 1.01, "LAD": 1.00, "NYM": 1.00,
    "TOR": 1.00, "STL": 0.99, "MIL": 0.99, "ARI": 0.99, "SDP": 0.98, "SFG": 0.97,
    "SEA": 0.97, "DET": 0.97, "MIA": 0.96, "OAK": 0.95, "PIT": 0.98, "CLE": 0.99,
    "KCR": 0.99, "MIN": 0.99, "CWS": 1.00, "TBR": 1.00, "WAS": 1.00, "LAA": 1.00
}

def normalize_team_id(team_id):
    return team_id_mapping.get(str(team_id), str(team_id))

def safe_int_year(year):
    try:
        if pd.isna(year):
            return None
        return int(year)
    except Exception:
        return None

def team_league(team_id, year=None):
    """Return AL/NL using historical league membership when a season year is available."""
    tid = normalize_team_id(team_id)
    yr = safe_int_year(year)
    if tid == "HOU" and yr is not None:
        return "NL" if yr <= 2012 else "AL"
    if tid == "MIL" and yr is not None:
        return "AL" if yr <= 1997 else "NL"
    if tid in AL_TEAMS:
        return "AL"
    if tid in NL_TEAMS:
        return "NL"
    return "Unknown"

def historical_team_name(team_id_original, year=None):
    """Display the real historical team name for the season while preserving franchise filtering.

    Cleveland is one franchise for filtering, but it should display as Indians through 2021
    and Guardians starting in 2022. Other historical franchise names use Lahman team IDs
    where available, such as BRO, FLO, MON, SLA, etc.
    """
    original = str(team_id_original)
    tid = normalize_team_id(original)
    yr = safe_int_year(year)
    if tid == "CLE" and yr is not None:
        return "Cleveland Indians" if yr <= 2021 else "Cleveland Guardians"
    return team_id_to_historical_name.get(original, team_id_to_historical_name.get(tid, original))

COUNT_STATS = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "G"]
RATE_STATS = ["BA", "OBP", "SLG", "OPS"]
TREND_COUNT_COLS = ["R Δ", "H Δ", "2B Δ", "3B Δ", "HR Δ", "RBI Δ", "SB Δ", "BB Δ"]
TREND_RATE_COLS = ["BA Δ", "OBP Δ", "SLG Δ", "OPS Δ"]
ML_TARGET_STATS = ["R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]
ML_BASE_FEATURE_STATS = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"]
ML_DERIVED_FEATURE_STATS = ["PA_est", "BB_rate", "K_rate", "SB_rate", "XBH", "XBH_rate", "HR_rate", "Speed_Index"]

st.set_page_config(page_title="⚾ Daniel Cohen Baseball Explorer ⚾", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
.title-box {background: linear-gradient(90deg, #0b1f3a, #1f4e79); padding: 22px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(0,0,0,0.18);}
.title-text {color: white; font-size: 36px; font-weight: 800; margin: 0;}
.subtitle-text {color: #dbe8f5; font-size: 16px; margin-top: 6px;}
.section-card {background-color: #f7f9fc; padding: 16px; border-radius: 12px; border: 1px solid #d9e2ec; margin-bottom: 16px;}
.section-title {font-size: 24px; font-weight: 800; color: #12324a; margin-bottom: 6px;}
.small-note {color: #4f6475; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <div class="title-text">⚾ Daniel Cohen Baseball Explorer</div>
    <div class="subtitle-text">
        Historical stats, career totals, custom leaderboards, player comparisons, trend direction, and valuation insights
    </div>
</div>
""", unsafe_allow_html=True)

def fmt_int(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x): return ""
    return f"{x:.0f}"

def fmt_count_1(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x): return ""
    return f"{x:.1f}"

def fmt_rate_3(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x): return ""
    return f"{x:.3f}"

def fmt_rate_4(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x): return ""
    return f"{x:.4f}"

def safe_round_rate_stats(df):
    df = df.copy()
    for col in ["BA", "OBP", "SLG", "OPS", "BA_roll", "OBP_roll", "SLG_roll", "OPS_roll"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
    return df

def build_player_label_map(df):
    options = df[["playerID", "fullName"]].drop_duplicates().sort_values(["fullName", "playerID"])
    return {f"{row.fullName} ({row.playerID})": row.playerID for row in options.itertuples()}

def compute_trend_slope(group, stat_col):
    group = group.sort_values("yearID")
    x = pd.to_numeric(group["yearID"], errors="coerce").values
    y = pd.to_numeric(group[stat_col], errors="coerce").values
    mask = ~pd.isna(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return np.polyfit(x, y, 1)[0]

def add_missing_numeric_columns(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def add_rate_stats(df):
    df = df.copy()
    needed_cols = ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SF"]
    df = add_missing_numeric_columns(df, needed_cols)
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    ab_denom = df["AB"].replace(0, np.nan)
    obp_denom = (df["AB"] + df["BB"] + df["HBP"] + df["SF"]).replace(0, np.nan)
    df["BA"] = pd.to_numeric(df["H"] / ab_denom, errors="coerce")
    df["OBP"] = pd.to_numeric((df["H"] + df["BB"] + df["HBP"]) / obp_denom, errors="coerce")
    df["SLG"] = pd.to_numeric((df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]) / ab_denom, errors="coerce")
    df["OPS"] = pd.to_numeric(df["OBP"] + df["SLG"], errors="coerce")
    return df

def apply_stat_min_filters(df, prefix):
    df = df.copy()
    stat_columns = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]
    for col in stat_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    st.subheader("Minimum Stat Filters")
    cols = st.columns(4)
    mins = {}
    for i, stat in enumerate(stat_columns):
        with cols[i % 4]:
            if stat in RATE_STATS:
                mins[stat] = st.number_input(f"Min {stat}", min_value=0.0, value=0.0, step=0.001, format="%.3f", key=f"{prefix}_{stat}_min")
            else:
                mins[stat] = st.number_input(f"Min {stat}", min_value=0, value=0, step=1, key=f"{prefix}_{stat}_min")
    for stat, min_val in mins.items():
        df = df[df[stat] >= min_val]
    return df

def color_trend(val):
    try:
        if pd.isna(val):
            return ""
        num = pd.to_numeric(str(val).replace(",", ""), errors="coerce")
        if pd.isna(num):
            return ""
        if num > 0:
            return "color: green; font-weight: bold;"
        if num < 0:
            return "color: red; font-weight: bold;"
        return "color: gray;"
    except Exception:
        return ""

def classify_trend(row):
    ops_trend = pd.to_numeric(row.get("OPS_trend", np.nan), errors="coerce")
    hr_trend = pd.to_numeric(row.get("HR_trend", np.nan), errors="coerce")
    if pd.isna(ops_trend): return "insufficient"
    if ops_trend >= 0.03 or (pd.notna(hr_trend) and hr_trend >= 3): return "breakout"
    if ops_trend <= -0.03 or (pd.notna(hr_trend) and hr_trend <= -3): return "decline"
    return "stable"

def describe_valuation_index(index):
    index = pd.to_numeric(index, errors="coerce")
    if pd.isna(index): return "not enough data to classify the valuation score"
    if index >= 0.90: return "an elite valuation score"
    if index >= 0.75: return "a very strong valuation score"
    if index >= 0.50: return "a solid middle-tier valuation score"
    if index >= 0.25: return "a low valuation score"
    return "a very low valuation score"

def make_trend_insight_summary(row):
    player = row.get("fullName", "This player")
    ops_trend = pd.to_numeric(row.get("OPS_trend", np.nan), errors="coerce")
    hr_trend = pd.to_numeric(row.get("HR_trend", np.nan), errors="coerce")
    rbi_trend = pd.to_numeric(row.get("RBI_trend", np.nan), errors="coerce")
    sb_trend = pd.to_numeric(row.get("SB_trend", np.nan), errors="coerce")
    xbh_trend = pd.to_numeric(row.get("XBH_noHR_trend", np.nan), errors="coerce")
    proj_ops = pd.to_numeric(row.get("proj_OPS", np.nan), errors="coerce")
    proj_hr = pd.to_numeric(row.get("proj_HR", np.nan), errors="coerce")
    proj_rbi = pd.to_numeric(row.get("proj_RBI", np.nan), errors="coerce")
    proj_sb = pd.to_numeric(row.get("proj_SB", np.nan), errors="coerce")
    proj_xbh = pd.to_numeric(row.get("proj_XBH", np.nan), errors="coerce")
    trend_type = classify_trend(row)
    label = "a breakout candidate" if trend_type == "breakout" else ("a decline risk" if trend_type == "decline" else "a stable profile")
    return (
        f"{player} looks like {label}. "
        f"OPS trend is {fmt_rate_4(ops_trend)} per year, HR trend is {fmt_count_1(hr_trend)}, "
        f"2B+3B trend is {fmt_count_1(xbh_trend)}, RBI trend is {fmt_count_1(rbi_trend)}, "
        f"and SB trend is {fmt_count_1(sb_trend)}. "
        f"If the recent pattern continues, the next-season trend estimate is roughly "
        f"{fmt_rate_4(proj_ops)} OPS, {fmt_count_1(proj_hr)} HR, {fmt_count_1(proj_xbh)} doubles/triples, "
        f"{fmt_count_1(proj_rbi)} RBI, and {fmt_count_1(proj_sb)} SB."
    )

def make_valuation_summary(row):
    player = row.get("fullName", "This player")
    trend_score = pd.to_numeric(row.get("Trend_Score", np.nan), errors="coerce")
    perf_score = pd.to_numeric(row.get("Perf_Score", np.nan), errors="coerce")
    valuation_score = pd.to_numeric(row.get("Valuation_Score", np.nan), errors="coerce")
    proj_ops = pd.to_numeric(row.get("proj_OPS", np.nan), errors="coerce")
    proj_hr = pd.to_numeric(row.get("proj_HR", np.nan), errors="coerce")
    proj_xbh = pd.to_numeric(row.get("proj_XBH", np.nan), errors="coerce")
    proj_rbi = pd.to_numeric(row.get("proj_RBI", np.nan), errors="coerce")
    proj_sb = pd.to_numeric(row.get("proj_SB", np.nan), errors="coerce")
    valuation_description = describe_valuation_index(valuation_score)
    return (
        f"{player}'s Trend Score is {fmt_count_1(trend_score)}, Current Score is {fmt_count_1(perf_score)}, "
        f"and Valuation Score is {fmt_rate_4(valuation_score)}. "
        f"That is {valuation_description}. "
        f"The Valuation Score combines current score with recent trend direction, "
        f"then scales the result from 0 to 1 compared with the other players in the filtered group. "
        f"If the recent pattern continues, the next-season trend estimate is roughly "
        f"{fmt_rate_4(proj_ops)} OPS, {fmt_count_1(proj_hr)} HR, {fmt_count_1(proj_xbh)} doubles/triples, "
        f"{fmt_count_1(proj_rbi)} RBI, and {fmt_count_1(proj_sb)} SB."
    )

def render_section_header(title, note):
    st.markdown(f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="small-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

def top_bar_chart(df, name_col, value_col, title, top_n=10):
    if df.empty or value_col not in df.columns or name_col not in df.columns:
        return
    chart_df = df[[name_col, value_col]].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)
    if chart_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(chart_df[name_col], chart_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.invert_yaxis()
    st.pyplot(fig)

def format_display_table(df, count_cols=None, rate_cols=None, score_cols=None, count_decimals=0, rate_decimals=3):
    """Return a plain DataFrame for maximum Streamlit Cloud stability.
    Formatting is handled by rounding numeric columns instead of pandas Styler.

    count_decimals / rate_decimals let the Trend page keep change values readable:
    counting-stat changes use 1 decimal, while OPS/BA/OBP/SLG changes use 4 decimals.
    """
    df = df.copy()
    count_cols = count_cols or []
    rate_cols = rate_cols or []
    score_cols = score_cols or []
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(count_decimals)
    for col in rate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(rate_decimals)
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4 if col == "Valuation Score" else 1)
    return df





MAX_TABLE_DISPLAY_ROWS = 500

@st.cache_data(show_spinner=False)
def _df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def render_output_table(df, *, key, file_name, display_rows=MAX_TABLE_DISPLAY_ROWS, style_cols=None):
    """Render a table quickly and add a CSV export button that opens cleanly in Excel."""
    table_df = df.copy()
    if len(table_df) > display_rows:
        st.caption(f"Showing first {display_rows:,} rows for speed. Export downloads all {len(table_df):,} rows.")
        display_df = table_df.head(display_rows)
    else:
        display_df = table_df

    style_cols = [c for c in (style_cols or []) if c in display_df.columns]
    # Avoid heavy styling on large tables. Styling was slowing Trend/Valuation pages and re-expanding decimals.
    if style_cols and display_df.size <= 6000:
        fmt = {}
        for col in display_df.columns:
            if col in TREND_RATE_COLS or col in ["OPS Δ", "BA Δ", "OBP Δ", "SLG Δ"]:
                fmt[col] = "{:.4f}"
            elif col in TREND_COUNT_COLS or col in ["HR Δ", "2B+3B Δ", "RBI Δ", "SB Δ", "R Δ", "H Δ", "2B Δ", "3B Δ", "BB Δ"]:
                fmt[col] = "{:.1f}"
            elif col in RATE_STATS or col in ["BA", "OBP", "SLG", "OPS"]:
                fmt[col] = "{:.3f}"
            elif col in ["Trend Score", "Current Score", "Performance Score", "Score"]:
                fmt[col] = "{:.1f}"
            elif col == "Valuation Score":
                fmt[col] = "{:.4f}"
        styled_df = display_df.style.map(color_trend, subset=style_cols).format(fmt)
        st.dataframe(styled_df, width="stretch", hide_index=True)
    else:
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.download_button(
        "Export CSV for Excel",
        data=_df_to_csv_bytes(table_df),
        file_name=file_name,
        mime="text/csv",
        width="content",
    )


def _numeric_plot_columns(df):
    """Return useful numeric columns for chart axes.

    The chart can use internal fields such as Age/Games, but it intentionally
    hides backend IDs and raw birth-date fields from the dropdowns.
    """
    blocked = {"birthday", "birthmonth", "birthyear", "birth day", "birth month", "birth year"}
    preferred = [
        "Age", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB",
        "BA", "OBP", "SLG", "OPS", "Current Score", "Valuation Score", "Score",
        "Debut Age", "Final Age", "Average Age"
    ]
    cols = []
    for c in preferred:
        if c in df.columns and c not in cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                cols.append(c)
    for c in df.columns:
        c_key = str(c).replace("_", " ").lower().strip()
        if c not in cols and c_key not in blocked:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0 and not str(c).lower().endswith("id"):
                cols.append(c)
    return cols

def _categorical_plot_columns(df):
    """Color-by options should stay consistent across Historical and Career scatterplots."""
    options = []
    for col in ["Primary Position", "Bats", "Team", "League"]:
        if col in df.columns:
            options.append(col)
    return options

def _prepare_historical_scatter_data(hist_df, team_col):
    """Build plot-ready data for Historical Explorer.

    The visible table stays clean, but the scatterplot can use internal fields such as
    season Age and G.
    """
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    plot_df = hist_df.copy()
    plot_df["Player"] = plot_df.get("fullName", "")
    plot_df["Year"] = pd.to_numeric(plot_df.get("yearID"), errors="coerce")
    plot_df["Team"] = plot_df.get(team_col, plot_df.get("teamHistoricalName", plot_df.get("primaryHistoricalTeamName", "")))
    plot_df["Primary Position"] = plot_df.get("displayPosition", plot_df.get("primaryPos", plot_df.get("careerPrimaryPos", "")))
    plot_df["Bats"] = plot_df.get("bats", "")
    if "teamLeague" in plot_df.columns:
        plot_df["League"] = plot_df["teamLeague"]
    elif "primaryLeague" in plot_df.columns:
        plot_df["League"] = plot_df["primaryLeague"]
    else:
        plot_df["League"] = "Unknown"
    plot_df["League"] = plot_df["League"].replace({"AL": "American League", "NL": "National League", "Unknown League": "Unknown", "": "Unknown"}).fillna("Unknown")

    if {"yearID", "birthYear"}.issubset(plot_df.columns):
        plot_df["Age"] = plot_df.apply(
            lambda r: baseball_age_for_season(
                r.get("yearID"),
                r.get("birthYear"),
                r.get("birthMonth", np.nan),
                r.get("birthDay", np.nan)
            ),
            axis=1
        )
    # Keep games labeled as G only. Do not create a duplicate "Games" field.
    if "G" in plot_df.columns:
        plot_df["G"] = pd.to_numeric(plot_df["G"], errors="coerce")
    return plot_df

def _prepare_career_scatter_data(career_df, filtered_source_df=None):
    """Build plot-ready data for Career Totals.

    Career age is less natural than season age, so this adds Debut Age, Final Age,
    and Average Age when birth/year fields are available in the filtered source.
    """
    if career_df is None or career_df.empty:
        return pd.DataFrame()
    plot_df = career_df.copy()
    plot_df["Player"] = plot_df.get("fullName", plot_df.get("Player", ""))
    plot_df["Team"] = plot_df.get("displayTeam", plot_df.get("Team", plot_df.get("teamHistoricalName", plot_df.get("primaryHistoricalTeamName", ""))))
    plot_df["Primary Position"] = plot_df.get("displayPosition", plot_df.get("Primary Position", plot_df.get("careerPrimaryPos", plot_df.get("primaryPos", ""))))
    plot_df["Bats"] = plot_df.get("bats", plot_df.get("Bats", ""))
    if "League" in plot_df.columns:
        plot_df["League"] = plot_df["League"]
    elif "primaryLeague" in plot_df.columns:
        plot_df["League"] = plot_df["primaryLeague"]
    elif "teamLeague" in plot_df.columns:
        plot_df["League"] = plot_df["teamLeague"]
    elif filtered_source_df is not None and "teamLeague" in filtered_source_df.columns and "playerID" in plot_df.columns:
        league_mode = (
            filtered_source_df.groupby(["playerID", "teamLeague"]).size().reset_index(name="count")
            .sort_values(["playerID", "count"], ascending=[True, False])
            .drop_duplicates("playerID")[["playerID", "teamLeague"]]
            .rename(columns={"teamLeague": "League"})
        )
        plot_df = plot_df.merge(league_mode, on="playerID", how="left")
    else:
        plot_df["League"] = "Unknown"
    plot_df["League"] = plot_df["League"].replace({"AL": "American League", "NL": "National League", "Unknown League": "Unknown", "": "Unknown"}).fillna("Unknown")
    # Keep games labeled as G only. Do not create a duplicate "Games" field.
    if "G" in plot_df.columns:
        plot_df["G"] = pd.to_numeric(plot_df["G"], errors="coerce")

    if filtered_source_df is not None and not filtered_source_df.empty and "playerID" in plot_df.columns:
        src = filtered_source_df.copy()
        if {"yearID", "birthYear"}.issubset(src.columns):
            src["Season Age"] = src.apply(
                lambda r: baseball_age_for_season(
                    r.get("yearID"),
                    r.get("birthYear"),
                    r.get("birthMonth", np.nan),
                    r.get("birthDay", np.nan)
                ),
                axis=1
            )
            weight_col = "AB" if "AB" in src.columns else ("G" if "G" in src.columns else None)
            if weight_col:
                src["_age_weight"] = pd.to_numeric(src[weight_col], errors="coerce").fillna(0)
                def wavg(g):
                    ages = pd.to_numeric(g["Season Age"], errors="coerce")
                    weights = pd.to_numeric(g["_age_weight"], errors="coerce").fillna(0)
                    mask = ages.notna() & (weights > 0)
                    if mask.sum() == 0:
                        return ages.mean()
                    return np.average(ages[mask], weights=weights[mask])
                age_summary = src.groupby("playerID").apply(
                    lambda g: pd.Series({
                        "Debut Age": pd.to_numeric(g["Season Age"], errors="coerce").min(),
                        "Final Age": pd.to_numeric(g["Season Age"], errors="coerce").max(),
                        "Average Age": wavg(g)
                    })
                ).reset_index()
            else:
                age_summary = src.groupby("playerID")["Season Age"].agg(
                    **{"Debut Age": "min", "Final Age": "max", "Average Age": "mean"}
                ).reset_index()
            plot_df = plot_df.merge(age_summary, on="playerID", how="left")
    return plot_df

def _year_axis_domain(series):
    """Zoom year axes around the dense part of the plotted data.

    This prevents a few very old/new rows from forcing a huge 1871-2025 range when
    nearly all visible dots are clustered in a narrower era.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return None
    full_min, full_max = float(s.min()), float(s.max())
    q_low, q_high = float(s.quantile(0.02)), float(s.quantile(0.98))
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        return None
    full_span = full_max - full_min
    dense_span = q_high - q_low
    # Only zoom if the dense range is meaningfully tighter than the full range.
    if full_span > 20 and dense_span < 0.75 * full_span:
        pad = max(1, round(dense_span * 0.06))
        return [int(np.floor(q_low - pad)), int(np.ceil(q_high + pad))]
    return [int(np.floor(full_min)), int(np.ceil(full_max))]

def _smart_axis_domain(series, pad=0.08, q_low=0.05, q_high=0.95):
    """Fit scatterplot axes to the dense part of the currently plotted data.

    This keeps charts readable when one or two outliers would otherwise force a
    huge empty range. It uses percentile clipping plus a small padding.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return None
    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high <= low:
        val = float(s.median()) if len(s) else 0.0
        width = max(abs(val) * 0.05, 1.0)
        return [val - width, val + width]
    padding = (high - low) * pad
    return [low - padding, high + padding]


def _axis_config_for_column(col_name, series):
    """Return an Altair scale/axis pair tuned for the selected statistic."""
    name = str(col_name).lower().strip()
    domain = _smart_axis_domain(series)
    axis_kwargs = {"title": col_name}

    if name == "year":
        domain = _year_axis_domain(series) or domain
        axis_kwargs["format"] = "d"
    elif name == "debut age":
        axis_kwargs["values"] = [18, 20, 22, 24, 26, 28]
    elif name == "final age":
        axis_kwargs["values"] = [32, 34, 36, 38, 40, 42]
    elif name == "average age":
        axis_kwargs["values"] = [23, 26, 29, 32, 35, 38, 41]
    elif name in {"ba", "avg", "batting average", "obp", "slg", "ops"}:
        axis_kwargs["format"] = ".3f" if name != "ops" else ".3f"

    return alt.Scale(domain=domain, zero=False) if domain else alt.Scale(zero=False), alt.Axis(**axis_kwargs)




def _full_axis_config_for_column(col_name, series):
    """Return an Altair scale/axis pair that includes every non-null point.

    Used for Full Outlier View. It avoids passing format=None to Altair, and it
    pads the min/max slightly so outliers do not sit directly on the chart border.
    """
    name = str(col_name).lower().strip()
    s = pd.to_numeric(series, errors="coerce").dropna()

    if name == "year":
        axis = alt.Axis(title=col_name, format="d")
    else:
        axis = alt.Axis(title=col_name)

    if s.empty:
        return alt.Scale(zero=False), axis

    low = float(s.min())
    high = float(s.max())

    if not np.isfinite(low) or not np.isfinite(high):
        return alt.Scale(zero=False), axis

    if high == low:
        pad = max(abs(high) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.06

    domain = [low - pad, high + pad]

    if name == "year":
        domain = [int(np.floor(low)), int(np.ceil(high))]

    return alt.Scale(domain=domain, zero=False, nice=True), axis

def _scatter_size_encoding(chart_df, size_col):
    """Scale dot size dynamically to the filtered data.

    Uses the 5th-95th percentile domain so one extreme outlier does not make all
    other dots look the same size.
    """
    if size_col == "None" or size_col not in chart_df.columns:
        return None
    vals = pd.to_numeric(chart_df[size_col], errors="coerce").dropna()
    if vals.empty:
        return None
    low = float(vals.quantile(0.05))
    high = float(vals.quantile(0.95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(vals.min()), float(vals.max())
    if high <= low:
        high = low + 1e-9
    return alt.Size(
        f"{size_col}:Q",
        title=size_col,
        scale=alt.Scale(domain=[low, high], range=[20, 300], clamp=True),
        legend=alt.Legend(title=size_col)
    )


def _scatter_color_encoding(chart_df, color_col):
    """Consistent color rules for league, handedness, team, and position scatterplots."""
    if color_col == "None" or color_col not in chart_df.columns:
        return None

    col_lower = color_col.lower()

    if "league" in col_lower:
        chart_df[color_col] = (
            chart_df[color_col]
            .replace({"AL": "American League", "NL": "National League", "Unknown League": "Unknown", "": "Unknown", None: "Unknown"})
            .fillna("Unknown")
        )
        return alt.Color(
            f"{color_col}:N",
            title=color_col,
            scale=alt.Scale(
                domain=["American League", "National League", "Unknown"],
                range=["#08519c", "#fb6a4a", "#bdbdbd"]
            ),
            legend=alt.Legend(title=color_col)
        )

    if color_col == "Bats":
        chart_df[color_col] = (
            chart_df[color_col]
            .replace({"": "Unknown", None: "Unknown"})
            .fillna("Unknown")
        )
        domain = ["L", "B", "R", "Unknown"]
        colors = ["#2ca25f", "#3182bd", "#de2d26", "#bdbdbd"]
        return alt.Color(
            f"{color_col}:N",
            title=color_col,
            scale=alt.Scale(domain=domain, range=colors),
            legend=alt.Legend(title=color_col)
        )

    if "position" in col_lower or color_col in ["POS", "Primary Position"]:
        chart_df[color_col] = (
            chart_df[color_col]
            .replace({"": "Unknown", None: "Unknown"})
            .fillna("Unknown")
        )
        return alt.Color(
            f"{color_col}:N",
            title=color_col,
            scale=alt.Scale(
                domain=["1B", "2B", "SS", "3B", "OF", "DH", "C", "P", "Unknown"],
                range=["#08306b", "#8c510a", "#238b45", "#ffd92f", "#e31a1c", "#756bb1", "#000000", "#bdbdbd", "#ffffff"]
            ),
            legend=alt.Legend(title=color_col)
        )

    if color_col == "Team" and "League" in chart_df.columns:
        chart_df["League"] = (
            chart_df["League"]
            .replace({"AL": "American League", "NL": "National League", "Unknown League": "Unknown", "": "Unknown", None: "Unknown"})
            .fillna("Unknown")
        )
        teams = [t for t in sorted(chart_df["Team"].dropna().astype(str).unique()) if t]
        al_palette = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#3182bd", "#084594", "#0868ac"]
        nl_palette = ["#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#fcae91", "#fdd0a2", "#e34a33"]
        other_palette = ["#bdbdbd", "#ffffff"]
        domain, colors = [], []
        for i, team in enumerate(teams):
            league_vals = chart_df.loc[chart_df["Team"].astype(str) == team, "League"].astype(str)
            league = league_vals.mode().iloc[0] if not league_vals.mode().empty else "Unknown"
            domain.append(team)
            if league == "American League":
                colors.append(al_palette[i % len(al_palette)])
            elif league == "National League":
                colors.append(nl_palette[i % len(nl_palette)])
            else:
                colors.append(other_palette[i % len(other_palette)])
        return alt.Color(
            f"{color_col}:N",
            title=color_col,
            scale=alt.Scale(domain=domain, range=colors),
            legend=alt.Legend(title=color_col)
        )

    return alt.Color(f"{color_col}:N", title=color_col, legend=alt.Legend(title=color_col))

def _best_fit_stats(chart_df, x_col, y_col):
    """Compute linear best-fit statistics for the selected scatterplot axes."""
    fit_df = chart_df[[x_col, y_col]].copy()
    fit_df[x_col] = pd.to_numeric(fit_df[x_col], errors="coerce")
    fit_df[y_col] = pd.to_numeric(fit_df[y_col], errors="coerce")
    fit_df = fit_df.dropna()
    fit_df = fit_df[np.isfinite(fit_df[x_col]) & np.isfinite(fit_df[y_col])]
    if len(fit_df) < 3 or fit_df[x_col].nunique() < 2 or fit_df[y_col].nunique() < 2:
        return None
    x = fit_df[x_col].to_numpy(dtype=float)
    y = fit_df[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    r2 = corr ** 2 if np.isfinite(corr) else np.nan
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    line_df = pd.DataFrame({
        x_col: [x_min, x_max],
        y_col: [slope * x_min + intercept, slope * x_max + intercept]
    })
    return {
        "slope": slope,
        "intercept": intercept,
        "corr": corr,
        "r2": r2,
        "line_df": line_df,
        "n": len(fit_df)
    }


def _format_equation_number(value):
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def render_scatterplot_section(plot_df, *, key_prefix, title="Visualize Results"):
    """Interactive scatterplot for the current filtered result set."""
    if plot_df is None or plot_df.empty:
        return

    plot_df = plot_df.copy()
    numeric_cols = _numeric_plot_columns(plot_df)
    if len(numeric_cols) < 2:
        return

    st.subheader(title)
    st.caption(
        "The scatterplot uses the current filtered results. It can also use internal chart fields "
        "like Age and Games even when those fields are not shown in the output table."
    )

    default_x = "HR" if "HR" in numeric_cols else numeric_cols[0]
    default_y = "SB" if "SB" in numeric_cols else ("OPS" if "OPS" in numeric_cols else numeric_cols[min(1, len(numeric_cols)-1)])

    p1, p2, p3, p4 = st.columns([1, 1, 1, 1])
    with p1:
        x_col = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index(default_x), key=f"{key_prefix}_scatter_x")
    with p2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(default_y), key=f"{key_prefix}_scatter_y")
    cat_options = ["None"] + _categorical_plot_columns(plot_df)
    with p3:
        color_col = st.selectbox("Color by", cat_options, index=0, key=f"{key_prefix}_scatter_color")
    size_options = ["None"] + numeric_cols
    with p4:
        size_col = st.selectbox("Size by", size_options, index=0, key=f"{key_prefix}_scatter_size")

    max_points = st.slider(
        "Maximum points to plot",
        min_value=250,
        max_value=5000,
        value=min(1500, max(250, len(plot_df))),
        step=250,
        key=f"{key_prefix}_scatter_max_points",
        help="Lower values make the chart faster when a filter returns a very large table."
    )

    view_mode = st.radio(
        "Scatterplot View",
        ["Focused View", "Full Outlier View"],
        horizontal=True,
        key=f"{key_prefix}_scatter_view_mode",
        help="Focused View keeps the main cluster readable. Full Outlier View expands the axes to include every outlier."
    )

    chart_df = plot_df.copy()
    chart_df[x_col] = pd.to_numeric(chart_df[x_col], errors="coerce")
    chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[x_col, y_col])
    if chart_df.empty:
        st.info("No rows have valid values for both selected axes.")
        return

    # Keep a full copy for axis domains so Full Outlier View truly includes the
    # entire filtered result set, even if we limit plotted points for speed.
    domain_df = chart_df.copy()

    if len(chart_df) > max_points:
        if view_mode == "Full Outlier View":
            # Preserve extreme x/y values before filling the rest with a broad sample.
            extreme_idx = set()
            for _col in [x_col, y_col]:
                extreme_idx.update(chart_df.nlargest(min(25, len(chart_df)), _col).index.tolist())
                extreme_idx.update(chart_df.nsmallest(min(25, len(chart_df)), _col).index.tolist())
            remaining = chart_df.drop(index=list(extreme_idx), errors="ignore")
            needed = max_points - len(extreme_idx)
            if needed > 0 and not remaining.empty:
                sampled = remaining.sample(n=min(needed, len(remaining)), random_state=42)
                chart_df = pd.concat([chart_df.loc[list(extreme_idx)], sampled], axis=0)
            else:
                chart_df = chart_df.loc[list(extreme_idx)].head(max_points)
            st.caption(f"Showing {len(chart_df):,} plotted points for speed, with extreme outliers preserved. Export/narrow filters for all rows.")
        else:
            chart_df = chart_df.sort_values(y_col, ascending=False).head(max_points)
            st.caption(f"Showing {max_points:,} plotted points for speed. Narrow filters for a complete visual.")

    tooltip_cols = [c for c in ["Player", "Year", "Team", "Primary Position", "Bats", "League", x_col, y_col, "G", "Age", "Debut Age", "Final Age", "Average Age", "OPS", "HR", "SB"] if c in chart_df.columns]
    tooltip_cols = list(dict.fromkeys(tooltip_cols))

    if view_mode == "Full Outlier View":
        # Full range: include every visible point/outlier on both axes.
        # This helper avoids passing format=None into Altair, which can break the chart.
        x_scale, x_axis = _full_axis_config_for_column(x_col, domain_df[x_col])
        y_scale, y_axis = _full_axis_config_for_column(y_col, domain_df[y_col])
    else:
        # Focused View: zooms to the dense middle of the data.
        x_scale, x_axis = _axis_config_for_column(x_col, chart_df[x_col])
        y_scale, y_axis = _axis_config_for_column(y_col, chart_df[y_col])

    enc = {
        "x": alt.X(f"{x_col}:Q", title=x_col, scale=x_scale, axis=x_axis),
        "y": alt.Y(f"{y_col}:Q", title=y_col, scale=y_scale, axis=y_axis),
        "tooltip": [alt.Tooltip(c, title=c) for c in tooltip_cols],
    }

    color_encoding = _scatter_color_encoding(chart_df, color_col)
    if color_encoding is not None:
        enc["color"] = color_encoding

    size_encoding = _scatter_size_encoding(chart_df, size_col)
    if size_encoding is not None:
        enc["size"] = size_encoding

    mark_kwargs = {"opacity": 0.74, "stroke": "#444444", "strokeWidth": 0.45}
    if size_encoding is None:
        mark_kwargs["size"] = 85

    points = (
        alt.Chart(chart_df)
        .mark_circle(**mark_kwargs)
        .encode(**enc)
    )

    fit = _best_fit_stats(chart_df, x_col, y_col)
    chart = points
    if fit is not None:
        fit_line = (
            alt.Chart(fit["line_df"])
            .mark_line(color="#111111", strokeWidth=2.5, strokeDash=[8, 5])
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_scale),
                y=alt.Y(f"{y_col}:Q", scale=y_scale),
            )
        )
        chart = points + fit_line

    chart = chart.interactive().properties(height=520)
    st.altair_chart(chart, width="stretch")

    if fit is not None:
        sign = "+" if fit["intercept"] >= 0 else "-"
        equation = f"{y_col} = {_format_equation_number(fit['slope'])} × {x_col} {sign} {_format_equation_number(abs(fit['intercept']))}"
        m1, m2, m3 = st.columns(3)
        m1.metric("Correlation (r)", f"{fit['corr']:.3f}")
        m2.metric("R²", f"{fit['r2']:.3f}")
        m3.metric("Rows Used", f"{fit['n']:,}")
        st.caption(f"Best-fit line: {equation}")
    else:
        st.caption("Best-fit line unavailable because there are not enough valid numeric points or one axis has no variation.")

def clean_feature_name(feature):
    """Make model feature names readable for the UI."""
    text = str(feature)
    replacements = {
        "age_entering_year": "Age",
        "hist_G_total": "Recent Games",
        "hist_AB_total": "Recent AB",
        "_mean_3yr": " 3-Year Avg",
        "_mean_4yr": " 4-Year Avg",
        "_mean_5yr": " 5-Year Avg",
        "_last": " Last Season",
        "_trend": " Trend",
        "yearID": "Year",
        "fullName": "Player",
        "bats": "Bats",
        "teamID": "Team",
        "playerID": "Player",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("_", " ").strip()
    return text

def clean_ui_columns(df):
    """Final safeguard so backend ID-style columns never appear in displayed tables."""
    df = df.copy()
    rename_map = {
        "yearID": "Year",
        "fullName": "Player",
        "bats": "Bats",
        "primaryPos": "Primary Position",
        "primaryHistoricalTeamName": "Team",
        "prediction_year": "Prediction Year",
        "predict_year": "Prediction Year",
        "last_year": "Last Year",
        "age_entering_year": "Age",
        "hist_G_total": "Recent Games",
        "hist_AB_total": "Recent AB",
        "score": "Score",
        "Trend_Score": "Trend Score",
        "Perf_Score": "Current Score",
        "Valuation_Score": "Valuation Score",
    }
    df = df.rename(columns=rename_map)
    drop_cols = [c for c in df.columns if str(c).lower().endswith("id") or "playerid" in str(c).lower() or "teamid" in str(c).lower()]
    return df.drop(columns=drop_cols, errors="ignore")



def baseball_age_for_season(season_year, birth_year, birth_month=np.nan, birth_day=np.nan):
    """Approximate MLB season age using July 1 of the season, not simply season_year - birth_year.
    This prevents late-year birthdays from being overstated by one year.
    """
    try:
        season_year = int(season_year)
        birth_year = int(float(birth_year))
    except Exception:
        return np.nan
    try:
        birth_month = int(float(birth_month))
        birth_day = int(float(birth_day))
    except Exception:
        birth_month, birth_day = 7, 1
    age = season_year - birth_year
    if (birth_month, birth_day) > (7, 1):
        age -= 1
    return age

def add_latest_and_projection_columns(base_df, recent_data):
    df = base_df.copy()
    latest_stats = (
        recent_data.sort_values(["playerID", "yearID"])
        .groupby("playerID")
        .tail(1)[["playerID", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]]
        .rename(columns={
            "R": "latest_R", "H": "latest_H", "2B": "latest_2B", "3B": "latest_3B",
            "HR": "latest_HR", "RBI": "latest_RBI", "SB": "latest_SB", "BB": "latest_BB",
            "BA": "latest_BA", "OBP": "latest_OBP", "SLG": "latest_SLG", "OPS": "latest_OPS"
        })
    )
    df = df.merge(latest_stats, on="playerID", how="left")
    df["XBH_noHR_trend"] = pd.to_numeric(df["2B_trend"], errors="coerce").fillna(0) + pd.to_numeric(df["3B_trend"], errors="coerce").fillna(0)
    df["latest_XBH_noHR"] = pd.to_numeric(df["latest_2B"], errors="coerce").fillna(0) + pd.to_numeric(df["latest_3B"], errors="coerce").fillna(0)
    df["proj_R"] = df["latest_R"] + df["R_trend"]
    df["proj_H"] = df["latest_H"] + df["H_trend"]
    df["proj_XBH"] = df["latest_XBH_noHR"] + df["XBH_noHR_trend"]
    df["proj_HR"] = df["latest_HR"] + df["HR_trend"]
    df["proj_RBI"] = df["latest_RBI"] + df["RBI_trend"]
    df["proj_SB"] = df["latest_SB"] + df["SB_trend"]
    df["proj_BB"] = df["latest_BB"] + df["BB_trend"]
    df["proj_BA"] = df["latest_BA"] + df["BA_trend"]
    df["proj_OBP"] = df["latest_OBP"] + df["OBP_trend"]
    df["proj_SLG"] = df["latest_SLG"] + df["SLG_trend"]
    df["proj_OPS"] = df["latest_OPS"] + df["OPS_trend"]
    return df


@st.cache_data(show_spinner=False)
def prepare_ml_yearly_source(yearly_source):
    """Add ML-ready context and derived features once, then cache it.

    This function is intentionally cached because it is used by both training-row creation
    and current-player projection creation.
    """
    df = yearly_source.copy()
    for col in ["yearID", "birthYear", "birthMonth", "birthDay"] + ML_BASE_FEATURE_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["PA_est"] = df["AB"] + df["BB"]
    safe_pa = df["PA_est"].replace(0, np.nan)
    safe_ab = df["AB"].replace(0, np.nan)
    steal_attempts = (df["SB"] + df["CS"]).replace(0, np.nan)
    df["BB_rate"] = (df["BB"] / safe_pa).fillna(0)
    df["K_rate"] = (df["SO"] / safe_pa).fillna(0)
    df["SB_rate"] = (df["SB"] / steal_attempts).fillna(0)
    df["XBH"] = df["2B"] + df["3B"] + df["HR"]
    df["XBH_rate"] = (df["XBH"] / safe_ab).fillna(0)
    df["HR_rate"] = (df["HR"] / safe_ab).fillna(0)
    df["Speed_Index"] = (df["SB"] + 2 * df["3B"]) / df["G"].replace(0, np.nan)
    df["Speed_Index"] = df["Speed_Index"].fillna(0)
    df["bats"] = df.get("bats", "Unknown").fillna("Unknown").replace({"": "Unknown"})
    df["primaryPos"] = df.get("primaryPos", "DH").fillna("DH").replace({"": "DH"})
    df["careerPrimaryPos"] = df.get("careerPrimaryPos", df["primaryPos"]).fillna(df["primaryPos"]).replace({"": "DH"})
    df["primaryTeamID"] = df.get("primaryTeamID", "UNK").fillna("UNK").replace({"": "UNK"})
    df["League"] = df["primaryTeamID"].apply(team_league)
    df["Park_Factor"] = df["primaryTeamID"].map(TEAM_PARK_FACTOR).fillna(1.0)
    return df.sort_values(["playerID", "yearID"]).reset_index(drop=True)


def add_context_dummy_features(row, source_row):
    """Add low-cardinality categorical features as numeric dummy variables."""
    bats = str(source_row.get("bats", "Unknown") or "Unknown")
    pos = str(source_row.get("primaryPos", "DH") or "DH")
    league = str(source_row.get("League", "Unknown") or "Unknown")
    team = str(source_row.get("primaryTeamID", "UNK") or "UNK")
    for b in ["L", "R", "B", "Unknown"]:
        row[f"bats_{b}"] = 1 if bats == b else 0
    for p in ["C", "1B", "2B", "3B", "SS", "OF", "P", "DH"]:
        row[f"pos_{p}"] = 1 if pos == p else 0
    for lg in ["AL", "NL", "Unknown"]:
        row[f"league_{lg}"] = 1 if league == lg else 0
    # Keep team as broad context without allowing hundreds of sparse old team codes.
    for t in sorted(AL_TEAMS | NL_TEAMS):
        row[f"team_{t}"] = 1 if team == t else 0
    row["Park_Factor"] = pd.to_numeric(source_row.get("Park_Factor", 1.0), errors="coerce")
    return row


@st.cache_data(show_spinner=False)
def build_ml_training_set(yearly_source, lookback_years=3, min_games_per_window=50, target_stats_tuple=tuple(ML_TARGET_STATS)):
    """Create supervised learning rows: last N years of features -> following-year stats.

    Added features include age/age², bats, primary position, team, league, park factor,
    recent stats, rolling means, trend slopes, playing time, walk rate, strikeout rate,
    OPS, and speed/durability proxies.
    """
    target_stats = list(target_stats_tuple)
    df = prepare_ml_yearly_source(yearly_source)
    rows = []
    all_feature_stats = ML_BASE_FEATURE_STATS + ML_DERIVED_FEATURE_STATS
    for player_id, g in df.groupby("playerID", sort=False):
        g = g.sort_values("yearID").reset_index(drop=True)
        for idx in range(lookback_years, len(g)):
            history = g.iloc[idx - lookback_years:idx]
            target = g.iloc[idx]
            expected_years = list(range(int(target["yearID"]) - lookback_years, int(target["yearID"])))
            if history["yearID"].astype(int).tolist() != expected_years:
                continue
            if pd.to_numeric(history["G"], errors="coerce").sum() < min_games_per_window:
                continue
            birth_year = pd.to_numeric(target.get("birthYear", np.nan), errors="coerce")
            age = baseball_age_for_season(target["yearID"], birth_year, target.get("birthMonth", np.nan), target.get("birthDay", np.nan))
            row = {
                "playerID": player_id,
                "fullName": target.get("fullName", ""),
                "bats": target.get("bats", ""),
                "primaryPos": target.get("primaryPos", ""),
                "League": target.get("League", "Unknown"),
                "primaryTeamID": target.get("primaryTeamID", "UNK"),
                "predict_year": int(target["yearID"]),
                "last_year": int(target["yearID"]) - 1,
                "age_entering_year": age,
                "age_squared": age ** 2 if pd.notna(age) else np.nan,
                "hist_G_total": pd.to_numeric(history["G"], errors="coerce").sum(),
                "hist_AB_total": pd.to_numeric(history["AB"], errors="coerce").sum(),
                "durability_3yr_avg_G": pd.to_numeric(history["G"], errors="coerce").mean(),
                "durability_3yr_min_G": pd.to_numeric(history["G"], errors="coerce").min(),
            }
            row = add_context_dummy_features(row, target)
            # weighted recency: latest year gets the largest weight
            weights = np.arange(1, len(history) + 1, dtype=float)
            weights = weights / weights.sum()
            for stat in all_feature_stats:
                if stat not in history.columns:
                    continue
                values = pd.to_numeric(history[stat], errors="coerce").fillna(0)
                row[f"{stat}_mean_{lookback_years}yr"] = values.mean()
                row[f"{stat}_weighted_recent"] = float(np.dot(values.to_numpy(), weights))
                row[f"{stat}_last"] = values.iloc[-1]
                row[f"{stat}_trend"] = compute_trend_slope(history, stat)
            for stat in target_stats:
                row[f"target_{stat}"] = pd.to_numeric(target.get(stat, np.nan), errors="coerce")
            rows.append(row)
    ml_df = pd.DataFrame(rows)
    if ml_df.empty:
        return ml_df, []
    exclude = {"playerID", "fullName", "bats", "primaryPos", "League", "primaryTeamID", "predict_year", "last_year"}
    feature_cols = [c for c in ml_df.columns if c not in exclude and not c.startswith("target_")]
    ml_df[feature_cols] = ml_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    target_cols = [f"target_{stat}" for stat in target_stats]
    ml_df[target_cols] = ml_df[target_cols].apply(pd.to_numeric, errors="coerce")
    ml_df = ml_df.dropna(subset=target_cols, how="all")
    return ml_df, feature_cols


@st.cache_resource(show_spinner=False)
def train_random_forest_models(ml_training_df, feature_cols_tuple, target_stats_tuple=tuple(ML_TARGET_STATS), random_state=42):
    """Train compact Random Forest models once and cache them for Streamlit Cloud speed."""
    target_stats = list(target_stats_tuple)
    feature_cols = list(feature_cols_tuple)
    results = {}
    if ml_training_df.empty or not feature_cols:
        return results
    X = ml_training_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    for stat in target_stats:
        target_col = f"target_{stat}"
        if target_col not in ml_training_df.columns:
            continue
        y = pd.to_numeric(ml_training_df[target_col], errors="coerce")
        valid = y.notna()
        if valid.sum() < 40:
            continue
        X_valid = X.loc[valid]
        y_valid = y.loc[valid]
        if len(X_valid) >= 80:
            X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.25, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = X_valid, X_valid, y_valid, y_valid
        model = RandomForestRegressor(
            n_estimators=12,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        importances = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        results[stat] = {
            "model": model,
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)) if len(y_test) > 1 else np.nan,
            "importance": importances,
        }
    return results


@st.cache_data(show_spinner=False)
def build_current_prediction_rows(yearly_source, lookback_years=3, min_games_per_window=50, max_player_pool=300):
    """Create one current row per active/recent player using each player's true max(yearID).

    Optimized for Streamlit Cloud:
    - only active/recent players are projected
    - optional cap on player pool, sorted by recent AB
    - feature construction mirrors training features exactly
    """
    df = prepare_ml_yearly_source(yearly_source)
    df = df.dropna(subset=["playerID", "yearID"]).copy()
    df["yearID"] = df["yearID"].astype(int)
    max_data_year = int(df["yearID"].max())
    rows = []
    all_feature_stats = ML_BASE_FEATURE_STATS + ML_DERIVED_FEATURE_STATS
    for player_id, g in df.groupby("playerID", sort=False):
        g = g.sort_values("yearID").reset_index(drop=True)
        latest_year = int(g["yearID"].max())
        if latest_year < max_data_year - 1:
            continue
        latest = g[g["yearID"] == latest_year].iloc[0]
        history = g[g["yearID"] <= latest_year].tail(lookback_years).copy()
        if len(history) < lookback_years:
            continue
        expected_years = list(range(latest_year - lookback_years + 1, latest_year + 1))
        if history["yearID"].astype(int).tolist() != expected_years:
            continue
        if pd.to_numeric(history["G"], errors="coerce").sum() < min_games_per_window:
            continue
        birth_year = pd.to_numeric(latest.get("birthYear", np.nan), errors="coerce")
        age = baseball_age_for_season(latest_year + 1, birth_year, latest.get("birthMonth", np.nan), latest.get("birthDay", np.nan))
        row = {
            "playerID": player_id,
            "fullName": latest.get("fullName", ""),
            "bats": latest.get("bats", ""),
            "primaryPos": latest.get("primaryPos", ""),
            "League": latest.get("League", "Unknown"),
            "primaryTeamID": latest.get("primaryTeamID", "UNK"),
            "last_year": latest_year,
            "prediction_year": latest_year + 1,
            "age_entering_year": age,
            "age_squared": age ** 2 if pd.notna(age) else np.nan,
            "hist_G_total": pd.to_numeric(history["G"], errors="coerce").sum(),
            "hist_AB_total": pd.to_numeric(history["AB"], errors="coerce").sum(),
            "durability_3yr_avg_G": pd.to_numeric(history["G"], errors="coerce").mean(),
            "durability_3yr_min_G": pd.to_numeric(history["G"], errors="coerce").min(),
        }
        row = add_context_dummy_features(row, latest)
        for stat in ML_BASE_FEATURE_STATS:
            row[f"Last {stat}"] = pd.to_numeric(latest.get(stat, np.nan), errors="coerce")
        weights = np.arange(1, len(history) + 1, dtype=float)
        weights = weights / weights.sum()
        for stat in all_feature_stats:
            if stat not in history.columns:
                continue
            values = pd.to_numeric(history[stat], errors="coerce").fillna(0)
            row[f"{stat}_mean_{lookback_years}yr"] = values.mean()
            row[f"{stat}_weighted_recent"] = float(np.dot(values.to_numpy(), weights))
            row[f"{stat}_last"] = values.iloc[-1]
            row[f"{stat}_trend"] = compute_trend_slope(history, stat)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and max_player_pool:
        out = out.sort_values("hist_AB_total", ascending=False).head(int(max_player_pool)).reset_index(drop=True)
    return out


def get_target_baselines(ml_training_df, target_stats):
    baselines = {}
    for stat in target_stats:
        col = f"target_{stat}"
        if col in ml_training_df.columns:
            baselines[stat] = pd.to_numeric(ml_training_df[col], errors="coerce").mean()
    return baselines


@st.cache_data(show_spinner=False)
def get_age_curve_adjustments(ml_training_df, target_stats_tuple=tuple(ML_TARGET_STATS)):
    """Estimate aging effects from historical training rows."""
    target_stats = list(target_stats_tuple)
    rows = []
    if ml_training_df.empty or "age_entering_year" not in ml_training_df.columns:
        return pd.DataFrame(columns=["Stat", "Age", "Age Adjustment"])
    tmp = ml_training_df.copy()
    tmp["age_bucket"] = pd.to_numeric(tmp["age_entering_year"], errors="coerce").round().clip(18, 45)
    for stat in target_stats:
        target_col = f"target_{stat}"
        last_col = f"{stat}_last"
        if target_col not in tmp.columns or last_col not in tmp.columns:
            continue
        stat_tmp = tmp[["age_bucket", target_col, last_col]].copy()
        stat_tmp[target_col] = pd.to_numeric(stat_tmp[target_col], errors="coerce")
        stat_tmp[last_col] = pd.to_numeric(stat_tmp[last_col], errors="coerce")
        stat_tmp["delta"] = stat_tmp[target_col] - stat_tmp[last_col]
        stat_tmp = stat_tmp.dropna(subset=["age_bucket", "delta"])
        grouped = stat_tmp.groupby("age_bucket")["delta"].agg(["mean", "count"]).reset_index()
        grouped = grouped[grouped["count"] >= 10].sort_values("age_bucket")
        if grouped.empty:
            continue
        grouped["smoothed"] = grouped["mean"].rolling(window=3, min_periods=1, center=True).mean()
        for r in grouped.itertuples(index=False):
            rows.append({"Stat": stat, "Age": int(r.age_bucket), "Age Adjustment": float(r.smoothed)})
    return pd.DataFrame(rows)


def lookup_age_adjustment(age_curve_df, stat, age):
    if age_curve_df.empty or pd.isna(age):
        return 0.0
    stat_curve = age_curve_df[age_curve_df["Stat"] == stat].copy()
    if stat_curve.empty:
        return 0.0
    age = int(round(float(age)))
    stat_curve["age_distance"] = (stat_curve["Age"] - age).abs()
    return float(stat_curve.sort_values("age_distance").iloc[0]["Age Adjustment"])


@st.cache_data(show_spinner=False)
def build_base_ml_predictions(yearly_source, lookback_years, min_games_per_window, max_player_pool=300):
    """Train once, predict once, and return reusable base objects for fast UI filtering."""
    target_stats_tuple = tuple(ML_TARGET_STATS)
    ml_training_df, feature_cols = build_ml_training_set(yearly_source, lookback_years, min_games_per_window, target_stats_tuple)
    if ml_training_df.empty or not feature_cols:
        return ml_training_df, [], {}, pd.DataFrame(), pd.DataFrame()
    feature_cols_tuple = tuple(feature_cols)
    ml_models = train_random_forest_models(ml_training_df, feature_cols_tuple, target_stats_tuple)
    current_rows = build_current_prediction_rows(yearly_source, lookback_years, min_games_per_window, max_player_pool=max_player_pool)
    if current_rows.empty:
        return ml_training_df, feature_cols, ml_models, current_rows, pd.DataFrame()
    X_current = current_rows.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
    base_pred_cols = ["playerID", "fullName", "bats", "primaryPos", "League", "primaryTeamID", "last_year", "prediction_year", "age_entering_year", "hist_G_total", "hist_AB_total"]
    last_audit_cols = [f"Last {s}" for s in ML_BASE_FEATURE_STATS if f"Last {s}" in current_rows.columns]
    pred_df = current_rows[[c for c in base_pred_cols + last_audit_cols if c in current_rows.columns]].copy()
    for stat, info in ml_models.items():
        pred_df[f"Raw ML {stat}"] = info["model"].predict(X_current)
    return ml_training_df, feature_cols, ml_models, current_rows, pred_df


@st.cache_data(show_spinner=False)
def build_similar_player_predictions(current_rows, ml_training_df, feature_cols_tuple, target_stats_tuple=tuple(ML_TARGET_STATS), k_neighbors=25, max_age_gap=3):
    """Fast nearest-neighbor comps. Excludes the target player from his own comps."""
    feature_cols = list(feature_cols_tuple)
    target_stats = list(target_stats_tuple)
    if current_rows.empty or ml_training_df.empty or not feature_cols:
        return pd.DataFrame()
    # Use a smaller, high-signal subset for similarity to avoid slow/noisy hundreds-column distances.
    preferred = [
        "age_entering_year", "age_squared", "hist_G_total", "hist_AB_total", "Park_Factor",
        "G_last", "AB_last", "HR_last", "RBI_last", "SB_last", "BB_last", "SO_last", "OPS_last", "BA_last", "OBP_last", "SLG_last",
        "BB_rate_last", "K_rate_last", "HR_rate_last", "XBH_rate_last", "Speed_Index_last",
        "HR_trend", "OPS_trend", "SB_trend", "BB_rate_trend", "K_rate_trend", "Speed_Index_trend",
    ]
    sim_cols = [c for c in preferred if c in feature_cols]
    if len(sim_cols) < 5:
        sim_cols = feature_cols[:30]
    train_X = ml_training_df.reindex(columns=sim_cols).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    current_X = current_rows.reindex(columns=sim_cols).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    means = train_X.mean(axis=0)
    stds = train_X.std(axis=0).replace(0, 1)
    train_Z = ((train_X - means) / stds).to_numpy()
    current_Z = ((current_X - means) / stds).to_numpy()
    train_ages = pd.to_numeric(ml_training_df.get("age_entering_year", np.nan), errors="coerce").to_numpy()
    current_ages = pd.to_numeric(current_rows.get("age_entering_year", np.nan), errors="coerce").to_numpy()
    train_player_ids = ml_training_df.get("playerID", pd.Series([""] * len(ml_training_df))).astype(str).to_numpy()
    out_rows = []
    current_reset = current_rows.reset_index(drop=True)
    for i, row in current_reset.iterrows():
        diff = train_Z - current_Z[i]
        distances = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        age = current_ages[i] if i < len(current_ages) else np.nan
        candidate_mask = np.ones(len(distances), dtype=bool)
        if pd.notna(row.get("playerID")):
            candidate_mask &= (train_player_ids != str(row.get("playerID")))
        if pd.notna(age):
            age_mask = np.abs(train_ages - age) <= max_age_gap
            if (candidate_mask & age_mask).sum() >= max(k_neighbors, 10):
                candidate_mask &= age_mask
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            continue
        # Pick nearest UNIQUE comparable players. The training set can contain multiple
        # seasons for the same historical player, so a simple top-k can repeat names.
        # Keep only the closest season for each comparable player.
        sorted_candidates = candidate_idx[np.argsort(distances[candidate_idx])]
        unique_nearest = []
        seen_comp_players = set()
        target_pid = str(row.get("playerID")) if pd.notna(row.get("playerID")) else ""
        for idx in sorted_candidates:
            comp_pid = str(train_player_ids[idx]) if idx < len(train_player_ids) else ""
            if not comp_pid or comp_pid == target_pid or comp_pid in seen_comp_players:
                continue
            unique_nearest.append(idx)
            seen_comp_players.add(comp_pid)
            if len(unique_nearest) >= k_neighbors:
                break
        if not unique_nearest:
            continue
        comps = ml_training_df.iloc[unique_nearest]
        out = {
            "playerID": row.get("playerID"),
            "Similar Player Sample": len(comps),
        }
        for stat in target_stats:
            tcol = f"target_{stat}"
            if tcol in comps.columns:
                out[f"Similar {stat}"] = pd.to_numeric(comps[tcol], errors="coerce").mean()
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def apply_advanced_projection_adjustments(pred_df, current_rows, ml_training_df, feature_cols, target_stats,
                                          regression_strength=0.20, age_strength=0.50, comp_weight=0.25, k_neighbors=25):
    """Blend compact RF output, similar-player comps, age curve, and regression-to-the-mean."""
    if pred_df.empty:
        return pred_df, pd.DataFrame(), pd.DataFrame()
    adjusted = pred_df.copy()
    baselines = get_target_baselines(ml_training_df, target_stats)
    age_curve_df = get_age_curve_adjustments(ml_training_df, tuple(target_stats))
    if comp_weight and comp_weight > 0:
        comp_df = build_similar_player_predictions(current_rows, ml_training_df, tuple(feature_cols), tuple(target_stats), k_neighbors=k_neighbors)
    else:
        comp_df = pd.DataFrame()
    if not comp_df.empty:
        adjusted = adjusted.merge(comp_df, on="playerID", how="left")
    else:
        adjusted["Similar Player Sample"] = np.nan
    for stat in target_stats:
        rf_col = f"Raw ML {stat}"
        final_col = f"Predicted {stat}"
        comp_col = f"Similar {stat}"
        if rf_col not in adjusted.columns:
            continue
        rf_pred = pd.to_numeric(adjusted[rf_col], errors="coerce")
        baseline = baselines.get(stat, rf_pred.mean())
        if comp_col in adjusted.columns:
            comp_pred = pd.to_numeric(adjusted[comp_col], errors="coerce")
            blended = (1 - comp_weight) * rf_pred + comp_weight * comp_pred.fillna(rf_pred)
        else:
            blended = rf_pred.copy()
        if "age_entering_year" in adjusted.columns:
            age_adj = adjusted["age_entering_year"].apply(lambda a: lookup_age_adjustment(age_curve_df, stat, a))
            blended = blended + age_strength * pd.to_numeric(age_adj, errors="coerce").fillna(0)
        recent_ab = pd.to_numeric(adjusted.get("hist_AB_total", np.nan), errors="coerce")
        reliability = (recent_ab / 1200).clip(lower=0.20, upper=1.0).fillna(0.50)
        dynamic_regression = regression_strength * (1.15 - reliability)
        final = (1 - dynamic_regression) * blended + dynamic_regression * baseline
        if stat in ["R", "H", "2B", "3B", "HR", "RBI", "SB", "BB"]:
            final = final.clip(lower=0)
        if stat in RATE_STATS:
            final = final.clip(lower=0, upper=1.5)
        adjusted[final_col] = final
    return adjusted, age_curve_df, comp_df


def make_ml_prediction_summary(row, sort_stat):
    player = row.get("Player", "This player")
    stat_val = row.get(f"Predicted {sort_stat}", np.nan)
    ops = row.get("Predicted OPS", np.nan)
    hr = row.get("Predicted HR", np.nan)
    rbi = row.get("Predicted RBI", np.nan)
    sb = row.get("Predicted SB", np.nan)
    stat_text = fmt_rate_3(stat_val) if sort_stat in RATE_STATS else fmt_int(stat_val)
    return (
        f"{player}'s advanced ML projection is strongest on {sort_stat}: {stat_text}. "
        f"The model projects about {fmt_rate_3(ops)} OPS, {fmt_int(hr)} HR, {fmt_int(rbi)} RBI, and {fmt_int(sb)} SB. "
        f"The displayed projection blends Random Forest, age/age², position, bats, league/team context, playing time, trends, similar-player history, aging curves, and regression-to-the-mean."
    )


def aggregate_player_year_team(df):
    """One row per player-year-actual team. Used when Historical Explorer shows split seasons."""
    if df.empty:
        return df.copy()
    group_cols = [
        "playerID", "fullName", "bats", "throws", "birthYear", "birthMonth", "birthDay", "birthCountry",
        "yearID", "teamID", "teamName", "teamHistoricalName", "teamLeague", "primaryPos", "careerPrimaryPos"
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    stat_cols = [c for c in ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"] if c in df.columns]
    out = df.groupby(group_cols, as_index=False)[stat_cols].sum()
    out = add_rate_stats(out)
    return out


def aggregate_player_year_primary_team(df):
    """One row per player-year, with stats combined and Team set to the primary team in the filtered data."""
    if df.empty:
        return df.copy()
    team_basis = aggregate_player_year_team(df)
    basis_col = "G" if "G" in team_basis.columns else "AB"
    primary_team = (
        team_basis.sort_values(["playerID", "yearID", basis_col, "AB"], ascending=[True, True, False, False])
        .drop_duplicates(["playerID", "yearID"])
        [["playerID", "yearID", "teamID", "teamName", "teamHistoricalName", "teamLeague"]]
        .rename(columns={"teamID": "primaryTeamID", "teamName": "primaryTeamName", "teamHistoricalName": "primaryHistoricalTeamName", "teamLeague": "primaryLeague"})
    )
    group_cols = [
        "playerID", "fullName", "bats", "throws", "birthYear", "birthMonth", "birthDay", "birthCountry", "careerPrimaryPos", "yearID"
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    stat_cols = [c for c in ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"] if c in df.columns]
    out = df.groupby(group_cols, as_index=False)[stat_cols].sum()
    out = add_rate_stats(out)
    out = out.merge(primary_team, on=["playerID", "yearID"], how="left")
    pos_basis = (
        df.groupby(["playerID", "yearID", "primaryPos"], as_index=False)["G"].sum()
        .sort_values(["playerID", "yearID", "G", "primaryPos"], ascending=[True, True, False, True])
        .drop_duplicates(["playerID", "yearID"])[["playerID", "yearID", "primaryPos"]]
    )
    out = out.merge(pos_basis, on=["playerID", "yearID"], how="left")
    return out


def add_primary_team_for_career(grouped_df, source_df):
    """Attach primary team/position to player-level career totals based on most games/AB in the filtered source."""
    if grouped_df.empty or source_df.empty:
        return grouped_df
    team_basis = aggregate_player_year_team(source_df)
    team_games = (
        team_basis.groupby(["playerID", "teamName", "teamHistoricalName"], as_index=False)[["G", "AB"]]
        .sum()
        .sort_values(["playerID", "G", "AB", "teamName"], ascending=[True, False, False, True])
        .drop_duplicates("playerID")
        [["playerID", "teamName", "teamHistoricalName"]]
        .rename(columns={"teamName": "primaryTeamName", "teamHistoricalName": "primaryHistoricalTeamName"})
    )
    season_pos_basis = (
        source_df.groupby(["playerID", "primaryPos"], as_index=False)[["G", "AB"]]
        .sum()
        .sort_values(["playerID", "G", "AB", "primaryPos"], ascending=[True, False, False, True])
        .drop_duplicates("playerID")
        [["playerID", "primaryPos"]]
    )
    career_pos_basis = source_df[["playerID", "careerPrimaryPos"]].drop_duplicates("playerID") if "careerPrimaryPos" in source_df.columns else pd.DataFrame(columns=["playerID", "careerPrimaryPos"])
    return (
        grouped_df
        .merge(team_games, on="playerID", how="left")
        .merge(season_pos_basis, on="playerID", how="left")
        .merge(career_pos_basis, on="playerID", how="left")
    )

@st.cache_data
def load_data():
    people = read_required_csv("People.csv")
    batting = read_required_csv("Batting.csv")
    fielding = read_required_csv("Fielding.csv")

    batting["teamID_original"] = batting["teamID"]
    fielding["teamID_original"] = fielding["teamID"]
    batting["teamHistoricalName"] = batting.apply(lambda r: historical_team_name(r["teamID_original"], r.get("yearID", None)), axis=1)
    fielding["teamHistoricalName"] = fielding.apply(lambda r: historical_team_name(r["teamID_original"], r.get("yearID", None)), axis=1)

    batting["teamID"] = batting["teamID"].replace(team_id_mapping)
    fielding["teamID"] = fielding["teamID"].replace(team_id_mapping)

    keep_people = ["playerID", "nameFirst", "nameLast", "birthYear", "birthMonth", "birthDay", "birthCountry", "bats", "throws"]
    keep_people = [c for c in keep_people if c in people.columns]
    people = people[keep_people].copy()
    people["nameFirst"] = people["nameFirst"].fillna("").astype(str).str.strip()
    people["nameLast"] = people["nameLast"].fillna("").astype(str).str.strip()
    people["fullName"] = (people["nameFirst"] + " " + people["nameLast"]).str.strip()

    batting_num_cols = ["yearID", "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]
    for col in batting_num_cols:
        if col in batting.columns:
            batting[col] = pd.to_numeric(batting[col], errors="coerce").fillna(0)

    if "yearID" in fielding.columns:
        fielding["yearID"] = pd.to_numeric(fielding["yearID"], errors="coerce").fillna(0)
    if "G" in fielding.columns:
        fielding["G"] = pd.to_numeric(fielding["G"], errors="coerce").fillna(0)
    else:
        fielding["G"] = 0

    # Group LF/CF/RF together so outfielders are classified as OF instead of splitting their games.
    fielding["POS_grouped"] = fielding["POS"].replace({"LF": "OF", "CF": "OF", "RF": "OF"})
    valid_primary_positions = ["C", "1B", "2B", "3B", "SS", "OF", "P", "DH"]
    fielding_for_pos = fielding[fielding["POS_grouped"].isin(valid_primary_positions)].copy()

    fielding_counts = (
        fielding_for_pos
        .groupby(["playerID", "yearID", "POS_grouped"], as_index=False)["G"]
        .sum()
        .rename(columns={"G": "games_at_pos"})
    )
    # Season primary position: most fielding games at a grouped position in that specific season.
    primary_positions = (
        fielding_counts.sort_values(["playerID", "yearID", "games_at_pos", "POS_grouped"], ascending=[True, True, False, True])
        .drop_duplicates(subset=["playerID", "yearID"])
        [["playerID", "yearID", "POS_grouped"]]
        .rename(columns={"POS_grouped": "primaryPos"})
    )

    # Career primary position: most fielding games at a grouped position across the player's entire career.
    # This is the correct career-page position definition: it is based on Fielding.csv games, not at-bats.
    career_primary_positions = (
        fielding_for_pos
        .groupby(["playerID", "POS_grouped"], as_index=False)["G"]
        .sum()
        .rename(columns={"G": "career_games_at_pos"})
        .sort_values(["playerID", "career_games_at_pos", "POS_grouped"], ascending=[True, False, True])
        .drop_duplicates(subset=["playerID"])
        [["playerID", "POS_grouped"]]
        .rename(columns={"POS_grouped": "careerPrimaryPos"})
    )

    batting = batting.merge(people, on="playerID", how="left")
    batting = batting.merge(primary_positions, on=["playerID", "yearID"], how="left")
    batting = batting.merge(career_primary_positions, on="playerID", how="left")
    batting["primaryPos"] = batting["primaryPos"].fillna("DH")
    batting["careerPrimaryPos"] = batting["careerPrimaryPos"].fillna(batting["primaryPos"]).fillna("DH")
    batting["teamName"] = batting["teamID"].map(team_id_to_name).fillna(batting["teamID"])
    batting["teamLeague"] = batting.apply(lambda r: team_league(r["teamID"], r["yearID"]), axis=1)
    batting = add_rate_stats(batting)

    yearly = (
        batting.groupby(["playerID", "fullName", "bats", "throws", "birthYear", "birthMonth", "birthDay", "birthCountry", "careerPrimaryPos", "yearID"], as_index=False)
        [["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]]
        .sum()
    )
    yearly = add_rate_stats(yearly)

    year_team_totals = batting.groupby(["playerID", "yearID", "teamID", "teamHistoricalName", "teamLeague"], as_index=False).agg({"AB": "sum"})
    primary_teams = (
        year_team_totals.sort_values(["playerID", "yearID", "AB"], ascending=[True, True, False])
        .drop_duplicates(subset=["playerID", "yearID"])
        [["playerID", "yearID", "teamID", "teamHistoricalName", "teamLeague"]]
        .rename(columns={"teamID": "primaryTeamID", "teamHistoricalName": "primaryHistoricalTeamName", "teamLeague": "primaryLeague"})
    )
    yearly = yearly.merge(primary_teams, on=["playerID", "yearID"], how="left")
    yearly["primaryTeamName"] = yearly["primaryTeamID"].map(team_id_to_name).fillna(yearly["primaryTeamID"])
    yearly["primaryHistoricalTeamName"] = yearly["primaryHistoricalTeamName"].fillna(yearly["primaryTeamName"])
    yearly["primaryLeague"] = yearly["primaryLeague"].fillna(yearly.apply(lambda r: team_league(r["primaryTeamID"], r["yearID"]), axis=1))

    yearly_pos = batting[["playerID", "yearID", "primaryPos"]].drop_duplicates(subset=["playerID", "yearID"])
    yearly = yearly.merge(yearly_pos, on=["playerID", "yearID"], how="left")
    yearly["primaryPos"] = yearly["primaryPos"].fillna("DH")
    yearly["careerPrimaryPos"] = yearly["careerPrimaryPos"].fillna(yearly["primaryPos"]).fillna("DH")
    yearly = yearly[~yearly["primaryPos"].isin(["PH", "PR"])]

    return batting, yearly, people

batting_df, yearly_df, people_df = load_data()
all_years = sorted(pd.to_numeric(yearly_df["yearID"], errors="coerce").dropna().astype(int).unique())
year_min = int(min(all_years))
year_max = int(max(all_years))
default_start_hist = max(year_min, 2010)
default_start_leaders = max(year_min, 2020)

PAGE_OPTIONS = ["Historical Explorer", "Career Totals", "Leaderboards", "Comparison Tool", "Trend Value", "Valuation", "ML Predictions"]

# Persist navigation and page-specific widget settings.
# IMPORTANT: Do not manually reassign widget keys in st.session_state.
# Streamlit forbids programmatic assignment for button/download_button widgets.
# Stable widget keys plus radio-style page navigation preserve filters and charts
# when moving between pages during the same session.

st.session_state.setdefault("active_page", "Historical Explorer")
active_page = st.sidebar.radio("Choose Page", PAGE_OPTIONS, key="active_page")
st.sidebar.caption("Speed note: using page navigation instead of tabs prevents Streamlit from recalculating every page after each filter change.")
st.sidebar.caption("Filters are remembered when you move between pages during the same app session.")

if active_page == "Historical Explorer":
    render_section_header(
        "🔎 Historical Explorer",
        "Find individual player seasons. Split-team seasons can stay as separate team rows or be combined into one primary-team season row."
    )
    c1, c2, c_mode, c3, c4 = st.columns([1.05, 1.0, 1.25, 1.0, 1.35])
    with c1:
        hist_year_range = st.slider("Year Range", year_min, year_max, (default_start_hist, year_max), key="hist_year")
    with c2:
        bats_options = sorted([x for x in batting_df["bats"].dropna().unique() if str(x).strip() != ""])
        hist_bats = st.multiselect("Batting Hand", bats_options, default=bats_options, key="hist_bats")
    with c_mode:
        hist_position_filter_mode = st.selectbox(
            "Position Filter Mode",
            ["Season Primary Position", "Career Primary Position"],
            index=0,
            key="hist_position_filter_mode",
            help="Season mode filters by the player’s primary position for that season. Career mode filters by the player’s full-career primary position from Fielding.csv games."
        )
    hist_position_source_col = "careerPrimaryPos" if hist_position_filter_mode == "Career Primary Position" else "primaryPos"
    with c3:
        pos_options = sorted([x for x in batting_df[hist_position_source_col].dropna().unique() if str(x).strip() != "" and x not in ["PH", "PR"]])
        hist_pos = st.multiselect("Primary Position", pos_options, default=pos_options, key="hist_pos")
    with c4:
        actual_team_names = sorted(set(batting_df["teamName"].dropna().astype(str)).intersection(set(team_id_to_name.values())))
        hist_team_options = ["All Teams", "American League", "National League"] + actual_team_names
        hist_teams = st.multiselect("Franchise / League", hist_team_options, default=["All Teams"], key="hist_team")

    combine_split_seasons = st.toggle(
        "Combine split-team seasons into one primary-team row",
        value=False,
        key="hist_combine_split_seasons",
        help="OFF = one row per player/year/team. ON = one row per player/year, with Team assigned to the team where he had the most games/AB in that season."
    )

    hist_source = batting_df[(batting_df["yearID"] >= hist_year_range[0]) & (batting_df["yearID"] <= hist_year_range[1])].copy()
    if hist_bats:
        hist_source = hist_source[hist_source["bats"].isin(hist_bats)]
    if hist_pos:
        hist_source = hist_source[hist_source[hist_position_source_col].isin(hist_pos)]

    hist_selected_all = (not hist_teams) or ("All Teams" in hist_teams)
    hist_selected_franchises = [x for x in hist_teams if x not in ["All Teams", "American League", "National League"]]
    hist_selected_leagues = []
    if "American League" in hist_teams:
        hist_selected_leagues.append("AL")
    if "National League" in hist_teams:
        hist_selected_leagues.append("NL")

    # Split-team mode: apply league/franchise filter to each actual displayed team row.
    if not combine_split_seasons and not hist_selected_all:
        hist_team_mask = pd.Series(False, index=hist_source.index)
        if hist_selected_franchises:
            hist_team_mask = hist_team_mask | hist_source["teamName"].isin(hist_selected_franchises)
        if hist_selected_leagues:
            hist_team_mask = hist_team_mask | hist_source["teamLeague"].isin(hist_selected_leagues)
        hist_source = hist_source[hist_team_mask]

    if combine_split_seasons:
        hist = aggregate_player_year_primary_team(hist_source)
        team_col_for_display = "primaryHistoricalTeamName"
        team_sort_col = "primaryTeamName"
        hist_note = "Combined mode: one row per player-season. Team is the primary team for that season. League filters use that primary team."
        if not hist_selected_all and not hist.empty:
            hist_team_mask = pd.Series(False, index=hist.index)
            if hist_selected_franchises:
                hist_team_mask = hist_team_mask | hist["primaryTeamName"].isin(hist_selected_franchises)
            if hist_selected_leagues and "primaryLeague" in hist.columns:
                hist_team_mask = hist_team_mask | hist["primaryLeague"].isin(hist_selected_leagues)
            hist = hist[hist_team_mask]
    else:
        hist = aggregate_player_year_team(hist_source)
        team_col_for_display = "teamHistoricalName"
        team_sort_col = "teamName"
        hist_note = "Split mode: one row per player-season-team. Split seasons stay separate."

    if hist_position_source_col in hist.columns:
        hist["displayPosition"] = hist[hist_position_source_col]
    elif "primaryPos" in hist.columns:
        hist["displayPosition"] = hist["primaryPos"]
    else:
        hist["displayPosition"] = ""

    hist = apply_stat_min_filters(hist, "hist")
    hist = safe_round_rate_stats(hist)
    st.caption(hist_note)

    c5, c6 = st.columns(2)
    # Keep Historical Explorer sorting focused on baseball statistics only.
    # Do not expose backend/name/team/position fields in the sort dropdown.
    sort_options_hist = [
        "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"
    ]
    with c5:
        hist_sort_stat = st.selectbox(
            "Sort Historical Explorer By",
            sort_options_hist,
            index=sort_options_hist.index("HR"),
            key="hist_sort_stat"
        )
    with c6:
        hist_sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"], index=0, key="hist_sort_order")

    display_cols_hist = [
        "yearID", "fullName", "bats", "displayPosition", team_col_for_display,
        "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"
    ]
    display_cols_hist = [c for c in display_cols_hist if c in hist.columns]
    hist_display_raw = hist[display_cols_hist].copy()
    if hist_sort_stat in hist_display_raw.columns:
        hist_display_raw = hist_display_raw.sort_values(by=hist_sort_stat, ascending=(hist_sort_order == "Ascending"), na_position="last")

    top_bar_chart(hist_display_raw, "fullName", hist_sort_stat, f"Top 10 Seasons by {hist_sort_stat}")

    c7, c8, c9 = st.columns(3)
    c7.metric("Rows Returned", len(hist_display_raw))
    top_value = pd.to_numeric(hist_display_raw[hist_sort_stat], errors="coerce").max() if len(hist_display_raw) and hist_sort_stat in hist_display_raw.columns else 0
    c8.metric("Top Stat Value", fmt_rate_3(top_value) if hist_sort_stat in RATE_STATS else (fmt_int(top_value) if hist_sort_stat in COUNT_STATS or hist_sort_stat == "yearID" else str(top_value)))
    c9.metric("Year Range", f"{hist_year_range[0]}-{hist_year_range[1]}")

    hist_display = hist_display_raw.rename(columns={
        "yearID": "Year", "fullName": "Player", "bats": "Bats", "displayPosition": "Primary Position",
        team_col_for_display: "Team"
    })
    st.divider()
    hist_table = format_display_table(clean_ui_columns(hist_display), count_cols=["Year", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"])
    render_output_table(hist_table, key="historical_explorer", file_name="historical_explorer.csv")
    st.divider()
    hist_plot_df = _prepare_historical_scatter_data(hist, team_col_for_display)
    render_scatterplot_section(hist_plot_df, key_prefix="hist", title="Visualize Historical Results")

if active_page == "Career Totals":
    render_section_header(
        "📚 Career Totals",
        "Aggregate career production with an independent display toggle: one primary-team career row or separate totals by each team."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        range_career = st.slider("Select Year Range", year_min, year_max, (max(year_min, 2010), year_max), key="career_year")
    with c2:
        bats_options_career = sorted([x for x in batting_df["bats"].dropna().unique() if str(x).strip() != ""])
        bats_filter_career = st.multiselect("Batting Hand", bats_options_career, default=bats_options_career, key="career_bats")
    with c3:
        position_filter_mode = st.selectbox(
            "Position Filter Mode",
            ["Career Primary Position", "Season Primary Position"],
            index=0,
            key="career_position_filter_mode",
            help="Career mode uses each player's full-career primary position from Fielding.csv games. Season mode includes only seasons where the selected position was that player's primary position that year."
        )
    with c4:
        position_source_col = "careerPrimaryPos" if position_filter_mode == "Career Primary Position" else "primaryPos"
        pos_options_career = sorted([x for x in batting_df[position_source_col].dropna().unique() if str(x).strip() != "" and x not in ["PH", "PR"]])
        pos_filter_career = st.multiselect("Position", pos_options_career, default=pos_options_career, key="career_pos")
    with c5:
        actual_team_names_career = sorted(set(batting_df["teamName"].dropna().astype(str)).intersection(set(team_id_to_name.values())))
        team_options_career = ["All Teams", "American League", "National League"] + actual_team_names_career
        team_filter_career = st.multiselect("Franchise / League", team_options_career, default=["All Teams"], key="career_team")

    show_career_by_team = st.toggle(
        "Show career totals separately by team",
        value=False,
        key="career_by_team_toggle",
        help="OFF = one row per player with a Primary Team. ON = one row per player/team, and stat minimums are applied to each team row separately."
    )

    filtered_career = batting_df[(batting_df["yearID"] >= range_career[0]) & (batting_df["yearID"] <= range_career[1])].copy()
    if bats_filter_career:
        filtered_career = filtered_career[filtered_career["bats"].isin(bats_filter_career)]
    if pos_filter_career:
        # Position filtering is explicitly based on fielding games, not at-bats.
        # Career Primary Position = most games at a grouped position over the full career.
        # Season Primary Position = most games at a grouped position in that season.
        filtered_career = filtered_career[filtered_career[position_source_col].isin(pos_filter_career)]
    career_selected_all = (not team_filter_career) or ("All Teams" in team_filter_career)
    career_selected_franchises = [x for x in team_filter_career if x not in ["All Teams", "American League", "National League"]]
    career_selected_leagues = []
    if "American League" in team_filter_career:
        career_selected_leagues.append("AL")
    if "National League" in team_filter_career:
        career_selected_leagues.append("NL")

    # League/franchise filtering changes the DATA included.
    # The by-team toggle only changes the DISPLAY structure after this filter is applied.
    if not career_selected_all:
        career_team_mask = pd.Series(False, index=filtered_career.index)
        if career_selected_franchises:
            career_team_mask = career_team_mask | filtered_career["teamName"].isin(career_selected_franchises)
        if career_selected_leagues and "teamLeague" in filtered_career.columns:
            career_team_mask = career_team_mask | filtered_career["teamLeague"].isin(career_selected_leagues)
        filtered_career = filtered_career[career_team_mask]

    stat_cols_career = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF", "G"]
    stat_cols_career = [c for c in stat_cols_career if c in filtered_career.columns]

    if show_career_by_team:
        # Filter first, aggregate by player + actual team second, then apply stat minimum filters to each team row.
        grouped_source = aggregate_player_year_team(filtered_career)
        group_cols = ["playerID", "fullName", "bats", "teamName", "teamHistoricalName"]
        career_totals = grouped_source.groupby(group_cols, as_index=False)[stat_cols_career].sum()
        if position_filter_mode == "Career Primary Position":
            pos_mode = grouped_source[["playerID", "careerPrimaryPos"]].drop_duplicates("playerID").rename(columns={"careerPrimaryPos": "displayPosition"})
            career_totals = career_totals.merge(pos_mode, on="playerID", how="left")
        else:
            pos_mode = (
                grouped_source.groupby(["playerID", "teamName", "primaryPos"], as_index=False)[["G", "AB"]]
                .sum()
                .sort_values(["playerID", "teamName", "G", "AB", "primaryPos"], ascending=[True, True, False, False, True])
                .drop_duplicates(["playerID", "teamName"])[["playerID", "teamName", "primaryPos"]]
                .rename(columns={"primaryPos": "displayPosition"})
            )
            career_totals = career_totals.merge(pos_mode, on=["playerID", "teamName"], how="left")
        career_totals["displayTeam"] = career_totals["teamHistoricalName"]
        career_mode_note = "By-team mode: each player/team row must pass stat minimum filters on its own. Franchise/league filters first limit the data included; position is based on Fielding.csv games."
    else:
        # Filter first, aggregate by player second, then apply stat minimum filters to the total row.
        career_totals = filtered_career.groupby(["playerID", "fullName", "bats"], as_index=False)[stat_cols_career].sum()
        career_totals = add_primary_team_for_career(career_totals, filtered_career)
        if position_filter_mode == "Career Primary Position":
            career_totals["displayPosition"] = career_totals["careerPrimaryPos"] if "careerPrimaryPos" in career_totals.columns else career_totals.get("primaryPos")
        else:
            career_totals["displayPosition"] = career_totals.get("primaryPos")
        career_totals["displayTeam"] = career_totals["primaryHistoricalTeamName"]
        career_mode_note = "Total-career mode: one row per player. Franchise/league filters first limit the data included, then Team becomes the primary team within that filtered data. Position is based on Fielding.csv games."

    career_totals = add_rate_stats(career_totals)
    career_totals = apply_stat_min_filters(career_totals, "career")
    career_totals = safe_round_rate_stats(career_totals)
    st.caption(career_mode_note)

    sort_stat_career = st.selectbox("Sort By", ["HR", "RBI", "SB", "R", "H", "2B", "3B", "BB", "BA", "OBP", "SLG", "OPS", "AB"], index=0, key="career_sort")
    top_bar_chart(career_totals, "fullName", sort_stat_career, f"Top 10 Career Totals by {sort_stat_career}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Rows", len(career_totals))
    c6.metric("Top Player", career_totals.sort_values(sort_stat_career, ascending=False).iloc[0]["fullName"] if len(career_totals) and sort_stat_career in career_totals.columns else "N/A")
    top_career_value = pd.to_numeric(career_totals[sort_stat_career], errors="coerce").max() if len(career_totals) and sort_stat_career in career_totals.columns else 0
    c7.metric("Top Value", fmt_rate_3(top_career_value) if sort_stat_career in RATE_STATS else fmt_int(top_career_value))

    career_display_cols = [
        "fullName", "bats", "displayPosition", "displayTeam",
        "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"
    ]
    career_display_cols = [c for c in career_display_cols if c in career_totals.columns]
    career_display = career_totals[career_display_cols].copy()
    if sort_stat_career in career_display.columns:
        career_display = career_display.sort_values(sort_stat_career, ascending=False)
    career_display = career_display.rename(columns={
        "fullName": "Player", "bats": "Bats", "displayPosition": "Primary Position", "displayTeam": "Team"
    })
    st.divider()
    career_table = format_display_table(clean_ui_columns(career_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"])
    render_output_table(career_table, key="career_totals", file_name="career_totals.csv")
    st.divider()
    career_plot_df = _prepare_career_scatter_data(career_totals, filtered_career)
    render_scatterplot_section(career_plot_df, key_prefix="career", title="Visualize Career Results")

if active_page == "Leaderboards":
    render_section_header("🏆 Leaderboards", "Build custom offensive rankings with weighted stats, filters, summary cards, and charts.")
    c1, c2 = st.columns(2)
    with c1:
        range_leaders = st.slider("Select Year Range", year_min, year_max, (max(year_min, 2020), year_max), key="leaders_year")
    with c2:
        top_n_leaders = st.slider("Show Top N Players", 5, 100, 25, key="leaders_top_n")

    st.subheader("Custom Stat Weights")
    weight_stats = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]
    default_weights = {"HR": 1.0, "RBI": 1.0, "SB": 1.0}
    weight_values = {}
    weight_cols = st.columns(4)
    for i, stat in enumerate(weight_stats):
        with weight_cols[i % 4]:
            weight_values[stat] = st.number_input(f"Weight for {stat}", min_value=0.0, max_value=10.0, value=default_weights.get(stat, 0.0), step=0.5, key=f"leaders_w_{stat}")

    filtered_leaders = yearly_df[(yearly_df["yearID"] >= range_leaders[0]) & (yearly_df["yearID"] <= range_leaders[1])].copy()
    leaderboard = filtered_leaders.groupby(["fullName", "bats"], as_index=False)[["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]].sum()
    leaderboard = add_rate_stats(leaderboard)
    leaderboard = apply_stat_min_filters(leaderboard, "leaders")
    leaderboard = safe_round_rate_stats(leaderboard)

    leaderboard["score"] = 0.0
    for stat, weight in weight_values.items():
        leaderboard["score"] += weight * (leaderboard[stat] * 100 if stat in RATE_STATS else leaderboard[stat])

    sort_stat_leaders = st.selectbox("Sort Leaderboard By", ["score", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"], index=0, key="leaders_sort")
    top_bar_chart(leaderboard, "fullName", sort_stat_leaders, f"Top 10 by {sort_stat_leaders}")

    c12, c13, c14 = st.columns(3)
    c12.metric("Top HR", fmt_int(leaderboard["HR"].max() if not leaderboard.empty else 0))
    c13.metric("Top OPS", fmt_rate_3(leaderboard["OPS"].max() if not leaderboard.empty else 0))
    c14.metric("Average OPS", fmt_rate_3(leaderboard["OPS"].mean() if not leaderboard.empty else 0))

    leaderboard_display = leaderboard[[
        "fullName", "bats", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS", "score"
    ]].sort_values(sort_stat_leaders, ascending=False).head(top_n_leaders).rename(columns={"fullName": "Player", "bats": "Bats", "score": "Score"})

    st.divider()
    leaderboard_table = format_display_table(clean_ui_columns(leaderboard_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"], score_cols=["Score"])
    render_output_table(leaderboard_table, key="leaderboards", file_name="leaderboards.csv")

if active_page == "Comparison Tool":
    render_section_header("📈 Comparison Tool", "Compare up to three players across years with tables and trend charts.")
    label_map_compare = build_player_label_map(yearly_df)
    selected_labels_compare = st.multiselect("Select up to 3 players", options=list(label_map_compare.keys()), max_selections=3, key="compare_players")
    selected_ids_compare = [label_map_compare[label] for label in selected_labels_compare]
    stat_choice_compare = st.selectbox("Choose stat to plot", ["R", "HR", "RBI", "SB", "H", "2B", "3B", "AB", "BA", "OBP", "SLG", "OPS"], index=0, key="compare_stat")

    if selected_ids_compare:
        compare = yearly_df[yearly_df["playerID"].isin(selected_ids_compare)].copy()
        compare = safe_round_rate_stats(compare)

        st.subheader("Year-by-Year Comparison")
        compare_display = compare[["yearID", "fullName", "R", "H", "2B", "3B", "HR", "RBI", "SB", "AB", "BA", "OBP", "SLG", "OPS"]].sort_values(["fullName", "yearID"]).rename(columns={"yearID": "Year", "fullName": "Player"})
        compare_table = format_display_table(clean_ui_columns(compare_display), count_cols=["Year", "R", "H", "2B", "3B", "HR", "RBI", "SB", "AB"], rate_cols=["BA", "OBP", "SLG", "OPS"])
        render_output_table(compare_table, key="comparison_yearly", file_name="comparison_year_by_year.csv")

        st.subheader("Career Totals")
        career_compare = compare.groupby(["fullName"], as_index=False)[["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]].sum()
        career_compare = add_rate_stats(career_compare)
        career_compare = safe_round_rate_stats(career_compare)
        career_compare_display = career_compare[["fullName", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BA", "OBP", "SLG", "OPS"]].sort_values("HR", ascending=False).rename(columns={"fullName": "Player"})
        career_compare_table = format_display_table(clean_ui_columns(career_compare_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB"], rate_cols=["BA", "OBP", "SLG", "OPS"])
        render_output_table(career_compare_table, key="comparison_career", file_name="comparison_career_totals.csv")

        st.subheader(f"{stat_choice_compare} Trends")
        fig, ax = plt.subplots(figsize=(10, 5))
        for pid in selected_ids_compare:
            subset = compare[compare["playerID"] == pid].sort_values("yearID")
            if not subset.empty:
                ax.plot(subset["yearID"], subset[stat_choice_compare], marker="o", label=subset["fullName"].iloc[0])
        all_compare_years = sorted(pd.to_numeric(compare["yearID"], errors="coerce").dropna().astype(int).unique())
        ax.set_xticks(all_compare_years)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year")
        ax.set_ylabel(stat_choice_compare)
        ax.set_title(f"{stat_choice_compare} Trends")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if active_page == "Trend Value":
    render_section_header("🔥 Trend Value", "Shows only trend numbers: which stats are rising or declining per year over the selected recent window.")
    c1, c2 = st.columns(2)
    with c1:
        lag_trend = st.selectbox("Trend Window (Years)", [3, 4, 5], index=0, key="trend_lag")
    with c2:
        min_g_trend = st.number_input("Minimum Games Played", 0, 800, 50, key="trend_min_g")

    max_year_trend = int(yearly_df["yearID"].max())
    recent_years_trend = list(range(max_year_trend - lag_trend + 1, max_year_trend + 1))
    st.write(f"Analyzing seasons: **{recent_years_trend[0]}–{recent_years_trend[-1]}**")
    st.caption(f"Trend estimates are next-season estimates for **{max_year_trend + 1}**, calculated as the player's latest season value plus the yearly trend slope from the selected window.")
    recent_data_trend = yearly_df[yearly_df["yearID"].isin(recent_years_trend)].copy().sort_values(["playerID", "yearID"])

    agg_trend = recent_data_trend.groupby(["playerID", "fullName", "bats"], as_index=False)[["G", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]].sum()
    agg_trend = add_rate_stats(agg_trend)
    agg_trend = agg_trend[agg_trend["G"] >= min_g_trend].copy()
    agg_trend = apply_stat_min_filters(agg_trend, "trend")

    trend_table = recent_data_trend.groupby("playerID").apply(lambda g: pd.Series({
        "R_trend": compute_trend_slope(g, "R"), "H_trend": compute_trend_slope(g, "H"),
        "2B_trend": compute_trend_slope(g, "2B"), "3B_trend": compute_trend_slope(g, "3B"),
        "HR_trend": compute_trend_slope(g, "HR"), "RBI_trend": compute_trend_slope(g, "RBI"),
        "SB_trend": compute_trend_slope(g, "SB"), "BB_trend": compute_trend_slope(g, "BB"),
        "BA_trend": compute_trend_slope(g, "BA"), "OBP_trend": compute_trend_slope(g, "OBP"),
        "SLG_trend": compute_trend_slope(g, "SLG"), "OPS_trend": compute_trend_slope(g, "OPS")
    })).reset_index()

    trend_value_df = agg_trend.merge(trend_table, on="playerID", how="left")
    trend_value_df = add_latest_and_projection_columns(trend_value_df, recent_data_trend)

    trend_display = trend_value_df[["fullName", "bats", "R_trend", "H_trend", "2B_trend", "3B_trend", "HR_trend", "RBI_trend", "SB_trend", "BB_trend", "BA_trend", "OBP_trend", "SLG_trend", "OPS_trend"]].copy()
    trend_display.columns = ["Player", "Bats", "R Δ", "H Δ", "2B Δ", "3B Δ", "HR Δ", "RBI Δ", "SB Δ", "BB Δ", "BA Δ", "OBP Δ", "SLG Δ", "OPS Δ"]

    sort_col = st.selectbox("Sort By Trend", ["R Δ", "H Δ", "2B Δ", "3B Δ", "HR Δ", "RBI Δ", "SB Δ", "BB Δ", "BA Δ", "OBP Δ", "SLG Δ", "OPS Δ"], index=11, key="trend_sort_col")
    trend_label_to_column = {"R Δ": "R_trend", "H Δ": "H_trend", "2B Δ": "2B_trend", "3B Δ": "3B_trend", "HR Δ": "HR_trend", "RBI Δ": "RBI_trend", "SB Δ": "SB_trend", "BB Δ": "BB_trend", "BA Δ": "BA_trend", "OBP Δ": "OBP_trend", "SLG Δ": "SLG_trend", "OPS Δ": "OPS_trend"}
    selected_trend_col = trend_label_to_column[sort_col]
    selected_trend_name = sort_col.replace(" Δ", "")
    selected_values = pd.to_numeric(trend_value_df[selected_trend_col], errors="coerce")

    c3m, c4m, c5m = st.columns(3)
    if selected_trend_name in RATE_STATS:
        c3m.metric(f"Best {selected_trend_name} Trend", fmt_rate_4(selected_values.max()))
        c4m.metric(f"Worst {selected_trend_name} Trend", fmt_rate_4(selected_values.min()))
        c5m.metric(f"Average {selected_trend_name} Trend", fmt_rate_4(selected_values.mean()))
    else:
        c3m.metric(f"Best {selected_trend_name} Trend", fmt_count_1(selected_values.max()))
        c4m.metric(f"Worst {selected_trend_name} Trend", fmt_count_1(selected_values.min()))
        c5m.metric(f"Average {selected_trend_name} Trend", fmt_count_1(selected_values.mean()))

    trend_sorted = clean_ui_columns(trend_display.sort_values(sort_col, ascending=False))
    st.subheader("Trend Table")
    st.caption("Showing the top 500 rows for speed. Use filters to narrow the table further.")
    trend_sorted_display = format_display_table(
        trend_sorted.head(500),
        count_cols=[c for c in TREND_COUNT_COLS if c in trend_sorted.columns],
        rate_cols=[c for c in TREND_RATE_COLS if c in trend_sorted.columns],
        count_decimals=1,
        rate_decimals=4,
    )
    render_output_table(trend_sorted_display, key="trend_table", file_name="trend_value.csv", style_cols=[c for c in trend_sorted_display.columns if "Δ" in c])
    breakout_df = trend_value_df[["fullName", "bats", "OPS_trend", "HR_trend", "XBH_noHR_trend", "RBI_trend", "SB_trend"]].copy()
    top_breakouts = breakout_df.sort_values("OPS_trend", ascending=False).head(10)
    biggest_declines = breakout_df.sort_values("OPS_trend", ascending=True).head(10)

    rename_breakout = {"fullName": "Player", "bats": "Bats", "OPS_trend": "OPS Δ", "HR_trend": "HR Δ", "XBH_noHR_trend": "2B+3B Δ", "RBI_trend": "RBI Δ", "SB_trend": "SB Δ"}
    top_breakouts_display = clean_ui_columns(top_breakouts.rename(columns=rename_breakout))
    biggest_declines_display = clean_ui_columns(biggest_declines.rename(columns=rename_breakout))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🔥 Top Breakout Players")
        breakout_table = format_display_table(top_breakouts_display, count_cols=["HR Δ", "2B+3B Δ", "RBI Δ", "SB Δ"], rate_cols=["OPS Δ"], count_decimals=1, rate_decimals=4)
        render_output_table(breakout_table, key="top_breakouts", file_name="top_breakouts.csv", style_cols=[c for c in breakout_table.columns if "Δ" in c])
    with c4:
        st.subheader("❄️ Biggest Declines")
        declines_table = format_display_table(biggest_declines_display, count_cols=["HR Δ", "2B+3B Δ", "RBI Δ", "SB Δ"], rate_cols=["OPS Δ"], count_decimals=1, rate_decimals=4)
        render_output_table(declines_table, key="biggest_declines", file_name="biggest_declines.csv", style_cols=[c for c in declines_table.columns if "Δ" in c])

    st.subheader("Insight Summaries")
    top_breakout_row = trend_value_df.sort_values("OPS_trend", ascending=False).head(1)
    top_decline_row = trend_value_df.sort_values("OPS_trend", ascending=True).head(1)
    if not top_breakout_row.empty: st.success(make_trend_insight_summary(top_breakout_row.iloc[0]))
    if not top_decline_row.empty: st.error(make_trend_insight_summary(top_decline_row.iloc[0]))

    st.subheader("Player Trend Visualization")
    label_map_trend = build_player_label_map(recent_data_trend)
    selected_label_trend = st.selectbox("Select Player to View Trend", list(label_map_trend.keys()), key="trend_player")
    selected_id_trend = label_map_trend[selected_label_trend]
    stat_choice_trend = st.selectbox("Choose Trend Stat to Plot", ["R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"], key="trend_plot_stat")
    player_trend = recent_data_trend[recent_data_trend["playerID"] == selected_id_trend].sort_values("yearID")
    player_trend = safe_round_rate_stats(player_trend)
    player_name = player_trend["fullName"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(player_trend["yearID"], player_trend[stat_choice_trend], marker="o", label=stat_choice_trend)
    trend_years = sorted(pd.to_numeric(player_trend["yearID"], errors="coerce").dropna().astype(int).unique())
    ax.set_xticks(trend_years)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel(stat_choice_trend)
    ax.set_title(f"{player_name} – {stat_choice_trend} over {lag_trend} Years")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    player_summary_row = trend_value_df[trend_value_df["playerID"] == selected_id_trend]
    if not player_summary_row.empty:
        st.info(make_trend_insight_summary(player_summary_row.iloc[0]))

if active_page == "Valuation":
    render_section_header("💰 Valuation", "Blend recent production and trend momentum into a valuation score.")
    c1, c2 = st.columns(2)
    with c1:
        lag_value = st.selectbox("Valuation Window (Years)", [3, 4, 5], index=0, key="value_lag")
    with c2:
        min_g_value = st.number_input("Minimum Games Played", 0, 800, 50, key="value_min_g")

    max_year_value = int(yearly_df["yearID"].max())
    recent_years_value = list(range(max_year_value - lag_value + 1, max_year_value + 1))
    st.write(f"Analyzing seasons: **{recent_years_value[0]}–{recent_years_value[-1]}**")
    recent_data_value = yearly_df[yearly_df["yearID"].isin(recent_years_value)].copy().sort_values(["playerID", "yearID"])

    agg_value = recent_data_value.groupby(["playerID", "fullName", "bats"], as_index=False)[["G", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]].sum()
    agg_value = add_rate_stats(agg_value)
    agg_value = agg_value[agg_value["G"] >= min_g_value].copy()
    agg_value = apply_stat_min_filters(agg_value, "value")

    trend_value = recent_data_value.groupby("playerID").apply(lambda g: pd.Series({
        "R_trend": compute_trend_slope(g, "R"), "H_trend": compute_trend_slope(g, "H"),
        "2B_trend": compute_trend_slope(g, "2B"), "3B_trend": compute_trend_slope(g, "3B"),
        "HR_trend": compute_trend_slope(g, "HR"), "RBI_trend": compute_trend_slope(g, "RBI"),
        "SB_trend": compute_trend_slope(g, "SB"), "BB_trend": compute_trend_slope(g, "BB"),
        "BA_trend": compute_trend_slope(g, "BA"), "OBP_trend": compute_trend_slope(g, "OBP"),
        "SLG_trend": compute_trend_slope(g, "SLG"), "OPS_trend": compute_trend_slope(g, "OPS")
    })).reset_index()

    valuation_df = agg_value.merge(trend_value, on="playerID", how="left")
    valuation_df = add_latest_and_projection_columns(valuation_df, recent_data_value)

    st.subheader("Valuation Weights")
    c5, c6 = st.columns(2)
    with c5:
        w_current = st.number_input("Weight: Current Score", 0.0, 10.0, 1.0, key="value_w_current")
    with c6:
        w_trend = st.number_input("Weight: Trend Score", 0.0, 10.0, 1.0, key="value_w_trend")

    valuation_df["Trend_Score"] = (
        valuation_df["R_trend"].fillna(0) * 1.0 + valuation_df["H_trend"].fillna(0) * 0.5 +
        valuation_df["2B_trend"].fillna(0) * 0.75 + valuation_df["3B_trend"].fillna(0) * 0.75 +
        valuation_df["HR_trend"].fillna(0) * 2.0 + valuation_df["RBI_trend"].fillna(0) * 1.5 +
        valuation_df["SB_trend"].fillna(0) * 1.0 + valuation_df["BB_trend"].fillna(0) * 0.5 +
        valuation_df["BA_trend"].fillna(0) * 100 + valuation_df["OBP_trend"].fillna(0) * 100 +
        valuation_df["SLG_trend"].fillna(0) * 100 + valuation_df["OPS_trend"].fillna(0) * 100
    )
    valuation_df["Perf_Score"] = (
        valuation_df["R"] * 0.10 + valuation_df["H"] * 0.05 + valuation_df["2B"] * 0.05 +
        valuation_df["3B"] * 0.05 + valuation_df["HR"] * 0.25 + valuation_df["RBI"] * 0.20 +
        valuation_df["SB"] * 0.10 + valuation_df["BA"].fillna(0) * 100 * 0.05 +
        valuation_df["OBP"].fillna(0) * 100 * 0.10 + valuation_df["SLG"].fillna(0) * 100 * 0.10 +
        valuation_df["OPS"].fillna(0) * 100 * 0.10
    )
    valuation_df["Valuation_Raw"] = w_current * valuation_df["Perf_Score"] + w_trend * valuation_df["Trend_Score"]
    val_min = valuation_df["Valuation_Raw"].min()
    val_max = valuation_df["Valuation_Raw"].max()
    if pd.notna(val_min) and pd.notna(val_max) and val_max != val_min:
        valuation_df["Valuation_Score"] = (valuation_df["Valuation_Raw"] - val_min) / (val_max - val_min)
    else:
        valuation_df["Valuation_Score"] = 0.0

    valuation_df = safe_round_rate_stats(valuation_df)
    top_bar_chart(valuation_df, "fullName", "Valuation_Score", "Top 10 Valuation Score")

    c7, c8, c9 = st.columns(3)
    c7.metric("Top Valuation Score", fmt_rate_4(valuation_df["Valuation_Score"].max() if not valuation_df.empty else 0))
    c8.metric("Average Valuation Score", fmt_rate_4(valuation_df["Valuation_Score"].mean() if not valuation_df.empty else 0))
    c9.metric("Top Valuation Player", valuation_df.sort_values("Valuation_Score", ascending=False).iloc[0]["fullName"] if not valuation_df.empty else "N/A")

    valuation_display = valuation_df[["fullName", "bats", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BA", "OBP", "SLG", "OPS", "Trend_Score", "Perf_Score", "Valuation_Score"]].sort_values("Valuation_Score", ascending=False).rename(columns={
        "fullName": "Player", "bats": "Bats", "Trend_Score": "Trend Score", "Perf_Score": "Current Score", "Valuation_Score": "Valuation Score"
    })
    valuation_table = format_display_table(clean_ui_columns(valuation_display), count_cols=["R", "H", "2B", "3B", "HR", "RBI", "SB"], rate_cols=["BA", "OBP", "SLG", "OPS"], score_cols=["Trend Score", "Current Score", "Valuation Score"])
    render_output_table(valuation_table, key="valuation", file_name="valuation.csv")

    st.subheader("Valuation Insight Summaries")
    best_value_row = valuation_df.sort_values("Valuation_Score", ascending=False).head(1)
    worst_value_row = valuation_df.sort_values("Valuation_Score", ascending=True).head(1)
    if not best_value_row.empty:
        st.success(f"💰 Best valuation profile: {make_valuation_summary(best_value_row.iloc[0])}")
    if not worst_value_row.empty:
        st.warning(f"⚠️ Weakest valuation profile: {make_valuation_summary(worst_value_row.iloc[0])}")

if active_page == "ML Predictions":
    render_section_header(
        "🤖 ML Predictions",
        "Generate next-season projections using machine learning, aging curves, regression-to-the-mean, and similar-player comparisons."
    )

    if not SKLEARN_AVAILABLE:
        st.error("Scikit-learn is not installed. In Command Prompt, run: pip install scikit-learn")
    else:
        c1, c2, c3, c4top = st.columns(4)
        with c1:
            ml_lookback = st.selectbox("Lookback Window", [3, 4, 5], index=0, key="ml_lookback")
        with c2:
            ml_min_games = st.number_input("Minimum Games in Lookback Window", 0, 800, 150, key="ml_min_games")
        with c3:
            ml_sort_stat = st.selectbox("Rank Predictions By", ["OPS", "HR", "RBI", "SB", "R", "H", "BA", "OBP", "SLG", "BB"], index=0, key="ml_sort_stat")
        with c4top:
            ml_max_players = st.selectbox("Projection Scope", [100, 150, 300, 500], index=1, key="ml_max_players", help="Lower numbers are much faster on Streamlit Cloud.")

        st.subheader("Advanced Projection Settings")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            regression_strength = st.slider("Regression to Mean", 0.00, 0.60, 0.20, 0.05, key="ml_regression_strength")
        with a2:
            age_strength = st.slider("Aging Curve Strength", 0.00, 1.00, 0.50, 0.05, key="ml_age_strength")
        with a3:
            comp_weight = st.slider("Similar Player Weight", 0.00, 0.60, 0.10, 0.05, key="ml_comp_weight")
        with a4:
            k_neighbors = st.slider("Similar Players Used", 5, 50, 10, 5, key="ml_k_neighbors")

        st.info(
            "Predictions use a cached feature-engineered ML pipeline with age, position, bats, team/league context, park factor, playing time, trend slopes, walk/K rates, speed, aging, regression-to-the-mean, and similarity adjustments. "
            "The displayed **Predicted** columns are the adjusted projections recommended for interpretation."
        )

        run_ml_predictions = st.button("Generate / Refresh ML Predictions", type="primary", key="run_ml_predictions_button")
        if run_ml_predictions:
            st.session_state["ml_predictions_have_run"] = True

        if not st.session_state.get("ml_predictions_have_run", False):
            st.info(
                "Choose the lookback window and projection settings above, then click **Generate / Refresh ML Predictions**. "
                "After it runs, scroll down to see the projection table, top prediction summary, and feature importance."
            )
        else:
            with st.spinner("Generating cached fast projections..."):
                ml_training_df, ml_feature_cols, ml_models, current_rows, base_pred_df = build_base_ml_predictions(
                    yearly_df, ml_lookback, ml_min_games, max_player_pool=ml_max_players
                )
            if ml_training_df.empty or not ml_feature_cols:
                st.warning("Not enough historical data to train the model with these settings. Lower the minimum games or use a shorter lookback window.")
            else:

                c4, c5, c6 = st.columns(3)
                c4.metric("Training Examples", f"{len(ml_training_df):,}")
                c5.metric("Features Used", f"{len(ml_feature_cols):,}")
                c6.metric("Models Trained", f"{len(ml_models):,}")

                metric_rows = []
                for stat, info in ml_models.items():
                    metric_rows.append({"Stat": stat, "MAE": info["mae"], "R²": info["r2"]})
                metrics_df = pd.DataFrame(metric_rows)
                if not metrics_df.empty:
                    st.subheader("Model Accuracy Check")
                    st.caption("MAE means average miss. For example, HR MAE of 4 means the model is typically off by about 4 home runs on the test seasons.")
                    metrics_table = clean_ui_columns(metrics_df.round({"MAE": 3, "R²": 3}))
                    render_output_table(metrics_table, key="ml_accuracy", file_name="ml_model_accuracy.csv")

                if current_rows.empty:
                    st.warning("No current players met the minimum playing-time filter for prediction.")
                else:
                    pred_df = base_pred_df.copy()
                    pred_df, age_curve_df, comp_df = apply_advanced_projection_adjustments(
                        pred_df, current_rows, ml_training_df, ml_feature_cols, ML_TARGET_STATS,
                        regression_strength=regression_strength,
                        age_strength=age_strength,
                        comp_weight=comp_weight,
                        k_neighbors=k_neighbors,
                    )

                    min_pred_ab = st.number_input("Minimum Recent AB in Lookback Window", 0, 2500, 300, key="ml_min_ab")
                    pred_df = pred_df[pd.to_numeric(pred_df["hist_AB_total"], errors="coerce") >= min_pred_ab].copy()

                    if pred_df.empty:
                        st.warning("No players met the Recent AB filter. Lower **Minimum Recent AB in Lookback Window** or lower **Minimum Games in Lookback Window** above, then generate again.")
                    else:
                        st.success(f"Generated {len(pred_df):,} player projections. Scroll down to view the table.")

                    sort_col = f"Predicted {ml_sort_stat}"
                    if sort_col in pred_df.columns:
                        pred_df = pred_df.sort_values(sort_col, ascending=False)

                    # User-facing ML output: show only identifying info and the recommended predicted stats.
                    # Historical/diagnostic columns such as Last Year, Last HR, Recent AB, Final, raw model outputs,
                    # and similar-player columns are intentionally hidden from the main table.
                    display_cols = [
                        "fullName", "bats", "prediction_year", "age_entering_year",
                        "Predicted R", "Predicted H", "Predicted 2B", "Predicted 3B", "Predicted HR", "Predicted RBI", "Predicted SB", "Predicted BB",
                        "Predicted BA", "Predicted OBP", "Predicted SLG", "Predicted OPS"
                    ]
                    display_cols = [c for c in display_cols if c in pred_df.columns]
                    projection_rename = {
                        "fullName": "Player", "bats": "Bats", "prediction_year": "Prediction Year",
                        "age_entering_year": "Age"
                    }
                    ml_display = clean_ui_columns(pred_df[display_cols].rename(columns=projection_rename))

                    st.subheader("Next-Season ML Projections")
                    st.caption("Predictions use machine learning with aging, regression-to-the-mean, and similarity adjustments. The table shows the recommended projected stats only.")
                    for _col in ml_display.columns:
                        if _col.startswith("Predicted "):
                            _stat = _col.replace("Predicted ", "")
                            ml_display[_col] = pd.to_numeric(ml_display[_col], errors="coerce").round(3 if _stat in RATE_STATS else 0)
                    if "Age" in ml_display.columns:
                        ml_display["Age"] = pd.to_numeric(ml_display["Age"], errors="coerce").round(0)
                    render_output_table(ml_display, key="ml_predictions", file_name="ml_predictions.csv")

                    if not ml_display.empty:
                        st.subheader("Top Prediction Summary")
                        st.success(make_ml_prediction_summary(ml_display.iloc[0], ml_sort_stat))

                    with st.expander("Show age curve details"):
                        st.write("The age curve estimates how players historically changed from their most recent season to the following season at each age. Similar-player comps are still used internally for the projection adjustment, but they are hidden from the main output to keep the page clean.")
                        if not age_curve_df.empty:
                            age_stats = [s for s in ML_TARGET_STATS if s in age_curve_df["Stat"].unique()]
                            if age_stats:
                                age_view_stat = st.selectbox("Age Curve Stat", age_stats, index=0, key="ml_age_curve_stat")
                                age_view = age_curve_df[age_curve_df["Stat"] == age_view_stat].rename(columns={"Age Adjustment": "Expected Age Change"})
                                age_curve_table = format_display_table(age_view, rate_cols=["Expected Age Change"])
                                render_output_table(age_curve_table, key="ml_age_curve", file_name="ml_age_curve.csv")

                    st.subheader("What Stats Matter Most?")
                    importance_options = [s for s in ML_TARGET_STATS if s in ml_models]
                    if importance_options:
                        importance_stat = st.selectbox("Feature Importance For", importance_options, index=0, key="ml_importance_stat")
                        importance_df = ml_models[importance_stat]["importance"].head(15).copy()
                        importance_df["Feature"] = importance_df["Feature"].apply(clean_feature_name)
                        importance_table = format_display_table(clean_ui_columns(importance_df), rate_cols=["Importance"])
                        render_output_table(importance_table, key="ml_feature_importance", file_name="ml_feature_importance.csv")
                        top_bar_chart(importance_df, "Feature", "Importance", f"Top Feature Importance for Predicting {importance_stat}", top_n=15)

                    st.info(
                        "How to explain this in an interview: I turned baseball history into a supervised ML projection problem. "
                        "For each player-season, the input is the previous 3–5 years of production plus age/age², position, batting hand, team, league, park factor, playing time, walk/K rates, speed proxies, recent averages, weighted recent production, and trend slopes. "
                        "The target is the next season. I use Random Forest for nonlinear prediction, then improve stability with regression-to-the-mean, "
                        "a learned aging curve, and a similar-player nearest-neighbor blend."
                    )

st.caption("Built with Streamlit, Pandas, and Matplotlib using People.csv, Batting.csv, and Fielding.csv.")
