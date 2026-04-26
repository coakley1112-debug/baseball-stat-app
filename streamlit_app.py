
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

COUNT_STATS = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "G"]
RATE_STATS = ["BA", "OBP", "SLG", "OPS"]
TREND_COUNT_COLS = ["R Δ", "H Δ", "2B Δ", "3B Δ", "HR Δ", "RBI Δ", "SB Δ", "BB Δ"]
TREND_RATE_COLS = ["BA Δ", "OBP Δ", "SLG Δ", "OPS Δ"]
ML_TARGET_STATS = ["R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]
ML_BASE_FEATURE_STATS = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"]

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
    if pd.isna(val): return ""
    if val > 0: return "color: green; font-weight: bold;"
    if val < 0: return "color: red; font-weight: bold;"
    return "color: gray;"

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
        f"If the recent pattern continues, next season projects roughly around "
        f"{fmt_rate_3(proj_ops)} OPS, {fmt_int(proj_hr)} HR, {fmt_int(proj_xbh)} doubles/triples, "
        f"{fmt_int(proj_rbi)} RBI, and {fmt_int(proj_sb)} SB."
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
        f"{player}'s Trend Score is {fmt_count_1(trend_score)}, Performance Score is {fmt_count_1(perf_score)}, "
        f"and Valuation Score is {fmt_rate_4(valuation_score)}. "
        f"That is {valuation_description}. "
        f"The Valuation Score combines current production with recent trend direction, "
        f"then scales the result from 0 to 1 compared with the other players in the filtered group. "
        f"If the recent pattern continues, next season projects roughly around "
        f"{fmt_rate_3(proj_ops)} OPS, {fmt_int(proj_hr)} HR, {fmt_int(proj_xbh)} doubles/triples, "
        f"{fmt_int(proj_rbi)} RBI, and {fmt_int(proj_sb)} SB."
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

def format_display_table(df, count_cols=None, rate_cols=None, score_cols=None):
    count_cols = count_cols or []
    rate_cols = rate_cols or []
    score_cols = score_cols or []
    format_dict = {}
    for col in count_cols:
        if col in df.columns:
            format_dict[col] = "{:.0f}"
    for col in rate_cols:
        if col in df.columns:
            format_dict[col] = "{:.3f}"
    for col in score_cols:
        if col in df.columns:
            format_dict[col] = "{:.4f}" if col == "Valuation Score" else "{:.1f}"
    return df.style.format(format_dict)



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
        "Perf_Score": "Performance Score",
        "Valuation_Score": "Valuation Score",
    }
    df = df.rename(columns=rename_map)
    drop_cols = [c for c in df.columns if str(c).lower().endswith("id") or "playerid" in str(c).lower() or "teamid" in str(c).lower()]
    return df.drop(columns=drop_cols, errors="ignore")

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

def build_ml_training_set(yearly_source, lookback_years=3, min_games_per_window=50, target_stats=None):
    """Create supervised learning rows: last N years of stats -> following year stats."""
    target_stats = target_stats or ML_TARGET_STATS
    df = yearly_source.copy().sort_values(["playerID", "yearID"])
    for col in ["yearID", "birthYear"] + ML_BASE_FEATURE_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for player_id, g in df.groupby("playerID"):
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
            row = {
                "playerID": player_id,
                "fullName": target.get("fullName", ""),
                "bats": target.get("bats", ""),
                "predict_year": int(target["yearID"]),
                "last_year": int(target["yearID"]) - 1,
                "age_entering_year": int(target["yearID"]) - birth_year if pd.notna(birth_year) else np.nan,
                "hist_G_total": pd.to_numeric(history["G"], errors="coerce").sum(),
                "hist_AB_total": pd.to_numeric(history["AB"], errors="coerce").sum(),
            }
            for stat in ML_BASE_FEATURE_STATS:
                values = pd.to_numeric(history[stat], errors="coerce")
                row[f"{stat}_mean_{lookback_years}yr"] = values.mean()
                row[f"{stat}_last"] = values.iloc[-1]
                row[f"{stat}_trend"] = compute_trend_slope(history, stat)
            for stat in target_stats:
                row[f"target_{stat}"] = pd.to_numeric(target.get(stat, np.nan), errors="coerce")
            rows.append(row)
    ml_df = pd.DataFrame(rows)
    if ml_df.empty:
        return ml_df, []
    feature_cols = [c for c in ml_df.columns if c not in ["playerID", "fullName", "bats", "predict_year", "last_year"] and not c.startswith("target_")]
    ml_df[feature_cols] = ml_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    target_cols = [f"target_{stat}" for stat in target_stats]
    ml_df[target_cols] = ml_df[target_cols].apply(pd.to_numeric, errors="coerce")
    ml_df = ml_df.dropna(subset=target_cols, how="all")
    return ml_df, feature_cols

@st.cache_data(show_spinner=False)
def train_random_forest_models(ml_training_df, feature_cols, target_stats, random_state=42):
    """Train one Random Forest per stat, so each stat has its own accuracy and feature importance."""
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
        model = RandomForestRegressor(n_estimators=350, max_depth=9, min_samples_leaf=4, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        importances = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        results[stat] = {"model": model, "mae": float(mean_absolute_error(y_test, preds)), "r2": float(r2_score(y_test, preds)) if len(y_test) > 1 else np.nan, "importance": importances}
    return results



def get_target_baselines(ml_training_df, target_stats):
    """League-wide historical next-year averages, used for regression-to-the-mean."""
    baselines = {}
    for stat in target_stats:
        col = f"target_{stat}"
        if col in ml_training_df.columns:
            baselines[stat] = pd.to_numeric(ml_training_df[col], errors="coerce").mean()
    return baselines


def get_age_curve_adjustments(ml_training_df, target_stats):
    """
    Estimate aging effects from historical training rows.
    For each age and stat, compare the following-year result to the player's most recent season.
    """
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


def build_similar_player_predictions(current_rows, ml_training_df, feature_cols, target_stats, k_neighbors=25, max_age_gap=3):
    """Compare current profiles to historical profiles, then average those players' next seasons."""
    if current_rows.empty or ml_training_df.empty or not feature_cols:
        return pd.DataFrame()
    train_X = ml_training_df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    current_X = current_rows.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    means = train_X.mean(axis=0)
    stds = train_X.std(axis=0).replace(0, 1)
    train_Z = ((train_X - means) / stds).to_numpy()
    current_Z = ((current_X - means) / stds).to_numpy()
    train_ages = pd.to_numeric(ml_training_df.get("age_entering_year", np.nan), errors="coerce").to_numpy()
    current_ages = pd.to_numeric(current_rows.get("age_entering_year", np.nan), errors="coerce").to_numpy()
    out_rows = []
    for i, row in current_rows.reset_index(drop=True).iterrows():
        distances = np.sqrt(np.sum((train_Z - current_Z[i]) ** 2, axis=1))
        age = current_ages[i] if i < len(current_ages) else np.nan
        candidate_mask = np.ones(len(distances), dtype=bool)
        if pd.notna(age):
            candidate_mask = np.abs(train_ages - age) <= max_age_gap
            if candidate_mask.sum() < max(k_neighbors, 10):
                candidate_mask = np.ones(len(distances), dtype=bool)
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            continue
        nearest = candidate_idx[np.argsort(distances[candidate_idx])[:k_neighbors]]
        comps = ml_training_df.iloc[nearest]
        out = {
            "playerID": row.get("playerID"),
            "Similar Player Sample": len(comps),
            "Similar Players": ", ".join(comps["fullName"].dropna().astype(str).head(5).tolist())
        }
        for stat in target_stats:
            tcol = f"target_{stat}"
            if tcol in comps.columns:
                out[f"Similar {stat}"] = pd.to_numeric(comps[tcol], errors="coerce").mean()
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def apply_advanced_projection_adjustments(pred_df, current_rows, ml_training_df, feature_cols, target_stats,
                                          regression_strength=0.20, age_strength=0.50, comp_weight=0.25, k_neighbors=25):
    """Blend Random Forest, similar-player comps, age curve, and regression-to-the-mean."""
    if pred_df.empty:
        return pred_df, pd.DataFrame(), pd.DataFrame()
    adjusted = pred_df.copy()
    baselines = get_target_baselines(ml_training_df, target_stats)
    age_curve_df = get_age_curve_adjustments(ml_training_df, target_stats)
    comp_df = build_similar_player_predictions(current_rows, ml_training_df, feature_cols, target_stats, k_neighbors=k_neighbors)
    if not comp_df.empty:
        adjusted = adjusted.merge(comp_df, on="playerID", how="left")
    else:
        adjusted["Similar Player Sample"] = np.nan
        adjusted["Similar Players"] = ""
    for stat in target_stats:
        rf_col = f"Predicted {stat}"
        final_col = f"Final {stat}"
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

def build_current_prediction_rows(yearly_source, lookback_years=3, min_games_per_window=50):
    """Create one current row per active/recent player using the latest N available years."""
    df = yearly_source.copy().sort_values(["playerID", "yearID"])
    for col in ["yearID", "birthYear"] + ML_BASE_FEATURE_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    max_year = int(df["yearID"].max())
    rows = []
    for player_id, g in df.groupby("playerID"):
        g = g.sort_values("yearID").reset_index(drop=True)
        if len(g) < lookback_years:
            continue
        history = g.tail(lookback_years)
        latest = history.iloc[-1]
        if int(latest["yearID"]) < max_year - 1:
            continue
        if pd.to_numeric(history["G"], errors="coerce").sum() < min_games_per_window:
            continue
        birth_year = pd.to_numeric(latest.get("birthYear", np.nan), errors="coerce")
        row = {
            "playerID": player_id,
            "fullName": latest.get("fullName", ""),
            "bats": latest.get("bats", ""),
            "last_year": int(latest["yearID"]),
            "prediction_year": int(latest["yearID"]) + 1,
            "age_entering_year": int(latest["yearID"] + 1) - birth_year if pd.notna(birth_year) else np.nan,
            "hist_G_total": pd.to_numeric(history["G"], errors="coerce").sum(),
            "hist_AB_total": pd.to_numeric(history["AB"], errors="coerce").sum(),
        }
        for stat in ML_BASE_FEATURE_STATS:
            values = pd.to_numeric(history[stat], errors="coerce")
            row[f"{stat}_mean_{lookback_years}yr"] = values.mean()
            row[f"{stat}_last"] = values.iloc[-1]
            row[f"{stat}_trend"] = compute_trend_slope(history, stat)
        rows.append(row)
    return pd.DataFrame(rows)

def make_ml_prediction_summary(row, sort_stat):
    player = row.get("Player", "This player")
    stat_val = row.get(f"Final {sort_stat}", row.get(f"Predicted {sort_stat}", np.nan))
    ops = row.get("Final OPS", row.get("Predicted OPS", np.nan))
    hr = row.get("Final HR", row.get("Predicted HR", np.nan))
    rbi = row.get("Final RBI", row.get("Predicted RBI", np.nan))
    sb = row.get("Final SB", row.get("Predicted SB", np.nan))
    stat_text = fmt_rate_3(stat_val) if sort_stat in RATE_STATS else fmt_int(stat_val)
    return (
        f"{player}'s advanced ML projection is strongest on {sort_stat}: {stat_text}. "
        f"The model projects about {fmt_rate_3(ops)} OPS, {fmt_int(hr)} HR, {fmt_int(rbi)} RBI, and {fmt_int(sb)} SB. "
        f"Unlike the simple trend tab, this projection blends Random Forest, age curves, similar-player history, and regression-to-the-mean."
    )

@st.cache_data
def load_data():
    people = pd.read_csv("People.csv", low_memory=False)
    batting = pd.read_csv("Batting.csv", low_memory=False)
    fielding = pd.read_csv("Fielding.csv", low_memory=False)

    batting["teamID_original"] = batting["teamID"]
    fielding["teamID_original"] = fielding["teamID"]
    batting["teamHistoricalName"] = batting["teamID_original"].map(team_id_to_historical_name).fillna(batting["teamID_original"])
    fielding["teamHistoricalName"] = fielding["teamID_original"].map(team_id_to_historical_name).fillna(fielding["teamID_original"])

    batting["teamID"] = batting["teamID"].replace(team_id_mapping)
    fielding["teamID"] = fielding["teamID"].replace(team_id_mapping)

    keep_people = ["playerID", "nameFirst", "nameLast", "birthYear", "birthCountry", "bats", "throws"]
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

    fielding_counts = fielding.groupby(["playerID", "yearID", "POS"]).size().reset_index(name="games_at_pos")
    primary_positions = (
        fielding_counts.sort_values(["playerID", "yearID", "games_at_pos"], ascending=[True, True, False])
        .drop_duplicates(subset=["playerID", "yearID"])
        [["playerID", "yearID", "POS"]]
        .rename(columns={"POS": "primaryPos"})
    )
    primary_positions = primary_positions[~primary_positions["primaryPos"].isin(["PH", "PR"])]

    batting = batting.merge(people, on="playerID", how="left")
    batting = batting.merge(primary_positions, on=["playerID", "yearID"], how="left")
    batting["primaryPos"] = batting["primaryPos"].fillna("DH")
    batting["teamName"] = batting["teamID"].map(team_id_to_name).fillna(batting["teamID"])
    batting = add_rate_stats(batting)

    yearly = (
        batting.groupby(["playerID", "fullName", "bats", "throws", "birthYear", "birthCountry", "yearID"], as_index=False)
        [["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]]
        .sum()
    )
    yearly = add_rate_stats(yearly)

    year_team_totals = batting.groupby(["playerID", "yearID", "teamID", "teamHistoricalName"], as_index=False).agg({"AB": "sum"})
    primary_teams = (
        year_team_totals.sort_values(["playerID", "yearID", "AB"], ascending=[True, True, False])
        .drop_duplicates(subset=["playerID", "yearID"])
        [["playerID", "yearID", "teamID", "teamHistoricalName"]]
        .rename(columns={"teamID": "primaryTeamID", "teamHistoricalName": "primaryHistoricalTeamName"})
    )
    yearly = yearly.merge(primary_teams, on=["playerID", "yearID"], how="left")
    yearly["primaryTeamName"] = yearly["primaryTeamID"].map(team_id_to_name).fillna(yearly["primaryTeamID"])
    yearly["primaryHistoricalTeamName"] = yearly["primaryHistoricalTeamName"].fillna(yearly["primaryTeamName"])

    yearly_pos = batting[["playerID", "yearID", "primaryPos"]].drop_duplicates(subset=["playerID", "yearID"])
    yearly = yearly.merge(yearly_pos, on=["playerID", "yearID"], how="left")
    yearly["primaryPos"] = yearly["primaryPos"].fillna("DH")
    yearly = yearly[~yearly["primaryPos"].isin(["PH", "PR"])]

    return batting, yearly, people

batting_df, yearly_df, people_df = load_data()
all_years = sorted(pd.to_numeric(yearly_df["yearID"], errors="coerce").dropna().astype(int).unique())
year_min = int(min(all_years))
year_max = int(max(all_years))
default_start_hist = max(year_min, 2010)
default_start_leaders = max(year_min, 2020)

tab_hist, tab_career, tab_leaders, tab_compare, tab_trend, tab_value, tab_ml = st.tabs([
    "Historical Explorer", "Career Totals", "Leaderboards", "Comparison Tool", "Trend Value", "Valuation", "ML Predictions"
])

with tab_hist:
    render_section_header("🔎 Historical Explorer", "Find individual player seasons using filters, minimum stat thresholds, sorting, and charts.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hist_year_range = st.slider("Year Range", year_min, year_max, (default_start_hist, year_max), key="hist_year")
    with c2:
        bats_options = sorted([x for x in yearly_df["bats"].dropna().unique() if str(x).strip() != ""])
        hist_bats = st.multiselect("Batting Hand", bats_options, default=bats_options, key="hist_bats")
    with c3:
        pos_options = sorted([x for x in yearly_df["primaryPos"].dropna().unique() if str(x).strip() != "" and x not in ["PH", "PR"]])
        hist_pos = st.multiselect("Primary Position", pos_options, default=pos_options, key="hist_pos")
    with c4:
        actual_team_names = sorted(set(yearly_df["primaryTeamName"].dropna().astype(str)).intersection(set(team_id_to_name.values())))
        hist_teams = st.multiselect("Franchise", actual_team_names, default=actual_team_names, key="hist_team")

    hist = yearly_df[(yearly_df["yearID"] >= hist_year_range[0]) & (yearly_df["yearID"] <= hist_year_range[1])].copy()
    if hist_bats: hist = hist[hist["bats"].isin(hist_bats)]
    if hist_pos: hist = hist[hist["primaryPos"].isin(hist_pos)]
    if hist_teams: hist = hist[hist["primaryTeamName"].isin(hist_teams)]

    hist = apply_stat_min_filters(hist, "hist")
    hist = safe_round_rate_stats(hist)

    c5, c6 = st.columns(2)
    with c5:
        hist_sort_stat = st.selectbox(
            "Sort Historical Explorer By",
            ["yearID", "fullName", "primaryTeamName", "primaryHistoricalTeamName", "primaryPos",
             "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"],
            index=10, key="hist_sort_stat"
        )
    with c6:
        hist_sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"], index=0, key="hist_sort_order")

    hist_display_raw = hist[[
        "yearID", "fullName", "bats", "primaryPos", "primaryHistoricalTeamName",
        "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"
    ]].copy().sort_values(by=hist_sort_stat, ascending=(hist_sort_order == "Ascending"), na_position="last")

    top_bar_chart(hist_display_raw, "fullName", hist_sort_stat, f"Top 10 Seasons by {hist_sort_label}")

    c7, c8, c9 = st.columns(3)
    c7.metric("Rows Returned", len(hist_display_raw))
    top_value = pd.to_numeric(hist_display_raw[hist_sort_stat], errors="coerce").max() if len(hist_display_raw) else 0
    c8.metric("Top Stat Value", fmt_rate_3(top_value) if hist_sort_stat in RATE_STATS else (fmt_int(top_value) if hist_sort_stat in COUNT_STATS or hist_sort_stat == "yearID" else str(top_value)))
    c9.metric("Year Range", f"{hist_year_range[0]}-{hist_year_range[1]}")

    hist_display = hist_display_raw.rename(columns={
        "yearID": "Year", "fullName": "Player", "bats": "Bats", "primaryPos": "Primary Position",
        "primaryHistoricalTeamName": "Team"
    })
    st.divider()
    st.dataframe(
        format_display_table(clean_ui_columns(hist_display), count_cols=["Year", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"]),
        use_container_width=True, hide_index=True
    )

with tab_career:
    render_section_header("📚 Career Totals", "Aggregate production across selected years with one clean row per player, using franchise filters but showing the actual historical team in the table.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        range_career = st.slider("Select Year Range", year_min, year_max, (max(year_min, 2010), year_max), key="career_year")
    with c2:
        bats_options_career = sorted([x for x in yearly_df["bats"].dropna().unique() if str(x).strip() != ""])
        bats_filter_career = st.multiselect("Batting Hand", bats_options_career, default=bats_options_career, key="career_bats")
    with c3:
        pos_options_career = sorted([x for x in yearly_df["primaryPos"].dropna().unique() if str(x).strip() != "" and x not in ["PH", "PR"]])
        pos_filter_career = st.multiselect("Primary Position", pos_options_career, default=pos_options_career, key="career_pos")
    with c4:
        team_options_career = sorted(set(yearly_df["primaryTeamName"].dropna().astype(str)).intersection(set(team_id_to_name.values())))
        team_filter_career = st.multiselect("Franchise", team_options_career, default=team_options_career, key="career_team")

    filtered_career = yearly_df[(yearly_df["yearID"] >= range_career[0]) & (yearly_df["yearID"] <= range_career[1])].copy()
    if bats_filter_career: filtered_career = filtered_career[filtered_career["bats"].isin(bats_filter_career)]
    if pos_filter_career: filtered_career = filtered_career[filtered_career["primaryPos"].isin(pos_filter_career)]
    if team_filter_career: filtered_career = filtered_career[filtered_career["primaryTeamName"].isin(team_filter_career)]

    stat_cols_career = ["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]
    career_totals = filtered_career.groupby(["playerID", "fullName", "bats"], as_index=False)[stat_cols_career].sum()

    pos_mode = (
        filtered_career.groupby(["playerID", "primaryPos"]).size().reset_index(name="count")
        .sort_values(["playerID", "count", "primaryPos"], ascending=[True, False, True])
        .drop_duplicates(subset=["playerID"])[["playerID", "primaryPos"]]
    )
    team_mode = (
        filtered_career.groupby(["playerID", "primaryTeamName", "primaryHistoricalTeamName"]).size().reset_index(name="count")
        .sort_values(["playerID", "count", "primaryTeamName", "primaryHistoricalTeamName"], ascending=[True, False, True, True])
        .drop_duplicates(subset=["playerID"])[["playerID", "primaryTeamName", "primaryHistoricalTeamName"]]
    )
    career_totals = career_totals.merge(pos_mode, on="playerID", how="left").merge(team_mode, on="playerID", how="left")
    career_totals = add_rate_stats(career_totals)
    career_totals = apply_stat_min_filters(career_totals, "career")
    career_totals = safe_round_rate_stats(career_totals)

    sort_stat_career = st.selectbox("Sort By", ["HR", "RBI", "SB", "R", "H", "2B", "3B", "BB", "BA", "OBP", "SLG", "OPS", "AB"], index=0, key="career_sort")
    top_bar_chart(career_totals, "fullName", sort_stat_career, f"Top 10 Career Totals by {sort_stat_career}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Players", len(career_totals))
    c6.metric("Top Player", career_totals.sort_values(sort_stat_career, ascending=False).iloc[0]["fullName"] if len(career_totals) else "N/A")
    top_career_value = pd.to_numeric(career_totals[sort_stat_career], errors="coerce").max() if len(career_totals) else 0
    c7.metric("Top Value", fmt_rate_3(top_career_value) if sort_stat_career in RATE_STATS else fmt_int(top_career_value))

    career_display = career_totals[[
        "fullName", "bats", "primaryPos", "primaryHistoricalTeamName",
        "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "BA", "OBP", "SLG", "OPS"
    ]].sort_values(sort_stat_career, ascending=False).rename(columns={
        "fullName": "Player", "bats": "Bats", "primaryPos": "Primary Position",
        "primaryHistoricalTeamName": "Team"
    })
    st.divider()
    st.dataframe(
        format_display_table(clean_ui_columns(career_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"]),
        use_container_width=True, hide_index=True
    )

with tab_leaders:
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
    st.dataframe(
        format_display_table(clean_ui_columns(leaderboard_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB"], rate_cols=["BA", "OBP", "SLG", "OPS"], score_cols=["Score"]),
        use_container_width=True, hide_index=True
    )

with tab_compare:
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
        st.dataframe(format_display_table(clean_ui_columns(compare_display), count_cols=["Year", "R", "H", "2B", "3B", "HR", "RBI", "SB", "AB"], rate_cols=["BA", "OBP", "SLG", "OPS"]), use_container_width=True, hide_index=True)

        st.subheader("Career Totals")
        career_compare = compare.groupby(["fullName"], as_index=False)[["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "SF"]].sum()
        career_compare = add_rate_stats(career_compare)
        career_compare = safe_round_rate_stats(career_compare)
        career_compare_display = career_compare[["fullName", "R", "AB", "H", "2B", "3B", "HR", "RBI", "SB", "BA", "OBP", "SLG", "OPS"]].sort_values("HR", ascending=False).rename(columns={"fullName": "Player"})
        st.dataframe(format_display_table(clean_ui_columns(career_compare_display), count_cols=["R", "AB", "H", "2B", "3B", "HR", "RBI", "SB"], rate_cols=["BA", "OBP", "SLG", "OPS"]), use_container_width=True, hide_index=True)

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

with tab_trend:
    render_section_header("🔥 Trend Value", "Shows only trend numbers: which stats are rising or declining per year over the selected recent window.")
    c1, c2 = st.columns(2)
    with c1:
        lag_trend = st.selectbox("Trend Window (Years)", [3, 4, 5], index=0, key="trend_lag")
    with c2:
        min_g_trend = st.number_input("Minimum Games Played", 0, 800, 50, key="trend_min_g")

    max_year_trend = int(yearly_df["yearID"].max())
    recent_years_trend = list(range(max_year_trend - lag_trend + 1, max_year_trend + 1))
    st.write(f"Analyzing seasons: **{recent_years_trend[0]}–{recent_years_trend[-1]}**")
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
    styled_trend = trend_sorted.style.map(color_trend, subset=TREND_COUNT_COLS + TREND_RATE_COLS).format({**{col: "{:.1f}" for col in TREND_COUNT_COLS}, **{col: "{:.4f}" for col in TREND_RATE_COLS}})
    st.subheader("Trend Table")
    st.dataframe(styled_trend, use_container_width=True, hide_index=True)

    breakout_df = trend_value_df[["fullName", "bats", "OPS_trend", "HR_trend", "XBH_noHR_trend", "RBI_trend", "SB_trend"]].copy()
    top_breakouts = breakout_df.sort_values("OPS_trend", ascending=False).head(10)
    biggest_declines = breakout_df.sort_values("OPS_trend", ascending=True).head(10)

    rename_breakout = {"fullName": "Player", "bats": "Bats", "OPS_trend": "OPS Δ", "HR_trend": "HR Δ", "XBH_noHR_trend": "2B+3B Δ", "RBI_trend": "RBI Δ", "SB_trend": "SB Δ"}
    top_breakouts_display = clean_ui_columns(top_breakouts.rename(columns=rename_breakout))
    biggest_declines_display = clean_ui_columns(biggest_declines.rename(columns=rename_breakout))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🔥 Top Breakout Players")
        st.dataframe(top_breakouts_display.style.map(color_trend, subset=["OPS Δ", "HR Δ", "2B+3B Δ", "RBI Δ", "SB Δ"]).format({"OPS Δ": "{:.4f}", "HR Δ": "{:.1f}", "2B+3B Δ": "{:.1f}", "RBI Δ": "{:.1f}", "SB Δ": "{:.1f}"}), use_container_width=True, hide_index=True)
    with c4:
        st.subheader("❄️ Biggest Declines")
        st.dataframe(biggest_declines_display.style.map(color_trend, subset=["OPS Δ", "HR Δ", "2B+3B Δ", "RBI Δ", "SB Δ"]).format({"OPS Δ": "{:.4f}", "HR Δ": "{:.1f}", "2B+3B Δ": "{:.1f}", "RBI Δ": "{:.1f}", "SB Δ": "{:.1f}"}), use_container_width=True, hide_index=True)

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

with tab_value:
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
        w_current = st.number_input("Weight: Current Production", 0.0, 10.0, 1.0, key="value_w_current")
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
        "fullName": "Player", "bats": "Bats", "Trend_Score": "Trend Score", "Perf_Score": "Performance Score", "Valuation_Score": "Valuation Score"
    })
    st.dataframe(
        format_display_table(clean_ui_columns(valuation_display), count_cols=["R", "H", "2B", "3B", "HR", "RBI", "SB"], rate_cols=["BA", "OBP", "SLG", "OPS"], score_cols=["Trend Score", "Performance Score", "Valuation Score"]),
        use_container_width=True, hide_index=True
    )

    st.subheader("Valuation Insight Summaries")
    best_value_row = valuation_df.sort_values("Valuation_Score", ascending=False).head(1)
    worst_value_row = valuation_df.sort_values("Valuation_Score", ascending=True).head(1)
    if not best_value_row.empty:
        st.success(f"💰 Best valuation profile: {make_valuation_summary(best_value_row.iloc[0])}")
    if not worst_value_row.empty:
        st.warning(f"⚠️ Weakest valuation profile: {make_valuation_summary(worst_value_row.iloc[0])}")

with tab_ml:
    render_section_header(
        "🤖 ML Predictions",
        "Train an advanced projection model from prior player seasons. It starts with Random Forest, then adds regression-to-the-mean, an age-curve adjustment, and similar-player comps."
    )

    if not SKLEARN_AVAILABLE:
        st.error("Scikit-learn is not installed. In Command Prompt, run: pip install scikit-learn")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            ml_lookback = st.selectbox("Lookback Window", [3, 4, 5], index=0, key="ml_lookback")
        with c2:
            ml_min_games = st.number_input("Minimum Games in Lookback Window", 0, 800, 150, key="ml_min_games")
        with c3:
            ml_sort_stat = st.selectbox("Rank Predictions By", ["OPS", "HR", "RBI", "SB", "R", "H", "BA", "OBP", "SLG", "BB"], index=0, key="ml_sort_stat")

        st.subheader("Advanced Projection Settings")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            regression_strength = st.slider("Regression to Mean", 0.00, 0.60, 0.20, 0.05, key="ml_regression_strength")
        with a2:
            age_strength = st.slider("Aging Curve Strength", 0.00, 1.00, 0.50, 0.05, key="ml_age_strength")
        with a3:
            comp_weight = st.slider("Similar Player Weight", 0.00, 0.60, 0.25, 0.05, key="ml_comp_weight")
        with a4:
            k_neighbors = st.slider("Similar Players Used", 5, 75, 25, 5, key="ml_k_neighbors")

        st.write(
            "This is a supervised learning setup: each training row says, "
            "**given a player's previous seasons, predict his next season**. "
            "The system combines Random Forest predictions with baseball-specific projection logic: similar-player comps, age-curve adjustment, and regression-to-the-mean for more stable forecasts."
        )

        ml_training_df, ml_feature_cols = build_ml_training_set(yearly_df, ml_lookback, ml_min_games, ML_TARGET_STATS)
        if ml_training_df.empty or not ml_feature_cols:
            st.warning("Not enough historical data to train the model with these settings. Lower the minimum games or use a shorter lookback window.")
        else:
            ml_models = train_random_forest_models(ml_training_df, ml_feature_cols, ML_TARGET_STATS)
            current_rows = build_current_prediction_rows(yearly_df, ml_lookback, ml_min_games)

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
                st.dataframe(clean_ui_columns(metrics_df).style.format({"MAE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True, hide_index=True)

            if current_rows.empty:
                st.warning("No current players met the minimum playing-time filter for prediction.")
            else:
                X_current = current_rows.reindex(columns=ml_feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
                pred_df = current_rows[["playerID", "fullName", "bats", "last_year", "prediction_year", "age_entering_year", "hist_G_total", "hist_AB_total"]].copy()
                for stat, info in ml_models.items():
                    pred_df[f"Predicted {stat}"] = info["model"].predict(X_current)

                for stat in ["R", "H", "2B", "3B", "HR", "RBI", "SB", "BB"]:
                    col = f"Predicted {stat}"
                    if col in pred_df.columns:
                        pred_df[col] = pred_df[col].clip(lower=0)
                for stat in RATE_STATS:
                    col = f"Predicted {stat}"
                    if col in pred_df.columns:
                        pred_df[col] = pred_df[col].clip(lower=0, upper=1.5)

                pred_df, age_curve_df, comp_df = apply_advanced_projection_adjustments(
                    pred_df, current_rows, ml_training_df, ml_feature_cols, ML_TARGET_STATS,
                    regression_strength=regression_strength,
                    age_strength=age_strength,
                    comp_weight=comp_weight,
                    k_neighbors=k_neighbors,
                )

                min_pred_ab = st.number_input("Minimum Recent AB in Lookback Window", 0, 2500, 300, key="ml_min_ab")
                pred_df = pred_df[pd.to_numeric(pred_df["hist_AB_total"], errors="coerce") >= min_pred_ab].copy()

                sort_col = f"Final {ml_sort_stat}"
                if sort_col in pred_df.columns:
                    pred_df = pred_df.sort_values(sort_col, ascending=False)

                display_cols = [
                    "fullName", "bats", "prediction_year", "age_entering_year", "hist_G_total", "hist_AB_total", "Similar Players",
                    "Final R", "Final H", "Final 2B", "Final 3B", "Final HR", "Final RBI", "Final SB", "Final BB",
                    "Final BA", "Final OBP", "Final SLG", "Final OPS",
                    "Predicted R", "Predicted H", "Predicted HR", "Predicted RBI", "Predicted SB", "Predicted OPS"
                ]
                display_cols = [c for c in display_cols if c in pred_df.columns]
                ml_display = clean_ui_columns(pred_df[display_cols].rename(columns={
                    "fullName": "Player", "bats": "Bats", "prediction_year": "Prediction Year",
                    "age_entering_year": "Age", "hist_G_total": "Recent Games", "hist_AB_total": "Recent AB"
                }))

                st.subheader("Next-Season Advanced ML Projections")
                st.dataframe(
                    ml_display.style.format({
                        **{c: "{:.0f}" for c in ml_display.columns if (c.startswith("Predicted ") or c.startswith("Final ")) and c.replace("Predicted ", "").replace("Final ", "") not in RATE_STATS},
                        **{c: "{:.3f}" for c in ml_display.columns if (c.startswith("Predicted ") or c.startswith("Final ")) and c.replace("Predicted ", "").replace("Final ", "") in RATE_STATS},
                        "Age": "{:.0f}", "Recent Games": "{:.0f}", "Recent AB": "{:.0f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                if not ml_display.empty:
                    st.subheader("Top Prediction Summary")
                    st.success(make_ml_prediction_summary(ml_display.iloc[0], ml_sort_stat))

                with st.expander("Show age curve and similar-player details"):
                    st.write("The age curve estimates how players historically changed from their most recent season to the following season at each age. Similar-player comps compare each player's recent profile to past players with similar age and statistics.")
                    if not age_curve_df.empty:
                        age_stats = [s for s in ML_TARGET_STATS if s in age_curve_df["Stat"].unique()]
                        if age_stats:
                            age_view_stat = st.selectbox("Age Curve Stat", age_stats, index=0, key="ml_age_curve_stat")
                            age_view = age_curve_df[age_curve_df["Stat"] == age_view_stat].rename(columns={"Age Adjustment": "Expected Age Change"})
                            st.dataframe(age_view.style.format({"Expected Age Change": "{:.4f}"}), use_container_width=True, hide_index=True)
                    if "Similar Players" in pred_df.columns:
                        comps_display = pred_df[["fullName", "age_entering_year", "Similar Player Sample", "Similar Players"]].rename(columns={"fullName": "Player", "age_entering_year": "Age"}).head(25)
                        st.dataframe(clean_ui_columns(comps_display), use_container_width=True, hide_index=True)

                st.subheader("What Stats Matter Most?")
                importance_options = [s for s in ML_TARGET_STATS if s in ml_models]
                if importance_options:
                    importance_stat = st.selectbox("Feature Importance For", importance_options, index=0, key="ml_importance_stat")
                    importance_df = ml_models[importance_stat]["importance"].head(15).copy()
                    importance_df["Feature"] = importance_df["Feature"].apply(clean_feature_name)
                    st.dataframe(clean_ui_columns(importance_df).style.format({"Importance": "{:.4f}"}), use_container_width=True, hide_index=True)
                    top_bar_chart(importance_df, "Feature", "Importance", f"Top Feature Importance for Predicting {importance_stat}", top_n=15)

                st.info(
                    "How to explain this in an interview: I turned baseball history into a supervised ML projection problem. "
                    "For each player-season, the input is the previous 3–5 years of production, age, recent averages, and trend slopes. "
                    "The target is the next season. I use Random Forest for nonlinear prediction, then improve stability with regression-to-the-mean, "
                    "a learned aging curve, and a similar-player nearest-neighbor blend."
                )

st.caption("Built with Streamlit, Pandas, and Matplotlib using People.csv, Batting.csv, and Fielding.csv.")
