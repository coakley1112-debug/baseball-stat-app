
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
