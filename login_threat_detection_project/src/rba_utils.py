from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_CSV_PATH = DATA_DIR / "rba-dataset.csv"

BASE_FEATURES = [
    "Country", "Region", "City", "ASN",
    "OS Name and Version", "Browser Name and Version", "Device Type",
    "Round-Trip Time [ms]", "Login Successful",
    "login_hour", "login_dayofweek", "is_weekend",
]

CATEGORICAL_FEATURES = [
    "Country", "Region", "City", "OS Name and Version",
    "Browser Name and Version", "Device Type",
]

NUMERIC_FEATURES = [
    "ASN", "Round-Trip Time [ms]", "Login Successful",
    "login_hour", "login_dayofweek", "is_weekend",
]

def find_dataset_csv() -> Path:
    if DEFAULT_CSV_PATH.exists():
        return DEFAULT_CSV_PATH
    matches = list(DATA_DIR.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError("Could not find data/rba-dataset.csv")
    for path in matches:
        if "rba" in path.name.lower():
            return path
    return matches[0]

def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    return normalized.map(lambda x: 1 if x in true_values else (0 if x in false_values else np.nan))

def parse_login_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        median_value = abs(numeric.dropna().median())
        if median_value > 1e17:
            unit = "ns"
        elif median_value > 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = parse_login_timestamp(df["Login Timestamp"])
    df["login_hour"] = dt.dt.hour.fillna(-1).astype(int)
    df["login_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["is_weekend"] = df["login_dayofweek"].isin([5, 6]).astype(int)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Login Successful", "Is Attack IP", "Is Account Takeover"]:
        if col in df.columns:
            df[col] = parse_bool_series(df[col])
    df = add_time_features(df)
    for col in ["ASN", "Round-Trip Time [ms]"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan":"Unknown", "None":"Unknown"}).fillna("Unknown")
    return df

def reduce_rare_categories(df: pd.DataFrame, categorical_cols, top_n=30):
    df = df.copy()
    category_maps = {}
    for col in categorical_cols:
        top_values = df[col].astype(str).value_counts(dropna=False).head(top_n).index.tolist()
        category_maps[col] = set(top_values)
        df[col] = df[col].astype(str).where(df[col].astype(str).isin(category_maps[col]), "Other")
    return df, category_maps

def apply_category_maps(df: pd.DataFrame, category_maps):
    df = df.copy()
    for col, allowed_values in category_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).where(df[col].astype(str).isin(allowed_values), "Other")
    return df
