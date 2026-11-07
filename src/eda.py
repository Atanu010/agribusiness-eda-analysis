"""
Core Exploratory Data Analysis (EDA) Engine for Agribusiness Datasets.

Provides reusable analytical functions for:
- Univariate parametric and non-parametric statistical profiling
- Production share, growth rates (YoY, CAGR), and spatial yield indices
- Market price spread, percentage change, and rolling volatility metrics
- Seasonal price and arrival indices
- Pearson and Spearman rank correlation matrices
- Anomaly diagnosis (IQR fences, Z-score, and time-series breaks)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils import logger


# ---------------------------------------------------------------------------
# 1. Univariate Statistical Profiling
# ---------------------------------------------------------------------------

def compute_numerical_profile(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Computes comprehensive parametric and non-parametric descriptive statistics.

    Args:
        df: Input DataFrame.
        columns: List of numerical column names to profile.

    Returns:
        pd.DataFrame: Tabular summary of descriptive statistics.
    """
    profile_records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25

        profile_records.append({
            "Variable": col,
            "Count": int(len(series)),
            "Mean": float(series.mean()),
            "Median": float(series.median()),
            "Std_Dev": float(series.std()),
            "Variance": float(series.var()),
            "Min": float(series.min()),
            "Q25": q25,
            "Q75": q75,
            "IQR": iqr,
            "Max": float(series.max()),
            "Skewness": float(series.skew()),
            "Kurtosis": float(series.kurtosis()),
        })

    return pd.DataFrame(profile_records)


def compute_categorical_profile(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Profiles categorical columns (cardinality, missingness, top category and share).
    """
    records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        val_counts = series.value_counts(dropna=False)
        top_cat = val_counts.index[0] if len(val_counts) > 0 else "N/A"
        top_count = val_counts.iloc[0] if len(val_counts) > 0 else 0

        records.append({
            "Variable": col,
            "Total_Rows": len(series),
            "Unique_Categories": int(series.nunique()),
            "Missing_Values": int(df[col].isna().sum()),
            "Top_Category": top_cat,
            "Top_Category_Count": int(top_count),
            "Top_Category_Share_Pct": round((top_count / len(series)) * 100.0, 2) if len(series) > 0 else 0.0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Production & Yield KPI Calculators
# ---------------------------------------------------------------------------

def calculate_yoy_growth(series: pd.Series) -> pd.Series:
    """
    Computes Year-over-Year (YoY) percentage change.
    Formula: YoY (%) = ((Value_t - Value_{t-1}) / Value_{t-1}) * 100
    """
    return series.pct_change(periods=1) * 100.0


def calculate_regional_share(df: pd.DataFrame, group_col: str, value_col: str = "production") -> pd.DataFrame:
    """
    Computes regional percentage contribution to total aggregate output.
    Formula: Share (%) = (Production_Region / Production_Total) * 100
    """
    total_val = df[value_col].sum()
    share_df = df.groupby(group_col)[value_col].sum().reset_index()
    share_df["Share_Pct"] = (share_df[value_col] / total_val) * 100.0
    return share_df.sort_values("Share_Pct", ascending=False).reset_index(drop=True)


def calculate_spatial_yield_index(
    df: pd.DataFrame,
    crop_col: str = "crop",
    region_col: str = "state_name",
    yield_col: str = "yield_kg_per_ha",
) -> pd.DataFrame:
    """
    Calculates Spatial Yield Index (SYI) relative to national crop benchmark:
    SYI = (Regional_Yield / National_Yield) * 100
    """
    records = []
    for crop, group in df.groupby(crop_col):
        nat_yield = group[yield_col].mean()
        if nat_yield <= 0 or np.isnan(nat_yield):
            continue
        region_yields = group.groupby(region_col)[yield_col].mean().reset_index()
        region_yields["National_Mean_Yield"] = nat_yield
        region_yields["Spatial_Yield_Index"] = (region_yields[yield_col] / nat_yield) * 100.0
        region_yields["Crop"] = crop
        records.append(region_yields)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Market Price Dynamics & Volatility
# ---------------------------------------------------------------------------

def calculate_price_metrics(df: pd.DataFrame, price_col: str = "modal_price") -> Dict[str, float]:
    """
    Computes aggregate price statistics including spread, returns, and historical volatility.
    """
    series = df[price_col].dropna()
    if len(series) < 2:
        return {}

    log_returns = np.log(series / series.shift(1)).dropna()
    volatility = float(log_returns.std() * np.sqrt(252)) if len(log_returns) > 1 else np.nan

    return {
        "Mean_Price": float(series.mean()),
        "Median_Price": float(series.median()),
        "Min_Price": float(series.min()),
        "Max_Price": float(series.max()),
        "Price_Spread": float(series.max() - series.min()),
        "Std_Dev": float(series.std()),
        "Annualized_Volatility": volatility,
    }


def compute_seasonal_price_index(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "modal_price",
) -> pd.DataFrame:
    """
    Computes monthly seasonal price index:
    Seasonal_Index (%) = (Monthly_Mean_Price / Overall_Annual_Mean_Price) * 100
    """
    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df["month"] = temp_df[date_col].dt.month
    temp_df["month_name"] = temp_df[date_col].dt.strftime("%b")

    overall_mean = temp_df[price_col].mean()
    monthly_summary = temp_df.groupby(["month", "month_name"])[price_col].mean().reset_index()
    monthly_summary["Seasonal_Index"] = (monthly_summary[price_col] / overall_mean) * 100.0
    return monthly_summary.sort_values("month").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Correlation Analysis
# ---------------------------------------------------------------------------

def compute_correlation_matrices(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Computes both Pearson (linear) and Spearman (rank monotonic) correlation matrices.
    """
    valid_cols = [c for c in numeric_cols if c in df.columns]
    numeric_df = df[valid_cols].apply(pd.to_numeric, errors="coerce")

    pearson_corr = numeric_df.corr(method="pearson")
    spearman_corr = numeric_df.corr(method="spearman")

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }


# ---------------------------------------------------------------------------
# 5. Statistical Anomaly Detection
# ---------------------------------------------------------------------------

def detect_statistical_anomalies(
    df: pd.DataFrame,
    target_col: str,
    group_col: Optional[str] = None,
    iqr_multiplier: float = 3.0,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Diagnoses potential anomalies using dual statistical hurdles:
    1. Grouped / Global Interquartile Range (IQR) fences
    2. Z-Score standardization threshold
    """
    result_df = df.copy()
    result_df["is_iqr_anomaly"] = False
    result_df["is_zscore_anomaly"] = False

    if target_col not in result_df.columns:
        return result_df

    # Global or Grouped IQR
    if group_col and group_col in result_df.columns:
        for _, group in result_df.groupby(group_col):
            if len(group) < 10:
                continue
            q25 = group[target_col].quantile(0.25)
            q75 = group[target_col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                upper = q75 + (iqr_multiplier * iqr)
                lower = max(0.0, q25 - (iqr_multiplier * iqr))
                outliers = group[(group[target_col] > upper) | (group[target_col] < lower)].index
                result_df.loc[outliers, "is_iqr_anomaly"] = True
    else:
        q25 = result_df[target_col].quantile(0.25)
        q75 = result_df[target_col].quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:
            upper = q75 + (iqr_multiplier * iqr)
            lower = max(0.0, q25 - (iqr_multiplier * iqr))
            result_df["is_iqr_anomaly"] = (result_df[target_col] > upper) | (result_df[target_col] < lower)

    # Z-Score
    mean_val = result_df[target_col].mean()
    std_val = result_df[target_col].std()
    if std_val > 0:
        z_scores = np.abs((result_df[target_col] - mean_val) / std_val)
        result_df["is_zscore_anomaly"] = z_scores > z_threshold

    return result_df
