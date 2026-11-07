"""
Visual Analytics and Charting Module for Agribusiness Trends.

Provides standardized plotting routines using Matplotlib and Seaborn for:
- Time-series trends and rolling trajectory lines
- Ranked horizontal/vertical bar charts
- Distribution histograms with Kernel Density Estimates (KDE)
- Multi-crop and regional box plots for dispersion analysis
- Bivariate scatter plots with OLS regression fits
- Pearson and Spearman correlation heatmaps
- Calendar and seasonal pivot heatmaps
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_agri_style() -> None:
    """Configures clean, publication-grade styling for agribusiness charts."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


def plot_time_series_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Agricultural Time Series Trend",
    ylabel: str = "Value",
    ma_window: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plots a continuous time-series trajectory with optional moving average."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    clean_df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    ax.plot(clean_df[date_col], clean_df[value_col], color="#2b5c8f", label="Observed", linewidth=1.5, alpha=0.8)

    if ma_window:
        rolling_series = clean_df[value_col].rolling(window=ma_window, min_periods=1).mean()
        ax.plot(clean_df[date_col], rolling_series, color="#d95f02", label=f"{ma_window}-Period MA", linewidth=2.0)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Date / Time", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_crop_comparison_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str = "Comparative Analysis Across Crops / Regions",
    xlabel: str = "Metric Value",
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Generates a ranked horizontal bar chart for categorical comparisons."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    grouped = df.groupby(category_col)[value_col].mean().sort_values(ascending=False).head(top_n)
    colors = sns.color_palette("crest", len(grouped))[::-1]

    bars = ax.barh(grouped.index[::-1], grouped.values[::-1], color=colors, edgecolor="#444444", linewidth=0.5)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(category_col.replace("_", " ").title(), labelpad=8)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:,.1f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_distribution_histogram(
    df: pd.DataFrame,
    col: str,
    title: str = "Variable Distribution Profile",
    xlabel: str = "Value",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plots a histogram with Kernel Density Estimate and mean/median reference lines."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    clean_series = pd.to_numeric(df[col], errors="coerce").dropna()
    mean_val = clean_series.mean()
    median_val = clean_series.median()

    sns.histplot(clean_series, bins=bins, kde=True, color="#386cb0", edgecolor="#ffffff", alpha=0.6, ax=ax)
    ax.axvline(mean_val, color="#e41a1c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")
    ax.axvline(median_val, color="#4daf4a", linestyle="-", linewidth=1.5, label=f"Median: {median_val:,.1f}")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Frequency Density", labelpad=8)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_yield_boxplots(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str = "Yield Dispersion & Outlier Boxplot",
    ylabel: str = "Yield (kg/ha)",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Generates box plots assessing dispersion, IQR, and outliers across groups."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x=category_col,
        y=value_col,
        palette="Set2",
        ax=ax,
        fliersize=3.5,
        linewidth=1.2,
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(category_col.replace("_", " ").title(), labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Bivariate Relationship with OLS Regression Trend",
    xlabel: str = "Independent Variable",
    ylabel: str = "Dependent Variable",
    hue_col: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Generates a scatter plot with an OLS regression fit line."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.5, "color": "#1f78b4", "s": 35},
        line_kws={"color": "#e31a1c", "linewidth": 2.0},
        ax=ax,
    )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plots an annotated correlation matrix heatmap."""
    set_agri_style()
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig
