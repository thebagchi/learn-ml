"""Utility functions for exploratory data analysis (EDA)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def display_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column (count and percentage).

    Only columns with at least one missing value are included, sorted by
    missing percentage descending. The ``missing_percent`` column is
    formatted as a string with a ``%`` suffix for display convenience.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``missing_count`` and ``missing_percent``
        indexed by column name.
    """
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
    })
    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_percent", ascending=False
    )
    missing["missing_percent"] = missing["missing_percent"].astype(str) + "%"
    return missing


def plot_missing_bars(
    df: pd.DataFrame,
    figsize: tuple = (14, 5),
) -> None:
    """Plot side-by-side bar charts of missing-value count and percentage.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    figsize : tuple
        Figure size ``(width, height)``.
    """
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().mean() * 100).round(2),
    })
    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_percent", ascending=False
    )
    if missing.empty:
        print("No missing values found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    missing["missing_count"].plot.barh(ax=axes[0], color="salmon")
    axes[0].set_title("Missing Value Count per Column")
    axes[0].set_xlabel("Count")

    missing["missing_percent"].plot.barh(ax=axes[1], color="steelblue")
    axes[1].set_title("Missing Value Percentage per Column")
    axes[1].set_xlabel("Percentage (%)")

    plt.tight_layout()
    plt.show()


def plot_missing_heatmap(
    df: pd.DataFrame,
    figsize: tuple = (12, 6),
) -> None:
    """Plot a heatmap showing the location of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    figsize : tuple
        Figure size ``(width, height)``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df.isnull(), cbar=True, yticklabels=False, cmap="YlOrRd", ax=ax,
    )
    ax.set_title("Missing Value Heatmap (yellow = present, red = missing)")
    plt.tight_layout()
    plt.show()


def compare_distributions(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    cols: List[str],
    figsize: tuple = (14, 10),
    title: str = "Before vs. After Imputation",
) -> None:
    """Overlay histograms of original vs imputed columns.

    Parameters
    ----------
    df_original : pd.DataFrame
        The original DataFrame (may contain NaN).
    df_imputed : pd.DataFrame
        The imputed DataFrame.
    cols : list of str
        Column names to compare.
    figsize : tuple
        Figure size ``(width, height)``.
    title : str
        Super-title for the figure.
    """
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for ax, col in zip(axes.flat, cols):
        ax.hist(
            df_original[col].dropna(), bins=25, alpha=0.5,
            label="Original", color="steelblue", density=True,
        )
        ax.hist(
            df_imputed[col], bins=25, alpha=0.5,
            label="Imputed", color="salmon", density=True,
        )
        ax.set_title(f"Distribution of '{col}'")
        ax.legend()

    # Hide unused axes
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def compare_boxplots(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    cols: List[str],
    figsize: Optional[tuple] = None,
    title: str = "Box Plot: Original vs Imputed",
) -> None:
    """Side-by-side box plots comparing original and imputed data.

    Parameters
    ----------
    df_original : pd.DataFrame
        The original DataFrame (may contain NaN).
    df_imputed : pd.DataFrame
        The imputed DataFrame.
    cols : list of str
        Column names to compare.
    figsize : tuple or None
        Figure size. Defaults to ``(4 * len(cols), 5)``.
    title : str
        Super-title for the figure.
    """
    if figsize is None:
        figsize = (4 * len(cols), 5)

    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        combined = pd.DataFrame({
            "Original": df_original[col],
            "Imputed": df_imputed[col],
        })
        combined.plot.box(ax=ax)
        ax.set_title(col)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def compare_stats(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """Return a comparison table of mean and std before and after imputation.

    Parameters
    ----------
    df_original : pd.DataFrame
        The original DataFrame (may contain NaN).
    df_imputed : pd.DataFrame
        The imputed DataFrame.
    cols : list of str
        Column names to compare.

    Returns
    -------
    pd.DataFrame
        Table with Original/Imputed Mean and Std per column.
    """
    return pd.DataFrame({
        "Original Mean": df_original[cols].mean(),
        "Imputed Mean": df_imputed[cols].mean(),
        "Original Std": df_original[cols].std(),
        "Imputed Std": df_imputed[cols].std(),
    }).round(3)


def add_missing_indicators(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add binary indicator columns flagging which values were missing.

    For each column in *cols*, a new column ``<col>_was_missing`` is added
    with value 1 where the original was NaN and 0 otherwise. The original
    columns are **not** modified.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to augment (not modified in place).
    cols : list of str or None
        Columns to create indicators for. If ``None``, indicators are
        created for every column that has at least one missing value.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the extra indicator columns appended.
    """
    df = df.copy()
    if cols is None:
        cols = df.columns[df.isnull().any()].tolist()
    for col in cols:
        df[f"{col}_was_missing"] = df[col].isnull().astype(int)
    return df
