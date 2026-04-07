"""
checkret.reports — Report generation combining metrics and plots.

Primary entry point:
    checkret.reports.full(pnl, base_pnl)
"""

from __future__ import annotations

import polars as pl
import matplotlib.pyplot as plt

from . import stats, plots


def _print_df(df: pl.DataFrame, title: str = "") -> None:
    """Pretty-print a Polars DataFrame as a formatted table."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # Get column widths
    cols = df.columns
    data = df.to_dict(as_series=False)
    widths = {}
    for col in cols:
        max_w = len(col)
        for val in data[col]:
            max_w = max(max_w, len(str(val)) if val is not None else 4)
        widths[col] = max_w + 2

    # Header
    header = "  ".join(str(col).rjust(widths[col]) for col in cols)
    print(header)
    print("  ".join("-" * widths[col] for col in cols))

    # Rows
    n_rows = len(data[cols[0]])
    for i in range(n_rows):
        row = "  ".join(
            str(data[col][i] if data[col][i] is not None else "—").rjust(widths[col])
            for col in cols
        )
        print(row)
    print()


def full(
    pnl: pl.DataFrame,
    base_pnl: pl.DataFrame | None = None,
    rf: float = 0.0,
    periods: int = 252,
    figsize_main: tuple = (12, 5),
    figsize_small: tuple = (12, 4),
    show: bool = True,
) -> dict:
    """
    Generate a full backtest report with metrics and plots.

    Args:
        pnl: Polars DataFrame with ["date", "pnl"] columns (daily returns).
        base_pnl: Optional benchmark DataFrame with same schema.
        rf: Risk-free rate (annualized, default 0.0).
        periods: Trading days per year (default 252).
        figsize_main: Figure size for main charts.
        figsize_small: Figure size for smaller charts.
        show: Whether to display plots inline (default True).

    Returns:
        dict with keys: "metrics", "drawdowns", "dow_stats", "figures"
    """
    # Validate inputs
    assert "date" in pnl.columns, "pnl must have a 'date' column"
    assert "pnl" in pnl.columns, "pnl must have a 'pnl' column"
    if base_pnl is not None:
        assert "date" in base_pnl.columns, "base_pnl must have a 'date' column"
        assert "pnl" in base_pnl.columns, "base_pnl must have a 'pnl' column"

    # Sort by date
    pnl = pnl.sort("date")
    if base_pnl is not None:
        base_pnl = base_pnl.sort("date")

    # ── 1. Metrics Summary ──────────────────────────────────────────
    metrics = stats.summary_metrics(pnl, base_pnl, rf, periods)
    _print_df(metrics, "Performance Metrics")

    # ── 2. Top Drawdowns ────────────────────────────────────────────
    dd = stats.drawdown_details(pnl)
    if dd.height > 0:
        _print_df(dd, "Top 5 Drawdowns")

    # ── 3. Day-of-Week Stats ────────────────────────────────────────
    dow = stats.day_of_week_stats(pnl)
    _print_df(dow, "Day-of-Week Statistics")

    # ── 4. Plots ────────────────────────────────────────────────────
    figures: dict[str, plt.Figure] = {}

    # Cumulative Returns
    fig = plots.plot_cumulative_returns(pnl, base_pnl, figsize=figsize_main)
    figures["cumulative_returns"] = fig
    if show:
        fig.show()

    # Drawdown
    fig = plots.plot_drawdown(pnl, figsize=figsize_small)
    figures["drawdown"] = fig
    if show:
        fig.show()

    # Monthly Heatmap
    fig = plots.plot_monthly_heatmap(pnl, figsize=figsize_main)
    figures["monthly_heatmap"] = fig
    if show:
        fig.show()

    # Yearly Returns
    fig = plots.plot_yearly_returns(pnl, base_pnl, figsize=figsize_main)
    figures["yearly_returns"] = fig
    if show:
        fig.show()

    # Return Distribution
    fig = plots.plot_return_distribution(pnl, base_pnl, figsize=figsize_main)
    figures["distribution"] = fig
    if show:
        fig.show()

    # Rolling Sharpe
    fig = plots.plot_rolling_sharpe(pnl, base_pnl, figsize=figsize_small)
    figures["rolling_sharpe"] = fig
    if show:
        fig.show()

    # Rolling Volatility
    fig = plots.plot_rolling_volatility(pnl, base_pnl, figsize=figsize_small)
    figures["rolling_volatility"] = fig
    if show:
        fig.show()

    # Day-of-Week Analysis
    fig = plots.plot_dow_returns(pnl)
    figures["dow_returns"] = fig
    if show:
        fig.show()

    return {
        "metrics": metrics,
        "drawdowns": dd,
        "dow_stats": dow,
        "figures": figures,
    }
