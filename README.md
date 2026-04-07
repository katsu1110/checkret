# Checkret

A simple backtest tool for your return series. Put your return series, and you get backtest results with visualizations and key metrics.

# How to use

## Installation

```bash
pip install checkret
```

Or with `uv`:

```bash
uv add checkret
```

## Data format

`checkret` expects a [Polars](https://pola.rs/) DataFrame with two columns:

| column | type | description |
|--------|------|-------------|
| `date` | `pl.Date` | Trading date |
| `pnl`  | `pl.Float64` | Daily return (e.g. `0.01` = +1%) |

## Basic usage

```python
import polars as pl
import checkret

# Build your return series
pnl = pl.DataFrame({
    "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), "1d", eager=True),
    "pnl": your_daily_returns,   # list / numpy array of floats
})

# Generate the full report (prints metrics + shows all plots)
results = checkret.reports.full(pnl)
```

`results` is a dict with the following keys:

| key | type | description |
|-----|------|-------------|
| `metrics` | `pl.DataFrame` | Summary metrics table |
| `drawdowns` | `pl.DataFrame` | Top-5 drawdown periods |
| `dow_stats` | `pl.DataFrame` | Day-of-week statistics |
| `figures` | `dict[str, Figure]` | All matplotlib figures |

## With a benchmark

```python
benchmark = pl.DataFrame({
    "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), "1d", eager=True),
    "pnl": benchmark_daily_returns,
})

results = checkret.reports.full(pnl, base_pnl=benchmark)
```

When a benchmark is provided, the metrics table also includes **Alpha**, **Beta**, **Correlation**, **Information Ratio**, and **Excess Return**.

## Advanced options

```python
results = checkret.reports.full(
    pnl,
    base_pnl=benchmark,
    rf=0.04,          # annualized risk-free rate (default 0.0)
    periods=252,      # trading days per year (default 252)
    show=False,       # suppress inline plot display
)
```

## Using individual modules

```python
import checkret

# --- Stats ---
checkret.stats.total_return(pnl)
checkret.stats.cagr(pnl)
checkret.stats.sharpe(pnl, rf=0.0)
checkret.stats.sortino(pnl)
checkret.stats.max_drawdown(pnl)
checkret.stats.calmar(pnl)
checkret.stats.volatility(pnl)
checkret.stats.win_rate(pnl)
checkret.stats.profit_factor(pnl)
checkret.stats.value_at_risk(pnl, alpha=0.05)

checkret.stats.drawdown_details(pnl, top_n=5)      # pl.DataFrame
checkret.stats.day_of_week_stats(pnl)              # pl.DataFrame
checkret.stats.summary_metrics(pnl, base_pnl)     # pl.DataFrame

# --- Plots ---
checkret.plots.plot_cumulative_returns(pnl, base_pnl)
checkret.plots.plot_drawdown(pnl)
checkret.plots.plot_monthly_heatmap(pnl)
checkret.plots.plot_yearly_returns(pnl, base_pnl)
checkret.plots.plot_return_distribution(pnl, base_pnl)
checkret.plots.plot_rolling_sharpe(pnl, base_pnl)
checkret.plots.plot_rolling_volatility(pnl, base_pnl)
checkret.plots.plot_dow_returns(pnl)
```

## Metrics produced

| metric | description |
|--------|-------------|
| Total Return | Compounded return over the full period |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Annualized risk-adjusted return |
| Sortino Ratio | Sharpe using only downside deviation |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | CAGR / \|Max Drawdown\| |
| Volatility (ann.) | Annualized standard deviation |
| Win Rate | % of days with positive returns |
| Profit Factor | Gross profit / gross loss |
| Best / Worst Day | Largest single-day gain / loss |
| Avg Win / Avg Loss | Mean return on winning / losing days |
| Daily VaR (95%) | 5th-percentile daily return |
| Recovery Factor | Total return / \|Max Drawdown\| |
| Skewness / Kurtosis | Distribution shape statistics |

