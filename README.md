# Stock Market Portfolio Optimization

This project reproduces the Modern Portfolio Theory analysis.

## What it does

- Downloads adjusted daily prices for VRTX, SPCX, NVDA, META, TSLA, AMC, and TEM.
- Checks which stocks have enough history for the five-year common-window model.
- Calculates daily returns, annualized expected returns, volatility, covariance, and correlation.
- Simulates 25,000 long-only portfolios whose weights total 100% and are capped at 40% per stock.
- Identifies the sampled maximum-Sharpe and minimum-volatility portfolios.
- Compares them with an equal-weight portfolio.
- Exports CSV tables, a JSON summary, and three presentation-ready charts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the real analysis

```powershell
python portfolio_optimization.py
```

The default date range includes July 20, 2021 through July 20, 2026. Outputs are saved under `outputs`.

## Test without internet

```powershell
python portfolio_optimization.py --demo --portfolios 5000 --output-directory demo_outputs
```

The demo verifies the complete workflow with deterministic synthetic data; its numbers will not match the presentation.

If Matplotlib is not installed, validate the calculations and table exports with:

```powershell
python portfolio_optimization.py --demo --skip-charts --output-directory demo_outputs
```

## Important assumptions

- Historical arithmetic-average returns are used as estimates.
- Covariance is estimated from the historical sample.
- 252 trading days are used for annualization.
- Portfolios are long-only and fully invested.
- Individual stocks are capped at 40%.
- The risk-free rate is 4%.
- Transaction costs, taxes, liquidity, and investor-specific preferences are omitted.

This is an educational historical analysis, not investment advice.
