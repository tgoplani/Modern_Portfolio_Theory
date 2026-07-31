"""Modern Portfolio Theory analysis with historical price data.

The script:
1. Downloads adjusted daily closing prices.
2. Excludes stocks without enough common history.
3. Engineers daily returns and annualized statistics.
4. Simulates constrained long-only portfolios.
5. Finds maximum-Sharpe and minimum-volatility candidates.
6. Exports data tables, charts, and a JSON summary.

This is an educational historical analysis, not investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["VRTX", "SPCX", "NVDA", "META", "TSLA", "AMC", "TEM"]
TRADING_DAYS = 252

COLORS = {
    "navy": "#070154",
    "blue": "#0047FF",
    "magenta": "#F900D3",
    "cyan": "#00CFE8",
    "slate": "#50658E",
    "pale": "#CED7E6",
}

TICKER_COLORS = {
    "VRTX": COLORS["cyan"],
    "NVDA": COLORS["magenta"],
    "META": COLORS["navy"],
    "TSLA": COLORS["blue"],
    "AMC": COLORS["slate"],
}


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    start: str
    end: str
    minimum_observations: int
    portfolios: int
    maximum_asset_weight: float
    risk_free_rate: float
    seed: int
    output_directory: Path
    demo: bool
    skip_charts: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Reproduce the MPT portfolio-optimization analysis.",
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2021-07-20")
    parser.add_argument(
        "--end",
        default="2026-07-21",
        help="Exclusive end date; 2026-07-21 includes prices through 2026-07-20.",
    )
    parser.add_argument("--minimum-observations", type=int, default=1000)
    parser.add_argument("--portfolios", type=int, default=25_000)
    parser.add_argument("--maximum-asset-weight", type=float, default=0.40)
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--output-directory", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data to test the workflow without internet access.",
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Export tables and summary only; useful when Matplotlib is unavailable.",
    )
    args = parser.parse_args()
    return Config(
        tickers=[ticker.upper() for ticker in args.tickers],
        start=args.start,
        end=args.end,
        minimum_observations=args.minimum_observations,
        portfolios=args.portfolios,
        maximum_asset_weight=args.maximum_asset_weight,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
        output_directory=args.output_directory.resolve(),
        demo=args.demo,
        skip_charts=args.skip_charts,
    )


def download_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download one ticker at a time so partial-history failures are visible."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for live downloads. "
            "Run `pip install -r requirements.txt` or use --demo."
        ) from exc

    series: dict[str, pd.Series] = {}
    failures: list[str] = []

    for ticker in tickers:
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if frame.empty:
                raise ValueError("no rows returned")

            price: pd.Series | pd.DataFrame | None = None
            if isinstance(frame.columns, pd.MultiIndex):
                for field in ("Adj Close", "Close"):
                    if field in frame.columns.get_level_values(0):
                        price = frame[field]
                        break
            else:
                for field in ("Adj Close", "Close"):
                    if field in frame.columns:
                        price = frame[field]
                        break

            if price is None:
                raise ValueError("adjusted-close and close columns are missing")
            if isinstance(price, pd.DataFrame):
                price = price.iloc[:, 0]

            clean = pd.to_numeric(price, errors="coerce").dropna()
            clean.index = pd.DatetimeIndex(clean.index).tz_localize(None)
            clean.name = ticker
            if clean.empty:
                raise ValueError("all downloaded prices were missing")
            series[ticker] = clean
        except Exception as exc:  # continue so one missing ticker does not end the run
            failures.append(f"{ticker}: {exc}")

    if not series:
        raise RuntimeError("No ticker data was downloaded.\n" + "\n".join(failures))
    if failures:
        print("Download warnings:")
        for failure in failures:
            print(f"  - {failure}")

    return pd.concat(series.values(), axis=1).sort_index()


def generate_demo_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    seed: int,
) -> pd.DataFrame:
    """Generate deterministic synthetic data for a network-free smoke test."""
    dates = pd.bdate_range(start=start, end=pd.Timestamp(end) - pd.Timedelta(days=1))
    rng = np.random.default_rng(seed)
    drifts = {
        "VRTX": 0.16,
        "SPCX": 0.20,
        "NVDA": 0.38,
        "META": 0.18,
        "TSLA": 0.22,
        "AMC": -0.25,
        "TEM": 0.12,
    }
    volatilities = {
        "VRTX": 0.28,
        "SPCX": 0.55,
        "NVDA": 0.50,
        "META": 0.42,
        "TSLA": 0.58,
        "AMC": 0.95,
        "TEM": 0.62,
    }
    data: dict[str, pd.Series] = {}

    for ticker in tickers:
        drift = drifts.get(ticker, 0.10)
        volatility = volatilities.get(ticker, 0.35)
        daily = rng.normal(
            drift / TRADING_DAYS,
            volatility / np.sqrt(TRADING_DAYS),
            len(dates),
        )
        values = 100 * np.exp(np.cumsum(daily))
        item = pd.Series(values, index=dates, name=ticker)
        if ticker == "SPCX":
            item = item.tail(25)
        elif ticker == "TEM":
            item = item.tail(524)
        data[ticker] = item

    return pd.concat(data.values(), axis=1).sort_index()


def coverage_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in prices.columns:
        observed = prices[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "start": observed.index.min().date().isoformat() if len(observed) else None,
                "end": observed.index.max().date().isoformat() if len(observed) else None,
                "observations": int(len(observed)),
                "years_approx": len(observed) / TRADING_DAYS,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def select_model_assets(
    prices: pd.DataFrame,
    coverage: pd.DataFrame,
    minimum_observations: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    eligible = coverage.index[
        coverage["observations"] >= minimum_observations
    ].tolist()
    excluded = [ticker for ticker in coverage.index if ticker not in eligible]
    if len(eligible) < 2:
        raise ValueError(
            "At least two tickers need the minimum history. "
            f"Eligible tickers: {eligible}"
        )
    aligned = prices[eligible].dropna(how="any")
    if len(aligned) < minimum_observations:
        raise ValueError(
            "Individually eligible assets do not have enough common dates. "
            f"Common observations: {len(aligned)}"
        )
    return aligned, eligible, excluded


def calculate_statistics(
    aligned_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    returns = aligned_prices.pct_change(fill_method=None).dropna(how="any")
    expected_returns = returns.mean() * TRADING_DAYS
    covariance = returns.cov() * TRADING_DAYS
    volatility = pd.Series(
        np.sqrt(np.diag(covariance)),
        index=covariance.index,
        name="annual_volatility",
    )
    correlation = returns.corr()
    return returns, expected_returns, covariance, volatility, correlation


def draw_constrained_weights(
    rng: np.random.Generator,
    assets: int,
    count: int,
    maximum_asset_weight: float,
) -> np.ndarray:
    if maximum_asset_weight * assets < 1 - 1e-12:
        raise ValueError(
            "The maximum-asset-weight constraint is infeasible: "
            "maximum weight × number of assets must be at least 1."
        )

    accepted: list[np.ndarray] = []
    remaining = count
    attempts = 0
    while remaining:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Could not draw enough feasible portfolios.")
        batch_size = max(10_000, remaining * 4)
        batch = rng.dirichlet(np.ones(assets), size=batch_size)
        feasible = batch[batch.max(axis=1) <= maximum_asset_weight + 1e-12]
        if len(feasible):
            take = feasible[:remaining]
            accepted.append(take)
            remaining -= len(take)
    return np.vstack(accepted)


def simulate_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    count: int,
    maximum_asset_weight: float,
    risk_free_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = draw_constrained_weights(
        rng=rng,
        assets=len(expected_returns),
        count=count,
        maximum_asset_weight=maximum_asset_weight,
    )
    mu = expected_returns.to_numpy()
    sigma = covariance.to_numpy()
    portfolio_returns = weights @ mu
    variances = np.einsum("ij,jk,ik->i", weights, sigma, weights)
    volatility = np.sqrt(np.maximum(variances, 0))
    sharpe = np.divide(
        portfolio_returns - risk_free_rate,
        volatility,
        out=np.full_like(volatility, np.nan),
        where=volatility > 0,
    )
    result = pd.DataFrame(
        {
            "expected_return": portfolio_returns,
            "volatility": volatility,
            "sharpe": sharpe,
        }
    )
    for index, ticker in enumerate(expected_returns.index):
        result[f"weight_{ticker}"] = weights[:, index]
    return result


def portfolio_metrics(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, float]:
    expected = float(weights @ expected_returns.to_numpy())
    variance = float(weights @ covariance.to_numpy() @ weights)
    volatility = float(np.sqrt(max(variance, 0)))
    sharpe = (expected - risk_free_rate) / volatility if volatility else np.nan
    return {
        "expected_return": expected,
        "volatility": volatility,
        "sharpe": float(sharpe),
    }


def efficient_frontier(portfolios: pd.DataFrame) -> pd.DataFrame:
    """Return the sampled upper envelope after sorting from low to high risk."""
    ordered = portfolios.sort_values(
        ["volatility", "expected_return"],
        ascending=[True, False],
    ).reset_index(drop=True)
    running_best = ordered["expected_return"].cummax()
    previous_best = running_best.shift(fill_value=-np.inf)
    return ordered.loc[ordered["expected_return"] > previous_best].copy()


def selected_results(
    portfolios: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_sharpe = portfolios.loc[portfolios["sharpe"].idxmax()]
    minimum_volatility = portfolios.loc[portfolios["volatility"].idxmin()]
    equal_weights = np.repeat(1 / len(expected_returns), len(expected_returns))
    equal_metrics = portfolio_metrics(
        equal_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    )

    summary = pd.DataFrame(
        [
            {
                "portfolio": "maximum_sharpe",
                "expected_return": max_sharpe["expected_return"],
                "volatility": max_sharpe["volatility"],
                "sharpe": max_sharpe["sharpe"],
            },
            {
                "portfolio": "minimum_volatility",
                "expected_return": minimum_volatility["expected_return"],
                "volatility": minimum_volatility["volatility"],
                "sharpe": minimum_volatility["sharpe"],
            },
            {"portfolio": "equal_weight", **equal_metrics},
        ]
    ).set_index("portfolio")

    weights = pd.DataFrame(index=expected_returns.index)
    weights["maximum_sharpe"] = [
        max_sharpe[f"weight_{ticker}"] for ticker in expected_returns.index
    ]
    weights["minimum_volatility"] = [
        minimum_volatility[f"weight_{ticker}"] for ticker in expected_returns.index
    ]
    weights["equal_weight"] = equal_weights
    weights.index.name = "ticker"
    return summary, weights


def chart_price_paths(aligned_prices: pd.DataFrame, output: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    normalized = aligned_prices.divide(aligned_prices.iloc[0]).multiply(100)
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for ticker in normalized:
        axis.plot(
            normalized.index,
            normalized[ticker],
            label=ticker,
            color=TICKER_COLORS.get(ticker),
            linewidth=2.2,
        )
    axis.set_title("Five-year price paths, normalized to 100", weight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Normalized price (start = 100)")
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=7))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("Jul %Y"))
    axis.grid(axis="y", color=COLORS["pale"], linewidth=0.8)
    axis.legend(ncol=len(normalized.columns), frameon=False, loc="upper center")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_frontier(
    portfolios: pd.DataFrame,
    frontier: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 7))
    sample = portfolios.iloc[:: max(1, len(portfolios) // 2500)]
    points = axis.scatter(
        sample["volatility"],
        sample["expected_return"],
        c=sample["sharpe"],
        cmap="viridis",
        s=10,
        alpha=0.35,
        label="Feasible portfolios",
    )
    axis.plot(
        frontier["volatility"],
        frontier["expected_return"],
        color=COLORS["cyan"],
        linewidth=3,
        label="Sampled efficient frontier",
    )
    axis.scatter(
        summary.loc["maximum_sharpe", "volatility"],
        summary.loc["maximum_sharpe", "expected_return"],
        color=COLORS["magenta"],
        marker="*",
        s=220,
        label="Maximum Sharpe",
        zorder=5,
    )
    axis.scatter(
        summary.loc["minimum_volatility", "volatility"],
        summary.loc["minimum_volatility", "expected_return"],
        color=COLORS["blue"],
        marker="D",
        s=70,
        label="Minimum volatility",
        zorder=5,
    )
    axis.set_title("Simulated portfolios and sampled efficient frontier", weight="bold")
    axis.set_xlabel("Annual volatility")
    axis.set_ylabel("Expected annual return")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(color=COLORS["pale"], linewidth=0.8)
    axis.legend(frameon=False)
    fig.colorbar(points, ax=axis, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_weights(weights: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    values = weights["maximum_sharpe"]
    colors = [TICKER_COLORS.get(ticker, COLORS["slate"]) for ticker in values.index]
    fig, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(values.index, values, color=colors)
    axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=4)
    axis.set_title("Historical maximum-Sharpe portfolio weights", weight="bold")
    axis.set_xlabel("Stock")
    axis.set_ylabel("Portfolio weight")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_ylim(0, max(0.45, values.max() * 1.18))
    axis.grid(axis="y", color=COLORS["pale"], linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="index")
    if isinstance(value, pd.Series):
        return value.to_dict()
    raise TypeError(f"Cannot convert {type(value).__name__} to JSON")


def run(config: Config) -> None:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    prices = (
        generate_demo_prices(config.tickers, config.start, config.end, config.seed)
        if config.demo
        else download_prices(config.tickers, config.start, config.end)
    )
    coverage = coverage_table(prices)
    aligned, eligible, excluded = select_model_assets(
        prices,
        coverage,
        config.minimum_observations,
    )
    returns, expected, covariance, volatility, correlation = calculate_statistics(aligned)
    portfolios = simulate_portfolios(
        expected_returns=expected,
        covariance=covariance,
        count=config.portfolios,
        maximum_asset_weight=config.maximum_asset_weight,
        risk_free_rate=config.risk_free_rate,
        seed=config.seed,
    )
    frontier = efficient_frontier(portfolios)
    summary, weights = selected_results(
        portfolios,
        expected,
        covariance,
        config.risk_free_rate,
    )

    annual_metrics = pd.DataFrame(
        {
            "expected_return": expected,
            "annual_volatility": volatility,
        }
    )

    coverage.to_csv(config.output_directory / "coverage.csv")
    aligned.to_csv(config.output_directory / "aligned_adjusted_prices.csv")
    returns.to_csv(config.output_directory / "daily_returns.csv")
    annual_metrics.to_csv(config.output_directory / "annual_asset_metrics.csv")
    covariance.to_csv(config.output_directory / "annual_covariance.csv")
    correlation.to_csv(config.output_directory / "correlation.csv")
    portfolios.to_csv(config.output_directory / "portfolio_candidates.csv", index=False)
    frontier.to_csv(config.output_directory / "efficient_frontier.csv", index=False)
    summary.to_csv(config.output_directory / "portfolio_summary.csv")
    weights.to_csv(config.output_directory / "optimized_weights.csv")

    if not config.skip_charts:
        chart_price_paths(aligned, config.output_directory / "price_paths.png")
        chart_frontier(
            portfolios,
            frontier,
            summary,
            config.output_directory / "efficient_frontier.png",
        )
        chart_weights(weights, config.output_directory / "optimized_weights.png")

    result = {
        "configuration": asdict(config),
        "source": "synthetic demo data" if config.demo else "Yahoo Finance via yfinance",
        "eligible_tickers": eligible,
        "excluded_tickers": excluded,
        "common_window": [
            aligned.index.min().date().isoformat(),
            aligned.index.max().date().isoformat(),
        ],
        "price_observations": len(aligned),
        "return_observations": len(returns),
        "coverage": coverage,
        "annual_asset_metrics": annual_metrics,
        "correlation": correlation,
        "portfolio_summary": summary,
        "optimized_weights": weights,
    }
    (config.output_directory / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, default=json_ready),
        encoding="utf-8",
    )

    print("\nEligible tickers:", ", ".join(eligible))
    print("Excluded tickers:", ", ".join(excluded) or "none")
    print("\nPortfolio summary:")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nMaximum-Sharpe weights:")
    print(
        weights["maximum_sharpe"].to_string(
            float_format=lambda value: f"{value:.1%}"
        )
    )
    print(f"\nFiles written to: {config.output_directory}")


if __name__ == "__main__":
    run(parse_args())
