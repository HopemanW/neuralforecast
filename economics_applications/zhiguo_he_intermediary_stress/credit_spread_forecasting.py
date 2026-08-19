"""Intermediary state variables and credit-spread forecasting.

Inspired by Zhiguo He's work on intermediary asset pricing, credit-spread
commonality, balance-sheet constraints, and financial-market dynamics.

This example tests whether nonlinear forecasting improves when the model
observes intermediary capital and dealer/inventory state variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT


HORIZON = 12
HIST_EXOG = [
    "intermediary_capital",
    "dealer_inventory",
    "default_rate",
    "market_volatility",
]


def simulate_monthly_credit_market(n_periods: int = 240, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_periods, freq="ME")

    capital = np.zeros(n_periods)
    inventory = np.zeros(n_periods)
    default_rate = np.zeros(n_periods)
    volatility = np.zeros(n_periods)
    spread = np.zeros(n_periods)

    capital[0] = 0.0
    inventory[0] = 0.0
    default_rate[0] = 0.02
    volatility[0] = 0.18
    spread[0] = 180

    for t in range(1, n_periods):
        capital[t] = 0.88 * capital[t - 1] + rng.normal(scale=0.35)
        inventory[t] = (
            0.70 * inventory[t - 1] - 0.25 * capital[t] + rng.normal(scale=0.45)
        )
        default_rate[t] = np.clip(
            0.86 * default_rate[t - 1]
            + 0.006
            - 0.002 * capital[t]
            + rng.normal(scale=0.004),
            0.001,
            0.20,
        )
        volatility[t] = np.clip(
            0.80 * volatility[t - 1]
            + 0.04
            - 0.012 * capital[t]
            + rng.normal(scale=0.025),
            0.05,
            0.80,
        )

        # Nonlinear intermediary-capital amplification:
        # weak capital has a larger marginal effect in stressed states.
        stress = max(-capital[t], 0)
        spread[t] = (
            95
            + 0.72 * spread[t - 1]
            - 20 * capital[t]
            + 10 * inventory[t]
            + 650 * default_rate[t]
            + 95 * volatility[t]
            + 18 * stress**2
            + rng.normal(scale=12)
        )

    return pd.DataFrame(
        {
            "unique_id": "credit_spread",
            "ds": dates,
            "y": spread,
            "intermediary_capital": capital,
            "dealer_inventory": inventory,
            "default_rate": default_rate,
            "market_volatility": volatility,
        }
    )


def fit_models(df: pd.DataFrame, horizon: int = HORIZON, seed: int = 31):
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    models = [
        NHITS(
            h=horizon,
            input_size=36,
            hist_exog_list=HIST_EXOG,
            max_steps=300,
            random_seed=seed,
        ),
        TFT(
            h=horizon,
            input_size=36,
            hist_exog_list=HIST_EXOG,
            max_steps=300,
            random_seed=seed,
        ),
    ]
    nf = NeuralForecast(models=models, freq="ME")
    nf.fit(df=train)
    forecast = nf.predict().reset_index()

    # NeuralForecast returns one row per horizon observation for each series.
    forecast["actual"] = test["y"].to_numpy()
    for col in [c for c in forecast.columns if c not in {"unique_id", "ds", "actual"}]:
        forecast[f"abs_error_{col}"] = np.abs(forecast[col] - forecast["actual"])
    return forecast


if __name__ == "__main__":
    data = simulate_monthly_credit_market()
    forecast = fit_models(data)
    print(forecast.to_string(index=False))

    mae_cols = [c for c in forecast if c.startswith("abs_error_")]
    print("\nMean absolute errors:")
    print(forecast[mae_cols].mean().to_string())
