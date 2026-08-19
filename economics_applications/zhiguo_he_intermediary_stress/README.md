# Intermediary Stress and Credit-Spread Forecasting (NeuralForecast)

This application connects deep time-series models to a central theme in Zhiguo He's asset-pricing and intermediation research: **financial prices depend on intermediary balance-sheet states, not only on their own lags.**

The motivating research includes work on intermediary asset pricing and commonality in credit-spread changes, where intermediary distress/capital and dealer balance-sheet conditions are economically important state variables.

## Forecasting question

> Do nonlinear neural forecasting models extract useful predictive structure from intermediary capital, dealer inventory, defaults, and market volatility?

The example compares:

- **N-HiTS** — a deep hierarchical interpolation model,
- **Temporal Fusion Transformer (TFT)** — an attention-based multi-horizon forecasting model.

Both receive historical exogenous state variables:

```text
intermediary_capital
dealer_inventory
default_rate
market_volatility
```

The synthetic DGP contains a nonlinear amplification mechanism: low intermediary capital matters more in stressed states.

## Run

```bash
python economics_applications/zhiguo_he_intermediary_stress/credit_spread_forecasting.py
```

The script holds out the final 12 months and reports forecast errors.

## Economic extension beyond pure forecasting

A useful research project should not stop at "Transformer beats ARIMA." The economically relevant questions are:

1. Does intermediary capital add predictive content beyond standard macro/market variables?
2. Is its contribution asymmetric in good versus stressed states?
3. Does model attention/feature importance line up with intermediary-constraint mechanisms?
4. Do shocks propagate differently when capital is scarce?
5. Is the forecast improvement economically large around crises rather than only on average?

## Public-data route

Zhiguo He's research site provides intermediary capital/risk-factor data associated with the intermediary asset-pricing work. A real version can combine those series with corporate credit spreads, dealer inventories, default rates, and volatility measures.

For a rigorous comparison, use a rolling/expanding-window evaluation and benchmark against AR/VAR, random forest/boosting, and a parsimonious factor model.

## Structural caution

Forecasting and causal identification answer different questions. Strong predictive contribution from intermediary capital is consistent with an economically important state variable, but it does not by itself identify a causal supply effect. That distinction is deliberate and complements the causal applications in the other forks.

This is an **inspired application, not a replication** and does not imply author endorsement.

References:
- https://zhiguohe.net/publications/research/
- https://nixtlaverse.nixtla.io/neuralforecast/
