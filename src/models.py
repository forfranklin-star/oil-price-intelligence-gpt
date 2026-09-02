from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

FEATURES = [
    "geopolitical_risk", "usd_return", "us10y_change", "cpi_surprise_proxy",
    "payroll_momentum", "institution_score", "brent_return", "momentum_5d",
]


@dataclass
class ModelBundle:
    weights: dict[str, float]
    rf: object
    lasso: object
    residual_sigma: float


def train_weight_models(features: pd.DataFrame, cfg: dict) -> ModelBundle:
    window = cfg["app"]["rolling_window_days"]
    train = features.dropna(subset=["target_next_return"]).tail(window).copy()
    X, y = train[FEATURES], train["target_next_return"]

    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=cfg["app"]["random_seed"])),
    ]).fit(X, y)
    lasso = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.00015, max_iter=20000, random_state=cfg["app"]["random_seed"])),
    ]).fit(X, y)

    rf_imp = rf.named_steps["model"].feature_importances_
    coef = lasso.named_steps["model"].coef_
    signed_rf = np.sign(coef + 1e-12) * rf_imp
    learned = {f: float(v) for f, v in zip(FEATURES, signed_rf)}
    denom = sum(abs(v) for v in learned.values()) or 1.0
    learned = {k: v / denom for k, v in learned.items()}

    manual = cfg["weights"]["manual"]
    a = float(cfg["weights"]["manual_prior_strength"])
    blended = {f: a * manual.get(f, 0.0) + (1 - a) * learned.get(f, 0.0) for f in FEATURES}
    denom = sum(abs(v) for v in blended.values()) or 1.0
    blended = {k: v / denom for k, v in blended.items()}

    pred = 0.5 * rf.predict(X) + 0.5 * lasso.predict(X)
    sigma = float(np.nanstd(y - pred)) or 0.015
    return ModelBundle(blended, rf, lasso, sigma)


def _probabilities(mean_price: float, current: float, sigma_price: float) -> tuple[float, float]:
    # Normal approximation without scipy.
    if sigma_price <= 0:
        return (1.0 if mean_price > current else 0.0, 1.0 if mean_price < current else 0.0)
    z = (mean_price - current) / sigma_price
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    bull = float(np.clip(cdf, 0.01, 0.99))
    return bull, 1 - bull


def short_forecast(features: pd.DataFrame, bundle: ModelBundle, days: int = 14) -> pd.DataFrame:
    series = features.dropna(subset=["wti"]).tail(240).set_index("date")["wti"]
    try:
        fit = SARIMAX(series, order=(1, 1, 1), trend="t", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(days)
        mean = fc.predicted_mean.to_numpy(float)
        ci = fc.conf_int(alpha=0.20).to_numpy(float)
    except Exception:
        current = float(series.iloc[-1])
        mean = np.repeat(current, days)
        vol = bundle.residual_sigma * current
        steps = np.arange(1, days + 1)
        ci = np.column_stack([mean - 1.282 * vol * np.sqrt(steps), mean + 1.282 * vol * np.sqrt(steps)])

    latest_x = features[FEATURES].tail(1)
    ml_ret = float(0.5 * bundle.rf.predict(latest_x)[0] + 0.5 * bundle.lasso.predict(latest_x)[0])
    drift = np.exp(np.clip(ml_ret, -0.03, 0.03) * np.arange(1, days + 1) * 0.35)
    mean = mean * drift
    ci[:, 0] *= drift
    ci[:, 1] *= drift
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    return pd.DataFrame({"date": dates, "mean": mean, "low": ci[:, 0], "high": ci[:, 1]})


def medium_forecast(features: pd.DataFrame, bundle: ModelBundle, days: int = 90) -> dict:
    current = float(features["wti"].dropna().iloc[-1])
    latest_x = features[FEATURES].tail(1)
    mu = float(0.5 * bundle.rf.predict(latest_x)[0] + 0.5 * bundle.lasso.predict(latest_x)[0])
    mu = float(np.clip(mu, -0.004, 0.004))
    horizon_scale = 0.45
    mean = current * math.exp(mu * days * horizon_scale)
    sigma = bundle.residual_sigma * math.sqrt(days) * current
    low, high = mean - 1.282 * sigma, mean + 1.282 * sigma
    bull, bear = _probabilities(mean, current, sigma)
    return {"mean": mean, "low": max(1.0, low), "high": high, "bull_probability": bull, "bear_probability": bear, "model": "RF+LASSO macro scenario"}


def long_scenarios(features: pd.DataFrame, bundle: ModelBundle) -> dict:
    current = float(features["wti"].dropna().iloc[-1])
    geo = float(features["geopolitical_risk"].tail(30).mean())
    inst = float(features["institution_score"].tail(30).mean())
    center_shift = np.clip(0.03 * geo + 0.04 * inst, -0.15, 0.20)
    center = current * (1 + center_shift)
    return {
        "low": {"mean": center * 0.78, "probability": 0.25},
        "base": {"mean": center, "probability": 0.50},
        "high": {"mean": center * 1.28, "probability": 0.25},
        "range": [center * 0.70, center * 1.38],
        "model": "structural scenario + news/institution tilt",
    }
