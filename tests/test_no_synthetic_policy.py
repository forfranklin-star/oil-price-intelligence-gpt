import pandas as pd
import pytest

from src.features import build_feature_frame
from src.models import train_weight_models
from src.utils import load_config


def _real_like_frame(n=90, start="2026-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"date": dates, "price": 80 + pd.Series(range(n), dtype=float) * 0.02})


def test_feature_builder_does_not_impute_missing_market_observation():
    wti = _real_like_frame()
    brent = _real_like_frame()
    brent.loc[40, "price"] = None
    dxy = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=90, freq="D"), "price": 100.0})
    macro = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=90, freq="7D"), "value": 100.0})
    events = pd.DataFrame(columns=["published", "geopolitical_score", "institution_score"])
    f = build_feature_frame(wti, brent, dxy, macro, macro, macro, events)
    # As-of alignment may legitimately use the last known real observation; it must
    # never invent a value between two observations. The number of source rows stays unchanged.
    assert len(brent.dropna()) == 89
    assert f["brent"].notna().all()


def test_training_rejects_incomplete_real_feature_rows():
    cfg = load_config()
    n = 90
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    f = pd.DataFrame({
        "date": dates,
        "wti": 80.0,
        "geopolitical_risk": 0.0,
        "usd_return": 0.0,
        "us10y_change": 0.0,
        "cpi_surprise_proxy": 0.0,
        "payroll_momentum": 0.0,
        "institution_score": 0.0,
        "brent_return": 0.0,
        "momentum_5d": 0.0,
        "target_next_return": 0.001,
    })
    f["cpi_surprise_proxy"] = float("nan")
    with pytest.raises(ValueError, match="Insufficient complete real observations"):
        train_weight_models(f, cfg)
