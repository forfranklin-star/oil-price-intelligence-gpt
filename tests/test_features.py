import pandas as pd

from src.features import build_feature_frame
from src.providers.news import NewsProvider


def test_news_scoring_direction():
    geo, inst, direction = NewsProvider.score_text("attack causes oil supply disruption and Goldman raise forecast")
    assert geo > 0
    assert inst > 0
    assert direction == "看涨"


def test_feature_frame_has_target():
    d = pd.date_range("2025-01-01", periods=120, freq="D")
    px = pd.DataFrame({"date": d, "price": range(70, 190)})
    macro = pd.DataFrame({"date": d, "value": range(100, 220)})
    events = pd.DataFrame(columns=["published", "geopolitical_score", "institution_score"])
    f = build_feature_frame(px, px, px, macro, macro, macro, events)
    assert "target_next_return" in f.columns
    assert len(f) >= 120
