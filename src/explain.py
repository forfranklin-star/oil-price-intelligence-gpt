from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "geopolitical_risk": "地缘政治风险",
    "usd_return": "美元指数变化",
    "us10y_change": "美国10Y国债收益率",
    "cpi_surprise_proxy": "通胀超预期代理",
    "payroll_momentum": "非农就业动能",
    "institution_score": "机构观点",
    "brent_return": "布伦特联动",
    "momentum_5d": "WTI 5日动量",
}


def event_impacts(events: pd.DataFrame, weights: dict, current_wti: float, residual_sigma: float) -> pd.DataFrame:
    if events.empty:
        return events
    x = events.copy()
    w_geo = weights.get("geopolitical_risk", 0)
    w_inst = weights.get("institution_score", 0)
    raw = x["geopolitical_score"] * w_geo + x["institution_score"] * w_inst
    # calibrated, deliberately conservative approximation of marginal impact.
    x["impact_pct"] = np.clip(raw * residual_sigma * 100 * 1.8, -4.0, 4.0)
    x["impact_usd"] = current_wti * x["impact_pct"] / 100
    x["current_weight"] = np.where(x["institution_score"].abs() > x["geopolitical_score"].abs(), abs(w_inst), abs(w_geo))
    return x


def narrative(current: float, short: dict, medium: dict, longterm: dict, weights: dict) -> list[str]:
    top = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    top_text = "、".join(f"{LABELS.get(k,k)}({v:+.0%})" for k, v in top)
    short_dir = "偏强" if short["mean"] > current else "偏弱"
    medium_dir = "偏强" if medium["mean"] > current else "偏弱"
    return [
        f"短期（约两周）模型中心值为 {short['mean']:.2f} 美元/桶，较当前价格呈{short_dir}格局；80%概率区间为 {short['low']:.2f}–{short['high']:.2f}。",
        f"中期（三个月）中心值为 {medium['mean']:.2f} 美元/桶，方向{medium_dir}；看涨概率约 {medium['bull_probability']:.0%}，区间宽度主要反映宏观变量与历史残差波动。",
        f"当前权重最高的驱动因素为：{top_text}。权重是滚动样本机器学习结果与人工先验的混合，不应解释为严格因果关系。",
        f"长期采用情景分析而非点预测：基准情景约 {longterm['base']['mean']:.2f} 美元/桶，压力/高价情景分别约 {longterm['low']['mean']:.2f}/{longterm['high']['mean']:.2f}。",
    ]
