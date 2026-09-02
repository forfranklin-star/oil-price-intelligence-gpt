from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="多因素油价智能分析与预测系统", page_icon="🛢️", layout="wide")
st.title("多因素油价智能分析与预测系统")
st.caption("国际原油 × 国内柴油 × 宏观金融 × 地缘事件 × 机构观点 | 每日自动更新")

files = sorted([p for p in REPORTS.glob("*.json") if p.name != "latest.json"], reverse=True)
if not files and (REPORTS / "latest.json").exists():
    files = [REPORTS / "latest.json"]
if not files:
    err_file = REPORTS / "last_run_error.json"
    if err_file.exists():
        try:
            err = json.loads(err_file.read_text(encoding="utf-8"))
            st.error("日报尚未生成：上一次流水线失败。")
            st.code(err.get("error", "未知错误"), language="text")
            st.caption("请在项目根目录重新运行 `python -m src.pipeline --print-summary`；程序不会使用模拟数据。")
        except Exception:
            st.error("尚无报告。请先运行：python -m src.pipeline --print-summary")
    else:
        st.error("尚无报告。请先运行：python -m src.pipeline --print-summary")
    st.stop()

labels = {p.stem: p for p in files}
selected = st.sidebar.selectbox("历史报告日期", list(labels.keys()))
with labels[selected].open("r", encoding="utf-8") as f:
    r = json.load(f)

st.sidebar.caption(f"生成时间：{r['generated_at']}")
with st.sidebar.expander("数据源状态"):
    st.json(r["data_status"])

latest = r["latest"]
fs = r["forecasts"]["wti_short"]
fm = r["forecasts"]["wti_medium"]
fl = r["forecasts"]["wti_long"]

c1, c2, c3 = st.columns(3)
c1.metric("WTI 最新", f"${latest['wti']:.2f}/桶")
c2.metric("Brent 最新", f"${latest['brent']:.2f}/桶")
c3.metric("国内柴油", f"¥{latest['china_diesel']:.0f}/吨", help="仅显示真实观测的人民币/吨数据；没有通过质量校验的数据不会进入报告")

st.subheader("预测卡片")
a, b, c = st.columns(3)
with a:
    st.markdown("#### 两周")
    st.metric("WTI 中心值", f"${fs['mean']:.2f}")
    st.write(f"80% 区间：${fs['low']:.2f} – ${fs['high']:.2f}")
    st.write(f"看涨 / 看跌：{fs['bull_probability']:.0%} / {fs['bear_probability']:.0%}")
with b:
    st.markdown("#### 三个月")
    st.metric("WTI 中心值", f"${fm['mean']:.2f}")
    st.write(f"80% 区间：${fm['low']:.2f} – ${fm['high']:.2f}")
    st.write(f"看涨 / 看跌：{fm['bull_probability']:.0%} / {fm['bear_probability']:.0%}")
with c:
    st.markdown("#### 长期情景")
    st.metric("基准情景", f"${fl['base']['mean']:.2f}")
    st.write(f"低 / 高情景：${fl['low']['mean']:.2f} / ${fl['high']['mean']:.2f}")
    st.write(f"综合区间：${fl['range'][0]:.2f} – ${fl['range'][1]:.2f}")

st.subheader("国际原油历史与短期预测")
wti = pd.DataFrame(r["history"]["wti"]); wti["date"] = pd.to_datetime(wti["date"])
brent = pd.DataFrame(r["history"]["brent"]); brent["date"] = pd.to_datetime(brent["date"])
fc = pd.DataFrame(r["history"]["short_path"]); fc["date"] = pd.to_datetime(fc["date"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=wti["date"], y=wti["price"], name="WTI", mode="lines"))
fig.add_trace(go.Scatter(x=brent["date"], y=brent["price"], name="Brent", mode="lines"))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["high"], name="预测上界", mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["low"], name="80%区间", mode="lines", fill="tonexty", line=dict(width=0)))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["mean"], name="WTI预测", mode="lines"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("国内柴油走势")
diesel = pd.DataFrame(r["history"]["diesel"]); diesel["date"] = pd.to_datetime(diesel["date"])
fig2 = go.Figure(go.Scatter(x=diesel["date"], y=diesel["price"], name="柴油", mode="lines"))
fig2.update_yaxes(title="人民币/吨")
st.plotly_chart(fig2, use_container_width=True)
dsf, dmf = r["forecasts"]["diesel_short"], r["forecasts"]["diesel_medium"]
st.write(f"两周预测：¥{dsf['mean']:.0f}/吨（{dsf['low']:.0f}–{dsf['high']:.0f}）；三个月预测：¥{dmf['mean']:.0f}/吨（{dmf['low']:.0f}–{dmf['high']:.0f}）。")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("关键事件及量化影响")
    ev = pd.DataFrame(r["events"])
    if not ev.empty:
        ev["published"] = pd.to_datetime(ev["published"]).dt.strftime("%Y-%m-%d")
        display = ev[["published", "title", "direction", "impact_pct", "impact_usd", "current_weight", "source"]].copy()
        display.columns = ["日期", "事件", "方向", "估算影响%", "估算影响$/桶", "当前权重", "来源"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("当前没有事件数据。")
with right:
    st.subheader("模型因素权重")
    weights = pd.DataFrame([{"factor": k, "weight": v, "abs_weight": abs(v)} for k, v in r["weights"].items()]).sort_values("abs_weight")
    fig3 = go.Figure(go.Bar(x=weights["weight"], y=weights["factor"], orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("正负号代表模型方向，绝对值代表相对重要性；权重不是严格因果系数。")

st.subheader("自动文字解读")
for p in r["narrative"]:
    st.markdown(f"- {p}")

st.warning(r["disclaimer"])
