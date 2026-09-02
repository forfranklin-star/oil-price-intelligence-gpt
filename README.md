# 多因素油价智能分析与预测系统 2.0（真实数据版）

## 核心原则
本项目不内置任何模拟历史数据，也不在运行时生成随机/插值/填充价格。真实数据不足时，Pipeline 失败，不生成日报。

## 数据源
- WTI / Brent：优先 Yahoo Finance 实时/延迟市场数据；失败则使用 FRED 的 EIA 日度现货序列 `DCOILWTICO` / `DCOILBRENTEU`。
- DXY：优先 Yahoo 的 `DX-Y.NYB`；失败则使用 FRED `DTWEXBGS`，并在报告中明确标为 Broad Trade-Weighted USD，不冒充 ICE DXY。
- CNY/USD：优先 Yahoo `CNY=X`；失败则 FRED `DEXCHUS`。
- 中国柴油：优先公开可访问的隆众柴油价格页面，独立交叉校验生意社公开页面；不绕过登录、验证码或订阅权限。

FRED 的 WTI/Brent 日度序列明确标注来源为 U.S. Energy Information Administration。当前网页核验也显示隆众柴油栏目在 2026-09-01 有当天的全国/区域成品油文章。

## 运行
```bash
pip install -r requirements.txt
python -m src.pipeline --print-summary
streamlit run streamlit_app.py
```

也可以直接打开 Streamlit，点击“立即采集真实数据并生成日报”。

## 自动日报
GitHub Actions 每天 UTC 02:00 运行，即北京时间 10:00。

## 重要
如果真实数据源不可访问，系统会明确失败。**不会为了让页面显示数字而伪造数据。**
