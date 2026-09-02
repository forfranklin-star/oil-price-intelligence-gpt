# 多因素油价智能分析与预测系统（真实数据严格版）

本项目的第一原则：**生产数据、训练数据、日报数据绝不使用模拟、随机生成、插值补齐或价格代理伪造。**

如果关键真实数据源不可用、过期、解析失败或口径不一致，pipeline 必须失败，且不生成报告。

## 真实数据源

### 国际原油

1. FRED / EIA 官方 WTI：`DCOILWTICO`
2. FRED / EIA 官方 Brent Europe：`DCOILBRENTEU`
3. EIA Open Data API：可选 `EIA_API_KEY`
4. Yahoo Finance：仅作为真实市场数据传输备用/交叉校验，不作为唯一数据源

FRED 中 WTI/Brent 的来源均明确标记为 U.S. Energy Information Administration，频率为 Daily。

### 国内柴油

按完整序列优先级降级，不混合不同口径：

1. 隆众资讯：全国主流市场柴油价格，元/吨
2. 生意社：柴油基准价格/行业报价
3. 卓创资讯：仅读取无需登录、公开且明确标注元/吨柴油价格的页面
4. 配置的真实 CSV/API：用于企业授权数据源
5. 新浪财经公开页面：主要作为真实省级价格/行业数据的备用源与交叉校验
6. 国家发改委：作为国内成品油调价政策/标准品价格的权威校验源，不与隆众批发均价拼接训练

隆众公开页面存在按日期发布的国内汽柴油价格表；国家发改委公开发布成品油调价和各省（区、市）最高零售价格。

## 严格数据规则

- 不生成随机价格
- 不用前后值插值
- 不用 Brent × 汇率推算柴油历史价格
- 不把不同统计口径拼接为一条目标序列
- 不把失败的数据源静默替换成模拟数据
- 不用中位数填充模型特征
- 时间序列使用 point-in-time / as-of join：只能使用预测时点已经可获得的观测
- 没有完整真实特征向量的训练行直接删除
- SARIMAX 失败不会退化为“当前价格不变”的伪预测
- 关键数据不足时直接失败，不生成 `latest.json`

## 数据降级

```text
WTI / Brent
  FRED(EIA official)
       ↓失败
  EIA API
       ↓失败
  Yahoo query1
       ↓失败
  Yahoo query2
       ↓全部失败
  PIPELINE FAILED

中国柴油
  隆众
       ↓失败/不足/过期
  生意社
       ↓失败/不足/过期
  卓创公开页
       ↓失败/不足/过期
  授权/配置 CSV/API
       ↓失败/不足/过期
  新浪公开页
       ↓全部失败
  PIPELINE FAILED
```

备用源是**整段切换**，不是把不同口径的数据拼接起来。

## 首次运行

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python scripts/check_sources.py
python -m src.pipeline --print-summary
streamlit run streamlit_app.py
```

如果 `check_sources.py` 报某个来源不可访问，不要用假数据替代；修复网络、源地址、授权或 API Key 后重新运行。

## EIA API Key

如果有 EIA API Key：

```bash
# Linux/macOS
export EIA_API_KEY='YOUR_REAL_KEY'

# PowerShell
$env:EIA_API_KEY='YOUR_REAL_KEY'
```

GitHub Actions 中配置 Repository Secret：`EIA_API_KEY`。

## 国内柴油授权源

如果你拥有隆众/卓创授权接口，推荐配置真实 CSV/API。CSV 必须至少包含：

```csv
date,price
2026-08-28,....
```

这里的数值必须来自真实供应商，系统不会生成缺失值。

## 模型

- 训练窗口：最近 180 天的完整真实特征
- 权重：Random Forest + LASSO + 人工先验
- 两周：SARIMAX + ML signal tilt
- 三个月：RF/LASSO
- 长期：情景分析
- 预测前进行 point-in-time 数据对齐，避免未来数据泄漏

注意：**预测值本身当然是模型输出，不是“真实数据”；真实数据约束适用于输入、训练集、事件和历史序列。**

## GitHub Actions

`.github/workflows/daily_report.yml` 每天 UTC 02:00 运行，即北京时间 10:00 左右。也支持手动 `workflow_dispatch`。

## 生产原则

宁可当天没有报告，也不允许系统为了“页面有数字”而生成任何虚构数据。
