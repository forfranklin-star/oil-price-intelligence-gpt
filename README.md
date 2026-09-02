# 多因素油价智能分析与预测系统

一个可部署到 Streamlit Community Cloud 的端到端 MVP：每天自动拉取市场、宏观与新闻数据，写入 SQLite，做滚动特征工程与模型训练，输出短期/中期/长期预测和可解释权重，并保存每日历史报告。

> **重要**：这是研究型预测系统，不构成投资、交易或采购建议。事件影响数值是模型估算的边际影响，不是因果认定。若 `config.yaml` 未配置真实国内柴油数据，系统会明确使用 Brent+CNY 代理序列。

## 1. 项目结构

```text
oil-price-intelligence/
├─ streamlit_app.py                 # 交互式报告入口
├─ config.yaml                      # 数据源、模型、权重配置
├─ requirements.txt
├─ Makefile
├─ src/
│  ├─ pipeline.py                   # 每日 ETL + 训练 + 预测 + 存档
│  ├─ features.py                   # 特征工程
│  ├─ models.py                     # RF/LASSO + SARIMAX + 情景模型
│  ├─ explain.py                    # 权重解释、事件影响与文字解读
│  ├─ storage.py                    # SQLite 存储
│  ├─ utils.py
│  └─ providers/
│     ├─ market.py                  # WTI/Brent/DXY/CNY，失败自动降级
│     ├─ macro.py                   # FRED CSV：10Y/CPI/非农/美元广义指数
│     ├─ news.py                    # Google News RSS + 规则评分
│     └─ diesel.py                  # 国内柴油多服务器 Provider + failover/交叉校验
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ features/
│  └─ oil_intelligence.db           # 首次运行后生成
├─ reports/
│  ├─ latest.json
│  └─ YYYY-MM-DD.json               # 历史日报
├─ models/
├─ tests/
└─ .github/workflows/daily_report.yml
```

## 2. 快速运行

建议 Python 3.12。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --print-summary
streamlit run streamlit_app.py
```

打开 Streamlit 输出的本地地址即可查看报告。

## 3. 数据源与回退策略

### 国际市场

默认使用 Yahoo Finance Chart 兼容接口读取：

- WTI：`CL=F`
- Brent：`BZ=F`
- DXY：`DX-Y.NYB`
- CNY/USD：`CNY=X`

旧版本曾在接口异常时静默回退到确定性模拟数据，这会导致日报看起来“有数据”但实际上是假的。**当前版本已经移除该行为**：真实行情源失败、数据超过新鲜度阈值或数据质量检查失败时，任务直接失败，不生成伪造日报。WTI/Brent 优先使用 EIA 官方石油现货数据（需要 `EIA_API_KEY`），否则使用 Yahoo Chart 的延迟行情作为后备，并强制检查最新观测不超过 3 天。

### 宏观数据

使用 FRED Graph CSV 公开端点：

- `DGS10`：美国 10 年期国债收益率
- `CPIAUCSL`：美国 CPI
- `PAYEMS`：非农就业
- `DTWEXBGS`：美元广义指数（作为 DXY 的官方宏观补充）

FRED 正式 API 也支持按 series 获取 observations；如你需要 vintage/revision 研究，可替换为带 API Key 的 FRED API。

### 新闻、地缘政治和机构观点

默认使用 Google News RSS 搜索，按关键词生成：

- `geopolitical_score`：战争、袭击、制裁、供应中断、停火、增产等事件风险分数；
- `institution_score`：Goldman/JPMorgan/UBS/IEA/EIA/OPEC 等机构观点方向分数。

这是轻量 MVP。生产版建议接入授权新闻 API，并用中文/英文金融 NLP 模型替代词典规则，同时保留原始新闻 URL、发布时间和去重指纹。

### 国内柴油：多服务器真实数据 + 整段降级

国内柴油不再依赖单一 CSV。`src/providers/diesel.py` 内置独立 Provider 列表：

| 优先级 | 数据源 | 服务器 | 主要用途 | 口径 |
|---|---|---|---|---|
| 1 | 隆众资讯 | `oilchem.net` / `list.oilchem.net` | 主训练序列 | 全国主流市场柴油均价，元/吨 |
| 2 | 生意社 | `100ppi.com` | 同类批发/基准价降级与校验 | 柴油基准/现货价，元/吨 |
| 3 | 卓创资讯 | `sci99.com` | 公开页面可用时的行业价格降级 | 仅接受页面明确公开的柴油元/吨价格 |
| 4 | 自定义 CSV/API | 任意独立服务器 | 企业授权数据或采购数据 | 由调用方定义，必须为真实观测 |
| 5 | 新浪财经 | `finance.sina.com.cn` | 省级零售/政策价交叉校验、必要时末级降级 | 只读取明确的元/吨值，不把元/升换算成元/吨 |

降级不是把不同来源拼接到一条时间序列。系统会先尝试取得**完整的同口径历史序列**，满足最小样本量和新鲜度要求后整段选用；该源不满足要求才切换到下一个 Provider。这样避免隆众均价、生意社基准价、省级最高零售价混拼导致结构性跳变。

隆众公开 7 日价格页不足以初始化模型时，Provider 会从其公开柴油资讯列表中回溯公开文章，并从含“中国/柴油/日期/价格”的表格补齐历史观测。不会访问登录后页面、不会处理验证码、不会绕过订阅限制。

所有源都会记录 `provider/server/metric/source_url/retrieved_at`；日报中的 `data_status.china_diesel_sources` 会留下成功、失败、拒绝训练及交叉校验信息。

默认校验规则：

- `min_diesel_rows: 30`：训练序列至少 30 个真实观测；
- `max_diesel_staleness_days: 7`：最新观测超过 7 天直接拒绝；
- 价格必须在 3000–20000 元/吨合理范围内；
- 批发类独立源最新值差异超过 15% 时，主源会被拒绝并继续降级；
- 元/升价格不做密度换算后冒充元/吨真实观测；
- 所有数据源失败时 pipeline **直接失败，不生成报告，不生成模拟数据**。

可在 `config.yaml -> diesel_sources` 调整启用状态、优先级、隆众回溯页数、卓创公开 URL 和新浪校验文章 URL。商用前请自行确认各供应商的数据授权、转载和缓存许可。

## 4. 模型设计

### 特征

当前包含：

- 地缘政治风险分数；
- 美元日收益率；
- 10Y 美债收益率变化；
- CPI “超预期代理”（基于趋势偏离，真实生产版应换成 actual-consensus）；
- 非农动能；
- 机构观点分数；
- Brent 日收益率；
- WTI 5 日动量。

### 权重

- Random Forest 给出非线性特征重要性；
- LASSO 给出稀疏方向；
- 用 LASSO 符号 × RF 重要性形成“学习权重”；
- 与 `config.yaml` 中人工先验权重按 `manual_prior_strength` 混合；
- 每天用最近 180 天滚动窗口重训，形成自适应权重。

权重用于“相对重要性与方向解释”，不是严格因果系数。

### 短期：两周

- SARIMAX(1,1,1) 预测 WTI 路径；
- RF/LASSO 下一日收益信号对 SARIMAX 路径做小幅倾斜；
- 输出 80% 概率区间与涨/跌概率。

### 中期：三个月

- RF/LASSO 宏观情景回归信号；
- 用滚动残差波动按平方根时间扩展区间；
- 输出中心值、80% 区间和涨跌概率。

生产版可进一步加入 VAR、XGBoost/LightGBM、宏观预期路径和 Monte Carlo 情景树。

### 长期

长期不做虚假的高精度点预测，使用低/基准/高三情景；中心由近期地缘政治和机构观点倾斜。生产版应加入 IEA/EIA/OPEC 长期供需平衡、OPEC spare capacity、美国页岩边际成本、EV 渗透率和机构目标价共识。

## 5. SQLite 数据表

首次运行后创建 `data/oil_intelligence.db`，包含：

- `prices`
- `macro`
- `events`
- `forecasts`
- `model_weights`
- `reports`

每日完整 JSON 同时写入 `reports/YYYY-MM-DD.json`，便于 Git 历史回溯和 Streamlit 历史日期选择。

## 6. GitHub Actions：每天北京时间 10:00

工作流：`.github/workflows/daily_report.yml`

```yaml
on:
  workflow_dispatch:
  schedule:
    - cron: "0 2 * * *"
```

GitHub Actions 的 schedule 默认按 UTC，因此 `02:00 UTC = 10:00 北京时间`。工作流会：

1. 拉取仓库；
2. 安装依赖；
3. 执行 `python -m src.pipeline`；
4. 将 `reports/`、`data/`、`models/` 的变化提交回默认分支；
5. Streamlit Community Cloud 监控 GitHub 仓库，提交后自动更新应用。

注意：GitHub scheduled workflow 并不保证严格在整点启动，平台繁忙时可能存在延迟。如果业务要求“10:00:00 准时”，应改用具备 SLA 的调度服务。

## 7. 部署到 Streamlit Community Cloud

1. 将整个目录推送到一个 GitHub **公开仓库**；
2. 登录 Streamlit Community Cloud 并连接 GitHub；
3. 选择该仓库、默认分支和入口 `streamlit_app.py`；
4. 部署；
5. 以后 GitHub 中的代码/报告提交会被 Community Cloud 自动读取并更新。

如果你将来加入付费 API Key，不要提交到仓库：在 GitHub Actions Secrets 和 Streamlit Secrets 中配置，然后通过环境变量读取。


### 生产环境必须配置的密钥

在 GitHub 仓库 `Settings -> Secrets and variables -> Actions` 中添加 `EIA_API_KEY`。不要把 API Key 写进 `config.yaml` 或提交到 Git。工作流会通过环境变量读取。

如果没有 EIA Key，WTI/Brent 会退回 Yahoo 延迟行情；如果 Yahoo 也不满足 3 天新鲜度阈值，任务会失败并保留错误日志，不会生成错误报告。

## 8. 接入真实国内柴油数据

推荐新建 Provider，而不是改页面代码。例如：

```python
class RealDieselProvider:
    def history(self) -> tuple[pd.DataFrame, str]:
        # 1. 调授权 API
        # 2. 校验字段/日期/异常值
        # 3. 返回 date, price
        return df, "vendor_name"
```

然后在 `pipeline.py` 中替换 `ChinaDieselProvider` 即可。建议同时存储：省份、0#/-10#牌号、批发/零售类型、含税口径、单位和来源。

## 9. 生产化增强建议

- 将 CPI/非农从“趋势代理”升级为 `actual - consensus` 的 surprise；
- CME FedWatch 改用获授权数据源或可靠财经 API，不建议通过页面逆向绕过限制；
- 新闻使用授权供应商 + FinBERT/中文金融情感模型 + LLM 结构化事件抽取；
- 加入 EIA 库存、OPEC 产量/闲置产能、炼厂开工率、裂解价差、期限结构、CFTC 持仓；
- 增加 walk-forward backtest、MAE/RMSE/方向准确率、interval coverage；
- 用 MLflow/DVC 做模型与数据版本管理；
- 对数据缺失、源异常、预测漂移增加告警；
- 对报告 commit 做大小控制，SQLite 变大后迁移到对象存储/云数据库。

## 10. 合规与数据抓取

本项目没有实现验证码绕过、代理池、Cookie 盗用或其他反爬绕过技术。对于网页源：

- 检查并遵守 `robots.txt` 和网站条款；
- 设置明确 User-Agent、超时和请求间隔；
- 优先使用官方 API/RSS/公开下载文件；
- 商用前核对数据转载、缓存和公开展示许可。

## Real-data policy (strict)

This version is **fail-closed**. It never creates synthetic prices, demo news, random observations, or hidden Brent/CNY proxy diesel prices. A key market series that is empty, malformed, too short, or stale causes the daily job to fail rather than publish a misleading report. News can be absent and is then represented as missing/zero event features; it is never invented.

For WTI/Brent, the daily market series is fetched from Yahoo Finance's real market-history endpoint first; EIA spot series are a secondary fallback and are recorded separately as `eia_spot:*`. Yahoo's data is delayed, so the report labels the source and retrieval time.

For China diesel, production uses the configured multi-server provider registry (Longzhong, Business Society, SCI99 public pages when available, custom CSV/API, and Sina validation pages). Every accepted value must be a real observed RMB/tonne figure. Providers are selected as complete series by priority; incompatible metrics are not spliced together. If no source passes quality/freshness checks, the pipeline stops.

The current official evidence confirms that EIA publishes WTI/Brent spot histories and that NDRC publishes domestic product-price adjustments; provincial development-and-reform commissions publish concrete diesel retail/wholesale prices. These are useful source families, but an official announcement alone is not treated as a complete historical diesel time series.
