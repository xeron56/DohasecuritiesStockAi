# DohasecuritiesStockAi

DohasecuritiesStockAi is a Bangladesh stock-research application for the Dhaka
Stock Exchange (DSE). It combines authenticated market data from the Doha
Securities gateway with a team of LLM agents that examine price action,
corporate disclosures, company fundamentals, competing investment arguments,
and portfolio risk.

The application is research software. It does not place orders, change an OMS
portfolio, or guarantee investment returns. Always verify important figures
against official DSE disclosures before making a financial decision.

## What it does

For a DSE ticker and analysis date, the application:

1. Resolves the company identity and normalizes ticker aliases.
2. Downloads historical candles and the latest data available on or before the
   selected date.
3. Collects company details, DSE announcements, financial performance,
   shareholding, dividends, NAV, EPS, NOCFPS, profit history, loans, and balance
   sheet data when those records are available.
4. Builds technical indicators such as moving averages, MACD, RSI, ATR, and
   volume-based measures from DSE candles.
5. Runs Market, DSE Disclosure, and Fundamentals analysts.
6. Sends their evidence to Bull and Bear researchers for a structured debate.
7. Produces a trader proposal and reviews it with aggressive, conservative, and
   neutral risk analysts.
8. Returns a Portfolio Manager decision with a rating, thesis, risk controls,
   price levels, and time horizon.
9. Saves the reports and completed state for later review, API access, or the
   optional local dashboard.

The normal workflow is:

```text
DSE data
  -> Market + Disclosure + Fundamentals analysts
  -> Bull/Bear research debate
  -> Trader proposal
  -> Risk-management debate
  -> Portfolio Manager decision
```

The default Bangladesh profile does not silently fall back to Yahoo Finance,
foreign media, Reddit, StockTwits, FRED, or prediction markets. Social-media,
macro, and prediction-market analysis are optional and disabled by default.

## Bangladesh market coverage

### Accepted symbols

Enter the base DSE trading code:

```text
GP
BRACBANK
SQURPHARMA
BXPHARMA
```

Board-qualified and common aliases are also accepted. These all resolve to
`GP`:

```text
GP
GP'PB
GP.DSE
GP.DH
```

DSE indices such as `DSEX`, `DS30`, and `DSES` are recognized where the
requested data operation supports indices.

### Data used by the default profile

| Area | Data |
|---|---|
| Market | Daily/intraday OHLCV candles, stock details, price history |
| Technical | Moving averages, MACD, RSI, ATR, volume and trend measures |
| Disclosures | Stock-specific and exchange-wide DSE announcements |
| Company | Company identity, sector and profile information |
| Financial | Annual and quarterly performance, EPS, NOCFPS and profit history |
| Ownership | Sponsor/director, institutional, foreign and public shareholding |
| Corporate actions | Cash/stock dividends, record dates and related notices |
| Balance sheet | Assets, liabilities, equity, NAV and disclosed loan status |

Historical candles, announcements, and dated financial records are filtered
again inside Python using the requested analysis date. This reduces look-ahead
leakage when researching a past date.

The detailed authenticated endpoint map and disclosure limitations are in
[docs/BANGLADESH_DSE.md](docs/BANGLADESH_DSE.md).

## What the analysis returns

Each completed run can contain:

| Report | Purpose |
|---|---|
| Market report | Price trend, momentum, volatility, volume, support and resistance |
| DSE disclosure report | Company announcements and exchange notices available by the selected date |
| Fundamentals report | Earnings, cash-flow proxies, balance sheet, dividends and ownership |
| Bull research | Strongest evidence-supported positive case |
| Bear research | Strongest evidence-supported negative case |
| Research manager | Balanced recommendation after the debate |
| Trader plan | Proposed action, conditions and risk levels |
| Risk debate | Aggressive, conservative and neutral critiques |
| Portfolio decision | Final rating, thesis, price target and time horizon |

The optional dashboard also creates deterministic presentation fields such as a
fundamental score, factor cards, bilingual English/Bangla explanations, and an
educational valuation range. Those calculations are shown separately from the
LLM-generated Portfolio Manager opinion.

## Requirements

- Python 3.10 or newer.
- A Doha Securities/DSE OAuth access token, or valid mobile-login credentials.
- An API key for one supported LLM provider.
- Node.js and npm only when using the optional Angular dashboard.
- PyTorch and the TimesFM compatibility source only when using the optional
  forecasting command.

## Installation

From the repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuration

The application automatically reads `.env` from the working directory. Keep
tokens and provider keys in that file or in process environment variables;
never commit real credentials.

### DSE authentication

Use an existing access token:

```dotenv
DSE_ACCESS_TOKEN=your-current-oms-oauth-token
```

Or allow the application to request an in-memory token through the supported
mobile authentication flow:

```dotenv
DSE_EMAIL_OR_PHONE=your-email-or-phone
DSE_PASSWORD=your-password
DSE_AUTH_GRANT_TYPE=urn:ietf:params:oauth:grant-type:mobile-application
DSE_AUTH_CLIENT_ID=oms
DSE_AUTH_SCOPE=android
```

An explicitly configured `DSE_ACCESS_TOKEN` takes precedence. Automatically
obtained access and refresh tokens are kept in process memory and are not
written to the configuration, report, cache, or log files.

### LLM provider

OpenRouter is the default profile:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
TRADINGAGENTS_LLM_PROVIDER=openrouter
TRADINGAGENTS_QUICK_THINK_LLM=google/gemini-3.5-flash
TRADINGAGENTS_DEEP_THINK_LLM=google/gemini-3.1-pro-preview
```

For native Gemini:

```dotenv
GOOGLE_API_KEY=your-google-key
TRADINGAGENTS_LLM_PROVIDER=google
TRADINGAGENTS_QUICK_THINK_LLM=gemini-3.5-flash
TRADINGAGENTS_DEEP_THINK_LLM=gemini-3.1-pro-preview
```

The provider layer also supports OpenAI, Anthropic, xAI, DeepSeek, Qwen, GLM,
MiniMax, NVIDIA, Kimi, Groq, Mistral, Azure OpenAI, AWS Bedrock, Ollama, and
custom OpenAI-compatible endpoints. Provider-specific credentials are described
in [.env.example](.env.example).

The `TRADINGAGENTS_*` configuration prefix is retained for compatibility even
though the Python package is named `dohasecuritiesstockai`.

## Run the interactive analysis

Start the CLI:

```bash
dohasecuritiesstockai
```

The exact product-name executable and the previous command remain available as
aliases:

```bash
DohasecuritiesStockAi
tradingagents
```

The CLI asks for:

1. DSE ticker.
2. Analysis date.
3. Output language: English or Bangla (`বাংলা`).
4. Analyst selection.
5. Research/debate depth.
6. LLM provider and models.
7. Provider-specific reasoning settings where supported.

Useful options:

```bash
dohasecuritiesstockai --checkpoint
dohasecuritiesstockai --clear-checkpoints
dohasecuritiesstockai --open-ui
dohasecuritiesstockai --no-open-ui
```

`--checkpoint` persists LangGraph progress after each node so an interrupted run
can resume. `--clear-checkpoints` forces a fresh run. `--open-ui` builds and
opens the optional local dashboard after the analysis finishes.

## Example output for a Bangladesh stock

The following is an abbreviated historical example from the stored
`GP`/Grameenphone run dated `2026-08-10`. It demonstrates the output structure;
it is not a current recommendation.

```text
Symbol: GP
Company: Grameenphone Ltd.
Market: Dhaka Stock Exchange
Analysis date: 2026-08-10

Market snapshot
  Latest price: Tk 257.80
  Daily change: +0.47%
  52-week range: Tk 237.40 - Tk 328.00

Deterministic dashboard summary
  Fundamental score: 72/100 (Good)
  Headline: Good company, but the share is quiet
  Valuation view: Fair price
  Educational valuation range: Tk 268.20 - Tk 349.50
  Main caution: Debt, cash flow, and liquidity deserve closer review

Agent Portfolio Manager decision
  Rating: Underweight
  Action: Reduce an overweight holding to a smaller core; do not add near Tk 258
  Technical evidence: Price below the falling 200-day SMA near Tk 259.80
  Fundamental evidence: Declining EPS/NOCFPS trend and weak current liquidity
  Risk level: Review a break below the 50-day SMA near Tk 254.50
  Re-entry condition: Confirmed 200-day SMA reclaim on volume, or a stabilized
                      value setup around Tk 237-240
  Time horizon: 3-6 months
```

This example also shows why the dashboard valuation and agent rating are kept
separate: one summarizes disclosed fundamentals using deterministic formulas,
while the other weighs trend, liquidity, catalysts, and risk through an LLM
debate.

The complete stored text example is available at
[docs/runtime/analyses/GP/2026-08-10/report.md](docs/runtime/analyses/GP/2026-08-10/report.md).

## Saved reports and state

During a run, working reports and logs are written under:

```text
~/.tradingagents/logs/<SYMBOL>/<YYYY-MM-DD>/
```

When the CLI asks to save the final report, its default export resembles:

```text
reports/GP_<timestamp>/
├── 1_analysts/
│   ├── market.md
│   ├── news.md
│   └── fundamentals.md
├── 2_research/
│   ├── bull.md
│   ├── bear.md
│   └── manager.md
├── 3_trading/trader.md
├── 4_risk/
│   ├── aggressive.md
│   ├── conservative.md
│   └── neutral.md
├── 5_portfolio/decision.md
└── complete_report.md
```

Override the default paths with `TRADINGAGENTS_RESULTS_DIR`,
`TRADINGAGENTS_CACHE_DIR`, and `TRADINGAGENTS_MEMORY_LOG_PATH`.

## Python usage

The package can run without the interactive questionnaire:

```python
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"

graph = TradingAgentsGraph(
    selected_analysts=("market", "news", "fundamentals"),
    config=config,
)
state, decision = graph.propagate("BRACBANK", "2026-08-10")

print(decision)
print(state["final_trade_decision"])
```

Environment overrides are applied to `DEFAULT_CONFIG` before this code runs.

## REST API and local dashboard

To fetch the yearly DSE evidence, run one grounded AI synthesis for the score,
valuation weighting, full research, and trader view, then open the UI without
running the long multi-agent graph, use:

```bash
dohasecuritiesstockai-dashboard GP --date 2026-08-10
```

To reopen an already completed multi-agent run in the UI without rerunning the
analysis or making another AI call, use:

```bash
dohasecuritiesstockai-dashboard SQURPHARMA --date 2026-08-17 --saved-run --no-ai
```

The command sends only the collected, date-bounded evidence to the configured
deep-thinking model and requires that provider's API key. Numeric valuation
methods remain calculation-backed; the AI assigns reliability weights, writes
the research, and produces the trader view. The first UI launch may still need
time to install/build the Angular frontend. Use `--no-open-ui` to generate and
save the AI payload without starting the server, or `--no-ai` for the original
calculation-only fallback.
The equivalent source-tree command is
`python -m dohasecuritiesstockai.dashboard_cli GP --date 2026-08-10`.

Start the read-only analysis API:

```bash
dohasecuritiesstockai-api
```

The equivalent module command is:

```bash
python -m dohasecuritiesstockai.api
```

Useful endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Service and market-profile status |
| `GET` | `/api/v1/stocks` | Search DSE symbols |
| `POST` | `/api/v1/analyses` | Queue an analysis |
| `GET` | `/api/v1/analyses/jobs/{job_id}` | Check job progress |
| `GET` | `/api/v1/analyses/{symbol}/latest` | Read the latest completed result |
| `GET` | `/api/v1/predictions/{run_id}` | Read a stored TimesFM forecast |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"GP","analysis_date":"2026-08-10"}'
```

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs` while the service is running.

For the end-to-end CLI workflow, enable the dashboard with:

```dotenv
TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS=true
```

Then run `dohasecuritiesstockai`. The first dashboard launch requires Node.js
and npm and may take longer because it installs locked frontend dependencies and
creates a production build. The server is read-only and remains open until
`Ctrl+C`.

## Long-term DSE opportunity screener

The separate opportunity command builds a research queue for a beginner who
wants to investigate less-followed DSE equities over a two-to-five-year horizon.
It does **not** predict a guaranteed winner or treat a low nominal share price as
cheap. The deterministic first pass screens the full listed universe for
profitability, relative valuation, financial safety, liquidity, and possible
value traps. Only the strongest diversified finalists receive deeper historical
evidence collection and an optional structured AI review.

Run a five-year scan and open its dedicated UI:

```bash
dohasecuritiesstockai-opportunities --horizon 5 --limit 10 --open-ui
```

Run the evidence-based ranking without sending finalist data to an LLM:

```bash
dohasecuritiesstockai-opportunities --no-ai --no-open-ui
```

The AI prompt receives only the saved evidence for the shortlisted companies.
It is instructed to identify missing data, avoid outside memory and precise
future-price claims, and return research priorities, risks, catalysts, and
checkpoints—not buy instructions. Every result includes source periods and
missing-evidence warnings. The methodology is informed by established research
on value, profitability, investment, and risk, including the
[Fama–French five-factor model](https://www.sciencedirect.com/science/article/pii/S0304405X14002323)
and [profitability research by Robert Novy-Marx](https://www.nber.org/papers/w15940).
Before using real money, review the
[Bangladesh Securities and Exchange Commission investor education material](https://sec.gov.bd/home/ieprogram)
and verify the latest audited report and DSE disclosures.

## Optional TimesFM stock forecasting

The independent prediction command backtests TimesFM on a held-out part of DSE
price history and then forecasts future bars. It is separate from the
multi-agent recommendation.

Install the optional dependencies:

```bash
git submodule update --init timesfm
pip install -e '.[timesfm]'
```

If cloning the repository for the first time, `git clone --recurse-submodules`
can initialize the pinned TimesFM checkout in the same step.

Run a daily BXPHARMA forecast using one year of history:

```bash
dohasecuritiesstockai-predict BXPHARMA \
  --resolution 1d \
  --lookback 1y \
  --future-steps 12 \
  --no-open-ui
```

The command reports held-out MAE, RMSE, directional accuracy, interval coverage,
and skill versus a last-price baseline. A positive-looking headline metric does
not prove useful predictive skill; compare the forecast with the reported naive
baseline before relying on it.

### Differential Graph Transformer forecasting

The DGT command uses the same authenticated DSE candle service and the same
prediction dashboard. Unlike TimesFM, it trains a small model for each run and
can learn from a target stock plus related DSE securities. It saves a real vs.
predicted holdout CSV, a future CSV, a chart, and a reusable PyTorch checkpoint.

```bash
pip install -e '.[dgt]'

dohasecuritiesstockai-dgt-predict GP \
  --resolution 1d \
  --lookback 2y \
  --peers DSEX,BRACBANK,SQURPHARMA \
  --future-steps 12 \
  --open-ui
```

The evaluation graph, scaler, and model use only data before the validation and
holdout periods. The holdout forecast is recursive, so its real-price comparison
has no future-price leakage. After scoring, a separate deployment model is refit
on all available real bars for the future forecast; use `--no-refit` to disable
that final step.

## Important limitations

- LLM output is non-deterministic and can differ between otherwise identical
  runs.
- A low temperature does not make reasoning-model output fully reproducible.
- Upstream DSE records can be missing, delayed, revised, or use inconsistent
  units.
- Quarterly NOCFPS is a cash-flow proxy, not a complete cash-flow statement.
- Shareholding percentages do not identify individual insider trades.
- Thinly traded DSE securities can make technical signals and estimated price
  levels unreliable.
- Historical-date filtering reduces look-ahead leakage but cannot repair an
  upstream record carrying an incorrect timestamp.
- TimesFM forecasts are experimental and may underperform a naive last-price
  baseline.
- Nothing in the CLI, API, dashboard, or generated report is financial advice
  or an instruction to trade.

## Security and data boundaries

- Credentials are read from environment variables or `.env`.
- Generated OAuth tokens are held in memory and are not intentionally logged or
  persisted.
- DSE analysis uses authentication plus read-only requests.
- The analysis workflow does not call order, portfolio, account-mutation, or
  market-data mutation endpoints.
- Reports may still contain sensitive research conclusions; review them before
  sharing.

## Project layout

```text
dohasecuritiesstockai/   Python package, agents, DSE data layer, API and forecasts
cli/                     Interactive Rich/Typer command-line interface
analysis-ui/             Optional Angular analysis dashboard
timesfm/                 Pinned Google TimesFM git submodule
tests/                   Unit and integration tests
docs/                    DSE endpoint and operational documentation
scripts/                 Development and smoke-test utilities
```

## Development

Install development dependencies and run the checks:

```bash
pip install -e '.[dev]'
ruff check dohasecuritiesstockai cli tests
pytest -q
```

Tests that require optional SDKs, live API credentials, or external services may
be skipped when those dependencies are not configured.
