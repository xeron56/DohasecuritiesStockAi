# Bangladesh / Dhaka Stock Exchange configuration

This project defaults to the `bangladesh_dse` market profile. The Python data
vendor mirrors the production endpoints referenced by the Angular application
in `../web-ui/src/app`.

## Authentication

The Doha Securities gateway returns HTTP 401/403 without an OAuth bearer token.
The Angular app obtains that token through authorization-code flow from
`https://auth.dohasecurities.com.bd`. TradingAgents accepts login credentials
from process environment variables, but does not write them itself or persist
the generated access token or refresh token.

Use either a pre-issued access token or the mobile login flow. For an existing
token:

```bash
export DSE_ACCESS_TOKEN='...'
```

For automatic login, leave `DSE_ACCESS_TOKEN` empty and configure:

```bash
export DSE_EMAIL_OR_PHONE='...'
export DSE_PASSWORD='...'
export DSE_AUTH_GRANT_TYPE='urn:ietf:params:oauth:grant-type:mobile-application'
export DSE_AUTH_CLIENT_ID='oms'
export DSE_AUTH_SCOPE='android'
```

The client posts those credentials only to
`/usermanagementservice/v1/auth_server/token/custom-flow`, extracts the returned
access token, and keeps it in process memory. It does not persist the generated
token. An explicitly set `DSE_ACCESS_TOKEN` takes precedence. Credentials and
tokens are never included in application logs or exception messages.

## Data-source map

| Framework tool | Doha Securities / DSE endpoint |
|---|---|
| Symbols | `/marketdataservice/market/symbols/sorted` |
| Daily OHLCV | `/market-analytics-service/v1/candlesticks/limited` |
| Stock details | `/marketdataservice/stock_details/{symbol}` |
| Stock news and historical filtering | `/marketdataservice/news/all/filter` |
| Exchange-wide DSE news | `/marketdataservice/news/all/filter` without `stockCode` |
| Company profile | `/market-analytics-service/v1/fundamentals/company_details/{symbol}` |
| Annual performance | `/fundamentals/financial_performance/{symbol}` |
| Quarterly performance | `/fundamentals/quarterly_performance/{symbol}` |
| Shareholding | `/fundamentals/share_holding/{symbol}` |
| Dividends | `/fundamentals/dividend_information/{symbol}` |
| NAV | `/fundamentals/nav_per_year/{symbol}` |
| Quarterly EPS | `/fundamentals/eps_history_quarter/{symbol}` |
| Quarterly NOCFPS | `/fundamentals/nocfps_history_quarter/{symbol}` |
| Quarterly profit | `/fundamentals/profit_history_quarter/{symbol}` |
| Loan status | `/fundamentals/loan_status/{symbol}` |
| Balance sheet | `/api/balance_sheet/balance-sheet/{symbol}` |

The table abbreviates the common analytics prefix
`/market-analytics-service/v1` for fundamental rows after the company profile.

Historical candles, news, and dated/yearly financial records are filtered again
in Python to the requested analysis date. This prevents future data from leaking
into a backtest even if an upstream API response is broader than requested.

## Known disclosure boundaries

- The Angular API exposes quarterly NOCFPS, not a complete cash-flow statement.
  The `get_cashflow` tool labels this clearly and does not synthesize missing
  line items.
- The API exposes periodic sponsor/director holding percentages, not individual
  insider trade filings. The insider tool reports percentage-point changes
  between disclosures and does not call them specific purchases or sales.
- Every relevant gateway route currently requires an authenticated OMS token.
  Automated service-to-service OAuth would require a separately issued client
  credential; the public Angular client does not supply one.
- DSE API field names and units are preserved in fundamental JSON blocks. The
  agent is instructed to treat amounts as BDT only where the field/API convention
  supports that interpretation.

## LLM provider profiles

OpenRouter is the built-in default:

```bash
export OPENROUTER_API_KEY='...'
export TRADINGAGENTS_LLM_PROVIDER=openrouter
export TRADINGAGENTS_QUICK_THINK_LLM=google/gemini-3.5-flash
export TRADINGAGENTS_DEEP_THINK_LLM=google/gemini-3.1-pro-preview
```

For native Gemini:

```bash
export GOOGLE_API_KEY='...'
export TRADINGAGENTS_LLM_PROVIDER=google
export TRADINGAGENTS_QUICK_THINK_LLM=gemini-3.5-flash
export TRADINGAGENTS_DEEP_THINK_LLM=gemini-3.1-pro-preview
```

One provider is used per run. Run the same ticker/date once with each provider
if you want to compare outputs; do not mix reports from different runs without
tracking which model produced each result.

## Feature flags

| Environment variable | Default | Effect |
|---|---:|---|
| `TRADINGAGENTS_MARKET_PROFILE` | `bangladesh_dse` | Selects DSE identity, benchmark, prompts, and configured candle loader |
| `TRADINGAGENTS_SOCIAL_MEDIA_ENABLED` | `false` | Adds/removes the Reddit + StockTwits Sentiment Analyst |
| `TRADINGAGENTS_MACRO_DATA_ENABLED` | `false` | Adds/removes FRED tools from the News Analyst |
| `TRADINGAGENTS_PREDICTION_MARKETS_ENABLED` | `false` | Adds/removes Polymarket tools from the News Analyst |
| `TRADINGAGENTS_DSE_GATEWAY_URL` | production gateway | Overrides the gateway used for login and read-only APIs |
| `TRADINGAGENTS_DSE_REQUEST_TIMEOUT` | `30` | HTTP timeout in seconds |
| `TRADINGAGENTS_DSE_VERIFY_SSL` | `true` | TLS certificate verification; keep enabled outside controlled development |
| `TRADINGAGENTS_DSE_BENCHMARK_TICKER` | `DSEX` | Benchmark for post-decision alpha reflection |

The default analysts are `market`, `news`, and `fundamentals`. To opt into the
social analyst programmatically, both enable the flag and include `social`:

```python
config["social_media_enabled"] = True
ta = TradingAgentsGraph(
    selected_analysts=("market", "social", "news", "fundamentals"),
    config=config,
)
```

## Run

```bash
pip install -e .
export DSE_ACCESS_TOKEN='...'
export OPENROUTER_API_KEY='...'
dohasecuritiesstockai
```

Alternatively, replace `DSE_ACCESS_TOKEN` with `DSE_EMAIL_OR_PHONE` and
`DSE_PASSWORD`. Analysis uses authentication plus GET requests; no order,
portfolio, account, or market-data mutation endpoints are called.

Enter a base trading code such as `GP`. `GP'PB`, `GP.DSE`, and `GP.DH` are also
accepted and normalized to `GP`. Reports are research outputs, not financial or
investment advice; validate material claims against official DSE disclosures
before acting.

## Analysis REST API and independent Angular UI

For the end-to-end CLI-first workflow, add this to `.env` and run the CLI:

```bash
TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS=true
python -m cli.main
```

The CLI persists its exact completed agent state, finishes the deterministic
presentation calculations, builds Angular when its sources changed, starts the
same-origin API/UI server, and opens `/?symbol=...&date=...`. Keep that terminal
open while using the dashboard and press `Ctrl+C` to stop it. The first launch
requires Node.js/npm and runs `npm ci` from the committed lockfile. Pass
`--no-open-ui` to override the env setting for one run.

The API and development UI can still be started separately when working on the
frontend.

Start the read-only analysis API from the repository directory:

```bash
python -m dohasecuritiesstockai.api
```

The API exposes health, DSE stock selection, asynchronous analysis jobs,
completed structured reports, the original agent reports, and the unchanged
full-state JSON under `/api/v1`. Interactive OpenAPI documentation is available
at `/docs` while the service is running.

The independent Angular prototype is in `analysis-ui/` and does not modify or
depend on the brokerage `web-ui` project:

```bash
cd analysis-ui
npm install
npm start
```

Visible scores and educational fair-value figures are deterministic
calculations from DSE disclosures. LLM output remains separately visible as
agent evidence rather than being used to invent the score.
