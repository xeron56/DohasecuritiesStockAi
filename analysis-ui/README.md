# DSE AI Analysis UI

Independent Angular client for the TradingAgents DSE REST API. It does not
depend on, import from, or modify the brokerage `web-ui` application.

## Run locally

The normal user flow starts here automatically after the Python CLI completes:

```bash
# From the repository root; .env can enable this permanently.
python -m cli.main --open-ui
```

That flow serves the production Angular bundle and opens the exact completed
ticker/date. For frontend development with live rebuilds, use two terminals:

From the `TradingAgents` directory, start the Python API:

```bash
python -m tradingagents.api
```

Then start this UI in another terminal:

```bash
cd analysis-ui
npm install
npm start
```

The development client calls `http://127.0.0.1:8000/api/v1`. Production builds
use same-origin `/api/v1`; change `src/environments/environment.prod.ts` if the
API is hosted elsewhere.

The UI only requests DSE reads and analysis jobs. Brokerage order, portfolio,
cash, transfer, and other mutation endpoints are not exposed by the API.
