# NFL Edge AI API — initial deployment scaffold

This is the first deployable API scaffold for Render. It provides a health endpoint and a fail-closed evaluation endpoint. It is **not yet a game prediction engine** and must not be represented as one.

## Endpoints
- `GET /` service status
- `GET /health` health check
- `GET /api/status` shows model connection and approval state
- `POST /api/evaluate` evaluates a supplied probability and odds, but always blocks GREEN because all markets remain unapproved
- `GET /docs` interactive API documentation

## Render setup
- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Important model status
The historical model package is included for the next integration stage. This initial API does not train or serve model predictions. The current historical evaluation did not demonstrate a reliable edge over the market, so all markets are deliberately unapproved. The included historical odds are not verified as current sportsbook or closing odds.
