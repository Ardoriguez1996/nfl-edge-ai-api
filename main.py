from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from decision_policy import assess_pick

app = FastAPI(title="NFL Edge AI Prediction API", version="0.1.0", description="Initial API scaffold. Model markets remain unapproved until validated.")

MARKET_APPROVAL = {"moneyline": False, "spread": False, "total": False, "player_prop": False}

class EvaluateRequest(BaseModel):
    market: str = Field(..., description="moneyline, spread, total, or player_prop")
    model_probability: float = Field(..., ge=0.0, le=1.0)
    american_odds: float
    event: Optional[str] = None

@app.get("/")
def root():
    return {"app": "NFL Edge AI", "status": "online", "stage": "backend scaffold", "prediction_markets_approved": MARKET_APPROVAL}

@app.get("/health")
def health():
    return {"status": "ok", "service": "nfl-edge-ai-api"}

@app.get("/api/status")
def api_status():
    return {
        "status": "online",
        "model_connected": False,
        "note": "This deployment confirms API hosting only. It does not generate game predictions yet.",
        "markets": {k: {"approved": v, "green_picks_enabled": False} for k, v in MARKET_APPROVAL.items()},
        "historical_data_included": True,
        "historical_odds_verified_as_current_sportsbook_lines": False
    }

@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    market = req.market.strip().lower()
    if market not in MARKET_APPROVAL:
        raise HTTPException(status_code=400, detail="Market must be moneyline, spread, total, or player_prop")
    try:
        result = assess_pick(req.model_probability, req.american_odds, approved=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result.update({
        "market": market,
        "event": req.event,
        "model_probability_source": "user_supplied_not_model_generated",
        "status": "NOT_A_MODEL_PREDICTION",
        "market_approved": False,
        "warning": "This endpoint only evaluates a supplied probability. No market is approved for GREEN picks."
    })
    return result
