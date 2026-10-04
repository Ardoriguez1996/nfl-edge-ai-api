"""Risk-control layer for NFL model v2.

The current model has not passed the historical market-benchmark gate.
Consequently, this module defaults to PASS even when a raw edge appears large.
"""
from dataclasses import dataclass, asdict

def implied_probability(american_odds: float) -> float:
    odds = float(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def no_vig_probability(home_odds: float, away_odds: float) -> float:
    ph = implied_probability(home_odds)
    pa = implied_probability(away_odds)
    return ph / (ph + pa)

def market_gate(model_brier: float, model_log_loss: float,
                market_brier: float, market_log_loss: float,
                min_holdout_games: int, holdout_games: int,
                require_both: bool = True) -> bool:
    """Approval requires sufficient holdout size and improvement on proper scores."""
    if holdout_games < min_holdout_games:
        return False
    if require_both:
        return model_brier < market_brier and model_log_loss < market_log_loss
    return model_brier < market_brier

def assess_pick(model_probability: float, american_odds: float,
                approved: bool = False, edge_threshold: float = 0.05) -> dict:
    """Return PASS unless the model is approved and both edge and EV clear thresholds."""
    p = min(max(float(model_probability), 1e-6), 1 - 1e-6)
    odds = float(american_odds)
    breakeven = implied_probability(odds)
    decimal_odds = 1 + (odds / 100 if odds > 0 else 100 / abs(odds))
    ev = p * (decimal_odds - 1) - (1 - p)
    edge = p - breakeven
    if not approved:
        grade = "PASS_MODEL_NOT_APPROVED"
    elif edge >= edge_threshold and ev > 0:
        grade = "GREEN"
    else:
        grade = "PASS"
    return {
        "grade": grade,
        "model_probability": p,
        "break_even_probability": breakeven,
        "edge_percentage_points": 100 * edge,
        "expected_value_per_unit": ev,
    }
