#!/usr/bin/env python3
"""
NFL walk-forward model prototype.
Uses pregame Elo features + market spread/total to estimate home ML and ATS probabilities.
Historical source CSV is community-maintained, not DraftKings-specific.
"""
from __future__ import annotations
import argparse, json, math, os, sys, urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/willvernon/NFL_Scores_Lines/main/betting_data_99-25.csv"
FEATURES = ["elo_diff", "elo_home_prob", "spread_line", "total_line", "week_num"]
DEFAULT_MIN_TRAIN_SEASONS = 8
DEFAULT_GREEN_EDGE = 0.05
DEFAULT_YELLOW_EDGE = 0.02

def american_to_decimal(odds):
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))

def implied_prob(odds):
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def no_vig_home_prob(home_odds, away_odds):
    ph, pa = implied_prob(home_odds), implied_prob(away_odds)
    return ph / (ph + pa)

def ev_per_unit(p, odds):
    dec = american_to_decimal(odds)
    return p * (dec - 1) - (1 - p)

def grade(p, odds, green=DEFAULT_GREEN_EDGE, yellow=DEFAULT_YELLOW_EDGE):
    breakeven = implied_prob(odds)
    edge = p - breakeven
    ev = ev_per_unit(p, odds)
    label = "GREEN" if edge >= green and ev > 0 else (
        "YELLOW" if edge >= yellow and ev > 0 else "RED"
    )
    return {"probability": float(p), "break_even": float(breakeven),
            "edge_pp": float(100 * edge), "ev_per_unit": float(ev), "grade": label}

def download_data(out_path):
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "NFLModelResearch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(raw)
    return {"path": str(out_path), "bytes": len(raw), "source_url": DATA_URL}

def load_games(path):
    df = pd.read_csv(path)
    required = ["game_id","season","week","game_type","home_team","home_score",
                "away_team","away_score","spread_line","total_line"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in ["season","week","home_score","away_score","spread_line","total_line",
              "home_moneyline","away_moneyline","home_spread_odds","away_spread_odds"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["season","week","home_score","away_score"]).copy()
    df["season"] = df["season"].astype(int)
    df["week_num"] = df["week"].astype(float)
    df = df[(df.home_score >= 0) & (df.away_score >= 0)]
    df = df.sort_values(["season","week_num","game_id"]).reset_index(drop=True)
    # Keep only completed games.
    return df

def make_pregame_features(games, k=20.0, home_advantage=55.0):
    """
    Compute Elo features before each week's games, then update ratings after the week.
    This avoids one game's result leaking into another game in the same week.
    Source convention: positive spread_line = home favored by that many points.
    """
    games = games.copy()
    if "week_num" not in games.columns:
        games["week_num"] = pd.to_numeric(games["week"], errors="coerce")
    ratings = {}
    out_rows = []
    for (season, week), block in games.groupby(["season","week_num"], sort=True):
        pending = []
        for _, r in block.iterrows():
            h, a = str(r.home_team), str(r.away_team)
            rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
            # Elo expected home win probability, with neutral-site caveat unavailable in this source.
            elo_home_prob = 1.0 / (1.0 + 10.0 ** (-(rh + home_advantage - ra) / 400.0))
            row = r.to_dict()
            row["elo_home"] = rh
            row["elo_away"] = ra
            row["elo_diff"] = rh - ra
            row["elo_home_prob"] = elo_home_prob
            row["home_margin"] = float(r.home_score - r.away_score)
            row["home_win"] = int(row["home_margin"] > 0)
            # nflverse convention: spread_line > 0 means home favored by that many points.
            # Cover margin = actual home margin minus the home-favorite line.
            row["cover_margin"] = row["home_margin"] - float(r.spread_line) if pd.notna(r.spread_line) else np.nan
            row["home_cover"] = int(row["cover_margin"] > 0) if pd.notna(row["cover_margin"]) and row["cover_margin"] != 0 else np.nan
            row["total_points"] = float(r.home_score + r.away_score)
            out_rows.append(row)
            pending.append((h, a, rh, ra, row["home_margin"]))
        # Update Elo after every game in the week has had its pregame rating assigned.
        for h, a, rh, ra, margin in pending:
            expected_home = 1.0 / (1.0 + 10.0 ** (-((rh + home_advantage - ra) / 400.0)))
            actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            mov_mult = math.log(abs(margin) + 1.0) * (2.2 / ((abs(rh-ra))*0.001 + 2.2))
            change = k * mov_mult * (actual_home - expected_home)
            ratings[h] = rh + change
            ratings[a] = ra - change
    return pd.DataFrame(out_rows).sort_values(["season","week_num","game_id"]).reset_index(drop=True)

def make_estimator():
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.5, max_iter=2000))
    ])
    return base

def calibrated_estimator():
    # Chronological calibration inside the training period. Requires enough rows and both classes.
    return CalibratedClassifierCV(
        estimator=make_estimator(),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
        ensemble=True
    )

def calibration_bins(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i+1] if i < n_bins-1 else p <= edges[i+1])
        if mask.any():
            bins.append({"range": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                         "n": int(mask.sum()), "mean_pred": round(float(p[mask].mean()), 4),
                         "observed": round(float(y[mask].mean()), 4)})
    return bins

def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return {"n": int(len(y)), "observed_rate": float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0,1])),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "calibration_bins": calibration_bins(y, p)}

def market_baseline_ml(df):
    m = df.dropna(subset=["home_moneyline","away_moneyline"]).copy()
    if m.empty:
        return None
    p = np.array([no_vig_home_prob(h, a) for h, a in zip(m.home_moneyline, m.away_moneyline)])
    y = m.home_win.to_numpy(dtype=int)
    # Ties are neither win nor loss; omit from binary comparison.
    keep = m.home_margin.to_numpy() != 0
    if keep.sum() == 0:
        return None
    return metrics(y[keep], p[keep])

def betting_roi_ml(df, p_home, edge_threshold=DEFAULT_GREEN_EDGE):
    bets = []
    for (_, r), ph in zip(df.iterrows(), p_home):
        if pd.notna(r.home_moneyline):
            g = grade(ph, r.home_moneyline)
            if g["edge_pp"] >= 100*edge_threshold and g["ev_per_unit"] > 0:
                if r.home_margin != 0:
                    profit = american_to_decimal(r.home_moneyline)-1 if r.home_margin > 0 else -1
                    bets.append((profit, "home"))
        if pd.notna(r.away_moneyline):
            pa = 1-ph
            g = grade(pa, r.away_moneyline)
            if g["edge_pp"] >= 100*edge_threshold and g["ev_per_unit"] > 0:
                if r.home_margin != 0:
                    profit = american_to_decimal(r.away_moneyline)-1 if r.home_margin < 0 else -1
                    bets.append((profit, "away"))
    if not bets:
        return {"bets": 0, "net_units": 0.0, "roi": None, "note": "No bets met the proposed edge threshold at available prices."}
    profits = np.array([x[0] for x in bets])
    return {"bets": int(len(profits)), "net_units": float(profits.sum()),
            "roi": float(profits.mean()), "wins": int((profits > 0).sum()),
            "win_rate": float((profits > 0).mean()),
            "edge_threshold_pp": 100*edge_threshold,
            "note": "Descriptive holdout ROI on available historical prices; not DraftKings-specific unless source odds are verified as DraftKings."}

def betting_roi_spread(df, p_home_cover, edge_threshold=DEFAULT_GREEN_EDGE):
    bets = []
    for (_, r), ph in zip(df.iterrows(), p_home_cover):
        if pd.isna(r.cover_margin) or r.cover_margin == 0:
            continue  # push excluded
        if pd.notna(r.home_spread_odds):
            g = grade(ph, r.home_spread_odds)
            if g["edge_pp"] >= 100*edge_threshold and g["ev_per_unit"] > 0:
                profit = american_to_decimal(r.home_spread_odds)-1 if r.cover_margin > 0 else -1
                bets.append(profit)
        if pd.notna(r.away_spread_odds):
            pa = 1-ph
            g = grade(pa, r.away_spread_odds)
            if g["edge_pp"] >= 100*edge_threshold and g["ev_per_unit"] > 0:
                profit = american_to_decimal(r.away_spread_odds)-1 if r.cover_margin < 0 else -1
                bets.append(profit)
    if not bets:
        return {"bets": 0, "net_units": 0.0, "roi": None, "note": "No bets met the proposed edge threshold at available prices."}
    profits = np.array(bets)
    return {"bets": int(len(profits)), "net_units": float(profits.sum()),
            "roi": float(profits.mean()), "wins": int((profits > 0).sum()),
            "win_rate": float((profits > 0).mean()),
            "edge_threshold_pp": 100*edge_threshold,
            "note": "Pushes excluded; historical odds may not be DraftKings prices."}

def walk_forward(games, min_train_seasons=DEFAULT_MIN_TRAIN_SEASONS, edge_threshold=DEFAULT_GREEN_EDGE):
    feat = make_pregame_features(games)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=["spread_line","total_line"]).copy()
    seasons = sorted(feat.season.unique())
    first_test = seasons[min_train_seasons] if len(seasons) > min_train_seasons else None
    if first_test is None:
        raise ValueError(f"Need at least {min_train_seasons+1} seasons with completed games.")
    outputs = []
    ml_features = FEATURES
    ats_features = FEATURES
    for season in seasons:
        if season < first_test:
            continue
        train = feat[feat.season < season].copy()
        test = feat[feat.season == season].copy()
        if len(train) < 500 or train.home_win.nunique() < 2:
            continue
        # ML: exclude ties from fitting; tie games are uncommon but are not wins/losses.
        train_ml = train[train.home_margin != 0]
        test_ml = test[test.home_margin != 0].copy()
        if len(test_ml) == 0:
            continue
        ml_model = calibrated_estimator()
        ml_model.fit(train_ml[ml_features], train_ml.home_win.astype(int))
        test_ml["p_home_win"] = ml_model.predict_proba(test_ml[ml_features])[:,1]
        test_ml["model_market"] = "moneyline"
        outputs.append(test_ml)
        # ATS: push outcomes are excluded from classifier training and ROI.
        train_ats = train.dropna(subset=["home_cover"])
        test_ats = test.dropna(subset=["home_cover"]).copy()
        test_ats = test_ats[test_ats.cover_margin != 0]
        if len(train_ats) >= 500 and len(test_ats):
            ats_model = calibrated_estimator()
            ats_model.fit(train_ats[ats_features], train_ats.home_cover.astype(int))
            test_ats["p_home_cover"] = ats_model.predict_proba(test_ats[ats_features])[:,1]
            test_ats["model_market"] = "spread"
            # Store predictions separately by game and market for metrics.
            outputs.append(test_ats)
    pred = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if pred.empty:
        raise ValueError("No walk-forward predictions produced.")
    ml = pred[pred.model_market == "moneyline"].copy()
    ats = pred[pred.model_market == "spread"].copy()
    report = {
        "status": "WALK_FORWARD_OUT_OF_SAMPLE",
        "data_source": DATA_URL,
        "seasons_tested": sorted(int(s) for s in pred.season.unique()),
        "assumptions": {
            "min_training_seasons": min_train_seasons,
            "elo_k": 20, "home_advantage_elo_points": 55,
            "green_edge_pp": edge_threshold*100,
            "yellow_edge_pp": DEFAULT_YELLOW_EDGE*100,
            "spread_sign": "source convention: positive spread_line means home favored; home cover margin = actual home margin - spread_line",
            "probability_calibration": "sigmoid calibration with chronological TimeSeriesSplit folds within each training window",
            "thresholds_empirically_optimized": False
        },
        "moneyline": {
            "model": metrics(ml.home_win.astype(int), ml.p_home_win),
            "market_no_vig_baseline_on_same_rows": market_baseline_ml(ml),
            "available_price_strategy": betting_roi_ml(ml, ml.p_home_win.to_numpy(), edge_threshold)
        },
        "spread": {
            "model": metrics(ats.home_cover.astype(int), ats.p_home_cover),
            "50_percent_baseline_brier": 0.25,
            "available_price_strategy": betting_roi_spread(ats, ats.p_home_cover.to_numpy(), edge_threshold)
        },
        "critical_limitations": [
            "This historical file is not verified as DraftKings-only; do not label its ROI as DraftKings ROI.",
            "Market spread and total are included as pregame predictors, so this is a market-informed model, not an independent team-only model.",
            "No player props are modeled by this script.",
            "A walk-forward backtest is historical evidence, not proof of future profit; odds timestamps and line provenance must be verified."
        ]
    }
    return report, pred

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", default="data/nfl_betting_data_99_25.csv")
    v = sub.add_parser("validate")
    v.add_argument("--csv", default="data/nfl_betting_data_99_25.csv")
    v.add_argument("--out", default="reports/nfl_walk_forward_report.json")
    v.add_argument("--predictions", default="reports/nfl_walk_forward_predictions.csv")
    v.add_argument("--min-train-seasons", type=int, default=8)
    v.add_argument("--edge", type=float, default=0.05)
    args = parser.parse_args()
    try:
        if args.command == "download":
            result = download_data(args.out)
            print(json.dumps(result, indent=2))
            return
        games = load_games(args.csv)
        report, pred = walk_forward(games, args.min_train_seasons, args.edge)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        pred.to_csv(args.predictions, index=False)
        print(json.dumps(report, indent=2))
        print(f"\nSaved report: {args.out}\nSaved predictions: {args.predictions}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
