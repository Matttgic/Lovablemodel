"""
FastAPI Backend for XGBoost European Predictions
Deploy on Railway, Fly.io, or Render

Features: h_gf, h_ga, h_cs, h_fw, h_fd, h_fl, a_gf, a_ga, a_cs, a_fw, a_fd, a_fl
"""

import os
import httpx
import xgboost as xgb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="XGBoost European Football Predictions API")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load XGBoost model
MODEL_PATH = os.getenv("MODEL_PATH", "model.json")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# API-Football configuration
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
CURRENT_SEASON = 2025

# Configuration des 5 grands championnats
LEAGUES = {
    "Premier League": 39,
    "Ligue 1": 61,
    "Bundesliga": 78,
    "Serie A": 135,
    "La Liga": 140
}

class PredictionRequest(BaseModel):
    """Request format from Edge Function"""
    match_id: int
    features: dict
    odds: Optional[dict] = None

class PredictionResponse(BaseModel):
    fixture_id: int
    league_id: int
    league_name: str
    home_team: str
    away_team: str
    date: str
    features: dict
    predictions: dict
    odds: Optional[dict]
    value_bets: list

class MatchListResponse(BaseModel):
    matches: list
    count: int

async def fetch_api_football(endpoint: str, params: dict = None):
    """Fetch data from API-Football"""
    if not API_FOOTBALL_KEY:
        raise HTTPException(status_code=500, detail="API_FOOTBALL_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_FOOTBALL_URL}/{endpoint}",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params=params or {},
            timeout=30.0
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="API-Football error")
        return response.json()

async def get_team_stats(team_id: int, league_id: int) -> dict:
    """Get team statistics for the current season in a specific league"""
    # Note: On passe league_id dynamiquement ici
    data = await fetch_api_football("teams/statistics", {
        "team": team_id,
        "league": league_id,
        "season": CURRENT_SEASON
    })
    
    if not data.get("response"):
        return {"gf": 0, "ga": 0, "cs": 0, "fw": 0, "fd": 0, "fl": 0}
    
    stats = data["response"]
    fixtures = stats.get("fixtures", {})
    goals = stats.get("goals", {})
    clean_sheets = stats.get("clean_sheet", {})
    
    played = fixtures.get("played", {}).get("total", 0) or 1
    wins = fixtures.get("wins", {}).get("total", 0)
    draws = fixtures.get("draws", {}).get("total", 0)
    loses = fixtures.get("loses", {}).get("total", 0)
    goals_for = goals.get("for", {}).get("total", {}).get("total", 0)
    goals_against = goals.get("against", {}).get("total", {}).get("total", 0)
    cs_total = clean_sheets.get("total", 0)
    
    return {
        "gf": round(goals_for / played, 2),
        "ga": round(goals_against / played, 2),
        "cs": cs_total,
        "fw": wins,
        "fd": draws,
        "fl": loses
    }

async def get_odds(fixture_id: int) -> Optional[dict]:
    """Get Bet365 odds for a fixture"""
    try:
        data = await fetch_api_football("odds", {
            "fixture": fixture_id,
            "bookmaker": 8,  # Bet365
            "bet": 1  # Match Winner
        })
        
        if not data.get("response") or len(data["response"]) == 0:
            return None
        
        bookmakers = data["response"][0].get("bookmakers", [])
        if not bookmakers:
            return None
        
        bets = bookmakers[0].get("bets", [])
        if not bets:
            return None
        
        values = bets[0].get("values", [])
        odds = {}
        for v in values:
            if v["value"] == "Home":
                odds["home"] = float(v["odd"])
            elif v["value"] == "Draw":
                odds["draw"] = float(v["odd"])
            elif v["value"] == "Away":
                odds["away"] = float(v["odd"])
        
        return odds if len(odds) == 3 else None
    except Exception:
        return None

def calculate_value_bets(predictions: dict, odds: dict) -> list:
    """Calculate value bets based on Kelly criterion"""
    value_bets = []
    
    outcomes = [
        ("home", predictions["home"], odds.get("home", 0)),
        ("draw", predictions["draw"], odds.get("draw", 0)),
        ("away", predictions["away"], odds.get("away", 0))
    ]
    
    for outcome, prob, odd in outcomes:
        if odd > 0:
            implied_prob = 1 / odd
            ev = (prob * odd) - 1
            edge = prob - implied_prob
            
            if ev > 0.05:  # 5% EV threshold
                value_bets.append({
                    "outcome": outcome,
                    "probability": round(prob * 100, 1),
                    "odds": odd,
                    "implied_probability": round(implied_prob * 100, 1),
                    "expected_value": round(ev * 100, 1),
                    "edge": round(edge * 100, 1)
                })
    
    return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "model": "XGBoost European Classifier", 
        "supported_leagues": LEAGUES
    }

@app.post("/predict")
async def predict_from_features(request: PredictionRequest):
    """Predict match outcome using pre-calculated features."""
    features = request.features
    feature_array = np.array([[
        features.get("home_gf", 0), features.get("home_ga", 0), features.get("home_cs", 0),
        features.get("home_fw", 0), features.get("home_fd", 0), features.get("home_fl", 0),
        features.get("away_gf", 0), features.get("away_ga", 0), features.get("away_cs", 0),
        features.get("away_fw", 0), features.get("away_fd", 0), features.get("away_fl", 0),
    ]])
    
    probabilities = model.predict_proba(feature_array)[0]
    predictions = {
        "home_win": round(float(probabilities[0]), 4),
        "draw": round(float(probabilities[1]), 4),
        "away_win": round(float(probabilities[2]), 4)
    }
    
    value_bets = []
    if request.odds:
        odds_mapped = {
            "home": request.odds.get("home_win", 0),
            "draw": request.odds.get("draw", 0),
            "away": request.odds.get("away_win", 0)
        }
        predictions_mapped = {
            "home": predictions["home_win"],
            "draw": predictions["draw"],
            "away": predictions["away_win"]
        }
        value_bets = calculate_value_bets(predictions_mapped, odds_mapped)
    
    return {
        "match_id": request.match_id,
        "predictions": predictions,
        "value_bets": value_bets
    }

@app.get("/matches", response_model=MatchListResponse)
async def get_upcoming_matches(league_id: int = Query(39, description="League ID (39=PL, 61=L1, 78=BL, 135=SA, 140=LL)")):
    """Get upcoming matches for a specific league (Default: Premier League)"""
    data = await fetch_api_football("fixtures", {
        "league": league_id,
        "season": CURRENT_SEASON,
        "next": 20
    })
    
    matches = []
    for fixture in data.get("response", []):
        matches.append({
            "fixture_id": fixture["fixture"]["id"],
            "date": fixture["fixture"]["date"],
            "home_team": fixture["teams"]["home"]["name"],
            "home_team_id": fixture["teams"]["home"]["id"],
            "away_team": fixture["teams"]["away"]["name"],
            "away_team_id": fixture["teams"]["away"]["id"],
            "venue": fixture["fixture"]["venue"]["name"] if fixture["fixture"].get("venue") else None
        })
    
    return {"matches": matches, "count": len(matches)}

@app.get("/predict/{fixture_id}", response_model=PredictionResponse)
async def predict_match(fixture_id: int):
    """Predict match outcome dynamically detecting the league"""
    
    # 1. Get fixture info to identify the league
    data = await fetch_api_football("fixtures", {"id": fixture_id})
    
    if not data.get("response"):
        raise HTTPException(status_code=404, detail="Match not found")
    
    fixture = data["response"][0]
    league_data = fixture["league"]
    league_id = league_data["id"]
    home_team = fixture["teams"]["home"]
    away_team = fixture["teams"]["away"]
    
    # 2. Get team statistics using the detected league_id
    home_stats = await get_team_stats(home_team["id"], league_id)
    away_stats = await get_team_stats(away_team["id"], league_id)
    
    # Build features
    features = np.array([[
        home_stats["gf"], home_stats["ga"], home_stats["cs"], home_stats["fw"], home_stats["fd"], home_stats["fl"],
        away_stats["gf"], away_stats["ga"], away_stats["cs"], away_stats["fw"], away_stats["fd"], away_stats["fl"],
    ]])
    
    # Predict
    probabilities = model.predict_proba(features)[0]
    predictions = {
        "home": round(float(probabilities[0]), 4),
        "draw": round(float(probabilities[1]), 4),
        "away": round(float(probabilities[2]), 4)
    }
    
    odds = await get_odds(fixture_id)
    value_bets = []
    if odds:
        value_bets = calculate_value_bets(predictions, odds)
    
    return PredictionResponse(
        fixture_id=fixture_id,
        league_id=league_id,
        league_name=league_data["name"],
        home_team=home_team["name"],
        away_team=away_team["name"],
        date=fixture["fixture"]["date"],
        features={"home": home_stats, "away": away_stats},
        predictions=predictions,
        odds=odds,
        value_bets=value_bets
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
 
