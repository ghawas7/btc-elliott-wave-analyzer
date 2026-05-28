"""
BTC/USD Elliott Wave Analyzer — FastAPI Server
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json

from engine.elliott import analyze, PatternType, FibonacciLevels
from web.fetcher import fetch_btc_data, prepare_analysis_data, TIMEFRAME_CONFIG

app = FastAPI(title="BTC Elliott Wave Analyzer", version="1.0.0")

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def pattern_to_dict(result):
    """Convert PatternResult to JSON-safe dict."""
    if result is None:
        return None
    
    data = {
        "pattern_type": result.pattern_type.value,
        "confidence": round(result.confidence, 2),
        "summary": result.summary,
        "details": result.details,
        "invalidation_level": result.invalidation_level,
        "confirmation_level": result.confirmation_level,
        "waves": [],
        "swings": [],
        "fib_levels": None,
    }
    
    for w in result.waves:
        data["waves"].append({
            "label": w.label,
            "start_price": round(w.start_swing.price, 2),
            "end_price": round(w.end_swing.price, 2),
            "pct_change": round(w.pct_change, 2),
            "start_idx": w.start_swing.index,
            "end_idx": w.end_swing.index,
        })
    
    for s in result.swings:
        data["swings"].append({
            "index": s.index,
            "price": round(s.price, 2),
            "is_high": s.is_high,
            "timestamp": str(s.timestamp),
        })
    
    if result.fib_levels:
        fib = result.fib_levels
        data["fib_levels"] = {
            "high": round(fib.high, 2),
            "low": round(fib.low, 2),
            "retrace_382": round(fib.retrace_382, 2),
            "retrace_500": round(fib.retrace_500, 2),
            "retrace_618": round(fib.retrace_618, 2),
            "retrace_786": round(fib.retrace_786, 2),
            "extension_1272": round(fib.extension_1272, 2),
            "extension_1618": round(fib.extension_1618, 2),
        }
    
    if result.alternate:
        data["alternate"] = pattern_to_dict(result.alternate)
    
    return data


@app.get("/api/analyze")
async def api_analyze(timeframe: str = Query("1d", regex="^(15m|1h|4h|1d|1w)$")):
    """Run Elliott Wave analysis and return JSON."""
    try:
        df = await fetch_btc_data(timeframe)
        prep = prepare_analysis_data(df, timeframe)
        
        result = analyze(
            prep["highs"],
            prep["lows"],
            prep["timestamps"],
            sensitivity=prep["sensitivity"]
        )
        
        # Build response
        response = {
            "timeframe": timeframe,
            "timeframe_label": prep["timeframe_label"],
            "candles_analyzed": len(df),
            "latest_price": round(float(df["close"].iloc[-1]), 2),
            "latest_time": str(df["timestamp"].iloc[-1]),
            "price_range": {
                "high": round(float(df["high"].max()), 2),
                "low": round(float(df["low"].min()), 2),
            },
            "analysis": pattern_to_dict(result),
            # Raw chart data for Plotly
            "chart_data": {
                "timestamps": [str(t) for t in prep["timestamps"]],
                "opens": [round(float(o), 2) for o in prep["opens"]],
                "highs": [round(float(h), 2) for h in prep["highs"]],
                "lows": [round(float(l), 2) for l in prep["lows"]],
                "closes": [round(float(c), 2) for c in prep["closes"]],
            }
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "dashboard.html")
    with open(html_path) as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BTC Elliott Wave Analyzer"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
