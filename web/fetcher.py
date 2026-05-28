"""
Coinbase BTC/USD data fetcher — matches Coinbase Pro API granularities.
Native support: 60 (1m), 300 (5m), 900 (15m), 3600 (1h), 21600 (6h), 86400 (1d)
For unsupported granularities (4h, 1w), we resample from supported ones.
"""

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


COINBASE_GRANULARITIES = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "6h":  21600,
    "1d":  86400,
}

TIMEFRAME_CONFIG = {
    "15m": {"granularity": 900,   "label": "15 Minutes", "candles": 280, "sensitivity": 3},
    "1h":  {"granularity": 3600,  "label": "1 Hour",     "candles": 280, "sensitivity": 4},
    "4h":  {"granularity": 3600,  "label": "4 Hours",    "candles": 72, "sensitivity": 5,
            "resample_from": "1h", "resample_factor": 4},
    "1d":  {"granularity": 86400, "label": "Daily",      "candles": 280, "sensitivity": 5},
    "1w":  {"granularity": 86400, "label": "Weekly",     "candles": 40, "sensitivity": 3,
            "resample_from": "1d", "resample_factor": 7},
}


async def fetch_btc_data(timeframe: str = "1d") -> pd.DataFrame:
    """Fetch BTC/USD OHLCV data from Coinbase Pro API."""
    config = TIMEFRAME_CONFIG.get(timeframe)
    if not config:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # If this timeframe needs resampling
    if "resample_from" in config:
        source_tf = config["resample_from"]
        source_config = TIMEFRAME_CONFIG[source_tf]
        candles_needed = config["candles"] * config["resample_factor"]
        df = await _fetch_raw(source_config["granularity"], candles_needed)
        return _resample_ohlc(df, f"{config['resample_factor']}h" if "h" in source_tf else f"{config['resample_factor']}D")
    
    return await _fetch_raw(config["granularity"], config["candles"])


async def _fetch_raw(granularity: int, candles: int) -> pd.DataFrame:
    """Fetch raw candles from Coinbase."""
    end = datetime.utcnow()
    start = end - timedelta(seconds=granularity * candles)
    
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    # Build URL manually to avoid httpx encoding colon chars
    url += f"?start={start.strftime('%Y-%m-%dT%H:%M:%S')}"
    url += f"&end={end.strftime('%Y-%m-%dT%H:%M:%S')}"
    url += f"&granularity={granularity}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    
    if not data:
        raise ValueError("No data returned from Coinbase")
    
    # Coinbase returns [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.assign(timestamp=pd.to_datetime(df["time"], unit="s"))
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    df = df.set_index("timestamp")
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return resampled


def prepare_analysis_data(df: pd.DataFrame, timeframe: str):
    """Convert DataFrame to format needed by the Elliott engine."""
    config = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["1d"])
    
    return {
        "highs": df["high"].values,
        "lows": df["low"].values,
        "closes": df["close"].values,
        "opens": df["open"].values,
        "volumes": df["volume"].values,
        "timestamps": df["timestamp"].tolist(),
        "sensitivity": config["sensitivity"],
        "timeframe_label": config["label"],
    }
