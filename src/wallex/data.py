"""Historical bar loading and resampling.

Source: Binance 1-minute klines stored as ``data/parquets/<SYMBOL>USDT.parquet.gz``
(naive UTC ``DatetimeIndex`` named ``time``, columns open/high/low/close/volume/...).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "parquets"
INTERVALS = {1: "1min", 15: "15min", 30: "30min", 60: "60min"}
OHLCV = ["open", "high", "low", "close", "volume"]
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def parquet_path(symbol: str, data_dir: Path | None = None) -> Path:
    """``BTC`` or ``BTCUSDT`` -> ``data/parquets/BTCUSDT.parquet.gz``."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    return (data_dir or DATA_DIR) / f"{sym}.parquet.gz"


def resample(bars_1m: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV bars to ``interval`` minutes (left-labelled, left-closed).

    Bars with no 1-minute data (gaps) are dropped rather than forward-filled.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {sorted(INTERVALS)}, got {interval}")
    if interval == 1:
        return bars_1m[OHLCV]
    out = bars_1m[OHLCV].resample(INTERVALS[interval], label="left", closed="left").agg(_AGG)
    return out.dropna(subset=["close"])


def load_bars(
    symbol: str,
    interval: int = 15,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load ``symbol`` bars resampled to ``interval`` minutes (1/15/30/60), UTC-naive index.

    ``start``/``end`` slice the 1-minute data before resampling (inclusive bounds).
    """
    df = pd.read_parquet(parquet_path(symbol, data_dir), columns=OHLCV)
    df = df.sort_index()
    if start is not None or end is not None:
        df = df.loc[start:end]
    return resample(df, interval)
