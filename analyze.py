# -*- coding: utf-8 -*-
"""
Trading Agent 自主掃描引擎第一版
讀取 universe.json 與 watchlist.json，分析行情、排序候選標的，
輸出 site/report.json，並在設定 Telegram Secrets 時發送摘要。
本工具使用固定技術規則，不會自動下單，也不構成投資建議。
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# 以程式檔案位置定位專案路徑，避免雲端執行目錄不同造成錯誤。
ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
UNIVERSE_FILE = ROOT / "universe.json"
WATCHLIST_FILE = ROOT / "watchlist.json"
REPORT_FILE = SITE_DIR / "report.json"


def read_json(path: Path) -> dict:
    """讀取 JSON 設定檔，檔案缺漏時明確回報錯誤。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到設定檔：{path.name}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_scan_items() -> tuple[list[dict], int]:
    """合併自主股票池與個人觀察清單，並依 ticker 去除重複項目。"""
    universe = read_json(UNIVERSE_FILE)
    watchlist = read_json(WATCHLIST_FILE)
    items = {}

    # 讀取台股、美股個股與 ETF 分類。
    for category, details in universe.get("markets", {}).items():
        market_name = details.get("name", category)
        for symbol in details.get("symbols", []):
            ticker = str(symbol).strip().upper()
            if ticker:
                items[ticker] = {
                    "ticker": ticker, "name": ticker,
                    "market": market_name, "source": "自主掃描"
                }

    # 個人觀察清單若有相同代號，保留使用者自訂名稱與市場資訊。
    for entry in watchlist.get("watchlist", []):
        ticker = str(entry.get("ticker", "")).strip().upper()
        if ticker:
            items[ticker] = {
                "ticker": ticker,
                "name": entry.get("name", ticker),
                "market": entry.get("market", "個人觀察清單"),
                "source": "個人觀察"
            }

    limit = int(universe.get("scan_settings", {}).get("daily_candidate_limit", 10))
    return list(items.values()), max(1, limit)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """使用 Wilder 平滑概念計算 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # 避免除以零；期間內完全沒有下跌時以 RSI 100 表示。
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - 100 / (1 + rs)).fillna(100)


def analyze_one(item: dict) -> dict:
    """下載單一標的日線資料，計算指標與規則式評分。"""
    ticker = item["ticker"]
    data = yf.Ticker(ticker).history(period="9mo", interval="1d", auto_adjust=True)
    if data.empty:
        raise ValueError("沒有可用行情資料")

    # 檢查必要欄位並移除缺漏列。
    columns = ["Close", "High", "Low", "Volume"]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"缺少行情欄位：{', '.join(missing)}")
    data = data[columns].dropna().copy()
    if len(data) < 60:
        raise ValueError("可用歷史資料不足 60 個交易日")

    close = data["Close"]
