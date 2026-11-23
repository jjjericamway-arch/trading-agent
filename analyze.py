# -*- coding: utf-8 -*-
"""
Trading Agent 免費入門版
功能：
1. 從 Yahoo Finance 取得台股與美股日線資料。
2. 計算 SMA、RSI、MACD 與成交量變化。
3. 依照固定、可重現的規則產生技術面觀察標籤。
4. 輸出繁體中文 JSON 報告給網頁儀表板讀取。
5. 如設定 Telegram Secrets，發送每日摘要通知。

重要限制：
- 本程式是規則式技術分析工具，不是能自主推理的 LLM Agent。
- Yahoo Finance/yfinance 資料可能延遲、中斷或修正。
- 訊號僅供研究，不構成投資建議，也不會下單。
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

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
WATCHLIST_FILE = ROOT / "watchlist.json"
REPORT_FILE = SITE_DIR / "report.json"


def load_watchlist() -> list[dict]:
    """讀取使用者設定的股票清單。"""
    with WATCHLIST_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("watchlist", [])


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """使用 Wilder 平滑概念計算 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # 避免除以零造成無限值。
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))

    # 若期間內完全沒有下跌，RSI 以 100 表示。
    rsi = rsi.fillna(100)
    return rsi


def fetch_and_analyze(item: dict) -> dict:
    """下載單一股票行情並產生技術面觀察結果。"""
    ticker = item["ticker"]
    name = item.get("name", ticker)
    market = item.get("market", "未分類")

    history = yf.Ticker(ticker).history(period="9mo", interval="1d", auto_adjust=True)

    if history.empty or len(history) < 60:
        raise ValueError(f"{ticker} 可用歷史資料不足")

    # 僅保留計算需要的欄位，並清除缺漏資料。
    history = history[["Close", "Volume"]].dropna().copy()
    close = history["Close"]
    volume = history["Volume"]

    # 計算常用技術指標。
    history["SMA20"] = close.rolling(20).mean()
    history["SMA50"] = close.rolling(50).mean()
    history["RSI14"] = calculate_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    history["MACD"] = ema12 - ema26
    history["MACD_signal"] = history["MACD"].ewm(span=9, adjust=False).mean()
    history["Volume20"] = volume.rolling(20).mean()

    latest = history.iloc[-1]
    previous = history.iloc[-2]

    # 以固定規則計分，避免讓語言模型直接決定交易訊號。
    score = 0
    reasons = []

    if latest["Close"] > latest["SMA20"]:
        score += 1
        reasons.append("收盤價高於 20 日均線")
    else:
        score -= 1
        reasons.append("收盤價低於 20 日均線")

    if latest["SMA20"] > latest["SMA50"]:
        score += 1
        reasons.append("20 日均線高於 50 日均線")
    else:
        score -= 1
        reasons.append("20 日均線低於 50 日均線")

    if latest["MACD"] > latest["MACD_signal"]:
        score += 1
        reasons.append("MACD 高於訊號線")
    else:
        score -= 1
        reasons.append("MACD 低於訊號線")

    rsi_value = float(latest["RSI14"])
    if 50 <= rsi_value <= 70:
        score += 1
        reasons.append("RSI 位於 50 至 70 區間")
    elif rsi_value >= 75:
        score -= 1
        reasons.append("RSI 偏高，留意短線過熱")
    elif rsi_value < 30:
        reasons.append("RSI 偏低，留意超賣與反彈風險")
    else:
        reasons.append("RSI 尚未呈現明確強勢")

    if pd.notna(latest["Volume20"]) and latest["Volume20"] > 0:
        if latest["Volume"] > latest["Volume20"] * 1.5 and latest["Close"] > previous["Close"]:
            score += 1
            reasons.append("上漲伴隨成交量高於 20 日均量 1.5 倍")
        else:
            reasons.append("成交量未出現明顯放大上漲")

    if score >= 3:
        label = "偏多"
        summary = "多項技術條件偏向正向，仍須留意價格回檔與市場風險。"
    elif score <= -2:
        label = "偏空"
        summary = "多項技術條件偏弱，應留意趨勢延續及反轉風險。"
    else:
        label = "中性"
        summary = "目前訊號方向不一致，建議持續觀察。"

    # 日期保留行情資料原始索引日期。
    market_date = history.index[-1].strftime("%Y-%m-%d")
    previous_close = float(previous["Close"])
    last_close = float(latest["Close"])
    change_pct = ((last_close / previous_close) - 1) * 100 if previous_close else 0

    return {
        "ticker": ticker,
        "name": name,
        "market": market,
        "data_date": market_date,
        "price": round(last_close, 3),
        "change_pct": round(change_pct, 2),
        "sma20": round(float(latest["SMA20"]), 3),
        "sma50": round(float(latest["SMA50"]), 3),
        "rsi14": round(rsi_value, 2),
        "macd": round(float(latest["MACD"]), 4),
        "macd_signal": round(float(latest["MACD_signal"]), 4),
        "volume": int(latest["Volume"]),
        "score": score,
        "signal": label,
        "summary": summary,
        "reasons": reasons
    }


def send_telegram_message(message: str) -> None:
    """使用 Telegram Bot API 發送文字通知；未設定 Secrets 時自動略過。"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("未設定 Telegram Secrets，略過通知。")
        return

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message
    }).encode("utf-8")

    request = urllib.request.Request(api_url, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError("Telegram 通知傳送失敗")


def main() -> None:
    """執行每日分析並輸出網站報告。"""
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for item in load_watchlist():
        try:
            results.append(fetch_and_analyze(item))
            print(f"完成：{item['ticker']}")
        except Exception as error:
            errors.append({
                "ticker": item.get("ticker", "未知"),
                "error": str(error)
            })
            print(f"失敗：{item.get('ticker')}，原因：{error}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timezone": "Asia/Taipei",
        "method": "固定技術規則計分，非 LLM 自主推理",
        "disclaimer": "資料可能延遲或缺漏；分析僅供研究，不構成投資建議，不會自動下單。",
        "results": results,
        "errors": errors
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    # 只推播有成功取得資料的結果。
    if results:
        lines = ["每日台股／美股技術分析", ""]
        for item in results:
            lines.append(
                f"{item['market']}｜{item['name']} ({item['ticker']})\n"
                f"收盤：{item['price']}｜漲跌：{item['change_pct']}%\n"
                f"觀察：{item['signal']}｜RSI：{item['rsi14']}"
            )
        lines.append("")
        lines.append("此為規則式技術觀察，非投資建議。")
        send_telegram_message("\n\n".join(lines))

    print(f"報告已輸出：{REPORT_FILE}")


if __name__ == "__main__":
    main()
