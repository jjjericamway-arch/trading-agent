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
    volume = data["Volume"]
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["RSI14"] = calculate_rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["Volume20"] = volume.rolling(20).mean()

    latest, previous = data.iloc[-1], data.iloc[-2]
    score, reasons = 0, []

    # 趨勢條件一：收盤價相對於 20 日均線的位置。
    if latest["Close"] > latest["SMA20"]:
        score += 1
        reasons.append("收盤價高於 20 日均線")
    else:
        score -= 1
        reasons.append("收盤價低於 20 日均線")

    # 趨勢條件二：短期均線相對於中期均線的位置。
    if latest["SMA20"] > latest["SMA50"]:
        score += 1
        reasons.append("20 日均線高於 50 日均線")
    else:
        score -= 1
        reasons.append("20 日均線低於 50 日均線")

    # 動能條件：MACD 與訊號線的相對位置。
    if latest["MACD"] > latest["MACD_signal"]:
        score += 1
        reasons.append("MACD 高於訊號線")
    else:
        score -= 1
        reasons.append("MACD 低於訊號線")

    rsi = float(latest["RSI14"])
    if 50 <= rsi <= 70:
        score += 1
        reasons.append("RSI 位於 50 至 70 區間")
    elif rsi >= 75:
        score -= 1
        reasons.append("RSI 偏高，留意短線過熱")
    elif rsi < 30:
        reasons.append("RSI 偏低，留意超賣與反彈風險")
    else:
        reasons.append("RSI 尚未呈現明確強勢")

    # 成交量條件：觀察上漲日是否伴隨顯著放量。
    volume_ratio = None
    if pd.notna(latest["Volume20"]) and latest["Volume20"] > 0:
        volume_ratio = float(latest["Volume"] / latest["Volume20"])
        if volume_ratio > 1.5 and latest["Close"] > previous["Close"]:
            score += 1
            reasons.append("上漲伴隨成交量高於 20 日均量 1.5 倍")
        else:
            reasons.append("成交量未出現明顯放大上漲")

    if score >= 3:
        signal, summary = "偏多", "多項技術條件偏向正向，仍須留意回檔與市場風險。"
    elif score <= -2:
        signal, summary = "偏空", "多項技術條件偏弱，應留意趨勢延續及反轉風險。"
    else:
        signal, summary = "中性", "目前訊號方向不一致，建議持續觀察。"

    previous_close = float(previous["Close"])
    price = float(latest["Close"])
    change = ((price / previous_close) - 1) * 100 if previous_close else 0

    return {
        "ticker": ticker,
        "name": item.get("name", ticker),
        "market": item.get("market", "未分類"),
        "source": item.get("source", "自主掃描"),
        "data_date": data.index[-1].strftime("%Y-%m-%d"),
        "price": round(price, 3),
        "change_pct": round(change, 2),
        "sma20": round(float(latest["SMA20"]), 3),
        "sma50": round(float(latest["SMA50"]), 3),
        "rsi14": round(rsi, 2),
        "macd": round(float(latest["MACD"]), 4),
        "macd_signal": round(float(latest["MACD_signal"]), 4),
        "volume": int(latest["Volume"]),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "score": score,
        "signal": signal,
        "summary": summary,
        "reasons": reasons
    }


def send_telegram(message: str) -> None:
    """透過 Telegram Bot API 發送通知；未設定 Secrets 時略過。"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("未設定 Telegram Secrets，略過通知。")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError("Telegram 通知傳送失敗")


def main() -> None:
    """執行掃描、排名、報告輸出與 Telegram 推播。"""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    items, candidate_limit = load_scan_items()
    if not items:
        raise RuntimeError("自主掃描股票池與個人觀察清單皆為空")

    results, errors = [], []

    # 單一標的失敗只記錄錯誤，繼續掃描其餘標的。
    for number, item in enumerate(items, start=1):
        ticker = item["ticker"]
        print(f"[{number}/{len(items)}] 開始分析：{ticker}")
        try:
            result = analyze_one(item)
            results.append(result)
            print(f"完成：{ticker}，評分：{result['score']}")
        except Exception as error:
            errors.append({
                "ticker": ticker,
                "market": item.get("market", "未分類"),
                "error": str(error)
            })
            print(f"失敗：{ticker}，原因：{error}")

    # 依評分及漲跌幅排序，取前 N 檔作為研究候選清單。
    ranked = sorted(
        results,
        key=lambda row: (row["score"], row["change_pct"]),
        reverse=True
    )
    candidates = ranked[:candidate_limit]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timezone": "Asia/Taipei",
        "method": "自主股票池掃描與固定技術規則計分，非 LLM 自主推理",
        "disclaimer": "資料可能延遲或缺漏；研究排序不構成投資建議，不會自動下單。",
        "scan_summary": {
            "total": len(items),
            "successful": len(results),
            "failed": len(errors),
            "candidate_limit": candidate_limit
        },
        "results": ranked,
        "candidates": candidates,
        "errors": errors
    }

    # 將完整分析資料寫入網站供儀表板讀取。
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    # 產生 Telegram 摘要，列出掃描統計與每日候選標的。
    if results:
        lines = [
            "Trading Agent 每日自主掃描報告",
            f"掃描數量：{len(items)}",
            f"成功取得資料：{len(results)}",
            f"資料失敗：{len(errors)}",
            ""
        ]
        for rank, row in enumerate(candidates, start=1):
            lines.append(
                f"{rank}. {row['market']}｜{row['name']} ({row['ticker']})\n"
                f"價格：{row['price']}｜漲跌：{row['change_pct']}%\n"
                f"評分：{row['score']}｜觀察：{row['signal']}｜RSI：{row['rsi14']}"
            )
        lines.extend(["", "此為規則式研究排序，非投資建議，不會自動下單。"])
        send_telegram("\n\n".join(lines))

    print(
        f"掃描完成：總數 {len(items)}，成功 {len(results)}，"
        f"失敗 {len(errors)}，候選 {len(candidates)}"
    )
    print(f"報告已輸出：{REPORT_FILE}")


if __name__ == "__main__":
    main()
