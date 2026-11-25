import os
import json
import numpy as np
import pandas as pd
import requests

WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
ACCOUNT_USD = float(os.environ.get("ACCOUNT_USD") or 0)   # optional, for lot size hint
TEST = os.environ.get("TEST") == "1"

TICKER = "GC=F"
K, RR = 1.5, 3                      # stop = 1.5 ATR, target = 1:3
FAST, MID, SLOW = 20, 50, 200
STATE_FILE = "state.json"
FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
WINDOWS = {"High": (30, 60), "Medium": (15, 30)}
LAGOS = "Africa/Lagos"


def send(text):
    if not WEBHOOK:
        print(text)
        return
    try:
        r = requests.post(WEBHOOK, json={"content": text},
                          headers={"User-Agent": "GoldAlert/1.0"}, timeout=20)
        if r.status_code >= 300:
            print("Discord error:", r.status_code)
    except Exception as ex:
        print("Discord error:", type(ex).__name__)


def download():
    import yfinance as yf
    df = yf.download(TICKER, period="180d", interval="1h",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def to4h(h):
    return h.resample("4h").agg({"Open": "first", "High": "max",
                                 "Low": "min", "Close": "last"}).dropna()


def build(df):
    d = df.copy()
    c = d["Close"]
    d["f"] = c.ewm(span=FAST, adjust=False).mean()
    d["m"] = c.ewm(span=MID, adjust=False).mean()
    d["s"] = c.ewm(span=SLOW, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(),
                    (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return d.dropna()


def sig(d):
    o, h, l, c = (d[x].values for x in ("Open", "High", "Low", "Close"))
    f, m, s = d["f"].values, d["m"].values, d["s"].values
    L = (m > s) & (c > m) & (l <= f) & (c > f) & (c > o)
    S = (m < s) & (c < m) & (h >= f) & (c < f) & (c < o)
    return L, S


def nearby_news(now_utc):
    try:
        r = requests.get(FEED, timeout=20)
        r.raise_for_status()
        out = []
        for e in r.json():
            if e.get("country") != "USD" or e.get("impact") not in WINDOWS:
                continue
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
            if now_utc - pd.Timedelta(minutes=60) <= t <= now_utc + pd.Timedelta(hours=2):
                lag = t.tz_convert(LAGOS).strftime("%H:%M")
                out.append(f"{e['impact']} USD: {e['title']} at {lag} Lagos")
        return out
    except Exception:
        return []


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def main(now=None):
    state = load_state()
    try:
        h = download()
    except Exception as ex:
        print("Data error:", type(ex).__name__)
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        if state.get("last_error_day") != today and not TEST:
            send("Gold bot: could not download price data. It will keep trying.")
            state["last_error_day"] = today
            save_state(state)
        return
    d4 = to4h(h)
    if now is None:
        now = pd.Timestamp.now(tz=d4.index.tz)
    if len(d4) and d4.index[-1] + pd.Timedelta(hours=4) > now:
        d4 = d4.iloc[:-1]          # last 4h candle is still forming
    d = build(d4)
    if len(d) < 50:
        print("Not enough data")
        return
    L, S = sig(d)
    last_t = d.index[-1]
    close = float(d["Close"].iloc[-1])
    atr = float(d["atr"].iloc[-1])
    risk = K * atr
    bar_end = last_t + pd.Timedelta(hours=4)
    age_h = (now - bar_end).total_seconds() / 3600
    side = "BUY" if L[-1] else ("SELL" if S[-1] else None)
    bar_id = last_t.isoformat()
    now_utc = now.tz_convert("UTC")
    lagos_bar = last_t.tz_convert(LAGOS).strftime("%a %H:%M")
    trend = "up" if d["m"].iloc[-1] > d["s"].iloc[-1] else "down"

    if TEST:
        send(f"Gold bot test OK. Last closed 4h bar started {lagos_bar} Lagos. "
             f"Trend: {trend}. Signal now: {side or 'none'}. ATR ${atr:.1f}.")
        return

    if side and state.get("last_alert") != bar_id and age_h <= 6:
        if side == "BUY":
            stop, tp = close - risk, close + RR * risk
            word_s, word_t, icon = "below", "above", "🟢"
        else:
            stop, tp = close + risk, close - RR * risk
            word_s, word_t, icon = "above", "below", "🔴"
        lines = [
            f"{icon} GOLD {side} SIGNAL (4h)",
            f"4h bar started {lagos_bar} Lagos and has closed.",
            "Enter at market now, then set from YOUR fill price:",
            f"  Stop: ${risk:.1f} {word_s} entry",
            f"  Target: ${RR * risk:.1f} {word_t} entry (1:{RR})",
            f"(Futures reference: price {close:.1f}, stop {stop:.1f}, target {tp:.1f}."
            " Your broker's gold price can differ by a few dollars, so use the distances.)",
            f"Risk at 0.01 lot is about ${risk:.0f}.",
        ]
        if ACCOUNT_USD > 0:
            lots = (ACCOUNT_USD * 0.01) / (risk * 100)
            lots = np.floor(lots * 100) / 100
            if lots < 0.01:
                lines.append(f"Warning: 0.01 lot risks more than 1% of ${ACCOUNT_USD:,.0f}. Skip or use a smaller-lot broker.")
            else:
                lines.append(f"Lot size for 1% risk on ${ACCOUNT_USD:,.0f}: about {lots:.2f}")
        news = nearby_news(now_utc)
        if news:
            lines.append("News nearby, be careful:")
            lines += [f"  - {n}" for n in news]
        lines.append("Paper-trade first and log it. Not financial advice.")
        send("\n".join(lines))
        state["last_alert"] = bar_id

    today = now_utc.strftime("%Y-%m-%d")
    if state.get("last_heartbeat") != today and now_utc.hour >= 7:
        send(f"Gold bot alive. Last closed 4h bar started {lagos_bar} Lagos. "
             f"Trend: {trend}. No action needed unless a signal is posted.")
        state["last_heartbeat"] = today

    save_state(state)


if __name__ == "__main__":
    main()
