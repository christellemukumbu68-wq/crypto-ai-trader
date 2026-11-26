import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG
# ============================================================
BINANCE_BASE = "https://api.binance.com/api/v3"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "AVAXUSDT"]
LABELS = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "AVAXUSDT": "AVAX/USDT"}

START_BALANCE        = 1000.0
MAX_LOSS_PER_TRADE   = 5.0
MAX_DAILY_LOSS       = 15.0
RISK_REWARD          = 2.0
SL_ATR_MULT          = 1.5
MAX_NOTIONAL_PCT     = 0.25

PORTFOLIO_FILE = Path(os.getenv("PORTFOLIO_FILE", "portfolio.json"))
CONFIG_FILE    = Path(os.getenv("CONFIG_FILE",    "config.json"))

DEFAULT_CONFIG = {
    "auto_refresh": True,
    "refresh_seconds": 15,
    "telegram_enabled": False,
    "telegram_token": "",
    "telegram_chat_id": "",
    "discord_enabled": False,
    "discord_webhook": "",
}

st.set_page_config(page_title="Crypto AI Trader", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.sig-buy  { background: linear-gradient(135deg,#0b8f4b,#14b866); }
.sig-sell { background: linear-gradient(135deg,#b3261e,#d94b42); }
.sig-wait { background: linear-gradient(135deg,#5f6368,#9aa0a6); }
.signal-card {
    padding: 14px; border-radius: 12px; color: white;
    text-align: center; font-weight: 700; font-size: 20px;
}
.price-tag { font-size: 22px; font-weight: 700; }
.small { font-size: 13px; opacity: 0.85; }
[data-testid="stMetricValue"] { font-size: 22px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PERSISTENCE
# ============================================================
def default_portfolio():
    return {
        "balance": START_BALANCE,
        "daily_pnl": 0.0,
        "day": datetime.now(timezone.utc).date().isoformat(),
        "open_trades": [],
        "history": [],
    }

def load_portfolio():
    if PORTFOLIO_FILE.exists():
        try:
            p = json.loads(PORTFOLIO_FILE.read_text())
            for k, v in default_portfolio().items():
                p.setdefault(k, v)
            p["balance"]   = float(p["balance"])
            p["daily_pnl"] = float(p["daily_pnl"])
            return p
        except Exception:
            pass
    return default_portfolio()

def save_portfolio(pf):
    try:
        PORTFOLIO_FILE.write_text(json.dumps(pf, indent=2, default=str))
    except Exception:
        pass

def load_config():
    c = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            c.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    try:
        if "TELEGRAM_TOKEN" in st.secrets:
            c["telegram_token"]   = st.secrets["TELEGRAM_TOKEN"]
            c["telegram_chat_id"] = st.secrets["TELEGRAM_CHAT_ID"]
            c["telegram_enabled"] = True
        if "DISCORD_WEBHOOK" in st.secrets and st.secrets["DISCORD_WEBHOOK"]:
            c["discord_webhook"] = st.secrets["DISCORD_WEBHOOK"]
            c["discord_enabled"] = True
    except Exception:
        pass
    return c

def save_config(cfg):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

# ============================================================
# MARKET DATA
# ============================================================
@st.cache_data(ttl=10, show_spinner=False)
def fetch_klines(symbol, interval, limit=300):
    r = requests.get(
        f"{BINANCE_BASE}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore",
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]

def fetch_klines_paginated(symbol, interval, total=3000, progress_cb=None):
    frames, end_time, remaining = [], None, total
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore",
        ])
        frames.append(df)
        end_time = int(data[0][0]) - 1
        remaining -= len(data)
        if progress_cb: progress_cb(min(1.0, (total - remaining) / total))
        if len(data) < limit:
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]

# ============================================================
# INDICATORS
# ============================================================
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr(df, p=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def add_indicators(df):
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"]   = rsi(df["close"], 14)
    df["atr"]   = atr(df, 14)
    return df

# ============================================================
# SIGNAL LOGIC
# ============================================================
def compute_signal(ema20, ema50, rsi_v, price, trend_1h):
    if ema20 > ema50 and price > ema20 and 40 <= rsi_v <= 70 and trend_1h == "bullish":
        return "BUY"
    if ema20 < ema50 and price < ema20 and 30 <= rsi_v <= 60 and trend_1h == "bearish":
        return "SELL"
    return "WAIT"

def sl_tp_for(signal, price, atr_v):
    if signal == "BUY":
        return price - SL_ATR_MULT*atr_v, price + RISK_REWARD*SL_ATR_MULT*atr_v
    if signal == "SELL":
        return price + SL_ATR_MULT*atr_v, price - RISK_REWARD*SL_ATR_MULT*atr_v
    return 0.0, 0.0

def position_size(price, sl, balance):
    risk = abs(price - sl)
    if risk <= 0: return 0.0, 0.0
    qty_risk     = MAX_LOSS_PER_TRADE / risk
    qty_notional = (MAX_NOTIONAL_PCT * balance) / price
    qty = min(qty_risk, qty_notional)
    return qty, qty * price

# ============================================================
# LIVE ANALYSIS
# ============================================================
def analyze(symbol, balance):
    df5  = add_indicators(fetch_klines(symbol, "5m", 300))
    df1h = add_indicators(fetch_klines(symbol, "1h", 300))
    last5, last1h = df5.iloc[-1], df1h.iloc[-1]
    price = float(last5["close"])

    trend_1h = "bullish" if last1h["ema20"] > last1h["ema50"] else "bearish"
    signal   = compute_signal(last5["ema20"], last5["ema50"], last5["rsi"], price, trend_1h)
    sl, tp   = sl_tp_for(signal, price, float(last5["atr"]))
    qty, notional = position_size(price, sl, balance) if signal != "WAIT" else (0.0, 0.0)

    reason = {
        "BUY":  "Tendance 5m + 1h haussières, RSI OK",
        "SELL": "Tendance 5m + 1h baissières, RSI OK",
        "WAIT": "Pas de confluence",
    }[signal]

    return dict(
        symbol=symbol, price=price, ema20=float(last5["ema20"]),
        ema50=float(last5["ema50"]), rsi=float(last5["rsi"]),
        atr=float(last5["atr"]), trend_1h=trend_1h, signal=signal,
        reason=reason, sl=sl, tp=tp, qty=qty, notional=notional, df5=df5,
    )

# ============================================================
# TRADE MGMT
# ============================================================
def reset_daily_if_needed(pf):
    today = datetime.now(timezone.utc).date().isoformat()
    if pf["day"] != today:
        pf["day"] = today
        pf["daily_pnl"] = 0.0

def check_open_trades(pf, prices):
    still_open = []
    for t in pf["open_trades"]:
        p = prices.get(t["symbol"])
        if p is None:
            still_open.append(t); continue
        hit_sl = hit_tp = False
        if t["side"] == "BUY":
            if p <= t["sl"]: hit_sl = True
            elif p >= t["tp"]: hit_tp = True
        else:
            if p >= t["sl"]: hit_sl = True
            elif p <= t["tp"]: hit_tp = True
        if hit_sl or hit_tp:
            exit_price = t["sl"] if hit_sl else t["tp"]
            reason     = "SL" if hit_sl else "TP"
            pnl = (exit_price - t["entry"]) * t["qty"] * (1 if t["side"]=="BUY" else -1)
            pf["balance"]   += pnl
            pf["daily_pnl"] += pnl
            pf["history"].append({
                **t, "exit": exit_price, "pnl": pnl, "reason": reason,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            still_open.append(t)
    pf["open_trades"] = still_open

def open_trade(pf, a):
    if a["signal"] not in ("BUY", "SELL"): return False, "Signal invalide"
    if pf["daily_pnl"] <= -MAX_DAILY_LOSS:
        return False, f"Limite quotidienne atteinte ({MAX_DAILY_LOSS:.0f} $)"
    if any(t["symbol"] == a["symbol"] for t in pf["open_trades"]):
        return False, f"Position déjà ouverte sur {LABELS[a['symbol']]}"
    pf["open_trades"].append({
        "symbol": a["symbol"], "side": a["signal"], "entry": a["price"],
        "sl": a["sl"], "tp": a["tp"], "qty": a["qty"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
    })
    return True, "OK"

def unrealized_pnl(pf, prices):
    total = 0.0
    for t in pf["open_trades"]:
        p = prices.get(t["symbol"])
        if p is None: continue
        diff = (p - t["entry"]) if t["side"]=="BUY" else (t["entry"] - p)
        total += diff * t["qty"]
    return total

# ============================================================
# ALERTS
# ============================================================
def send_telegram(cfg, msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": msg, "parse_mode": "Markdown"},
            timeout=6,
        )
    except Exception: pass

def send_discord(cfg, msg):
    try:
        requests.post(cfg["discord_webhook"], json={"content": msg}, timeout=6)
    except Exception: pass

def send_alert(cfg, msg):
    if cfg.get("telegram_enabled") and cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        send_telegram(cfg, msg)
    if cfg.get("discord_enabled") and cfg.get("discord_webhook"):
        send_discord(cfg, msg)

# ============================================================
# BACKTEST
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def run_backtest(symbol, n_5m=3000, initial_balance=START_BALANCE):
    df5  = add_indicators(fetch_klines_paginated(symbol, "5m", n_5m))
    hours_needed = (n_5m * 5) // 60 + 60
    df1h = add_indicators(fetch_klines_paginated(symbol, "1h", hours_needed))

    df1h_small = df1h.sort_values("open_time").rename(
        columns={"ema20":"ema20_1h","ema50":"ema50_1h"}
    )[["open_time","ema20_1h","ema50_1h"]]

    merged = pd.merge_asof(
        df5.sort_values("open_time"),
        df1h_small,
        on="open_time", direction="backward",
    )

    balance, trades, open_t = initial_balance, [], None

    for i in range(60, len(merged) - 1):
        row = merged.iloc[i]
        nxt = merged.iloc[i + 1]

        if open_t:
            exit_price = reason = None
            if open_t["side"] == "BUY":
                if nxt["low"]  <= open_t["sl"]: exit_price, reason = open_t["sl"], "SL"
                elif nxt["high"] >= open_t["tp"]: exit_price, reason = open_t["tp"], "TP"
            else:
                if nxt["high"] >= open_t["sl"]: exit_price, reason = open_t["sl"], "SL"
                elif nxt["low"]  <= open_t["tp"]: exit_price, reason = open_t["tp"], "TP"
            if exit_price is not None:
                pnl = ((exit_price - open_t["entry"]) if open_t["side"]=="BUY"
                       else (open_t["entry"] - exit_price)) * open_t["qty"]
                balance += pnl
                trades.append({**open_t, "exit": exit_price, "pnl": pnl, "reason": reason,
                               "closed_at": str(nxt["open_time"])})
                open_t = None
            continue

        if pd.isna(row["ema20_1h"]):
            continue

        price     = row["close"]
        trend_1h  = "bullish" if row["ema20_1h"] > row["ema50_1h"] else "bearish"
        signal    = compute_signal(row["ema20"], row["ema50"], row["rsi"], price, trend_1h)
        if signal == "WAIT":
            continue

        sl, tp = sl_tp_for(signal, price, row["atr"])
        qty, _ = position_size(price, sl, balance)
        if qty <= 0:
            continue

        open_t = {
            "side": signal, "entry": price, "sl": sl, "tp": tp, "qty": qty,
            "opened_at": str(row["open_time"]),
        }

    if not trades:
        return {
            "symbol": symbol, "trades_df": pd.DataFrame(),
            "balance": balance, "stats": {
                "n_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "final_balance": balance, "return_pct": 0.0,
                "profit_factor": 0.0, "max_dd": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            },
        }

    trades_df  = pd.DataFrame(trades)
    wins       = int((trades_df["pnl"] > 0).sum())
    losses     = int((trades_df["pnl"] <= 0).sum())
    total_pnl  = float(trades_df["pnl"].sum())
    gross_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()))
    pf_ratio   = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = (trades_df["pnl"].cumsum() + initial_balance).values
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak * 100
    max_dd = float(dd.min())

    stats = {
        "n_trades":      len(trades_df),
        "wins":          wins,
        "losses":        losses,
        "win_rate":      wins / len(trades_df) * 100,
        "total_pnl":     total_pnl,
        "final_balance": balance,
        "return_pct":    (balance - initial_balance) / initial_balance * 100,
        "profit_factor": pf_ratio,
        "max_dd":        max_dd,
        "avg_win":       float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins else 0.0,
        "avg_loss":      float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()) if losses else 0.0,
    }
    return {"symbol": symbol, "trades_df": trades_df,
            "balance": balance, "stats": stats}

# ============================================================
# INIT STATE & SIDEBAR
# ============================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()
if "config" not in st.session_state:
    st.session_state.config = load_config()
if "last_signals" not in st.session_state:
    st.session_state.last_signals = {}

pf   = st.session_state.portfolio
cfg  = st.session_state.config
reset_daily_if_needed(pf)

with st.sidebar:
    st.title("🤖 Crypto AI Trader")
    st.caption("Binance Spot · Simulation uniquement")

    st.markdown("### ⚙️ Rafraîchissement")
    cfg["auto_refresh"]    = st.toggle("Auto-refresh", value=cfg["auto_refresh"])
    cfg["refresh_seconds"] = st.slider(
        "Intervalle (s)", 10, 120, cfg["refresh_seconds"], step=5,
        disabled=not cfg["auto_refresh"],
    )

    st.markdown("### 📢 Alertes")
    cfg["telegram_enabled"] = st.toggle("Telegram", value=cfg["telegram_enabled"])
    if cfg["telegram_enabled"]:
        cfg["telegram_token"]   = st.text_input("Bot token",   value=cfg["telegram_token"],   type="password")
        cfg["telegram_chat_id"] = st.text_input("Chat ID",      value=cfg["telegram_chat_id"])

    if st.button("💾 Sauvegarder la config", use_container_width=True):
        save_config(cfg)
        st.success("Config sauvegardée")

    st.divider()
    if st.button("🔁 Réinitialiser le portefeuille", use_container_width=True):
        st.session_state.portfolio = default_portfolio()
        save_portfolio(st.session_state.portfolio)
        st.success("Portefeuille réinitialisé à 1000 $")
        st.rerun()

if cfg["auto_refresh"]:
    st_autorefresh(interval=cfg["refresh_seconds"] * 1000, key="autorefresh")

# ============================================================
# FETCH & ANALYZE
# ============================================================
results, errors, prices = [], [], {}
for s in SYMBOLS:
    try:
        a = analyze(s, pf["balance"])
        results.append(a)
        prices[s] = a["price"]
    except Exception as e:
        errors.append((s, str(e)))

if errors:
    for s, e in errors:
        st.error(f"{LABELS.get(s,s)} : {e}")

before = len(pf["open_trades"])
check_open_trades(pf, prices)
if len(pf["open_trades"]) < before:
    save_portfolio(pf)
    for h in pf["history"][-3:]:
        send_alert(cfg, f"🔔 *{LABELS[h['symbol']]}* {h['side']} fermé ({h['reason']})\nPnL : {h['pnl']:+.2f} $")

for a in results:
    prev = st.session_state.last_signals.get(a["symbol"])
    if a["signal"] in ("BUY", "SELL") and prev != a["signal"]:
        send_alert(cfg, f"🚨 *Signal {a['signal']}* — {LABELS[a['symbol']]}\nPrix : {a['price']:,.4f}\nSL : {a['sl']:,.4f} · TP : {a['tp']:,.4f}")
    st.session_state.last_signals[a["symbol"]] = a["signal"]

# ============================================================
# TABS
# ============================================================
tab_dash, tab_bt, tab_hist, tab_settings = st.tabs(["📊 Dashboard", "🧪 Backtest", "📜 Historique", "⚙️ Réglages"])

with tab_dash:
    upnl   = unrealized_pnl(pf, prices)
    equity = pf["balance"] + upnl

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Solde",          f"${pf['balance']:,.2f}")
    k2.metric("📈 Équity",          f"${equity:,.2f}", f"{upnl:+.2f} $")
    k3.metric("📉 PnL du jour",     f"{pf['daily_pnl']:+.2f} $")
    k4.metric("📂 Trades ouverts",  len(pf["open_trades"]))

    st.progress(min(1.0, abs(min(0.0, pf["daily_pnl"])) / MAX_DAILY_LOSS), text=f"Risque quotidien : {abs(min(0.0, pf['daily_pnl'])):.2f} $ / {MAX_DAILY_LOSS:.0f} $")
    st.divider()

    st.subheader("📊 Signaux temps réel")
    for a in results:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 3])
            with c1:
                st.markdown(f"### {LABELS[a['symbol']]}")
                st.markdown(f"<div class='price-tag'>${a['price']:,.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<span class='small'>Tendance 1h : <b>{a['trend_1h']}</b></span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"EMA20 : **{a['ema20']:,.4f}**")
                st.markdown(f"EMA50 : **{a['ema50']:,.4f}**")
                st.markdown(f"RSI 14 : **{a['rsi']:.1f}**")
                st.markdown(f"ATR : {a['atr']:.4f}")
            with c3:
                cls = {"BUY":"sig-buy","SELL":"sig-sell","WAIT":"sig-wait"}[a["signal"]]
                st.markdown(f"<div class='signal-card {cls}'>SIGNAL : {a['signal']}</div>", unsafe_allow_html=True)
                st.caption(a["reason"])
                if a["signal"] in ("BUY", "SELL"):
                    st.markdown(f"**Entry** : {a['price']:,.4f}  \n**SL** : {a['sl']:,.4f}  \n**TP** : {a['tp']:,.4f}  \n**Qté** : {a['qty']:.6f} (~{a['notional']:.2f} $)")
                    if st.button(f"▶️ Simuler {a['signal']}", key=f"btn_{a['symbol']}", use_container_width=True):
                        ok, msg = open_trade(pf, a)
                        if ok:
                            save_portfolio(pf)
                            send_alert(cfg, f"✅ *Trade ouvert* {LABELS[a['symbol']]} {a['signal']}\nEntry {a['price']:,.4f} · SL {a['sl']:,.4f} · TP {a['tp']:,.4f}")
                            st.success("Trade simulé ouvert")
                            st.rerun()
                        else:
                            st.warning(msg)

            with st.expander("📉 Graphique 5 min"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=a["df5"]["open_time"], y=a["df5"]["close"], name="Close", line=dict(color="#888")))
                fig.add_trace(go.Scatter(x=a["df5"]["open_time"], y=a["df5"]["ema20"], name="EMA20", line=dict(color="#2196f3")))
                fig.add_trace(go.Scatter(x=a["df5"]["open_time"], y=a["df5"]["ema50"], name="EMA50", line=dict(color="#ff9800")))
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("📂 Trades ouverts")
    if pf["open_trades"]:
        df_open = pd.DataFrame(pf["open_trades"])
        df_open["PnL latent"] = df_open.apply(lambda r: ((prices.get(r["symbol"], r["entry"]) - r["entry"]) if r["side"]=="BUY" else (r["entry"] - prices.get(r["symbol"], r["entry"]))) * r["qty"], axis=1)
        st.dataframe(df_open[["symbol","side","entry","sl","tp","qty","opened_at","PnL latent"]], use_container_width=True, hide_index=True)
    else:
        st.info("Aucun trade ouvert.")

with tab_bt:
    st.subheader("🧪 Backtest de la stratégie")
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        bt_symbols = st.multiselect("Symboles", SYMBOLS, default=SYMBOLS, format_func=lambda s: LABELS[s])
        n_5m = st.slider("Nombre de bougies 5m", 500, 5000, 2000, step=500)
    with bc2:
        run_btn = st.button("▶️ Lancer le backtest", type="primary", use_container_width=True)

    if run_btn and bt_symbols:
        results_bt = []
        prog = st.progress(0.0, text="Chargement…")
        for i, s in enumerate(bt_symbols):
            prog.progress((i) / len(bt_symbols), text=f"Backtest {LABELS[s]}…")
            try:
                res = run_backtest(s, n_5m=n_5m)
                results_bt.append(res)
            except Exception as e:
                st.error(f"{LABELS[s]} : {e}")
        prog.progress(1.0, text="Terminé")

        for res in results_bt:
            st.markdown(f"#### {LABELS[res['symbol']]}")
            if res["stats"]["n_trades"] == 0:
                st.warning("Aucun trade généré sur cette période.")
                continue
            s = res["stats"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", s["n_trades"])
            m2.metric("Win rate", f"{s['win_rate']:.1f} %")
            m3.metric("PnL total", f"{s['total_pnl']:+.2f} $")
            m4.metric("Rendement", f"{s['return_pct']:+.1f} %")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Profit factor", f"{s['profit_factor']:.2f}")
            m6.metric("Max drawdown", f"{s['max_dd']:.1f} %")
            m7.metric("Avg win", f"{s['avg_win']:+.2f} $")
            m8.metric("Avg loss", f"{s['avg_loss']:+.2f} $")

            trades_df = res["trades_df"]
            eq = trades_df["pnl"].cumsum() + START_BALANCE
            fig = go.Figure(go.Scatter(y=eq.values, mode="lines", name="Équity", line=dict(color="#14b866", width=2)))
            fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", yaxis_title="Équity ($)", xaxis_title="Trade #")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Voir les trades"):
                show = trades_df[["opened_at","side","entry","sl","tp","exit","pnl","reason"]].copy()
                show["pnl"] = show["pnl"].map(lambda v: f"{v:+.2f}")
                st.dataframe(show, use_container_width=True, hide_index=True)

with tab_hist:
    st.subheader("📜 Historique des trades simulés")
    if pf["history"]:
        df_h = pd.DataFrame(pf["history"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Trades clôturés", len(df_h))
        m2.metric("PnL réalisé", f"{df_h['pnl'].sum():+.2f} $")
        m3.metric("Win rate", f"{(df_h['pnl']>0).mean()*100:.1f} %")
        st.dataframe(df_h[["symbol","side","entry","exit","qty","pnl","reason","closed_at"]], use_container_width=True, hide_index=True)
        fig = go.Figure(go.Scatter(y=(df_h["pnl"].cumsum() + START_BALANCE).values, mode="lines", line=dict(color="#2196f3", width=2)))
        fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", yaxis_title="Équity ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun trade clôturé pour le moment.")

with tab_settings:
    st.subheader("🛡️ Règles de gestion du risque")
    st.markdown(f"- **Perte max par trade** : {MAX_LOSS_PER_TRADE:.0f} $\n- **Perte max quotidienne** : {MAX_DAILY_LOSS:.0f} $\n- **Ratio Risque/Récompense** : 1 : {RISK_REWARD:.0f}\n- **Stop-Loss** : {SL_ATR_MULT} × ATR(14)\n- **Taille de position** : plafonnée à {MAX_NOTIONAL_PCT*100:.0f} % du solde\n- **Candles** : signal 5 min, confirmation 1 h\n- **Portefeuille simulé** : {START_BALANCE:.0f} $")
    st.caption("⚠️ Outil pédagogique et de simulation uniquement. Aucun ordre réel n'est envoyé.")

save_portfolio(pf)
