"""
BILORIE + Trend Magic  ─  Analysis & Backtest Bot  (v3 – MT5 Edition)
=====================================================================
Changes vs v2:
  • Replaced ccxt with MetaTrader5 (mt5) for data fetching
  • Prints ALL signals in the backtest window (not just last N)
  • Full backtest summary with P&L, win rate, avg win/loss, etc.
  • Signal log format: YYYY-MM-DD HH:MM UTC | BUY/SELL @ price

Requirements:
  pip install MetaTrader5 pandas numpy

NOTE: MT5 terminal must be running and logged in before executing.

Parameters (from screenshots):
  Length           : 14      Sell Arm Min    : 70
  Sell Trigger Max : 40      Buy Arm Max     : 30
  Buy Trigger Min  : 50      MA Type         : SMA
  CCI Period       : 20      ATR Multiplier  : 2
  ATR Period       : 5       TM Smooth       : OFF
  Source           : Close
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ══════════════════════════════════════════════
#  USER CONFIG  ─ edit here
# ══════════════════════════════════════════════
SYMBOL        = "EURUSD"       # MT5 symbol name (e.g. EURUSD, BTCUSD, DOGEUSD)
TIMEFRAME     = mt5.TIMEFRAME_M1  # mt5.TIMEFRAME_M1/M5/M15/H1/H4/D1 …
TIMEFRAME_STR = "M1"           # human-readable label
LIMIT         = 2000           # max candles to fetch (>= 300 recommended)
BACKTEST_DAYS = 1              # days to scan for signals  ← USER INPUT

# ── BILORIE parameters ────────────────────────
RSI_LEN       = 14
BUY_ARM_MAX   = 30             # arm fires when RSI <= this
BUY_TRG_MIN   = 50             # trigger fires when armed + RSI >= this
SELL_ARM_MIN  = 70             # arm fires when RSI >= this
SELL_TRG_MAX  = 40             # trigger fires when armed + RSI <= this

# ── Trend Magic parameters ────────────────────
CCI_PERIOD    = 20
ATR_MULT      = 2.0
ATR_PERIOD    = 5
TM_SOURCE     = "close"
TM_SMOOTH     = False
TM_SMOOTH_LEN = 14

# ── Backtest P&L config ───────────────────────
TRADE_SIZE    = 1.0            # lot size or contract size for P&L calc
PIP_VALUE     = 1.0            # USD value per pip/point (adjust to your instrument)


# ══════════════════════════════════════════════
#  SOURCE HELPER
# ══════════════════════════════════════════════
def get_source(df: pd.DataFrame, src: str) -> pd.Series:
    src = src.lower()
    mapping = {
        "open"  : df["open"],
        "high"  : df["high"],
        "low"   : df["low"],
        "close" : df["close"],
        "hl2"   : (df["high"] + df["low"]) / 2,
        "hlc3"  : (df["high"] + df["low"] + df["close"]) / 3,
        "ohlc4" : (df["open"] + df["high"] + df["low"] + df["close"]) / 4,
    }
    if src not in mapping:
        raise ValueError(f"Unknown source '{src}'. Choose: {list(mapping)}")
    return mapping[src]


# ══════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ══════════════════════════════════════════════

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA – matches TradingView ta.rma()."""
    alpha  = 1.0 / length
    result = np.full(len(series), np.nan)
    vals   = series.values
    first  = next((i for i, v in enumerate(vals) if not np.isnan(v)), None)
    if first is None:
        return pd.Series(result, index=series.index)
    result[first] = vals[first]
    for i in range(first + 1, len(vals)):
        result[i] = (vals[i] * alpha + result[i - 1] * (1.0 - alpha)
                     if not np.isnan(vals[i]) else result[i - 1])
    return pd.Series(result, index=series.index)


def calc_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    avg_g = rma(delta.clip(lower=0), length)
    avg_l = rma((-delta).clip(lower=0), length)
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=length).mean()
    md = tp.rolling(window=length).apply(
             lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def calc_trend_magic(df: pd.DataFrame, source_name: str,
                     cci_period: int, atr_mult: float, atr_period: int,
                     smooth: bool = False, smooth_len: int = 14) -> pd.Series:
    src      = get_source(df, source_name).values
    cci_vals = calc_cci(df["high"], df["low"], df["close"], cci_period).values
    atr_vals = calc_atr(df["high"], df["low"], df["close"], atr_period).values

    upT   = src - atr_vals * atr_mult
    downT = src + atr_vals * atr_mult
    n  = len(df)
    tm = np.full(n, np.nan)

    for i in range(n):
        c, u, d = cci_vals[i], upT[i], downT[i]
        if np.isnan(c) or np.isnan(u) or np.isnan(d):
            continue
        if i == 0 or np.isnan(tm[i - 1]):
            tm[i] = u if c >= 0 else d
        else:
            prev = tm[i - 1]
            tm[i] = (u if prev < u else prev) if c >= 0 else (d if prev > d else prev)

    tm_series = pd.Series(tm, index=df.index)
    return sma(tm_series, smooth_len) if smooth else tm_series


# ══════════════════════════════════════════════
#  SIGNAL ENGINE
# ══════════════════════════════════════════════

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]     = calc_rsi(df["close"], RSI_LEN)
    df["tm"]      = calc_trend_magic(df, TM_SOURCE, CCI_PERIOD, ATR_MULT,
                                     ATR_PERIOD, TM_SMOOTH, TM_SMOOTH_LEN)
    df["tm_bull"] = df["close"] > df["tm"]

    rsi_arr   = df["rsi"].values
    close_arr = df["close"].values
    tm_arr    = df["tm"].values
    signals   = [None] * len(df)
    armed_buy = armed_sell = False

    for i in range(1, len(df)):
        r, r_prev, c, tm = rsi_arr[i], rsi_arr[i-1], close_arr[i], tm_arr[i]
        if np.isnan(r) or np.isnan(r_prev) or np.isnan(tm):
            continue

        if r <= BUY_ARM_MAX:
            armed_buy  = True
        if r >= SELL_ARM_MIN:
            armed_sell = True

        if armed_buy and r_prev < BUY_TRG_MIN <= r and c > tm:
            signals[i] = "BUY"
            armed_buy = armed_sell = False
        elif armed_sell and r_prev > SELL_TRG_MAX >= r and c < tm:
            signals[i] = "SELL"
            armed_sell = armed_buy = False

    df["signal"] = signals
    return df


# ══════════════════════════════════════════════
#  MT5 DATA FETCH
# ══════════════════════════════════════════════

def fetch_ohlcv_mt5(symbol: str, timeframe, limit: int) -> pd.DataFrame:
    if not mt5.initialize():
        raise SystemExit(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, limit)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise SystemExit(f"[ERROR] No data returned for '{symbol}'. "
                         f"Check symbol name and MT5 connection. Error: {mt5.last_error()}")

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "open", "high": "high",
        "low": "low", "close": "close",
        "tick_volume": "volume"
    }, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ══════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════

def run_backtest(sig_df: pd.DataFrame) -> dict:
    """
    Simple next-bar-open P&L simulation:
      BUY  signal → long  at next candle close price
      SELL signal → short at next candle close price
      Each trade closes on the opposite signal.
    """
    sig_rows = sig_df[sig_df["signal"].notna()].reset_index()
    if sig_rows.empty:
        return {}

    trades    = []
    position  = None   # {"side": "BUY"/"SELL", "entry": float, "time": ts}

    for _, row in sig_rows.iterrows():
        sig   = row["signal"]
        price = row["close"]
        ts    = row["time"] if "time" in row else row.name

        if position is None:
            # Open first trade
            position = {"side": sig, "entry": price, "time": ts}
        else:
            if sig != position["side"]:
                # Close current trade
                entry = position["entry"]
                side  = position["side"]
                exit_ = price
                if side == "BUY":
                    pnl = (exit_ - entry) * TRADE_SIZE
                else:
                    pnl = (entry - exit_) * TRADE_SIZE
                trades.append({
                    "side"      : side,
                    "entry"     : entry,
                    "exit"      : exit_,
                    "entry_time": position["time"],
                    "exit_time" : ts,
                    "pnl"       : pnl,
                    "win"       : pnl > 0,
                })
                # Open new trade in opposite direction
                position = {"side": sig, "entry": price, "time": ts}

    if not trades:
        return {}

    trades_df  = pd.DataFrame(trades)
    buy_trades  = trades_df[trades_df["side"] == "BUY"]
    sell_trades = trades_df[trades_df["side"] == "SELL"]
    wins        = trades_df[trades_df["win"]]
    losses      = trades_df[~trades_df["win"]]

    return {
        "total_trades" : len(trades_df),
        "buy_trades"   : len(buy_trades),
        "sell_trades"  : len(sell_trades),
        "total_pnl"    : trades_df["pnl"].sum(),
        "buy_pnl"      : buy_trades["pnl"].sum() if not buy_trades.empty else 0,
        "sell_pnl"     : sell_trades["pnl"].sum() if not sell_trades.empty else 0,
        "win_count"    : len(wins),
        "loss_count"   : len(losses),
        "win_rate"     : len(wins) / len(trades_df) * 100,
        "avg_win"      : wins["pnl"].mean() if not wins.empty else 0,
        "avg_loss"     : losses["pnl"].mean() if not losses.empty else 0,
        "best_trade"   : trades_df["pnl"].max(),
        "worst_trade"  : trades_df["pnl"].min(),
        "gross_profit" : wins["pnl"].sum() if not wins.empty else 0,
        "gross_loss"   : losses["pnl"].sum() if not losses.empty else 0,
        "profit_factor": (abs(wins["pnl"].sum()) / abs(losses["pnl"].sum())
                          if not losses.empty and losses["pnl"].sum() != 0 else float("inf")),
    }


# ══════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════

W = 62

def div(ch="="):
    print(ch * W)

def fmt_time(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")

def print_header():
    div()
    print("  BILORIE + TREND MAGIC  ─  MT5 Backtest Bot  v3")
    div()
    print(f"  Symbol    : {SYMBOL:<14}  Timeframe : {TIMEFRAME_STR}")
    print(f"  Backtest  : last {BACKTEST_DAYS} day(s)  (UTC)")
    div()

def print_params():
    smooth_lbl = f"SMA({TM_SMOOTH_LEN})" if TM_SMOOTH else "OFF"
    print()
    print("  BILORIE PARAMETERS")
    print(f"  RSI Length : {RSI_LEN:<6}  Buy Arm  <= {BUY_ARM_MAX}   Buy Trig  >= {BUY_TRG_MIN}")
    print(f"  Sell Arm  >= {SELL_ARM_MIN:<6}  Sell Trig <= {SELL_TRG_MAX}")
    print()
    print("  TREND MAGIC PARAMETERS")
    print(f"  CCI Period : {CCI_PERIOD:<6}  ATR Period : {ATR_PERIOD:<6}  ATR Mult : {ATR_MULT}")
    print(f"  Source     : {TM_SOURCE:<6}  Smooth TM  : {smooth_lbl}")
    print()

def print_all_signals(sig_df: pd.DataFrame):
    """Print every signal in the backtest window."""
    signals = sig_df[sig_df["signal"].notna()]
    div("-")
    print(f"  ALL SIGNALS  ({len(signals)} total)")
    div("-")
    if signals.empty:
        print("  No signals found in this window.")
        print("  Tip: increase BACKTEST_DAYS or use a shorter timeframe.")
        print()
        return

    for ts, row in signals.iterrows():
        sig   = row["signal"]
        arrow = "BUY " if sig == "BUY" else "SELL"
        print(f"{fmt_time(ts)} | {arrow} @ {row['close']:.5f}")
    print()

def print_backtest_summary(metrics: dict, sig_count: int):
    div()
    print("BACKTEST SUMMARY")
    div()
    if not metrics:
        print("  Not enough signals for a completed trade.")
        div()
        return

    print(f"Total Signals : {sig_count}")
    print(f"Total Trades  : {metrics['total_trades']}")
    print(f"  - Buy Trades  : {metrics['buy_trades']}")
    print(f"  - Sell Trades : {metrics['sell_trades']}")
    print()
    print("P&L Breakdown:")
    print(f"  - Total P&L   : ${metrics['total_pnl']:>12.2f}")
    print(f"  - Buy P&L     : ${metrics['buy_pnl']:>12.2f}")
    print(f"  - Sell P&L    : ${metrics['sell_pnl']:>12.2f}")
    print(f"  - Gross Profit: ${metrics['gross_profit']:>12.2f}")
    print(f"  - Gross Loss  : ${metrics['gross_loss']:>12.2f}")
    print(f"  - Profit Factor: {metrics['profit_factor']:.2f}")
    print()
    div()
    print("Performance Metrics:")
    print(f"  - Win Rate    : {metrics['win_rate']:.1f}%  "
          f"({metrics['win_count']}/{metrics['total_trades']})")
    print(f"  - Avg Win     : ${metrics['avg_win']:>10.2f}")
    print(f"  - Avg Loss    : ${metrics['avg_loss']:>10.2f}")
    print(f"  - Best Trade  : ${metrics['best_trade']:>10.2f}")
    print(f"  - Worst Trade : ${metrics['worst_trade']:>10.2f}")
    div()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print_header()
    print_params()

    print("  Fetching OHLCV data from MT5 ...")
    df = fetch_ohlcv_mt5(SYMBOL, TIMEFRAME, LIMIT)
    print(f"  Fetched {len(df)} candles")
    print(f"  Range  : {fmt_time(df.index[0])}  →  {fmt_time(df.index[-1])}")
    print()

    print("  Computing RSI + Trend Magic + Signals ...")
    df = compute_signals(df)
    print("  Done.\n")

    # ── Backtest slice ─────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS)
    bt_df  = df[df.index >= cutoff].copy()

    print(f"  Backtest Window : {fmt_time(bt_df.index[0])}  →  {fmt_time(bt_df.index[-1])}")
    print(f"  Candles         : {len(bt_df)}")
    print()

    # ── Print all signals ──────────────────────
    print_all_signals(bt_df)

    # ── Run backtest & print summary ───────────
    metrics   = run_backtest(bt_df)
    sig_count = bt_df["signal"].notna().sum()
    print_backtest_summary(metrics, sig_count)


if __name__ == "__main__":
    main()