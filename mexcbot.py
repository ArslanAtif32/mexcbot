"""
BILORIE + Trend Magic  ─  Analysis & Backtest Bot  (v2 – fixed)
================================================================
KEY FIXES vs v1:
  • Trend Magic is now a proper RATCHETING TRAILING STOP (not raw calc)
      Bullish (CCI>=0): TM only ever moves UP   → max(upT, prev_TM)
      Bearish (CCI< 0): TM only ever moves DOWN → min(downT, prev_TM)
  • Smooth Trend Magic = OFF  (checkbox is unchecked in screenshot)
  • Source = Close (for upT / downT baseline)
  • RSI uses Wilder RMA  (matches TradingView ta.rsi() exactly)
  • Armed latch resets correctly on signal fire

Parameters (from screenshots):
  Length           : 14      Sell Arm Min    : 70
  Sell Trigger Max : 40      Buy Arm Max     : 30
  Buy Trigger Min  : 50      MA Type         : SMA
  CCI Period       : 20      ATR Multiplier  : 2
  ATR Period       : 5       TM Smooth       : OFF
  Source           : Close

Usage:
  pip install ccxt pandas numpy
  python bilorie_bot.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ══════════════════════════════════════════════
#  USER CONFIG  ─ edit here
# ══════════════════════════════════════════════
EXCHANGE_ID   = "binance"      # bybit | kraken | kucoin | okx …
SYMBOL        = "DOGE/USDT"
TIMEFRAME     = "1m"          # 1m | 5m | 15m | 1h | 4h | 1d
LIMIT         = 600            # candles to fetch  (>= 300 recommended)
BACKTEST_DAYS = 1              # days to scan for signals
TOP_N_SIGNALS = 5              # how many recent signals to print

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
TM_SOURCE     = "close"        # open | high | low | close | hl2 | hlc3 | ohlc4
TM_SMOOTH     = False          # matches unchecked checkbox in screenshot
TM_SMOOTH_LEN = 14             # only used if TM_SMOOTH = True


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
        "hlco4" : (df["high"] + df["low"] + df["close"] + df["open"]) / 4,
    }
    if src not in mapping:
        raise ValueError(f"Unknown source '{src}'. Choose from: {list(mapping)}")
    return mapping[src]


# ══════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ══════════════════════════════════════════════

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's RMA (Running Moving Average).
    Matches TradingView ta.rma() exactly.
    alpha = 1 / length
    """
    alpha  = 1.0 / length
    result = np.full(len(series), np.nan)
    vals   = series.values

    # Find first non-NaN index
    first  = next((i for i, v in enumerate(vals) if not np.isnan(v)), None)
    if first is None:
        return pd.Series(result, index=series.index)

    result[first] = vals[first]
    for i in range(first + 1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * vals[i] + (1.0 - alpha) * result[i - 1]
    return pd.Series(result, index=series.index)


def calc_rsi(close: pd.Series, length: int) -> pd.Series:
    """RSI using Wilder RMA – identical to TradingView ta.rsi()."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = rma(gain, length)
    avg_l = rma(loss, length)
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             length: int) -> pd.Series:
    """ATR using Wilder RMA – identical to TradingView ta.atr()."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series,
             length: int) -> pd.Series:
    tp  = (high + low + close) / 3.0
    ma  = tp.rolling(window=length).mean()
    md  = tp.rolling(window=length).apply(
              lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def calc_trend_magic(df: pd.DataFrame,
                     source_name: str,
                     cci_period:  int,
                     atr_mult:    float,
                     atr_period:  int,
                     smooth:      bool = False,
                     smooth_len:  int  = 14) -> pd.Series:
    """
    Trend Magic – ratcheting trailing stop.

    Pine Script (original E.W. Traders logic):
        upT        = src - ATR * mult          # support (bullish side)
        downT      = src + ATR * mult          # resistance (bearish side)
        TrendMagic = CCI >= 0
                       ? (TrendMagic[1] < upT   ? upT   : TrendMagic[1])
                       : (TrendMagic[1] > downT  ? downT : TrendMagic[1])

    Behaviour:
        CCI >= 0 (bullish): TM ratchets UP only  → max(upT, prev_TM)
        CCI <  0 (bearish): TM ratchets DOWN only → min(downT, prev_TM)
    """
    src      = get_source(df, source_name).values
    cci_vals = calc_cci(df["high"], df["low"], df["close"], cci_period).values
    atr_vals = calc_atr(df["high"], df["low"], df["close"], atr_period).values

    upT   = src - atr_vals * atr_mult     # bullish stop level
    downT = src + atr_vals * atr_mult     # bearish stop level

    n  = len(df)
    tm = np.full(n, np.nan)

    for i in range(n):
        c = cci_vals[i]
        u = upT[i]
        d = downT[i]

        if np.isnan(c) or np.isnan(u) or np.isnan(d):
            continue

        if i == 0 or np.isnan(tm[i - 1]):
            # Seed on first valid bar
            tm[i] = u if c >= 0 else d
        else:
            prev = tm[i - 1]
            if c >= 0:
                # Bullish regime: never let TM fall below upT
                tm[i] = u if prev < u else prev
            else:
                # Bearish regime: never let TM rise above downT
                tm[i] = d if prev > d else prev

    tm_series = pd.Series(tm, index=df.index)

    if smooth:
        tm_series = sma(tm_series, smooth_len)

    return tm_series


# ══════════════════════════════════════════════
#  SIGNAL ENGINE
# ══════════════════════════════════════════════

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    State machine:
      BUY  – arm when RSI <= BUY_ARM_MAX (30)
              fire when armed + RSI crosses UP through BUY_TRG_MIN (50)
                   + close > TM (price above Trend Magic = bullish)
      SELL – arm when RSI >= SELL_ARM_MIN (70)
              fire when armed + RSI crosses DOWN through SELL_TRG_MAX (40)
                   + close < TM (price below Trend Magic = bearish)
    """
    df = df.copy()

    df["rsi"]     = calc_rsi(df["close"], RSI_LEN)
    df["tm"]      = calc_trend_magic(df, TM_SOURCE,
                                     CCI_PERIOD, ATR_MULT, ATR_PERIOD,
                                     TM_SMOOTH, TM_SMOOTH_LEN)
    df["tm_bull"] = df["close"] > df["tm"]   # True = price above TM (bullish)

    rsi_arr   = df["rsi"].values
    close_arr = df["close"].values
    tm_arr    = df["tm"].values

    signals    = [None] * len(df)
    armed_buy  = False
    armed_sell = False

    for i in range(1, len(df)):
        r      = rsi_arr[i]
        r_prev = rsi_arr[i - 1]
        c      = close_arr[i]
        tm     = tm_arr[i]

        if np.isnan(r) or np.isnan(r_prev) or np.isnan(tm):
            continue

        # ── Update arm latches first ───────────
        if r <= BUY_ARM_MAX:
            armed_buy  = True
        if r >= SELL_ARM_MIN:
            armed_sell = True

        # ── BUY signal ─────────────────────────
        if (armed_buy
                and r_prev < BUY_TRG_MIN <= r   # RSI crossed above trigger
                and c > tm):                     # price above Trend Magic
            signals[i] = "BUY"
            armed_buy  = False
            armed_sell = False

        # ── SELL signal ────────────────────────
        elif (armed_sell
                and r_prev > SELL_TRG_MAX >= r  # RSI crossed below trigger
                and c < tm):                    # price below Trend Magic
            signals[i] = "SELL"
            armed_sell = False
            armed_buy  = False

    df["signal"] = signals
    return df


# ══════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════

def fetch_ohlcv(exchange_id: str, symbol: str,
                timeframe: str, limit: int) -> pd.DataFrame:
    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        raw      = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BadSymbol:
        raise SystemExit(f"[ERROR] Symbol '{symbol}' not found on {exchange_id}.")
    except ccxt.NetworkError as e:
        raise SystemExit(f"[ERROR] Network error: {e}")
    except Exception as e:
        raise SystemExit(f"[ERROR] {e}")

    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


# ══════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════

W = 68

def fmt_time(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")

def div(ch="═"):
    print("  " + ch * W)

def print_header():
    div()
    print("  BILORIE + TREND MAGIC  ─  Analysis & Backtest Bot  v2")
    div()
    print(f"  Exchange  : {EXCHANGE_ID.upper():<14} Symbol    : {SYMBOL}")
    print(f"  Timeframe : {TIMEFRAME:<14} Backtest  : last {BACKTEST_DAYS} day(s)  (UTC)")
    div()

def print_params():
    smooth_lbl = f"SMA({TM_SMOOTH_LEN})" if TM_SMOOTH else "OFF"
    print()
    print("  BILORIE PARAMETERS")
    print(f"  RSI Length : {RSI_LEN:<6} MA Type         : SMA")
    print(f"  Buy  Arm  <= {BUY_ARM_MAX:<5} Buy  Trigger >= : {BUY_TRG_MIN}")
    print(f"  Sell Arm  >= {SELL_ARM_MIN:<5} Sell Trigger <= : {SELL_TRG_MAX}")
    print()
    print("  TREND MAGIC PARAMETERS")
    print(f"  CCI Period : {CCI_PERIOD:<6} ATR Period  : {ATR_PERIOD:<6} ATR Mult : {ATR_MULT}")
    print(f"  Source     : {TM_SOURCE:<6} Smooth TM   : {smooth_lbl}")
    print()

def print_signals_table(sig_df: pd.DataFrame, n: int, label: str):
    all_sig = sig_df[sig_df["signal"].notna()].tail(n)

    print(f"  LAST {n} SIGNALS  [{label}]")
    div("─")

    if all_sig.empty:
        print("  No signals in this window.")
        print("  Tip: increase BACKTEST_DAYS or use a shorter TIMEFRAME.")
        print()
        return

    print(f"  {'#':<3} {'TYPE':<6} {'CANDLE TIME (UTC)':<25} {'CLOSE':>12}  {'RSI':>6}  {'TM LINE':>12}  TM TREND")
    div("─")
    for idx, (ts, row) in enumerate(all_sig.iterrows(), 1):
        icon  = "BUY  " if row["signal"] == "BUY" else "SELL "
        arrow = "+" if row["signal"] == "BUY" else "-"
        trend = "BULL +" if row["tm_bull"] else "BEAR -"
        print(f"  {idx:<3} [{arrow}]{icon} {fmt_time(ts):<25} "
              f"{row['close']:>12.4f}  {row['rsi']:>6.2f}  {row['tm']:>12.4f}  {trend}")
    print()
    total_b = (sig_df["signal"] == "BUY").sum()
    total_s = (sig_df["signal"] == "SELL").sum()
    print(f"  Window totals  ->  [+] BUY : {total_b}   [-] SELL : {total_s}")

def print_latest_signal(df: pd.DataFrame):
    last_sig = df[df["signal"].notna()]
    print()
    print("  MOST RECENT SIGNAL  (across full fetch history)")
    div("─")
    if last_sig.empty:
        print("  None found in fetched data.")
        return
    ts    = last_sig.index[-1]
    row   = last_sig.iloc[-1]
    is_buy = row["signal"] == "BUY"
    label  = "[+] BUY " if is_buy else "[-] SELL"
    trend  = "BULLISH +" if row["tm_bull"] else "BEARISH -"
    print(f"  {label}  @  {fmt_time(ts)}")
    print(f"  Close : {row['close']:.4f}   RSI : {row['rsi']:.2f}   TM : {row['tm']:.4f}   {trend}")

def print_current_state(df: pd.DataFrame):
    last = df.iloc[-1]
    ts   = df.index[-1]
    print()
    print("  CURRENT CANDLE STATE")
    div("─")
    print(f"  Time   : {fmt_time(ts)}")
    print(f"  Close  : {last['close']:.4f}")
    rsi_v   = last["rsi"]
    rsi_lbl = ""
    if rsi_v <= BUY_ARM_MAX:
        rsi_lbl = "  <-- OVERSOLD   (BUY ARM ACTIVE)"
    elif rsi_v >= SELL_ARM_MIN:
        rsi_lbl = "  <-- OVERBOUGHT (SELL ARM ACTIVE)"
    elif BUY_TRG_MIN <= rsi_v < SELL_ARM_MIN:
        rsi_lbl = "  (neutral zone)"
    print(f"  RSI    : {rsi_v:.2f}{rsi_lbl}")
    trend = "BULLISH + (close > TM)" if last["tm_bull"] else "BEARISH - (close < TM)"
    print(f"  TM     : {last['tm']:.4f}   -> {trend}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print_header()
    print_params()

    print("  Fetching OHLCV data ...")
    df = fetch_ohlcv(EXCHANGE_ID, SYMBOL, TIMEFRAME, LIMIT)
    print(f"  Fetched {len(df)} candles")
    print(f"  Range  : {fmt_time(df.index[0])}  to  {fmt_time(df.index[-1])}")
    print()

    print("  Computing RSI + Trend Magic + signals ...")
    df = compute_signals(df)
    print("  Done.\n")

    # backtest slice
    cutoff = datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS)
    bt_df  = df[df.index >= cutoff]
    print(f"  Backtest window : {fmt_time(bt_df.index[0])}  to  {fmt_time(bt_df.index[-1])}")
    print(f"  Candles         : {len(bt_df)}")
    print()

    label = f"last {BACKTEST_DAYS}d  |  {TIMEFRAME}"
    print_signals_table(bt_df, TOP_N_SIGNALS, label)
    print_latest_signal(df)
    print_current_state(df)
    print()
    div()

if __name__ == "__main__":
    main()