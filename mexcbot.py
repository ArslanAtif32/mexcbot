"""
BILORIE + Trend Magic  ─  Analysis & Backtest Bot  (v3 – full stats)
=====================================================================
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ══════════════════════════════════════════════
#  USER CONFIG
# ══════════════════════════════════════════════
EXCHANGE_ID   = "binance"
SYMBOL        = "BTC/USDT"
TIMEFRAME     = "1m"
BACKTEST_DAYS = 99999
TOP_N_SIGNALS = 99999

# ── BILORIE parameters ────────────────────────
RSI_LEN       = 14
BUY_ARM_MAX   = 30
BUY_TRG_MIN   = 50
SELL_ARM_MIN  = 70
SELL_TRG_MAX  = 40

# ── Trend Magic parameters ────────────────────
CCI_PERIOD    = 20
ATR_MULT      = 2.0
ATR_PERIOD    = 5
TM_SOURCE     = "close"
TM_SMOOTH     = False
TM_SMOOTH_LEN = 14

# ── PnL / Backtest settings ───────────────────
TRADE_SIZE_USDT  = 100.0   # fixed position size per trade in USDT
FEE_PCT          = 0.001   # 0.1% per side (Binance taker)
CLOSE_ON         = "next"  # "next" = close at next candle open
                            # "signal" = close at signal candle close


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
        raise ValueError(f"Unknown source '{src}'.")
    return mapping[src]


# ══════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ══════════════════════════════════════════════
def sma(series, length):
    return series.rolling(window=length).mean()

def rma(series: pd.Series, length: int) -> pd.Series:
    alpha  = 1.0 / length
    result = np.full(len(series), np.nan)
    vals   = series.values
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

def calc_rsi(close, length):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    return 100.0 - (100.0 / (1.0 + rma(gain, length) / rma(loss, length).replace(0, np.nan)))

def calc_atr(high, low, close, length):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)

def calc_cci(high, low, close, length):
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=length).mean()
    md = tp.rolling(window=length).apply(
             lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def calc_trend_magic(df, source_name, cci_period, atr_mult,
                     atr_period, smooth=False, smooth_len=14):
    src      = get_source(df, source_name).values
    cci_vals = calc_cci(df["high"], df["low"], df["close"], cci_period).values
    atr_vals = calc_atr(df["high"], df["low"], df["close"], atr_period).values
    upT      = src - atr_vals * atr_mult
    downT    = src + atr_vals * atr_mult
    n        = len(df)
    tm       = np.full(n, np.nan)
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
    open_arr  = df["open"].values
    tm_arr    = df["tm"].values
    signals   = [None] * len(df)
    armed_buy = armed_sell = False

    for i in range(1, len(df)):
        r, r_prev = rsi_arr[i], rsi_arr[i - 1]
        c, tm     = close_arr[i], tm_arr[i]
        if np.isnan(r) or np.isnan(r_prev) or np.isnan(tm):
            continue
        if r <= BUY_ARM_MAX:
            armed_buy  = True
        if r >= SELL_ARM_MIN:
            armed_sell = True
        if armed_buy and r_prev < BUY_TRG_MIN <= r and c > tm:
            signals[i] = "BUY"
            armed_buy  = armed_sell = False
        elif armed_sell and r_prev > SELL_TRG_MAX >= r and c < tm:
            signals[i] = "SELL"
            armed_sell = armed_buy = False

    df["signal"] = signals
    return df


# ══════════════════════════════════════════════
#  PnL ENGINE
# ══════════════════════════════════════════════
def calc_pnl(df: pd.DataFrame) -> list:
    """
    For each BUY/SELL signal, pair it with the next opposite signal
    to close the trade. Entry price = signal candle close (or next open).
    Returns list of trade dicts.
    """
    sig_rows = df[df["signal"].notna()].copy()
    trades   = []
    i        = 0
    rows     = list(sig_rows.itertuples())

    while i < len(rows):
        entry_row = rows[i]
        direction = entry_row.signal          # "BUY" or "SELL"
        exit_type = "SELL" if direction == "BUY" else "BUY"

        # Entry price
        if CLOSE_ON == "next":
            # find next candle open after signal
            loc = df.index.get_loc(entry_row.Index)
            if loc + 1 >= len(df):
                break
            entry_price = df.iloc[loc + 1]["open"]
        else:
            entry_price = entry_row.close

        # Find next opposite signal to close
        j = i + 1
        while j < len(rows) and rows[j].signal != exit_type:
            j += 1

        if j >= len(rows):
            # No closing signal — open trade, skip
            break

        exit_row = rows[j]

        if CLOSE_ON == "next":
            loc2 = df.index.get_loc(exit_row.Index)
            if loc2 + 1 >= len(df):
                break
            exit_price = df.iloc[loc2 + 1]["open"]
        else:
            exit_price = exit_row.close

        # PnL calculation
        qty        = TRADE_SIZE_USDT / entry_price
        fee_entry  = TRADE_SIZE_USDT * FEE_PCT
        fee_exit   = qty * exit_price * FEE_PCT
        gross_pnl  = (exit_price - entry_price) * qty if direction == "BUY" \
                     else (entry_price - exit_price) * qty
        net_pnl    = gross_pnl - fee_entry - fee_exit
        pnl_pct    = (net_pnl / TRADE_SIZE_USDT) * 100

        trades.append({
            "direction"   : direction,
            "entry_time"  : entry_row.Index,
            "exit_time"   : exit_row.Index,
            "entry_price" : entry_price,
            "exit_price"  : exit_price,
            "gross_pnl"   : gross_pnl,
            "net_pnl"     : net_pnl,
            "pnl_pct"     : pnl_pct,
            "win"         : net_pnl > 0,
        })
        i = j  # move to exit signal, next iteration starts from here

    return trades


# ══════════════════════════════════════════════
#  DATA FETCH  (paginated)
# ══════════════════════════════════════════════
def fetch_ohlcv(exchange_id, symbol, timeframe, limit=999999):
    try:
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        exchange.load_markets()
        all_raw = []
        since   = None
        per_req = 1000
        while True:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                       since=since, limit=per_req)
            if not raw:
                break
            all_raw.extend(raw)
            if len(raw) < per_req:
                break
            since = raw[-1][0] + 1
    except ccxt.BadSymbol:
        raise SystemExit(f"[ERROR] Symbol '{symbol}' not found on {exchange_id}.")
    except ccxt.NetworkError as e:
        raise SystemExit(f"[ERROR] Network error: {e}")
    except Exception as e:
        raise SystemExit(f"[ERROR] {e}")

    df = pd.DataFrame(all_raw, columns=["timestamp","open","high","low","close","volume"])
    df.drop_duplicates("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


# ══════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════
W = 72

def fmt_time(ts):
    return ts.strftime("%Y-%m-%d %H:%M UTC")

def div(ch="═"):
    print("  " + ch * W)

def print_header():
    div()
    print("  BILORIE + TREND MAGIC  ─  Analysis & Backtest Bot  v3")
    div()
    print(f"  Exchange  : {EXCHANGE_ID.upper():<14} Symbol    : {SYMBOL}")
    print(f"  Timeframe : {TIMEFRAME:<14} Trade Size: ${TRADE_SIZE_USDT:.0f} USDT  |  Fee: {FEE_PCT*100:.2f}%/side")
    div()

def print_params():
    smooth_lbl = f"SMA({TM_SMOOTH_LEN})" if TM_SMOOTH else "OFF"
    print()
    print("  BILORIE PARAMETERS")
    print(f"  RSI Length : {RSI_LEN:<6} Buy Arm  <= {BUY_ARM_MAX:<5} Buy  Trigger >= {BUY_TRG_MIN}")
    print(f"               {'':6} Sell Arm >= {SELL_ARM_MIN:<5} Sell Trigger <= {SELL_TRG_MAX}")
    print()
    print("  TREND MAGIC PARAMETERS")
    print(f"  CCI Period : {CCI_PERIOD:<6} ATR Period : {ATR_PERIOD:<6} ATR Mult : {ATR_MULT}  Smooth : {smooth_lbl}")
    print()

def print_signals_table(sig_df, n, label):
    all_sig = sig_df[sig_df["signal"].notna()].tail(n)
    print(f"  SIGNALS  [{label}]  —  showing last {min(n, len(all_sig))} of {len(all_sig)} total")
    div("─")
    if all_sig.empty:
        print("  No signals in this window.")
        return
    print(f"  {'#':<4} {'TYPE':<6} {'CANDLE TIME':<22} {'CLOSE':>10}  {'RSI':>6}  {'TM':>10}  TREND")
    div("─")
    for idx, (ts, row) in enumerate(all_sig.iterrows(), 1):
        arrow = "+" if row["signal"] == "BUY" else "-"
        trend = "BULL" if row["tm_bull"] else "BEAR"
        print(f"  {idx:<4} [{arrow}]{row['signal']:<5} {fmt_time(ts):<22} "
              f"{row['close']:>10.5f}  {row['rsi']:>6.2f}  {row['tm']:>10.5f}  {trend}")
    print()
    total_b = (sig_df["signal"] == "BUY").sum()
    total_s = (sig_df["signal"] == "SELL").sum()
    print(f"  Total in window  →  [+] BUY : {total_b}   [-] SELL : {total_s}   ALL : {total_b + total_s}")

def print_trades_table(trades, n=50):
    print()
    print(f"  CLOSED TRADES  —  showing last {max(n, len(trades))} of {len(trades)}")
    div("─")
    if not trades:
        print("  No completed trades (need at least one BUY→SELL or SELL→BUY pair).")
        return
    print(f"  {'#':<4} {'DIR':<5} {'ENTRY TIME':<18} {'EXIT TIME':<18} "
          f"{'ENTRY':>9} {'EXIT':>9} {'NET PnL':>9} {'%':>7}  RESULT")
    div("─")
    for i, t in enumerate(trades[-n:], 1):
        result = "WIN  ✓" if t["win"] else "LOSS ✗"
        sign   = "+" if t["net_pnl"] >= 0 else ""
        print(f"  {i:<4} {t['direction']:<5} "
              f"{fmt_time(t['entry_time']):<18} {fmt_time(t['exit_time']):<18} "
              f"{t['entry_price']:>9.5f} {t['exit_price']:>9.5f} "
              f"{sign}{t['net_pnl']:>8.4f} {sign}{t['pnl_pct']:>6.2f}%  {result}")

def print_stats(trades):
    print()
    div()
    print("  BACKTEST PERFORMANCE SUMMARY")
    div()

    if not trades:
        print("  Not enough signal pairs to compute stats.")
        div()
        return

    total       = len(trades)
    wins        = sum(1 for t in trades if t["win"])
    losses      = total - wins
    win_rate    = (wins / total) * 100
    accuracy    = win_rate                           # same metric, alias
    total_pnl   = sum(t["net_pnl"]  for t in trades)
    total_gross = sum(t["gross_pnl"] for t in trades)
    avg_win     = np.mean([t["net_pnl"] for t in trades if t["win"]])     if wins   else 0
    avg_loss    = np.mean([t["net_pnl"] for t in trades if not t["win"]]) if losses else 0
    best_trade  = max(trades, key=lambda t: t["net_pnl"])
    worst_trade = min(trades, key=lambda t: t["net_pnl"])
    profit_factor = (
        abs(sum(t["net_pnl"] for t in trades if t["win"]))
        / abs(sum(t["net_pnl"] for t in trades if not t["win"]) or 1)
    )

    # Drawdown
    equity = np.cumsum([t["net_pnl"] for t in trades])
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = dd.min()

    buys  = sum(1 for t in trades if t["direction"] == "BUY")
    sells = sum(1 for t in trades if t["direction"] == "SELL")

    s = "+" if total_pnl >= 0 else ""
    print(f"  {'Total Trades':<28}: {total:<6}   (BUY: {buys}  |  SELL: {sells})")
    print(f"  {'Profitable Trades (Wins)':<28}: {wins:<6}   ({win_rate:.1f}%)")
    print(f"  {'Losing  Trades':<28}: {losses:<6}   ({100-win_rate:.1f}%)")
    print(f"  {'Win Rate / Accuracy':<28}: {accuracy:.2f}%")
    div("─")
    print(f"  {'Total Net PnL':<28}: {s}${total_pnl:.4f}  USDT")
    print(f"  {'Total Gross PnL':<28}: {s}${total_gross:.4f}  USDT")
    print(f"  {'Avg Win':<28}: +${avg_win:.4f}  USDT")
    print(f"  {'Avg Loss':<28}: ${avg_loss:.4f}  USDT")
    print(f"  {'Best  Trade':<28}: +${best_trade['net_pnl']:.4f}  ({best_trade['pnl_pct']:+.2f}%)")
    print(f"  {'Worst Trade':<28}: ${worst_trade['net_pnl']:.4f}  ({worst_trade['pnl_pct']:+.2f}%)")
    print(f"  {'Profit Factor':<28}: {profit_factor:.3f}")
    print(f"  {'Max Drawdown':<28}: ${max_dd:.4f}  USDT")
    div("─")
    roi = (total_pnl / TRADE_SIZE_USDT) * 100
    print(f"  {'ROI (on ${:.0f} base)':<28}: {roi:+.2f}%".format(TRADE_SIZE_USDT))
    div()

def print_latest_signal(df):
    last_sig = df[df["signal"].notna()]
    print()
    print("  MOST RECENT SIGNAL")
    div("─")
    if last_sig.empty:
        print("  None found.")
        return
    ts, row = last_sig.index[-1], last_sig.iloc[-1]
    label   = "[+] BUY " if row["signal"] == "BUY" else "[-] SELL"
    trend   = "BULLISH +" if row["tm_bull"] else "BEARISH -"
    print(f"  {label}  @  {fmt_time(ts)}")
    print(f"  Close : {row['close']:.5f}   RSI : {row['rsi']:.2f}   TM : {row['tm']:.5f}   {trend}")

def print_current_state(df):
    last, ts = df.iloc[-1], df.index[-1]
    print()
    print("  CURRENT CANDLE STATE")
    div("─")
    print(f"  Time  : {fmt_time(ts)}")
    print(f"  Close : {last['close']:.5f}")
    rsi_v   = last["rsi"]
    rsi_lbl = ("  ← OVERSOLD   (BUY ARM ACTIVE)"  if rsi_v <= BUY_ARM_MAX  else
               "  ← OVERBOUGHT (SELL ARM ACTIVE)" if rsi_v >= SELL_ARM_MIN else
               "  (neutral zone)")
    print(f"  RSI   : {rsi_v:.2f}{rsi_lbl}")
    trend = "BULLISH + (close > TM)" if last["tm_bull"] else "BEARISH - (close < TM)"
    print(f"  TM    : {last['tm']:.5f}   → {trend}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    print_header()
    print_params()

    print("  Fetching ALL available OHLCV candles (paginated) ...")
    df = fetch_ohlcv(EXCHANGE_ID, SYMBOL, TIMEFRAME)
    print(f"  Fetched  : {len(df):,} candles")
    print(f"  Range    : {fmt_time(df.index[0])}  →  {fmt_time(df.index[-1])}")
    print()

    print("  Computing RSI + Trend Magic + Signals ...")
    df = compute_signals(df)

    # backtest slice
    cutoff = datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS)
    bt_df  = df[df.index >= cutoff]
    print(f"  Backtest : {fmt_time(bt_df.index[0])}  →  {fmt_time(bt_df.index[-1])}")
    print(f"  Candles  : {len(bt_df):,}")
    print()

    label = f"all data  |  {TIMEFRAME}"
    print_signals_table(bt_df, TOP_N_SIGNALS, label)

    print()
    print("  Computing PnL for closed trades ...")
    trades = calc_pnl(bt_df)
    print(f"  Closed trades found : {len(trades)}")

    print_trades_table(trades, n=50)
    print_stats(trades)
    print_latest_signal(df)
    print_current_state(df)
    print()
    div()

if __name__ == "__main__":
    main()