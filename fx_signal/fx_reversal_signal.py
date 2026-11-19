#!/usr/bin/env python3
"""
FX currency-strength REVERSAL: daily signal calculator (paper-trading / forward-test tool)
=========================================================================================

Idea (found by backtest, NOT proven): over the last `lookback` trading days, rank the 7 major
currencies by their performance against USD.  BUY the weakest `k`, SELL the strongest `k`.
Each day's fresh signal is held for `hold` days (positions are the average of the last `hold`
daily signals, i.e. 1/hold of the book is re-ranked every day).

Backtest (daily closes, cost 1bp/unit turnover, lookback 21, hold 5):
  2009-2026  Sharpe ~0.42 (t~1.8), ann ~3.5% @ ~8% vol, maxDD ~-18%
  2009-2020  Sharpe ~0.24 (weak) | 2021-2026 Sharpe ~1.0 (strong; may be a temporary regime)
  => a small, unproven edge.  Treat as an EXPERIMENT: demo account only, for 3-6 months.

Usage
-----
  python fx_reversal_signal.py --source csv --csv sample_prices.csv      # offline test
  python fx_reversal_signal.py --source yahoo                            # needs: pip install yfinance
  TWELVEDATA_API_KEY=xxx python fx_reversal_signal.py --source twelvedata
  python fx_reversal_signal.py --source yahoo --log signals_log.csv      # also append to a log
  python fx_reversal_signal.py --source yahoo --report --log signals_log.csv   # paper P&L of the log
  python fx_reversal_signal.py --source csv --csv sample_prices.csv --backtest # reproduce the backtest

Run it once per day AFTER the FX daily close (17:00 New York), Monday-Friday, then place the
changes the table asks for.  Not financial advice.
"""
import argparse, os, sys, json, time, urllib.request, urllib.parse
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]
CCY = {"EURUSD": "EUR", "GBPUSD": "GBP", "AUDUSD": "AUD", "NZDUSD": "NZD",
       "USDJPY": "JPY", "USDCAD": "CAD", "USDCHF": "CHF"}
# currency return vs USD = sign * pair return   (USDxxx pairs are inverted)
SIGN = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1, "USDJPY": -1, "USDCAD": -1, "USDCHF": -1}
YAHOO = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
         "USDJPY": "JPY=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X"}
TD_SYMBOL = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "AUDUSD": "AUD/USD", "NZDUSD": "NZD/USD",
             "USDJPY": "USD/JPY", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF"}
COST_BP = 1.0   # assumed cost per unit of turnover (about a 1-pip round turn)


# ------------------------------------------------------------------ data
def load_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = [p for p in PAIRS if p not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing} (need {PAIRS})")
    return df[PAIRS]


def load_yahoo(days=400):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed:  pip install yfinance   (or use --source csv / twelvedata)")
    t = [YAHOO[p] for p in PAIRS]
    raw = yf.download(t, period=f"{days}d", interval="1d", auto_adjust=False, progress=False)["Close"]
    raw = raw.rename(columns={v: k for k, v in YAHOO.items()})
    return raw[PAIRS]


def load_twelvedata(outputsize=400):
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        sys.exit("Set TWELVEDATA_API_KEY in the environment (free plan allows 8 requests/min).")
    out = {}
    for p in PAIRS:
        q = urllib.parse.urlencode({"symbol": TD_SYMBOL[p], "interval": "1day", "outputsize": outputsize, "apikey": key})
        with urllib.request.urlopen(f"https://api.twelvedata.com/time_series?{q}", timeout=30) as r:
            js = json.load(r)
        if "values" not in js:
            sys.exit(f"Twelve Data error for {p}: {js.get('message', js)}")
        s = pd.Series({pd.Timestamp(v["datetime"]): float(v["close"]) for v in js["values"]}).sort_index()
        out[p] = s
        time.sleep(0.5)
    return pd.DataFrame(out)


def clean(px, asof=None, drop_last=False):
    px = px.copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    px = px[px.index.dayofweek < 5].dropna()           # no weekend bars, no gaps across pairs
    if asof:
        px = px[px.index <= pd.Timestamp(asof)]
    if drop_last:
        px = px.iloc[:-1]
    if (px <= 0).any().any():
        sys.exit("Non-positive prices found - check the data source.")
    return px


# ------------------------------------------------------------------ strategy
def currency_returns(px):
    r = np.log(px).diff()
    return pd.DataFrame({CCY[p]: SIGN[p] * r[p] for p in PAIRS}).iloc[1:]


def fresh_weights(px, lookback, k):
    """Weights of the signal formed at each close: +1/k on the weakest k currencies, -1/k on the strongest k."""
    lp = np.log(px)
    score = pd.DataFrame({CCY[p]: SIGN[p] * (lp[p] - lp[p].shift(lookback)) for p in PAIRS})
    rk = score.rank(axis=1)
    n = score.shape[1]
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    w[rk <= k] = 1.0 / k
    w[rk > n - k] = -1.0 / k
    return w[score.notna().all(axis=1)], score


def target_weights(px, lookback=21, hold=5, k=2):
    """Blended target = average of the last `hold` daily signals (what the backtest holds)."""
    fw, score = fresh_weights(px, lookback, k)
    tw = fw.reindex(px.index).rolling(hold, min_periods=hold).mean()
    return tw, fw.reindex(px.index), score


def currency_to_pair(w_ccy):
    """Currency weight w (long the currency) -> pair position: p = SIGN * w."""
    return pd.Series({p: SIGN[p] * w_ccy[CCY[p]] for p in PAIRS})


def run_returns(px, tw, cost_bp=COST_BP):
    cr = currency_returns(px)
    wl = tw.shift(1).reindex(cr.index)                  # signal at close t earns return of t+1
    gross = (wl * cr).sum(axis=1, min_count=1)
    turn = wl.diff().abs().sum(axis=1)
    return (gross - turn * cost_bp / 1e4).dropna()


def stats(r):
    eq = (1 + r).cumprod()
    return dict(days=len(r), sharpe=float(r.mean() / r.std() * np.sqrt(252)), t=float(r.mean() / r.std() * np.sqrt(len(r))),
                ann_ret_pct=float(r.mean() * 252 * 100), vol_pct=float(r.std() * np.sqrt(252) * 100),
                max_dd_pct=float((eq / eq.cummax() - 1).min() * 100))


def fmt_stats(d):
    return (f"days={d['days']}  Sharpe={d['sharpe']:.2f}  t={d['t']:.2f}  ann={d['ann_ret_pct']:.2f}%  "
            f"vol={d['vol_pct']:.1f}%  maxDD={d['max_dd_pct']:.1f}%")


# ------------------------------------------------------------------ commands
def cmd_signal(px, a):
    tw, fw, score = target_weights(px, a.lookback, a.hold, a.k)
    d = px.index[-1]
    if pd.isna(tw.loc[d]).any():
        sys.exit(f"Not enough history: need at least {a.lookback + a.hold} daily rows, have {len(px)}.")
    today = datetime.now(timezone.utc)
    age = (pd.Timestamp(today.date()) - d).days
    print(f"\nLast close in data: {d.date()}   (lookback={a.lookback}d, k={a.k}, hold={a.hold}d)")
    if age > 4:
        print(f"  !! WARNING: data is {age} days old - refresh the data source before trading.")
    if age == 0 and today.hour < 22:
        print("  !! NOTE: today's bar may be incomplete (FX daily close is 17:00 New York). "
              "Re-run after the close, or use --drop-last.")
    cur = pd.DataFrame({
        f"{a.lookback}d ret vs USD %": (np.exp(score.loc[d]) - 1) * 100,
        "rank (1=weakest)": score.loc[d].rank().astype(int),
        "TODAY's fresh signal": fw.loc[d].map(lambda x: "BUY (weak)" if x > 0 else ("SELL (strong)" if x < 0 else "-")),
        "target weight %": tw.loc[d] * 100,
    }).sort_values("rank (1=weakest)")
    pd.set_option("display.width", 140)
    print("\nCurrency ranking (performance vs USD):")
    print(cur.round(2).to_string())
    pp = currency_to_pair(tw.loc[d])
    last = px.iloc[-1]
    print(f"\nTarget PAIR positions for equity = {a.equity:,.0f} USD   (weights are % of equity notional; gross up to 200%)")
    rows = []
    for p in PAIRS:
        notional = pp[p] * a.equity
        base_usd = last[p] if p.endswith("USD") else 1.0      # USD value of 1 unit of the pair's base currency
        lots = notional / (100000 * base_usd)
        side = "BUY" if pp[p] > 1e-9 else ("SELL" if pp[p] < -1e-9 else "flat")
        rows.append((p, side, pp[p] * 100 + 0.0, notional + 0.0, lots + 0.0))
    print(pd.DataFrame(rows, columns=["pair", "side", "weight %", "notional USD", "lots (100k)"]).round(2).to_string(index=False))
    if len(tw.dropna()) > 1:
        prev = tw.dropna().iloc[-2]
        ch = (tw.loc[d] - prev) * 100
        print("\nCHANGE vs yesterday's target (currency weight %, + = buy the currency):")
        print(ch.round(1).to_string())
    if a.log:
        row = pd.DataFrame([tw.loc[d]], index=[d]); row.index.name = "date"
        if os.path.exists(a.log):
            old = pd.read_csv(a.log, index_col=0, parse_dates=True); row = pd.concat([old[old.index != d], row]).sort_index()
        row.to_csv(a.log); print(f"\nLogged target weights for {d.date()} -> {a.log}")


def cmd_backtest(px, a):
    tw, _, _ = target_weights(px, a.lookback, a.hold, a.k)
    r = run_returns(px, tw)
    r = r[a.lookback + a.hold + 2:]
    print(f"\nBacktest {r.index[0].date()} -> {r.index[-1].date()}  (cost {COST_BP}bp/turnover, lookback={a.lookback}, hold={a.hold}, k={a.k})")
    for lab, s in [("all", r), ("2009-2020", r[:"2020"]), ("2021+", r["2021":])]:
        if len(s) > 100:
            print(f"  {lab:10s} {fmt_stats(stats(s))}")


def cmd_report(px, a):
    if not a.log or not os.path.exists(a.log):
        sys.exit("--report needs an existing --log file (run the signal for several days first).")
    lg = pd.read_csv(a.log, index_col=0, parse_dates=True).sort_index()
    cr = currency_returns(px)
    # the target logged at close d earns the return of the NEXT trading day
    rets = {}
    prev_w = None
    for d in lg.index:
        nxt = cr.index[cr.index > d]
        if len(nxt) == 0:
            continue
        w = lg.loc[d]
        turn = (w - prev_w).abs().sum() if prev_w is not None else w.abs().sum()
        rets[nxt[0]] = float((w * cr.loc[nxt[0]]).sum() - turn * COST_BP / 1e4)
        prev_w = w
    if not rets:
        sys.exit("No completed days to evaluate yet (the log needs at least one day older than the latest price).")
    r = pd.Series(rets).sort_index()
    print(f"\nPaper P&L from log ({len(r)} days, {r.index[0].date()} -> {r.index[-1].date()}):")
    print(f"  total return {((1 + r).prod() - 1) * 100:.2f}%  |  best day {r.max() * 100:.2f}%  |  worst day {r.min() * 100:.2f}%")
    if len(r) >= 20:
        print("  ", fmt_stats(stats(r)))
    else:
        print("  (fewer than 20 days: statistics would be meaningless - keep logging)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["csv", "yahoo", "twelvedata"], default="csv")
    ap.add_argument("--csv", default="sample_prices.csv")
    ap.add_argument("--lookback", type=int, default=21)
    ap.add_argument("--hold", type=int, default=5, help="days each signal is held (3 or 5 were tested)")
    ap.add_argument("--k", type=int, default=2, help="how many currencies on each side")
    ap.add_argument("--equity", type=float, default=10000.0, help="account equity in USD (for sizing)")
    ap.add_argument("--asof", help="use data up to this date (YYYY-MM-DD)")
    ap.add_argument("--drop-last", action="store_true", help="ignore the most recent (maybe incomplete) bar")
    ap.add_argument("--log", help="CSV file to append the target weights to")
    ap.add_argument("--report", action="store_true", help="paper P&L from the log")
    ap.add_argument("--backtest", action="store_true", help="re-run the historical backtest on the data")
    a = ap.parse_args()
    raw = {"csv": lambda: load_csv(a.csv), "yahoo": load_yahoo, "twelvedata": load_twelvedata}[a.source]()
    px = clean(raw, a.asof, a.drop_last)
    if len(px) < a.lookback + a.hold + 5:
        sys.exit(f"Only {len(px)} usable daily rows; need more history.")
    if a.backtest:
        cmd_backtest(px, a)
    elif a.report:
        cmd_report(px, a)
    else:
        cmd_signal(px, a)


if __name__ == "__main__":
    main()
