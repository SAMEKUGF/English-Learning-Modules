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
import argparse, io, os, sys, json, time, urllib.request, urllib.parse
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


def load_avfiles(map_path):
    """Read Alpha Vantage / Twelve Data daily results saved to files.  map_path = JSON {"EURUSD": "/path/to/file", ...}.
    Each file is the tool's saved JSON ({"sample_data": ...} for Alpha Vantage, {"result": ...} for Twelve Data) or plain CSV text."""
    m = json.load(open(map_path))
    out = {}
    for p in PAIRS:
        if p not in m:
            sys.exit(f"{map_path}: missing pair {p}")
        txt = open(m[p]).read()
        try:
            js = json.loads(txt)
            txt = js["sample_data"] if isinstance(js, dict) and "sample_data" in js else js["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        lines = [l for l in txt.replace("\r", "").split("\n") if l.strip()]
        header = lines[0].lower()
        if "timestamp" not in header and "date" not in header:
            sys.exit(f"{m[p]}: unexpected header '{lines[0]}'")
        sep = ";" if ";" in lines[0] else ","                                  # Twelve Data uses ';', Alpha Vantage ','
        body = [l for l in lines[1:] if l.count(sep) == lines[0].count(sep)]    # drop a possibly truncated last line
        df = pd.read_csv(io.StringIO("\n".join([lines[0]] + body)), sep=sep)
        df = df.rename(columns={df.columns[0]: "ts"})
        out[p] = pd.Series(df["close"].values, index=pd.to_datetime(df["ts"])).sort_index()
    return pd.DataFrame(out)


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


def target_weights(px, lookback=21, hold=5, k=2, start=None):
    """Blended target = average of the last `hold` daily signals (what the backtest holds).
    start: first day you trade.  Signals before it count as zero, so the book ramps up by 1/hold per day."""
    fw, score = fresh_weights(px, lookback, k)
    fwr = fw.reindex(px.index)
    if start is not None:
        fwr.loc[fwr.index < pd.Timestamp(start)] = 0.0
    tw = fwr.rolling(hold, min_periods=hold).mean()
    return tw, fwr, score


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
    tw, fw, score = target_weights(px, a.lookback, a.hold, a.k, a.start)
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
        lots = lots_for(pp[p], a.equity, last[p], p)
        side = "BUY" if pp[p] > 1e-9 else ("SELL" if pp[p] < -1e-9 else "flat")
        rows.append((p, side, pp[p] * 100 + 0.0, notional + 0.0, lots + 0.0))
    print(pd.DataFrame(rows, columns=["pair", "side", "weight %", "notional USD", "lots (100k)"]).round(3).to_string(index=False))
    if len(tw.dropna()) > 1:
        prev = tw.dropna().iloc[-2]
        ch = (tw.loc[d] - prev) * 100
        print("\nCHANGE vs yesterday's target (currency weight %, + = buy the currency):")
        print(ch.round(1).to_string())
    if a.email_json:
        em = build_email(px, tw, fw, score, a)
        json.dump(em, open(a.email_json, "w"), ensure_ascii=False, indent=1)
        print(f"\nEmail written -> {a.email_json}  (subject: {em['subject']})")
    if a.log:
        row = pd.DataFrame([tw.loc[d]], index=[d]); row.index.name = "date"
        if os.path.exists(a.log):
            old = pd.read_csv(a.log, index_col=0, parse_dates=True); row = pd.concat([old[old.index != d], row]).sort_index()
        row.to_csv(a.log); print(f"\nLogged target weights for {d.date()} -> {a.log}")


AR = {"BUY": "اشترِ", "SELL": "بِع"}


LOT_STEP = 0.001      # set by --lot-step (0.01 = standard lots, 0.001 = micro/cent accounts)


def lots_for(weight, equity, price, pair):
    base_usd = price if pair.endswith("USD") else 1.0        # USD value of one unit of the base currency
    raw = weight * equity / (100000.0 * base_usd)
    return round(round(raw / LOT_STEP) * LOT_STEP + 0.0, 4)


def fl(x):
    return f"{abs(x):.3f}" if LOT_STEP < 0.01 else f"{abs(x):.2f}"


def units(x):
    return f"{abs(x) * 100000:,.0f}"


def build_email(px, tw, fw, score, a):
    d = px.index[-1]
    idx = tw.dropna().index
    prev = idx[idx < d][-1] if (idx < d).any() else None
    pt = currency_to_pair(tw.loc[d])
    py = currency_to_pair(tw.loc[prev]) if prev is not None else pt * 0.0
    last = px.iloc[-1]
    orders, targets = [], []
    for p in PAIRS:
        lt = lots_for(pt[p], a.equity, last[p], p)
        ly = lots_for(py[p], a.equity, last[p], p)         # yesterday's target, sized at today's price (no price-drift noise)
        diff = round(lt - ly + 0.0, 4)
        nm = f"{p[:3]}/{p[3:]}"
        side = "BUY" if diff > 0 else ("SELL" if diff < 0 else None)
        pos = (f"شراء {fl(lt)} لوت (≈ {units(lt)} وحدة)" if lt > 0 else (f"بيع {fl(lt)} لوت (≈ {units(lt)} وحدة)" if lt < 0 else "لا مركز"))
        targets.append(dict(pair=nm, target_lots=lt, text=pos, weight_pct=round(pt[p] * 100, 1)))
        if side:
            orders.append(dict(pair=nm, side=side, lots=abs(diff), text=f"{AR[side]} {fl(diff)} لوت {nm} (≈ {units(diff)} وحدة)", before=ly, after=lt))
    small = [f"{p[:3]}/{p[3:]}" for p in PAIRS if abs(pt[p]) > 1e-9 and lots_for(pt[p], a.equity, last[p], p) == 0.0]
    stale = bool(a.expect_date) and str(d.date()) != a.expect_date
    first_day = prev is None or (a.start is not None and str(d.date()) == a.start)
    rank = pd.DataFrame({"r": (np.exp(score.loc[d]) - 1) * 100}).sort_values("r")
    rank_lines = [f"{c}: {v:+.2f}%" for c, v in rank["r"].items()]
    subj = f"[FX] إشارة {d.date()} بعد إغلاق نيويورك — " + (f"{len(orders)} أوامر" if orders else "لا أوامر اليوم")
    if stale:
        subj = "[بيانات قديمة] " + subj
    L = []
    if stale:
        L.append(f"⚠️ تحذير: آخر إغلاق في البيانات هو {d.date()} وليس {a.expect_date}. لا تنفّذ قبل التأكد من تحديث الأسعار.")
    L.append(f"إغلاق نيويورك: {d.date()}  |  رصيد الحساب المفترض: {a.equity:,.0f}$  |  حساب تجريبي")
    if first_day:
        L.append("اليوم أول يوم تشغيل: المحفظة تبدأ فارغة وتُبنى تدريجياً بمعدل خُمس الإشارة كل يوم.")
    L.append("")
    L.append("ماذا تفعل الآن (نفّذ كل أمر كأمر سوقي):")
    if orders:
        for i, o in enumerate(orders, 1):
            dirn = lambda x: ("شراء" if x > 0 else ("بيع" if x < 0 else "لا مركز"))
            L.append(f"  {i}) {o['text']}   (المركز بعد التنفيذ: {dirn(o['after'])} {fl(o['after'])})")
    else:
        L.append("  لا توجد أوامر اليوم. لا تغيّر شيئاً.")
    L.append("")
    L.append("المراكز المطلوب أن تكون مفتوحة بعد التنفيذ:")
    for t in targets:
        L.append(f"  - {t['pair']}: {t['text']}")
    if small:
        L.append(f"  ملاحظة: الأحجام الأقل من {LOT_STEP} لوت تُقرَّب إلى صفر ({', '.join(small)}). رصيد أكبر يعطي دقة أفضل.")
    L.append("")
    L.append("ترتيب العملات (أداء 21 يوماً مقابل الدولار، من الأضعف إلى الأقوى):")
    L.append("  " + "  |  ".join(rank_lines))
    L.append("")
    L.append("تذكير: لا يوجد وقف خسارة ولا هدف. الخروج يتم تدريجياً بتعديلات الأيام القادمة. إذا اختلف مركزك الفعلي عن الجدول أعلاه، عدّله ليطابقه.")
    L.append("استراتيجية تجريبية غير مثبتة. لا تستخدم مالاً حقيقياً قبل 3–6 أشهر من المتابعة.")
    text = "\n".join(L)
    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag} style='padding:6px 10px;border:1px solid #ccc;text-align:right'>{c}</{tag}>" for c in cells) + "</tr>"
    html = ["<div dir='rtl' style='font-family:Arial,Tahoma,sans-serif;font-size:15px;line-height:1.6;text-align:right'>"]
    if stale:
        html.append(f"<p style='color:#b00020'><b>⚠️ آخر إغلاق في البيانات هو {d.date()} وليس {a.expect_date}. لا تنفّذ قبل التأكد من تحديث الأسعار.</b></p>")
    html.append(f"<p>إغلاق نيويورك: <b>{d.date()}</b> · رصيد الحساب المفترض: <b>{a.equity:,.0f}$</b> · حساب تجريبي</p>")
    if first_day:
        html.append("<p><b>اليوم أول يوم تشغيل:</b> المحفظة تبدأ فارغة وتُبنى تدريجياً بمعدل خُمس الإشارة كل يوم.</p>")
    html.append("<h3>ماذا تفعل الآن (أوامر سوقية)</h3>")
    if orders:
        rows = [tr(["#", "الأمر", "المركز بعد التنفيذ"], "th")]
        for i, o in enumerate(orders, 1):
            aft = "شراء" if o["after"] > 0 else ("بيع" if o["after"] < 0 else "لا مركز")
            rows.append(tr([i, f"<b>{o['text']}</b>", f"{aft} {fl(o['after'])}" if o["after"] else "لا مركز"]))
        html.append("<table style='border-collapse:collapse'>" + "".join(rows) + "</table>")
    else:
        html.append("<p><b>لا توجد أوامر اليوم. لا تغيّر شيئاً.</b></p>")
    html.append("<h3>المراكز المطلوب أن تكون مفتوحة بعد التنفيذ</h3>")
    html.append("<table style='border-collapse:collapse'>" + tr(["الزوج", "المركز"], "th") + "".join(tr([t["pair"], t["text"]]) for t in targets) + "</table>")
    if small:
        html.append(f"<p style='color:#666'>الأحجام الأقل من {LOT_STEP} لوت تُقرَّب إلى صفر ({', '.join(small)}).</p>")
    html.append("<h3>ترتيب العملات (21 يوماً مقابل الدولار، من الأضعف إلى الأقوى)</h3><p>" + " &nbsp;|&nbsp; ".join(rank_lines) + "</p>")
    html.append("<p style='color:#555'>لا يوجد وقف خسارة ولا هدف؛ الخروج بتعديلات الأيام القادمة. إذا اختلف مركزك الفعلي عن الجدول فعدّله ليطابقه.<br>استراتيجية تجريبية غير مثبتة: لا مال حقيقي قبل 3–6 أشهر من المتابعة.</p></div>")
    return dict(subject=subj, body_text=text, body_html="".join(html), orders=orders, targets=targets, stale=stale, last_date=str(d.date()))


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
    ap.add_argument("--source", choices=["csv", "yahoo", "twelvedata", "avfiles"], default="csv")
    ap.add_argument("--av-map", help="JSON map {pair: file} of saved Alpha Vantage FX_DAILY results (--source avfiles)")
    ap.add_argument("--start", help="first trading day (YYYY-MM-DD): the book ramps up from empty")
    ap.add_argument("--expect-date", help="NY date of the close you expect to be the last bar; flags stale data")
    ap.add_argument("--email-json", help="write the ready-to-send email (subject/text/html/orders) to this JSON file")
    ap.add_argument("--csv", default="sample_prices.csv")
    ap.add_argument("--lookback", type=int, default=21)
    ap.add_argument("--hold", type=int, default=5, help="days each signal is held (3 or 5 were tested)")
    ap.add_argument("--k", type=int, default=2, help="how many currencies on each side")
    ap.add_argument("--equity", type=float, default=10000.0, help="account equity in USD (for sizing)")
    ap.add_argument("--lot-step", type=float, default=0.001, help="lot rounding step: 0.01 standard, 0.001 micro/cent")
    ap.add_argument("--asof", help="use data up to this date (YYYY-MM-DD)")
    ap.add_argument("--drop-last", action="store_true", help="ignore the most recent (maybe incomplete) bar")
    ap.add_argument("--log", help="CSV file to append the target weights to")
    ap.add_argument("--report", action="store_true", help="paper P&L from the log")
    ap.add_argument("--backtest", action="store_true", help="re-run the historical backtest on the data")
    a = ap.parse_args()
    global LOT_STEP
    LOT_STEP = a.lot_step
    raw = {"csv": lambda: load_csv(a.csv), "yahoo": load_yahoo, "twelvedata": load_twelvedata, "avfiles": lambda: load_avfiles(a.av_map)}[a.source]()
    px = clean(raw, a.asof, a.drop_last)
    if a.expect_date:      # providers open the NEXT day's bar right after the 17:00 NY close: ignore it
        px = px[px.index <= pd.Timestamp(a.expect_date)]
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
