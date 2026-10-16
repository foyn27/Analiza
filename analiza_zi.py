"""
ANALIZA UNEI ACTIUNI INTR-O ZI  (NU trimite ordine)

Descarca de la Alpaca tot ce trebuie pentru o analiza completa:
  - lumanari de 1 minut pentru ziua aleasa (premarket 4:00 - after-hours 20:00, ora NY)
  - lumanari zilnice pe ultimele ~90 de zile (cum arata actiunea inainte)
  - spread-ul bid/ask la fiecare 15 minute (cat costa de fapt intrarea si iesirea)
  - stirile din ultimele 3 zile
  - informatii despre actiune (bursa, daca se poate tranzactiona / vinde in lipsa)
si calculeaza: VWAP, EMA 9/20, RSI 14, ATR%, Efficiency Ratio, volum relativ,
maximul din premarket, saltul fata de ieri, semnalele "lumanare mare verde + volum"
pe 5 si 15 minute si ce ar fi iesit din fiecare (cu cost 1% pe parte).

Rulare:  python analiza_zi.py                    (te intreaba simbolul si data)
         python analiza_zi.py SDST               (azi)
         python analiza_zi.py SDST 2026-10-15
Rezultat (in ~/alpaca_auto_istoric/analiza_zi/):
         zi_SDST_2026-10-15.html     raportul cu grafic
         zi_SDST_2026-10-15_1min.csv datele minut cu minut, cu indicatori
"""

import importlib
import json
import math
import sys
import time
import traceback
import webbrowser
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path.home() / "alpaca_auto_istoric" / "analiza_zi"
NY = ZoneInfo("America/New_York")
RO = ZoneInfo("Europe/Bucharest")
COST = 0.01                    # cost estimat pe fiecare parte (spread + alunecare)
SIG_MOVE, SIG_VOL = 0.10, 1.0  # lumanare >= +10%, volum peste media ultimelor 9


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ==========================================================
# DATE
# ==========================================================

def load_bot():
    """Clienti Alpaca proprii (nu depinde de alti boti). Cheile: din fereastra oricarui
    bot (Trading 212, Alpaca, RSI2, crypto - salvate criptat) sau din variabilele de mediu."""
    import os
    from types import SimpleNamespace
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    key = secret = None
    for mod, kid, sec in (("gui", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
                          ("gui_alpaca", "APCA_AUTO_KEY_ID", "APCA_AUTO_SECRET_KEY"),
                          ("gui_rsi2", "APCA_RSI2_KEY_ID", "APCA_RSI2_SECRET_KEY"),
                          ("gui_crypto", "APCA_CRYPTO_KEY_ID", "APCA_CRYPTO_SECRET_KEY"),
                          ("gui_crypto_semi", "APCA_CRYPTO_KEY_ID", "APCA_CRYPTO_SECRET_KEY"),
                          (None, "APCA_API_KEY_ID", "APCA_API_SECRET_KEY")):
        if mod:
            try:
                importlib.import_module(mod).load_saved_keys()
            except Exception:
                pass
        if os.getenv(kid) and os.getenv(sec):
            key = os.getenv(kid).strip().strip('"').strip("'")
            secret = os.getenv(sec).strip().strip('"').strip("'")
            break
    for name, val in (("APCA_API_KEY_ID (Key ID)", key), ("APCA_API_SECRET_KEY (Secret)", secret)):
        bad = [(i + 1, ch) for i, ch in enumerate(val or "") if not (ch.isascii() and ch.isprintable())]
        if bad:
            pos, ch = bad[0]
            raise RuntimeError(
                f"Cheia {name} contine un caracter nepermis pe pozitia {pos}: {ch!r} "
                f"(cod U+{ord(ch):04X}). Cheile Alpaca au doar litere latine si cifre. "
                "Probabil tastatura a pus o litera chirilica (ex. Х in loc de X) sau un "
                "spatiu special. Sterge secretul si lipeste-l din nou, copiat direct din Alpaca.")
    if not key:
        raise RuntimeError("Nu gasesc cheile Alpaca. Pune analiza_zi.py in folderul unui bot "
                           "(ex. botul Trading 212) si deschide o data fereastra lui ca sa "
                           "salveze cheile, sau seteaza APCA_API_KEY_ID si APCA_API_SECRET_KEY.")
    return SimpleNamespace(
        market_data=StockHistoricalDataClient(key, secret),
        trading=TradingClient(key, secret, paper=True),
        StockBarsRequest=StockBarsRequest, TimeFrame=TimeFrame,
        APCA_KEY=key, APCA_SECRET=secret)


def _retry(fn, what):
    for attempt in range(3):
        try:
            return fn()
        except Exception as exc:
            txt = str(exc)
            if "401" in txt or "403" in txt or "Authorization Required" in txt \
                    or "forbidden" in txt.lower() or "unauthorized" in txt.lower():
                raise RuntimeError(
                    "Alpaca a respins cheile (401/403). Verifica APCA_API_KEY_ID si "
                    "APCA_API_SECRET_KEY: sa fie cheile contului Paper, copiate fara spatii "
                    "sau ghilimele, Key ID si Secret sa nu fie inversate, si sa nu fi "
                    "regenerat cheile intre timp (cheile vechi nu mai merg).") from None
            say(f"  {what}: {str(exc)[:100]} (incercarea {attempt + 1}/3)")
            time.sleep(3 * (attempt + 1))
    return None


def sip_end(t):
    """Contul gratuit nu poate cere date SIP din ultimele 15 minute."""
    lim = datetime.now(timezone.utc) - timedelta(minutes=16)
    return min(t, lim)


def fetch_minutes(bot, sym, d):
    from alpaca.data.enums import Adjustment, DataFeed
    start = datetime(d.year, d.month, d.day, 4, 0, tzinfo=NY)
    end = datetime(d.year, d.month, d.day, 20, 0, tzinfo=NY)
    feeds = [(DataFeed.SIP, sip_end(end)), (DataFeed.IEX, end)]

    def get(feed, e):
        if e <= start:
            return pd.DataFrame()
        res = _retry(lambda: bot.market_data.get_stock_bars(bot.StockBarsRequest(
            symbol_or_symbols=sym, timeframe=bot.TimeFrame.Minute, start=start, end=e,
            adjustment=Adjustment.RAW, feed=feed)).df, f"minute ({feed.value})")
        if res is None or res.empty:
            return pd.DataFrame()
        df = res.xs(sym, level="symbol") if isinstance(res.index, pd.MultiIndex) else res
        df.index = pd.to_datetime(df.index).tz_convert(NY)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    GAPS.clear()
    for feed, e in feeds:
        df = get(feed, e)
        if df.empty:
            continue
        if feed == DataFeed.SIP:
            df = fill_gaps(df, lambda: get(DataFeed.IEX, end))
        return df, feed.value
    return pd.DataFrame(), ""


GAPS = []   # goluri gasite in datele SIP: (de la, pana la, umplut din IEX?)


def fill_gaps(df, get_iex, min_gap=10):
    """Uneori datele SIP de la Alpaca au gauri (zeci de minute lipsa in sesiunea
    normala, desi actiunea se tranzactiona). Le umplem cu lumanarile IEX din acel
    interval (volumul IEX e doar o parte din volumul real) si le semnalam."""
    reg = df.between_time("09:30", "15:59")
    if len(reg) < 2:
        return df
    t = reg.index.to_series()
    gaps = [(a, b) for a, b in zip(t.iloc[:-1], t.iloc[1:])
            if (b - a).total_seconds() / 60 > min_gap]
    if not gaps:
        return df
    iex = get_iex()
    extra = []
    for a, b in gaps:
        part = iex[(iex.index > a) & (iex.index < b)] if not iex.empty else iex
        GAPS.append((a, b, not part.empty))
        if not part.empty:
            extra.append(part)
    if extra:
        df = pd.concat([df] + extra).sort_index()
        df = df[~df.index.duplicated(keep="first")]
    for a, b, ok in GAPS:
        say(f"  ATENTIE: lipsesc date SIP intre {a:%H:%M} si {b:%H:%M} NY"
            + (" - completate din IEX." if ok else " - nici IEX nu are date."))
    return df


def fetch_daily(bot, sym, d):
    from alpaca.data.enums import Adjustment, DataFeed
    start = datetime(d.year, d.month, d.day, tzinfo=NY) - timedelta(days=130)
    end = sip_end(datetime(d.year, d.month, d.day, 23, 59, tzinfo=NY))
    res = _retry(lambda: bot.market_data.get_stock_bars(bot.StockBarsRequest(
        symbol_or_symbols=sym, timeframe=bot.TimeFrame.Day, start=start, end=end,
        adjustment=Adjustment.RAW, feed=DataFeed.SIP)).df, "zilnic")
    if res is None or res.empty:
        return pd.DataFrame()
    df = res.xs(sym, level="symbol") if isinstance(res.index, pd.MultiIndex) else res
    df.index = pd.to_datetime(df.index).tz_convert(NY).date
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def fetch_spreads(bot, sym, d, minutes):
    """Primul bid/ask dupa fiecare sfert de ora in care s-a tranzactionat."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    rows = []
    if minutes.empty:
        return pd.DataFrame()
    slots = sorted(set(minutes.index.floor("15min")))
    for t in slots:
        e = sip_end(t + timedelta(minutes=15))
        if e <= t:
            continue
        res = _retry(lambda: bot.market_data.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=sym, start=t, end=e, limit=20, feed=DataFeed.SIP)).df,
            f"spread {t:%H:%M}")
        if res is None or res.empty:
            continue
        q = res.reset_index()
        q = q[(q["bid_price"] > 0) & (q["ask_price"] > q["bid_price"])]
        if q.empty:
            continue
        mid = (q["bid_price"] + q["ask_price"]) / 2
        sp = ((q["ask_price"] - q["bid_price"]) / mid * 100).median()
        rows.append({"ora_ny": t, "bid": q["bid_price"].iloc[0], "ask": q["ask_price"].iloc[0],
                     "spread_pct": sp})
        time.sleep(0.25)
    return pd.DataFrame(rows)


def fetch_news(bot, sym, d):
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
    except Exception:
        return []
    key = getattr(bot, "APCA_KEY", None)
    sec = getattr(bot, "APCA_SECRET", None)
    if not key:
        return []
    cl = NewsClient(key, sec)
    start = datetime(d.year, d.month, d.day, tzinfo=NY) - timedelta(days=3)
    end = min(datetime(d.year, d.month, d.day, 23, 59, tzinfo=NY), datetime.now(timezone.utc))
    res = _retry(lambda: cl.get_news(NewsRequest(symbols=sym, start=start, end=end,
                                                  limit=50, sort="asc")), "stiri")
    out = []
    if res is None:
        return out
    items = res.data.get("news", []) if hasattr(res, "data") else []
    for n in items:
        out.append({"ora": pd.Timestamp(n.created_at).tz_convert(NY), "titlu": n.headline,
                    "sursa": n.source, "rezumat": n.summary or "", "url": n.url or "",
                    "simboluri": ", ".join(n.symbols or [])})
    return out


def fetch_asset(bot, sym):
    try:
        a = bot.trading.get_asset(sym)
        g = lambda x: str(getattr(x, "value", x)) if x is not None else "—"
        return {"nume": a.name, "bursa": g(a.exchange), "status": g(a.status),
                "tranzactionabila": a.tradable, "vanzare_in_lipsa": a.shortable,
                "usor_de_imprumutat": a.easy_to_borrow}
    except Exception as exc:
        return {"eroare": str(exc)[:100]}


# ==========================================================
# INDICATORI
# ==========================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df):
    """Indicatori pe orice interval (1, 5, 15 minute)."""
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cv = df["volume"].cumsum()
    df["vwap"] = (tp * df["volume"]).cumsum() / cv.replace(0, np.nan)
    df["fata_de_vwap_pct"] = (df["close"] / df["vwap"] - 1) * 100
    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    df["atr_pct14"] = tr.rolling(14, min_periods=5).mean() / df["close"] * 100
    path = df["close"].diff().abs().rolling(14, min_periods=5).sum()
    df["er14"] = (df["close"] - df["close"].shift(14)).abs() / path.replace(0, np.nan)
    df["schimbare_pct"] = (df["close"] / df["open"] - 1) * 100
    return df


def resample(m, minutes):
    """Lumanari de N minute din cele de 1 minut. Intervalele fara tranzactii au volum 0
    (conteaza la media volumului), dar nu apar ca lumanari."""
    r = m.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    r["vol_sma9"] = r["volume"].rolling(9).mean().shift(1)     # ultimele 9, fara cea curenta
    r = r.dropna(subset=["open"])
    r = add_indicators(r)
    r["volum_fata_de_media9"] = r["volume"] / r["vol_sma9"].replace(0, np.nan)
    r["semnal"] = (r["close"] >= r["open"] * (1 + SIG_MOVE)) & (r["volum_fata_de_media9"] > SIG_VOL)
    return r


def outcome(m, t_entry, minutes):
    """Intrare la primul minut de la t_entry; iesire dupa `minutes` minute."""
    after = m[m.index >= t_entry]
    if after.empty:
        return None
    px = after["open"].iloc[0]
    t0 = after.index[0]
    win = after[after.index < t0 + timedelta(minutes=minutes)]
    if win.empty or not px > 0:
        return None
    g = win["close"].iloc[-1] / px - 1
    return {"intrare": px, "ora_intrare": t0,
            "brut_pct": g * 100,
            "net_pct": ((1 + g) * (1 - COST) / (1 + COST) - 1) * 100,
            "max_pct": (win["high"].max() / px - 1) * 100,
            "min_pct": (win["low"].min() / px - 1) * 100}


def session_name(t):
    hm = t.hour * 60 + t.minute
    return "premarket" if hm < 570 else "normal" if hm < 960 else "after-hours"


# ==========================================================
# RAPORT
# ==========================================================

CSS = """
:root{--bg:#f3f4f7;--card:#fff;--ink:#1d2433;--dim:#6b7280;--line:#e3e6ec;--pos:#15803d;
--neg:#c62828;--warnbg:#fff5d6;--head:#14213d}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e6e9ef;--dim:#9aa3b2;
--line:#2c3340;--pos:#4ade80;--neg:#f87171;--warnbg:#3a3320;--head:#0b1220}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:var(--head);color:#fff;padding:18px 28px}header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;opacity:.8;font-size:13px}main{max-width:1200px;margin:0 auto;padding:16px}
h2{font-size:17px;margin:26px 0 10px}.kpis{display:grid;grid-template-columns:repeat(auto-fill,
minmax(170px,1fr));gap:10px}.kpi{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 12px}.kpi b{display:block;font-size:19px}.kpi span{color:var(--dim);
font-size:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 16px}.warn{background:var(--warnbg);border-radius:10px;padding:10px 14px}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;
max-height:560px}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:var(--card);color:var(--dim);font-weight:600}
.n{text-align:right;font-variant-numeric:tabular-nums}.pos{color:var(--pos)}.neg{color:var(--neg)}
.dim{color:var(--dim)}.small{font-size:12px;color:var(--dim)}tr.sig{background:var(--warnbg)}
#chart{height:640px;background:var(--card);border:1px solid var(--line);border-radius:10px}
a{color:inherit}
"""


def fp(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if abs(x) < 1:
        return f"{x:.4f}"
    return f"{x:,.{min(d, 2)}f}"


def fpct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"


def cls(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return "pos" if x > 0 else "neg" if x < 0 else ""


def kpi(label, value, c=""):
    return f"<div class='kpi'><b class='{c}'>{value}</b><span>{escape(label)}</span></div>"


def hro(t):
    return f"{t.astimezone(RO):%H:%M} <span class='dim'>({t:%H:%M} NY)</span>"


def svg_chart(b, sigs, news):
    """Grafic cu lumanari in SVG (fara biblioteci externe, merge si offline)."""
    if b.empty:
        return ""
    W, H, L, Rm, T = 1160, 560, 64, 10, 12
    PH, VH, GAP = 380, 110, 24
    n = len(b)
    step = (W - L - Rm) / n
    lo = float(min(b["low"].min(), b["vwap"].min()))
    hi = float(max(b["high"].max(), b["vwap"].max()))
    pad = (hi - lo) * 0.04 or hi * 0.01 or 1
    lo, hi = lo - pad, hi + pad
    vmax = float(b["volume"].max()) or 1
    y = lambda p: T + (hi - p) / (hi - lo) * PH
    yv0 = T + PH + GAP + VH
    xs = lambda k: L + step * (k + 0.5)
    out = [f"<svg viewBox='0 0 {W} {H}' width='100%' style='display:block;font:11px system-ui'>"]
    times = list(b.index)
    # sesiuni extinse
    for k, t in enumerate(times):
        hm = t.hour * 60 + t.minute
        if hm < 570 or hm >= 960:
            out.append(f"<rect x='{L + step * k:.1f}' y='{T}' width='{step + 0.3:.1f}' "
                       f"height='{yv0 - T}' fill='currentColor' opacity='.05'/>")
    # grila pret
    for g in np.linspace(lo, hi, 6)[1:-1]:
        out.append(f"<line x1='{L}' x2='{W - Rm}' y1='{y(g):.1f}' y2='{y(g):.1f}' "
                   f"stroke='currentColor' opacity='.12'/>"
                   f"<text x='{L - 6}' y='{y(g) + 4:.1f}' text-anchor='end' fill='currentColor' "
                   f"opacity='.7'>{fp(float(g))}</text>")
    # ore
    for k, t in enumerate(times):
        tr = t.astimezone(RO)
        if tr.minute == 0 and (k == 0 or times[k - 1].astimezone(RO).hour != tr.hour):
            out.append(f"<line x1='{xs(k):.1f}' x2='{xs(k):.1f}' y1='{T}' y2='{yv0}' "
                       f"stroke='currentColor' opacity='.08'/>"
                       f"<text x='{xs(k):.1f}' y='{yv0 + 16}' text-anchor='middle' "
                       f"fill='currentColor' opacity='.7'>{tr:%H:%M}</text>")
    # stiri
    for nw in news:
        t = nw["ora"]
        k = int(np.searchsorted(np.array([x.value for x in times]), t.value))
        k = min(max(k, 0), n - 1)
        out.append(f"<line x1='{xs(k):.1f}' x2='{xs(k):.1f}' y1='{T}' y2='{yv0}' stroke='#64748b' "
                   f"stroke-dasharray='3 3'><title>{escape(t.astimezone(RO).strftime('%H:%M'))} "
                   f"{escape(nw['titlu'][:120])}</title></line>")
    # lumanari + volum
    bw = max(1.0, step * 0.65)
    for k, (t, r) in enumerate(b.iterrows()):
        up = r["close"] >= r["open"]
        col = "#16a34a" if up else "#dc2626"
        x = xs(k)
        y1, y2 = y(max(r["open"], r["close"])), y(min(r["open"], r["close"]))
        tip = (f"{t.astimezone(RO):%H:%M} RO ({t:%H:%M} NY)&#10;D {fp(r['open'])}  M {fp(r['high'])}"
               f"  m {fp(r['low'])}  I {fp(r['close'])}&#10;{r['schimbare_pct']:+.1f}%  "
               f"volum {r['volume']:,.0f}&#10;VWAP {fp(r['vwap'])}  RSI {r['rsi14']:.0f}")
        out.append(f"<g><title>{tip}</title>"
                   f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{y(r['high']):.1f}' y2='{y(r['low']):.1f}' "
                   f"stroke='{col}'/><rect x='{x - bw / 2:.1f}' y='{y1:.1f}' width='{bw:.1f}' "
                   f"height='{max(1.0, y2 - y1):.1f}' fill='{col}'/>"
                   f"<rect x='{x - bw / 2:.1f}' y='{yv0 - r['volume'] / vmax * VH:.1f}' "
                   f"width='{bw:.1f}' height='{r['volume'] / vmax * VH:.1f}' fill='{col}' "
                   f"opacity='.45'/></g>")
    # linii
    for colname, color, w in (("vwap", "#f59e0b", 1.8), ("ema9", "#3b82f6", 1.1),
                              ("ema20", "#a855f7", 1.1)):
        pts = " ".join(f"{xs(k):.1f},{y(v):.1f}" for k, v in enumerate(b[colname])
                       if np.isfinite(v))
        out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{w}'/>")
    # semnale (la sfarsitul lumanarii de 15 min)
    tv = np.array([x.value for x in times])
    for sg in sigs:
        t_end = sg["t"] + timedelta(minutes=15)
        k = min(int(np.searchsorted(tv, t_end.value)) - 1, n - 1)
        if k < 0:
            continue
        yy = y(float(b["low"].iloc[max(0, k - 2):k + 1].min())) + 6
        out.append(f"<path d='M{xs(k):.1f},{yy:.1f} l6,11 l-12,0 z' fill='#f59e0b' "
                   f"stroke='currentColor' stroke-width='.6'><title>semnal 15 min "
                   f"{sg['t'].astimezone(RO):%H:%M}</title></path>")
    out.append(f"<text x='{L}' y='{T + PH + GAP - 6}' fill='currentColor' opacity='.6'>volum</text>")
    out.append("</svg>")
    return "".join(out)


def build_report(ctx):
    sym, d = ctx["sym"], ctx["d"]
    m, daily, b5, b15 = ctx["m"], ctx["daily"], ctx["b5"], ctx["b15"]
    S = ctx["summary"]
    P = [f"<!doctype html><html lang='ro'><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{escape(sym)} {d}</title><style>{CSS}</style>"
         "<body>"
         f"<header><h1>{escape(sym)} · {d:%d.%m.%Y}</h1><p>"
         f"{escape(str(ctx['asset'].get('nume', '')))} · {escape(str(ctx['asset'].get('bursa', '')))}"
         f" · date {escape(ctx['feed'].upper())} · ore Romania, in paranteza ora New York · "
         f"generat {datetime.now(RO):%Y-%m-%d %H:%M}</p></header><main>"]
    for a, b, ok in ctx.get("gaps", []):
        P.append("<div class='warn'>Date lipsa la Alpaca (SIP) intre "
                 f"{a.astimezone(RO):%H:%M} si {b.astimezone(RO):%H:%M} "
                 f"({a:%H:%M}-{b:%H:%M} NY). "
                 + ("Intervalul a fost completat cu date IEX: preturile sunt bune, "
                    "volumul e mai mic decat cel real." if ok else
                    "Nici IEX nu are date; intervalul lipseste din grafic si tabele.")
                 + " Ruleaza din nou analiza mai tarziu (sau maine) pentru date complete.</div>")
    if ctx["feed"] == "iex":
        P.append("<div class='warn'>Am primit doar date IEX (o parte mica din volum). "
                 "Pentru zile mai vechi de 15 minute se folosesc datele complete (SIP).</div>")

    # --- rezumat
    P.append("<h2>Rezumat</h2><div class='kpis'>")
    for lab, val, c in S["kpis"]:
        P.append(kpi(lab, val, c))
    P.append("</div>")

    # --- grafic
    P.append("<h2>Grafic (5 minute)</h2><div class='card' style='padding:6px'>"
             + svg_chart(b5, ctx["sig15"], [n for n in ctx["news"] if n["ora"].date() == d])
             + "</div>")
    P.append("<div class='small'>Portocaliu = VWAP, albastru = EMA 9, mov = EMA 20. "
             "Triunghiurile = semnale „lumanare de 15 min ≥ +10% cu volum peste media 9”. "
             "Liniile verticale punctate = stiri. Tine mouse-ul pe o lumanare ca sa vezi valorile. "
             "Fundalul gri = premarket si after-hours.</div>")

    # --- semnale
    P.append(f"<h2>Semnale „lumanare mare verde + volum” si ce a urmat (cost {COST * 100:g}% pe parte)</h2>")
    for lab, rows in (("15 minute", ctx["sig15"]), ("5 minute", ctx["sig5"])):
        if not rows:
            P.append(f"<div class='card'>Pe {lab}: niciun semnal.</div>")
            continue
        P.append(f"<h3>Pe {lab}: {len(rows)} semnale</h3><div class='tbl'><table><tr>"
                 "<th>#</th><th>Lumanare semnal</th><th>Sesiune</th><th class='n'>Lumanare</th>"
                 "<th class='n'>Volum / media 9</th><th class='n'>ER inainte</th>"
                 "<th class='n'>ATR% inainte</th><th class='n'>Intrare</th>"
                 "<th class='n'>Dupa 15 min</th><th class='n'>Dupa 45 min</th>"
                 "<th class='n'>Max in 45 min</th><th class='n'>Min in 45 min</th></tr>")
        for k, r in enumerate(rows, 1):
            o15, o45 = r["o15"], r["o45"]
            P.append(
                f"<tr><td>{k}</td><td>{hro(r['t'])}</td><td>{r['sesiune']}</td>"
                f"<td class='n pos'>{fpct(r['lumanare'])}</td><td class='n'>{r['volx']:.1f}x</td>"
                f"<td class='n'>{f'{r["er"]:.2f}' if r['er'] == r['er'] else '—'}</td>"
                f"<td class='n'>{fpct(r['atr'], 1, False)}</td>"
                f"<td class='n'>{fp(o45['intrare']) if o45 else '—'}</td>"
                + "".join(f"<td class='n {cls(o['net_pct']) if o else ''}'>"
                          f"{fpct(o['net_pct']) if o else '—'}</td>" for o in (o15, o45))
                + f"<td class='n pos'>{fpct(o45['max_pct']) if o45 else '—'}</td>"
                f"<td class='n neg'>{fpct(o45['min_pct']) if o45 else '—'}</td></tr>")
        P.append("</table></div>")
    P.append("<div class='small'>Intrare la primul pret dupa inchiderea lumanarii de semnal. "
             "„Dupa 15/45 min” include costul. ER/ATR% inainte = ultimele 14 lumanari dinaintea "
             "semnalului (ER aproape de 1 = miscare lina, aproape de 0 = zig-zag).</div>")

    # --- spread
    sp = ctx["spreads"]
    if not sp.empty:
        P.append("<h2>Spread bid/ask (cat costa o intrare + iesire)</h2><div class='tbl'><table><tr>"
                 "<th>Ora</th><th>Sesiune</th><th class='n'>Bid</th><th class='n'>Ask</th>"
                 "<th class='n'>Spread median</th></tr>")
        for r in sp.itertuples():
            P.append(f"<tr><td>{hro(r.ora_ny)}</td><td>{session_name(r.ora_ny)}</td>"
                     f"<td class='n'>{fp(r.bid)}</td><td class='n'>{fp(r.ask)}</td>"
                     f"<td class='n'>{fpct(r.spread_pct, 2, False)}</td></tr>")
        P.append("</table></div><div class='small'>Cumperi la ask si vinzi la bid: spread-ul il "
                 "platesti o data la fiecare tranzactie dus-intors, plus alunecarea.</div>")

    # --- 15 min tabel
    P.append("<h2>Toate lumanarile de 15 minute, cu indicatori</h2><div class='tbl'><table><tr>"
             "<th>Ora</th><th class='n'>Deschidere</th><th class='n'>Maxim</th><th class='n'>Minim</th>"
             "<th class='n'>Inchidere</th><th class='n'>Schimbare</th><th class='n'>Volum</th>"
             "<th class='n'>Vol/media 9</th><th class='n'>VWAP</th><th class='n'>Fata de VWAP</th>"
             "<th class='n'>EMA 9</th><th class='n'>EMA 20</th><th class='n'>RSI 14</th>"
             "<th class='n'>ATR% 14</th><th class='n'>ER 14</th></tr>")
    for t, r in b15.iterrows():
        P.append(f"<tr class='{'sig' if r['semnal'] else ''}'><td>{hro(t)}</td>"
                 f"<td class='n'>{fp(r['open'])}</td><td class='n'>{fp(r['high'])}</td>"
                 f"<td class='n'>{fp(r['low'])}</td><td class='n'>{fp(r['close'])}</td>"
                 f"<td class='n {cls(r['schimbare_pct'])}'>{fpct(r['schimbare_pct'])}</td>"
                 f"<td class='n'>{r['volume']:,.0f}</td>"
                 f"<td class='n'>{'' if not np.isfinite(r['volum_fata_de_media9']) else f'{r.volum_fata_de_media9:.1f}x'}</td>"
                 f"<td class='n'>{fp(r['vwap'])}</td>"
                 f"<td class='n {cls(r['fata_de_vwap_pct'])}'>{fpct(r['fata_de_vwap_pct'])}</td>"
                 f"<td class='n'>{fp(r['ema9'])}</td><td class='n'>{fp(r['ema20'])}</td>"
                 f"<td class='n'>{r['rsi14']:.0f}</td><td class='n'>{fpct(r['atr_pct14'], 1, False)}</td>"
                 f"<td class='n'>{'—' if not np.isfinite(r['er14']) else f'{r.er14:.2f}'}</td></tr>")
    P.append("</table></div><div class='small'>Randurile galbene = semnal. RSI peste 70 = "
             "supracumparat, sub 30 = supravandut (pe penny stocks raman des la extreme).</div>")

    # --- stiri
    P.append("<h2>Stiri (ultimele 3 zile, sursa Alpaca/Benzinga)</h2>")
    if ctx["news"]:
        P.append("<div class='tbl'><table><tr><th>Data / ora</th><th>Titlu</th><th>Sursa</th></tr>")
        for n in ctx["news"]:
            t = n["ora"]
            title = escape(n["titlu"])
            if n["url"]:
                title = f"<a href='{escape(n['url'])}' target='_blank'>{title}</a>"
            P.append(f"<tr><td>{t.astimezone(RO):%d.%m %H:%M}</td><td style='white-space:normal'>"
                     f"{title}<div class='small'>{escape(n['rezumat'][:300])}</div></td>"
                     f"<td>{escape(n['sursa'])}</td></tr>")
        P.append("</table></div>")
    else:
        P.append("<div class='card'>Nicio stire in fluxul Alpaca. La penny stocks, multe "
                 "comunicate apar doar pe site-ul companiei sau la SEC.</div>")

    # --- istoric zilnic
    if not daily.empty:
        dd = daily.tail(30).iloc[::-1]
        P.append("<h2>Ultimele 30 de zile</h2><div class='tbl'><table><tr><th>Data</th>"
                 "<th class='n'>Deschidere</th><th class='n'>Maxim</th><th class='n'>Minim</th>"
                 "<th class='n'>Inchidere</th><th class='n'>Schimbare</th><th class='n'>Amplitudine</th>"
                 "<th class='n'>Volum</th></tr>")
        chg = daily["close"].pct_change() * 100
        for dt, r in dd.iterrows():
            rng = (r["high"] / r["low"] - 1) * 100 if r["low"] > 0 else float("nan")
            P.append(f"<tr><td>{dt:%d.%m.%Y}</td><td class='n'>{fp(r['open'])}</td>"
                     f"<td class='n'>{fp(r['high'])}</td><td class='n'>{fp(r['low'])}</td>"
                     f"<td class='n'>{fp(r['close'])}</td>"
                     f"<td class='n {cls(chg.get(dt))}'>{fpct(chg.get(dt))}</td>"
                     f"<td class='n'>{fpct(rng, 0, False)}</td><td class='n'>{r['volume']:,.0f}</td></tr>")
        P.append("</table></div><div class='small'>Preturi neajustate. Un salt brusc de pret "
                 "intre doua zile, fara volum, poate fi un reverse split.</div>")

    # --- de verificat online
    s = escape(sym)
    P.append("<h2>De verificat online (nu sunt la Alpaca)</h2><div class='card'><ul>"
             f"<li><b>Float si acțiuni in circulatie, short interest</b>: "
             f"<a href='https://finviz.com/quote.ashx?t={s}' target='_blank'>Finviz</a></li>"
             f"<li><b>Documente SEC</b> (emisiuni noi S-1 / 424B, warrant-uri, reverse split): "
             f"<a href='https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={s}"
             f"&type=&dateb=&owner=include&count=40' target='_blank'>SEC EDGAR</a></li>"
             f"<li><b>Halt-uri</b> (opriri pentru volatilitate): "
             f"<a href='https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts' target='_blank'>"
             f"Nasdaq Trader</a></li></ul>"
             "<div class='small'>Mi le poti lasa mie: cand imi trimiti raportul, le caut eu "
             "dupa simbol.</div></div>")

    P.append(f"<h2>Despre actiune (Alpaca)</h2><div class='card small'>"
             f"{escape(json.dumps(ctx['asset'], ensure_ascii=False, default=str))}</div>")

    P.append("</main></body></html>")
    return "".join(P)


# ==========================================================
# MAIN
# ==========================================================

def analyze(bot, sym, d):
    say(f"Descarc datele de minut pentru {sym} pe {d}...")
    m, feed = fetch_minutes(bot, sym, d)
    if m.empty:
        raise RuntimeError(f"Nu exista tranzactii pentru {sym} pe {d} (zi nelucratoare, "
                           "simbol gresit sau prea devreme).")
    say(f"  {len(m)} minute cu tranzactii. Descarc istoricul zilnic, stirile, spread-ul...")
    daily = fetch_daily(bot, sym, d)
    news = fetch_news(bot, sym, d)
    asset = fetch_asset(bot, sym)
    spreads = fetch_spreads(bot, sym, d, m)

    m1 = add_indicators(m)
    b5, b15 = resample(m, 5), resample(m, 15)

    def sig_rows(b, step):
        rows = []
        idx = list(b.index)
        for k, t in enumerate(idx):
            r = b.loc[t]
            if not r["semnal"]:
                continue
            prev = b.iloc[max(0, k - 14):k]
            er = atr = float("nan")
            if len(prev) >= 10:
                path = prev["close"].diff().abs().sum()
                er = abs(prev["close"].iloc[-1] - prev["close"].iloc[0]) / path if path > 0 else float("nan")
                atr = ((prev["high"] - prev["low"]) / prev["close"]).mean() * 100
            t_in = t + timedelta(minutes=step)
            rows.append({"t": t, "sesiune": session_name(t), "lumanare": r["schimbare_pct"],
                         "volx": r["volum_fata_de_media9"], "er": er, "atr": atr, "c": r["close"],
                         "o15": outcome(m, t_in, 15), "o45": outcome(m, t_in, 45)})
        return rows

    sig15, sig5 = sig_rows(b15, 15), sig_rows(b5, 5)

    # --- rezumat
    prev_close = None
    if not daily.empty:
        before = daily[daily.index < d]
        if not before.empty:
            prev_close = float(before["close"].iloc[-1])
    hm = m.index.hour * 60 + m.index.minute
    pre, reg, aft = m[hm < 570], m[(hm >= 570) & (hm < 960)], m[hm >= 960]
    k = []
    last = float(m["close"].iloc[-1])
    k.append(("Inchidere ieri", fp(prev_close), ""))
    if prev_close:
        first = float(m["open"].iloc[0])
        k.append((f"Primul pret al zilei ({m.index[0].astimezone(RO):%H:%M})",
                  f"{fp(first)} ({fpct((first / prev_close - 1) * 100, 0)})", cls(first - prev_close)))
    if not pre.empty:
        k.append((f"Maxim premarket ({pre['high'].idxmax().astimezone(RO):%H:%M})",
                  fp(pre["high"].max()), ""))
        k.append(("Volum premarket", f"{pre['volume'].sum():,.0f}", ""))
    if not reg.empty:
        k.append(("Deschidere 16:30 (9:30 NY)", fp(reg["open"].iloc[0]), ""))
        if reg.index[-1].hour >= 15 and reg.index[-1].minute >= 55:
            k.append(("Inchidere 23:00 (16:00 NY)", fp(reg["close"].iloc[-1]), ""))
    k.append((f"Maximul zilei ({m['high'].idxmax().astimezone(RO):%H:%M})", fp(m["high"].max()), ""))
    k.append((f"Minimul zilei ({m['low'].idxmin().astimezone(RO):%H:%M})", fp(m["low"].min()), ""))
    k.append((f"Ultimul pret ({m.index[-1].astimezone(RO):%H:%M})", fp(last), ""))
    if prev_close:
        k.append(("Schimbare fata de ieri", fpct((last / prev_close - 1) * 100),
                  cls(last - prev_close)))
    k.append(("Scadere de la maxim pana la ultimul pret",
              fpct((last / m["high"].max() - 1) * 100), "neg"))
    vol_day = m["volume"].sum()
    k.append(("Volum total (pana acum)", f"{vol_day:,.0f}", ""))
    if not daily.empty:
        hist = daily[daily.index < d].tail(20)
        if len(hist) >= 5 and hist["volume"].mean() > 0:
            k.append(("Volum relativ (fata de media pe 20 de zile)",
                      f"{vol_day / hist['volume'].mean():.1f}x", ""))
            rngs = (hist["high"] / hist["low"] - 1) * 100
            k.append(("Amplitudine zilnica obisnuita (mediana 20 zile)", fpct(rngs.median(), 0, False), ""))
    day_rng = (m["high"].max() / m["low"].min() - 1) * 100
    k.append(("Amplitudinea de azi (maxim/minim)", fpct(day_rng, 0, False), ""))
    k.append(("Semnale 15 min / 5 min", f"{len(sig15)} / {len(sig5)}", ""))
    if not spreads.empty:
        k.append(("Spread median (bid/ask)", fpct(spreads["spread_pct"].median(), 2, False), ""))
    k.append(("Stiri in ultimele 3 zile", str(len(news)), ""))

    ctx = {"sym": sym, "d": d, "m": m1, "daily": daily, "b5": b5, "b15": b15,
           "sig15": sig15, "sig5": sig5, "news": news, "asset": asset, "spreads": spreads,
           "feed": feed, "summary": {"kpis": k}, "gaps": list(GAPS)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / f"zi_{sym}_{d}.html"
    csv_path = OUT_DIR / f"zi_{sym}_{d}_1min.csv"
    html_path.write_text(build_report(ctx), encoding="utf-8")
    out = m1.copy()
    out.insert(0, "ora_romania", [t.astimezone(RO).strftime("%Y-%m-%d %H:%M") for t in out.index])
    out.insert(1, "sesiune", [session_name(t) for t in out.index])
    out.index = out.index.strftime("%Y-%m-%d %H:%M")
    out.index.name = "ora_new_york"
    out.round(6).to_csv(csv_path, encoding="utf-8-sig")
    say(f"Raport: {html_path}")
    say(f"Date:   {csv_path}")
    return html_path


def parse_date(text):
    """Accepta 2026-10-15, 2026.10.15, 2026/10/15, 15.10.2026, 15-10-2026, 15/10/2026."""
    import re
    parts = [p for p in re.split(r"[^0-9]+", text.strip()) if p]
    if len(parts) == 3:
        a, b, c = parts
        try:
            if len(a) == 4:
                return date(int(a), int(b), int(c))
            if len(c) == 4:
                return date(int(c), int(b), int(a))
        except ValueError:
            pass
    raise SystemExit(f"Data „{text}” nu e valida. Scrie-o ca 2026-10-15 sau 15.10.2026.")


def ask_inputs(argv):
    sym = argv[1] if len(argv) > 1 else input("Simbolul actiunii (ex. SDST): ")
    sym = sym.strip().upper()
    if len(argv) > 2:
        ds = argv[2]
    else:
        ds = input("Data (AAAA-LL-ZZ, Enter = azi): ").strip()
    if ds:
        d = parse_date(ds)
    else:
        d = datetime.now(NY).date()
    return sym, d


def main():
    try:
        sym, d = ask_inputs(sys.argv)
        if not sym:
            print("Nu ai introdus un simbol.")
            return
        bot = load_bot()
        path = analyze(bot, sym, d)
        try:
            webbrowser.open(path.as_uri())
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
        print("\nEROARE (detaliile sunt mai sus).")


if __name__ == "__main__":
    main()
