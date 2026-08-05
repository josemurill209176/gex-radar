"""
GEX / VANNA Radar — dealer-exposure dashboard (SPY / SPX / QQQ / IWM …)
Data: Yahoo Finance option chains (free, delayed ~15 min; OI = prior-night snapshot, static intraday)
Run:    streamlit run gex_dashboard.py
Deploy: push to GitHub -> share.streamlit.io

Indicators:
  • NetGEX (gamma), NetVEX (vanna), NetCharm (delta-decay) matrices — strike x expiration.
  • Volume matrix — intraday contract volume (the free proxy for flow).
  • Confidence score on the buy/put signal (magnet · regime · reach · skew).
  • Put/Call ratios (OI + volume), Max pain, Expected move, IV skew.
  • Auto-refresh toggle.
"""

import math
import time
from datetime import datetime, date, timezone

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# ---------------------------------------------------------------- greeks --

def _d1d2(spot, strike, t, iv, r=0.045):
    st_ = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * st_)
    return d1, d1 - iv * st_, st_


def bs_gamma(spot, strike, t, iv, r=0.045):
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1, _, st_ = _d1d2(spot, strike, t, iv, r)
    return (math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)) / (spot * iv * st_)


def bs_vanna(spot, strike, t, iv, r=0.045):
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1, d2, _ = _d1d2(spot, strike, t, iv, r)
    return -(math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)) * d2 / iv


def bs_charm(spot, strike, t, iv, r=0.045):
    """Charm = dDelta/dTime (per year). Approx, calls; used structurally."""
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1, d2, st_ = _d1d2(spot, strike, t, iv, r)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return -pdf * (2 * r * t - d2 * iv * st_) / (2 * t * iv * st_)


def years_to_expiry(exp_str):
    exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
    now = datetime.now(timezone.utc)
    days = (exp - now.date()).days
    frac_today = max((20 - now.hour - now.minute / 60) / 6.5 / 24, 0.03 / 24 * 6.5)
    t = max(days, 0) / 365.0 + (frac_today / 365.0 if days == 0 else 0)
    return max(t, 0.0008)


def series_flip(col):
    col = col.dropna().sort_index()
    if col.empty:
        return None
    cum = col.cumsum()
    prev = None
    for k, v in cum.items():
        sg = 1 if v >= 0 else -1
        if prev is not None and sg != prev:
            return float(k)
        prev = sg
    return None


# ------------------------------------------------------------- chain agg --

def process_chain(calls, puts, spot, t):
    """One pass over a chain -> per-strike GEX/VEX/Charm/Volume + OI/vol sums + IV."""
    gex, vex, chm, vol, coi, poi, ivs = {}, {}, {}, {}, {}, {}, {}
    sums = {"cvol": 0.0, "pvol": 0.0}

    def acc(df, sign, is_call):
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            k = float(row.get("strike", 0) or 0)
            if k <= 0:
                continue
            oi = float(row.get("openInterest", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0)
            v = float(row.get("volume", 0) or 0)
            if oi > 0 and iv > 0.01:
                gex[k] = gex.get(k, 0.0) + sign * (spot ** 2) * 0.01 * bs_gamma(spot, k, t, iv) * oi * 100
                vex[k] = vex.get(k, 0.0) + sign * spot * 0.01 * bs_vanna(spot, k, t, iv) * oi * 100
                chm[k] = chm.get(k, 0.0) + sign * spot * 0.01 * bs_charm(spot, k, t, iv) * oi * 100
            if v > 0:
                vol[k] = vol.get(k, 0.0) + v
                sums["cvol" if is_call else "pvol"] += v
            if oi > 0:
                (coi if is_call else poi)[k] = (coi if is_call else poi).get(k, 0.0) + oi
            if iv > 0.01:
                ivs[k] = (ivs[k] + iv) / 2 if k in ivs else iv

    acc(calls, +1, True)
    acc(puts, -1, False)
    atm_iv = ivs[min(ivs, key=lambda x: abs(x - spot))] if ivs else None
    return {"gex": gex, "vex": vex, "charm": chm, "vol": vol,
            "coi": coi, "poi": poi, "ivs": ivs,
            "cvol": sums["cvol"], "pvol": sums["pvol"], "atm_iv": atm_iv}


def max_pain(coi, poi):
    strikes = sorted(set(coi) | set(poi))
    if not strikes:
        return None
    best, best_pain = None, None
    for P in strikes:
        pain = sum(coi.get(s, 0) * max(P - s, 0) + poi.get(s, 0) * max(s - P, 0) for s in strikes)
        if best_pain is None or pain < best_pain:
            best_pain, best = pain, P
    return float(best)


# ------------------------------------------------------------- data -------

@st.cache_data(ttl=300, show_spinner=False)
def load_all(ticker, n_exps, _bucket=0):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot = None
    try:
        spot = tk.fast_info.last_price
    except Exception:
        pass
    if not spot:
        h = tk.history(period="1d")["Close"]
        spot = float(h.iloc[-1]) if len(h) else None
    if not spot:
        return None
    try:
        exps = list(tk.options)[:n_exps]
    except Exception:
        exps = []
    gcols, vcols, ccols, volcols, stats = {}, {}, {}, {}, {}
    iv_skew = None
    for i, exp in enumerate(exps):
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        t = years_to_expiry(exp)
        pc = process_chain(ch.calls, ch.puts, spot, t)
        if pc["gex"]:
            gcols[exp] = pd.Series(pc["gex"])
        if pc["vex"]:
            vcols[exp] = pd.Series(pc["vex"])
        if pc["charm"]:
            ccols[exp] = pd.Series(pc["charm"])
        if pc["vol"]:
            volcols[exp] = pd.Series(pc["vol"])
        coi_sum, poi_sum = sum(pc["coi"].values()), sum(pc["poi"].values())
        stats[exp] = {
            "pc_oi": (poi_sum / coi_sum) if coi_sum > 0 else None,
            "pc_vol": (pc["pvol"] / pc["cvol"]) if pc["cvol"] > 0 else None,
            "max_pain": max_pain(pc["coi"], pc["poi"]),
            "atm_iv": pc["atm_iv"],
            "dte": (datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days,
        }
        if i == 0 and pc["ivs"]:
            iv_skew = pd.Series(pc["ivs"]).sort_index()
    try:
        prev = getattr(tk.fast_info, "previous_close", None)
        day_chg = (spot - prev) / prev * 100 if prev else 0.0
    except Exception:
        day_chg = 0.0
    return {
        "spot": spot, "day_chg": day_chg,
        "gex": pd.DataFrame(gcols).sort_index() if gcols else pd.DataFrame(),
        "vex": pd.DataFrame(vcols).sort_index() if vcols else pd.DataFrame(),
        "charm": pd.DataFrame(ccols).sort_index() if ccols else pd.DataFrame(),
        "vol": pd.DataFrame(volcols).sort_index() if volcols else pd.DataFrame(),
        "stats": stats, "iv_skew": iv_skew,
    }


# ------------------------------------------------------- color + format ---

def _lerp(a, b, t):
    return a + (b - a) * t


def grid_color(v, max_abs):
    if max_abs <= 0 or pd.isna(v):
        return "#3f7d7d", "#c9d6d6"
    t = max(-1.0, min(1.0, v / max_abs))
    a = abs(t) ** 0.5
    if t >= 0:
        if a < 0.5:
            u = a / 0.5; r, g, b = _lerp(63, 111, u), _lerp(125, 184, u), _lerp(125, 111, u)
        else:
            u = (a - 0.5) / 0.5; r, g, b = _lerp(111, 232, u), _lerp(184, 232, u), _lerp(111, 74, u)
        txt = "#12210f" if a > 0.62 else "#eafaf0"
    else:
        if a < 0.5:
            u = a / 0.5; r, g, b = _lerp(63, 74, u), _lerp(125, 109, u), _lerp(125, 157, u)
        else:
            u = (a - 0.5) / 0.5; r, g, b = _lerp(74, 61, u), _lerp(109, 43, u), _lerp(157, 122, u)
        txt = "#f2ecff"
    return f"rgb({int(r)},{int(g)},{int(b)})", txt


def fmt_k(v):
    return "" if pd.isna(v) else f"{'-' if v < 0 else ''}${abs(v)/1000:,.1f}K"


def fmt_big(v):
    ax = abs(v)
    if ax >= 1e9: return f"${v/1e9:,.2f}B"
    if ax >= 1e6: return f"${v/1e6:,.1f}M"
    if ax >= 1e3: return f"${v/1e3:,.1f}K"
    return f"${v:,.0f}"


def fmt_vol(v):
    return "" if pd.isna(v) or v == 0 else f"{int(v):,}"


# ----------------------------------------------------------- matrix -------

def render_matrix(df, spot, value_fmt=fmt_k):
    if df is None or df.empty:
        st.info("No data available right now.")
        return
    dfx = df.sort_index(ascending=False)
    exps = list(dfx.columns)
    stacked = dfx.stack()
    max_abs = float(stacked.abs().max()) if len(stacked) else 1.0
    max_abs = max_abs or 1.0
    pos_c = stacked.idxmax() if (stacked > 0).any() else None
    neg_c = stacked.idxmin() if (stacked < 0).any() else None
    spot_strike = min(dfx.index, key=lambda k: abs(k - spot))
    today = date.today()

    head = ['<th style="position:sticky;top:0;left:0;z-index:3;background:#0c0d12;color:#6b7490;'
            'padding:6px 8px;text-align:left;">Strike</th>']
    for e in exps:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            tag = "0DTE" if dte == 0 else f"{dte}d"
        except Exception:
            tag = ""
        head.append(f'<th style="position:sticky;top:0;z-index:2;background:#0c0d12;color:#e8eaf0;'
                    f'padding:6px 10px;text-align:right;font-weight:700;min-width:96px;">{e}'
                    f'<div style="color:#6b7490;font-weight:400;font-size:9px;">{tag}</div></th>')
    rows = ["<tr>" + "".join(head) + "</tr>"]

    for k in dfx.index:
        is_spot = (k == spot_strike)
        klabel = (f'<span style="color:#e8eaf0;">&#9656; {k:,.1f}</span>' if is_spot
                  else f'<span style="color:#7a8296;">{k:,.1f}</span>')
        cells = [f'<td style="position:sticky;left:0;z-index:1;background:#0c0d12;padding:3px 8px;'
                 f'text-align:left;font-weight:{700 if is_spot else 400};'
                 f'border-bottom:1px solid rgba(255,255,255,0.03);">{klabel}</td>']
        for e in exps:
            v = dfx.loc[k, e]
            bg, txt = grid_color(v, max_abs)
            border, marker = "", ""
            if pos_c is not None and (k, e) == tuple(pos_c):
                border = "box-shadow:inset 0 0 0 1.5px rgba(255,255,255,0.55);"
                marker = ' <span>&#9733;</span>'
            elif neg_c is not None and (k, e) == tuple(neg_c):
                border = "box-shadow:inset 0 0 0 1.5px rgba(255,255,255,0.55);"
                marker = ' <span>&#8226;</span>'
            cells.append(f'<td style="background:{bg};color:{txt};padding:3px 10px;text-align:right;'
                         f'font-variant-numeric:tabular-nums;{border}'
                         f'border-bottom:1px solid rgba(0,0,0,0.15);">{value_fmt(v)}{marker}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')

    st.markdown('<div style="overflow:auto;max-height:560px;border:1px solid #1c1f2b;border-radius:12px;">'
                '<table style="border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;'
                'font-size:11px;">' + "".join(rows) + '</table></div>', unsafe_allow_html=True)


# ------------------------------------------------------ multi-ladder -----

def ladder_color(v, max_abs):
    """Green positive · magenta/purple negative (matches the multi-ticker screenshot)."""
    t = min(1.0, abs(v) / max_abs) if max_abs > 0 else 0.0
    e = t ** 0.62
    if v >= 0:
        r, g, b = _lerp(14, 53, e), _lerp(60, 212, e), _lerp(50, 154, e)      # -> #35d49a
    else:
        r, g, b = _lerp(80, 201, e), _lerp(30, 60, e), _lerp(120, 239, e)     # -> magenta
    return f"rgba({int(r)},{int(g)},{int(b)},{0.12 + 0.72 * e:.2f})", (e > 0.55)


def render_ladder_col(tk, bucket, window):
    """One vertical net-GEX ladder (nearest expiration) for the multi-ticker panel."""
    d = load_all(tk, 1, bucket)
    if not d or d["gex"].empty:
        st.warning(f"**{tk}** — no data (SPX `^SPX` is often empty on Yahoo free).")
        return
    spot, day = d["spot"], d["day_chg"]
    col = d["gex"].iloc[:, 0].dropna()
    lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
    col = col[(col.index >= lo) & (col.index <= hi)].sort_index(ascending=False)
    if col.empty:
        st.warning(f"**{tk}** — no strikes in window.")
        return
    max_abs = float(col.abs().max()) or 1.0
    king = float(col.abs().idxmax())
    spot_k = min(col.index, key=lambda k: abs(k - spot))
    king_pct = (king - spot) / spot * 100
    chg_c = "#35d49a" if day >= 0 else "#ff5c7a"
    kp_c = "#35d49a" if king_pct >= 0 else "#ff5c7a"

    header = (
        f'<div style="padding:7px 10px;border-bottom:1px solid #1c1f2b;font-family:ui-monospace,monospace;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:#e8eaf0;font-weight:700;font-size:13px;">&#9679; {tk}</span>'
        f'<span><span style="color:#e8eaf0;font-weight:600;font-size:12px;">${spot:,.2f}</span>'
        f'<span style="color:{chg_c};font-size:11px;font-weight:700;margin-left:5px;">'
        f'{"+" if day>=0 else ""}{day:.2f}%</span></span></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:3px;font-size:10.5px;">'
        f'<span style="color:#c9a227;">&#9733; King {king:,.0f}</span>'
        f'<span style="color:{kp_c};">{abs(king_pct):.1f}% {"&#8593;" if king_pct>=0 else "&#8595;"}</span>'
        f'</div></div>'
    )
    rows = []
    for k in col.index:
        v = float(col.loc[k])
        bg, bright = ladder_color(v, max_abs)
        is_king, is_spot = (k == king), (k == spot_k)
        strike = (f'<span style="background:#fff;color:#0b0e14;font-weight:700;font-size:10px;'
                  f'padding:0 5px;border-radius:7px;">{k:,.0f}</span>'
                  if is_spot else f'<span style="color:#565d70;padding-left:6px;">{k:,.0f}</span>')
        star = '<span style="color:#c9a227;margin-left:3px;">&#9733;</span>' if is_king else ""
        txt = "#fff" if bright else "#c5cad6"
        rows.append(
            f'<div style="display:flex;align-items:center;height:20px;font-family:ui-monospace,monospace;'
            f'font-size:10.5px;border-bottom:1px solid rgba(255,255,255,0.02);">'
            f'<div style="width:46px;flex:0 0 auto;">{strike}</div>'
            f'<div style="flex:1;display:flex;align-items:center;background:{bg};height:100%;">'
            f'<span style="margin-left:auto;padding-right:7px;color:{txt};">{fmt_k(v)}{star}</span>'
            f'</div></div>')
    st.markdown(f'<div style="background:#0c0d12;border:1px solid #1c1f2b;border-radius:10px;overflow:hidden;">'
                f'{header}<div style="max-height:520px;overflow-y:auto;">{"".join(rows)}</div></div>',
                unsafe_allow_html=True)


def render_multi(tickers, bucket, window):
    cols = st.columns(len(tickers))
    for c, tk in zip(cols, tickers):
        with c:
            render_ladder_col(tk, bucket, window)


# ------------------------------------------------------- confidence -------

def confidence(net_total, king_ex, gross_abs, king, spot, em, pc_oi, direction):
    dom = min(1.0, abs(king_ex) / gross_abs) if gross_abs > 0 else 0
    clar = min(1.0, abs(net_total) / gross_abs) if gross_abs > 0 else 0
    reach = (1 - min(1.0, abs(king - spot) / em)) if em and em > 0 else 0.5
    if direction == "calls":
        agree = 1.0 if (pc_oi is not None and pc_oi < 1) else 0.4
    elif direction == "puts":
        agree = 1.0 if (pc_oi is not None and pc_oi > 1) else 0.4
    else:
        agree = 0.5
    parts = {"magnet": round(dom * 30), "regime": round(clar * 25),
             "reach": round(reach * 25), "skew": round(agree * 20)}
    return sum(parts.values()), parts


# --------------------------------------------------------------- UI -------

st.set_page_config(page_title="GEX / Vanna Radar", page_icon="🎯", layout="wide")
st.markdown("""
<style>
  .stApp { background:#0b0e14; }
  html, body, [class*="css"] { font-family:'Inter',-apple-system,sans-serif; }
  h1 { color:#e8eaf0 !important; font-weight:800; letter-spacing:-0.02em; }
  .card { background:#131826; border:1px solid #1f2637; border-radius:14px; padding:12px 16px; }
  .lab { color:#6b7490; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; }
  .val { color:#e8eaf0; font-size:1.2rem; font-weight:700; font-variant-numeric:tabular-nums; }
  .pos { color:#35d49a; } .neg { color:#ff5c7a; } .neu { color:#f5b64c; } .blu { color:#7aa2ff; }
  section[data-testid="stSidebar"] { background:#0e1220; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 GEX / Vanna Radar")
st.caption("Quotes ~15-min delayed · OI = prior-night snapshot (static intraday) · greeks: live Black-Scholes (r=4.5%)")

with st.sidebar:
    st.header("Settings")
    window = st.slider("Strike window around spot (±%)", 2, 20, 6)
    n_exps = st.slider("Expirations across the top", 3, 8, 5)
    st.button("🔄 Refresh data", on_click=st.cache_data.clear)
    st.markdown("---")
    auto = st.toggle("Auto-refresh", value=False)
    interval = st.selectbox("Every", [30, 60, 120, 300], index=1,
                            format_func=lambda s: f"{s}s" if s < 60 else f"{s//60} min",
                            disabled=not auto)
    st.caption("Yahoo is ~15-min delayed & OI updates once daily, so 60s+ is plenty. "
               "Faster can get you rate-limited.")
    st.markdown("---")
    multi = st.toggle("Multi-ticker ladders", value=False)
    multi_syms = st.multiselect("Tickers", ["SPY", "SPX", "QQQ", "IWM", "DIA"],
                                default=["SPY", "SPX", "QQQ"], disabled=not multi)
    placement = st.radio("Placement", ["Bottom", "Popup"], horizontal=True, disabled=not multi)

choice = st.radio("Ticker", ["SPY", "SPX", "QQQ", "IWM", "Custom"],
                  horizontal=True, label_visibility="collapsed")
ticker = (st.text_input("Symbol", "AAPL").strip().upper() or "AAPL") if choice == "Custom" \
    else {"SPY": "SPY", "SPX": "^SPX", "QQQ": "QQQ", "IWM": "IWM"}[choice]

bucket = 0
if auto:
    bucket = int(time.time() // interval)
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=interval * 1000, key="auto_refresh")
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(53,212,154,0.12);'
        f'border:1px solid rgba(53,212,154,0.4);border-radius:8px;padding:2px 10px;margin-bottom:6px;'
        f'font-family:ui-monospace,monospace;font-size:11px;color:#35d49a;">'
        f'<span style="width:7px;height:7px;border-radius:99px;background:#35d49a;"></span>'
        f'LIVE · every {interval}s · updated {datetime.now().strftime("%H:%M:%S")} (server time)</div>',
        unsafe_allow_html=True)

multi_tickers = [{"SPX": "^SPX"}.get(s, s) for s in multi_syms] if multi else []
if multi and multi_tickers and placement == "Popup":
    if hasattr(st, "popover"):
        with st.popover("📊 Multi-ticker ladders"):
            st.caption("Green = positive · magenta = negative · ★ King · white pill = spot · nearest expiration")
            render_multi(multi_tickers, bucket, window)
    else:
        placement = "Bottom"   # older Streamlit without popovers -> fall back to bottom

data = load_all(ticker, n_exps, bucket)
if not data:
    st.error(f"Couldn't load {ticker} from Yahoo. If this is SPX (`^SPX`), the free feed is often "
             f"empty for index options — try SPY as a proxy (SPX ≈ SPY × 10).")
    st.stop()
spot, day_chg = data["spot"], data["day_chg"]
gdf, vdf, cdf, voldf = data["gex"], data["vex"], data["charm"], data["vol"]
if gdf.empty:
    st.warning(f"{ticker}: no usable open interest yet (0DTE OI posts pre-market).")
    st.stop()

# window filter
lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
def win(df): return df[(df.index >= lo) & (df.index <= hi)] if not df.empty else df
gdf, vdf, cdf, voldf = win(gdf), win(vdf), win(cdf), win(voldf)

# nearest-expiration signal + confidence
near = gdf.columns[0]
col = gdf[near].dropna()
net_total = float(col.sum())
gross_abs = float(col.abs().sum()) or 1.0
king = float(col.abs().idxmax()) if len(col) else spot
king_ex = float(col.loc[king]) if king in col.index else 0.0
flip = series_flip(col)
st_near = data["stats"].get(near, {})
atm_iv = st_near.get("atm_iv")
t_near = years_to_expiry(near)
em = spot * atm_iv * math.sqrt(t_near) if atm_iv else 0.0   # 1-SD expected move ($)
pc_oi = st_near.get("pc_oi")

NODE = 0.004
dist = (king - spot) / spot
sig, sig_cls, direction, why = "SIT OUT", "neu", "none", ""
if net_total >= 0:
    if dist >= NODE:
        sig, sig_cls, direction = "BUY CALLS", "pos", "calls"
        why = f"Positive gamma (calm). Price below magnet ${king:,.0f} — drifts up toward it."
    elif dist <= -NODE:
        sig, sig_cls, direction = "BUY PUTS", "neg", "puts"
        why = f"Positive gamma (calm). Price above magnet ${king:,.0f} — drifts back down."
    else:
        why = f"Price on the magnet (${king:,.0f}) — likely chops, no edge."
else:
    if flip and spot < flip:
        sig, sig_cls, direction = "BUY PUTS", "neg", "puts"
        why = f"Negative gamma (wild), below flip ${flip:,.0f} — downside runs. Ride momentum."
    else:
        why = "Negative gamma (wild), no clean level — sitting out is the smart trade."

conf, parts = confidence(net_total, king_ex, gross_abs, king, spot, em, pc_oi, direction)
conf_cls = "pos" if conf >= 70 else ("neu" if conf >= 40 else "neg")
conf_lab = "high" if conf >= 70 else ("moderate" if conf >= 40 else "low")

# ---- signal (with confidence) + key stats ----
s_col, m_col = st.columns([1.5, 4])
with s_col:
    st.markdown(
        f'<div class="card" style="border-width:2px;border-color:#2a3350;">'
        f'<div class="lab">Signal · nearest {near}</div>'
        f'<div class="val {sig_cls}" style="font-size:1.7rem;margin:2px 0 4px;">{sig}</div>'
        f'<div style="color:#aab2c8;font-size:0.8rem;line-height:1.35;">{why}</div>'
        f'<div class="lab" style="margin-top:10px;">Confidence · {conf_lab}</div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;height:8px;background:#0b0e14;border-radius:6px;overflow:hidden;">'
        f'<div style="width:{conf}%;height:100%;background:{"#35d49a" if conf>=70 else "#f5b64c" if conf>=40 else "#ff5c7a"};"></div></div>'
        f'<div class="val {conf_cls}" style="font-size:1.1rem;">{conf}</div></div>'
        f'<div style="color:#6b7490;font-size:0.66rem;margin-top:4px;">'
        f'magnet {parts["magnet"]}/30 · regime {parts["regime"]}/25 · reach {parts["reach"]}/25 · skew {parts["skew"]}/20</div>'
        f'</div>', unsafe_allow_html=True)
with m_col:
    a, b, c, d = st.columns(4)
    chg_cls = "pos" if day_chg >= 0 else "neg"
    net_cls = "pos" if net_total >= 0 else "neg"
    a.markdown(f'<div class="card"><div class="lab">{ticker} spot</div><div class="val">${spot:,.2f}</div>'
               f'<div class="{chg_cls}" style="font-size:0.8rem;">{"+" if day_chg>=0 else ""}{day_chg:.2f}%</div></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="card"><div class="lab">Net GEX · {near}</div><div class="val {net_cls}">{fmt_big(net_total)}</div>'
               f'<div class="{net_cls}" style="font-size:0.8rem;">{"dealers long γ" if net_total>=0 else "dealers short γ"}</div></div>',
               unsafe_allow_html=True)
    c.markdown(f'<div class="card"><div class="lab">King node</div><div class="val neu">${king:,.0f}</div>'
               f'<div class="neu" style="font-size:0.8rem;">price magnet</div></div>', unsafe_allow_html=True)
    d.markdown(f'<div class="card"><div class="lab">Gamma flip</div><div class="val blu">{f"${flip:,.0f}" if flip else "—"}</div>'
               f'<div class="blu" style="font-size:0.8rem;">regime line</div></div>', unsafe_allow_html=True)

# ---- flow & sentiment row ----
mp = st_near.get("max_pain")
pc_vol = st_near.get("pc_vol")

def stat_card(label, value_html, sub=""):
    sub_html = f'<div style="color:#6b7490;font-size:0.7rem;">{sub}</div>' if sub else ""
    return (f'<div class="card"><div class="lab">{label}</div>'
            f'<div class="val">{value_html}</div>{sub_html}</div>')

def pc_cls(x):
    return "neg" if (x is not None and x > 1) else "pos"

e1, e2, e3, e4 = st.columns(4)
e1.markdown(stat_card("Put/Call · OI",
            f'<span class="{pc_cls(pc_oi)}">{pc_oi:.2f}</span>' if pc_oi is not None else "—",
            ">1 = put-heavy"), unsafe_allow_html=True)
e2.markdown(stat_card("Put/Call · Volume",
            f'<span class="{pc_cls(pc_vol)}">{pc_vol:.2f}</span>' if pc_vol is not None else "—",
            "today's flow"), unsafe_allow_html=True)
e3.markdown(stat_card(f"Max pain · {near}",
            f'<span class="neu">${mp:,.0f}</span>' if mp else "—",
            "pin bias"), unsafe_allow_html=True)
e4.markdown(stat_card(f"Expected move · {near}",
            f'±${em:,.0f}' if em > 0 else "—",
            f'±{em/spot*100:.2f}% · ${spot-em:,.0f}–${spot+em:,.0f}' if em > 0 else ""),
            unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---- the matrices ----
st.subheader("NetGEX — gamma exposure")
st.caption("Dates across the top · strikes down the left · ★ biggest positive node · • biggest negative · ▸ spot row")
render_matrix(gdf, spot)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.subheader("NetVEX — vanna exposure")
st.caption("How dealer delta shifts as IV moves · strongest into OpEx / vol-regime changes")
render_matrix(vdf, spot)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.subheader("NetCharm — delta decay")
st.caption("How dealer delta drifts purely from time passing · dominant on 0DTE afternoons / into expiry")
render_matrix(cdf, spot)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.subheader("Volume — today's contract flow")
st.caption("Intraday contracts traded (call+put) · the free proxy for where fresh positioning is happening")
render_matrix(voldf, spot, value_fmt=fmt_vol)

# ---- IV skew ----
with st.expander("IV skew (nearest expiration)"):
    sk = data.get("iv_skew")
    if sk is not None and not sk.empty:
        sk = sk[(sk.index >= lo) & (sk.index <= hi)]
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=sk.index, y=(sk.values * 100), mode="lines+markers",
                                   line=dict(color="#7aa2ff", width=2), marker=dict(size=4)))
        fig.add_vline(x=spot, line_color="#e8eaf0", line_dash="dot",
                      annotation_text=f"spot {spot:,.0f}", annotation_font_color="#e8eaf0")
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="#0b0e14",
                          plot_bgcolor="#0b0e14", margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="Strike", yaxis_title="Implied vol (%)",
                          font=dict(family="Inter, sans-serif", color="#aab2c8"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Downward slope L→R = put skew (crash premium). Steeper skew = more fear priced in.")
    else:
        st.info("No IV data for the nearest expiration.")

with st.expander("How to read this / what each indicator does"):
    st.markdown("""
- **NetGEX (gamma):** dealer hedging map. Positive = they buy dips/sell rips (chop, pinning). Negative = they chase moves (trend). The **flip** is the regime line; the **King** is the strongest magnet.
- **NetVEX (vanna):** how dealer delta shifts when *IV* moves. Big into OpEx and vol-regime changes — tells you where a vol crush/spike would create mechanical buying or selling.
- **NetCharm (delta decay):** how dealer delta drifts just from *time passing*. The engine behind afternoon 0DTE pins and OpEx drift.
- **Volume matrix:** today's actual contracts traded — your free stand-in for live flow (OI is stale intraday, volume is not).
- **Confidence score:** blends magnet dominance, regime one-sidedness, whether the King sits inside the expected move (reachable), and put/call skew agreement — a 0–100 gut-check on the signal.
- **Put/Call, Max pain, Expected move, IV skew:** sentiment + the market's own 1-SD range + where "pin" pressure sits.
- **Limitation:** OI is a prior-night snapshot; intraday *positioning changes* are only inferable from volume/flow (what paid feeds sell). SPX (`^SPX`) is often empty on Yahoo's free feed — use SPY as a proxy.
""")

# ---- multi-ticker ladders at the bottom ----
if multi and multi_tickers and placement == "Bottom":
    st.markdown("---")
    st.subheader("Multi-ticker — net GEX ladders (nearest expiration)")
    st.caption("Green = positive · magenta = negative · ★ King node · white pill = spot")
    render_multi(multi_tickers, bucket, window)

# fallback auto-refresh if the helper package isn't installed
if auto and not HAS_AUTOREFRESH:
    time.sleep(interval)
    st.rerun()
