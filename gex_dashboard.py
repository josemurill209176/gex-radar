"""
GEX / VANNA Radar — strike × expiration matrix (SPY / SPX / QQQ / IWM …)
Data: Yahoo Finance option chains (free, delayed ~15 min; OI updates once daily pre-market)
Run:    streamlit run gex_dashboard.py
Deploy: push to GitHub -> share.streamlit.io

Layout:
  • Ticker tabs up top.
  • BUY CALLS / BUY PUTS signal + key stats on the side.
  • Two matrices: NetGEX and NetVEX — expiration dates across the top, strikes down the left,
    dollar exposure in each colored cell, King node + spot row highlighted.
"""

import math
from datetime import datetime, date, timezone

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- math ----

def bs_gamma(spot, strike, t, iv, r=0.045):
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return pdf / (spot * iv * math.sqrt(t))


def bs_vanna(spot, strike, t, iv, r=0.045):
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    st_ = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * st_)
    d2 = d1 - iv * st_
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return -pdf * d2 / iv


METRICS = {
    "gamma": {"fn": bs_gamma, "spot_pow": 2, "label": "NetGEX"},
    "vanna": {"fn": bs_vanna, "spot_pow": 1, "label": "NetVEX"},
}


def compute_exposure(calls, puts, spot, t, kind):
    """Per-strike net exposure. Calls +, puts −."""
    m = METRICS[kind]
    fn, scale = m["fn"], (spot ** m["spot_pow"]) * 0.01
    rows = {}
    for df, sign in ((calls, +1), (puts, -1)):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            k = float(row.get("strike", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0)
            if k <= 0 or oi <= 0 or iv <= 0.01:
                continue
            ex = sign * scale * fn(spot, k, t, iv) * oi * 100
            rows[k] = rows.get(k, 0.0) + ex
    return rows


def years_to_expiry(exp_str):
    exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
    now = datetime.now(timezone.utc)
    days = (exp - now.date()).days
    frac_today = max((20 - now.hour - now.minute / 60) / 6.5 / 24, 0.03 / 24 * 6.5)
    t = max(days, 0) / 365.0 + (frac_today / 365.0 if days == 0 else 0)
    return max(t, 0.0008)


def series_flip(col):
    """Strike where cumulative net exposure crosses zero."""
    col = col.dropna().sort_index()
    if col.empty:
        return None
    cum = col.cumsum()
    prev_sign = None
    for strike, v in cum.items():
        sign = 1 if v >= 0 else -1
        if prev_sign is not None and sign != prev_sign:
            return float(strike)
        prev_sign = sign
    return None


# ------------------------------------------------------------- data -------

@st.cache_data(ttl=300, show_spinner=False)
def load_grid_both(ticker, n_exps):
    """One fetch per expiration; returns spot + GEX grid + VEX grid (strike x expiration)."""
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
        return None, pd.DataFrame(), pd.DataFrame(), 0.0
    try:
        exps = list(tk.options)[:n_exps]
    except Exception:
        exps = []
    gcols, vcols = {}, {}
    for exp in exps:
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        t = years_to_expiry(exp)
        g = compute_exposure(ch.calls, ch.puts, spot, t, "gamma")
        v = compute_exposure(ch.calls, ch.puts, spot, t, "vanna")
        if g:
            gcols[exp] = pd.Series(g)
        if v:
            vcols[exp] = pd.Series(v)
    gdf = pd.DataFrame(gcols).sort_index() if gcols else pd.DataFrame()
    vdf = pd.DataFrame(vcols).sort_index() if vcols else pd.DataFrame()
    # day change %
    try:
        prev = getattr(tk.fast_info, "previous_close", None)
        day_chg = (spot - prev) / prev * 100 if prev else 0.0
    except Exception:
        day_chg = 0.0
    return spot, gdf, vdf, day_chg


# ------------------------------------------------------- color + format ---

def _lerp(a, b, t):
    return a + (b - a) * t


def grid_color(v, max_abs):
    """Teal base · positive -> green -> yellow · negative -> blue -> purple."""
    if max_abs <= 0 or pd.isna(v):
        return "#3f7d7d", "#c9d6d6"
    t = max(-1.0, min(1.0, v / max_abs))
    a = abs(t) ** 0.5
    if t >= 0:
        if a < 0.5:
            tt = a / 0.5
            r, g, b = _lerp(63, 111, tt), _lerp(125, 184, tt), _lerp(125, 111, tt)
        else:
            tt = (a - 0.5) / 0.5
            r, g, b = _lerp(111, 232, tt), _lerp(184, 232, tt), _lerp(111, 74, tt)
        txt = "#12210f" if a > 0.62 else "#eafaf0"
    else:
        if a < 0.5:
            tt = a / 0.5
            r, g, b = _lerp(63, 74, tt), _lerp(125, 109, tt), _lerp(125, 157, tt)
        else:
            tt = (a - 0.5) / 0.5
            r, g, b = _lerp(74, 61, tt), _lerp(109, 43, tt), _lerp(157, 122, tt)
        txt = "#f2ecff"
    return f"rgb({int(r)},{int(g)},{int(b)})", txt


def fmt_k(v):
    if pd.isna(v):
        return ""
    return f"{'-' if v < 0 else ''}${abs(v)/1000:,.1f}K"


def fmt_big(v):
    ax = abs(v)
    if ax >= 1e9: return f"${v/1e9:,.2f}B"
    if ax >= 1e6: return f"${v/1e6:,.1f}M"
    if ax >= 1e3: return f"${v/1e3:,.1f}K"
    return f"${v:,.0f}"


# ----------------------------------------------------------- matrix -------

def render_matrix(df, spot, label):
    if df is None or df.empty:
        st.info(f"No {label} data available right now.")
        return
    dfx = df.sort_index(ascending=False)          # strikes high -> low
    exps = list(dfx.columns)
    stacked = dfx.stack()
    max_abs = float(stacked.abs().max()) if len(stacked) else 1.0
    max_abs = max_abs or 1.0
    pos_coord = stacked.idxmax() if (stacked > 0).any() else None   # (strike, exp)
    neg_coord = stacked.idxmin() if (stacked < 0).any() else None
    spot_strike = min(dfx.index, key=lambda k: abs(k - spot))
    today = date.today()

    # header
    head = ['<th style="position:sticky;top:0;left:0;z-index:3;background:#0c0d12;'
            'color:#6b7490;padding:6px 8px;text-align:left;">Strike</th>']
    for e in exps:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            tag = "0DTE" if dte == 0 else f"{dte}d"
        except Exception:
            tag = ""
        head.append(f'<th style="position:sticky;top:0;z-index:2;background:#0c0d12;'
                    f'color:#e8eaf0;padding:6px 10px;text-align:right;font-weight:700;'
                    f'min-width:96px;">{e}<div style="color:#6b7490;font-weight:400;'
                    f'font-size:9px;">{tag}</div></th>')
    rows = ["<tr>" + "".join(head) + "</tr>"]

    for k in dfx.index:
        is_spot = (k == spot_strike)
        klabel = (f'<span style="color:#e8eaf0;">&#9656; {k:,.1f}</span>'
                  if is_spot else f'<span style="color:#7a8296;">{k:,.1f}</span>')
        cells = [f'<td style="position:sticky;left:0;z-index:1;background:#0c0d12;'
                 f'padding:3px 8px;text-align:left;font-weight:{700 if is_spot else 400};'
                 f'border-bottom:1px solid rgba(255,255,255,0.03);">{klabel}</td>']
        for e in exps:
            v = dfx.loc[k, e]
            bg, txt = grid_color(v, max_abs)
            border = ""
            marker = ""
            if pos_coord is not None and (k, e) == tuple(pos_coord):
                border = "box-shadow:inset 0 0 0 1.5px rgba(255,255,255,0.55);"
                marker = ' <span style="opacity:.9;">&#9733;</span>'
            elif neg_coord is not None and (k, e) == tuple(neg_coord):
                border = "box-shadow:inset 0 0 0 1.5px rgba(255,255,255,0.55);"
                marker = ' <span style="opacity:.9;">&#8226;</span>'
            cells.append(
                f'<td style="background:{bg};color:{txt};padding:3px 10px;text-align:right;'
                f'font-variant-numeric:tabular-nums;{border}'
                f'border-bottom:1px solid rgba(0,0,0,0.15);">{fmt_k(v)}{marker}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')

    html = (
        '<div style="overflow:auto;max-height:640px;border:1px solid #1c1f2b;border-radius:12px;">'
        '<table style="border-collapse:collapse;width:100%;font-family:ui-monospace,monospace;'
        'font-size:11px;">' + "".join(rows) + '</table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------- UI -------

st.set_page_config(page_title="GEX / Vanna Radar", page_icon="🎯", layout="wide")
st.markdown("""
<style>
  .stApp { background:#0b0e14; }
  html, body, [class*="css"] { font-family:'Inter',-apple-system,sans-serif; }
  h1 { color:#e8eaf0 !important; font-weight:800; letter-spacing:-0.02em; }
  .card { background:#131826; border:1px solid #1f2637; border-radius:14px; padding:12px 16px; }
  .lab { color:#6b7490; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; }
  .val { color:#e8eaf0; font-size:1.25rem; font-weight:700; font-variant-numeric:tabular-nums; }
  .pos { color:#35d49a; } .neg { color:#ff5c7a; } .neu { color:#f5b64c; } .blu { color:#7aa2ff; }
  section[data-testid="stSidebar"] { background:#0e1220; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 GEX / Vanna Radar")
st.caption("Dealer gamma & vanna from free Yahoo option data · OI updates once daily pre-market · prices delayed ~15 min")

with st.sidebar:
    st.header("Settings")
    window = st.slider("Strike window around spot (±%)", 2, 20, 6)
    n_exps = st.slider("Expirations across the top", 3, 8, 5)
    st.button("🔄 Refresh data", on_click=st.cache_data.clear)
    st.markdown("---")
    st.caption("NetGEX = S²·0.01·Γ·OI·100. NetVEX = S·0.01·vanna·OI·100. Calls +, puts −.")

# ticker tabs
choice = st.radio("Ticker", ["SPY", "SPX", "QQQ", "IWM", "Custom"],
                  horizontal=True, label_visibility="collapsed")
if choice == "Custom":
    ticker = st.text_input("Symbol", "AAPL").strip().upper() or "AAPL"
else:
    ticker = {"SPY": "SPY", "SPX": "^SPX", "QQQ": "QQQ", "IWM": "IWM"}[choice]

spot, gdf, vdf, day_chg = load_grid_both(ticker, n_exps)

if not spot:
    st.error(f"Couldn't load {ticker} from Yahoo. If this is SPX (`^SPX`), the free feed is often "
             f"empty for index options — try SPY as a proxy (SPX ≈ SPY × 10).")
    st.stop()
if gdf.empty:
    st.warning(f"{ticker}: no usable open interest yet (0DTE OI posts pre-market).")
    st.stop()

# window filter
lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
gdf = gdf[(gdf.index >= lo) & (gdf.index <= hi)]
vdf = vdf[(vdf.index >= lo) & (vdf.index <= hi)] if not vdf.empty else vdf

# signal + stats from nearest expiration (gamma)
near = gdf.columns[0]
col = gdf[near].dropna()
net_total = float(col.sum())
king = float(col.abs().idxmax()) if len(col) else spot
flip = series_flip(col)

NODE = 0.004
dist = (king - spot) / spot
sig, sig_cls, why = "SIT OUT", "neu", ""
if net_total >= 0:
    if dist >= NODE:
        sig, sig_cls = "BUY CALLS", "pos"
        why = f"Positive gamma (calm). Price below magnet ${king:,.0f} — tends to drift up to it."
    elif dist <= -NODE:
        sig, sig_cls = "BUY PUTS", "neg"
        why = f"Positive gamma (calm). Price above magnet ${king:,.0f} — tends to drift back down."
    else:
        why = f"Price sitting on the magnet (${king:,.0f}) — likely chops, no edge."
else:
    if flip and spot < flip:
        sig, sig_cls = "BUY PUTS", "neg"
        why = f"Negative gamma (wild), below flip ${flip:,.0f} — downside tends to run. Ride momentum."
    else:
        why = "Negative gamma (wild), no clean level — sitting out is the smart trade."

# ---- top row: signal on the side + key stats ----
s_col, m_col = st.columns([1.3, 4])
with s_col:
    st.markdown(
        f'<div class="card" style="border-width:2px;border-color:#2a3350;">'
        f'<div class="lab">Signal · nearest {near}</div>'
        f'<div class="val {sig_cls}" style="font-size:1.7rem;margin:2px 0 4px;">{sig}</div>'
        f'<div style="color:#aab2c8;font-size:0.8rem;line-height:1.35;">{why}</div>'
        f'<div style="color:#6b7490;font-size:0.68rem;margin-top:8px;">Rules: risk ≤12% · stop −50% '
        f'· TP +100% or magnet tag · one trade/day · stale after ~10 AM.</div></div>',
        unsafe_allow_html=True)
with m_col:
    a, b, c, d = st.columns(4)
    chg_cls = "pos" if day_chg >= 0 else "neg"
    net_cls = "pos" if net_total >= 0 else "neg"
    a.markdown(f'<div class="card"><div class="lab">{ticker} spot</div>'
               f'<div class="val">${spot:,.2f}</div>'
               f'<div class="{chg_cls}" style="font-size:0.8rem;">{"+" if day_chg>=0 else ""}{day_chg:.2f}%</div></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="card"><div class="lab">Net GEX · {near}</div>'
               f'<div class="val {net_cls}">{fmt_big(net_total)}</div>'
               f'<div class="{net_cls}" style="font-size:0.8rem;">'
               f'{"dealers long γ" if net_total>=0 else "dealers short γ"}</div></div>',
               unsafe_allow_html=True)
    c.markdown(f'<div class="card"><div class="lab">King node</div>'
               f'<div class="val neu">${king:,.0f}</div>'
               f'<div class="neu" style="font-size:0.8rem;">price magnet</div></div>',
               unsafe_allow_html=True)
    d.markdown(f'<div class="card"><div class="lab">Gamma flip</div>'
               f'<div class="val blu">{f"${flip:,.0f}" if flip else "—"}</div>'
               f'<div class="blu" style="font-size:0.8rem;">regime line</div></div>',
               unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---- the two matrices ----
st.subheader("NetGEX — gamma exposure")
st.caption("Dates across the top · strikes down the left · ★ = biggest positive node · • = biggest negative · ▸ = spot row")
render_matrix(gdf, spot, "NetGEX")

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

st.subheader("NetVEX — vanna exposure")
st.caption("How dealer delta shifts as IV moves · strongest into OpEx / vol-regime changes")
render_matrix(vdf, spot, "NetVEX")

with st.expander("How to read this"):
    st.markdown("""
- **Rows = strikes, columns = expiration dates.** Each cell is that strike/expiry's net dealer exposure.
- **Yellow-green** = large positive (call-heavy / dealer long). **Purple-blue** = large negative (put-heavy / dealer short). **Teal** = near zero.
- **★** marks the single biggest positive node, **•** the biggest negative — the price magnets.
- **▸ on the left** marks the strike nearest spot.
- **Signal** uses the nearest expiration's gamma regime — a market-structure lean, **not** guaranteed.
- **Limitation:** OI updates once daily pre-market, so intraday flow is invisible (paid feeds add that). SPX (`^SPX`) is often empty on Yahoo's free feed — use SPY as a proxy if so.
""")
