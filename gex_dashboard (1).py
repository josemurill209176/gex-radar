"""
GEX / VANNA Radar — free dealer-exposure dashboard
Data: Yahoo Finance option chains (free, delayed ~15 min; OI updates once daily pre-market)
Run locally:   streamlit run gex_dashboard.py
Deploy free:   push to GitHub -> share.streamlit.io

CHANGES vs original:
  • Added Vanna exposure (VEX) alongside Gamma (GEX) — Gamma/Vanna toggle in sidebar.
  • Generalized exposure engine so metrics, King, flip, chart, grid all follow the toggle.
  • Added a strike-ladder heatmap panel (colored rows, King ★, spot pill) matching the
    screenshot aesthetic — rendered as styled HTML so it looks like the reference tool.
  • Everything still runs on your existing live yfinance pipeline.
"""

import math
from datetime import datetime, date, timezone

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- math ----

def bs_gamma(spot: float, strike: float, t_years: float, iv: float, r: float = 0.045) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return pdf / (spot * iv * math.sqrt(t_years))


def bs_vanna(spot: float, strike: float, t_years: float, iv: float, r: float = 0.045) -> float:
    """Black-Scholes vanna = dDelta/dVol = dVega/dSpot (same for calls and puts).
    vanna = -phi(d1) * d2 / iv"""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return -pdf * d2 / iv


# metric registry: name -> (greek fn, spot-power scaling, human label, per-move label)
METRICS = {
    "gamma": {
        "fn": bs_gamma,
        "spot_pow": 2,   # GEX = S^2 * 0.01 * gamma * OI * 100  ($/1% move)
        "label": "GEX",
        "axis": "Dealer gamma exposure per 1% move ($)",
    },
    "vanna": {
        "fn": bs_vanna,
        "spot_pow": 1,   # VEX = S   * 0.01 * vanna * OI * 100  ($/1 vol pt)
        "label": "VEX",
        "axis": "Dealer vanna exposure per 1 vol point ($)",
    },
}


def compute_exposure(calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
                     t_years: float, kind: str = "gamma") -> pd.DataFrame:
    """Per-strike dealer exposure for the chosen greek.
    Convention: calls positive, puts negative (standard dealer-positioning assumption)."""
    m = METRICS[kind]
    fn, spot_pow = m["fn"], m["spot_pow"]
    scale = (spot ** spot_pow) * 0.01
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
            greek = fn(spot, k, t_years, iv)
            ex = sign * scale * greek * oi * 100
            entry = rows.setdefault(k, {"strike": k, "call_ex": 0.0, "put_ex": 0.0})
            if sign > 0:
                entry["call_ex"] += ex
            else:
                entry["put_ex"] += ex
    out = pd.DataFrame(rows.values())
    if not out.empty:
        out = out.sort_values("strike").reset_index(drop=True)
        out["net_ex"] = out["call_ex"] + out["put_ex"]
        out["abs_ex"] = out["net_ex"].abs()
    return out


# thin wrapper so the multi-exp grid keeps working; follows selected metric
def compute_gex(calls, puts, spot, t_years, kind="gamma"):
    return compute_exposure(calls, puts, spot, t_years, kind)


def find_flip(gex: pd.DataFrame):
    """Strike where cumulative net exposure crosses zero (gamma/vanna flip)."""
    if gex.empty:
        return None
    cum = gex["net_ex"].cumsum()
    sign = cum.apply(lambda x: 1 if x >= 0 else -1)
    for i in range(1, len(sign)):
        if sign.iloc[i] != sign.iloc[i - 1]:
            return float(gex["strike"].iloc[i])
    return None


def years_to_expiry(exp_str: str) -> float:
    """Time to 4pm ET on expiry date, in years. Floors 0DTE at ~45 minutes."""
    exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
    now = datetime.now(timezone.utc)
    days = (exp - now.date()).days
    close_utc = 20
    frac_today = max((close_utc - now.hour - now.minute / 60) / 6.5 / 24, 0.03 / 24 * 6.5)
    t = max(days, 0) / 365.0 + (frac_today / 365.0 if days == 0 else 0)
    return max(t, 0.0008)


# ------------------------------------------------------------- data -------

@st.cache_data(ttl=300, show_spinner=False)
def load_chain(ticker: str, expiration: str):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot = None
    try:
        spot = tk.fast_info.last_price
    except Exception:
        pass
    if not spot:
        hist = tk.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
    chain = tk.option_chain(expiration)
    return spot, chain.calls, chain.puts


@st.cache_data(ttl=1800, show_spinner=False)
def load_expirations(ticker: str):
    import yfinance as yf
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=600, show_spinner=False)
def load_grid(ticker: str, n_exps: int = 5, kind: str = "gamma"):
    """Net exposure per strike across the next n expirations (heatmap grid)."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    spot_g = tk.fast_info.last_price
    today = date.today()
    cols = {}
    for exp in list(tk.options)[:n_exps]:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        t = max(dte, 0.2) / 365.0
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        g = compute_exposure(ch.calls, ch.puts, spot_g, t, kind)
        if not g.empty:
            cols[exp] = g.set_index("strike")["net_ex"]
    return spot_g, pd.DataFrame(cols).sort_index()


# ------------------------------------------------------- ladder render ----

def _lerp(a, b, t):
    return a + (b - a) * t


def _ladder_color(value, max_abs):
    """Diverging heatmap: green positive, pink/red negative, intensity by magnitude."""
    t = min(1.0, abs(value) / max_abs) if max_abs > 0 else 0.0
    ease = t ** 0.62
    if value >= 0:                      # -> #35d49a
        r, g, b = _lerp(14, 53, ease), _lerp(60, 212, ease), _lerp(50, 154, ease)
    else:                               # -> #ff5c7a
        r, g, b = _lerp(70, 255, ease), _lerp(20, 92, ease), _lerp(90, 122, ease)
    alpha = 0.10 + 0.72 * ease
    bg = f"rgba({int(r)},{int(g)},{int(b)},{alpha:.2f})"
    bar = f"rgb({int(r)},{int(g)},{int(b)})"
    return bg, bar, ease


def _fmt_val(v):
    ax = abs(v)
    if ax >= 1e9:
        return f"{'-' if v < 0 else ''}${ax/1e9:,.2f}B"
    if ax >= 1e6:
        return f"{'-' if v < 0 else ''}${ax/1e6:,.1f}M"
    return f"{'-' if v < 0 else ''}${ax/1e3:,.1f}K"


def render_ladder(g: pd.DataFrame, spot: float, label: str, prev_map: dict | None = None):
    """Screenshot-style vertical strike ladder as styled HTML."""
    if g.empty:
        st.info("No ladder data.")
        return
    gg = g.sort_values("strike", ascending=False).reset_index(drop=True)
    max_abs = float(gg["abs_ex"].max()) or 1.0
    king_k = float(gg.loc[gg["abs_ex"].idxmax(), "strike"])
    spot_k = float(gg.iloc[(gg["strike"] - spot).abs().argmin()]["strike"])

    rows_html = []
    for _, r in gg.iterrows():
        k = float(r["strike"]); val = float(r["net_ex"])
        bg, bar, ease = _ladder_color(val, max_abs)
        is_king = (k == king_k)
        is_spot = (k == spot_k)
        # % change badge vs prior snapshot, if available
        badge = ""
        if prev_map and k in prev_map and abs(prev_map[k]) > 1:
            d = (val - prev_map[k]) / abs(prev_map[k]) * 100
            if abs(d) >= 1.5:
                bc = "#0b0e14"
                bgc = "rgba(53,212,154,0.92)" if d >= 0 else "rgba(255,92,122,0.92)"
                badge = (f'<span style="position:absolute;left:6px;font-size:9px;font-weight:700;'
                         f'padding:0 4px;border-radius:8px;line-height:14px;background:{bgc};'
                         f'color:{bc};">{"+" if d>0 else ""}{d:.0f}%</span>')
        strike_cell = (
            f'<span style="background:#fff;color:#0b0e14;font-weight:700;font-size:10.5px;'
            f'padding:0 6px;border-radius:8px;">{k:,.0f}</span>'
            f'<span style="color:#fff;font-size:9px;margin-left:-1px;">&#9656;</span>'
            if is_spot else
            f'<span style="color:#565d70;padding-left:8px;">{k:,.0f}</span>'
        )
        star = ('<span style="color:#c9a227;margin-left:4px;">&#9733;</span>' if is_king else "")
        txt_color = "#ffffff" if ease > 0.55 else "#c5cad6"
        weight = 600 if ease > 0.55 else 400
        rows_html.append(
            f'<div style="display:flex;align-items:center;height:22px;font-family:ui-monospace,'
            f'monospace;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.02);">'
            f'<div style="width:56px;flex:0 0 auto;display:flex;align-items:center;">{strike_cell}</div>'
            f'<div style="position:relative;flex:1;height:100%;display:flex;align-items:center;'
            f'background:{bg};">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;width:{ease*100:.0f}%;'
            f'background:{bar};opacity:0.16;"></div>'
            f'{badge}'
            f'<span style="margin-left:auto;padding-right:8px;color:{txt_color};font-weight:{weight};'
            f'display:flex;align-items:center;">{_fmt_val(val)}{star}</span>'
            f'</div></div>'
        )

    html = (
        f'<div style="background:#0c0d12;border:1px solid #1c1f2b;border-radius:12px;'
        f'overflow:hidden;">'
        f'<div style="padding:6px 10px;border-bottom:1px solid #1c1f2b;color:#6b7490;'
        f'font-family:ui-monospace,monospace;font-size:11px;display:flex;'
        f'justify-content:space-between;">'
        f'<span>{label} ladder</span>'
        f'<span style="color:#c9a227;">&#9733; King {king_k:,.0f}</span></div>'
        f'<div style="max-height:600px;overflow-y:auto;">{"".join(rows_html)}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------- UI -------

st.set_page_config(page_title="GEX / Vanna Radar", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0b0e14; }
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
  h1 { color: #e8eaf0 !important; font-weight: 800; letter-spacing: -0.02em; }
  .metric-card { background: #131826; border: 1px solid #1f2637; border-radius: 14px;
    padding: 14px 18px; text-align: left; }
  .metric-label { color: #6b7490; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 2px; }
  .metric-value { color: #e8eaf0; font-size: 1.45rem; font-weight: 700;
    font-variant-numeric: tabular-nums; }
  .metric-sub { font-size: 0.75rem; margin-top: 2px; }
  .pos { color: #35d49a; } .neg { color: #ff5c7a; } .neu { color: #f5b64c; }
  .regime-band { border-radius: 14px; padding: 12px 18px; font-weight: 600;
    font-size: 0.95rem; margin: 4px 0 10px 0; }
  .regime-pos { background: rgba(53,212,154,0.10); border: 1px solid rgba(53,212,154,0.35); color:#35d49a; }
  .regime-neg { background: rgba(255,92,122,0.10); border: 1px solid rgba(255,92,122,0.35); color:#ff5c7a; }
  section[data-testid="stSidebar"] { background: #0e1220; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 GEX / Vanna Radar")
st.caption("Dealer gamma & vanna exposure from free Yahoo option data · OI refreshes once daily pre-market · prices delayed ~15 min")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    metric_label = st.radio("Exposure", ["Gamma (GEX)", "Vanna (VEX)"], index=0)
    kind = "gamma" if metric_label.startswith("Gamma") else "vanna"
    MLAB = METRICS[kind]["label"]
    view_mode = st.radio("View", ["Single expiration", "Multi-expiration grid"], index=0)
    try:
        expirations = load_expirations(ticker)
    except Exception as e:
        st.error(f"Couldn't load expirations for {ticker}: {e}")
        st.stop()
    if not expirations:
        st.error("No option expirations found.")
        st.stop()
    labels = []
    today = date.today()
    for x in expirations[:15]:
        d = (datetime.strptime(x, "%Y-%m-%d").date() - today).days
        tag = "0DTE" if d == 0 else f"{d}DTE"
        labels.append(f"{x}  ({tag})")
    pick = st.selectbox("Expiration", labels, index=0)
    expiration = pick.split()[0]
    window = st.slider("Strike window around spot (±%)", 1, 10, 4)
    st.button("🔄 Refresh data", on_click=st.cache_data.clear)
    st.markdown("---")
    if kind == "gamma":
        st.caption("GEX = S² × 0.01 × Γ × OI × 100. Calls +, puts − (standard dealer-positioning assumption).")
    else:
        st.caption("VEX = S × 0.01 × vanna × OI × 100. Vanna = ∂Δ/∂σ. Calls +, puts −. "
                   "Positive VEX above spot / negative below is the classic profile.")

try:
    spot, calls, puts = load_chain(ticker, expiration)
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

if not spot:
    st.error("Couldn't get spot price.")
    st.stop()

t_years = years_to_expiry(expiration)


@st.cache_data(ttl=600, show_spinner=False)
def load_rsi(ticker: str, period: int = 14):
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="3mo")["Close"]
    if len(hist) < period + 1:
        return None
    delta = hist.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return float((100 - 100 / (1 + rs)).iloc[-1])


rsi = None
try:
    rsi = load_rsi(ticker)
except Exception:
    pass

gex = compute_exposure(calls, puts, spot, t_years, kind)
if gex.empty:
    st.warning("No usable open interest for this expiration yet (0DTE OI posts pre-market).")
    st.stop()

lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
g = gex[(gex["strike"] >= lo) & (gex["strike"] <= hi)].copy()
if g.empty:
    g = gex.copy()

net_total = g["net_ex"].sum()
flip = find_flip(g)
king_row = g.loc[g["abs_ex"].idxmax()]
king = float(king_row["strike"])


def fmt_b(x):
    ax = abs(x)
    if ax >= 1e9: return f"${x/1e9:,.2f}B"
    if ax >= 1e6: return f"${x/1e6:,.1f}M"
    return f"${x:,.0f}"


c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{ticker} spot</div>'
                f'<div class="metric-value">${spot:,.2f}</div>'
                f'<div class="metric-sub neu">{expiration}</div></div>', unsafe_allow_html=True)
with c2:
    cls = "pos" if net_total >= 0 else "neg"
    if kind == "gamma":
        sub = "dealers long gamma" if net_total >= 0 else "dealers short gamma"
    else:
        sub = "positive vanna tilt" if net_total >= 0 else "negative vanna tilt"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Net {MLAB} (window)</div>'
                f'<div class="metric-value {cls}">{fmt_b(net_total)}</div>'
                f'<div class="metric-sub {cls}">{sub}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{MLAB} flip</div>'
                f'<div class="metric-value">{f"${flip:,.0f}" if flip else "—"}</div>'
                f'<div class="metric-sub neu">regime changes here</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">King node</div>'
                f'<div class="metric-value">${king:,.0f}</div>'
                f'<div class="metric-sub neu">{fmt_b(float(king_row["net_ex"]))} · price magnet</div></div>',
                unsafe_allow_html=True)
with c5:
    if rsi is not None:
        r_cls = "neg" if rsi >= 70 else ("pos" if rsi <= 30 else "neu")
        r_note = "overbought — stretched up" if rsi >= 70 else (
                 "oversold — stretched down" if rsi <= 30 else "neutral zone")
        st.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14, daily)</div>'
                    f'<div class="metric-value {r_cls}">{rsi:.0f}</div>'
                    f'<div class="metric-sub {r_cls}">{r_note}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="metric-label">RSI (14, daily)</div>'
                    '<div class="metric-value">—</div></div>', unsafe_allow_html=True)

# regime band (gamma only — vanna regime interpretation differs)
if kind == "gamma":
    if net_total >= 0:
        st.markdown('<div class="regime-band regime-pos">🟢 POSITIVE GAMMA — dealers dampen moves. '
                    'Expect chop / mean reversion. Fading toward big nodes is the playbook; momentum chases get punished.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="regime-band regime-neg">🔴 NEGATIVE GAMMA — dealers amplify moves. '
                    'Trend/momentum regime. Breakouts can run; fading is dangerous. Watch the flip level.</div>',
                    unsafe_allow_html=True)
else:
    st.markdown('<div class="regime-band regime-pos" style="background:rgba(122,162,255,0.10);'
                'border-color:rgba(122,162,255,0.35);color:#7aa2ff;">🔵 VANNA — measures how dealer '
                'delta shifts as IV moves. Big negative-vanna strikes below spot = vol-up selling pressure; '
                'positive above = supportive on a vol crush. Strongest signal into OpEx / vol regime changes.</div>',
                unsafe_allow_html=True)

# --------------------------- strike ladder (screenshot style) -------------
st.subheader(f"{MLAB} strike ladder")
st.caption("Green = positive exposure · Pink = negative · ★ King (largest node) · white pill = spot")
render_ladder(g, spot, MLAB)

# ------------------------- TODAY'S SIGNAL (gamma regime only) --------------
if kind == "gamma":
    NODE_DIST = 0.004
    dist = (king - spot) / spot
    sig, sig_cls, why = "NO TRADE — SIT OUT", "neu", ""
    if net_total >= 0:
        if dist >= NODE_DIST:
            sig, sig_cls = "🟢 BUY CALLS", "pos"
            why = (f"Calm market (positive gamma). Price is below the big magnet at ${king:,.0f}. "
                   f"The market tends to drift toward it. Target: ${king:,.0f}.")
        elif dist <= -NODE_DIST:
            sig, sig_cls = "🔴 BUY PUTS", "neg"
            why = (f"Calm market (positive gamma). Price is above the big magnet at ${king:,.0f}. "
                   f"The market tends to drift back down to it. Target: ${king:,.0f}.")
        else:
            why = (f"Price is already sitting on the magnet (${king:,.0f}). "
                   "It will likely chop sideways — no edge, save your money for a better day.")
    else:
        if flip and spot < flip:
            sig, sig_cls = "🔴 BUY PUTS", "neg"
            why = (f"Wild market (negative gamma) and price is below the flip level ${flip:,.0f}. "
                   "Moves down tend to keep going. Ride the momentum, don't fight it.")
        else:
            why = ("Wild market (negative gamma) with no clear level to lean on. "
                   "This is when accounts get hurt — sitting out IS the smart trade today.")

    st.markdown(f"""
    <div class="metric-card" style="margin:4px 0 14px 0; border-width:2px;">
      <div class="metric-label">Today's signal — rules-based, not a guarantee</div>
      <div class="metric-value {sig_cls}" style="font-size:2rem;">{sig}</div>
      <div class="metric-sub" style="color:#aab2c8; font-size:0.9rem;">{why}</div>
      <div class="metric-sub" style="color:#6b7490;">Rules: risk max 12% of account · stop at −50% of premium ·
      take profit at +100% or when the magnet price is tagged · one trade per day · signal is stale after ~10 AM.</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- expected move & overnight OI shifts ----------------
import os


def _mid(df, k):
    row = df[df["strike"] == k]
    if row.empty:
        return None
    b = float(row["bid"].iloc[0] or 0)
    a = float(row["ask"].iloc[0] or 0)
    l = float(row["lastPrice"].iloc[0] or 0)
    return (b + a) / 2 if b > 0 and a > 0 else (l or None)


em_txt = "Quotes unavailable right now."
try:
    atm = float(min(calls["strike"], key=lambda x: abs(x - spot)))
    cm, pm = _mid(calls, atm), _mid(puts, atm)
    if cm and pm:
        em = cm + pm
        lo_em, hi_em = spot - em, spot + em
        reach = ("king node WITHIN reach ✓" if lo_em <= king <= hi_em
                 else "king node OUTSIDE expected move — magnet may not get hit, lower confidence")
        em_txt = (f"±${em:.2f} (±{em/spot*100:.2f}%) by {expiration} → "
                  f"range ${lo_em:,.0f}–${hi_em:,.0f} · {reach}")
except Exception:
    pass

oi_txt = "First snapshot saved today — overnight changes will appear from tomorrow."
try:
    snap_path = "oi_snapshots.csv"
    today_s = str(date.today())
    cur = pd.concat([
        calls[["strike", "openInterest"]].assign(side="C"),
        puts[["strike", "openInterest"]].assign(side="P"),
    ]).assign(date=today_s, exp=expiration)
    cur["openInterest"] = cur["openInterest"].fillna(0)
    hist_s = pd.read_csv(snap_path) if os.path.exists(snap_path) else pd.DataFrame(columns=cur.columns)
    already = (((hist_s["date"] == today_s) & (hist_s["exp"] == expiration)).any()
               if not hist_s.empty else False)
    if not already:
        pd.concat([hist_s, cur]).to_csv(snap_path, index=False)
    if not hist_s.empty:
        prior = sorted(hist_s.loc[(hist_s["exp"] == expiration) & (hist_s["date"] != today_s), "date"].unique())
        if prior:
            prev = hist_s[(hist_s["date"] == prior[-1]) & (hist_s["exp"] == expiration)]
            m = cur.merge(prev, on=["strike", "side"], suffixes=("", "_prev"))
            m["chg"] = m["openInterest"] - m["openInterest_prev"]
            top = m.reindex(m["chg"].abs().sort_values(ascending=False).index).head(3)
            top = top[top["chg"].abs() > 0]
            if not top.empty:
                bits = [f"{'+' if r.chg >= 0 else ''}{int(r.chg):,} "
                        f"{'calls' if r.side == 'C' else 'puts'} @ ${r.strike:,.0f}"
                        for r in top.itertuples()]
                oi_txt = f"Biggest positioning shifts since {prior[-1]}: " + " · ".join(bits)
            else:
                oi_txt = f"No meaningful OI changes vs {prior[-1]}."
except Exception:
    oi_txt = "OI tracker will start recording from the next market day."

st.markdown(f"""
<div class="metric-card" style="margin-bottom:10px;">
  <div class="metric-label">Expected move — the market's own forecast</div>
  <div class="metric-sub" style="color:#e8eaf0; font-size:0.95rem;">{em_txt}</div>
  <div class="metric-label" style="margin-top:10px;">Overnight smart-money positioning (OI shifts)</div>
  <div class="metric-sub" style="color:#aab2c8; font-size:0.9rem;">{oi_txt}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------- plotly bar chart ----------------------------
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(y=g["strike"], x=g["call_ex"], orientation="h", name=f"Call {MLAB}",
                     marker_color="#35d49a", opacity=0.9))
fig.add_trace(go.Bar(y=g["strike"], x=g["put_ex"], orientation="h", name=f"Put {MLAB}",
                     marker_color="#ff5c7a", opacity=0.9))
fig.add_trace(go.Scatter(y=g["strike"], x=g["net_ex"], mode="lines", name=f"Net {MLAB}",
                         line=dict(color="#f5b64c", width=2)))
fig.add_hline(y=spot, line_color="#e8eaf0", line_dash="dot",
              annotation_text=f"SPOT ${spot:,.2f}", annotation_font_color="#e8eaf0")
if flip:
    fig.add_hline(y=flip, line_color="#f5b64c", line_dash="dash",
                  annotation_text=f"FLIP ${flip:,.0f}", annotation_font_color="#f5b64c")
fig.add_hline(y=king, line_color="#7aa2ff", line_dash="dash",
              annotation_text=f"KING ${king:,.0f}", annotation_font_color="#7aa2ff")
fig.update_layout(
    barmode="relative", template="plotly_dark", height=650,
    paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.05),
    xaxis_title=METRICS[kind]["axis"],
    yaxis_title="Strike",
    font=dict(family="Inter, sans-serif", color="#aab2c8"),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- multi-expiration heatmap grid ----------------------
st.subheader("Multi-expiration grid")
st.caption("Green = positive exposure walls · Red = negative walls · Brightness = size")
n_exps = st.slider("Expirations to show", 3, 8, 5)
try:
    spot_g, grid = load_grid(ticker, n_exps, kind)
    if grid.empty:
        st.info("Grid data not available right now.")
    else:
        glo, ghi = spot_g * (1 - window / 100), spot_g * (1 + window / 100)
        gm = grid[(grid.index >= glo) & (grid.index <= ghi)]
        if gm.empty:
            gm = grid
        heat = go.Figure(go.Heatmap(
            z=gm.values, x=list(gm.columns), y=list(gm.index),
            colorscale=[[0.0, "#ff2d55"], [0.5, "#0b0e14"], [1.0, "#35d49a"]],
            zmid=0, showscale=False,
            hovertemplate="Strike $%{y}<br>Exp %{x}<br>Net " + MLAB + " $%{z:,.0f}<extra></extra>"))
        heat.add_hline(y=spot_g, line_color="#e8eaf0", line_dash="dot",
                       annotation_text=f"SPOT ${spot_g:,.2f}", annotation_font_color="#e8eaf0")
        heat.update_layout(template="plotly_dark", height=560,
                           paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                           margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title="Expiration", yaxis_title="Strike",
                           font=dict(family="Inter, sans-serif", color="#aab2c8"))
        st.plotly_chart(heat, use_container_width=True)
except Exception as e:
    st.info(f"Grid unavailable: {e}")

with st.expander("How to read this"):
    st.markdown("""
- **King node (blue):** largest exposure wall — price tends to gravitate toward it into expiry, especially on 0DTE afternoons ("pinning").
- **Flip (yellow):** above it, dealers stabilize price (fade extremes); below it, they accelerate moves (trade momentum).
- **Vanna view:** shows where dealer delta will shift as IV moves — most useful into OpEx and vol regime changes, not for intraday pinning.
- **Air pockets:** strike zones with tiny bars — little dealer hedging there, so price moves *fast* through them.
- **Limitations:** open interest updates once per day pre-market, so intraday flow (new 0DTE positioning) is invisible here. That's the part paid "velocity" feeds add.
- This is a market-structure map, **not** a buy/sell signal.
""")
