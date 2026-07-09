"""
GEX Radar — free SPY gamma exposure dashboard
Data: Yahoo Finance option chains (free, delayed ~15 min; OI updates once daily pre-market)
Run locally:   streamlit run gex_dashboard.py
Deploy free:   push to GitHub -> share.streamlit.io -> get a URL you can open on your phone
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


def compute_gex(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t_years: float) -> pd.DataFrame:
    """Per-strike dealer gamma exposure, standard convention:
    calls positive, puts negative. GEX = spot^2 * gamma * OI * 100 * 0.01
    (dollar exposure per 1% move)."""
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
            g = bs_gamma(spot, k, t_years, iv)
            gex = sign * spot * spot * 0.01 * g * oi * 100
            entry = rows.setdefault(k, {"strike": k, "call_gex": 0.0, "put_gex": 0.0})
            if sign > 0:
                entry["call_gex"] += gex
            else:
                entry["put_gex"] += gex
    out = pd.DataFrame(rows.values()).sort_values("strike").reset_index(drop=True)
    if not out.empty:
        out["net_gex"] = out["call_gex"] + out["put_gex"]
        out["abs_gex"] = out["net_gex"].abs()
    return out


def find_flip(gex: pd.DataFrame):
    """Strike where cumulative net GEX crosses zero (gamma flip)."""
    if gex.empty:
        return None
    cum = gex["net_gex"].cumsum()
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
    close_utc = 20  # 4pm ET ~ 20:00/21:00 UTC depending on DST; close enough for gamma
    frac_today = max((close_utc - now.hour - now.minute / 60) / 6.5 / 24, 0.03 / 24 * 6.5)
    t = max(days, 0) / 365.0 + (frac_today / 365.0 if days == 0 else 0)
    return max(t, 0.0008)  # never zero


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
def load_grid(ticker: str, n_exps: int = 5):
    """Net GEX per strike across the next n expirations (for the heatmap grid)."""
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
        g = compute_gex(ch.calls, ch.puts, spot_g, t)
        if not g.empty:
            cols[exp] = g.set_index("strike")["net_gex"]
    return spot_g, pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------- UI -------

st.set_page_config(page_title="GEX Radar", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0b0e14; }
  html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
  h1 { color: #e8eaf0 !important; font-weight: 800; letter-spacing: -0.02em; }
  .metric-card {
    background: #131826; border: 1px solid #1f2637; border-radius: 14px;
    padding: 14px 18px; text-align: left;
  }
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

st.title("🎯 GEX Radar")
st.caption("Dealer gamma exposure from free Yahoo option data · OI refreshes once daily pre-market · prices delayed ~15 min")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
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
    st.caption("GEX = S² × 0.01 × Γ × OI × 100. Calls +, puts − (standard dealer-positioning assumption).")

try:
    spot, calls, puts = load_chain(ticker, expiration)
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

if not spot:
    st.error("Couldn't get spot price.")
    st.stop()

t_years = years_to_expiry(expiration)
gex = compute_gex(calls, puts, spot, t_years)
if gex.empty:
    st.warning("No usable open interest for this expiration yet (0DTE OI posts pre-market).")
    st.stop()

lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
g = gex[(gex["strike"] >= lo) & (gex["strike"] <= hi)].copy()
if g.empty:
    g = gex.copy()

net_total = g["net_gex"].sum()
flip = find_flip(g)
king_row = g.loc[g["abs_gex"].idxmax()]
king = float(king_row["strike"])

def fmt_b(x):
    ax = abs(x)
    if ax >= 1e9: return f"${x/1e9:,.2f}B"
    if ax >= 1e6: return f"${x/1e6:,.1f}M"
    return f"${x:,.0f}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{ticker} spot</div>'
                f'<div class="metric-value">${spot:,.2f}</div>'
                f'<div class="metric-sub neu">{expiration}</div></div>', unsafe_allow_html=True)
with c2:
    cls = "pos" if net_total >= 0 else "neg"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Net GEX (window)</div>'
                f'<div class="metric-value {cls}">{fmt_b(net_total)}</div>'
                f'<div class="metric-sub {cls}">{"dealers long gamma" if net_total>=0 else "dealers short gamma"}</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Gamma flip</div>'
                f'<div class="metric-value">{f"${flip:,.0f}" if flip else "—"}</div>'
                f'<div class="metric-sub neu">regime changes here</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">King node</div>'
                f'<div class="metric-value">${king:,.0f}</div>'
                f'<div class="metric-sub neu">{fmt_b(float(king_row["net_gex"]))} · price magnet</div></div>',
                unsafe_allow_html=True)

if net_total >= 0:
    st.markdown('<div class="regime-band regime-pos">🟢 POSITIVE GAMMA — dealers dampen moves. '
                'Expect chop / mean reversion. Fading moves toward big nodes is the playbook; momentum chases get punished.</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="regime-band regime-neg">🔴 NEGATIVE GAMMA — dealers amplify moves. '
                'Trend/momentum regime. Breakouts can run; fading is dangerous. Watch the flip level.</div>',
                unsafe_allow_html=True)

# ------------------------- TODAY'S SIGNAL (same rules as the paper bot) ----
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

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(y=g["strike"], x=g["call_gex"], orientation="h", name="Call GEX",
                     marker_color="#35d49a", opacity=0.9))
fig.add_trace(go.Bar(y=g["strike"], x=g["put_gex"], orientation="h", name="Put GEX",
                     marker_color="#ff5c7a", opacity=0.9))
fig.add_trace(go.Scatter(y=g["strike"], x=g["net_gex"], mode="lines", name="Net GEX",
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
    xaxis_title="Dealer gamma exposure per 1% move ($)",
    yaxis_title="Strike",
    font=dict(family="Inter, sans-serif", color="#aab2c8"),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- multi-expiration heatmap grid ----------------------
st.subheader("Multi-expiration grid")
st.caption("Green = call walls (magnets / speed bumps) · Red = put walls (trapdoors) · Brightness = size")
n_exps = st.slider("Expirations to show", 3, 8, 5)
try:
    spot_g, grid = load_grid(ticker, n_exps)
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
            hovertemplate="Strike $%{y}<br>Exp %{x}<br>Net GEX $%{z:,.0f}<extra></extra>"))
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
- **King node (blue):** largest gamma wall — price tends to gravitate toward it into expiry, especially on 0DTE afternoons ("pinning").
- **Flip (yellow):** above it, dealers stabilize price (fade extremes); below it, they accelerate moves (trade momentum).
- **Air pockets:** strike zones with tiny bars — little dealer hedging there, so price moves *fast* through them.
- **Limitations:** open interest updates once per day pre-market, so intraday flow (new 0DTE positioning) is invisible here. That's the part paid "velocity" feeds add.
- This is a market-structure map, **not** a buy/sell signal.
""")
