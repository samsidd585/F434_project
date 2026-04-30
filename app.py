import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="SPY/VOO Dislocation Explorer", layout="wide")

PLOTLY_TEMPLATE = "plotly_dark"
import plotly.io as pio
pio.templates.default = PLOTLY_TEMPLATE

# ---------- Data loading ----------

@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("America/New_York")
    df["MID_SPY"] = (df["bid_px_00_SPY"] + df["ask_px_00_SPY"]) / 2
    df["MID_VOO"] = (df["bid_px_00_VOO"] + df["ask_px_00_VOO"]) / 2
    df["date"] = df["ts_event"].dt.date
    return df.sort_values("ts_event").reset_index(drop=True)

@st.cache_data(show_spinner="Computing z-scores...")
def compute_signal(df: pd.DataFrame, win: int, thresh: float) -> pd.DataFrame:
    d = df[["ts_event", "MID_SPY", "MID_VOO", "date"]].copy()
    d["disc_ratio"] = d["MID_SPY"] / d["MID_VOO"]
    d["rolling_mean"] = d.groupby("date")["disc_ratio"].transform(lambda x: x.rolling(win).mean())
    d["rolling_std"] = d.groupby("date")["disc_ratio"].transform(lambda x: x.rolling(win).std())
    d["z_score"] = (d["disc_ratio"] - d["rolling_mean"]) / d["rolling_std"]
    out = d[["ts_event", "z_score"]].dropna().copy()
    out["signal"] = (out["z_score"].abs() > thresh).astype(int)
    return out

DATA_PATH = Path(__file__).parent / "data_xy_month.parquet"
if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH.name}. Place it next to app.py.")
    st.stop()

df = load_data(str(DATA_PATH))
ALL_DAYS = sorted(df["date"].unique())

# ---------- Helpers ----------

def day_picker(key_prefix: str, days: list):
    """Renders Prev / dropdown / Next on one row, returns the chosen day."""
    sel_key = f"{key_prefix}_select"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = days[0]

    col_prev, col_pick, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("\u25C0 Prev", key=f"{key_prefix}_prev", use_container_width=True):
            i = days.index(st.session_state[sel_key])
            st.session_state[sel_key] = days[max(0, i - 1)]
    with col_next:
        if st.button("Next \u25B6", key=f"{key_prefix}_next", use_container_width=True):
            i = days.index(st.session_state[sel_key])
            st.session_state[sel_key] = days[min(len(days) - 1, i + 1)]
    with col_pick:
        chosen = st.selectbox("Day", days, key=sel_key, label_visibility="collapsed")
    return chosen

# ---------- Tabs ----------

tab_about, tab_ba, tab_macro, tab_event, tab_stats = st.tabs(
    ["About", "Bid/Ask View", "Macro View", "Event Inspector", "Summary Stats"]
)

with tab_about:
    st.header("SPY / VOO Dislocation Explorer")

    st.markdown("""
### Project goal
Identify and characterize **dislocations** between two ETFs that track the same underlying
index using sub-sampled level-1 data. The project takes inspiration from Marshall, Nguyen &
Visaltanachoti (2013), and aims to empirically answer two questions: how often do dislocations
occur, and what does the order book look like when they do?

### Why SPY and VOO
Both ETFs track the S&P 500, and are highly liquid making them an ideal pair
for measuring price dislocations.

### Defining a dislocation
A *dislocation* is a moment when the Mid of SPY/Mid of VOO price ratio is meaningfully far from its
recent normal. We measure this with a **rolling z-score** on the mid-price ratio:

$$
\\text{ratio}_t = \\frac{\\text{mid}^{\\text{SPY}}_t}{\\text{mid}^{\\text{VOO}}_t}, \\qquad
z_t = \\frac{\\text{ratio}_t - \\mu_t}{\\sigma_t}
$$

where $\\mu_t$ and $\\sigma_t$ are the rolling mean and standard deviation over a configurable
window (default **600 seconds**). If $|z_t| > $ threshold (default **2.58**, ~99% of a normal),
that timestamp is flagged as a dislocation.

### Why a z-score instead of a fixed % threshold
The original paper defines dislocation as a tradable arbitrage opportunity (~0.2% after costs).
We use a z-score because:

- It is **normalized**, comparable across days with different volatility regimes.
- It captures *how unusual* a given gap is relative to its own recent history, rather than
  imposing a fixed dollar/percentage cutoff that may be too loose on calm days and too tight
  on volatile ones.
- The SPY/VOO ratio is **stationary** (confirmed by the ADF test, p approx 0), so a rolling
  mean and standard deviation are well-defined and the z-score is a valid measure of how far
  the ratio sits from its equilibrium.
- We chose this over **returns / percent-change**, which only measure how fast each ETF is
  moving. Two ETFs can move sharply together without dislocating. What we want to detect is
  the *gap* between them, and the z-score on the ratio measures that directly.

---

### Tab guide
- **Bid/Ask View** — top-of-book bid, ask, and mid for SPY and VOO over a chosen day.
  Sanity-checks the data and shows how each leg quotes throughout the session.
- **Macro View** — SPY and VOO mid-prices on twin axes for a given day, with red diamonds
  marking every dislocation. Sliders let you re-define the rolling window and threshold and
  watch the events update.
- **Event Inspector** — step through individual dislocations. Shows top-of-book imbalance
  for each leg, the spread, and a $\\pm$60s zoom of the price + z-score so you can see whether
  and how fast the ratio reverts.
- **Summary Stats** — aggregate findings across the month: event counts, reversion times,
  and time-of-day patterns.

### Data
One month of BBO-1s data (best-bid/best-offer at 1-second granularity) for SPY and VOO,
covering Oct 10 – Nov 10 2025, cleaned and time-aligned within a 1-second tolerance.
""")

with tab_ba:
    st.header("Bid / Ask / Mid")
    st.caption("Top-of-book bid (green), ask (red), and mid (dotted black) for each ETF across the chosen day.")

    day = day_picker("ba", ALL_DAYS)
    d = df[df["date"] == day]

    def ba_chart(d, sym, color_mid):
        bid = d[f"bid_px_00_{sym}"]; ask = d[f"ask_px_00_{sym}"]; mid = d[f"MID_{sym}"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["ts_event"], y=ask, name="ask",
                                 line=dict(color="#d62728", width=1)))
        fig.add_trace(go.Scatter(x=d["ts_event"], y=bid, name="bid",
                                 line=dict(color="#2ca02c", width=1)))
        fig.add_trace(go.Scatter(x=d["ts_event"], y=mid, name="mid",
                                 line=dict(color=color_mid, width=1.2, dash="dot")))
        fig.update_layout(
            title=f"{sym}",
            xaxis_title=None, yaxis_title="price",
            height=340, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    st.plotly_chart(ba_chart(d, "SPY", "#1f77b4"), use_container_width=True)
    st.plotly_chart(ba_chart(d, "VOO", "#ff7f0e"), use_container_width=True)

with tab_macro:
    st.header("Macro View")
    st.caption("SPY (left axis) and VOO (right axis) mid-prices over a chosen day. Red diamonds mark dislocation events under the current window/threshold.")

    c1, c2 = st.columns(2)
    with c1:
        win = st.slider("Rolling window (sec)", min_value=60, max_value=1800, value=600, step=60, key="macro_win")
    with c2:
        thresh = st.slider("|z| threshold", min_value=1.0, max_value=4.0, value=2.58, step=0.05, key="macro_thresh")

    sig = compute_signal(df, win, thresh)
    days_with_events = sorted(sig.loc[sig["signal"] == 1, "ts_event"].dt.date.unique())
    if not days_with_events:
        st.warning("No dislocations under current settings. Try a lower threshold.")
    else:
        day = day_picker("macro", days_with_events)
        d = df[df["date"] == day].copy()
        z_map = sig.set_index("ts_event")["z_score"]
        sig_map = sig.set_index("ts_event")["signal"]
        d["z_score"] = d["ts_event"].map(z_map)
        d["signal"] = d["ts_event"].map(sig_map).fillna(0).astype(int)
        events = d[d["signal"] == 1].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["ts_event"], y=d["MID_SPY"], name="SPY mid",
                                 line=dict(color="#1f77b4", width=1.2)))
        fig.add_trace(go.Scatter(x=d["ts_event"], y=d["MID_VOO"], name="VOO mid",
                                 line=dict(color="#ff7f0e", width=1.2), yaxis="y2"))
        fig.add_trace(go.Scatter(
            x=events["ts_event"], y=events["MID_SPY"], mode="markers", name="Dislocation",
            marker=dict(symbol="diamond", size=10, color="#d62728", line=dict(color="white", width=0.5)),
            hovertemplate="%{x}<br>SPY=%{y:.2f}<br>z=%{customdata:.2f}<extra></extra>",
            customdata=events["z_score"],
        ))
        spy_lo, spy_hi = d["MID_SPY"].min(), d["MID_SPY"].max()
        voo_lo, voo_hi = d["MID_VOO"].min(), d["MID_VOO"].max()
        spy_pad = (spy_hi - spy_lo) * 0.05 or 0.05
        voo_pad = (voo_hi - voo_lo) * 0.05 or 0.05

        fig.update_layout(
            title=f"{day}  |  win={win}s, thresh={thresh:.2f}  |  {len(events)} dislocations",
            xaxis_title="Time (ET)",
            yaxis=dict(title="SPY mid", side="left",
                       range=[spy_lo - spy_pad, spy_hi + spy_pad]),
            yaxis2=dict(title="VOO mid", overlaying="y", side="right",
                        range=[voo_lo - voo_pad, voo_hi + voo_pad]),
            height=560, hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_event:
    st.header("Event Inspector")
    st.caption("Step through individual dislocation events. See top-of-book state and z-score trajectory in a \u00B160s window.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ev_win = st.slider("Rolling window (sec)", 60, 1800, 600, 60, key="ev_win")
    with c2:
        ev_thresh = st.slider("|z| threshold", 1.0, 4.0, 2.58, 0.05, key="ev_thresh")
    with c3:
        zoom_sec = st.slider("Zoom window (\u00B1 sec)", 15, 300, 60, 15, key="ev_zoom")

    sig_e = compute_signal(df, ev_win, ev_thresh)
    z_map_e = sig_e.set_index("ts_event")["z_score"]
    sig_map_e = sig_e.set_index("ts_event")["signal"]
    events_df = df.copy()
    events_df["z_score"] = events_df["ts_event"].map(z_map_e).fillna(0)
    events_df["signal"] = events_df["ts_event"].map(sig_map_e).fillna(0).astype(int)
    events_df = events_df.sort_values("ts_event").reset_index(drop=True)
    event_idx = events_df.index[events_df["signal"] == 1].tolist()

    if not event_idx:
        st.warning("No events under current settings. Lower the threshold.")
    else:
        if "ev_i" not in st.session_state:
            st.session_state["ev_i"] = 0
        st.session_state["ev_i"] = min(st.session_state["ev_i"], len(event_idx) - 1)

        b1, b2, b3, b4 = st.columns([1, 1, 1, 4])
        with b1:
            if st.button("\u25C0 Prev", key="ev_prev", use_container_width=True):
                st.session_state["ev_i"] = max(0, st.session_state["ev_i"] - 1)
        with b2:
            if st.button("Next \u25B6", key="ev_next", use_container_width=True):
                st.session_state["ev_i"] = min(len(event_idx) - 1, st.session_state["ev_i"] + 1)
        with b3:
            if st.button("Random", key="ev_rand", use_container_width=True):
                st.session_state["ev_i"] = int(np.random.randint(0, len(event_idx)))
        with b4:
            st.markdown(f"**Event {st.session_state['ev_i']+1} / {len(event_idx)}**")

        row_idx = event_idx[st.session_state["ev_i"]]
        r = events_df.loc[row_idx]
        t = r["ts_event"]
        lo, hi = t - pd.Timedelta(seconds=zoom_sec), t + pd.Timedelta(seconds=zoom_sec)
        win_df = events_df[(events_df["ts_event"] >= lo) & (events_df["ts_event"] <= hi)]
        after = win_df[win_df["ts_event"] > t]
        rev = after[after["z_score"].abs() < 1]
        rev_sec = (rev["ts_event"].iloc[0] - t).total_seconds() if len(rev) > 0 else None
        rev_str = f"{rev_sec:.0f}s" if rev_sec is not None else f"no revert in \u00B1{zoom_sec}s"

        m1, m2, m3 = st.columns(3)
        m1.metric("Z-score", f"{r['z_score']:+.2f}")
        m2.metric("SPY/VOO ratio", f"{r['MID_SPY']/r['MID_VOO']:.5f}")
        m3.metric("Reversion to |z|<1", rev_str)

        def book_html(sym: str, r):
            bid_px = r[f"bid_px_00_{sym}"]; ask_px = r[f"ask_px_00_{sym}"]
            bid_sz = int(r[f"bid_sz_00_{sym}"]); ask_sz = int(r[f"ask_sz_00_{sym}"])
            spread = ask_px - bid_px
            tot = bid_sz + ask_sz
            imb = (bid_sz - ask_sz) / tot if tot > 0 else 0.0
            if imb > 0.2:
                tag = f"<span style='color:#2ca02c;font-weight:bold'>BID-heavy ({imb:+.0%})</span>"
            elif imb < -0.2:
                tag = f"<span style='color:#d62728;font-weight:bold'>ASK-heavy ({imb:+.0%})</span>"
            else:
                tag = f"<span style='color:#888'>balanced ({imb:+.0%})</span>"
            mx = max(bid_sz, ask_sz, 1)
            bid_w = int(100 * bid_sz / mx); ask_w = int(100 * ask_sz / mx)
            return f"""
            <div style='font-family:sans-serif; font-size:13px; margin-bottom:8px; color:#fafafa'>
              <b>{sym}</b> &nbsp; order book &nbsp;&mdash;&nbsp; {tag}
            </div>
            <table style='border-collapse:collapse; width:100%; font-family:monospace; font-size:13px; color:#fafafa'>
              <tr style='background:#2a1414'>
                <td style='padding:6px 10px; text-align:right; width:35%'></td>
                <td style='padding:6px 10px; text-align:center; color:#ff6b6b; font-weight:bold; width:30%'>
                  {ask_px:.2f}
                </td>
                <td style='padding:6px 10px; text-align:left; width:35%'>
                  <div style='display:inline-block; height:14px; width:{ask_w}%; background:#d62728; opacity:0.7'></div>
                  <span style='margin-left:8px'>{ask_sz:,}</span>
                </td>
              </tr>
              <tr style='background:#1a1d24'>
                <td colspan='3' style='padding:4px 10px; text-align:center; color:#888'>
                  &uarr; spread {spread:.2f} &darr;
                </td>
              </tr>
              <tr style='background:#142a14'>
                <td style='padding:6px 10px; text-align:right; width:35%'>
                  <span style='margin-right:8px'>{bid_sz:,}</span>
                  <div style='display:inline-block; height:14px; width:{bid_w}%; background:#2ca02c; opacity:0.7'></div>
                </td>
                <td style='padding:6px 10px; text-align:center; color:#5fd96b; font-weight:bold; width:30%'>
                  {bid_px:.2f}
                </td>
                <td style='padding:6px 10px; text-align:left'></td>
              </tr>
            </table>
            """

        bcol1, bcol2 = st.columns(2)
        bcol1.markdown(book_html("SPY", r), unsafe_allow_html=True)
        bcol2.markdown(book_html("VOO", r), unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:12px;color:#666;margin-top:6px'>Event @ {t}</div>", unsafe_allow_html=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.6, 0.4],
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=win_df["ts_event"], y=win_df["MID_SPY"],
                                 name="SPY mid", line=dict(color="#1f77b4")),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=win_df["ts_event"], y=win_df["MID_VOO"],
                                 name="VOO mid", line=dict(color="#ff7f0e")),
                      row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=[t], y=[r["MID_SPY"]], mode="markers",
                                 marker=dict(symbol="diamond", size=12, color="#d62728"),
                                 name="event"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=win_df["ts_event"], y=win_df["z_score"],
                                 name="z", line=dict(color="black")),
                      row=2, col=1)
        fig.add_hline(y=ev_thresh, line=dict(dash="dash", color="#E86A33"), row=2, col=1)
        fig.add_hline(y=-ev_thresh, line=dict(dash="dash", color="#E86A33"), row=2, col=1)
        fig.add_hline(y=1, line=dict(dash="dot", color="grey"), row=2, col=1)
        fig.add_hline(y=-1, line=dict(dash="dot", color="grey"), row=2, col=1)
        fig.add_vline(x=t, line=dict(color="#d62728", width=1))
        fig.update_yaxes(title_text="SPY mid", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="VOO mid", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="z-score", row=2, col=1)
        fig.update_layout(height=560, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.12),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab_stats:
    st.header("Summary Stats")
    st.write("Coming soon.")
