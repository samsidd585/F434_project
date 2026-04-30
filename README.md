# SPY/VOO Dislocation Explorer

Interactive Streamlit app for exploring price dislocations between SPY and VOO using high-frequency BBO-1s data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Data

Expects `data_xy_month.csv` in the same folder — one month (Oct 10 – Nov 10, 2025) of cleaned, time-aligned SPY/VOO top-of-book snapshots at 1-second granularity.

## Tabs

- **About** — methodology, formulas, how to read the app.
- **Bid/Ask View** — per-day bid, ask, mid lines for each ETF.
- **Macro View** — SPY/VOO mids on twin axes with dislocation markers; window/threshold sliders.
- **Event Inspector** — step through individual dislocation events; orderbook snapshot + zoom plot.
- **Summary Stats** — aggregate event statistics over the month.
