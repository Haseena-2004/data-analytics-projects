# Mobile Game Player Engagement & Retention Analysis (SQL + Python)

Product-analytics project: retention, engagement segmentation, monetization and hypothesis testing on game telemetry stored in SQLite.

## Data
**Synthetic** (no real player data): 8,000 players installed Jan-Feb 2025, ~132K sessions and ~1.4K purchases through 30 Apr 2025. Built by `generate_data.py` with a hidden engagement variable that drives play time, retention and spend. **Findings describe this simulated data, not a real game.**

Tables: `players(player_id, install_date, platform, channel, country)`, `sessions(player_id, session_date, duration_sec)`, `purchases(player_id, purchase_date, amount_usd)`.

## SQL (in `sql/`)
| File | What it does | Techniques |
|---|---|---|
| `01_retention.sql` | D1/D7/D14 retention by acquisition channel | CTEs, conditional aggregation, joins |
| `02_cohort_curve.sql` | Retention curve by install-week cohort | CTEs, joins, cohort sizing |
| `03_engagement_segments.sql` | Quartiles of first-week playtime vs week-4 activity, payer rate, ARPU | `NTILE` window function, multi-CTE |
| `04_daily_kpis.sql` | DAU, revenue, ARPDAU, 7-day moving average | Window frame `ROWS BETWEEN` |

Only players with a complete observation window are counted for each metric.

## Python (`analysis.py`)
Runs the SQL, saves CSVs and charts, and tests two hypotheses: a two-proportion z-test on D7 retention between channels, and a point-biserial correlation between first-week minutes and week-4 activity. It also flags anomalous DAU days using residuals from the 7-day average.

## Results (synthetic data)
- D7 retention: referral 32.6%, paid social 24.2%, organic 23.2%, ads network 21.3%.
- Referral vs ads network D7: +11.3 percentage points (95% CI 7.6 to 15.0), z = 6.34, p < 0.001.
- Players in the top first-week playtime quartile were active in week 4 at 38.8% vs 13.2% in the bottom quartile; payer rate 31.1% vs 4.5%; ARPU $1.60 vs $0.19.
- First-week minutes vs week-4 activity: r = 0.22 (n = 8,000, p < 0.001).
- No anomalous DAU days (|z| > 3).

## Limitations and assumptions
- Relationships exist because the simulation built them in; the value here is the method and SQL, not the numbers.
- Correlation between early engagement and retention is not causal.
- Observational channel comparison; no randomized experiment or multiple-test correction.

## Run
```
pip install -r requirements.txt
python generate_data.py
python analysis.py
```
Charts and CSVs are written to `results/`.
