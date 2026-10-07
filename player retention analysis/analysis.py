"""Run the SQL files against game.db, test hypotheses in Python, and save charts + results."""
import sqlite3, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
con = sqlite3.connect("game.db")
q = lambda f: pd.read_sql(open(f"sql/{f}").read(), con)

ret = q("01_retention.sql"); print("Retention by channel\n", ret.to_string(index=False))
curve = q("02_cohort_curve.sql")
seg = q("03_engagement_segments.sql"); print("\nEngagement quartiles\n", seg.to_string(index=False))
kpi = q("04_daily_kpis.sql")
for n, d in [("retention_by_channel", ret), ("engagement_segments", seg), ("daily_kpis", kpi)]:
    d.to_csv(f"results/{n}.csv", index=False)

# Hypothesis 1: paid_social vs referral D7 retention (two-proportion z-test, player level)
flags = pd.read_sql("""
WITH d AS (SELECT DISTINCT player_id, session_date FROM sessions)
SELECT p.player_id, p.channel,
       MAX(CASE WHEN julianday(d.session_date)-julianday(p.install_date)=7 THEN 1 ELSE 0 END) AS d7
FROM players p LEFT JOIN d ON d.player_id=p.player_id
WHERE julianday('2025-04-30')-julianday(p.install_date) >= 14 GROUP BY p.player_id""", con)
a, b = (flags[flags.channel == c].d7 for c in ("referral", "ads_network"))
p_pool = (a.sum() + b.sum()) / (len(a) + len(b))
z = (a.mean() - b.mean()) / np.sqrt(p_pool * (1 - p_pool) * (1/len(a) + 1/len(b)))
pval = 2 * (1 - stats.norm.cdf(abs(z)))
diff = a.mean() - b.mean(); se = np.sqrt(a.mean()*(1-a.mean())/len(a) + b.mean()*(1-b.mean())/len(b))
test = dict(test="two-proportion z-test D7 retention: referral vs ads_network", referral=float(a.mean()), ads_network=float(b.mean()),
            diff=float(diff), ci95=[float(diff-1.96*se), float(diff+1.96*se)], z=float(z), p_value=float(pval))
print("\nHypothesis test:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()})

# Hypothesis 2: first-week minutes vs week-4 activity (point-biserial correlation)
f7 = pd.read_sql("""SELECT p.player_id, COALESCE(SUM(s.duration_sec),0)/60.0 AS m7,
  (SELECT COUNT(*) FROM sessions s2 WHERE s2.player_id=p.player_id AND julianday(s2.session_date)-julianday(p.install_date) BETWEEN 21 AND 29)>0 AS wk4
  FROM players p LEFT JOIN sessions s ON s.player_id=p.player_id AND julianday(s.session_date)-julianday(p.install_date) BETWEEN 0 AND 6
  WHERE julianday('2025-04-30')-julianday(p.install_date)>=30 GROUP BY p.player_id""", con)
rho, p2 = stats.pointbiserialr(f7.wk4.astype(int), f7.m7)
corr = dict(r=float(rho), p_value=float(p2), n=int(len(f7))); print("First-week minutes vs week-4 activity:", {k: round(v, 4) for k, v in corr.items()})

# Anomaly check: days where DAU deviates > 3 std from 7-day average (rolling z-score on residuals)
k = kpi.copy(); k["resid"] = k.dau - k.dau_7d_avg
k["z"] = (k.resid - k.resid.mean()) / k.resid.std()
anom = k[k.z.abs() > 3][["day", "dau", "dau_7d_avg", "z"]]
print("Anomalous days (|z|>3):", len(anom))

# Charts
fig, ax = plt.subplots(figsize=(7, 4.5))
for wk, g in curve.groupby("install_week"):
    ax.plot(g.day_n, g.retention_pct, alpha=.6, label=wk)
ax.set_xlabel("Days since install"); ax.set_ylabel("% of cohort active"); ax.set_title("Retention curves by install week")
ax.legend(fontsize=6, ncol=3); plt.tight_layout(); plt.savefig("results/retention_curves.png", dpi=130); plt.close()
ret.set_index("channel")[["d1_pct", "d7_pct", "d14_pct"]].plot.bar(figsize=(7, 4.5), title="Retention by acquisition channel", rot=0)
plt.ylabel("%"); plt.tight_layout(); plt.savefig("results/retention_by_channel.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(seg.quartile.astype(str), seg.active_in_week4_pct); ax[0].set_title("Week-4 activity by week-1 engagement quartile"); ax[0].set_xlabel("quartile (4 = most engaged)")
ax[1].bar(seg.quartile.astype(str), seg.arpu_usd, color="tab:green"); ax[1].set_title("ARPU (USD) by quartile"); ax[1].set_xlabel("quartile")
plt.tight_layout(); plt.savefig("results/engagement_segments.png", dpi=130); plt.close()
fig, ax = plt.subplots(figsize=(9, 3.8)); ax.plot(pd.to_datetime(kpi.day), kpi.dau, lw=.8, label="DAU"); ax.plot(pd.to_datetime(kpi.day), kpi.dau_7d_avg, label="7-day avg")
ax.legend(); ax.set_title("Daily active users"); plt.tight_layout(); plt.savefig("results/dau.png", dpi=130); plt.close()

json.dump(dict(data="synthetic", players=int(con.execute("select count(*) from players").fetchone()[0]),
               channel_test=test, engagement_vs_week4=corr, anomalous_days=int(len(anom))), open("results/metrics.json", "w"), indent=2)
