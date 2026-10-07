"""Generate a SYNTHETIC mobile-game dataset (players, sessions, purchases) into SQLite."""
import sqlite3, numpy as np, pandas as pd
r = np.random.default_rng(42)
N = 8000
start = pd.Timestamp("2025-01-01")
install = start + pd.to_timedelta(r.integers(0, 60, N), unit="D")
platform = r.choice(["iOS", "Android"], N, p=[.4, .6])
channel = r.choice(["organic", "paid_social", "referral", "ads_network"], N, p=[.4, .25, .1, .25])
country = r.choice(["IN", "US", "BR", "DE", "GB"], N, p=[.35, .25, .15, .13, .12])
# latent engagement drives sessions, retention and spend
eng = r.beta(2, 4, N) + np.where(channel == "referral", 0.1, 0) - np.where(channel == "ads_network", 0.05, 0)
eng = np.clip(eng, 0.02, 1)
players = pd.DataFrame(dict(player_id=range(1, N+1), install_date=install.strftime("%Y-%m-%d"),
                            platform=platform, channel=channel, country=country))
sess, buys = [], []
horizon = pd.Timestamp("2025-04-30")
for pid, inst, e in zip(players.player_id, install, eng):
    for d in range(0, 31):
        day = inst + pd.Timedelta(days=d)
        if day > horizon: break
        p_active = min(0.98, e * 1.5 * np.exp(-0.11 * d) + (0.9 if d == 0 else 0))
        if r.random() < p_active:
            n = 1 + r.poisson(1 + 3 * e)
            for _ in range(n):
                sess.append((pid, day.strftime("%Y-%m-%d"), int(max(30, r.normal(300 + 600 * e, 120)))))
            if r.random() < 0.015 + 0.06 * e * (d > 0):
                buys.append((pid, day.strftime("%Y-%m-%d"), float(r.choice([0.99, 1.99, 4.99, 9.99, 19.99], p=[.35, .25, .2, .15, .05]))))
con = sqlite3.connect("game.db"); players.to_sql("players", con, if_exists="replace", index=False)
pd.DataFrame(sess, columns=["player_id", "session_date", "duration_sec"]).to_sql("sessions", con, if_exists="replace", index=False)
pd.DataFrame(buys, columns=["player_id", "purchase_date", "amount_usd"]).to_sql("purchases", con, if_exists="replace", index=False)
con.commit(); print("players", len(players), "sessions", len(sess), "purchases", len(buys))
