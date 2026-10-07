-- Segment players by first-7-day minutes played (NTILE window function); compare week-4 activity and spend.
WITH first7 AS (
    SELECT p.player_id, p.install_date,
           COUNT(s.session_date)                       AS sessions_7d,
           COUNT(DISTINCT s.session_date)              AS active_days_7d,
           COALESCE(SUM(s.duration_sec), 0) / 60.0     AS minutes_7d
    FROM players p
    LEFT JOIN sessions s ON s.player_id = p.player_id
         AND julianday(s.session_date) - julianday(p.install_date) BETWEEN 0 AND 6
    WHERE julianday('2025-04-30') - julianday(p.install_date) >= 30
    GROUP BY p.player_id
),
ranked AS (
    SELECT *, NTILE(4) OVER (ORDER BY minutes_7d) AS quartile FROM first7
),
late AS (
    SELECT r.player_id,
           MAX(CASE WHEN julianday(s.session_date) - julianday(r.install_date) BETWEEN 21 AND 29 THEN 1 ELSE 0 END) AS active_wk4
    FROM ranked r LEFT JOIN sessions s ON s.player_id = r.player_id
    GROUP BY r.player_id
),
spend AS (
    SELECT player_id, SUM(amount_usd) AS revenue FROM purchases GROUP BY player_id
)
SELECT r.quartile,
       COUNT(*)                                           AS players,
       ROUND(AVG(r.minutes_7d), 1)                        AS avg_minutes_7d,
       ROUND(100.0 * AVG(l.active_wk4), 1)                AS active_in_week4_pct,
       ROUND(100.0 * AVG(CASE WHEN sp.revenue > 0 THEN 1 ELSE 0 END), 1) AS payer_pct,
       ROUND(AVG(COALESCE(sp.revenue, 0)), 2)             AS arpu_usd
FROM ranked r JOIN late l USING (player_id) LEFT JOIN spend sp USING (player_id)
GROUP BY r.quartile ORDER BY r.quartile;
