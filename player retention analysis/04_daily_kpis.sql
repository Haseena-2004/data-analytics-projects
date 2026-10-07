-- Daily active users, revenue, ARPDAU and a 7-day moving average (window function) for anomaly checks.
WITH dau AS (SELECT session_date AS day, COUNT(DISTINCT player_id) AS dau FROM sessions GROUP BY session_date),
rev AS (SELECT purchase_date AS day, SUM(amount_usd) AS revenue FROM purchases GROUP BY purchase_date)
SELECT d.day, d.dau, ROUND(COALESCE(r.revenue, 0), 2) AS revenue,
       ROUND(COALESCE(r.revenue, 0) / d.dau, 4) AS arpdau,
       ROUND(AVG(d.dau) OVER (ORDER BY d.day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS dau_7d_avg
FROM dau d LEFT JOIN rev r USING (day)
ORDER BY d.day;
