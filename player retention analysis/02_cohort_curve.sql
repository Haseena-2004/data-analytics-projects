-- Retention curve: share of each install-week cohort active on day N (0-14).
WITH daily AS (SELECT DISTINCT player_id, session_date FROM sessions),
base AS (
    SELECT p.player_id, strftime('%Y-W%W', p.install_date) AS install_week, p.install_date
    FROM players p WHERE julianday('2025-04-30') - julianday(p.install_date) >= 14
),
activity AS (
    SELECT b.install_week, b.player_id,
           CAST(julianday(d.session_date) - julianday(b.install_date) AS INTEGER) AS day_n
    FROM base b JOIN daily d ON d.player_id = b.player_id
),
sizes AS (SELECT install_week, COUNT(*) AS cohort_size FROM base GROUP BY install_week)
SELECT a.install_week, a.day_n, s.cohort_size,
       ROUND(100.0 * COUNT(DISTINCT a.player_id) / s.cohort_size, 1) AS retention_pct
FROM activity a JOIN sizes s USING (install_week)
WHERE a.day_n BETWEEN 0 AND 14
GROUP BY a.install_week, a.day_n
ORDER BY a.install_week, a.day_n;
