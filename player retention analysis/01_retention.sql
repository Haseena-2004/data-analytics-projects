-- D1 / D7 / D14 retention by acquisition channel.
-- Only players whose observation window is complete (installed >= 14 days before data end) are counted.
WITH daily AS (
    SELECT DISTINCT player_id, session_date FROM sessions
),
flags AS (
    SELECT p.player_id, p.channel, p.install_date,
           MAX(CASE WHEN julianday(d.session_date) - julianday(p.install_date) = 1  THEN 1 ELSE 0 END) AS d1,
           MAX(CASE WHEN julianday(d.session_date) - julianday(p.install_date) = 7  THEN 1 ELSE 0 END) AS d7,
           MAX(CASE WHEN julianday(d.session_date) - julianday(p.install_date) = 14 THEN 1 ELSE 0 END) AS d14
    FROM players p LEFT JOIN daily d ON d.player_id = p.player_id
    GROUP BY p.player_id
)
SELECT channel,
       COUNT(*)                          AS installs,
       ROUND(100.0 * AVG(d1), 1)         AS d1_pct,
       ROUND(100.0 * AVG(d7), 1)         AS d7_pct,
       ROUND(100.0 * AVG(d14), 1)        AS d14_pct
FROM flags
WHERE julianday('2025-04-30') - julianday(install_date) >= 14
GROUP BY channel
ORDER BY d7_pct DESC;
