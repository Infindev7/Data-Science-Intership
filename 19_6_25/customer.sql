SELECT mode, COUNT(amount) AS total
FROM payment

GROUP BY mode

HAVING COUNT(amount) >= 2 and count(amount) < 4
order by total asc;