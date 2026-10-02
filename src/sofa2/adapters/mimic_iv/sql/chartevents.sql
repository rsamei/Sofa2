-- Charted items (vital signs, FiO2, weight, GCS, respiratory devices, ECMO, MCS, RRT).
SELECT ce.stay_id, ce.charttime, ce.itemid, ce.value, ce.valuenum
FROM {icu}.chartevents ce
JOIN {icu}.icustays ie ON ie.stay_id = ce.stay_id
WHERE ce.itemid IN ({items})
  AND ce.charttime IS NOT NULL
{stay_filter}
