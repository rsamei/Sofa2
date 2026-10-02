-- Urine output.
SELECT oe.stay_id, oe.charttime, oe.itemid, oe.value
FROM {icu}.outputevents oe
JOIN {icu}.icustays ie ON ie.stay_id = oe.stay_id
WHERE oe.itemid IN ({items})
  AND oe.value IS NOT NULL
{stay_filter}
