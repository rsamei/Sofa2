-- Dialysis and ventilation procedures with start and end times.
SELECT pe.stay_id, pe.starttime, pe.endtime, pe.itemid
FROM {icu}.procedureevents pe
JOIN {icu}.icustays ie ON ie.stay_id = pe.stay_id
WHERE pe.itemid IN ({items})
  AND pe.value IS NOT NULL
  AND pe.starttime IS NOT NULL
{stay_filter}
