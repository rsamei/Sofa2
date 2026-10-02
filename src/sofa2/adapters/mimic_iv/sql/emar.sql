-- Delirium drugs given during the ICU stay.
SELECT ie.stay_id, e.charttime, e.medication, e.event_txt
FROM {hosp}.emar e
JOIN {icu}.icustays ie ON ie.hadm_id = e.hadm_id
WHERE e.charttime >= ie.intime
  AND e.charttime <= ie.outtime
  AND ({medication_like})
  AND e.event_txt IN ({events})
{stay_filter}
