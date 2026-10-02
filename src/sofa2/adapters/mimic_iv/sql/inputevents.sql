-- Infusions and boluses of the mapped drugs.
SELECT iv.stay_id, iv.starttime, iv.endtime, iv.itemid, iv.rate, iv.rateuom, iv.amount,
       iv.patientweight
FROM {icu}.inputevents iv
JOIN {icu}.icustays ie ON ie.stay_id = iv.stay_id
WHERE iv.itemid IN ({items})
  AND iv.starttime IS NOT NULL
{stay_filter}
