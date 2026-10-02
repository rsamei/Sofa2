-- Laboratory values from the pre-ICU lookback to ICU discharge, with the blood-gas specimen type.
WITH spec AS (
    SELECT specimen_id, MAX(value) AS specimen_type
    FROM {hosp}.labevents
    WHERE itemid = {specimen_item}
    GROUP BY specimen_id
)
SELECT ie.stay_id, le.charttime, le.itemid, le.valuenum, le.valueuom, le.specimen_id,
       spec.specimen_type
FROM {hosp}.labevents le
JOIN {icu}.icustays ie ON ie.subject_id = le.subject_id
LEFT JOIN spec ON spec.specimen_id = le.specimen_id
WHERE le.itemid IN ({items})
  AND le.valuenum IS NOT NULL
  AND le.charttime >= {lab_start}
  AND le.charttime <= ie.outtime
{stay_filter}
