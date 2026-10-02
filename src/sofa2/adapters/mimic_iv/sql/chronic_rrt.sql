-- Stays whose hospital admission carries a chronic dialysis diagnosis.
SELECT DISTINCT ie.stay_id
FROM {icu}.icustays ie
JOIN {hosp}.diagnoses_icd dx ON dx.hadm_id = ie.hadm_id
WHERE ((dx.icd_version = 9 AND dx.icd_code IN ({icd9}))
    OR (dx.icd_version = 10 AND dx.icd_code IN ({icd10})))
{stay_filter}
