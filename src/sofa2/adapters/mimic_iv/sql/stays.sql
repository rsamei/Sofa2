-- ICU stays.
SELECT ie.stay_id, ie.subject_id AS patient_id, ie.hadm_id, ie.intime, ie.outtime
FROM {icu}.icustays ie
WHERE ie.intime IS NOT NULL AND ie.outtime IS NOT NULL AND ie.outtime > ie.intime
{stay_filter}
