"""Write tiny made-up tables in the MIMIC-IV layout (no patient data) for adapter tests.

Run from the repository root: ``python tests/data/make_mimic_synthetic.py``.
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "mimic_synthetic"
T0 = pd.Timestamp("2150-01-01 00:00")


def t(h):
    return (T0 + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")


icustays = pd.DataFrame([
    dict(subject_id=10, hadm_id=100, stay_id=1000, first_careunit="MICU", last_careunit="MICU",
         intime=t(0), outtime=t(48), los=2.0),
    dict(subject_id=20, hadm_id=200, stay_id=2000, first_careunit="CVICU", last_careunit="CVICU",
         intime=t(0), outtime=t(24), los=1.0),
])

ce = []


def chart(stay, subj, h, item, value, valuenum=None):
    ce.append(dict(subject_id=subj, hadm_id=subj * 10, stay_id=stay, caregiver_id=1, charttime=t(h),
                   storetime=t(h), itemid=item, value=str(value),
                   valuenum=valuenum if valuenum is not None else (value if isinstance(value, (int, float)) else None),
                   valueuom=None, warning=0))


# stay 1000: ventilated (ETT + vent mode charts 0-20 h), then nasal cannula; GCS with ETT
for h in range(0, 21, 4):
    chart(1000, 10, h, 226732, "Endotracheal tube")
    chart(1000, 10, h + 0.1, 223849, "CMV/ASSIST/AutoFlow")
chart(1000, 10, 24, 226732, "Nasal cannula")
chart(1000, 10, 30, 226732, "Nasal cannula")
chart(1000, 10, 1, 223835, 50)          # FiO2 as %
chart(1000, 10, 0.5, 226512, 80)        # admission weight
chart(1000, 10, 2, 220052, 64)          # MAP
chart(1000, 10, 2, 220739, 1)
chart(1000, 10, 2, 223900, "No Response-ETT", 1)
chart(1000, 10, 2, 223901, 4)           # motor 4 -> SOFA-2 brain 2
chart(1000, 10, 30, 220181, 75)
chart(1000, 10, 30, 220277, 95)
# stay 2000: IABP running, VA ECMO configuration, CRRT chart, BiPAP mask with trailing space
chart(2000, 20, 3, 224322, 70)
chart(2000, 20, 4, 229268, "VA")
chart(2000, 20, 5, 224144, 200)         # CRRT blood flow
chart(2000, 20, 6, 226732, "Bipap mask ")
chart(2000, 20, 7, 226732, "Bipap mask ")
chart(2000, 20, 8, 223835, 0.6)         # FiO2 as a fraction
chart(2000, 20, 8.5, 220277, 90)        # SpO2 -> S/F 150
chart(2000, 20, 2, 220739, 4)
chart(2000, 20, 2, 223900, 5)
chart(2000, 20, 2, 223901, 6)
chartevents = pd.DataFrame(ce)

labevents = pd.DataFrame([
    # stay 1000: arterial blood gas PaO2 70 with FiO2 50% in the same specimen -> PF 140
    dict(labevent_id=1, subject_id=10, hadm_id=100, specimen_id=501, itemid=52033, charttime=t(2), value="ART.", valuenum=None, valueuom=None),
    dict(labevent_id=2, subject_id=10, hadm_id=100, specimen_id=501, itemid=50821, charttime=t(2), value="70", valuenum=70, valueuom="mm Hg"),
    dict(labevent_id=3, subject_id=10, hadm_id=100, specimen_id=501, itemid=50816, charttime=t(2), value="50", valuenum=50, valueuom="%"),
    # venous PO2 must not be used
    dict(labevent_id=4, subject_id=10, hadm_id=100, specimen_id=502, itemid=52033, charttime=t(3), value="VEN.", valuenum=None, valueuom=None),
    dict(labevent_id=5, subject_id=10, hadm_id=100, specimen_id=502, itemid=50821, charttime=t(3), value="30", valuenum=30, valueuom="mm Hg"),
    # creatinine 4 h before ICU (counts on day 1) and bilirubin 10 h before (does not)
    dict(labevent_id=6, subject_id=10, hadm_id=None, specimen_id=503, itemid=50912, charttime=t(-4), value="2.5", valuenum=2.5, valueuom="mg/dL"),
    dict(labevent_id=7, subject_id=10, hadm_id=100, specimen_id=504, itemid=50885, charttime=t(-10), value="13", valuenum=13, valueuom="mg/dL"),
    dict(labevent_id=8, subject_id=10, hadm_id=100, specimen_id=505, itemid=51265, charttime=t(5), value="45", valuenum=45, valueuom="K/uL"),
    # stay 2000: potassium 6.3 and creatinine 1.5 (RRT is running anyway)
    dict(labevent_id=9, subject_id=20, hadm_id=200, specimen_id=601, itemid=50971, charttime=t(1), value="6.3", valuenum=6.3, valueuom="mEq/L"),
    dict(labevent_id=10, subject_id=20, hadm_id=200, specimen_id=602, itemid=50912, charttime=t(1), value="1.5", valuenum=1.5, valueuom="mg/dL"),
])

inputevents = pd.DataFrame([
    # norepinephrine 0.25 mcg/kg/min for 3 h, then 8 mcg/min (80 kg) = 0.1 for 2 h
    dict(subject_id=10, hadm_id=100, stay_id=1000, starttime=t(1), endtime=t(4), itemid=221906, amount=3.6, amountuom="mg", rate=0.25, rateuom="mcg/kg/min", patientweight=80),
    dict(subject_id=10, hadm_id=100, stay_id=1000, starttime=t(26), endtime=t(28), itemid=221906, amount=1.0, amountuom="mg", rate=8, rateuom="mcg/min", patientweight=80),
    # a norepinephrine row in mg/kg/min is dropped (unit not accepted)
    dict(subject_id=10, hadm_id=100, stay_id=1000, starttime=t(40), endtime=t(42), itemid=221906, amount=1.0, amountuom="mg", rate=0.5, rateuom="mg/kg/min", patientweight=80),
    # propofol infusion 6-20 h; haloperidol bolus at 30 h
    dict(subject_id=10, hadm_id=100, stay_id=1000, starttime=t(6), endtime=t(20), itemid=222168, amount=500, amountuom="mg", rate=30, rateuom="mcg/kg/min", patientweight=80),
    dict(subject_id=10, hadm_id=100, stay_id=1000, starttime=t(30), endtime=t(30.02), itemid=221824, amount=2, amountuom="mg", rate=None, rateuom=None, patientweight=80),
    # stay 2000: vasopressin units/hour (presence only)
    dict(subject_id=20, hadm_id=200, stay_id=2000, starttime=t(1), endtime=t(5), itemid=222315, amount=10, amountuom="units", rate=2.4, rateuom="units/hour", patientweight=70),
])

outputevents = pd.DataFrame(
    [dict(subject_id=10, hadm_id=100, stay_id=1000, caregiver_id=1, charttime=t(h), storetime=t(h),
          itemid=226559, value=20.0, valueuom="mL") for h in range(1, 25)]
    + [dict(subject_id=10, hadm_id=100, stay_id=1000, caregiver_id=1, charttime=t(5), storetime=t(5),
            itemid=227488, value=30.0, valueuom="mL")]  # irrigant in: 20 - 30 -> 0 at 5 h
)

procedureevents = pd.DataFrame([
    dict(subject_id=20, hadm_id=200, stay_id=2000, starttime=t(10), endtime=t(14), itemid=225441,
         value=240, valueuom="min"),
])

emar = pd.DataFrame([
    dict(subject_id=10, hadm_id=100, emar_id="a", emar_seq=1, charttime=t(36), medication="QUEtiapine Fumarate", event_txt="Administered"),
    dict(subject_id=10, hadm_id=100, emar_id="b", emar_seq=2, charttime=t(40), medication="Haloperidol", event_txt="Not Given"),
])

diagnoses_icd = pd.DataFrame([
    dict(subject_id=20, hadm_id=200, seq_num=1, icd_code="N186", icd_version=10),
    dict(subject_id=10, hadm_id=100, seq_num=1, icd_code="J189", icd_version=10),
])

if __name__ == "__main__":
    (OUT / "icu").mkdir(parents=True, exist_ok=True)
    (OUT / "hosp").mkdir(parents=True, exist_ok=True)
    for name, df in {"icustays": icustays, "chartevents": chartevents, "inputevents": inputevents,
                     "outputevents": outputevents, "procedureevents": procedureevents}.items():
        df.to_csv(OUT / "icu" / f"{name}.csv", index=False)
    for name, df in {"labevents": labevents, "emar": emar, "diagnoses_icd": diagnoses_icd}.items():
        df.to_csv(OUT / "hosp" / f"{name}.csv", index=False)
    print(f"wrote {OUT}")
