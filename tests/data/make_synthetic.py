"""Write the synthetic test dataset (internal schema). Entirely made up; no patient data.

Run from the repository root: ``python tests/data/make_synthetic.py``.
Expected daily scores are listed in tests/test_pipeline_e2e.py.
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "synthetic"
T0 = pd.Timestamp("2100-01-01 00:00")


def t(hours: float) -> str:
    return str(T0 + pd.Timedelta(hours=hours))


stays = pd.DataFrame(
    [
        # stay 1: ventilated, vasopressors, sedation, two days
        dict(stay_id=1, patient_id=101, intime=t(0), outtime=t(48), weight_kg=None, chronic_rrt=False),
        # stay 2: no data at all
        dict(stay_id=2, patient_id=102, intime=t(0), outtime=t(30), weight_kg=None, chronic_rrt=False),
        # stay 3: meets RRT criteria, SpO2:FiO2 only, short dopamine, dobutamine
        dict(stay_id=3, patient_id=103, intime=t(0), outtime=t(24), weight_kg=70, chronic_rrt=False),
        # stay 4: VA ECMO, chronic RRT, GCS 7
        dict(stay_id=4, patient_id=104, intime=t(0), outtime=t(24), weight_kg=70, chronic_rrt=True),
    ]
)

m = [
    # stay 1
    (1, t(2), "pao2", 70, "bg1"), (1, t(2), "fio2", 0.5, "bg1"),
    (1, t(10), "map", 65, None), (1, t(30), "map", 75, None),
    (1, t(-3), "bilirubin", 2.5, None),
    (1, t(5), "platelets", 90, None), (1, t(30), "platelets", 45, None),
    (1, t(5), "creatinine", 1.5, None),
    (1, t(0.5), "weight", 80, None),
    # stay 3
    (3, t(3), "creatinine", 2.0, None), (3, t(4), "potassium", 6.2, None),
    (3, t(10), "fio2", 0.4, None), (3, t(10.5), "spo2", 92, None),
]
measurements = pd.DataFrame(m, columns=["stay_id", "time", "variable", "value", "specimen_id"])

gcs = pd.DataFrame(
    [
        (1, t(1), 4, 5, 6, 15, False),
        (1, t(10), 2, 2, 4, 8, False),       # during propofol: ignored
        (1, t(36), 3, None, 5, None, True),  # intubated, verbal not assessable, motor 5
        (4, t(2), 2, 2, 3, 7, False),
    ],
    columns=["stay_id", "time", "eye", "verbal", "motor", "total", "unassessable"],
)

infusions = pd.DataFrame(
    [
        (1, t(1), t(5), "norepinephrine", 0.15),
        (1, t(1), t(5), "vasopressin", 0.03),
        (1, t(6), t(30), "propofol", 30),
        (3, t(2), t(2.5), "dopamine", 25),   # 30 min only: not scored
        (3, t(2), t(4), "dobutamine", 5),
    ],
    columns=["stay_id", "start", "end", "drug", "rate"],
)

medications = pd.DataFrame([(1, t(40), "haloperidol")], columns=["stay_id", "time", "drug"])

support = pd.DataFrame(
    [(1, t(0), t(48), "imv"), (4, t(0), t(24), "ecmo_va")],
    columns=["stay_id", "start", "end", "type"],
)

urine = pd.DataFrame(
    [(1, t(h), 30.0) for h in range(1, 25)], columns=["stay_id", "time", "volume_ml"]
)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "stays": stays, "measurements": measurements, "gcs": gcs, "infusions": infusions,
        "medications": medications, "support": support, "urine_output": urine,
    }.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"wrote {OUT}")
