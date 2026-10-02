# sofa2

Compute **SOFA-2** scores (Ranzani et al., JAMA 2025) and the **original SOFA** score (Vincent et
al., Intensive Care Med 1996) from ICU data, with the same pipeline, so the two can be compared.

- One row per ICU stay per window (daily, or hourly over the preceding 24 h), with the total, the
  six organ sub-scores and a status flag per organ (observed, carried forward, imputed normal).
- Every threshold lives in YAML (`src/sofa2/config/`), with the source table and footnote next to
  it.
- Data extraction is separate from scoring: adapters turn a database into a small internal schema.
  The first adapter is MIMIC-IV, on BigQuery or on local CSV/Parquet files.

## Install

```bash
pip install -e .                 # core
pip install -e ".[local]"        # + DuckDB, for MIMIC-IV files on disk
pip install -e ".[bigquery]"     # + Google BigQuery client
pip install -e ".[dev]"          # + pytest
```

Python 3.10 or later.

## Quick start

On the synthetic example data in this repository (made up, no patient data):

```python
from sofa2.adapters.internal import InternalAdapter
from sofa2.pipeline import compute_scores

scores = compute_scores(InternalAdapter("tests/data/synthetic"))  # daily windows
print(scores[["stay_id", "window_index", "sofa2_total", "sofa1_total"]])
```

MIMIC-IV from local files laid out as on PhysioNet (`<dir>/icu/*.csv.gz`, `<dir>/hosp/*.csv.gz`):

```python
from sofa2.adapters.mimic_iv import MimicIVAdapter
from sofa2.pipeline import compute_scores_batched

adapter = MimicIVAdapter.local("/data/mimiciv/3.1")
scores = compute_scores_batched(adapter, batch_size=2000, freq="daily")
```

MIMIC-IV on BigQuery (needs PhysioNet credentialed access and a billing project):

```python
adapter = MimicIVAdapter.bigquery(project="my-gcp-project")   # physionet-data.mimiciv_3_1_*
scores = compute_scores_batched(adapter, batch_size=5000)
```

Command line:

```bash
sofa2 score --source internal --path tests/data/synthetic --out scores.csv
sofa2 score --source mimic-local --path /data/mimiciv/3.1 --batch-size 2000 --out scores.parquet
sofa2 score --source mimic-bigquery --project my-gcp-project --batch-size 5000 --out scores.parquet
```

Options: `--scores sofa2,sofa1`, `--missing locf|normal|none`, `--values` (add the worst raw values
behind each score), `--sofa2-config/--sofa1-config/--pipeline-config` (replace a YAML file).

## Output

| column | meaning |
|---|---|
| `stay_id`, `window_index` | stay and 0-based window (day 1 = 0; hourly: hour since admission) |
| `window_start`, `window_end` | window bounds |
| `sofa2_total`, `sofa1_total` | sum of the six organ scores (0–24) |
| `<score>_<organ>` | organ score, organ in respiratory, cardiovascular, brain, liver, kidney, hemostasis |
| `<score>_<organ>_status` | `observed`, `carried_forward`, `imputed_normal` or `missing` |
| `<score>_brain_presedation_gcs` | true if the window used the last GCS before sedation (footnote c) |
| `<score>_<value>` | with `include_values=True`: eg `sofa2_pf_min`, `sofa2_ne_epi_max`, `sofa1_uo_ml_day` |

The original SOFA uses the same organ names (coagulation → hemostasis, central nervous system →
brain, renal → kidney).

## Configuration

`load_config()` reads the packaged YAML files; pass paths or a nested `overrides` mapping to change
them:

```python
from sofa2.config import load_config
cfg = load_config(overrides={"pipeline": {"missing": {"strategy": "normal"}},
                             "sofa2": {"cardiovascular": {"map_only_fallback": True}}})
scores = compute_scores(data, config=cfg)
```

- `sofa2.yaml`: SOFA-2 Table 2 and footnotes a–q.
- `sofa1.yaml`: Vincent 1996 Table 3.
- `pipeline.yaml`: windows, pre-ICU lab lookback, missing data, FiO2 pairing, sedation drugs, urine
  output coverage, weight bounds.

A threshold list (`*_bands`) is a list of `{points, op, value}`. A value scores the highest
`points` whose condition holds, else 0. Lists are checked when loaded: points must increase and
thresholds must move monotonically.

## Internal schema

Adapters produce `sofa2.schema.ICUData`. All values are in canonical units (`sofa2.units` has
explicit conversions: creatinine µmol/L ÷ 88.4, bilirubin µmol/L ÷ 17.1, kPa × 7.5, FiO2 % ÷ 100,
norepinephrine salt → base, infusion rates → µg/kg/min).

| table | columns |
|---|---|
| `stays` | stay_id, patient_id, intime, outtime, weight_kg (optional), chronic_rrt (optional) |
| `measurements` | stay_id, time, variable, value, specimen_id (optional). Variables: pao2, fio2, spo2, map, bilirubin, creatinine, platelets, potassium, ph, bicarbonate, weight |
| `gcs` | stay_id, time, eye, verbal, motor, total, unassessable |
| `infusions` | stay_id, start, end, drug, rate |
| `medications` | stay_id, time, drug |
| `support` | stay_id, start, end, type (imv, niv, cpap, bipap, hfnc, home_vent, ecmo, ecmo_vv, ecmo_va, iabp, impella, lvad, rvad, rrt_continuous, rrt_intermittent) |
| `urine_output` | stay_id, time, volume_ml |

An interval with no end time (still running at extraction) is taken to run until ICU discharge.
Boolean columns accept true/false, t/f, yes/no or 1/0.

A new database needs an adapter that returns these tables. No scoring code changes. For files that
are already in this schema, `InternalAdapter(directory)` reads `<table>.csv` or `<table>.parquet`.
A `unit` column in `measurements` is converted automatically.

## How scores are computed

1. Every stay is split into hours from ICU admission. Each organ is scored per hour from the worst
   values of that hour.
2. A window's organ score is the worst hourly score in the window (Table 2 footnote a).
   - Daily windows are consecutive 24-hour blocks from admission.
   - Hourly windows cover the preceding 24 hours.
3. Missing organs are handled per `pipeline.missing` (below).

### Implementation decisions

These rules are not fully specified by the papers. Each was agreed before implementation.
Thresholds, drug lists, time limits and switches are in the YAML. The pairing logic, the
concurrent norepinephrine + epinephrine sum, dopamine as an "other" agent, SpO2:FiO2 only
without PaO2:FiO2, and the lowest-rate-per-hour rule are fixed in code.

**Respiratory**
- PaO2 is paired with the FiO2 of the same blood gas, else with the last FiO2 up to 4 h before.
- SpO2:FiO2 (SOFA-2 footnote f) is used only in hours without a PaO2:FiO2, only with SpO2 < 98%,
  and with an FiO2 charted up to 1 h before.
- Footnote g (changes within 1 h, eg after suctioning): a reading is ignored if a later reading
  of the same kind within 60 min has a higher ratio and scores fewer points.
- Advanced respiratory support (footnote g) is HFNC, CPAP, BiPAP, NIV, IMV or home ventilation.
  For the original SOFA, "respiratory support" is IMV, NIV, CPAP, BiPAP or home ventilation (not
  HFNC).
- ECMO of any kind scores respiratory 4. VA/VAV ECMO also scores cardiovascular 4 (taken as a
  cardiovascular indication).

**Cardiovascular**
- Vasoactive drugs count only within a continuous infusion of at least 60 min (footnote j). Rows
  within 1 min of each other are merged.
- Norepinephrine + epinephrine is the highest concurrent sum in the hour.
- Dopamine given with norepinephrine/epinephrine counts as an "other" agent.
- Dopamine given with another agent (not norepinephrine/epinephrine) uses the dopamine bands
  (footnote l), plus 1 point (maximum 4).
- Other vasopressors/inotropes: dobutamine, vasopressin, phenylephrine, milrinone, levosimendan,
  angiotensin II, isoproterenol.
- Footnote m (MAP-only scoring when vasoactives are unavailable) is off by default.
- The original SOFA uses the per-drug bands of Vincent 1996 (doses not summed, only
  norepinephrine, epinephrine, dopamine and dobutamine).

**Brain**
- GCS charted during a sedative infusion (propofol, midazolam, lorazepam, dexmedetomidine) is
  ignored. In sedated hours the last GCS before sedation is carried, with no time cap. With
  none, the hour has no GCS and is handled as missing, which gives 0 under `locf` and `normal`
  (footnote c). The original SOFA uses the same carry.
- A GCS flagged unassessable (eg intubated) scores on the motor response in SOFA-2 only
  (footnote d): M6 → 0, M5 → 1, M4 → 2, M3 → 3, M2/M1 → 4.
- Delirium drugs (haloperidol, quetiapine, olanzapine, risperidone, ziprasidone) score at least
  1 point (footnote e, SOFA-2 only).

**Kidney**
- Urine output rate over a trailing 6, 12 or 24 h window = volume / summed collection intervals /
  weight.
  - A chart's collection interval runs from the previous chart (the first: from admission).
  - The window is scored only when the intervals cover it fully.
  - Windows are evaluated at every chart time, because a window ending between charts would
    include uncovered time. An hour takes its lowest evaluable rate.
  - On day 1, a 24 h window is fully covered only if a chart falls exactly 24 h after
    admission. In practice the 24-hour criteria therefore never apply on day 1: 0 of 140 day-1
    windows in the MIMIC-IV demo.
- Anuria means 0 mL over a covered 12 h window with at least 2 charts.
- Weight is the last weight charted, else the first of the stay, else `stays.weight_kg`.
- RRT scores 4 while a session runs.
  - Intermittent RRT also scores 4 for 72 h after each session (footnote q).
  - Chronic dialysis scores 4 throughout.
- Footnote p ("fulfils criteria for RRT") must be met within the same window.
- The original SOFA uses the latest evaluable 24-hour urine volume in the window (mL/day).

**Labs and boundaries**
- Laboratory values up to 6 h before ICU admission count in hour 0. PaO2, FiO2 and SpO2 are
  excluded from this lookback, because pre-ICU support status is often unrecorded.
- The original SOFA's printed ranges are read as half-open intervals, so 1.95 mg/dL bilirubin
  scores 1. Bilirubin 12.0 and creatinine 5.0 score 4, consistent with Vincent's µmol/L columns.
- SI creatinine is converted with 88.4. Table 2's rounded µmol/L cut-offs (110/170/300) therefore
  differ slightly from exact conversions.

### Missing data

`pipeline.missing.strategy`:

- `locf` (default): an organ without data in a window takes the score of its last measured hour,
  if that hour is at most `locf_max_hours` (24) before the window start (`carried_forward`).
  Otherwise it scores 0 (`imputed_normal`). On day 1 nothing earlier exists, so a missing organ
  scores 0, as Table 2 footnote b recommends.
  - Only measurements are carried: PaO2:FiO2, SpO2:FiO2, MAP, GCS, labs and urine output.
  - Points from a treatment that has stopped are not carried, because its record shows it ended.
    This covers ECMO, vasoactive drugs, mechanical support, RRT and delirium drugs.
  - Each measured variable is carried separately (eg creatinine and urine output), and the
    organ takes the worst.
  - A carried respiratory score is capped at 2 when the window has no respiratory support
    (footnote h).
- `normal`: missing organs score 0.
- `none`: missing organs stay empty, and so does the total.

### Not identifiable in EHR data

These rules are not implemented, or are off by default:
- Ceiling of treatment or unavailable support (footnote h; footnote m is available as a switch).
- RRT given only for non-renal causes (footnote o).
- Generalized myoclonus (brain 4).
- The indication for ECMO (VA is assumed to be cardiovascular).

## MIMIC-IV adapter

`src/sofa2/adapters/mimic_iv/mapping.yaml` lists every item ID used. All were checked against
`d_items`/`d_labitems` of the MIMIC-IV demo.

Where a rule comes from MIT-LCP mimic-code, the mapping says so: ventilation episodes,
urine-output items, RRT items and lab ranges. Deviations from mimic-code:
- Only arterial PaO2 is used.
- **Ventilation:**
  - All oxygen devices charted at a time are considered.
  - "Tracheostomy tube" alone gives no status, because the ventilator mode decides; "Trach mask"
    counts as oxygen.
  - The Hamilton mode SPONT counts as invasive, and the ventilator mode CPAP as CPAP.
  - Invasive and non-invasive ventilation procedures (procedureevents) are a second source.
- **RRT:** dialysis catheter charts alone are not RRT.
- **Mechanical circulatory support:** only items measured while the device runs (flows,
  pressures, speeds).
- **Infusions:** for dose-scored drugs (catecholamines, dopamine, dobutamine, phenylephrine,
  milrinone), rows in rate units other than mcg/kg/min or mcg/min are dropped with a warning.
- **Weight:** charted weight (kg, or lbs converted), else the first `inputevents.patientweight`
  of the extracted drug rows.
- **eMAR:** depot antipsychotics (decanoate, pamoate, Consta, microspheres) are not delirium
  treatment.

For many batches on local files, convert once to Parquet:
`sofa2.adapters.mimic_iv.convert_to_parquet(src, dst)`.

The same SQL runs on BigQuery and on DuckDB; only table paths and date arithmetic differ.

**Validation status:**
- The local adapter was run on the open MIMIC-IV demo.
- The BigQuery path is tested by checking the rendered SQL. It has not yet been run against
  BigQuery.

## Development

```bash
pip install -e ".[dev]"
pytest
python tests/data/make_synthetic.py         # regenerate the synthetic dataset
python tests/data/make_mimic_synthetic.py   # regenerate the MIMIC-shaped test files
```

The tests cover:
- every sub-score at, just above and just below each threshold;
- the derivation rules;
- the missing-data strategies;
- end-to-end scores on the synthetic data, checked by hand;
- the MIMIC-IV adapter on made-up MIMIC-shaped files.

## References

- Ranzani OT, Singer M, Salluh JIF, et al. Development and validation of the Sequential Organ
  Failure Assessment (SOFA)-2 score. *JAMA*. 2025. https://jamanetwork.com/journals/jama/fullarticle/2840822
- Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related Organ Failure Assessment) score
  to describe organ dysfunction/failure. *Intensive Care Med*. 1996;22:707-710.
- Johnson AEW, et al. MIMIC-IV. PhysioNet. mimic-code: https://github.com/MIT-LCP/mimic-code

## License

MIT (see `LICENSE`).
