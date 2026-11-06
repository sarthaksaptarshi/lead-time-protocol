# Lab Guide — Running on Public Datasets

> **Anonymized artifact for double-blind review.** No author or institutional
> information is included.

How to obtain the public datasets and run the protocol on them. The synthetic
study (`experiment.py`) needs no download; see `README.md`. This guide covers the
two real-data adapters.

Datasets are third-party and are **not** redistributed here. Download them from
their official sources and respect their licenses.

---

## 1. Exathlon (primary real-data experiment)

**Get the data.** From the Exathlon project (`github.com/exathlonbenchmark/exathlon`),
run its data-extraction step. You will have:
- per-application trace archives; extract all trace CSVs into one folder (call it
  `all_traces`), and
- a `ground_truth.csv` with columns including `trace_name`, `anomaly_type`,
  `root_cause_start`, `root_cause_end`, `extended_effect_end` (Unix-second
  timestamps). Traces are recorded once per second.

**Confirm the column names** before running, then verify the pipeline:
```bash
python run_exathlon.py --selftest
```
This builds a synthetic Exathlon-format fixture and runs end to end. Its numbers
are synthetic — do not cite them.

**Run the controlled low-density operating point** (used in the paper):
```bash
python run_exathlon.py \
  --data-root  <path>/all_traces \
  --ground-truth <path>/ground_truth.csv \
  --ts-col t --gt-trace-col trace_name \
  --gt-onset-col root_cause_start --gt-end-col extended_effect_end \
  --gt-type-col anomaly_type \
  --horizon-min 30 --budget 1.0 --calib test-normal --max-features 200
```

**What it does.**
- Treats traces listed in the ground truth as *disturbed* (test) and all other
  trace CSVs as *undisturbed* (train/calibrate).
- Uses each anomaly's **root-cause start** as the onset (the injection time),
  and the timestamp column to report **lead in minutes**.
- Fits detectors on the undisturbed pool (standardized; a single global
  top-variance feature subset of size `--max-features`), reduces every trace to
  that same subset, and scores causally.
- Forms persistence-gated alarm episodes, matches within the horizon, and
  reports coverage, actionable recall AR(L), excess ΔAR(L) over uniform and
  circular nulls (95% CI over traces), and the achieved alert rate.

**Calibration modes.**
- `--calib clean-split` (default, deployable): threshold from the undisturbed
  calibration split. Under strong distribution shift this does **not** control
  the test rate (it can run far above the target).
- `--calib test-normal` (disclosed retrospective): threshold set to a target
  episode rate on the **normal regions of the test traces** — a controlled
  low-density operating point. Not deployable; label it as a diagnostic.

**Outputs** (`exathlon_out/`): `exathlon_pooled.csv`, `exathlon_manifest.csv`
(trace roles and per-type anomaly counts; shows any omitted/unreadable trace),
`exathlon_results.csv` (per trace), `exathlon_dAR.png`.

**Notes and caveats.**
- The root-cause start is the injection time, which is a defensible onset for the
  injected event but **not** the onset of a downstream service failure; for an
  externally injected event without a precursor, pre-injection warning may be
  unavailable by construction.
- Exathlon is high-dimensional (~2000 metrics). `--max-features` keeps the
  highest-variance columns for tractable runtime; set `--max-features 0` to keep
  all (slower). Record the value you use — it is a preprocessing choice.
- Unreadable/empty trace files are skipped with a `[skip]` message and listed in
  the manifest.

---

## 2. SMD (supporting)

**Get the data.** SMD ships with OmniAnomaly
(`github.com/NetManAIOps/OmniAnomaly`): a `ServerMachineDataset` folder with
`train/`, `test/`, and `test_label/` subfolders.

**Verify, then run:**
```bash
python run_smd.py --selftest
python run_smd.py --smd-root <path>/ServerMachineDataset --machines all \
                  --fpr 0.01 --horizon-min 60
```

**What it does / caveats.**
- Trains on the clean training split, calibrates on a held-out portion, and
  evaluates the test split; episodes are maximal label runs.
- SMD labels **windows, not onsets**, and its physical sampling interval is not
  stated with the data, so lead is reported in **samples** and each window start
  is used as an **onset proxy** of unknown offset.
- Outputs `smd_out/` (`smd_pooled.csv`, `smd_results.csv`, `smd_dAR.png`).

---

## 3. Shared protocol parameters

Both adapters use the same protocol code as the synthetic study:
persistence `p=3`, cooldown `c=15` samples (minimum alarm gap `d=c+p−1=17`),
`B=200` null draws, causal smoothing, and percentile-bootstrap 95% CIs over the
aggregation unit (streams for synthetic, traces for Exathlon, machines for SMD).
All are fixed in the released code.
