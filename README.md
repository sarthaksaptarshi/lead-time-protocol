# Measuring the Warning — Code and Reproducibility

> **Anonymized artifact for double-blind review.** Author names, emails,
> affiliations, and acknowledgments have been removed. Do not add identifying
> information to this mirror.

Reproducible code for a lead-time evaluation **protocol** for time-series
anomaly detectors. The protocol measures how much *warning* a detector gives
before a failure, whether that warning exceeds a budget-matched random-alarm
baseline, and across how many events — reported jointly with coverage and
confidence intervals. The protocol is the contribution; this repository is not a
new detector.

Everything here is deterministic (fixed seed) and runs on CPU with four or five
standard Python packages.

---

## 1. Requirements

- Python 3.10+ (developed on 3.12)
- `numpy`, `scipy`, `scikit-learn`, `matplotlib` (synthetic study)
- `pandas` additionally for the Exathlon adapter

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Exact versions used: Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.9,
Matplotlib 3.11. No network or GPU is needed to run the experiments (datasets
are downloaded separately; see Section 4).

---

## 2. Synthetic study (one command)

```bash
python experiment.py
```

Runs 15 independent seeded streams and writes `out_v2/`:

| File | Content |
|---|---|
| `table1_main.csv` | Main comparison: coverage, AR(30), ΔAR(30) with 95% CI, lead, beyond-chance |
| `table1b_validation.csv` | Control-detector validation (random / onset-only / precursor-oracle) |
| `table2_opreal.csv` | Calibration target vs achieved alert rate at three targets |
| `table3_ablation.csv` | level / trend / combined ablation |
| `table4_opvsoracle.csv` | Operational vs retrospective-oracle thresholds |
| `episode_level.csv` | Per-(stream, detector, episode) onset and lead |
| `fig1_concept.png` … `fig4_horizon.png` | The four figures |
| `summary.txt` | All scalars (Spearman CI, null coverage, Monte-Carlo s.e.) |

Deterministic: master seed `20270419`. Re-running reproduces identical numbers
and byte-identical figures.

---

## 3. Public-data experiments

### 3a. Exathlon (primary real-data experiment)

First verify the pipeline on a synthetic Exathlon-format fixture (not a real
result):

```bash
python run_exathlon.py --selftest
```

Then, after obtaining the dataset (Section 4), run the controlled low-density
operating point used in the paper:

```bash
python run_exathlon.py \
  --data-root  <path>/all_traces \
  --ground-truth <path>/ground_truth.csv \
  --ts-col t --gt-trace-col trace_name \
  --gt-onset-col root_cause_start --gt-end-col extended_effect_end \
  --gt-type-col anomaly_type \
  --horizon-min 30 --budget 1.0 --calib test-normal --max-features 200
```

Writes `exathlon_out/`: `exathlon_pooled.csv`, `exathlon_manifest.csv`
(trace roles and per-type anomaly counts), `exathlon_results.csv`, and
`exathlon_dAR.png`. `--calib test-normal` is a disclosed retrospective
calibration that reaches a controlled alert rate; `--calib clean-split` is the
deployable (but non-transferring) default. The onset used is the labelled
root-cause (injection) start; lead is reported in minutes.

### 3b. SMD (supporting)

```bash
python run_smd.py --selftest
python run_smd.py --smd-root <path>/ServerMachineDataset --machines all
```

Writes `smd_out/` (`smd_pooled.csv`, `smd_results.csv`, `smd_dAR.png`). SMD
labels windows, not onsets, and its physical sampling interval is not stated, so
lead is reported in samples with the window start as an onset proxy.

---

## 4. Datasets (download separately)

These are third-party datasets; download from their official sources and respect
their licenses. Do **not** redistribute the raw data in this repository.

- **Exathlon** — `github.com/exathlonbenchmark/exathlon` (run its `extract_data`
  step; point `--data-root` at the extracted traces and `--ground-truth` at the
  ground-truth CSV). License: CC BY-NC-SA 4.0.
- **SMD (Server Machine Dataset)** — released with OmniAnomaly,
  `github.com/NetManAIOps/OmniAnomaly` (folder `ServerMachineDataset` with
  `train/`, `test/`, `test_label/`).

---

## 5. What is shared across the scripts

`experiment.py`, `run_smd.py`, and `run_exathlon.py` share the same detector and
metric code (causal persistence-gated alarm episodes; greedy matching; uniform
and circular-shift nulls; excess actionable recall ΔAR with bootstrap CIs). "One
audited pipeline" refers to that shared code invoked by three entry points.

---

## 6. Determinism and honesty

Every number in the paper is produced by these scripts; nothing is hand-entered.
The synthetic control detectors (random, onset-only, precursor-oracle) are
included so the chance-corrected metric can be read honestly. If you change the
seed or parameters, report what you actually observe.
