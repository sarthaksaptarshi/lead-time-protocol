# Reproducibility Guide

> **Anonymized artifact for double-blind review.** No author or institutional
> information is included.

Step-by-step instructions to reproduce every table and figure, and to re-run the
protocol on public data. The whole study is seeded, so synthetic reproduction is
exact.

---

## 1. Environment

| Component | Version used |
|---|---|
| Python | 3.12 |
| NumPy | 2.5 |
| SciPy | 1.18 |
| scikit-learn | 1.9 |
| Matplotlib | 3.11 |
| pandas (Exathlon only) | any recent |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Reproduce the synthetic study

```bash
python experiment.py
```

Runtime: a few minutes on a laptop. Outputs land in `out_v2/` and map to the
paper as follows.

| Output | Maps to |
|---|---|
| `table1_main.csv` | Table 1 — cov, AR(30), ΔAR(30) [95% CI], lead, beyond-chance [CI] |
| `table1b_validation.csv` | Table 2 — control-detector validation (gradual faults) |
| `table2_opreal.csv` | Table 3 — target vs achieved alert rate at targets 3/6/12 |
| `table3_ablation.csv` | Table 4 content — level/trend/combined ablation |
| `table4_opvsoracle.csv` | operational vs oracle thresholds |
| `fig1_concept.png` | Fig. 1 (one detector, two thresholds) |
| `fig2_beyond_chance.png` | Fig. 2 (excess recall vs L, gradual, with controls) |
| `fig3_actionable_recall.png` | Fig. 3 (actionable recall vs L) |
| `fig4_horizon.png` | Fig. 4 (horizon sweep) |
| `episode_level.csv` | per-episode onsets and leads |
| `summary.txt` | all scalars (Spearman CI, null coverage, Monte-Carlo s.e.) |

### Determinism
Master seed `20270419`; per-stream seeds are `MASTER_SEED + i`. Re-running
reproduces identical numbers and byte-identical figures. To test robustness,
change `MASTER_SEED` and confirm the qualitative findings persist (positive
beyond-chance lead on gradual faults; ≈0 on sudden; weak F1–lead correlation;
collapse of lead at the F1-optimal oracle; negative ablation for trend scoring).

Rebuild figures into a paper build (if compiling the manuscript): copy
`out_v2/fig*.png` to the manuscript's `figs/` directory.

---

## 3. Mapping of methodology to code

| Methodology element | Where in `experiment.py` |
|---|---|
| Causal (trailing) smoothing | `causal_ma`, `causal_ewma_pos_diff` |
| Disjoint train / calibration / test | `TRAIN_END`, `CAL_END`, `fit_and_score`, `calibrate` |
| Persistence + cooldown alarm episodes (min gap d=c+p−1) | `alarm_onsets`, `MIN_GAP` |
| Greedy matching, symmetric boundary | `match_leads`, `clean_mask` |
| Uniform (distinct, min-gap) null | `scatter_minspaced`, `null_distributions` |
| Circular-shift null | `null_distributions(..., circ_src=…)` |
| Coverage + actionable recall + ΔAR | `evaluate_stream` (`cov_*`, `arec_*`, `dAR_*`) |
| Multiple seeds + bootstrap CIs | `aggregate`, `boot_ci` |
| Horizon sweep, comparable-budget check | `H_grid`, `TG_LOW` block |
| Ablation (level/trend/combined) | `fit_and_score` variants |
| NAB-style and range-based comparators | `nab_style`, `range_recall_front` |
| Affiliation-style comparator | `affiliation_style` |
| Null self-calibration | `null_selfcal` |

---

## 4. Reproduce the public-data experiments

Datasets are downloaded separately (see `README.md`, Section 4). No data is
bundled here.

### Exathlon (primary)
```bash
python run_exathlon.py --selftest        # pipeline check on a synthetic fixture
python run_exathlon.py \
  --data-root <path>/all_traces --ground-truth <path>/ground_truth.csv \
  --ts-col t --gt-trace-col trace_name \
  --gt-onset-col root_cause_start --gt-end-col extended_effect_end \
  --gt-type-col anomaly_type \
  --horizon-min 30 --budget 1.0 --calib test-normal --max-features 200
```
Outputs `exathlon_out/`: `exathlon_pooled.csv` (coverage, AR(15), ΔAR(15) under
uniform and circular nulls with 95% CIs, achieved rate), `exathlon_manifest.csv`
(trace roles + per-type anomaly counts, including any omitted/unreadable trace),
`exathlon_results.csv` (per trace), `exathlon_dAR.png`. Aggregation is at the
trace level; the 95% CI bootstraps traces as clusters. `--calib test-normal`
gives a controlled low-density operating point (a disclosed retrospective
calibration on test-normal regions); `--calib clean-split` is the deployable
default. Onset = root-cause (injection) start; lead in minutes.

### SMD (supporting)
```bash
python run_smd.py --selftest
python run_smd.py --smd-root <path>/ServerMachineDataset --machines all
```
Outputs `smd_out/`. Lead in samples; window start used as an onset proxy.

---

## 5. Reproducibility checklist

- [ ] `python experiment.py` runs clean and regenerates all `out_v2/` artifacts.
- [ ] Reported synthetic numbers match `out_v2/` exactly (no hand edits).
- [ ] Environment versions match Section 1.
- [ ] Master seed stated; a second seed confirms the qualitative findings.
- [ ] Exathlon run uses `--calib test-normal` and the achieved rate is reported
      (it will differ from the target under distribution shift).
- [ ] Any retrospective-oracle or test-normal calibration is labelled as such.
