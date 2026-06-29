# Reviewer Response: what to fix and run

Action list from the IEEE-HPEC reviewer feedback. Items are tagged
**[done-in-code]** (already implemented + committed, just needs a re-run / paper text),
**[run]** (you run it on the Mac with the data/GPU), or **[paper]** (text edit only).

Priorities in order: 0 (critical) > 1 > 3 > 2 > 5 > presentation.

---

## 0. CRITICAL (not in the review, found while auditing): RADIATE RQ-H ran on only 300 frames

`profiling.n_frames: 300` in `config.yaml` bounded the RADIATE profiling to the first
**300 frames** of each ~2,600-frame sequence. Because the simulation length is the number
of profiled frames, **the RADIATE RQ-H deltas were computed on 300 frames**, while Table II
reports full-sequence frame counts and fault fractions. They do not match:

| track | Table II (frames / fault%) | actual RQ-H window | actual in-window fault% |
|---|---|---|---|
| fog   | 2659 / 1.9%  | **300** | **4.0%** |
| rain  | 2651 / 44.4% | **300** | **27.0%** |
| snow  | 2589 / 18.5% | **300** | **11.3%** |
| night | 2644 / 12.7% | **300** | **8.0%** |
| Track C (soiling) | 3303 / 8.1% | 3303 (full) | 8.1% (matches) |
| Track A off-road seg. | 309 / 30.4% | 300 | 31.3% (matches) |

Track C and Track A are fine; the four RADIATE tracks are not. This is the single biggest
validity gap (it also makes the "5 seeds reuse the same frames" point worse: it's 300
frames, 5 seeds). **Fix by re-profiling RADIATE on more frames**, then the reported
frames/fault% become representative and the CIs/temporal-split get more power.

**[run]** Re-profile each RADIATE sequence on >=1500 frames:
```bash
# bump the profiling window (use 0 for ALL frames if the Mac can take it)
# edit config.yaml:  profiling.n_frames: 1500     (was 300)

for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ \
    python src/run_pipeline.py --skip-extract
done

# then re-run the add-ons on the new outputs
python src/run_addons.py --tracks outputs/trackD_rain_4_0 outputs/trackD_snow_1_0 \
                                   outputs/trackD_fog_6_0 outputs/trackD_night_1_0
for t in trackD_fog_6_0 trackD_rain_4_0 trackD_snow_1_0 trackD_night_1_0; do
  SW=""; [ "$t" = "trackD_fog_6_0" ] && SW="--sweeps"
  python src/run_robustness.py --track outputs/$t $SW
done
python src/gen_results.py
python src/make_summary_figure.py
git add outputs RESULTS.md paper/figures && git commit -m "Re-profile RADIATE on 1500-frame windows" && git push
```
**[paper]** After re-profiling, update Table II frames/fault% to the new numbers and check
the RQ-H deltas (they should stay the same sign; report whatever they are). If you instead
keep 300 frames, you MUST report 300 and the in-window fault% in Table II, not the
full-sequence numbers.

---

## 1. Contention-schedule circularity (reviewer #1) -- mostly already safe

The strong version ("schedule uses the router's own detected onset/HMM output") is **false
in the code**, and you should say so explicitly:

- The contention schedule is built from `fault_active = lab["any_fault"]` -- the **Phase-1
  threshold-and-smooth labels** (precomputed, deterministic), see `regimes.coupled_schedule`
  and `simulate.build_substrate`. It is **not** the router's HMM output.
- The online router receives only `s_belief` (sensor HMM filtering **noisy** observations,
  per-seed noise `observation_noise_std=0.15`) and `c_belief` (compute HMM on probe
  latency). **It never sees `state`** (the schedule variable).
- The fault HMM must infer the event from the noisy observations independently.

**[paper]** State the four bullets above in the Experimental Setup, plus: kappa is fit on
**independent calibration contention draws** (different seeds), never the test draw
(`simulate.calibration_kappa`), and is a structural regime constant.

**[done-in-code]** Requirement #4 (calibration/test temporal separation) is now implemented:
`run_robustness.py` adds a `calibration_split` -- fit kappa on the first half of the trace,
evaluate RQ-H on the held-out second half. Results (see `RESULTS.md` 7d):
- kappa is stable where the calibration half has faults (0.78-0.86 vs full ~0.80).
- The held-out test is significant on the **long Track C** sequence (+1.53 pp, CI excludes 0).
- On the 300-frame RADIATE windows the held-out half is fault-sparse (the split is
  underpowered) -- another reason to re-profile longer (item 0). After item 0 this becomes a
  clean rebuttal on all tracks.

---

## 2. Pseudo-GT accuracy validation (reviewer #2)

**[run]** `src/validate_pseudo_gt.py` is a scaffold. Wire up `load_radiate_gt()` (parse the
RADIATE `annotations.json` and rectify boxes through the same calib as
`src/extract_radiate.py`) and `detect()` (mirror `profile_frontier.py`), then:
```bash
python src/validate_pseudo_gt.py --seq-dir data/radiate/rain_4_0 --max-frames 500
```
It correlates per-frame pseudo-GT (agreement-with-C1) against real F1 vs the RADIATE
labels. One number (high Pearson/Spearman) shows agreement-with-reference tracks real
detection quality. **[paper]** Add it as a validation paragraph; keep the rest of the
experiments on pseudo-GT.

---

## 3. Deadline looks convenient (reviewer #3) -- DONE in code

**[done-in-code]** `run_robustness.py` now sweeps the deadline for **every** condition (not
just fog) and adds **externally-motivated fixed deadlines** (10 Hz / 100 ms and 5 Hz /
200 ms), not self-calibrated. See `RESULTS.md` 7b (all-condition sweep) and 7c (fixed Hz).
Pattern: the benefit peaks at 1.0x for all conditions and shrinks at looser deadlines but
stays positive for most (off-road stays +2.53 pp at 1.2x; fog concentrates near 1.0x).
**[paper]** Replace the fog-only sweep with the all-condition table; use "self-calibrated
deadline" consistently and frame it as an operating-point experiment, not a robot
requirement. Cite the fixed-Hz result as the externally-motivated point.

---

## 4. Statistics (reviewer #4)

- **[paper]** Promote the moving-block bootstrap to the **primary** sequential analysis (it's
  already computed for all tracks, `RESULTS.md` 7). Keep the seed CI as secondary.
- **[run, optional]** More noise seeds: edit `seeds:` in `config.yaml` (e.g. 0-9) and re-run
  phases 5-6. This tightens the observation-noise CI only; it does NOT add cross-environment
  generalization, so keep the honest disclosure. Best done together with item 0.
- **[paper]** Tone down significance language so it does not sound like cross-route/vehicle
  generalization; the CIs are sensitivity to injected sensor noise on fixed frames.

---

## 5. Method under-specified (reviewer #5) -- concrete values to put in the paper

All from `config.yaml` + the named functions; add these so reviewers do not need the repo:

- **Health signals**: blur = variance of Laplacian; illumination = grayscale histogram
  entropy; occlusion = sparse optical-flow track-survival rate (fisheye: real soiling-mask
  pixel fraction; blur computed after polar-unwrap). Each robust-z normalized on an
  EMA(alpha=0.3) baseline.
- **Fault labels** (threshold-and-smooth): robust-z threshold 2.5, min segment 5 frames,
  recovery hysteresis 8 frames; per-channel "low tail = bad".
- **Sensor HMM**: 4 states {nominal, degrading, faulted, recovering}, Gaussian emissions
  (per-state std floor 0.3), transition init stay_nominal=0.92, stay_faulted=0.90, Laplace
  smoothing 1.0 on fitted counts; labels expanded with 3 onset / 5 recovery frames.
  `b_f^t = max_k P(faulted | o_{1:t})` (forward filter), see `sensor_belief.py`.
- **Compute HMM**: 2 states, features [p95, p99, queue_depth] over a 30-frame window,
  standardized; transition init stay_nominal=0.95, stay_contended=0.90;
  `b_c^t = P(contended | l_{1:t})`, see `compute_belief.py`.
- **E[acc_c | b_f]** = (1 - b_f)·acc_nominal[c] + b_f·acc_faulted[c] where acc_nominal/faulted
  are per-config mean agreement on non-fault/fault frames (`policies.FrontierModel.exp_accuracy`).
- **q_c (feasibility)**: a config is feasible in the predicted compute state s if its median
  latency in s meets the deadline, i.e. p_meet[c][s] >= 0.5 (`policies._choose`).
- **kappa fit**: kappa = clip(P(cont | fault_{t-lag}) - P(cont | nominal_{t-lag}), 0, 1),
  lag = 3 frames, fit on independent calibration draws (`policies.fit_coupling`,
  `simulate.calibration_kappa`). Coupled-regime generator uses P(cont|fault)=0.85,
  P(cont|nominal)=0.05.
- **Deadline**: median end-to-end latency of C1 under nominal compute (locked rule).
- **Hysteresis dwell**: fixed = 10 frames; model-derived from target false-transition rate
  0.02 (`policies.model_derived_dwell`). Reliability target 0.85.
- **Seeds**: 5; per-seed observation noise std 0.15 (sole source of seed variance).

---

## Presentation fixes (reviewer)

- **[done-in-code]** Figure 1 sign + fonts + clipping: regenerate with
  `python src/make_summary_figure.py` -> `paper/figures/fig_rqh_summary.png`. Axis is now
  "Deadline-miss reduction: decoupled - coupled (pp)" (positive = coupling helps, matches
  Table II), large fonts, padded layout.
- **[paper]** None of the paper figures are in the repo (`paper/figures/` was empty). Commit
  ALL figures so the paper compiles from the repo. Regenerate the per-track ones from
  `outputs/<track>/phase5,phase6,extras/`.
- **[paper]** Standardize terminology: use **"coupled"** everywhere (or "joint" everywhere),
  not both.
- **[paper]** Define RQ-H and RQ-A1 in the intro before the labels are used.
- **[paper]** The learned-router / sequence-split table (rain miss 0.000): label clearly that
  it uses different temporal test partitions than Table II, or drop it. Given the MLP result
  is a wash, consider cutting it for space and using the room for items 2 and 5.
- **[paper]** State that "per-channel" = per health signal (blur / illumination / occlusion),
  not per camera or per modality.

---

## One-shot run sequence (after editing config.yaml: profiling.n_frames: 1500)
```bash
# 0+3+1: re-profile RADIATE, re-run add-ons + robustness, regenerate results + figure
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ python src/run_pipeline.py --skip-extract
done
for t in trackD_fog_6_0 trackD_rain_4_0 trackD_snow_1_0 trackD_night_1_0 trackC trackA_309; do
  SW=""; [ "$t" = "trackD_fog_6_0" ] && SW="--sweeps"
  python src/run_robustness.py --track outputs/$t $SW
done
python src/gen_results.py && python src/make_summary_figure.py
# 2: wire up + run the pseudo-GT validation (needs RADIATE annotations)
# python src/validate_pseudo_gt.py --seq-dir data/radiate/rain_4_0 --max-frames 500
git add -A && git commit -m "Re-profile RADIATE + reviewer robustness add-ons" && git push
```
