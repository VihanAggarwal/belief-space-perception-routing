# REPORT: running self-assessment and per-phase verdicts

This file is the project's honest log. Every former human-eyeball checkpoint is
replaced by a self-assessment here with a quantitative pass/fail criterion stated
before the decision, plus a diagnosis whenever an acceptance criterion does not
pass. Negative results are recorded plainly, not reframed.

---

## Phase 0: setup, repo, dataset, environment

### Handshake outcome
- GitHub: authenticated as `VihanAggarwal`, `repo` scope present.
- Commit identity: `Vihan Aggarwal <VihanAggarwal@users.noreply.github.com>`
  (no Claude attribution anywhere, per project rule).
- Repo: private, `belief-space-perception-routing`.
- Dataset choice: download one full TartanDrive 2.0 v2 bag chunk (~32 GB) and
  extract its full camera stream.

### Hardware detected (this dev machine)
- CPU: 13th Gen Intel Core i7-1360P, 12 physical / 16 logical cores.
- RAM: 15.7 GB.
- GPU: NVIDIA RTX A500 Laptop GPU, 4 GB VRAM, driver 596.47 (CUDA capable);
  plus Intel Iris Xe integrated.
- Disk: 53 GB free on C: at start.
- OS: Windows 11 Pro. Python 3.11.4 in a dedicated venv (default system Python is
  3.14, too new for current torch wheels, so a 3.11 venv was created from the
  Anaconda interpreter).

**Consequence for the method.** 4 GB VRAM is tight for YOLO11x at 1280px fp32 and
may OOM locally. This is exactly why reportable latency numbers are deferred to
`run_on_colab.ipynb`. Local runs prove correctness and produce dev-only numbers,
clearly labeled non-reportable.

### Dataset situation (surfaced honestly)
The archive the human downloaded, `tartan_drive-main.zip` (~10 MB), is the
**TartanDrive toolkit repository**, not the dataset. It contains `download_files.py`,
`azfiles.txt`, rosbag-to-dataset converters, and pretrained world-model
checkpoints, but no camera frames.

The real data is on CMU AirLab's public Swift store:
- bucket `tartandrive` (v1): 22 files, each ~100 GB (verified one HEAD =
  102,500,212,979 bytes), delivered as rosbag bundles that need ROS melodic to
  convert.
- bucket `tartandrive2` (v2): per-trajectory `.bag` chunks ~32 GB each, each holding
  ~1,198 color frames on `/multisense/left/image_rect_color`.

We use one v2 chunk (`turnpike_afternoon_fall_0.bag`). Verified by a 4 MB ranged
read that it is a genuine `#ROSBAG V2.0` file with **uncompressed** chunks
(`compression=none`), which lets a pure-Python reader extract the camera stream
without any ROS install.

### Environment confirmed
- Python 3.11.4 venv; torch 2.6.0+cu124 with `cuda.is_available() == True`;
  ultralytics 8.4.78, opencv 4.13, rosbags 0.11.3, numpy/scipy/pandas/matplotlib.
- YOLO11 inference confirmed on this GPU: yolo11n@640 fp16 and **yolo11x@1280 fp16
  both run without OOM** (the latter is the reference config C1; it fits in 4 GB
  locally, so local frontier profiling can include it). On a random synthetic frame
  both return a valid empty detection set, as expected; real off-road frames will
  contain objects.

### Foundation validated on a synthetic slice (dev-only, not reportable)
To prove pipeline correctness before the 32 GB bag finished downloading, the full
Phase 1 and Phase 2 code was run end-to-end on 80 synthetic frames with known
degradation segments:
- Phase 1: degradation observations computed; deterministic fault labels fired on
  the degraded segments; Track B injection plan applied; contention generator gave a
  clean p95 shift 139 ms -> 390 ms (2.8x) under GPU contention; both coupling
  regimes generated.
- Phase 2: per-channel belief estimator tracked injected faults with 0-frame lag and
  balanced accuracy 0.87 (blur) / 0.91 (occlusion) on the Track B known timeline.
These synthetic numbers validate code paths only. All reportable results use the
real TartanDrive frames.

### Dataset acquisition: partial download + streaming extraction (honest account)
The 32 GB v2 bag download from CMU AirLab proved slow and flaky: the connection
dropped mid-transfer (curl reported `end of response with 26 GB missing`) and
resumed at ~1 MB/s, which would take many hours for the full chunk. Rather than
block the build, we exploited the fact that v2 chunks are uncompressed: a hand-rolled
sequential ROS1 bag reader (`src/extract_frames_streaming.py`) recovers every image
message present in whatever bytes have downloaded so far, without needing the trailing
index (which the `rosbags` library requires). From the partial 7.6 GB we extracted
**309 real camera frames** (`/multisense/left/image_rect_color`, bgr8, 1024x544),
the first ~31 s of the `turnpike_afternoon_fall` trajectory at 10 Hz.

Consequence and scope: the local dev pipeline runs on these 309 real frames (a real
slice, not synthetic). The full ~1198-frame chunk and the reportable GPU numbers come
from `run_on_colab.ipynb`, which re-downloads on Colab's fast link. The background
download continues to retry for completeness. Frame extraction was validated: frames
are valid images (pixel range 2..255, mean ~114) and YOLO11x@1280 returns real
detections on them. Off-road scenes contain few COCO-class objects, so the pseudo-GT
(agreement with C1) is sparse; whether cheaper configs disagree enough under faults to
create an accuracy tradeoff is measured in Phase 3, not assumed.

### Acceptance (self-verified) -- PHASE 0 COMPLETE
- [x] Repo created on GitHub (private, human-authored, no Claude attribution):
      https://github.com/VihanAggarwal/belief-space-perception-routing
- [x] Hardware detected and recorded (above).
- [x] Environment works; YOLO11x inference returns detections on a real frame.
- [x] Dataset frames extracted; **309 real frames** enumerated with a manifest
      (timestamps preserved). Layout: data/frames/turnpike_afternoon_fall_0/.

### Code review (writer + reviewer workflow)
Before any real-data experiment, an independent bug-reviewer pass audited the
scientific modules. It confirmed two RQ-H-biasing bugs, both fixed: (1) differential
abstention deflating the joint policy's miss rate (abstention now OFF for the RQ-H
headline, evaluated separately against the oracle), and (2) the decoupled baseline's
sensor branch being inert (both policies now share one soft objective and differ ONLY
by the coupling term, so the baseline is a fair union). It also flagged in-sample
coupling fitting (now uses a held-out calibration draw) and verified the HMM filters,
CI math, IoU matching, and oracle are correct.

---

## Phase 1: data harness, degradation, contention, regimes (real, 309 frames)

**Criteria:** contention produces a measurable p95/p99 shift under load; fault labels
align with degradation; both coupling regimes and both tracks generate.

**Results (real frames):**
- Degradation + deterministic threshold-and-smooth labels: **30.4% of frames flagged
  faulted** across blur/illumination/occlusion. The first ~31 s of the trajectory
  contains real degradation, so the experiments have genuine fault structure.
- Contention generator (GPU-appropriate: co-running GPU workload + H2D pressure):
  **p95 142 -> 358 ms (2.52x), p99 163 -> 390 ms** under load. Saved
  `outputs/phase1/contention_shift.png`. (Dev-env latency, not reportable.)
- Coupling regimes well separated: **coupled** Pearson r=0.74 (P(contention|fault)=0.81
  vs P(contention|nominal)=0.07); **uncoupled** r=0.06 (fault-independent). This is the
  separation RQ-H requires: a regime where the coupling is real and one where it is not.
- Both tracks generated: Track A (real, headline) and Track B (controlled injection,
  3 events, validation-only). Overlays saved to `outputs/phase1/`.

**Verdict: PASS.** Contention shift measurable, faults present and labeled, regimes
separated. Proceeding to Phase 2.

---

## Phase 2: sensor fault belief estimator (real)

**Criteria:** P(z=faulted) tracks the fault windows with small lag; state the lag and a
tracking criterion numerically. If beliefs do not track, RQ-A1 is compromised for a
boring reason -- flag it.

**Results:**
- Track A (real): the illumination channel produced a labeled fault segment and the
  belief tracked it with **lag 0 frames, balanced accuracy 0.95** (TPR 0.95, TNR 0.96).
  Blur and occlusion produced **no** Track A segments on this slice -- the
  `turnpike_afternoon_fall` degradation is illumination-dominated (canopy/sun/shadow),
  and the global-shutter rectified camera shows little motion blur and high optical-flow
  track survival on the open trail. This is a property of the data, not an estimator
  failure.
- Track B (controlled injection, known timeline): validates the two channels Track A
  did not exercise -- **blur lag 2.5 frames, balanced acc 0.96; occlusion lag 5.0
  frames, balanced acc 0.97** (the seed-0 plan placed blur and occlusion events). These
  are validation aids, clearly labeled, not headline numbers.
- Tracking criterion: belief crosses 0.5 inside the fault window within < 6 frames and
  frame-level balanced accuracy >= 0.9. Met on every channel that had a segment.

**Verdict: PASS.** Beliefs track faults at small lag and high accuracy across all three
channels (illumination via real Track A; blur + occlusion via Track B). The persistence
premise (RQ-A1) is not compromised by poor tracking. Per-channel/track belief plots in
`outputs/phase2/`.
