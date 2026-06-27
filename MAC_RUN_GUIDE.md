# Mac Run Guide (Apple Silicon / Metal GPU)

For running the belief-space-perception-routing experiments on a Mac with an Apple
Silicon GPU (M1/M2/M3/M4/M5). Covers the two built tracks:

- **Track A — TartanDrive** (off-road, normal lens)
- **Track D — RADIATE** (real adverse weather: rain / snow / fog / night)

Track C (WoodScape fisheye) is not built yet and is intentionally out of scope here.

This runs the pipeline exactly as designed (no methodology changes): per-channel
degradation observations, deterministic threshold-and-smooth fault labels, switching-HMM
sensor belief, two-state compute belief, joint vs decoupled routing, both hysteresis
variants, abstention, oracle labeler, and the RQ-H / RQ-A1 / RQ-A2 experiments with CIs
over 5 seeds.

Note on latencies: numbers measured on your Mac's Metal GPU are environment-specific
(disclose the chip). The RQ-H deadline-miss comparison is the result; absolute latencies
are dev-environment.

---

## 0. Prerequisites
- macOS on Apple Silicon (M-series).
- Python 3.11 or 3.12, and `git`.
- Disk: ~40 GB free if you run Track A (TartanDrive downloads a ~32 GB bag). Track D
  needs only a few GB per RADIATE sequence.

## 1. One-time setup
```bash
git clone https://github.com/VihanAggarwal/belief-space-perception-routing.git
cd belief-space-perception-routing

python3.12 -m venv .venv        # or python3.11
source .venv/bin/activate
pip install --upgrade pip

# IMPORTANT: on Mac use the DEFAULT torch wheel (ships Metal/MPS).
# Do NOT use the CUDA --index-url that requirements.txt mentions for NVIDIA.
pip install torch torchvision
pip install -r requirements.txt

# confirm the Metal GPU is visible (expect: mps available: True)
python -c "import torch; print('mps available:', torch.backends.mps.is_available())"
```
`config.yaml` already ships `device.mode: mps`, so the pipeline auto-uses the Metal GPU.
(Force it with `DEVICE=mps` or `DEVICE=cpu` in a `.env` file if ever needed.)

---

## 2. Track A — TartanDrive (off-road, normal lens)
```bash
python src/download_data.py                 # ~32 GB bag chunk (resumable)
python src/extract_frames.py                # bag -> PNG frames + manifest
OUTPUTS_DIR=outputs/trackA python src/run_pipeline.py --skip-extract
```
- Results land in `outputs/trackA/` (RQ-H figure: `outputs/trackA/phase5/rqh_centerpiece.png`).
- This runs the FULL trajectory (~1,198 frames) — more fault onsets than the committed
  309-frame dev slice, so it is a stronger Track A point.
- Profiling all frames is heavy; if you want to bound it, set `profiling.n_frames` in
  `config.yaml` (e.g. 600) or leave it for the full run.

---

## 3. Track D — RADIATE (rain / snow / fog / night)

**Get the data first.** You need the 4 sequence folders (each containing `zed_left/`).
Either get them from Vihan (the 4 zips: `rain_4_0`, `snow_1_0`, `fog_6_0`, `night_1_0`),
or register at https://pro.hw.ac.uk/radiate/downloads/ . Unzip each so you have:
```
data/radiate/rain_4_0/zed_left/000001.png ...
data/radiate/snow_1_0/zed_left/...
data/radiate/fog_6_0/zed_left/...
data/radiate/night_1_0/zed_left/...
```

**Run all four conditions:**
```bash
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ \
    python src/run_pipeline.py --skip-extract
done
```
- Each condition writes `outputs/trackD_<seq>/` (its own RQ-H/RQ-A1/RQ-A2).
- The Phase 1 log prints the per-condition fault-onset count — note it next to each result.
- `extract_radiate.py` rectifies the left ZED camera to `camera_left_rect` (matching the
  RADIATE SDK); radar/lidar/annotations are unused, pseudo-GT is YOLO11x, same as every track.

**If a condition is slow** (Phase 1's optical flow is CPU-bound on Mac too), cap frames —
this is the same option used for the committed rain result (1,500 frames, still significant):
```bash
python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
```

---

## 4. Return the results
If you have push access (collaborator):
```bash
git add outputs/trackA outputs/trackD_*
git commit -m "Track A (full TartanDrive) and Track D (RADIATE rain/snow/fog/night) on Apple MPS"
git push
```
Otherwise zip `outputs/` and send it to Vihan.

---

## 5. Mac / MPS specifics (handled automatically, listed for transparency)
- **Precision axis is CUDA-only.** On MPS the configs run fp32 (resolution still varies),
  because MPS fp16 inference is unreliable for this study. Disclosed in the write-up.
- **Contention runs as a separate process on MPS** (Metal command buffers are not
  thread-safe), so the co-running GPU workload is launched as a child process. Automatic.
- After Phase 1, check `outputs/<track>/phase1/contention_summary.json` -> `p95_shift_ratio`
  (want roughly 2-3x). If it is far off, adjust `contention.gpu.competitor_sleep_s` in
  `config.yaml` (smaller = more contention) and re-run.
- Latencies are your Mac's; the RQ-H comparison is the reportable result, not the absolute ms.

---

## 6. Track C (WoodScape fisheye) — coming later
Not built yet. When added it will need the WoodScape Soiling dataset (gated, Valeo
registration) and fisheye-correct channels (mask-based occlusion ground truth, polar
blur metric, undistort-before-detection). A separate section will be added here then.
