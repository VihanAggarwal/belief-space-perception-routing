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

### Acceptance (self-verified) [status updated as background jobs complete]
- [ ] Repo created on GitHub (private, human-authored, no Claude attribution).
- [ ] Dataset bag downloaded and frames extracted; frame count recorded.
- [ ] One YOLO11x inference returns detections on a real frame.
- [ ] Hardware, layout, frame count recorded here.

(Verdict and numbers appended once the 32 GB download and dependency install
finish; both are running in the background.)
