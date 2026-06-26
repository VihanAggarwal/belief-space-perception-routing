"""Generate run_on_colab.ipynb: the self-contained notebook that produces the
REPORTABLE GPU profiling and latency numbers (Colab T4/A100). Latency on a laptop
CPU/weak GPU is not reportable; the precision (fp16) axis is only meaningful on a
real GPU, and YOLO11x@1280 needs GPU headroom. The notebook implements
GPU-appropriate contention (co-running GPU workload + host-to-device pressure), the
same as src/contention.py.
"""
import json
from pathlib import Path

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = [
    MD("""# Belief-Space Perception Routing: reportable GPU run (Colab)

Run this on a Colab **GPU** runtime (Runtime -> Change runtime type -> T4 or A100).
It produces the reportable profiling, latency, and experiment numbers. Locally
measured laptop latencies are dev-only and are never the reported result.

What it does:
1. installs dependencies and clones the project repo,
2. downloads one TartanDrive 2.0 bag chunk and extracts the camera frames,
3. runs Phase 1 (labels), Phase 3 (frontier + deadline), and Phases 4-6
   (RQ-H, RQ-A1, RQ-A2) with GPU-appropriate contention,
4. zips `outputs/` so you can download the reportable plots and tables.

Disclose the exact environment (GPU model from the first cell) in the write-up.
"""),
    MD("## 1. Environment"),
    CODE("!nvidia-smi"),
    CODE("""
# Dependencies. torch/cuda is preinstalled on Colab GPU runtimes.
!pip -q install "ultralytics>=8.3.0" opencv-python-headless rosbags filterpy python-dotenv pyyaml
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""),
    MD("""## 2. Get the code

The repo is private. Paste a GitHub token (repo scope) when prompted, or set
`USE_UPLOAD = True` and upload a `src.zip` of the `src/` folder + `config.yaml`.
"""),
    CODE("""
import os, getpass, subprocess
USE_UPLOAD = False
REPO = "VihanAggarwal/belief-space-perception-routing"
if not USE_UPLOAD:
    tok = getpass.getpass("GitHub token (repo scope): ").strip()
    url = f"https://{tok}@github.com/{REPO}.git"
    subprocess.run(["git", "clone", "--depth", "1", url, "/content/proj"], check=True)
    os.chdir("/content/proj")
else:
    from google.colab import files
    up = files.upload()  # upload src.zip and config.yaml
    os.makedirs("/content/proj", exist_ok=True); os.chdir("/content/proj")
    !unzip -o src.zip -d /content/proj
print("cwd", os.getcwd()); print(os.listdir("."))
"""),
    MD("""## 3. Download a TartanDrive 2.0 bag chunk and extract frames

~32 GB download; Colab disk is ~100 GB and the link is fast (a few minutes). Set
`EXTRACT_MAX_FRAMES` to cap frames (0 = all ~1198 in the chunk).
"""),
    CODE("""
import os
os.makedirs("data/raw", exist_ok=True)
SWIFT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BAG = f"{SWIFT}/tartandrive2/bags/2023-10-26-14-42-35_turnpike_afternoon_fall/turnpike_afternoon_fall_0.bag"
DEST = "data/raw/turnpike_afternoon_fall_0.bag"
if not os.path.exists(DEST) or os.path.getsize(DEST) < 32_000_000_000:
    !curl -k -C - --retry 8 --retry-all-errors -o "$DEST" "$BAG"
print("bag GB:", os.path.getsize(DEST)/1e9)
"""),
    CODE("""
os.environ["EXTRACT_MAX_FRAMES"] = "0"   # 0 = all frames in the chunk
!python src/extract_frames.py
"""),
    MD("## 4. Phase 1 labels, Phase 3 frontier + deadline (reportable)"),
    CODE("""
# Phase 1 produces the Track A fault labels used for per-fault accuracy buckets.
!python src/phase1_run.py
# Phase 3 profiles the full frontier on the GPU and sets the deadline. These latency
# numbers ARE reportable (real GPU, real fp16 axis, GPU-appropriate contention).
!python src/profile_frontier.py
"""),
    MD("## 5. Experiments: RQ-H (centerpiece), RQ-A1, RQ-A2 (reportable)"),
    CODE("""
!python src/phase4_run.py
!python src/phase5_run.py
!python src/phase6_run.py
"""),
    MD("## 6. Collect reportable outputs"),
    CODE("""
import shutil
shutil.make_archive("/content/outputs_colab", "zip", "outputs")
from google.colab import files
files.download("/content/outputs_colab.zip")
print("Download started: outputs_colab.zip")
"""),
    MD("""## Notes for the write-up
- Report the GPU model printed in cell 1 (e.g., Tesla T4 16 GB / A100 40 GB).
- All latency numbers in `outputs/phase3/` and the deadline in `config.yaml`
  (`value_ms`) come from THIS environment, not the laptop.
- Contention here is GPU-appropriate (co-running GPU workload + host-to-device
  pressure), per `src/contention.py::GPUContention`.
- Pseudo-GT note: accuracy is agreement with the C1 reference, not external truth.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "run_on_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
