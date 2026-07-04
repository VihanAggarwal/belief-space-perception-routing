"""RQ-Coupling (c): cross-SEQUENCE calibration transfer.

Phase 5's kappa is fit on calibration draws of the SAME trace being tested
(different contention-schedule seeds, same real fault timeline) -- this is a
temporal/seed separation, not a sequence separation. This script instead fits
kappa entirely on a DIFFERENT sequence of the same weather condition, then
reruns RQ-H on the test sequence with that transplanted kappa, comparing to
the test sequence's own natively-fit kappa. If the benefit survives with a
kappa learned on a different real trajectory, the coupling coefficient
generalizes across sequences, not just across seeds of one trace.

    python src/measure_cross_sequence_calibration.py \
        --calib-track outputs/trackD_rain_2_0 --test-track outputs/trackD_rain_4_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_substrates(track: str, cfg_loader, sim, seeds, regime="coupled"):
    os.environ["OUTPUTS_DIR"] = track
    cfg = cfg_loader()
    kappa_native = sim.calibration_kappa(cfg, regime, [s + 100 for s in seeds])
    subs = [sim.build_substrate(cfg, regime=regime, seed=s) for s in seeds]
    return cfg, kappa_native, subs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-track", required=True, help="sequence to FIT kappa on")
    ap.add_argument("--test-track", required=True, help="sequence to EVALUATE RQ-H on")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import policies as pol
    import ci as cistats

    if not (abspath(args.test_track) / "phase3" / "phase3_summary.json").exists():
        print(f"ERROR: {args.test_track} has no profiled frontier."); return 2
    if not (abspath(args.calib_track) / "phase3" / "phase3_summary.json").exists():
        print(f"ERROR: {args.calib_track} has no profiled frontier."); return 2

    # kappa fit entirely on the CALIB track (different real trajectory)
    cfg_c, kappa_transfer, _ = _load_substrates(args.calib_track, load_config, sim, [1, 2, 3, 4, 5])

    # test track's own substrate + its own native kappa
    cfg_t, kappa_native, subs_test = _load_substrates(args.test_track, load_config, sim, cfg_c["seeds"])
    rt = cfg_t["routing"]["reliability_target"]
    dwell = cfg_t["routing"]["hysteresis"]["fixed"]["dwell_frames"]

    def rqh_with_kappa(kappa):
        jl, dl = [], []
        for sub in subs_test:
            joint = pol.JointPolicy(sub.fm, kappa, rt, pol.Hysteresis(dwell))
            decoup = pol.DecoupledPolicy(sub.fm, rt, pol.Hysteresis(dwell))
            jl.append(sim.run_policy(joint, sub)["deadline_miss_rate"])
            dl.append(sim.run_policy(decoup, sub)["deadline_miss_rate"])
        return cistats.paired_diff_ci(dl, jl)

    native = rqh_with_kappa(kappa_native)
    transferred = rqh_with_kappa(kappa_transfer)

    summary = {
        "calib_track": args.calib_track, "test_track": args.test_track,
        "kappa_native_on_test_track": kappa_native,
        "kappa_transferred_from_calib_track": kappa_transfer,
        "rqh_reduction_native_kappa_pp": {"mean": native["mean"] * 100, "lo": native["lo"] * 100,
                                          "hi": native["hi"] * 100, "significant": native["significant"]},
        "rqh_reduction_transferred_kappa_pp": {"mean": transferred["mean"] * 100, "lo": transferred["lo"] * 100,
                                               "hi": transferred["hi"] * 100, "significant": transferred["significant"]},
    }
    out_dir = abspath(args.test_track) / "extras"; out_dir.mkdir(parents=True, exist_ok=True)
    calib_name = Path(args.calib_track).name
    json.dump(summary, open(out_dir / f"cross_seq_calib_from_{calib_name}.json", "w"), indent=2, default=str)

    print(f"\n{'='*70}\n[cross-sequence calibration] calib={args.calib_track} -> test={args.test_track}"
          f"\n{'='*70}")
    print(f"kappa native (fit on test track itself): {kappa_native:.3f}")
    print(f"kappa transferred (fit on {calib_name}):  {kappa_transfer:.3f}")
    print(f"RQ-H reduction, native kappa:      {native['mean']*100:+.2f}pp "
          f"[{native['lo']*100:.2f},{native['hi']*100:.2f}] (sig={native['significant']})")
    print(f"RQ-H reduction, transferred kappa: {transferred['mean']*100:+.2f}pp "
          f"[{transferred['lo']*100:.2f},{transferred['hi']*100:.2f}] (sig={transferred['significant']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
