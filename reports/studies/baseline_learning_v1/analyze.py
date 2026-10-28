"""Analyze the approved longer baseline without changing frozen pipeline sources."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/restoration_study_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils.checkpoint import load_checkpoint
from utils.logger import write_json
from utils.reproducibility import source_identity

HERE = Path(__file__).resolve().parent
RUN = ROOT / "experiments/studies/baseline_learning_v1/B_seed42"
OLD = ROOT / "reports/studies/mps_screening_v1"


def read_json(path: Path) -> dict:
    """Read retained evidence; missing artifacts fail rather than imply completion."""
    return json.loads(path.read_text())


def main() -> None:
    """Check selected replay, independently aggregate endpoints, plot observed data."""
    summary = read_json(RUN / "summary.json")
    report = read_json(HERE / "development.json")
    full = read_json(HERE / "full_image_development.json")
    old = read_json(OLD / "B_dev.json")
    jpeg_full = read_json(HERE / "jpeg_input_full_image_development.json")
    with (RUN / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    valid = [r for r in rows if r["worst_phase_psnr"]]
    selected = max(valid, key=lambda r: float(r["worst_phase_psnr"]))
    latest = load_checkpoint(RUN / "latest.pth")
    best = load_checkpoint(RUN / "best_model.pth")
    identity = source_identity()["files"]
    assert best["source"]["files"] == latest["source"]["files"] == identity
    assert best["config"] == latest["config"]
    assert latest["step"] == summary["step"] <= 5000
    updates = summary["step"]
    assert summary["completed"] == (updates == 5000)
    assert summary["skipped_updates"] == 0
    assert latest["cursor"] == 8 * updates and latest["microsteps"] == 4 * updates
    assert len(rows) == updates
    assert len(valid) == updates // 100 if summary["completed"] else updates // 100 - 1 <= len(valid) <= updates // 100
    assert best["step"] == report["step"] == int(selected["step"])
    assert math.isclose(report["worst_phase_psnr"], float(selected["worst_phase_psnr"]), rel_tol=0, abs_tol=1e-10)
    assert report["checkpoint_sha256"] == hashlib.sha256((RUN / "best_model.pth").read_bytes()).hexdigest()
    assert full["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert report["split"] == full["split"] == "val"
    assert report["count"] == full["count"] == 8
    assert report["dataset_hash"] == old["dataset_hash"]
    assert full["dataset_hash"] == jpeg_full["dataset_hash"]
    assert report["codec"] == old["codec"]
    assert report["phase_panel"] == old["phase_panel"]
    per_image = []
    for image in report["images"]:
        phases = image["phases"]
        assert len(phases) == image["count"] == 16
        worst = min(p["psnr"] for p in phases)
        input_worst = min(p["input_psnr"] for p in phases)
        assert math.isclose(worst, image["worst_phase_psnr"], rel_tol=0, abs_tol=1e-12)
        per_image.append({"id":image["id"], "worst_phase_psnr":worst,
                          "input_worst_phase_psnr":input_worst, "gain_db":worst-input_worst,
                          "mean_phase_psnr":float(np.mean([p["psnr"] for p in phases])),
                          "mean_phase_ssim":float(np.mean([p["ssim"] for p in phases]))})
    independent = {"worst_phase_psnr":float(np.mean([r["worst_phase_psnr"] for r in per_image])),
                   "psnr":float(np.mean([r["mean_phase_psnr"] for r in per_image])),
                   "ssim":float(np.mean([r["mean_phase_ssim"] for r in per_image]))}
    for key,value in independent.items():
        assert math.isclose(value, report[key], rel_tol=0, abs_tol=1e-12), key
    gain = report["worst_phase_psnr"] - report["input_worst_phase_psnr"]
    ssim_delta = report["ssim"] - report["input_mean_ssim"]
    result = {"status":"completed_and_replayed" if summary["completed"] else "incomplete_and_replayed", "diagnostic_only":True, "completed":summary["completed"], "stop_reason":summary["stop_reason"],
              "source_unchanged":True, "checkpoint_selection_and_replay_verified":True,
              "independent_aggregation":independent, "successful_updates":updates, "target_updates":5000,
              "selected_update":best["step"], "training_seconds":summary["elapsed_seconds"],
              "train_source_count":32, "development_source_count":8, "training_seed":42,
              "development_phases":16, "quality":20,
              "worst_phase_gain_over_input_db":gain, "mean_phase_ssim_delta_over_input":ssim_delta,
              "pragmatic_continuation_threshold":{"worst_phase_gain_at_least_db":0.10,
                                                    "no_mean_phase_ssim_decrease":True,
                                                    "passed":bool(summary["completed"] and gain >= 0.10 and ssim_delta >= 0)},
              "full_image_psnr":full["psnr"], "full_image_ssim":full["ssim"],
              "full_image_gain_over_input_db":full["psnr"]-jpeg_full["psnr"],
              "selected_checkpoint_sha256":report["checkpoint_sha256"],
              "latest_checkpoint_sha256":hashlib.sha256((RUN / "latest.pth").read_bytes()).hexdigest(),
              "per_image":per_image, "protected_test_scored":False,
              "novelty_established":False,
              "comparison_limit":"The 5000-update cosine horizon differs from the 1000-update screening. Budget/schedule diagnostic only, not matched inference about the consistency loss or an isolated duration effect. Same dev data select checkpoints; no held-out generalization claim.",
              "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(HERE / "summary.json", result)
    with (HERE / "validation_curve.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(valid[0]));writer.writeheader();writer.writerows(valid)
    with (HERE / "per_image.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for axis,key,label,reference in zip(axes,("worst_phase_psnr","ssim"),
        ("Mean image-worst-phase PSNR (dB)","Mean phase RGB SSIM"),
        (report["input_worst_phase_psnr"],report["input_mean_ssim"])):
        axis.plot([int(r["step"]) for r in valid],[float(r[key]) for r in valid],color="#228833",marker="o",markersize=2,label="B / 5000-update schedule")
        axis.axhline(reference,color="#777777",linestyle="--",label="JPEG input")
        if key=="worst_phase_psnr":
            axis.axhline(reference+0.10,color="#AA7733",linestyle=":",label="Pragmatic +0.10 dB threshold")
            axis.scatter([best["step"]],[report[key]],marker="X",s=80,color="#228833",zorder=4)
        axis.set_xlabel("Successful optimizer updates");axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend(frameon=False,fontsize=8)
    fig.suptitle("Longer baseline diagnostic: 8 development images, 16 phases, one seed")
    fig.savefig(HERE/"learning_curves.png",dpi=180);fig.savefig(HERE/"learning_curves.svg");plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!="per_image"},indent=2))


if __name__ == "__main__":
    main()
