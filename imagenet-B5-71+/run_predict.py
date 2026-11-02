"""Run the recorded Stage8 six-view FP32 prediction and validate its ZIP."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from landcover.bootstrap import ROOT, SEGMENTATION


def _arguments():
    options = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "images", "output"):
        options.add_argument("--" + name, type=Path, required=True)
    options.add_argument("--ema-decay", type=float, default=0.995)
    return options.parse_args()


def _environment():
    values = dict(os.environ)
    values.update(AIC_PROJECT_ROOT=str(ROOT), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    values["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(SEGMENTATION),
                                          values.get("PYTHONPATH", "")))
    return values


def main():
    options = _arguments()
    weights, images, output = (getattr(options, key).resolve()
                               for key in ("checkpoint", "images", "output"))
    if not weights.is_file() or not images.is_dir():
        raise FileNotFoundError("Check the checkpoint and Test2 image directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prediction requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    environment = _environment()

    def command(module, *arguments):
        invocation = [sys.executable, "-u", "-m", "landcover.commands." + module]
        subprocess.run(invocation + list(map(str, arguments)), cwd=ROOT,
                       env=environment, check=True)

    import torch
    record = torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
    backup = any(key.startswith("ema_") for key in record["state_dict"])
    del record
    inference_weights = output / "ema_inference.pth" if backup else weights
    if backup:
        command("export_average", weights, inference_weights, "--ema-decay", options.ema_decay)
    command("sixview_predict", "--config", ROOT / "experiments" / "prediction_sixview.py",
            "--checkpoint", inference_weights, "--original-checkpoint", weights,
            "--test-dir", images, "--output-root", output)
    submission = output / "stage8_refresh_stage7_832_best_p1_test2.zip"
    command("seal_submission", "--prediction-dir", output / "test2_predictions",
            "--test-dir", images, "--output", submission, "--expected-count", 1300,
            "--checkpoint", weights)
    report_path = output / "inference_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(submission.with_suffix(".zip.manifest.json").read_text(encoding="utf-8"))
    if report["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during prediction")
    report.update(status="complete", submission=str(submission),
                  zip_sha256=manifest["zip_sha256"], validated_png_count=1300)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE", submission)


if __name__ == "__main__":
    main()
