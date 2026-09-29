"""Five-site four-map guidance; same 55-epoch crop/HV GT-Mean recipe."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from train_brightness_cross import build_command as build_bottom_command


ROOT = Path(__file__).resolve().parent
NAME = "a_mdta_brightness_cross4_multiscale_gtmean_crop192x384_hv_b8_e55_seed100_20260929"
BOTTOM_NAME = "a_mdta_brightness_cross4_gtmean_crop192x384_hv_b8_e55_seed100_20260929"


def build_command(root):
    command, config, _, reference = build_bottom_command(root)
    run = root / "runs" / NAME
    command[command.index("--run-dir") + 1] = str(run)
    command[command.index("--brightness-prior-attention") + 1] = "multiscale-cross4"
    return command, config, run, reference


def main():
    command, config, run, reference = build_command(ROOT)
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Run directory is not empty: {run}")
    result = dict(reference_base=str(reference), reference_bottom=str(ROOT / "runs" / BOTTOM_NAME),
                  parameters=388473, added_parameters_vs_base=142754, added_parameters_vs_bottom=55534,
                  intervention="same four maps and bottom block; add full/half encoder and half/full decoder cross-attention+2xFFN; shared prior pyramid",
                  initialization="from scratch seed100; common A+MDTA and bottom-guidance initialization retained",
                  steps=10340, epochs=55, val_every=500, save_every=500, train_command=command,
                  test_policy="one raw-val-L1-best test; every500-step checkpoint retained")
    print(json.dumps(dict(phase="training", **result), ensure_ascii=False), flush=True)
    result_path = run / "experiment_result.json"
    try:
        subprocess.run(command, check=True)
        import torch
        best = run / "best_val.pt"
        payload = torch.load(best, map_location="cpu", weights_only=False)
        step = payload["step"]
        del payload
        output = run / f"test_best{step}_{datetime.now():%Y%m%d}"
        result.update(state="testing", best_step=step, output=str(output), updated_at=datetime.now().isoformat())
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, "-B", "-u", "-X", "utf8", str(ROOT / "test_best.py"),
                        "--checkpoint", str(best), "--data-root", config["data_root"],
                        "--output-dir", str(output), "--method-label",
                        "A-BrightnessCross4-MultiScale-MDTA-GTMean-crop192x384-HV-55ep-test", "--device", "cuda"], check=True)
        result.update(state="complete", updated_at=datetime.now().isoformat())
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("A_BRIGHTNESS_MULTISCALE_TRAIN_AND_TEST_COMPLETE", flush=True)
    except Exception as error:
        if run.exists():
            result.update(state="failed", error=str(error), updated_at=datetime.now().isoformat())
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
