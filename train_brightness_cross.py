"""A+four-map brightness cross-attention+MDTA, same 55-epoch GT-Mean recipe."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
NAME = "a_mdta_brightness_cross4_gtmean_crop192x384_hv_b8_e55_seed100_20260929"
REFERENCE_NAME = "a_mdta_gtmean_crop192x384_hv_b8_e55_seed100_20260928"


def build_command(root):
    reference = root / "runs" / REFERENCE_NAME / "config.json"
    config = json.loads(reference.read_text(encoding="utf-8"))
    expected = dict(width=24, spectral_mode="off", output_mode="direct", bottleneck_attention="mdta")
    if config["model"] != expected or config["loss_mode"] != "gt-mean-l1":
        raise ValueError("Reference must be unchanged A+bottom MDTA+GT-Mean")
    if (config["steps"] != 10340 or config["epochs"] != 55 or config["seed"] != 100
            or config["batch_size"] != 8 or config["crop_size"] != [192, 384]
            or not config["flip_hv"] or config.get("geometric_augs", False)
            or config["gt_mean_sigma"] != 0.1):
        raise ValueError("Reference must retain seed100, crop/HV, batch8, 55 epochs / 10340 steps")
    run = root / "runs" / NAME
    command = [sys.executable, "-B", "-u", "-X", "utf8", str(root / "run_demo.py"),
               "train", "--run-dir", str(run), "--data-root", config["data_root"]]
    keys = {"epochs": "epochs", "batch-size": "batch_size", "seed": "seed", "workers": "workers",
            "lr": "lr", "lr-schedule": "lr_schedule", "min-lr": "min_lr", "weight-decay": "weight_decay",
            "grad-clip": "grad_clip", "best-metric": "best_metric"}
    for flag, key in keys.items():
        command.extend(["--" + flag, str(config[key])])
    command.extend(["--width", "24", "--spectral-mode", "off", "--output-mode", "direct",
                    "--bottleneck-attention", "mdta", "--brightness-prior-attention", "bottom-cross4",
                    "--loss-mode", "gt-mean-l1", "--gt-mean-sigma", "0.1",
                    "--crop-size", "192", "384", "--flip-hv", "--val-every", "500", "--save-every", "500",
                    "--device", "cuda", "--amp" if config["amp"] else "--no-amp", "--log-every", "50"])
    return command, config, run, reference


def main():
    command, config, run, reference = build_command(ROOT)
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Run directory is not empty: {run}")
    result = dict(reference=str(reference), parameters=332939, added_parameters=87220,
                  intervention="four fixed low-RGB brightness maps; encoder4/16/32/96; mainQ/priorKV cross-attention+2xFFN before unchanged MDTA",
                  initialization="from scratch seed100; shared A+MDTA initialization retained",
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
                        "A-BrightnessCross4-MDTA-GTMean-crop192x384-HV-55ep-test", "--device", "cuda"], check=True)
        result.update(state="complete", updated_at=datetime.now().isoformat())
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("A_BRIGHTNESS_CROSS4_TRAIN_AND_TEST_COMPLETE", flush=True)
    except Exception as error:
        if run.exists():
            result.update(state="failed", error=str(error), updated_at=datetime.now().isoformat())
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
