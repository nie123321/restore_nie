"""Run the agreed A/B/C ablations and all four validation evaluations serially."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
DATA = Path(r"M:\picture data\cholec80_t\train_test")
REFERENCE = ROOT / "runs" / "demo_10k_bs8_cos_20260921"
EVALUATOR = DATA / "HVI_CIDNet" / "evaluate_four_metrics.py"
GROUPS = {"A": ("direct", "off"), "B": ("structured", "off"),
          "C": ("structured", "static"), "D": ("structured", "conditional")}
RECIPE = dict(steps=10000, batch_size=8, lr=2e-4, lr_schedule="cosine", min_lr=2e-6,
              weight_decay=1e-4, grad_clip=1.0, seed=100, workers=0, width=24,
              val_every=500, save_every=1000, log_every=50, device="cuda")
METRICS = ["output_psnr", "output_ssim", "output_lpips_alex", "output_ciede2000"]


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def preflight():
    with (DATA / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for split, count in (("train", 1500), ("val", 200)):
        selected = [r for r in rows if r["split"] == split]
        if len(selected) != count or len({r["sample_id"] for r in selected}) != count:
            raise RuntimeError(f"Unexpected {split} manifest count")
        for kind in ("lowlight", "gt"):
            expected = {Path(r[f"{kind}_relpath"]).name for r in selected}
            actual = {p.name for p in (DATA / split / kind).glob("*.png")}
            if expected != actual:
                raise RuntimeError(f"{split}/{kind} filenames differ from manifest")
    reference = checkpoint(REFERENCE / "best_val.pt")
    config = reference["config"]
    for key, value in RECIPE.items():
        actual = config["model"]["width"] if key == "width" else config.get(key)
        if key == "log_every":  # Not recorded in the old training config.
            continue
        if actual != value:
            raise RuntimeError(f"Reference recipe differs at {key}: {actual} != {value}")
    if config["model"] != dict(width=24, spectral_mode="conditional", output_mode="structured"):
        raise RuntimeError("Unexpected reference architecture")
    return sorted((r for r in rows if r["split"] == "val"), key=lambda r: int(r["sample_id"]))


class Queue:
    def __init__(self, directory, rows):
        self.root, self.rows = directory, rows
        self.code = directory / "code"
        self.logs = directory / "logs"
        self.logs.mkdir(exist_ok=True)
        if (directory / "queue_status.json").exists():
            self.state = read_json(directory / "queue_status.json")
        else:
            self.state = dict(created_at=now(), groups={g: {"state": "pending"} for g in GROUPS})
        self.state.update(state="running", pid=os.getpid(), active=None)
        self.persist()

    def persist(self):
        self.state["updated_at"] = now()
        save_json(self.root / "queue_status.json", self.state)

    def command(self, group, stage, argv):
        log = self.logs / f"{group}_{stage}.log"
        print(f"{now()} START {group} {stage}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{now()} {json.dumps(argv, ensure_ascii=False)}\n")
            handle.flush()
            child = subprocess.Popen(argv, cwd=self.code, stdout=handle, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            self.state["active"] = dict(group=group, stage=stage, child_pid=child.pid, log=str(log))
            self.persist()
            result = child.wait()
        if result:
            raise RuntimeError(f"{group} {stage} exited {result}; see {log}")
        print(f"{now()} FINISH {group} {stage}", flush=True)

    def train(self, group):
        run = self.root / f"{group}_train"
        final = run / "step_010000.pt"
        if final.exists() and checkpoint(final)["step"] == 10000:
            if not (run / "best_val.pt").is_file():
                raise RuntimeError(f"Completed {group} has no best validation checkpoint")
            return run
        output, spectral = GROUPS[group]
        argv = [sys.executable, "-u", "-X", "utf8", str(self.code / "run_demo.py"), "train",
                "--data-root", str(DATA), "--run-dir", str(run), "--amp",
                "--output-mode", output, "--spectral-mode", spectral]
        for key, value in RECIPE.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        last = run / "last.pt"
        if last.exists():
            restored = checkpoint(last)
            completed = restored["step"]
            # Preserve records beyond the last committed checkpoint before replay.
            history = run / "loss.jsonl"
            if history.exists():
                lines = history.read_text(encoding="utf-8").splitlines()
                keep, abandoned = [], []
                for line in lines:
                    try:
                        valid = json.loads(line)["step"] <= completed
                    except (ValueError, KeyError):
                        valid = False
                    (keep if valid else abandoned).append(line)
                if abandoned:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    (run / f"interrupted_{stamp}.jsonl").write_text("\n".join(abandoned) + "\n", encoding="utf-8")
                    temp = history.with_suffix(".jsonl.tmp")
                    temp.write_text("\n".join(keep) + "\n", encoding="utf-8")
                    os.replace(temp, history)
            argv.extend(["--resume", str(last)])
        elif run.exists() and any(run.iterdir()):
            # A startup failure before the first checkpoint can safely restart.
            archive = self.root / (run.name + "_interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            if run.resolve().parent != self.root.resolve() or archive.resolve().parent != self.root.resolve():
                raise RuntimeError("Unsafe run archive path")
            run.rename(archive)
        self.command(group, "train", argv)
        status = read_json(run / "status.json")
        if status["state"] != "complete" or status["step"] != 10000 or not final.is_file():
            raise RuntimeError(f"{group} did not finish 10000 steps")
        return run

    def evaluate(self, group, run):
        weight = run / "best_val.pt"
        ckpt = checkpoint(weight)
        expected = dict(width=24, output_mode=GROUPS[group][0], spectral_mode=GROUPS[group][1])
        if ckpt["config"]["model"] != expected:
            raise RuntimeError(f"Wrong architecture for {group}")
        weight_sha = digest(weight)
        root = self.root / f"{group}_val"
        root.mkdir(exist_ok=True)
        status_path = root / "status.json"
        if status_path.exists():
            done = read_json(status_path)
            if done.get("state") == "complete" and done.get("checkpoint_sha256") == weight_sha:
                return done
        attempt = 1
        while (root / f"attempt_{attempt:02d}").exists():
            attempt += 1
        experiment = root / f"attempt_{attempt:02d}"
        experiment.mkdir()
        fields = ["sample_id", "video", "lowlight_relpath", "gt_relpath", "output_filename"]
        with (experiment / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({**{k: row[k] for k in fields[:-1]},
                                 "output_filename": Path(row["lowlight_relpath"]).name})
        self.command(group, "infer_val", [sys.executable, "-u", "-X", "utf8", str(self.code / "run_demo.py"),
                     "infer", "--checkpoint", str(weight), "--input", str(DATA / "val" / "lowlight"),
                     "--output-dir", str(experiment / "enhanced"), "--device", "cuda"])
        expected_names = {Path(r["lowlight_relpath"]).name for r in self.rows}
        if {p.name for p in (experiment / "enhanced").glob("*.png")} != expected_names:
            raise RuntimeError(f"{group}: incomplete validation predictions")
        self.command(group, "metrics_val", [sys.executable, "-u", "-X", "utf8",
                     str(self.code / "evaluate_four_metrics.py"), "--experiment-root", str(experiment),
                     "--method-label", f"Demo-ablation-{group}", "--dataset-root", str(DATA)])
        with (experiment / "metrics_per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        if len(metrics) != 200 or {r["sample_id"] for r in metrics} != {r["sample_id"] for r in self.rows}:
            raise RuntimeError(f"{group}: incomplete metrics")
        if not all(math.isfinite(float(r[k])) for r in metrics for k in METRICS):
            raise RuntimeError(f"{group}: nonfinite metrics")
        means = {k: sum(float(r[k]) for r in metrics) / len(metrics) for k in METRICS}
        done = dict(state="complete", group=group, checkpoint=str(weight), checkpoint_step=ckpt["step"],
                    checkpoint_sha256=weight_sha, selection="minimum validation raw L1",
                    samples=200, split="val", parameters=sum(v.numel() for k, v in ckpt["model"].items()
                    if k not in {"luminance_weights", "chroma_basis"}),
                    best_val_l1=ckpt["best_val"], metrics=means, experiment=str(experiment), completed_at=now())
        save_json(status_path, done)
        save_json(experiment / "evaluation_contract.json", done)
        return done

    def summarize(self):
        columns = ["group", "state", "checkpoint_step", "parameters", "output_psnr", "output_ssim",
                   "output_lpips_alex", "output_ciede2000"]
        records, markdown = [], ["# A/B/C/D validation comparison", "",
            "200 validation pairs; best checkpoint by raw validation L1; metrics computed on rounded RGB PNGs.",
            "", "| Group | State | Best step | Parameters | PSNR | SSIM | LPIPS | CIEDE2000 |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        for group, item in self.state["groups"].items():
            result = item.get("result", {})
            record = dict(group=group, state=item["state"], checkpoint_step=result.get("checkpoint_step", ""),
                          parameters=result.get("parameters", ""), **result.get("metrics", {}))
            records.append(record)
            cells = [f"{record[k]:.6f}" if k in METRICS and k in record else str(record.get(k, "")) for k in columns]
            markdown.append("| " + " | ".join(cells) + " |")
        with (self.root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        (self.root / "COMPARISON.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    def execute(self):
        # D's existing checkpoint exercises the evaluation pipeline before the long jobs.
        for group in ("D", "A", "B", "C"):
            item = self.state["groups"][group]
            if item["state"] == "complete":
                continue
            for attempt in range(2):
                item.update(state="running", attempt=attempt + 1, started_at=now())
                self.persist()
                self.summarize()
                try:
                    run = REFERENCE if group == "D" else self.train(group)
                    item.update(state="complete", result=self.evaluate(group, run), completed_at=now())
                    self.state["active"] = None
                    self.persist()
                    self.summarize()
                    break
                except Exception as error:
                    item.update(state="retry_pending" if attempt == 0 else "failed", error=str(error))
                    self.persist()
                    self.summarize()
                    traceback.print_exc()
                    if attempt == 1:
                        self.state.update(state="failed", active=None)
                        self.persist()
                        raise
                    time.sleep(5)
        self.state.update(state="complete", active=None, completed_at=now())
        self.persist()
        self.summarize()
        print("ALL_ABLATIONS_AND_VALIDATION_METRICS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=ROOT / "runs" / "ablation_10k_seed100_20260921")
    args = parser.parse_args()
    directory = args.queue_dir.resolve()
    if directory.parent != (ROOT / "runs").resolve():
        raise ValueError("Queue must be a direct child of this demo's runs directory")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".queue.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("QUEUE_ALREADY_RUNNING", flush=True)
            return
        rows = preflight()
        code = directory / "code"
        code.mkdir(exist_ok=True)
        for source in (ROOT / "model.py", ROOT / "run_demo.py", EVALUATOR):
            destination = code / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
        contract = dict(recipe=RECIPE, groups=GROUPS, reference_run=str(REFERENCE), validation_samples=200,
                        execution_order=["D evaluation", "A train/evaluate", "B train/evaluate", "C train/evaluate"],
                        code_sha256={p.name: digest(p) for p in code.glob("*.py")},
                        split_manifest_sha256=digest(DATA / "split_manifest.csv"))
        contract_path = directory / "queue_contract.json"
        if contract_path.exists() and read_json(contract_path) != json.loads(json.dumps(contract)):
            raise RuntimeError("Frozen queue contract differs; investigate before resuming")
        save_json(contract_path, contract)
        Queue(directory, rows).execute()


if __name__ == "__main__":
    main()
