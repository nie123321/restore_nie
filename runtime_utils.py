"""Atomic run ownership and bounded code provenance; no dataset mutations."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

import torch


def configure_fp32():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".j.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunLock:
    """OS releases ownership on process death. Keep the file to avoid inode races."""
    def __init__(self, output):
        self.path = Path(output) / ".run.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600), "r+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"Output is owned by another process: {self.path.parent}") from error
        self.stream.seek(1)
        self.stream.truncate()
        self.stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                     "acquired_unix": time.time()}).encode())
        self.stream.flush()
        return self

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def output_entries(output):
    return [path for path in Path(output).iterdir() if path.name != ".run.lock"]


def source_manifest(source):
    source = Path(source)
    root_names = {"train.py", "test.py", "dataset.py", "losses.py", "metrics.py",
                  "runtime_utils.py", "config.json", "variant.json", "requirements.txt",
                  "train_server.sh", "train_local.ps1"}
    paths = [source / name for name in root_names if (source / name).is_file()]
    paths += [p for p in source.glob("config*.json") if p not in paths]
    paths += [p for p in (source / "models").rglob("*.py") if "__pycache__" not in p.parts]
    entries = [{"path": path.relative_to(source).as_posix(), "sha256": sha256_file(path)}
               for path in sorted(paths, key=lambda p: p.relative_to(source).as_posix())]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "files": entries}


def snapshot_source(source, output, manifest):
    destination = Path(output) / "code"
    if destination.exists():
        recorded = destination / "source_manifest.json"
        if not recorded.is_file() or json.loads(recorded.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Code/config changed since this run; preserve the old run and use a new output")
        for entry in manifest["files"]:
            path = destination / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Run source snapshot changed: {entry['path']}")
        return
    temporary = Path(output) / f".c.{uuid.uuid4().hex[:12]}.tmp"
    try:
        temporary.mkdir()
        for entry in manifest["files"]:
            target = temporary / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source) / entry["path"], target)
        atomic_json(temporary / "source_manifest.json", manifest)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            # Invocation-owned temporary tree strictly inside its run output.
            if temporary.resolve().parent != Path(output).resolve():
                raise RuntimeError("Temporary source snapshot escaped its output directory")
            shutil.rmtree(temporary)


def split_manifest_info(root):
    if root is not None:
        path = Path(root) / "split_manifest.csv"
        if path.is_file():
            return {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return None
