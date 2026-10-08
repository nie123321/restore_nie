"""Run this folder's fixed ablation training from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.launcher import main


if __name__ == "__main__":
    main()
