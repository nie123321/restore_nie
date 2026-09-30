"""Single-GPU entry point for the official StarIR trainer."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import types


def install_scipy_shim():
    try:
        import scipy.ndimage.filters
        return
    except ModuleNotFoundError:
        pass
    from scipy import ndimage
    shim = types.ModuleType("scipy.ndimage.filters")
    for name in ("convolve", "correlate", "gaussian_filter", "gaussian_filter1d",
                 "uniform_filter", "uniform_filter1d", "median_filter",
                 "maximum_filter", "minimum_filter", "order_filter",
                 "rank_filter", "sobel", "prewitt"):
        if hasattr(ndimage, name):
            setattr(shim, name, getattr(ndimage, name))
    sys.modules["scipy.ndimage.filters"] = shim
    setattr(ndimage, "filters", shim)


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=repo / "cholec80_single_gpu_100k.yml")
    args = parser.parse_args()
    config = args.config.resolve()
    install_scipy_shim()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.argv = ["basicsr/train.py", "-opt", str(config), "--launcher", "none"]
    runpy.run_path(str(repo / "basicsr/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
