"""环境信息收集与多进程设置."""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 环境信息
# ---------------------------------------------------------------------------

def collect_env() -> dict[str, str]:
    """收集运行环境信息."""
    import torch

    env: dict[str, str] = {
        "sys.platform": sys.platform,
        "Python": sys.version.replace("\n", ""),
        "PyTorch": torch.__version__,
        "CUDA available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env["CUDA"] = torch.version.cuda or "N/A"
        env["GPU"] = torch.cuda.get_device_name(0)
        env["CUDA_HOME"] = os.environ.get("CUDA_HOME", "N/A")

    import torchvision
    env["TorchVision"] = torchvision.__version__

    import flashocc
    env["FlashOCC"] = flashocc.__version__

    try:
        import subprocess
        gcc = subprocess.check_output("gcc --version", shell=True, text=True).split("\n")[0]
        env["GCC"] = gcc
    except Exception:
        env["GCC"] = "N/A"
    return env


def setup_multi_processes(cfg):
    """根据配置设置多进程环境变量."""
    getter = getattr(cfg, "get", None)
    if getter:
        omp = getter("omp_num_threads", None)
        mkl = getter("mkl_num_threads", None)
        mp_start = getter("mp_start_method", None)
    else:
        omp = mkl = mp_start = None

    if omp is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp)
    if mkl is not None:
        os.environ["MKL_NUM_THREADS"] = str(mkl)
    if mp_start is not None:
        import torch.multiprocessing as mp
        mp.set_start_method(mp_start, force=True)
