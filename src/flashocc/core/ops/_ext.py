"""CUDA 扩展加载器.

统一管理所有 C++/CUDA 扩展的加载。支持:
1. 预编译扩展 (pip install 安装)
2. JIT 编译 (torch.utils.cpp_extension)

所有 C++/CUDA 源码位于 ``ops/csrc/`` 目录下。
"""
import os
import warnings

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")


def _try_load_ext(name, sources=None, extra_cuda_cflags=None):
    """尝试加载 CUDA 扩展."""
    # 先尝试已安装的扩展
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        pass

    # JIT 编译
    if sources:
        try:
            from torch.utils.cpp_extension import load as _load
            return _load(
                name=name,
                sources=sources,
                extra_cuda_cflags=extra_cuda_cflags or ["-O3"],
                verbose=False,
            )
        except Exception as e:
            warnings.warn(f"无法 JIT 编译 CUDA 扩展 {name}: {e}")
    return None


# ---- BEV Pool v2 ----
try:
    import bev_pool_v2_ext  # type: ignore
except ImportError:
    _bev_pool_v2_src = os.path.join(_CSRC_DIR, "bev_pool_v2")
    bev_pool_v2_ext = _try_load_ext(
        "bev_pool_v2_ext",
        sources=[
            os.path.join(_bev_pool_v2_src, "bev_pool.cpp"),
            os.path.join(_bev_pool_v2_src, "bev_pool_cuda.cu"),
        ],
        extra_cuda_cflags=["-O3"],
    )


# ---- BEV Pool v3 ----
try:
    import bev_pool_v3_ext  # type: ignore
except ImportError:
    _bev_pool_v3_src = os.path.join(_CSRC_DIR, "bev_pool_v3")
    bev_pool_v3_ext = _try_load_ext(
        "bev_pool_v3_ext",
        sources=[
            os.path.join(_bev_pool_v3_src, "bev_pool_v3.cpp"),
            os.path.join(_bev_pool_v3_src, "bev_pool_v3_cuda.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    )


__all__ = ["bev_pool_v2_ext", "bev_pool_v3_ext"]
