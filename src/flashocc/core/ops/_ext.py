"""CUDA 扩展加载器.

统一管理所有 C++/CUDA 扩展的加载。支持:
1. 预编译扩展 (pip install 安装)
2. JIT 编译 (torch.utils.cpp_extension)

所有 C++/CUDA 源码位于 ``ops/csrc/`` 目录下。
环境由 uv 管理, CUDA 工具链应在安装阶段就绑定. 加载失败时直接报错。
"""
import importlib
import os

from torch.utils.cpp_extension import load as _jit_load

_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")


def _load_ext(name: str, sources: list[str],
              extra_cuda_cflags: list[str] | None = None):
    """加载 CUDA 扩展: 优先使用已安装的, 否则 JIT 编译.

    Raises:
        RuntimeError: JIT 编译和预编译均失败.
    """
    # 优先: 已安装的预编译扩展
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    # 后备: JIT 编译
    return _jit_load(
        name=name,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags or ["-O3"],
        verbose=False,
    )


# ---- BEV Pool v2 ----
_bev_pool_v2_src = os.path.join(_CSRC_DIR, "bev_pool_v2")
bev_pool_v2_ext = _load_ext(
    "bev_pool_v2_ext",
    sources=[
        os.path.join(_bev_pool_v2_src, "bev_pool.cpp"),
        os.path.join(_bev_pool_v2_src, "bev_pool_cuda.cu"),
    ],
    extra_cuda_cflags=["-O3"],
)

# ---- BEV Pool v3 ----
_bev_pool_v3_src = os.path.join(_CSRC_DIR, "bev_pool_v3")
bev_pool_v3_ext = _load_ext(
    "bev_pool_v3_ext",
    sources=[
        os.path.join(_bev_pool_v3_src, "bev_pool_v3.cpp"),
        os.path.join(_bev_pool_v3_src, "bev_pool_v3_cuda.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


__all__ = ["bev_pool_v2_ext", "bev_pool_v3_ext"]
