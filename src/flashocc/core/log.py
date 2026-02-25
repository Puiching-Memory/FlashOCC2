"""统一日志 & 进度条 — 基于 loguru + tqdm.

设计要点
--------
1. 全局唯一 ``logger`` 实例 (``from flashocc.core.log import logger``)。
2. DDP 模式下, 非 rank-0 进程自动静默 (logger + tqdm)。
3. ``setup_logger()`` 在入口脚本 (train / test) 中调用一次即可。
4. ``progress_bar()`` 封装 tqdm, 非 rank-0 自动 ``disable=True``。
"""

from __future__ import annotations

import os
import sys

from loguru import logger
from tqdm import tqdm as _tqdm


# ── 内部状态 ─────────────────────────────────────────────
_setup_done: bool = False

# 默认格式
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_LOG_FORMAT_SIMPLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - "
    "<level>{level: <8}</level> - "
    "<level>{message}</level>"
)


def _is_rank0() -> bool:
    """判断当前进程是否是 rank-0 (兼容非分布式)."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0


def setup_logger(
    log_file: str | None = None,
    level: str = "INFO",
    rank0_only: bool = True,
    simple_format: bool = True,
) -> None:
    """初始化全局 loguru logger.

    Parameters
    ----------
    log_file : str | None
        日志文件路径 (可选)。仅 rank-0 写入。
    level : str
        日志级别, 默认 ``"INFO"``。
    rank0_only : bool
        DDP 模式下是否仅 rank-0 输出。
    simple_format : bool
        使用简洁格式 (True) 或详细格式 (False)。
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    fmt = _LOG_FORMAT_SIMPLE if simple_format else _LOG_FORMAT

    # 移除默认 handler
    logger.remove()

    is_rank0 = _is_rank0()

    if not rank0_only or is_rank0:
        # stderr → 控制台 (不与 tqdm 冲突, 因为 loguru 默认写 stderr)
        logger.add(
            sys.stderr,
            format=fmt,
            level=level,
            colorize=True,
            enqueue=True,   # 多进程安全
        )
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            logger.add(
                log_file,
                format=fmt,
                level=level,
                rotation="500 MB",
                enqueue=True,
            )
    else:
        # 非 rank-0: 添加一个永远丢弃的 sink
        logger.add(lambda _: None, level="CRITICAL", format=fmt)


# ── 进度条 ───────────────────────────────────────────────

def progress_bar(iterable=None, total=None, desc=None, rank0_only=True, **kwargs):
    """loguru + tqdm 友好的进度条.

    非 rank-0 自动 ``disable=True``，避免多卡干扰。
    """
    disable = rank0_only and not _is_rank0()
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        disable=disable,
        dynamic_ncols=True,
        **kwargs,
    )


__all__ = ["logger", "setup_logger", "progress_bar"]
