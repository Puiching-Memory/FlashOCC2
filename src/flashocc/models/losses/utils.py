"""损失函数工具."""
import torch


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """对 loss 施加权重和 reduction."""
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        elif reduction != "none":
            raise ValueError(f"Invalid reduction: {reduction}")
    return loss


__all__ = ["weight_reduce_loss"]
