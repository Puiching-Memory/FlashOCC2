"""测试引擎."""
import torch
from flashocc.core.log import progress_bar
from .trainer import scatter_data
from flashocc.datasets.dali_decode import dali_decode_batch


@torch.no_grad()
def single_gpu_test(model, data_loader, show=False, out_dir=None, **kwargs):
    """单 GPU 测试.

    Args:
        model: 模型。
        data_loader: 测试数据加载器。
        show: 是否可视化。
        out_dir: 输出目录。

    Returns:
        list: 预测结果列表。
    """
    model.eval()
    results = []
    for data in progress_bar(data_loader, desc="Testing"):
        data = scatter_data(data)
        if 'jpeg_bytes' in data:
            data = dali_decode_batch(data)
        result = model(return_loss=False, **data)
        if hasattr(result, '__iter__') and not hasattr(result, 'keys') and not hasattr(result, 'shape'):
            results.extend(result)
        else:
            results.append(result)
    return results


__all__ = ["single_gpu_test"]
