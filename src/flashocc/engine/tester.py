"""测试引擎."""
import numpy as np
import torch
from flashocc.core.log import progress_bar
from .trainer import scatter_data
from flashocc.datasets.dali_decode import dali_decode_batch


def single_gpu_test(model, data_loader, show=False, out_dir=None, *,
                    amp_dtype=None, use_channels_last=False,
                    non_blocking=False, img_color_order="RGB", **kwargs):
    """单 GPU 测试.

    Args:
        model: 模型。
        data_loader: 测试数据加载器。
        show: 是否可视化。
        out_dir: 输出目录。
        amp_dtype: AMP 推理数据类型 (如 torch.bfloat16), None 表示不启用。
        use_channels_last: 是否将输入图像转换为 channels_last 格式。
        non_blocking: scatter_data 是否使用 non_blocking 传输。

    Returns:
        list: 预测结果列表。
    """
    model.eval()
    use_amp = amp_dtype is not None
    results = []
    with torch.inference_mode():
        for data in progress_bar(data_loader, desc="Testing"):
            data = scatter_data(data, non_blocking=non_blocking)
            if 'jpeg_bytes' in data:
                data = dali_decode_batch(data, color_order=img_color_order)
            # channels_last: 将输入图像转换为 NHWC 格式
            if use_channels_last and 'img_inputs' in data:
                img_inputs = data['img_inputs']
                if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0:
                    imgs = img_inputs[0]
                    if hasattr(imgs, 'ndim') and imgs.ndim == 5:
                        B, N, C, H, W = imgs.shape
                        imgs = imgs.view(B * N, C, H, W).to(
                            memory_format=torch.channels_last).view(B, N, C, H, W)
                        data['img_inputs'] = [imgs] + list(img_inputs[1:])
            # AMP 推理
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                result = model(return_loss=False, **data)
            # GPU tensor → numpy uint8 (在编译图之外, 避免 CUDA graph 分区)
            if isinstance(result, torch.Tensor) and result.is_cuda:
                result = list(result.cpu().numpy().astype(np.uint8))
            if hasattr(result, '__iter__') and not hasattr(result, 'keys') and not hasattr(result, 'shape'):
                results.extend(result)
            else:
                results.append(result)
    return results


__all__ = ["single_gpu_test"]
