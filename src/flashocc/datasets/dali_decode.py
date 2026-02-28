"""NVIDIA DALI GPU 图像管道 — 全 DALI 实现, 零 PyTorch 后处理.

所有图像变换全部在 DALI GPU 管道中完成:
  1) DataLoader workers 读取 JPEG 原始字节 + 计算增广参数 (CPU)
  2) collate 后, GPU 上一次管道调用批量完成:
     nvJPEG 解码(BGR) → Resize → Crop(支持负偏移) → Rotate → Flip+Normalize+CHW
  3) 统一形状输出 → as_tensor() 批量拷贝到 PyTorch

典型耗时: ~20ms/batch (H800, 4×6=24 images, 1600×900 JPEG → 704×256 float32)
"""
from __future__ import annotations

import numpy as np
import torch

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.pytorch as dali_torch

from flashocc.constants import IMAGENET_MEAN, IMAGENET_STD


# =====================================================================
#  全 DALI Pipeline:
#  decode(BGR) → resize → slice(crop) → rotate → crop_mirror_normalize
# =====================================================================

class DALIBatchDecoder:
    """全 DALI GPU 图像处理管道.

    Pipeline (全部 GPU):
        1. fn.decoders.image — nvJPEG 硬件解码, 直接输出 BGR
        2. fn.resize — per-image resize
        3. fn.slice — per-image crop, out_of_bounds_policy='pad' 处理负坐标
        4. fn.rotate — per-image 旋转 (角度已补偿 flip 顺序)
        5. fn.crop_mirror_normalize — flip + normalize + HWC→CHW

    输出: (N, 3, tH, tW) float32 BGR normalized, 全部 shape 相同.

    归一化说明:
        原管道: PIL(RGB) → [::-1](BGR) → subtract IMAGENET_MEAN → divide IMAGENET_STD
        等效于: BGR 输入 + mean=[123.675, 116.28, 103.53] std=[58.395, 57.12, 57.375]
        直接解码 BGR 并用相同 mean/std 即完全复现.
    """

    def __init__(self, max_batch_size: int = 48, device_id: int = 0,
                 num_threads: int = 4, target_h: int = 256, target_w: int = 704):
        self.max_batch_size = max_batch_size
        self.device_id = device_id
        self.num_threads = num_threads
        self.target_h = target_h
        self.target_w = target_w
        self._pipe: Pipeline | None = None
        self._built_bs: int = 0

    def _ensure_pipeline(self, batch_size: int) -> None:
        if self._pipe is not None and batch_size == self._built_bs:
            return
        self._build_pipeline(batch_size)

    def _build_pipeline(self, batch_size: int) -> None:
        tH, tW = self.target_h, self.target_w
        pipe = Pipeline(
            batch_size=batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            prefetch_queue_depth=1,
            exec_async=False,
            exec_pipelined=False,
        )
        with pipe:
            # ---- External Sources (CPU → DALI 自动搬运到 GPU) ----
            jpegs = fn.external_source(name="jpegs", device="cpu",
                                       dtype=types.UINT8)
            resize_w = fn.external_source(name="resize_w", device="cpu")
            resize_h = fn.external_source(name="resize_h", device="cpu")
            crop_start = fn.external_source(name="crop_start", device="cpu")
            rotate_angle = fn.external_source(name="rotate_angle",
                                              device="cpu")
            do_flip = fn.external_source(name="do_flip", device="cpu")

            # 1) nvJPEG 硬件解码 → BGR (直接匹配原管道通道顺序)
            images = fn.decoders.image(jpegs, device="mixed",
                                       output_type=types.BGR,
                                       hw_decoder_load=0.7)

            # 2) Per-image GPU resize
            images = fn.resize(images,
                               resize_x=resize_w,
                               resize_y=resize_h,
                               interp_type=types.INTERP_LINEAR)

            # 3) Crop — fn.slice 支持负坐标, OOB 区域填 0 (等效 PIL.crop)
            images = fn.slice(images,
                              start=crop_start,
                              shape=[tH, tW],
                              axes=[0, 1],
                              out_of_bounds_policy="pad",
                              fill_values=0)

            # 4) Rotate — 角度已预补偿 flip 顺序:
            #    原管道: flip → rotate(θ)
            #    等价: rotate(-θ) → flip  (利用 R(θ)F = FR(-θ))
            images = fn.rotate(images,
                               angle=rotate_angle,
                               fill_value=0,
                               keep_size=True,
                               interp_type=types.INTERP_LINEAR)

            # 5) Flip + Normalize + HWC→CHW (融合内核)
            #    BGR 解码 + IMAGENET_MEAN/STD → 精确复现原管道:
            #    B − 123.675, G − 116.28, R − 103.53  (除以对应 std)
            images = fn.crop_mirror_normalize(
                images,
                crop=(tH, tW),
                mirror=do_flip,
                mean=[float(IMAGENET_MEAN[0]), float(IMAGENET_MEAN[1]),
                      float(IMAGENET_MEAN[2])],
                std=[float(IMAGENET_STD[0]), float(IMAGENET_STD[1]),
                     float(IMAGENET_STD[2])],
                output_layout="CHW",
                dtype=types.FLOAT,
            )

            pipe.set_outputs(images)
        pipe.build()
        self._pipe = pipe
        self._built_bs = batch_size

    @torch.no_grad()
    def decode_and_transform(
        self,
        jpeg_bytes_list: list[bytes],
        aug_params_list: list[tuple],
    ) -> torch.Tensor:
        """GPU 批量解码 + 全变换.

        Args:
            jpeg_bytes_list: N 个 JPEG 文件原始字节.
            aug_params_list: N 个 (resize_dims, crop, flip, rotate) 元组.

        Returns:
            (N, 3, target_h, target_w) float32 BGR normalized GPU tensor.
        """
        N = len(jpeg_bytes_list)
        device = f"cuda:{self.device_id}"
        tH, tW = self.target_h, self.target_w

        if N == 0:
            return torch.empty(0, 3, tH, tW, device=device)

        self._ensure_pipeline(N)

        # ---- 准备 per-image 参数 ----
        np_jpegs = [np.frombuffer(b, dtype=np.uint8) for b in jpeg_bytes_list]
        resize_ws = []
        resize_hs = []
        crop_starts = []
        rotate_angles = []
        flips = []

        for resize_dims, crop, flip, rotate_deg in aug_params_list:
            rw, rh = resize_dims
            resize_ws.append(np.array([rw], dtype=np.float32))
            resize_hs.append(np.array([rh], dtype=np.float32))

            cx, cy = crop[0], crop[1]
            crop_starts.append(np.array([int(cy), int(cx)], dtype=np.int32))

            # flip → rotate(θ) 等价于 rotate(-θ) → flip
            adj_angle = -float(rotate_deg) if flip else float(rotate_deg)
            rotate_angles.append(np.array([adj_angle], dtype=np.float32))

            flips.append(np.array([int(bool(flip))], dtype=np.int32))

        # ---- Feed & Run DALI (全 GPU) ----
        self._pipe.feed_input("jpegs", np_jpegs)
        self._pipe.feed_input("resize_w", resize_ws)
        self._pipe.feed_input("resize_h", resize_hs)
        self._pipe.feed_input("crop_start", crop_starts)
        self._pipe.feed_input("rotate_angle", rotate_angles)
        self._pipe.feed_input("do_flip", flips)

        output = self._pipe.run()
        dali_tl = output[0]  # TensorListGPU, each (3, tH, tW) float32

        # ---- Batch copy: DALI → PyTorch ----
        # 所有样本 shape 完全相同 → as_tensor() 一次批量传输
        result = torch.empty(N, 3, tH, tW, dtype=torch.float32, device=device)
        try:
            dense_tl = dali_tl.as_tensor()
            dali_torch.feed_ndarray(dense_tl, result)
        except Exception:
            # Fallback: per-image copy
            for i in range(N):
                dali_torch.feed_ndarray(dali_tl[i], result[i])

        return result


# =====================================================================
#  全局单例
# =====================================================================

_GLOBAL_DECODER: DALIBatchDecoder | None = None


def get_dali_decoder(device_id: int = 0,
                     target_h: int = 256,
                     target_w: int = 704) -> DALIBatchDecoder:
    global _GLOBAL_DECODER
    if _GLOBAL_DECODER is None or _GLOBAL_DECODER.device_id != device_id:
        _GLOBAL_DECODER = DALIBatchDecoder(
            max_batch_size=48,
            device_id=device_id,
            num_threads=4,
            target_h=target_h,
            target_w=target_w,
        )
    return _GLOBAL_DECODER


# =====================================================================
#  批量处理入口 (供 trainer / tester 调用)
# =====================================================================

@torch.no_grad()
def dali_decode_batch(data_batch: dict, target_h: int = 256,
                      target_w: int = 704) -> dict:
    """collate 后在 GPU 上批量 DALI 解码 + 全变换."""
    if "jpeg_bytes" not in data_batch:
        return data_batch

    jpeg_bytes_batch = data_batch.pop("jpeg_bytes")
    aug_params_batch = data_batch.pop("img_aug_params")

    device_id = torch.cuda.current_device()
    decoder = get_dali_decoder(device_id, target_h, target_w)

    B = len(jpeg_bytes_batch)
    N_cam = len(jpeg_bytes_batch[0])

    flat_bytes = []
    flat_params = []
    for b in range(B):
        for c in range(N_cam):
            flat_bytes.append(jpeg_bytes_batch[b][c])
            flat_params.append(aug_params_batch[b][c])

    # 全 DALI GPU: decode + resize + crop + rotate + flip + normalize
    imgs = decoder.decode_and_transform(flat_bytes, flat_params)
    imgs = imgs.view(B, N_cam, 3, target_h, target_w)

    # 更新 img_inputs
    img_inputs = data_batch["img_inputs"]
    if isinstance(img_inputs, (list, tuple)):
        data_batch["img_inputs"] = [imgs] + list(img_inputs[1:])

    return data_batch


__all__ = [
    "DALIBatchDecoder",
    "dali_decode_batch",
    "get_dali_decoder",
]
