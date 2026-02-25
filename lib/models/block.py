import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton

from flash_mla import get_mla_metadata, flash_mla_with_kvcache
torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.bfloat16)
# MAX_Speed:b=128, s_q=2, mean_sk=8192, h_q=128, h_kv=1, d=576, dv=512, causal=True, varlen=False
s_q = 2 # query seq len
h_q = 128 # query head
h_kv  = 1 # key/value head
b = 128 # batch size
dv = 512 # value dimension
d = 576 # query dimension
mean_sk = 8192 # mean seq len

cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)
print(tile_scheduler_metadata, num_splits)

max_seqlen = cache_seqlens.max().item()
max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
q_i = torch.randn(b, s_q, h_q, d)
block_size = 64
block_table = torch.arange(
    b * max_seqlen_pad // block_size, dtype=torch.int32
).view(b, max_seqlen_pad // block_size)
blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
for i in range(b):
    blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
        float("nan")
    )

kvcache_i = blocked_k

for i in range(1):
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, causal=False,
    )
    print(o_i.shape, lse_i.shape)

# if __name__ == "__main__":
#     model = nn.Sequential(
#         MultiHeadLatentAttention(),
#     ).to("cuda:0")
#     loss = nn.MSELoss().to("cuda:0")
#     opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

#     with torch.profiler.profile(
#             activities=[
#                 #torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
#             schedule=torch.profiler.schedule(
#                 wait=1,  # 前1步不采样
#                 warmup=1,  # 第2步作为热身，不计入结果
#                 active=3,  # 采集后面3步的性能数据
#                 repeat=1),  # 重复X轮
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log',use_gzip=True),  # 保存日志以供 TensorBoard 可视化
#             record_shapes=True,  # 记录输入张量的形状
#             profile_memory=True,  # 分析内存分配
#             with_stack=True,  # 记录操作的调用堆栈信息
#             with_flops=True,
#             with_modules=True,
#         ) as profiler:

#         for i in range(100):
#             inp = torch.randn(1, 10, 256).to("cuda:0")
#             opt.zero_grad()

#             out = model(inp)
#             loss_val = loss(out, inp)
#             loss_val.backward()
#             opt.step()

#             profiler.step()

#             print(loss_val)
#             print(inp)
#             print(out)

