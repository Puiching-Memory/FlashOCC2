import sys
import os
sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, base=10000, max_len=512):
        """
        RoPE位置编码模块

        Args:
            hidden_size (int): 模型维度
            num_heads (int): 注意力头数量
            base (int): 频率基值
            max_len (int): 最大序列长度
        """
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        self.cos_pos_cache, self.sin_pos_cache = self._compute_pos_emb()

    def _compute_pos_emb(self):
        theta_i = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(self.max_len)
        pos_emb = positions.unsqueeze(1) * theta_i.unsqueeze(0)

        cos_pos = pos_emb.sin().repeat_interleave(2, dim=-1)
        sin_pos = pos_emb.cos().repeat_interleave(2, dim=-1)

        return cos_pos, sin_pos

    def forward(self, q):
        """
        RoPE位置编码应用

        Args:
            q (torch.Tensor): 输入张量 [bs, num_heads, seq_len, head_dim]

        Returns:
            torch.Tensor: 应用位置编码后的张量
        """
        bs, seq_len = q.shape[0], q.shape[2]
        cos_pos = self.cos_pos_cache[:seq_len].to(q.device)  # [seq_len, head_dim]
        sin_pos = self.sin_pos_cache[:seq_len].to(q.device)  # [seq_len, head_dim]

        # 扩展维度以匹配batch和head维度
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

        # RoPE变换
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)  # 奇偶交替
        q2 = q2.reshape(q.shape).contiguous()

        return q * cos_pos + q2 * sin_pos

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size=256, down_dim=64, up_dim=128, num_heads=8, rope_head_dim=26, dropout_prob=0.0):
        """
        Multi-Head Latent Attention 实现

        Args:
            hidden_size (int): 输入特征维度
            down_dim (int): 降维后的维度
            up_dim (int): 升维后的维度
            num_heads (int): 注意力头数量
            rope_head_dim (int): RoPE编码的头维度
            dropout_prob (float): Dropout概率
        """
        super(MultiHeadLatentAttention, self).__init__()

        self.d_model = hidden_size
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads

        # 降维投影
        self.down_proj_kv = nn.Linear(hidden_size, down_dim)
        self.down_proj_q = nn.Linear(hidden_size, down_dim)

        # 升维投影
        self.up_proj_k = nn.Linear(down_dim, up_dim)
        self.up_proj_v = nn.Linear(down_dim, up_dim)
        self.up_proj_q = nn.Linear(down_dim, up_dim)

        # 解耦Q/K投影
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(hidden_size, rope_head_dim)

        # RoPE位置编码
        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)

        # 输出层
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, hidden_size)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, h, mask=None):
        """
        前向传播

        Args:
            h (torch.Tensor): 输入张量 [bs, seq_len, d_model]
            mask (torch.Tensor): 注意力掩码 [bs, seq_len]

        Returns:
            torch.Tensor: 输出张量 [bs, seq_len, d_model]
        """
        bs, seq_len, _ = h.size()

        # Step 1: 低秩转换
        c_t_kv = self.down_proj_kv(h)  # [bs, seq_len, down_dim]
        k_t_c = self.up_proj_k(c_t_kv)  # [bs, seq_len, up_dim]
        v_t_c = self.up_proj_v(c_t_kv)  # [bs, seq_len, up_dim]
        c_t_q = self.down_proj_q(h)  # [bs, seq_len, down_dim]
        q_t_c = self.up_proj_q(c_t_q)  # [bs, seq_len, up_dim]

        # Step 2: 解耦Q/K处理
        # RoPE投影处理
        q_t_r = self.proj_qr(c_t_q)  # [bs, seq_len, rope_head_dim*num_heads]
        q_t_r = q_t_r.view(bs, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)  # [bs, num_heads, seq_len, rope_head_dim]
        q_t_r = self.rope_q(q_t_r)  # 应用RoPE编码

        k_t_r = self.proj_kr(h)  # [bs, seq_len, rope_head_dim]
        k_t_r = k_t_r.unsqueeze(1)  # [bs, 1, seq_len, rope_head_dim]
        k_t_r = self.rope_k(k_t_r)  # 应用RoPE编码

        # Step 3: 注意力计算
        # Q/K/V维度调整
        q_t_c = q_t_c.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)  # [bs, num_heads, seq_len, up_dim/num_heads]
        q = torch.cat([q_t_c, q_t_r], dim=-1)  # [bs, num_heads, seq_len, (up_dim+rope_head_dim)/num_heads]

        k_t_c = k_t_c.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)  # [bs, num_heads, seq_len, up_dim/num_heads]
        k_t_r = k_t_r.expand(bs, self.num_heads, seq_len, -1)  # [bs, num_heads, seq_len, rope_head_dim]
        k = torch.cat([k_t_c, k_t_r], dim=-1)  # [bs, num_heads, seq_len, (up_dim+rope_head_dim)/num_heads]

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
        scores = scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim))

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))  # [bs, num_heads, seq_len, seq_len]

        attn_weights = torch.softmax(scores, dim=-1)  # [bs, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # V维度调整
        v_t_c = v_t_c.view(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)  # [bs, num_heads, seq_len, v_head_dim]

        # 计算上下文向量
        context = torch.matmul(attn_weights, v_t_c)  # [bs, num_heads, seq_len, v_head_dim]

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, -1)  # [bs, seq_len, num_heads*v_head_dim]

        # 输出投影
        output = self.fc(context)  # [bs, seq_len, d_model]
        output = self.res_dropout(output)

        return output

    
if __name__ == "__main__":
    model = nn.Sequential(
        MultiHeadLatentAttention(),
    ).to("cuda:0")
    loss = nn.MSELoss().to("cuda:0")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with torch.profiler.profile(
            activities=[
                #torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
            schedule=torch.profiler.schedule(
                wait=1,  # 前1步不采样
                warmup=1,  # 第2步作为热身，不计入结果
                active=3,  # 采集后面3步的性能数据
                repeat=1),  # 重复X轮
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_log',use_gzip=True),  # 保存日志以供 TensorBoard 可视化
            record_shapes=True,  # 记录输入张量的形状
            profile_memory=True,  # 分析内存分配
            with_stack=True,  # 记录操作的调用堆栈信息
            with_flops=True,
            with_modules=True,
        ) as profiler:

        for i in range(100):
            inp = torch.randn(1, 10, 256).to("cuda:0")
            opt.zero_grad()

            out = model(inp)
            loss_val = loss(out, inp)
            loss_val.backward()
            opt.step()

            profiler.step()

            print(loss_val)
            print(inp)
            print(out)

