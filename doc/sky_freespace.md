# Sky Freespace — 射线级 3D 体素空间 Free-Space 监督

## 背景

自动驾驶 3D 占用预测中，天空区域在物理世界中意味着 "无穷远 / 无任何遮挡"。
我们利用 SAM3 大模型离线生成的高质量天空二值掩码，为模型提供额外的
"已知空闲 (Known Free)" 3D 监督信号，帮助模型减少空中的虚假占用预测。

## 早期尝试: Sky Gate (硬门控) — 已废弃

> **结论: 在 2D 特征阶段做硬门控不可行，已从代码中移除。**
>
> 早期曾尝试在 `LSSViewTransformer` 中引入 `gate_net` 分支预测前景/天空得分，
> 以 `tran_feat *= sigmoid(gate)` 的方式在 2D 特征阶段硬性阻断天空区域的特征。
> 实验表明该方案存在三个致命问题:
>
> 1. **射线伪影**: Sigmoid 截断在图像空间引入锐利的 0/1 边界，经 LSS 沿视线
>    Splat 到 BEV 后产生密集的辐射状射束噪声，严重污染 BEV 特征
> 2. **梯度阻断**: 被门控压为零的区域，OCC Loss 梯度无法回传到主干网络，
>    导致天空边界附近的特征缺乏优化压力，深度估计退化
> 3. **SAM3 误杀**: SAM3 的任何误判（高光车体、桥梁底部等）都会直接消灭
>    该区域的目标特征，属于不可逆的信息损失

## 算法原理

```
对于 2D 图像中被 SAM3 标注为 "天空" 的像素 (fH, fW):
  ├── 沿 LSS 视锥的所有 D 个深度采样点 (frustum)
  │     ├── 通过 get_ego_coor() 映射到 ego 坐标系 (x, y, z)
  │     └── 量化为 3D 体素索引 (ix, iy, iz) ∈ [0, Dx) × [0, Dy) × [0, Dz)
  └── 将这些体素标记为 "应当为空 (Free, class 17)"
      └── 对 OCC head 输出施加辅助 Cross-Entropy Loss
```

### 关键数学

给定天空像素集 $\mathcal{S}$，其视锥采样点集 $\mathcal{V}_\text{sky} = \{(x, y, z)\}$，
对应的体素索引集 $\mathcal{I}_\text{sky}$：

$$
\mathcal{L}_\text{freespace} = \frac{1}{|\mathcal{I}_\text{sky}|}
\sum_{i \in \mathcal{I}_\text{sky}}
\text{CE}(\hat{y}_i, \text{Free})
$$

其中 $\hat{y}_i$ 是 OCC head 在体素 $i$ 的 logits，$\text{Free}$ 对应 class 17。

## 文件变更

### 模型

- **`src/flashocc/models/detectors/bevdet_occ.py`**
  - `sky_freespace_loss_weight` 参数控制射线级 Loss 权重
  - `_compute_sky_freespace_loss()` 方法：
    1. 在 `torch.no_grad()` 下计算视锥 ego 坐标
    2. 将天空像素的 D 个深度点映射到 3D 体素索引
    3. 去重后取 OCC 预测，对 Free 类计算 Cross-Entropy
  - `forward_occ_train()` 返回 `(loss_dict, occ_pred)` 二元组

- **`src/flashocc/models/detectors/bevdet.py`**
  - `extract_img_feat()` 缓存 `_cached_sensor_inputs` 供子类复用

### 配置

- **`configs/flashocc_convnext_tiny_dinov3_sky_freespace.py`**
  - `sky_freespace_loss_weight=1.0` — 启用射线级 Loss
  - `LoadSkyMask` pipeline 加载天空掩码
  - `Collect3D` 包含 `sky_mask` key

## 使用方式

### 训练

```bash
# 单 GPU
python tools/train.py configs/flashocc_convnext_tiny_dinov3_sky_freespace.py

# 多 GPU
bash tools/dist_train.sh configs/flashocc_convnext_tiny_dinov3_sky_freespace.py 4
```

### 调参建议

- `SKY_FREESPACE_LOSS_WEIGHT`:
  - 推荐范围: `0.5 ~ 2.0`
  - 过高可能导致模型过度偏向预测 Free，小物体召回下降
  - 过低则效果不明显
  - 默认值 `1.0` 是保守起点

### 前置依赖

确保已运行 SAM3 天空掩码生成:

```bash
bash tools/dist_generate_sky_masks.sh 4
```

掩码存储在 `data/SAM3/samples/CAM_XXX/*.png`，由 `LoadSkyMask` pipeline 读取。

## 设计决策 FAQ

**Q: 为什么要去重 (unique) 体素索引？**

A: 多条射线（来自不同相机/不同深度）可能命中同一体素。去重确保每个体素
在 Loss 中只贡献一次，避免高重叠区域（如近处地面）被过度惩罚。

**Q: 推理时有额外开销吗？**

A: 完全没有。Sky Freespace 仅在 `forward_train` 中计算辅助 Loss，
`simple_test` 路径不涉及任何天空相关逻辑。
