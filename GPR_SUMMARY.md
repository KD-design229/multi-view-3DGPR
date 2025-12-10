# 探地雷达 (GPR) 数据处理与模型适配总结

## 1. 探地雷达数据处理 (Data Processing)

该库通过 `code/data.py` 和 `code/data_augmention_loader_ori.py` 等文件实现了针对 GPR 数据的加载、预处理和增强。

### 多视图数据配对与标签融合
针对 GPR 数据通常包含不同视图（如 B-scan 侧视图和 C-scan 顶视图）的特性，代码实现了严格的文件名匹配逻辑。同时，为了处理不同视图对同一病害的不同表现，实现了标签融合逻辑 `get_problem`。

*   **配对逻辑**: `find_indices_both_views` 函数将同一检测点的“顶视图”（`_h.png`）与“主视图”（`_v.png`）进行配对。
*   **标签映射**: 通过 `get_problem` 函数将两个视图的原始分类标签映射为最终的 4 类病害标签（0: 空洞, 1: 层间脱空, 2: 裂缝, 3: 正常）。

```python
# code/data.py

def get_problem(top_view_type, main_view_type):
    # main_view_type: 0_双曲线, 1_高亮, 2_正常
    # top_view_type: 0_平行线, 1_暗斑, 2_亮斑, 3_正常
    
    problems = ['空洞', '层间脱空', '裂缝', '正常']
    # 优先级判定逻辑：
    if main_view_type == 1: # 主视图为“高亮” -> 判定为 0: 空洞
        return 0
    else:
        if top_view_type == 1 or top_view_type == 2: # 顶视图为“暗斑”或“亮斑” -> 判定为 1: 层间脱空
            return 1
        elif top_view_type == 0: # 顶视图为“平行线” -> 判定为 2: 裂缝
            return 2
        else:
            return 3 # 正常
```

### GPR 特有的数据增强
*   **物理噪声模拟**:
    *   `AddPepperNoise`: 模拟椒盐噪声，模拟地下介质不均匀性产生的随机强反射点。
    *   `Gaussian_noise`: 模拟高斯白噪声，模拟设备热噪声或背景干扰。
*   **信号缺失模拟**:
    *   `Cutout` / `HidePatch`: 随机遮挡图像块，模拟雷达信号在特定区域的丢失或严重衰减。

```python
# code/data.py
class AddPepperNoise(object):
    """增加椒盐噪声"""
    def __init__(self, snr, p=0.9):
        # ...
    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            # 随机生成噪声掩码，模拟信号坏点
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # ...
```

*   **离线数据扩充**:
    *   `augmention_dataset` 类支持在训练前对数据集进行倍增 (`aug_times`)。
    *   在 `mode=2` (训练模式) 下，会对图像进行随机色调调整 (`adjust_hue`) 和高斯模糊 (`GaussianBlur`)，以增加样本多样性。

```python
# code/data_augmention_loader_ori.py
class augmention_dataset(data.Dataset):
    def maketraindata(self, repeat=0):
        # 数据倍增逻辑，根据 repeat 次数生成不同的色调调整参数
        if abs(repeat) > 0:
            # ...
            for y in range(...):
                self.samples = self.samples + self.non_norm_sampling(...)
```

### 类别不平衡处理
*   `up_sample`: 通过随机复制少数类样本（如“空洞”样本通常较少）来平衡训练集分布，防止模型偏向于“正常”类别。

## 2. 针对 GPR 特性的代码模型适配 (Model Adaptations)

该库在 `models/densenet_MVFD.py` 中实现了专门的 **FusionDenseNet** 模型，并在 `code/train_DenseNet121_MVFD.py` 中实现了相应的训练策略。

### 双分支网络架构与图注意力融合 (GAT Fusion)
模型包含两个独立的 DenseNet 分支，分别处理顶视图和主视图。在特征提取末端，使用图注意力机制动态计算融合权重，从而自适应地利用不同视图的信息。

*   `mean_att`: 计算两个视图特征的注意力权重。
*   `FusionDenseNet`: 结合两个分支的特征和权重进行加权融合。

```python
# models/densenet_MVFD.py
def mean_att(x: Tensor, y: Tensor) -> Tensor:
    # x, y shape: [batch, features]
    # 对特征进行拼接并计算 softmax 权重
    betas_x, betas_y = [], []
    for idx in range(3): # 假设有3个注意力头
        weight_x_temp, weight_y_temp = x[:, idx].view(-1, 1), y[:, idx].view(-1, 1)
        attention_merged = torch.cat((weight_x_temp, weight_y_temp), 1)
        attention_merged = torch.softmax(attention_merged, dim=1)
        beta_x, beta_y = attention_merged.split(1, dim=1)
        betas_x.append(beta_x)
        betas_y.append(beta_y)
    
    # 取平均作为最终权重
    beta_x_mean = torch.mean(torch.cat(betas_x, dim=1), dim=1).view(-1, 1)
    beta_y_mean = torch.mean(torch.cat(betas_y, dim=1), dim=1).view(-1, 1)
    return torch.cat((beta_x_mean, beta_y_mean), dim=1)
```

### 切换式在线知识蒸馏 (SwitOKD)
为了解决多视图融合中单一视图分支可能训练不足的问题，采用了“切换式在线知识蒸馏”策略。模型会动态评估主视图、顶视图和融合视图的置信度，并让表现好的视图“指导”表现差的视图。

*   **动态阈值**: 计算 `epsilon` 和 `delta`，衡量视图预测与 Ground Truth 之间的距离差异。
*   **相互蒸馏**: 根据阈值判定，选择性地将 KL 散度损失 (`kl_div`) 加入到总损失中。

```python
# code/train_DenseNet121_MVFD.py

# 距离度量函数
def dist_s_label(y, q): # 学生与标签的距离 (L1)
    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)
    return torch.mean(dist)

def dist_s_t(p_logit, q_logit, T): # 学生与教师的距离 (L1 of Softmax)
    # ...
    dist = torch.sum(torch.abs(q - p), 1)
    return torch.mean(dist)

# 训练循环中的逻辑
# ...
epsilon = torch.exp(-1 * top_label / (main_label + top_label))
delta = main_label - epsilon * top_label
# ...

# 经典四步判断 (Classic Four Steps)
if (pmain_ptop > delta and top_label < main_label) and ...:
    # 场景1: 主视图优于顶视图，且融合视图也表现良好 -> 主视图作为教师指导其他
    loss_main = criterion(outs_main, problems_label_torch) + \
                (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
    loss = loss_main
    loss.backward()
    optimizer_main.step()
elif ...:
    # 场景2/3/4: 根据相对性能，动态调整谁作为教师(Teacher)，谁作为学生(Student)
    # 从而实现 Top View, Main View, Fusion View 三者之间的相互促进
```
