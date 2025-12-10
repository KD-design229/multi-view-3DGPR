# 探地雷达 (GPR) 数据处理与模型适配总结

## 1. 探地雷达数据处理 (Data Processing)

该库通过 `code/data.py` 和 `code/data_augmention_loader_ori.py` 等文件实现了针对 GPR 数据的加载、预处理和增强。

### 多视图数据配对与加载
*   **配对逻辑**: 针对 GPR 数据通常包含不同视图（如 B-scan 侧视图和 C-scan 顶视图）的特性，代码实现了严格的文件名匹配逻辑（如 `find_indices_both_views` 函数），将同一检测点的“顶视图”（Top View, `_h.png`）与“主视图”（Main View, `_v.png`）进行配对加载。

```python
# code/data.py

def find_indices_both_views(data_dir, indices, dataset_topview, dataset_mainview):
    indices_img_v, indices_img_h = [], []
    top_path, main_path = [], []
    for i in indices:
        img_name = str(i + 1)
        # 构建主视图和顶视图的文件路径
        img_tuple_0_v = (os.path.join(data_dir, "1_mainview", "0_双曲线", img_name + "_v.png"), 0)
        img_tuple_0_h = (os.path.join(data_dir, "0_topview", "0_平行线", img_name + "_h.png"), 0)
        # ... (省略部分代码) ...
        
        # 在数据集中查找对应的索引
        if img_tuple_0_v in dataset_mainview.imgs:
            img_indice_v = dataset_mainview.imgs.index(img_tuple_0_v)
            main_path.append(img_tuple_0_v)
        # ...
        
        indices_img_v.append(img_indice_v)
        indices_img_h.append(img_indice_h)

    return [indices_img_h, indices_img_v], [top_path, main_path]
```

### GPR 特有的数据增强
*   **噪声模拟**: 针对雷达信号易受干扰的特点，实现了 `AddPepperNoise` (椒盐噪声) 和 `Gaussian_noise` (高斯噪声) 类，用于模拟地下介质不均匀性或设备噪声。
*   **缺失模拟**: 实现了 `Cutout` 和 `HidePatch` 类，随机遮挡图像块，模拟信号丢失或局部数据不完整的情况。

```python
# code/data.py

class AddPepperNoise(object):
    """增加椒盐噪声"""
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            # 随机生成噪声掩码
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
```

### 类别不平衡处理
*   实现了 `up_sample` 函数，通过随机复制少数类样本来解决地下病害样本分布不均的问题。

```python
# code/data.py

def up_sample(indices, classes):
    max_num = max(classes)
    up_indices = indices
    up_classes = []
    for i in range(len(classes)):
        up_classes.append(classes[i] + up_classes[i - 1]) if up_classes else up_classes.append(classes[i])
        if classes[i] < max_num:
            # 随机选择样本进行复制，直到数量达到 max_num
            index = np.random.randint(classes[i], size=(max_num - classes[i]))
            for item in index:
                up_indices.append(indices[item + up_classes[i - 1]]) if i > 0 else up_indices.append(indices[item])
    return up_indices
```

## 2. 针对 GPR 特性的代码模型适配 (Model Adaptations)

该库在 `models/densenet_MVFD.py` 中实现了专门的 **FusionDenseNet** 模型，并在 `code/train_DenseNet121_MVFD.py` 中实现了相应的训练策略，以适应 GPR 的多视图特性。

### 双分支网络架构与图注意力融合
*   **双分支**: 模型包含 `dn1` 和 `dn2` 两个 DenseNet 分支。
*   **注意力融合**: 使用 `GAT` 模块计算权重，融合两个视图的特征。

```python
# models/densenet_MVFD.py

class FusionDenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_init_features, num_classes, **kwargs):
        super(FusionDenseNet, self).__init__()
        # 双分支 DenseNet
        self.dn1 = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
        self.dn2 = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
        
        # 融合层与分类器
        self.classifier = nn.Linear(1024, 100)
        self.classifier2 = nn.Linear(100, num_classes)
        self.gat = GAT(1024, 3) # 图注意力网络

    def forward(self, x, y):
        out_x, x = self.dn1(x) # 顶视图特征提取
        w_x = self.gat(x)      # 计算注意力权重
        out_y, y = self.dn2(y) # 主视图特征提取
        w_y = self.gat(y)
        
        # 融合注意力权重
        attention_merged = mean_att(w_x, w_y)
        beta_x, beta_y = attention_merged.split(1, dim=1)
        
        # 加权融合特征
        layer_merged = beta_x * x + beta_y * y
        layer_merged = torch.flatten(layer_merged, 1)
        
        out = self.classifier(layer_merged)
        out = self.relu(out)
        out = self.classifier2(out)
        return out, out_x, out_y
```

### 切换式在线知识蒸馏 (SwitOKD)
*   **动态蒸馏**: 根据各分支和融合分支的预测置信度（通过 `epsilon` 和 `delta` 动态阈值判断），决定反向传播的 Loss 构成。
*   **互相指导**: 表现好的分支会指导表现差的分支（通过 KL 散度）。

```python
# code/train_DenseNet121_MVFD.py

# ... (计算 outputs 和 one-hot labels) ...

# 动态计算阈值
epsilon = torch.exp(-1 * top_label / (main_label + top_label))
delta = main_label - epsilon * top_label
epsilon1 = torch.exp(-1 * fusion_label / (main_label + fusion_label))
delta1 = main_label - epsilon1 * fusion_label
# ...

# 根据阈值判断当前样本应该优化哪些分支，以及谁指导谁
if (pmain_ptop > delta and top_label < main_label) and ...:
    # 例如：主视图表现好，用主视图指导其他分支
    loss_main = criterion(outs_main, problems_label_torch) + \
                (kl_div(outs_top.detach(), outs_main) * 1 + kl_div(outs_fusion.detach(), outs_main)) * 1
    loss = loss_main
    loss.backward()
    optimizer_main.step()
elif ...:
    # 其他情况的蒸馏逻辑
    # ...
```
