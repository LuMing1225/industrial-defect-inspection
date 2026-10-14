# 工业盖板缺陷抗干扰与 SAM 掩膜融合技术规格书

| 项目 | 内容 |
|---|---|
| 文档名称 | 工业盖板缺陷抗干扰与 SAM 掩膜融合技术规格书 |
| 文档版本 | V1.0（定稿） |
| 所属项目 | 多 Agent 协同与 CV 智能感知 —— 工业缺陷干扰项过滤系统 |
| 项目代号 | 工业盖板缺陷抗干扰与 SAM 掩膜融合 |
| 编制日期 | 2026-09-30 |
| 编制方 | 检测算法组（detector / segmenter / vlm） |
| 评审机制 | 算法组评审定稿 |
| 文档性质 | 工程设计文档（开源） |

---

## 1. 文档说明

### 1.1 适用范围

本规格书适用于手机盖板玻璃产线在线光学缺陷检测系统中的"干扰项过滤与掩膜修正"子系统。它规定了：

1. 针对空气粉尘、水渍、反光防伪条等干扰项的**多特征初筛**算法规格；
2. 针对 SAM（Segment Anything Model）因条纹背景产生的**碎片化掩膜**进行凸包融合与形态学平滑的算法规格；
3. 针对落入"模糊区间"样本的 **Qwen2-VL 本地大模型兜底终审**规格；
4. 各模块之间的**接口规格、性能预算与验收指标**。

本规格书不约束底层缺陷检测器（如 YOLO/无监督异常分割主干）的内部实现，仅约束其**输出候选区域（Region Proposal）之后**的过滤与修正链路。

### 1.2 引用标准与参考

| 编号 | 名称 | 用途 |
|---|---|---|
| REF-1 | Ojala T., Pietikäinen M., Harwood D. *A Comparative Study of Texture Measures with Classification Based on Feature Distributions*, Pattern Recognition, 1996 | LBP 算子原始定义 |
| REF-2 | OpenCV 5.0 官方文档（`connectedComponents`、`findContours`、`convexHull`、`morphologyEx`） | 底层算子 API |
| REF-3 | Kirillov et al. *Segment Anything*, ICCV 2023 | SAM 分割模型行为参考 |
| REF-4 | Qwen2.5-VL 技术报告 & Ollama 本地部署文档 | VLM 推理接口 |
| REF-5 | KolektorSDD / MVTec AD 数据集标注规范 | 真实缺陷标签口径 |
| REF-6 | GB/T 2828.1 计数抽样检验程序 | 产线误报率统计口径参考 |

### 1.3 术语与缩略语

| 缩略语 | 全称 | 含义 |
|---|---|---|
| RoI | Region of Interest | 检测器输出的候选缺陷外接区域 |
| LBP | Local Binary Pattern | 局部二值模式纹理算子 |
| SAM | Segment Anything Model | 分割一切基础模型 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型 |
| TNR | True Negative Rate | 真负率，此处即干扰项（粉尘/水渍）被正确排除的比例 |
| FNR | False Negative Rate | 漏检率，真实缺陷被误过滤的比例 |
| FPS | Frames Per Second | 系统每秒可处理帧数 |
| Hausdorff | Hausdorff Distance | 点集之间的最大最小距离度量 |

---

## 2. 系统概述

### 2.1 业务背景与痛点

在手机盖板玻璃生产线上，采用**强背光透射成像**以凸显玻璃内部与表面的微小凹坑、崩边、划痕。该成像方式存在两个结构性干扰源：

1. **空气粉尘误报**：镜头上方空气中悬浮的微尘落入相机视场，在强背光下呈现为**圆形、边缘柔和、灰度均匀**的暗斑，其灰度与形状在像素层面与"微小凹坑/崩边"高度相似。现场实测该类干扰导致**误报率高达 18%**，频繁触发停机人工复检，严重影响节拍。
2. **条纹背景导致 SAM 碎片化**：盖板上的金属条纹栅栏与反光防伪条，其周期性纹理与玻璃缺陷纹理高度一致。直接将 RoI 送入 SAM 进行提示分割时，防伪标被打碎成**上百个局部掩膜碎片**，无法提取完整轮廓，既造成"一个防伪标被报成上百个缺陷"的过分割误报，也无法对真实缺陷做实例级定位。

业务验收的硬性约束为：**粉尘/水渍干扰排除率 ≥ 96%，真实缺陷漏检率 = 0，SAM 碎片 Mask 拼接连通率 ≥ 92%**。

### 2.2 系统架构（三级流水线）

系统采用"**几何纹理初筛 → SAM 碎片凸包融合 → VLM 模糊区间兜底**"的三级漏斗式流水线。整体数据流如下（文字架构图）：

```
┌──────────────────────────────────────────────────────────────────────┐
│                        产线相机帧 10Hz @1024×768                        │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage-0  缺陷检测器（Detector 主干，本规格书范围外）                     │
│           输出候选 RoI 列表：[(bbox, score, mask_seed), ...]           │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage-1  多特征初筛  FeatureFilter（detector/feature_filter.py）       │
│  ├─ 连通域查找 contour / connectedComponents                           │
│  ├─ 几何特征：Roundness、AspectRatio                                    │
│  ├─ 纹理特征：VarGray、Contrast_LBP                                     │
│  ├─ 颜色特征：HSV 直方图（轻量）                                        │
│  └─ 决策：硬阈值 + 加权评分 Score_filter                                │
│     ├─ 高置信干扰（Score ≤ τ_low）   → 直接丢弃（误报压制）              │
│     ├─ 高置信缺陷（Score ≥ τ_high）  → 直接上报                         │
│     └─ 模糊区间（τ_low < Score < τ_high）→ 进入 Stage-2 / Stage-3       │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage-2  SAM 碎片融合  SAMCombiner（segmenter/sam_combiner.py）       │
│  ├─ SAM 提示分割 → 碎片 mask 列表                                       │
│  ├─ 碎片关联：中心距 + Hausdorff + 边缘灰度梯度共生矩阵                  │
│  ├─ 近邻碎片聚类（图连通）                                              │
│  ├─ 凸包 Convex Hull 缝合（Andrew 单调链）                              │
│  └─ 形态学闭运算（先膨胀后腐蚀）平滑边缘、填充间隙                        │
│     输出：合并后完整实例 mask，校验连通度 ≥ 92%                         │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage-3  VLM 兜底终审  VLMChatter（vlm/validator.py）                  │
│  ├─ 仅对"模糊区间"样本（约 20%）裁切缺陷图送入                           │
│  ├─ Ollama 本地 qwen2.5vl:7b 推理                                      │
│  ├─ System Prompt 判定：粉尘/水渍/划痕 vs 真实结构缺陷                   │
│  └─ 超时降级：>450ms 未返回则按"可疑保留"处理（宁可人工复判，不漏检）    │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
        ┌───────────────┐
        │ 质检结果上报     │  {bbox, mask, class, confidence, source_stage}
        └───────────────┘
```

### 2.3 设计目标

| 维度 | 设计目标 |
|---|---|
| 准确性 | 粉尘/水渍排除率 ≥ 96%；真实缺陷漏检率 = 0（安全侧偏置） |
| 鲁棒性 | 对未见过的新型干扰（新型水渍、油污）通过 VLM 兜底，不依赖固定白名单 |
| 实时性 | 初筛 + SAM 融合 ≤ 80ms/帧；VLM 终审 ≤ 450ms/帧（仅模糊区间触发） |
| 资源 | RTX 5070 8GB 显存下不发生 OOM；VLM 约占 6GB，其余模型与缓冲可控 |
| 可维护性 | 权重 w1/w2/w3、阈值 τ_low/τ_high、膨胀核尺寸均可配置，支持在线反馈更新 |
| 数据安全 | VLM 完全本地部署（Ollama），缺陷图像不出产线内网 |

---

## 3. 多特征初筛规格

### 3.1 候选区域连通域提取

对每帧灰度图（可叠加边缘梯度图）先做二值化（自适应阈值 `cv2.adaptiveThreshold`），再调用 `cv2.connectedComponentsWithStats` 得到每个连通域：

- 输出：`num_labels, labels, stats, centroids`；
- 对面积过小（`area < S_min`，默认 9 像素）与过大（`area > S_max`，默认整图 1/4）的连通域直接剔除，避免噪声与整板背景；
- 每个连通域提取其像素集合 $C = \{ (x_i, y_i) \}_{i=1}^{N}$ 作为后续特征计算的输入。

### 3.2 几何与纹理特征定义

#### 3.2.1 圆度 Roundness

$$
\text{Roundness} = \frac{4\pi \cdot A}{P^{2}}
$$

其中 $A$ 为连通域面积（像素数），$P$ 为轮廓周长。**物理意义**：衡量形状接近理想圆的程度。完美圆取到理论最大值 $4\pi \cdot \pi r^2 / (2\pi r)^2 = 1.0$；越不规则、越细长，Roundness 越趋近 0。粉尘在强背光下呈近圆形暗斑，Roundness 通常 > 0.7；真实凹坑边缘不规则、常带拖尾，Roundness 多 < 0.5。

> 注：轮廓周长 $P$ 采用 `cv2.findContours` 近似多边形长度；面积 $A$ 用同一 contour 的 `cv2.contourArea`，保证二者口径一致。

#### 3.2.2 长宽比 Aspect Ratio

$$
\text{AspectRatio} = \frac{\max(W_{mbr}, H_{mbr})}{\min(W_{mbr}, H_{mbr})}
$$

其中 $W_{mbr}, H_{mbr}$ 为连通域**最小外接旋转矩形**（Min Area Rectangle）的宽与高。**物理意义**：刻画目标的细长程度。近圆粉尘 AspectRatio ≈ 1.0；真实划痕常呈长条状，AspectRatio 可 > 3.0。该特征作为圆度的补充，避免把"近圆但实为多角崩边"误判。

#### 3.2.3 灰度方差 VarGray

$$
\text{VarGray} = \frac{1}{N}\sum_{i=1}^{N}\left(I_i - \mu\right)^{2}, \qquad \mu = \frac{1}{N}\sum_{i=1}^{N} I_i
$$

$I_i$ 为连通域内像素灰度，归一化到 $[0,1]$。**物理意义**：区域内部灰度的离散程度。粉尘由均匀光晕形成，内部灰度高度一致，VarGray 低；真实凹坑含深浅渐变与边缘过渡，VarGray 高。

#### 3.2.4 LBP 纹理对比度 Contrast_LBP

$$
\text{LBP}_{P,R}(x_c) = \sum_{p=0}^{P-1} s(g_p - g_c)\cdot 2^{p}, \quad s(x)=\begin{cases}1, & x\ge 0\\ 0, & x<0\end{cases}
$$

在半径 $R=2$、邻域采样点 $P=16$ 的圆形邻域上计算每个像素的 LBP 编码，统计整域 LBP 直方图 $\{h_k\}_{k=0}^{255}$，再取其方差：

$$
\text{Contrast\_LBP} = \frac{1}{256}\sum_{k=0}^{255}\left(h_k - \bar{h}\right)^{2}
$$

**物理意义**：LBP 直方图越集中（少数 bin 占主导）说明纹理越均匀单调，Contrast_LBP 越小；直方图越分散说明边缘/纹理丰富，Contrast_LBP 越大。粉尘区域纹理均匀 → 直方图集中 → 对比度低；凹坑边缘锐利、纹理复杂 → 直方图分散 → 对比度高。

### 3.3 评分公式与权重设计

初筛决策采用加权线性评分：

$$
\text{Score}_{filter} = w_1\cdot \text{Roundness} + w_2\cdot \widetilde{\text{VarGray}} + w_3\cdot \widetilde{\text{Contrast\_LBP}}
$$

> 为量纲统一，VarGray 与 Contrast_LBP 需先做 Min-Max 归一化（$\widetilde{x} = (x - x_{min})/(x_{max}-x_{min})$），统计上下界来自训练集。

**权重设计依据（w1=0.4, w2=0.3, w3=0.3）**：

| 权重 | 特征 | 设计依据 |
|---|---|---|
| $w_1=0.4$ | Roundness | 粉尘在几何上最稳定、最具区分度的先验是"圆形"，现场粉尘 90% 以上为近圆暗斑，故赋予最高权重 |
| $w_2=0.3$ | VarGray | 灰度均匀性对光晕类粉尘稳健，但受背光强度漂移影响，故次高 |
| $w_3=0.3$ | Contrast_LBP | 纹理对比度对边缘缺陷敏感，但 LBP 对噪声敏感、计算略重，与 VarGray 并列 |

权重在初始化后，使用标注样本（100 真实缺陷 + 50 粉尘）通过 **网格搜索 + ROC 约登指数** 在验证集上标定，使 TNR（粉尘排除率）最大化的同时约束 FNR = 0。

### 3.4 硬阈值与模糊区间

决策采用**三段式**而非单一阈值，以在"漏检 = 0"的安全约束下最大化误报压制：

| 区间 | 判定 | 处理 | 概率解释 |
|---|---|---|---|
| $\text{Score}_{filter} \le \tau_{low}$（默认 0.35） | 高置信干扰（典型粉尘） | 直接丢弃，不上报 | 后验 $P(\text{粉尘}\mid x) > 0.95$ |
| $\text{Score}_{filter} \ge \tau_{high}$（默认 0.65） | 高置信缺陷 | 直接上报，跳过 VLM | 后验 $P(\text{缺陷}\mid x) > 0.9$ |
| $\tau_{low} < \text{Score}_{filter} < \tau_{high}$ | **模糊区间** | 进入 Stage-2 SAM 融合；并送 Stage-3 VLM 终审 | 后验 $0.5 \sim 0.9$，决策不确定区 |

**硬阈值补充规则**：当 `AspectRatio > 3.0`（明显长条划痕）或 `Roundness > 0.85 且 VarGray < 阈值`（极规则圆形且内部极均匀，典型粉尘）时，可直接走硬规则拦截，不进入评分，以覆盖 80% 的规则圆形粉尘（对应 MVP 目标）。

### 3.5 LBP 纹理算子原理

#### 3.5.1 计算方法

对中心像素 $g_c$，在半径 $R=2$ 的圆上均匀取 $P=16$ 个邻域点 $g_p$（非整数坐标用双线性插值）。将每个邻域灰度与中心比较，大于等于中心记 1，否则记 0，按位权 $2^p$ 加权求和得到该点 LBP 值（0–255）。

```
for each pixel (x,y) in RoI:
    gc = gray(x, y)
    code = 0
    for p in 0..P-1:                       # P=16
        theta = 2*pi*p / P
        nx = x + R*cos(theta); ny = y + R*sin(theta)   # R=2, 双线性插值
        gp = bilinear_sample(nx, ny)
        code |= (1 << p) if gp >= gc else 0
    lbp_map[y][x] = code
hist = histogram(lbp_map, bins=256)        # 归一化
Contrast_LBP = variance(hist)
```

选用 $(P=16, R=2)$ 的理由：$R=2$ 可覆盖粉尘光晕的典型柔和过渡带（约 4–6 像素直径），$P=16$ 提供足够角度分辨率以区分锐利凹坑边缘与柔和粉尘边缘；半径过小会把噪声当成纹理，过大会模糊真实缺陷的细微边缘。

#### 3.5.2 粉尘 vs 真实缺陷在 LBP 空间的分布差异

- **粉尘**：强背光下的圆形光晕，由中心向外灰度平滑渐变，邻域点与中心的大小关系在各方向上**接近、单调、变化缓慢**。绝大多数像素的 LBP 编码落在少数几个"近似均匀"的 bin（如全 0/全 1 或微渐变码），LBP 直方图**高度集中** → 方差小 → `Contrast_LBP` 低。
- **真实凹坑/崩边**：边缘锐利、存在相位反转、坑内深浅不一，邻域符号关系在边缘处剧烈翻转，LBP 编码**散布在大量 bin** → 直方图**分散** → 方差大 → `Contrast_LBP` 高。

二者在 `(Roundness, Contrast_LBP)` 二维平面上近似呈"粉尘聚集于左下角、缺陷散布于右上"的可分簇，这是初筛评分有效的物理基础。

---

## 4. SAM 碎片融合规格

### 4.1 SAM 碎片化问题分析

条纹防伪标 / 条纹栅栏与缺陷纹理高度一致时，SAM 基于提示点/框进行提示分割，会把同一防伪标沿周期性条纹边界切成**数十至上百个碎片 mask**。根因有二：

1. **提示歧义**：SAM 依据提示点周边的"图像边界能量"生长 mask，条纹本身就是强边界，导致 mask 在条纹处提前终止；
2. **纹理连续性断裂**：相邻条纹间的灰度突变破坏了 mask 生长所需的前景一致性。

直接使用这些碎片会造成"一个防伪标 → 上百条误报"，必须在实例层面重新缝合。

### 4.2 碎片关联度量

对碎片集合 $\{M_1, M_2, \dots, M_n\}$，两两计算以下关联度，构造关联图 $G=(V,E)$：

| 度量 | 定义 | 作用 |
|---|---|---|
| 中心距离 $d_{cent}$ | 两碎片质心欧氏距离 | 粗筛近邻，先只对 $d_{cent}<\rho_{cent}$ 的对建边 |
| Hausdorff 距离 $d_{H}$ | 无向 Hausdorff 距离 | 衡量两片轮廓的形状贴近程度，轮廓越接近 $d_H$ 越小 |
| 边缘灰度梯度共生矩阵 | 提取两片边缘梯度方向/幅值统计，算直方图交并 | 判断两片是否属于同一连续表面（梯度方向是否衔接） |

**关联建边规则**：当 $d_{cent}<\rho_{cent}$ 且 $d_{H}<\rho_{H}$ 且边缘梯度相似度 $>\rho_{grad}$ 时，在 $M_i, M_j$ 间连边；随后对图 $G$ 做连通分量聚类，得到若干"碎片组"，每组视为一个待缝合实例。

### 4.3 凸包（Convex Hull）缝合算法

对同一碎片组内所有前景像素点集合并，求其凸包，用凸包多边形作为缝合后的实例轮廓，填补碎片之间的间隙。伪代码：

```python
def combine_masks(fragment_masks, grad_map,
                  rho_cent=40.0, rho_h=15.0, rho_grad=0.6):
    # 1. 收集每个碎片的点集、质心、边缘梯度直方图
    nodes = [describe(m) for m in fragment_masks]
    n = len(nodes)

    # 2. 建关联图（并查集）
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            dcent = centroid_dist(nodes[i], nodes[j])
            dh    = hausdorff(nodes[i].edge_pts, nodes[j].edge_pts)
            gsim  = grad_hist_intersect(nodes[i].ghist, nodes[j].ghist)
            if dcent < rho_cent and dh < rho_h and gsim > rho_grad:
                uf.union(i, j)

    groups = uf.components()          # 若干碎片组
    merged = []
    for g in groups:
        # 3. 合并组内所有前景点
        pts = np.vstack([nodes[k].foreground_pts for k in g])
        # 4. Andrew 单调链求凸包顶点
        hull = convex_hull_andrew(pts)
        # 5. 凸包多边形填充为 mask
        mask = fill_polygon(hull, shape=IMAGE_SHAPE)
        # 6. 形态学闭运算平滑（先膨胀后腐蚀）
        mask = morphology_close(mask, kernel=cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (7,7)))
        merged.append(mask)
    return merged
```

> 凸包缝合的物理意义：防伪标虽然被条纹切成碎片，但其外轮廓整体是一个凸（或近凸）矩形区域；用凸包把碎片包围起来，即可把"沿条纹裂开的内部间隙"一并补回，恢复完整实例轮廓。

### 4.4 形态学膨胀/腐蚀平滑

凸包多边形边缘可能存在锯齿，且组内碎片间仍可能残留小孔洞。采用**闭运算（先膨胀后腐蚀）**：

- 膨胀 `dilate`：把碎片边缘向外扩，弥合碎片间的细小缝隙；
- 腐蚀 `erode`：再向内收回，恢复原始尺寸；
- 椭圆核 $7\times7$：比矩形核更贴合弧形缺陷边缘，避免直角突出。

闭运算在不显著改变 mask 总面积的前提下，消除边缘毛刺与内部小孔，使连通域更紧致。

### 4.5 Mask 连通度指标与验证

**连通度定义**：

$$
\text{Connectivity} = \frac{\text{Area}(\text{最大连通域})}{\text{Area}(\text{全部前景像素})}
$$

- 缝合前：碎片彼此分离，最大连通域面积占比低（可能 < 30%）；
- 缝合后：理想情况下所有碎片被包入同一凸包，连通度应 ≥ 92%。

**验证方法**：对缝合后 mask 执行 `cv2.connectedComponentsWithStats`，取最大连通域面积除以总前景面积。在 30 张条纹碎裂测试图上逐张计算，要求**平均连通度 ≥ 92%，且无单图低于 85%**。若不达标，按 Loop Engineering 自动调大闭运算核尺寸或降低 $\rho_H$ 阈值后重跑。

---

## 5. VLM 兜底校验规格

### 5.1 本地部署方案

| 项 | 配置 |
|---|---|
| 推理引擎 | Ollama（本地常驻服务，REST API） |
| 模型 | `qwen2.5vl:7b`（视觉-语言多模态） |
| 显存占用 | 约 6 GB（FP16 量化加载） |
| 硬件 | RTX 5070 8GB，CUDA 12.9 |
| 调用方式 | HTTP `POST /api/generate`（带 `images` base64 字段） |
| 触发比例 | 仅模糊区间样本，约占全量 20% |

部署优势：缺陷图像**不出产线内网**，满足工业数据安全要求；本地推理避免公网往返，P95 可控在 450ms 内。

### 5.2 System Prompt 设计规范

VLM 仅做"是干扰还是真实结构缺陷"的二分类辅助，Prompt 必须封闭、可判定、避免开放生成：

```
你是手机盖板玻璃产线的质检复核员。下面是一张背光透射成像的局部裁切图。
请严格依据以下判定准则，仅输出 JSON：
- "dust_or_water"：空气中粉尘、水渍、油污、指纹等非结构性、可随时间消失/迁移的干扰；
- "real_defect"：玻璃真实结构缺陷，如凹坑、崩边、穿透性划痕、裂纹等会残留的结构损伤。
判定依据：
1. 形态：边缘是否锐利、是否有深浅不一的结构阴影；
2. 纹理：是否为均匀柔和光晕（干扰），还是不规则边缘与内部结构（缺陷）；
3. 圆形度：过于规则的正圆柔光斑倾向于粉尘。
只输出：{"label": "dust_or_water" | "real_defect", "confidence": 0.0~1.0, "reason": "一句话"}
```

### 5.3 触发条件、API 流程与降级

- **触发条件**：仅当 Stage-1 落在模糊区间 $\tau_{low}<Score<\tau_{high}$，或 Stage-2 凸包连通度处于临界（0.85–0.92）时触发 VLM。高置信区不调用，节省算力。
- **API 调用流程**：
  1. 按 bbox 从原图裁切缺陷 RoI，resize 到 VLM 输入尺寸，base64 编码；
  2. `POST /api/generate`，携带 system prompt + 用户提问 + 图像；
  3. 解析返回 JSON，取 `label` 与 `confidence`；
  4. `label=dust_or_water` 且 `confidence>0.7` → 判为干扰丢弃；否则保留上报。
- **超时降级策略**：设置 450ms 硬超时。超时或解析失败时，**按"可疑保留"处理（上报人工复判）**，绝不因 VLM 不可用而漏检真实缺陷——这是安全侧偏置的硬性要求。

---

## 6. 接口规格

### 6.1 多特征初筛 `FeatureFilter`（detector/feature_filter.py）

```python
class FeatureFilter:
    def __init__(self,
                 w1: float = 0.4, w2: float = 0.3, w3: float = 0.3,
                 tau_low: float = 0.35, tau_high: float = 0.65,
                 r_lbp: int = 2, p_lbp: int = 16,
                 s_min: int = 9, s_max_ratio: float = 0.25):
        ...

    def compute_roundness(self, contour) -> float:
        """返回 4*pi*A / P^2，范围约 (0, 1]"""

    def compute_aspect_ratio(self, contour) -> float:
        """最小外接旋转矩形 长边/短边"""

    def compute_var_gray(self, roi_gray: np.ndarray,
                         mask: np.ndarray) -> float:
        """连通域内灰度方差（已归一化到[0,1]）"""

    def compute_contrast_lbp(self, roi_gray: np.ndarray,
                            mask: np.ndarray) -> float:
        """LBP(P=16,R=2) 直方图方差"""

    def score(self, features: dict) -> float:
        """Score_filter = w1*Roundness + w2*VarGray + w3*Contrast_LBP"""

    def filter(self, frame_gray: np.ndarray,
               proposals: list[dict]) -> list[dict]:
        """
        入参: 整帧灰度图 + 检测器候选列表 [{bbox, score, mask_seed}]
        返回: 每个 proposal 追加 {
            'features': {...},
            'score_filter': float,
            'decision': 'reject' | 'accept' | 'ambiguous'
        }
        decision: reject=高置信干扰直接丢弃; accept=高置信缺陷直报; ambiguous=进入下一阶段
        """
```

### 6.2 SAM 碎片融合 `SAMCombiner`（segmenter/sam_combiner.py）

```python
class SAMCombiner:
    def __init__(self, rho_cent: float = 40.0, rho_h: float = 15.0,
                 rho_grad: float = 0.6, close_kernel: int = 7,
                 min_connectivity: float = 0.92):
        ...

    def fragment_masks(self, frame_bgr: np.ndarray,
                       proposal: dict) -> list[np.ndarray]:
        """调用 SAM(或OpenCV连通域模拟) 输出碎片 mask 列表"""

    @staticmethod
    def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
        """无向 Hausdorff 距离 H(A,B)"""

    @staticmethod
    def convex_hull_andrew(points: np.ndarray) -> np.ndarray:
        """Andrew 单调链返回凸包顶点"""

    @staticmethod
    def connectivity(mask: np.ndarray) -> float:
        """最大连通域面积 / 总前景面积"""

    def combine(self, frame_bgr: np.ndarray,
                proposal: dict) -> dict:
        """
        返回 {
            'merged_mask': np.ndarray,
            'connectivity': float,        # 应 >= min_connectivity
            'num_fragments': int,
            'semantic_review_needed': bool # 连通度临界 -> 触发 VLM
        }
        """
```

### 6.3 VLM 终审 `VLMChatter`（vlm/validator.py）

```python
class VLMChatter:
    def __init__(self, model: str = "qwen2.5vl:7b",
                 ollama_url: str = "http://127.0.0.1:11434",
                 timeout_ms: int = 450, conf_threshold: float = 0.7):
        ...

    def should_review(self, proposal: dict) -> bool:
        """模糊区间 / 连通度临界 时返回 True"""

    def review(self, frame_bgr: np.ndarray, bbox) -> dict:
        """
        裁切 RoI -> base64 -> POST Ollama
        返回 {'label': 'dust_or_water'|'real_defect',
              'confidence': float, 'reason': str, 'timeout': bool}
        超时 -> 返回 timeout=True, 调用方按"可疑保留"处理
        """
```

### 6.4 流水线编排 `pipeline`

```python
class DefectPipeline:
    def __init__(self):
        self.detector   = ...            # 外部缺陷检测器
        self.filter     = FeatureFilter()
        self.combiner   = SAMCombiner()
        self.vlm        = VLMChatter()

    def run(self, frame_bgr: np.ndarray) -> list[dict]:
        rois      = self.detector.detect(frame_bgr)
        filtered  = self.filter.filter(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), rois)
        results   = []
        for p in filtered:
            if p['decision'] == 'reject':
                continue
            if p['decision'] == 'ambiguous':
                merged = self.combiner.combine(frame_bgr, p)
                if self.vlm.should_review(p) or merged['semantic_review_needed']:
                    verdict = self.vlm.review(frame_bgr, p['bbox'])
                    if verdict['label'] == 'dust_or_water' and verdict['confidence'] > 0.7 \
                       and not verdict['timeout']:
                        continue   # VLM 终审为干扰，丢弃
                p['mask'] = merged['merged_mask']
            results.append(p)
        return results
```

---

## 7. 性能规格（时延预算分配）

| 阶段 | 操作 | 时延预算 | 说明 |
|---|---|---|---|
| Stage-1 多特征初筛 | 连通域查找 + 4 类特征提取 + 评分 | **≤ 30 ms**（目标 ≤15 ms） | CPU/小 GPU 均可，算子均为 OpenCV 原生 |
| Stage-2 SAM 融合 | SAM 提示分割 + 碎片关联 + 凸包 + 形态学 | **≤ 50 ms**（目标 ≤40 ms） | 仅对模糊区间 RoI 执行，非全图 |
| Stage-3 VLM 终审 | qwen2.5vl:7b 本地推理 | **≤ 450 ms**（目标 ≤400 ms） | 仅 ~20% 样本触发，异步队列处理 |
| 端到端（不含 VLM） | Stage-1 + Stage-2 | **≤ 80 ms** | 对应 ≥12 FPS，满足 10Hz 产线 |
| 端到端（含 VLM） | 主线程 + VLM 异步 | 主线程 ≤80ms，VLM 后台不阻塞节拍 | 通过异步队列消化 |

显存预算：SAM（vit_b）约 375 MB + VLM 约 6 GB + 图像缓冲约数百 MB，总计需在 RTX 5070 8GB 内预留余量。

---

## 8. 验收标准对照表

| 编号 | 验收项 | 指标阈值 | 测量方法 | 是否达标 |
|---|---|---|---|---|
| AC-1 | 粉尘/水渍干扰排除率（TNR） | ≥ 96% | 50 张粉尘/水渍样本，正确排除数 / 总数 | 待测试集回测 |
| AC-2 | 真实缺陷漏检率（FNR） | = 0 | 100 张真实缺陷图，被误过滤数 = 0 | 待测试集回测 |
| AC-3 | SAM 碎片 Mask 拼接连通率 | ≥ 92% | 30 张条纹碎裂图，缝合后最大连通域/总面积均值 | 待测试集回测 |
| AC-4 | 初筛 + SAM 融合时延 | ≤ 80 ms/帧 | P95 实测，10Hz 压测脚本 | 待压测 |
| AC-5 | VLM 终审时延 | ≤ 450 ms/帧 | P95 实测（仅模糊区间） | 待压测 |
| AC-6 | 系统稳定性 | 不发生 OOM | 10Hz 连续压测 ≥ 30 min，显存/内存无泄漏 | 待压测 |
| AC-7 | 数学推导完整性 | 含 LBP + 颜色直方图融合的粉尘过滤数理推导 | 见《核心特征判定与凸包计算的数学模型》 | 已交付 |

> 测试脚本：`scripts/benchmark_line_speed.py`（产线模拟压测）；测试用例：`tests/test_defect_filter.py::test_dust_false_alarm_reduction` 与 `tests/test_stripe_mask_union`。Loop Engineering 机制：若 AC-1/AC-3 不达标，自动迭代 LBP 窗口半径 $R$ 与凸包闭运算核尺寸，直至全部通过。

---

*（本规格书经算法组评审定稿，后续权重/阈值变更需更新版本号并在 changelog 记录。）*
