# 模型权重

权重文件体积较大，不纳入 Git 版本库。请按下面的方式自行下载并放到本目录。

## SAM（Segment Anything，可选）

| 项 | 说明 |
|---|---|
| 文件名 | `sam_vit_b.pth` |
| 官方权重 | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)（约 358 MB） |
| 项目主页 | https://github.com/facebookresearch/segment-anything |

下载后重命名为 `sam_vit_b.pth` 并放到本目录：

```
weights/
└── sam_vit_b.pth
```

然后在 `config.py` 中把 `SAM_COMBINER_CONFIG["use_real_sam"]` 改为 `True`。

## 不下载权重也可以运行

默认 `use_real_sam = False`，此时碎片分割会回退到基于 OpenCV 连通域 / 自适应阈值的
实现（`SAMCombiner._segment_mimic`），聚类、Hausdorff 距离与凸包融合的逻辑与真实 SAM
完全一致，**不影响接口、测试与前后端演示**，仅分割精细度与真实模型有差距。

这意味着：

- `pip install -r requirements.txt` 后即可直接运行全部测试与 Demo；
- 若需在真实产线图像上获得最佳分割效果，再按上表下载 SAM 权重即可。

> VLM 终审使用本地 [Ollama](https://ollama.com/) 运行的 `qwen2.5vl:7b`，
> 未启动 Ollama 时自动降级为规则推断，不会阻塞主流程。
