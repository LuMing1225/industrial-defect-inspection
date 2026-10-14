# -*- coding: utf-8 -*-
"""
工业缺陷干扰项过滤系统 - 全局配置
所有模块共享此配置，端口、阈值、路径统一管理
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# 真实工业缺陷数据集路径（可选，KolektorSDD），未设置时使用程序生成的合成缺陷样本
REAL_DEFECT_DATASET = os.environ.get("KOLEKTOR_SDD_DIR", "")

# ==================== 服务端口 ====================
BACKEND_PORT = 8562
FRONTEND_PORT = 8522
BACKEND_HOST = "0.0.0.0"
FRONTEND_HOST = "0.0.0.0"

# ==================== 多特征初筛参数 ====================
FILTER_CONFIG = {
    # 评分公式: Score_filter = w1*Roundness + w2*VarGray + w3*Contrast_LBP
    "weights": {
        "w1_roundness": 0.4,    # 圆度权重（粉尘圆度高→得分高→过滤）
        "w2_var_gray": 0.3,     # 灰度方差权重（粉尘方差低→得分高→过滤）
        "w3_contrast_lbp": 0.3, # LBP对比度权重（粉尘纹理均匀→得分高→过滤）
    },
    # 硬阈值：Score >= threshold 判定为干扰项（粉尘/水渍）
    "score_threshold": 0.65,
    # 模糊区间：置信度在此区间内触发VLM终审
    "vlm_confidence_low": 0.45,
    "vlm_confidence_high": 0.75,
    # 圆度阈值：圆度 > 0.85 视为近圆形（粉尘特征）
    "roundness_threshold": 0.82,
    # 长宽比阈值
    "aspect_ratio_max": 1.5,
    # 灰度方差阈值（归一化0-1）：粉尘内部灰度均匀，方差低
    "var_gray_threshold": 0.15,
    # LBP对比度阈值
    "lbp_contrast_threshold": 0.2,
    # LBP提取参数
    "lbp_radius": 2,
    "lbp_n_points": 16,
    # 最小缺陷面积（像素）
    "min_area": 50,
}

# ==================== SAM碎片融合参数 ====================
SAM_COMBINER_CONFIG = {
    # Hausdorff距离阈值（像素）：低于此值的碎片进行融合
    "hausdorff_threshold": 30.0,
    # 中心距离阈值
    "center_distance_threshold": 50.0,
    # 形态学膨胀核大小
    "dilate_kernel": 5,
    # 形态学腐蚀核大小
    "erode_kernel": 3,
    # 凸包平滑迭代次数
    "convex_smooth_iterations": 2,
    # Mask连通度目标 >= 92%
    "target_connectivity": 0.92,
    # SAM模型类型（vit_b最轻量）
    "sam_model_type": "vit_b",
    # SAM权重路径（若不存在则使用OpenCV连通域模拟）
    "sam_checkpoint": os.path.join(WEIGHTS_DIR, "sam_vit_b.pth"),
    # 是否使用真实SAM（False时用连通域分析模拟碎片效果）
    "use_real_sam": False,
}

# ==================== VLM校验参数 ====================
VLM_CONFIG = {
    # Ollama API地址
    "ollama_url": "http://localhost:11434",
    # 使用的视觉语言模型
    "model": "qwen2.5vl:7b",
    # 超时时间（秒）
    "timeout": 30,
    # System Prompt：判断是否为粉尘/水渍干扰
    "system_prompt": (
        "你是一名工业质检专家。请判断图片中的暗斑/异常区域是【真实缺陷】还是【干扰项】。\n"
        "干扰项包括：空气中的粉尘颗粒、水渍、反光、镜头污渍。\n"
        "真实缺陷包括：玻璃凹坑、崩边、划痕、裂纹、气泡。\n"
        "判断依据：\n"
        "1. 粉尘通常呈规则圆形、边缘模糊、内部灰度均匀\n"
        "2. 水渍通常呈不规则扩散状、有折射光晕\n"
        "3. 真实缺陷边缘锐利、形状不规则、有深度感\n"
        "请只回答JSON格式：{\"label\": \"缺陷\"或\"干扰\", \"confidence\": 0.0-1.0, \"reason\": \"简短原因\"}"
    ),
    # 模糊区间触发VLM
    "trigger_low": 0.45,
    "trigger_high": 0.75,
}

# ==================== 测试数据参数 ====================
TEST_DATA_CONFIG = {
    "n_real_defects": 100,      # 真实缺陷图数量
    "n_dust_interference": 50,  # 粉尘干扰图数量
    "n_stripe_masks": 30,       # 条纹背景碎裂Mask图数量
    "image_size": (512, 512),   # 合成图像尺寸
    "random_seed": 42,
}

# ==================== 压测参数 ====================
BENCHMARK_CONFIG = {
    "fps_target": 10,           # 产线拍摄速率 10Hz
    "duration_seconds": 30,     # 压测时长
    "image_size": (1024, 768),  # 产线图像分辨率
    "n_warmup": 5,              # 预热帧数
}

# ==================== 验收硬指标 ====================
ACCEPTANCE_CRITERIA = {
    "dust_exclusion_rate": 0.96,    # 粉尘/水渍干扰排除率 ≥96%
    "real_defect_miss_rate": 0.0,   # 真实缺陷漏检率 = 0
    "mask_connectivity": 0.92,      # SAM碎片Mask拼接连通率 ≥92%
    "filter_sam_latency_ms": 80,    # 初筛+SAM融合 ≤80ms/帧
    "vlm_latency_ms": 450,          # VLM终审 ≤450ms/帧
    "no_oom": True,                 # 系统不发生OOM
}
