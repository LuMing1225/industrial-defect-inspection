# -*- coding: utf-8 -*-
"""
工业缺陷干扰项过滤系统 - FastAPI 后端服务
==========================================
职责：
  1. 对外提供 REST API（健康检查 / 配置读写 / 缺陷检测 / 压测 / 统计）
  2. 串联完整检测流水线：多特征初筛 -> SAM 碎片融合 -> VLM 终审
  3. 对核心算法模块（detector / segmenter / vlm）做全局单例管理与优雅降级

设计要点：
  - 核心算法模块由另一路并行开发，本服务在 import 失败时自动切换到
    "演示降级模式"（用 OpenCV 连通域 / 形态学 / 直接调用 Ollama 实现同等接口），
    从而保证后端永远可以启动，等真实模块就绪后无需改动本文件即可自动接管。
  - VLM 调用做了 try/except + 超时保护，异常时不阻塞主流程，返回降级结果。
  - 运行时配置保存在内存字典中，/api/config GET/POST 实时读写，不影响落盘 config.py。
"""
import os
import sys
import time
import json
import base64
import asyncio
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------------
# 路径处理：把项目根目录加入 sys.path，保证 `import config` 与 `import detector...` 可用
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as CFG  # noqa: E402

# ----------------------------------------------------------------------------
# 日志
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("defect-backend")

# ============================================================================
# 第一部分：核心算法模块加载（真实模块优先，失败则降级）
# ============================================================================
REAL_MODULES_AVAILABLE = {"feature_filter": False, "sam_combiner": False, "vlm": False}

# ---- 尝试加载真实 FeatureFilter ----
try:
    from detector.feature_filter import FeatureFilter as _RealFeatureFilter  # type: ignore
    REAL_MODULES_AVAILABLE["feature_filter"] = True
except Exception as _e:  # pragma: no cover - 演示降级分支
    _RealFeatureFilter = None
    logger.warning("未找到真实 detector.feature_filter，启用内置降级实现。原因：%s", _e)

# ---- 尝试加载真实 SAMCombiner ----
try:
    from segmenter.sam_combiner import SAMCombiner as _RealSAMCombiner  # type: ignore
    REAL_MODULES_AVAILABLE["sam_combiner"] = True
except Exception as _e:  # pragma: no cover
    _RealSAMCombiner = None
    logger.warning("未找到真实 segmenter.sam_combiner，启用内置降级实现。原因：%s", _e)

# ---- 尝试加载真实 VLMValidator ----
try:
    from vlm.validator import VLMValidator as _RealVLMValidator  # type: ignore
    REAL_MODULES_AVAILABLE["vlm"] = True
except Exception as _e:  # pragma: no cover
    _RealVLMValidator = None
    logger.warning("未找到真实 vlm.validator，启用内置降级实现。原因：%s", _e)


# ============================================================================
# 第二部分：内置降级实现（保证后端可独立启动演示）
# 以下类与真实模块保持同样的方法签名，便于无缝替换。
# ============================================================================
class _DemoFeatureFilter:
    """降级版多特征初筛：用 OpenCV 连通域 + 圆度/方差/LBP 近似特征实现。
    构造参数与真实 FeatureFilter 一致，直接接收 FILTER_CONFIG 形状的字典。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or CFG.FILTER_CONFIG

    # 运行时配置可能被 /api/config 热更新，每次调用都读最新值
    def _thresholds(self) -> Dict[str, Any]:
        return self.cfg

    def find_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """用自适应阈值 + 连通域找出候选 bbox 列表。"""
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # 自适应阈值，对光照不均的工业图更鲁棒
        binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        min_area = self._thresholds().get("min_area", 50)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_area:
                continue
            out.append((int(x), int(y), int(w), int(h)))
        # 限制候选数量，避免一次返回过多
        return out[:40]

    def extract_features(self, image: np.ndarray, bbox) -> Dict[str, float]:
        """提取圆度、长宽比、灰度方差、LBP 对比度、面积。"""
        x, y, w, h = [int(v) for v in bbox]
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        crop = g[max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            return {"roundness": 0.0, "aspect_ratio": 1.0, "var_gray": 0.0,
                    "contrast_lbp": 0.0, "area": 0.0}
        area = float(w * h)
        perimeter = float(2 * (w + h))  # 近似周长
        # 圆度 = 4πA / P² （越接近1越圆）
        roundness = float(4 * np.pi * area / (perimeter * perimeter + 1e-6))
        roundness = float(np.clip(roundness, 0.0, 1.0))
        aspect_ratio = float(max(w, h) / (min(w, h) + 1e-6))
        # 灰度方差（归一化到 0-1，方差越小越像粉尘）
        var_gray = float(np.var(crop) / (255.0 * 255.0))
        # LBP 对比度近似：用 Sobel 梯度幅值均值代替（无需 skimage 依赖）
        gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(gx, gy)
        contrast_lbp = float(np.mean(grad_mag) / 255.0)
        return {
            "roundness": roundness,
            "aspect_ratio": aspect_ratio,
            "var_gray": float(np.clip(var_gray * 4.0, 0.0, 1.0)),  # 放大到可感知区间
            "contrast_lbp": float(np.clip(contrast_lbp, 0.0, 1.0)),
            "area": area,
        }

    def compute_score(self, features: Dict[str, float]) -> float:
        """
        过滤得分：Score = w1*圆度 + w2*(1-灰度方差) + w3*(1-LBP对比度)
        得分越高越像粉尘/干扰。
        """
        th = self._thresholds()
        w = th.get("weights", CFG.FILTER_CONFIG["weights"])
        w1 = w.get("w1_roundness", 0.4)
        w2 = w.get("w2_var_gray", 0.3)
        w3 = w.get("w3_contrast_lbp", 0.3)
        roundness = features.get("roundness", 0.0)
        var_gray = features.get("var_gray", 0.0)
        contrast = features.get("contrast_lbp", 0.0)
        # 粉尘：圆度高、内部均匀(方差低)、纹理平坦(LBP对比度低)
        score = (w1 * roundness
                 + w2 * (1.0 - var_gray)
                 + w3 * (1.0 - contrast))
        return float(np.clip(score, 0.0, 1.0))

    def is_interference(self, features: Dict[str, float], score: float) -> bool:
        """硬规则 + 评分联合判定：是否被初筛过滤为干扰项。"""
        th = self._thresholds()
        if score >= th.get("score_threshold", 0.65):
            return True
        # 近圆形 + 低方差 + 低纹理 → 粉尘
        if (features.get("roundness", 0) >= th.get("roundness_threshold", 0.82)
                and features.get("var_gray", 1) <= th.get("var_gray_threshold", 0.15)
                and features.get("aspect_ratio", 9) <= th.get("aspect_ratio_max", 1.5)):
            return True
        return False

    def filter_detections(self, image: np.ndarray,
                          contours) -> List[Dict[str, Any]]:
        """对一批候选 bbox 批量打分，返回结构化结果。"""
        results = []
        for bbox in contours:
            feats = self.extract_features(image, bbox)
            score = self.compute_score(feats)
            filtered = self.is_interference(feats, score)
            results.append({
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "features": feats,
                "score": score,
                "is_filtered": filtered,
                "filter_reason": "高圆度/低方差/低纹理 → 疑似粉尘干扰" if filtered else "通过初筛",
            })
        return results


class _DemoSAMCombiner:
    """降级版 SAM 融合：连通域 + 形态学闭运算 + 凸包重组。
    构造参数与真实 SAMCombiner 一致，直接接收 SAM_COMBINER_CONFIG 形状字典。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or CFG.SAM_COMBINER_CONFIG

    def _th(self) -> Dict[str, Any]:
        return self.cfg

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """把图中的暗斑区域切成若干 mask 碎片（模拟 SAM 输出）。"""
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        masks = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:
                continue
            m = (labels == i).astype(np.uint8) * 255
            masks.append(m)
        return masks

    def combine_masks(self, masks: List[np.ndarray],
                      image: np.ndarray) -> np.ndarray:
        """碎片融合：膨胀闭运算 + 凸包闭合，把相邻碎片缝成整体。"""
        if not masks:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        merged = np.zeros_like(masks[0])
        for m in masks:
            merged = cv2.bitwise_or(merged, m)
        k = self._th().get("dilate_kernel", 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 凸包重组
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        hull_img = np.zeros_like(merged)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_img, [hull], -1, 255, thickness=cv2.FILLED)
        return hull_img

    def mask_connectivity(self, combined: np.ndarray) -> float:
        """计算 mask 连通度：最大连通域面积 / mask 总面积。越接近1越好。"""
        if combined is None or combined.sum() == 0:
            return 0.0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        if num <= 1:
            return 1.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = float(areas.max())
        total = float(areas.sum())
        return float(np.clip(largest / (total + 1e-6), 0.0, 1.0))


class _DemoVLMValidator:
    """降级版 VLM：直接调用本地 Ollama，失败则返回降级结论。
    构造参数与真实 VLMValidator 一致，直接接收 VLM_CONFIG 形状字典。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or CFG.VLM_CONFIG

    def _th(self) -> Dict[str, Any]:
        return self.cfg

    def is_available(self) -> bool:
        """探测 Ollama 服务是否在线。"""
        try:
            import requests
            r = requests.get(self._th().get("ollama_url", "http://localhost:11434") + "/api/tags",
                             timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def needs_vlm_check(self, confidence: float) -> bool:
        th = self._th()
        lo = th.get("trigger_low", th.get("vlm_confidence_low", 0.45))
        hi = th.get("trigger_high", th.get("vlm_confidence_high", 0.75))
        return float(lo) <= float(confidence) <= float(hi)

    def validate(self, image_crop: np.ndarray) -> Dict[str, Any]:
        """调用 Qwen2.5-VL 终审；超时或异常时降级返回。"""
        th = self._th()
        # 编码图片为 base64（JPEG 减小体积）
        ok, buf = cv2.imencode(".jpg", image_crop)
        if not ok:
            return {"label": "缺陷", "confidence": 0.5, "reason": "图片编码失败，降级为待人工复核"}
        b64 = base64.b64encode(buf).decode("ascii")
        payload = {
            "model": th.get("model", "qwen2.5vl:7b"),
            "prompt": th.get("system_prompt", ""),
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            import requests
            url = th.get("ollama_url", "http://localhost:11434") + "/api/generate"
            r = requests.post(url, json=payload, timeout=th.get("timeout", 30))
            r.raise_for_status()
            text = r.json().get("response", "")
            return self._parse_response(text)
        except Exception as e:
            logger.warning("VLM 调用失败，降级处理：%s", e)
            return {"label": "缺陷", "confidence": 0.5,
                    "reason": f"VLM 不可用({type(e).__name__})，按真实缺陷保留待复核"}

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """从模型返回文本中尽量解析出 JSON；失败则做关键字兜底。"""
        text = text.strip()
        # 尝试提取 {...} 片段
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            return {
                "label": obj.get("label", "缺陷"),
                "confidence": float(obj.get("confidence", 0.6)),
                "reason": obj.get("reason", text[:80]),
            }
        except Exception:
            if "干扰" in text or "粉尘" in text or "水渍" in text:
                return {"label": "干扰", "confidence": 0.7, "reason": text[:80]}
            return {"label": "缺陷", "confidence": 0.6, "reason": text[:80]}


# ============================================================================
# 第三部分：全局单例管理
# ============================================================================
# 运行时配置（内存态，可被 /api/config 热更新）
RUNTIME_CONFIG: Dict[str, Any] = {
    "filter": CFG.FILTER_CONFIG,
    "sam": CFG.SAM_COMBINER_CONFIG,
    "vlm": CFG.VLM_CONFIG,
}


def _build_singletons() -> Dict[str, Any]:
    """根据真实模块是否存在，构造三套单例。
    注意：真实模块构造函数直接接收 FILTER_CONFIG / SAM_COMBINER_CONFIG / VLM_CONFIG 形状的字典，
    因此这里传 RUNTIME_CONFIG 的子字典，而非外层包装。
    """
    ff_cfg = RUNTIME_CONFIG["filter"]
    sam_cfg = RUNTIME_CONFIG["sam"]
    vlm_cfg = RUNTIME_CONFIG["vlm"]

    if REAL_MODULES_AVAILABLE["feature_filter"] and _RealFeatureFilter is not None:
        ff = _RealFeatureFilter(ff_cfg)
    else:
        ff = _DemoFeatureFilter(ff_cfg)

    if REAL_MODULES_AVAILABLE["sam_combiner"] and _RealSAMCombiner is not None:
        sc = _RealSAMCombiner(sam_cfg)
    else:
        sc = _DemoSAMCombiner(sam_cfg)

    if REAL_MODULES_AVAILABLE["vlm"] and _RealVLMValidator is not None:
        vv = _RealVLMValidator(vlm_cfg)
    else:
        vv = _DemoVLMValidator(vlm_cfg)

    return {"feature_filter": ff, "sam_combiner": sc, "vlm_validator": vv}


SINGLES = _build_singletons()

# 历史统计（内存态）
STATS: Dict[str, Any] = {
    "total_frames": 0,
    "total_candidates": 0,
    "filtered_by_feature": 0,
    "passed_to_sam": 0,
    "vlm_checked": 0,
    "final_defects": 0,
    "total_interference": 0,
    "latency_history_ms": [],   # 最近若干帧总耗时
}


# ============================================================================
# 第四部分：可视化标注工具
# ============================================================================
def _load_chinese_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """加载一个支持中文的字体（Windows 自带 msyh.ttc）。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",     # 黑体
        r"C:\Windows\Fonts\simsun.ttc",     # 宋体
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def visualize(image: np.ndarray,
              defects: List[Dict[str, Any]],
              combined_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    在原图上绘制：
      - 绿色框 = 最终判定缺陷
      - 橙色框 = 被过滤的干扰项（半透明）
      - 半透明彩色 mask 叠加
      - 中文标签（用 PIL 绘制，避免 cv2 中文乱码）
    """
    out = image.copy()
    h, w = out.shape[:2]

    # 1) mask 半透明叠加（青色）
    if combined_mask is not None and combined_mask.sum() > 0:
        mask_resized = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = out.copy()
        overlay[mask_resized > 0] = (0.6 * overlay[mask_resized > 0]
                                     + 0.4 * np.array([255, 200, 0])).astype(np.uint8)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    # 2) 转 PIL 画中文
    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_chinese_font(18)
    font_small = _load_chinese_font(14)

    for d in defects:
        x, y, bw, bh = d["bbox"]
        label = d.get("final_label", "缺陷")
        if label == "缺陷":
            color_rgb = (0, 220, 80)   # 绿
            thickness = 2
        else:
            color_rgb = (255, 140, 0)  # 橙
            thickness = 1
        # PIL 坐标
        draw.rectangle([x, y, x + bw, y + bh], outline=color_rgb, width=thickness)
        # 标签背景条
        tag = f"{label} {d.get('score', 0):.2f}"
        tw = draw.textlength(tag, font=font_small)
        draw.rectangle([x, max(0, y - 18), x + tw + 6, y], fill=color_rgb)
        draw.text((x + 3, max(0, y - 17)), tag, fill=(0, 0, 0), font=font_small)

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out


# ============================================================================
# 第五部分：FastAPI 应用
# ============================================================================
app = FastAPI(title="工业缺陷干扰项过滤系统", version="1.0.0")

# CORS：允许前端（8522）及任何来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求体模型 ----
class ConfigUpdate(BaseModel):
    """/api/config POST 的请求体：允许局部更新 filter/sam/vlm 配置。"""
    filter: Optional[Dict[str, Any]] = None
    sam: Optional[Dict[str, Any]] = None
    vlm: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health():
    """健康检查：返回后端端口、Ollama 状态、GPU 信息。"""
    ff_ok = REAL_MODULES_AVAILABLE["feature_filter"]
    sam_ok = REAL_MODULES_AVAILABLE["sam_combiner"]
    vlm_ok = REAL_MODULES_AVAILABLE["vlm"]
    try:
        ollama_available = SINGLES["vlm_validator"].is_available()
    except Exception:
        ollama_available = False

    gpu_info = "未知"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        else:
            gpu_info = "CPU 模式"
    except Exception:
        gpu_info = "torch 未安装，CPU 模式"

    return {
        "status": "ok",
        "backend_port": CFG.BACKEND_PORT,
        "ollama_available": ollama_available,
        "gpu_info": gpu_info,
        "modules_loaded": {
            "feature_filter": ff_ok,
            "sam_combiner": sam_ok,
            "vlm": vlm_ok,
        },
        "uptime_s": int(time.time() - START_TIME),
    }


@app.get("/api/config")
def get_config():
    """返回当前运行时配置（供前端参数面板回显）。"""
    return RUNTIME_CONFIG


@app.post("/api/config")
def update_config(cfg: ConfigUpdate):
    """局部更新运行时配置。"""
    if cfg.filter is not None:
        # 允许更新 weights / score_threshold 等字段；深度合并
        RUNTIME_CONFIG["filter"].update(cfg.filter)
    if cfg.sam is not None:
        RUNTIME_CONFIG["sam"].update(cfg.sam)
    if cfg.vlm is not None:
        RUNTIME_CONFIG["vlm"].update(cfg.vlm)
    return {"status": "ok", "config": RUNTIME_CONFIG}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    完整检测流水线：
      1. cv2 解码上传图片
      2. 多特征初筛
      3. 未过滤区域走 SAM 碎片融合
      4. 模糊区间走 VLM 终审
      5. 可视化并返回 JSON
    """
    t_total0 = time.perf_counter()

    # ---- 1. 解码图片 ----
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="图片解码失败，请上传有效图片")
    h, w = image.shape[:2]

    ff = SINGLES["feature_filter"]
    sc = SINGLES["sam_combiner"]
    vv = SINGLES["vlm_validator"]

    # ---- 2. 多特征初筛 ----
    t0 = time.perf_counter()
    contours = ff.find_contours(image)
    raw_results = ff.filter_detections(image, contours)
    feature_ms = (time.perf_counter() - t0) * 1000.0

    total_candidates = len(raw_results)
    filtered = [r for r in raw_results if r["is_filtered"]]
    passed = [r for r in raw_results if not r["is_filtered"]]

    # ---- 3. SAM 碎片融合（对通过初筛的整图做一次融合） ----
    t1 = time.perf_counter()
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    connectivity = 0.0
    try:
        if passed:
            masks = sc.segment(image)
            combined_mask = sc.combine_masks(masks, image)
            connectivity = sc.mask_connectivity(combined_mask)
    except Exception as e:
        logger.warning("SAM 融合异常：%s", e)
    sam_ms = (time.perf_counter() - t1) * 1000.0

    # ---- 4. VLM 终审（对模糊区间结果异步并发，带超时保护） ----
    t2 = time.perf_counter()
    vlm_checked = 0
    VLM_TIMEOUT_SEC = 10.0  # VLM单次调用超时，超时则默认保留为缺陷（不漏检）

    async def _run_vlm_with_timeout(crop: np.ndarray):
        """在线程池中跑同步 requests，带超时保护，避免阻塞事件循环。"""
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, vv.validate, crop),
                timeout=VLM_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            return {"label": "缺陷", "confidence": 0.5,
                    "reason": f"VLM超时({VLM_TIMEOUT_SEC}s)，默认保留为真实缺陷待复核"}
        except Exception as e:
            return {"label": "缺陷", "confidence": 0.5,
                    "reason": f"VLM异常:{type(e).__name__}，默认保留为缺陷"}

    vlm_tasks = []
    vlm_indices = []
    for idx, r in enumerate(passed):
        conf = 1.0 - r["score"]  # 把过滤得分转成"是缺陷"的置信度
        if vv.needs_vlm_check(conf):
            x, y, bw, bh = r["bbox"]
            crop = image[max(y, 0):y + bh, max(x, 0):x + bw]
            if crop.size > 0:
                vlm_tasks.append(_run_vlm_with_timeout(crop))
                vlm_indices.append(idx)

    vlm_results = [None] * len(passed)
    if vlm_tasks:
        done = await asyncio.gather(*vlm_tasks, return_exceptions=True)
        for i, idx in enumerate(vlm_indices):
            res = done[i]
            if isinstance(res, Exception):
                res = {"label": "缺陷", "confidence": 0.5, "reason": f"VLM异常:{res}"}
            vlm_results[idx] = res
            vlm_checked += 1
    vlm_ms = (time.perf_counter() - t2) * 1000.0

    # ---- 5. 组装最终缺陷列表 ----
    final_defects: List[Dict[str, Any]] = []
    for idx, r in enumerate(passed):
        conf = 1.0 - r["score"]
        v_res = vlm_results[idx]
        if v_res is not None:
            # VLM已终审，采纳VLM结论
            final_label = v_res.get("label", "缺陷")
        else:
            # 未触发VLM：通过初筛即视为真实缺陷（保证漏检率=0）
            # conf>0.75(score<0.25)为高置信缺陷；conf<0.45(score>0.55)为临界但未过滤，保守保留
            final_label = "缺陷"
        final_defects.append({
            "bbox": r["bbox"],
            "features": r["features"],
            "score": round(r["score"], 4),
            "sam_mask_connectivity": round(float(connectivity), 4),
            "vlm_result": v_res if v_res else {"label": "无需VLM", "confidence": round(conf, 3), "reason": "通过初筛，置信度明确，判定为真实缺陷"},
            "final_label": final_label,
        })

    # 被初筛过滤的也画出来（橙色）
    for r in filtered:
        final_defects.append({
            "bbox": r["bbox"],
            "features": r["features"],
            "score": round(r["score"], 4),
            "sam_mask_connectivity": 0.0,
            "vlm_result": {"label": "干扰", "confidence": round(1 - r["score"], 3),
                           "reason": r.get("filter_reason", "初筛过滤")},
            "final_label": "干扰",
        })

    total_ms = (time.perf_counter() - t_total0) * 1000.0

    # ---- 6. 可视化 ----
    vis = visualize(image, final_defects, combined_mask)
    ok, buf = cv2.imencode(".png", vis)
    vis_b64 = "data:image/png;base64," + base64.b64encode(buf).decode("ascii") if ok else ""

    # ---- 7. 更新统计 ----
    STATS["total_frames"] += 1
    STATS["total_candidates"] += total_candidates
    STATS["filtered_by_feature"] += len(filtered)
    STATS["passed_to_sam"] += len(passed)
    STATS["vlm_checked"] += vlm_checked
    STATS["final_defects"] += sum(1 for d in final_defects if d["final_label"] == "缺陷")
    STATS["total_interference"] += sum(1 for d in final_defects if d["final_label"] == "干扰")
    STATS["latency_history_ms"].append(round(total_ms, 2))
    if len(STATS["latency_history_ms"]) > 100:
        STATS["latency_history_ms"] = STATS["latency_history_ms"][-100:]

    return {
        "image_size": [int(w), int(h)],
        "total_candidates": total_candidates,
        "filtered_by_feature": len(filtered),
        "passed_to_sam": len(passed),
        "vlm_checked": vlm_checked,
        "final_defects": final_defects,
        "timing_ms": {
            "feature_filter": round(feature_ms, 2),
            "sam_combine": round(sam_ms, 2),
            "vlm_validate": round(vlm_ms, 2),
            "total": round(total_ms, 2),
        },
        "visualization": vis_b64,
    }


@app.post("/api/benchmark")
def benchmark():
    """
    产线压测：模拟 10Hz 输入合成图像，统计各阶段平均耗时与吞吐。
    """
    n_frames = CFG.BENCHMARK_CONFIG.get("duration_seconds", 30) * CFG.BENCHMARK_CONFIG.get("fps_target", 10)
    n_frames = min(n_frames, 60)  # 演示环境限制最多跑60帧
    fw, fh = CFG.BENCHMARK_CONFIG.get("image_size", (1024, 768))
    ff = SINGLES["feature_filter"]
    sc = SINGLES["sam_combiner"]

    timings = []
    for i in range(n_frames):
        # 合成一张带噪声的工业图
        img = np.random.randint(200, 230, (fh, fw, 3), dtype=np.uint8)
        # 随机撒几个暗斑
        for _ in range(np.random.randint(1, 5)):
            cx, cy = np.random.randint(50, fw - 50), np.random.randint(50, fh - 50)
            cv2.circle(img, (cx, cy), np.random.randint(3, 12), (30, 30, 30), -1)
        t0 = time.perf_counter()
        contours = ff.find_contours(img)
        _ = ff.filter_detections(img, contours)
        masks = sc.segment(img)
        _ = sc.combine_masks(masks, img)
        timings.append((time.perf_counter() - t0) * 1000.0)

    arr_t = np.array(timings)
    result = {
        "frames": int(n_frames),
        "avg_ms": round(float(arr_t.mean()), 2),
        "p95_ms": round(float(np.percentile(arr_t, 95)), 2),
        "max_ms": round(float(arr_t.max()), 2),
        "min_ms": round(float(arr_t.min()), 2),
        "throughput_fps": round(1000.0 / max(float(arr_t.mean()), 1e-3), 2),
        "target_latency_ms": CFG.ACCEPTANCE_CRITERIA["filter_sam_latency_ms"],
        "pass": bool(arr_t.mean() <= CFG.ACCEPTANCE_CRITERIA["filter_sam_latency_ms"]),
    }
    logger.info("压测完成：%s", result)
    return result


@app.get("/api/stats")
def stats():
    """返回历史检测统计（内存态）。"""
    n = STATS["total_frames"]
    avg_lat = float(np.mean(STATS["latency_history_ms"])) if STATS["latency_history_ms"] else 0.0
    interference_rate = 0.0
    if STATS["total_candidates"] > 0:
        interference_rate = STATS["filtered_by_feature"] / STATS["total_candidates"]
    return {
        "total_frames": n,
        "total_candidates": STATS["total_candidates"],
        "filtered_by_feature": STATS["filtered_by_feature"],
        "passed_to_sam": STATS["passed_to_sam"],
        "vlm_checked": STATS["vlm_checked"],
        "final_defects": STATS["final_defects"],
        "total_interference": STATS["total_interference"],
        "interference_exclusion_rate": round(interference_rate, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "recent_latency_ms": STATS["latency_history_ms"][-30:],
    }


# 启动时间戳
START_TIME = time.time()


# 入口：python -m uvicorn backend.app:app --host 0.0.0.0 --port 8562
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=CFG.BACKEND_HOST, port=CFG.BACKEND_PORT, reload=False)
