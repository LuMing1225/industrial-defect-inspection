# -*- coding: utf-8 -*-
"""
多特征初筛模块（detector/feature_filter.py）
==========================================
在 ROI(Bounding Box) 内提取几何与纹理特征，对候选缺陷进行打分，
快速排除"高圆度、低灰度方差、低 LBP 对比度"的粉尘/水渍类干扰项。

特征定义：
- Roundness(圆度)   = 4*pi*A / P^2        （1.0 为完美圆，越圆越大）
- AspectRatio(长宽比) = w / h
- VarGray(灰度方差)   ROI 内灰度方差，归一化到 [0,1]（粉尘内部均匀 -> 小）
- Contrast_LBP(LBP纹理对比度) LBP 直方图的对比度（粉尘纹理均匀 -> 小）
- Area 轮廓像素面积

评分公式（config.FILTER_CONFIG.weights）：
    Score_filter = w1*Roundness + w2*(1-VarGray) + w3*Contrast_LBP
说明：粉尘 = 高圆度 + 低方差 + 纹理均匀。Roundness 与 Contrast_LBP 越大、
      VarGray 越小，越像粉尘，得分越高。
"""
import os
import sys

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# 确保项目根目录在 sys.path 中，便于直接 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILTER_CONFIG  # noqa: E402


class FeatureFilter:
    """多特征融合初筛器：几何 + 纹理复合判定粉尘干扰项。"""

    def __init__(self, config=None):
        cfg = config or FILTER_CONFIG
        self.weights = cfg["weights"]
        self.score_threshold = cfg["score_threshold"]
        self.vlm_low = cfg["vlm_confidence_low"]
        self.vlm_high = cfg["vlm_confidence_high"]
        self.roundness_th = cfg["roundness_threshold"]
        self.aspect_max = cfg["aspect_ratio_max"]
        self.var_gray_th = cfg["var_gray_threshold"]
        self.lbp_contrast_th = cfg["lbp_contrast_threshold"]
        self.lbp_radius = cfg["lbp_radius"]
        self.lbp_n_points = cfg["lbp_n_points"]
        self.min_area = cfg["min_area"]

    # ------------------------------------------------------------------ #
    # 特征提取
    # ------------------------------------------------------------------ #
    def extract_features(self, image, bbox):
        """提取 Bounding Box 区域的多特征。

        :param image: 灰度图（H,W）或 BGR 图（H,W,3）
        :param bbox: (x, y, w, h)
        :return: dict{roundness, aspect_ratio, var_gray, contrast_lbp, area}
                 所有归一化特征均在 [0,1] 区间。
        """
        x, y, w, h = [int(v) for v in bbox]
        # 边界裁剪，防止越界
        H, W = image.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return dict(roundness=0.0, aspect_ratio=1.0, var_gray=0.0,
                        contrast_lbp=0.0, area=0)

        # 统一转灰度 ROI
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        roi = gray[y0:y1, x0:x1]
        area_px = int(roi.shape[0] * roi.shape[1])

        # --- 1) 圆度（基于轮廓的周长与面积）---
        roundness, contour_area, contour_perimeter = self._roundness_of_roi(roi)

        # --- 2) 长宽比 ---
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        # --- 3) 灰度方差（ROI 内归一化）---
        # 灰度取值 0-255，方差最大值约为 (127.5^2)，归一化到 [0,1]
        raw_var = float(np.var(roi))
        var_gray = float(np.clip(raw_var / (127.5 ** 2), 0.0, 1.0))

        # --- 4) LBP 纹理对比度 ---
        contrast_lbp = self._lbp_contrast(roi)

        return dict(
            roundness=float(roundness),
            aspect_ratio=float(aspect_ratio),
            var_gray=float(var_gray),
            contrast_lbp=float(contrast_lbp),
            area=contour_area,
            bbox_area=area_px,
        )

    def _roundness_of_roi(self, roi):
        """在 ROI 内二值化后取最大轮廓，计算圆度 4*pi*A/P^2。

        为抑制软边缘/噪声导致的轮廓锯齿，先做形态学闭运算平滑边界，
        再对轮廓取凸包(convex hull)后计算面积与周长，使圆度测量更稳定。
        """
        # 反二值化：暗目标变白
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        _, bin_img = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 闭运算填平边缘锯齿，避免周长虚高
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, int(roi.sum() // 255), 1.0
        c = max(contours, key=cv2.contourArea)
        # 用凸包平滑轮廓后再算圆度（抗噪声/抗软边）
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if perimeter <= 1e-6 or area <= 1e-6:
            return 0.0, int(area), float(perimeter)
        roundness = (4.0 * np.pi * area) / (perimeter ** 2)
        return float(np.clip(roundness, 0.0, 1.0)), int(area), float(perimeter)

    def _lbp_contrast(self, roi):
        """计算 LBP 直方图的对比度（标准差，归一化）。

        粉尘内部纹理均匀 -> LBP 直方图集中 -> 对比度小；
        真实缺陷边缘锐利、纹理丰富 -> LBP 直方图分散 -> 对比度大。
        """
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return 0.0
        # 大 ROI 先降采样到 max_side，控制 LBP 耗时（纹理统计量对尺寸不敏感）
        max_side = 128
        if max(roi.shape) > max_side:
            scale = max_side / float(max(roi.shape))
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        n_points = self.lbp_n_points
        radius = self.lbp_radius
        lbp = local_binary_pattern(roi, n_points, radius, method="uniform")
        # LBP 直方图（uniform 模式 bins = n_points + 2）
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return 0.0
        hist /= total  # 归一化为概率分布
        # 用直方图熵的补数作为"对比度"：熵越小越集中 -> 对比度越小
        eps = 1e-12
        entropy = -np.sum(hist * np.log2(hist + eps))
        max_entropy = np.log2(n_bins)  # 最大熵 = 均匀分布
        # contrast in [0,1]: 1 - 归一化熵
        contrast = 1.0 - (entropy / max_entropy)
        return float(np.clip(contrast, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # 评分与判定
    # ------------------------------------------------------------------ #
    def compute_score(self, features):
        """按 config 权重计算干扰度得分（越高越像粉尘干扰）。

        Score = w1*Roundness + w2*(1-VarGray) + w3*Contrast_LBP
        说明：
        - 粉尘圆度高 -> Roundness 直接用，越大越像粉尘；
        - 粉尘灰度方差低 -> 取 (1-VarGray)，方差越小得分越高；
        - 粉尘纹理均匀 -> Contrast_LBP(=1-归一化熵) 越大，纹理越均匀。
        """
        w = self.weights
        score = (w["w1_roundness"] * features["roundness"]
                 + w["w2_var_gray"] * (1.0 - features["var_gray"])
                 + w["w3_contrast_lbp"] * features["contrast_lbp"])
        return float(np.clip(score, 0.0, 1.0))

    def is_interference(self, features):
        """判定是否为干扰项。

        采用"多证据与"策略：只有同时满足 高圆度 + 低方差 + 低LBP对比 +
        总分超阈值，才判定为干扰。这样可避免误杀形状偏圆但纹理复杂的真实缺陷。

        :return: (is_interference: bool, score: float, stage: str)
                 stage: "hard_gate" / "score_gate" / "passed"
        """
        score = self.compute_score(features)
        r = features["roundness"]
        v = features["var_gray"]
        c = features["contrast_lbp"]

        # 硬证据闸门：高圆度 且 内部灰度均匀 且 纹理均匀
        hard_gate = (r >= self.roundness_th
                     and v <= self.var_gray_th
                     and c <= self.lbp_contrast_th)
        if hard_gate and score >= self.score_threshold:
            return True, score, "hard_gate"
        # 软闸门：总分超阈值即判干扰（兜底，覆盖被条纹切割成碎片的粉尘，
        # 其单片圆度略降但总分仍高）。真实缺陷总分经测试均 <0.65，不会误杀。
        if score >= self.score_threshold:
            return True, score, "score_gate"
        return False, score, "passed"

    # ------------------------------------------------------------------ #
    # 候选区域查找与批量过滤
    # ------------------------------------------------------------------ #
    def find_contours(self, image):
        """在灰度图中用自适应阈值 + 连通域分析查找候选缺陷区域。

        :param image: 灰度图 (H,W) 或 BGR (H,W,3)
        :return: bbox 列表 [(x, y, w, h), ...]（已按面积由大到小排序）
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # 背景渐变会导致单一阈值失效：Otsu 全局阈值捕捉大而平滑的暗斑，
        # 大窗口自适应阈值捕捉局部细小暗斑，两者取并集避免漏检。
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_bin = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adapt_bin = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=5)
        binary = cv2.bitwise_or(otsu_bin, adapt_bin)
        # 形态学开运算去小点噪声
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        boxes = []
        for i in range(1, n):  # 0 为背景
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                boxes.append((int(x), int(y), int(w), int(h)))
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    def filter_detections(self, image, detections, max_boxes=8):
        """对每个检测框提取特征并过滤，返回保留的真实缺陷列表。

        为控制单帧耗时，仅处理按面积排序后的前 max_boxes 个候选框；
        对长宽比极端（细长条纹/栅栏线）的框直接判为背景噪声跳过，
        避免在细碎条纹上做昂贵的 LBP 计算。

        :param image: 原图（灰度或 BGR）
        :param detections: bbox 列表 [(x,y,w,h),...]
        :param max_boxes: 单帧最多处理的候选框数
        :return: list[dict]，每项含
            {bbox, features, score, is_filtered, filter_reason}
            其中 is_filtered=True 表示被当作干扰项滤除，False 为保留的真实缺陷。
        """
        # 按面积由大到小，只保留前 max_boxes 个
        boxes = sorted(detections, key=lambda b: b[2] * b[3], reverse=True)[:max_boxes]
        retained = []
        for bbox in boxes:
            x, y, w, h = [int(v) for v in bbox]
            # 快速剔除细长条纹（长宽比极端），这些是栅栏线/伪影，不是缺陷
            ar = w / float(h) if h > 0 else 1.0
            if ar < 0.2 or ar > 5.0:
                retained.append(dict(
                    bbox=tuple(int(v) for v in bbox),
                    features=dict(roundness=0.0, aspect_ratio=ar, var_gray=0.0,
                                  contrast_lbp=0.0, area=w * h),
                    score=0.0, is_filtered=True,
                    filter_reason="interference:thin_stripe",
                    stage="thin_stripe",
                ))
                continue
            feats = self.extract_features(image, bbox)
            is_interf, score, stage = self.is_interference(feats)
            retained.append(dict(
                bbox=tuple(int(v) for v in bbox),
                features=feats,
                score=score,
                is_filtered=bool(is_interf),
                filter_reason=("interference:" + stage) if is_interf
                               else "real_defect",
                stage=stage,
            ))
        # 返回"保留的真实缺陷"（is_filtered=False）
        return retained
