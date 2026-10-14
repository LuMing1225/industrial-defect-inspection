# -*- coding: utf-8 -*-
"""
SAM 碎片融合模块（segmenter/sam_combiner.py）
============================================
条纹栅栏/防伪标背景下，SAM 会把一个完整目标打碎成上百个 mask 碎片。
本模块：
1. segment(image)                分割图像返回 mask 碎片列表（优先真实 SAM，否则用 OpenCV 模拟碎片）
2. compute_hausdorff(mask1,mask2) 两个二值 mask 边缘点集的 Hausdorff 距离
3. combine_masks(masks)           按中心距离 + Hausdorff 距离聚类合并，凸包缝合 + 形态学平滑
4. convex_hull_merge(masks)       所有 mask 取并集后求凸包并填充
5. mask_connectivity(mask)        连通度 = 最大连通域面积 / 总前景面积
"""
import os
import sys

import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAM_COMBINER_CONFIG  # noqa: E402


class SAMCombiner:
    """SAM 碎片化掩膜的重组与边缘平滑引擎。"""

    def __init__(self, config=None):
        cfg = config or SAM_COMBINER_CONFIG
        self.hausdorff_th = cfg["hausdorff_threshold"]
        self.center_dist_th = cfg["center_distance_threshold"]
        self.dilate_kernel = cfg["dilate_kernel"]
        self.erode_kernel = cfg["erode_kernel"]
        self.smooth_iter = cfg["convex_smooth_iterations"]
        self.target_connectivity = cfg["target_connectivity"]
        self.use_real_sam = cfg["use_real_sam"]
        self.sam_checkpoint = cfg["sam_checkpoint"]
        self.sam_model_type = cfg["sam_model_type"]
        self._sam_predictor = None  # 懒加载

    # ------------------------------------------------------------------ #
    # 1) 分割
    # ------------------------------------------------------------------ #
    def segment(self, image):
        """分割图像，返回 mask 碎片列表（每个 mask 为 HxW 的 uint8 二值图）。

        优先加载真实 SAM vit_b；权重不存在时退化为 OpenCV 自适应阈值 +
        连通域分析，并把大连通域"随机碎裂"成多个小碎片，以模拟条纹背景下
        SAM 过分割的失败现象。
        """
        if self.use_real_sam and os.path.exists(self.sam_checkpoint):
            return self._segment_with_sam(image)
        return self._segment_mimic(image)

    def _segment_with_sam(self, image):
        """真实 SAM 推理（权重存在时启用）。"""
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            if self._sam_predictor is None:
                sam = sam_model_registry[self.sam_model_type](
                    checkpoint=self.sam_checkpoint)
                sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
                self._sam_predictor = SamAutomaticMaskGenerator(sam)
            bgr = image
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            anns = self._sam_predictor.generate(rgb)
            return [np.array(a["segmentation"], dtype=np.uint8) * 255
                    for a in anns]
        except Exception:
            # 任何异常都回退到模拟模式，保证流水线可用
            return self._segment_mimic(image)

    def _segment_mimic(self, image):
        """OpenCV 模拟 SAM 碎片：自适应阈值 -> 连通域 -> 大连通域随机碎裂。"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=31, C=7)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        masks = []
        rng = np.random.default_rng(0)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < 50:
                continue
            # 跳过细长条纹栅栏线（长宽比极端），避免产生大量碎片拖慢融合
            ar = ww / float(hh) if hh > 0 else 1.0
            if ar < 0.15 or ar > 6.0:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            if area > 800:
                # 大连通域模拟 SAM 过分割：用若干条竖向切割线把它打碎
                frags = self._shatter_component(comp, rng)
                masks.extend(frags)
            else:
                masks.append(comp)
        return masks

    @staticmethod
    def _shatter_component(comp, rng, n_cuts=4):
        """把大连通域沿随机竖向缝隙切成若干小碎片，模拟条纹切割。"""
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            return [comp]
        x_min, x_max = xs.min(), xs.max()
        width = x_max - x_min
        if width < 20:
            return [comp]
        cuts = np.sort(rng.choice(
            np.arange(x_min + 5, x_max - 5),
            size=min(n_cuts, max(1, width // 20)), replace=False))
        fragments = []
        prev = x_min - 1
        for cx in cuts:
            piece = np.zeros_like(comp)
            piece[:, prev + 1:cx] = comp[:, prev + 1:cx]
            if (piece > 0).sum() > 30:
                fragments.append(piece)
            prev = cx
        piece = np.zeros_like(comp)
        piece[:, prev + 1:x_max + 1] = comp[:, prev + 1:x_max + 1]
        if (piece > 0).sum() > 30:
            fragments.append(piece)
        return fragments if fragments else [comp]

    # ------------------------------------------------------------------ #
    # 2) Hausdorff 距离
    # ------------------------------------------------------------------ #
    def _edge_points(self, mask, max_points=120):
        """提取二值 mask 的边缘点坐标集合 (N,2)。

        直接用 findContours 取外轮廓点（比全图膨胀/腐蚀快很多），
        边缘点超过 max_points 时等间距降采样以加速 Hausdorff 计算。
        """
        m = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            ys, xs = np.where(m > 0)
            pts = np.column_stack([xs, ys]).astype(np.float32)
        else:
            c = max(contours, key=cv2.contourArea)
            pts = c[:, 0, :].astype(np.float32)  # (N,2)
        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = pts[idx]
        return pts

    def compute_hausdorff(self, mask1, mask2):
        """两个二值 mask 边缘点集的 Hausdorff 距离（像素）。"""
        pts1 = self._edge_points(mask1)
        pts2 = self._edge_points(mask2)
        return self._hausdorff_pts(pts1, pts2)

    @staticmethod
    def _hausdorff_pts(pts1, pts2):
        """两组已提取边缘点集的 Hausdorff 距离（双向最大值）。"""
        if len(pts1) == 0 or len(pts2) == 0:
            return float("inf")
        d1 = directed_hausdorff(pts1, pts2)[0]
        d2 = directed_hausdorff(pts2, pts1)[0]
        return float(max(d1, d2))

    # ------------------------------------------------------------------ #
    # 3) 融合核心
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mask_center(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.array([0.0, 0.0])
        return np.array([xs.mean(), ys.mean()])

    def combine_masks(self, masks, image=None):
        """核心融合算法：
        1. 计算相邻碎片中心距离 + Hausdorff 距离，低于阈值的碎片归为同一簇；
        2. 对每簇取并集 -> 凸包(Convex Hull) 填充；
        3. 对最终结果做 膨胀 -> 腐蚀 形态学平滑，保证边缘连续无缝。

        :param masks: list[np.ndarray] 二值 mask 碎片
        :param image: 可选原图（预留，当前融合不依赖）
        :return: 合并后的二值 mask (H,W) uint8
        """
        valid = [m for m in masks if m is not None and (m > 0).sum() > 0]
        if not valid:
            return np.zeros((image.shape[:2]) if image is not None else (1, 1),
                            dtype=np.uint8)
        H = max(m.shape[0] for m in valid)
        W = max(m.shape[1] for m in valid)

        # ---- 步骤1: 基于中心距离 + Hausdorff 距离的凝聚聚类 ----
        n = len(valid)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        centers = [self._mask_center(m) for m in valid]
        # 预计算每个 mask 的边缘点集，避免两两比较时重复提取轮廓
        edge_pts = [self._edge_points(m) for m in valid]
        for i in range(n):
            for j in range(i + 1, n):
                cdist = float(np.linalg.norm(centers[i] - centers[j]))
                if cdist > self.center_dist_th * 3:
                    # 中心距离过远，直接不合并
                    continue
                hd = self._hausdorff_pts(edge_pts[i], edge_pts[j])
                if cdist <= self.center_dist_th or hd <= self.hausdorff_th:
                    union(i, j)

        clusters = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        # ---- 步骤2: 每簇并集 + 凸包填充 ----
        merged = np.zeros((H, W), dtype=np.uint8)
        for _, idxs in clusters.items():
            cluster_union = np.zeros((H, W), dtype=np.uint8)
            for i in idxs:
                h, w = valid[i].shape
                cluster_union[:h, :w] = cv2.bitwise_or(
                    cluster_union[:h, :w], (valid[i] > 0).astype(np.uint8) * 255)
            hull = self._convex_hull_fill(cluster_union)
            merged = cv2.bitwise_or(merged, hull)

        # ---- 步骤3: 形态学平滑（膨胀->腐蚀）缝合边缘 ----
        kd = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel))
        ke = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.erode_kernel, self.erode_kernel))
        merged = cv2.dilate(merged, kd, iterations=self.smooth_iter)
        merged = cv2.erode(merged, ke, iterations=self.smooth_iter)
        return merged

    def _convex_hull_fill(self, mask):
        """对单个 mask 的前景求凸包并填充。"""
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        out = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) < 10:
                continue
            hull = cv2.convexHull(c)
            cv2.fillConvexPoly(out, hull, 255)
        return out

    # ------------------------------------------------------------------ #
    # 4) 凸包全局合并
    # ------------------------------------------------------------------ #
    def convex_hull_merge(self, masks):
        """对所有 mask 取并集后计算凸包，填充整个凸包区域。"""
        valid = [m for m in masks if m is not None and (m > 0).sum() > 0]
        if not valid:
            return np.zeros((1, 1), dtype=np.uint8)
        H = max(m.shape[0] for m in valid)
        W = max(m.shape[1] for m in valid)
        union = np.zeros((H, W), dtype=np.uint8)
        for m in valid:
            h, w = m.shape
            union[:h, :w] = cv2.bitwise_or(
                union[:h, :w], (m > 0).astype(np.uint8) * 255)
        return self._convex_hull_fill(union)

    # ------------------------------------------------------------------ #
    # 5) 连通度
    # ------------------------------------------------------------------ #
    def mask_connectivity(self, mask):
        """连通度 = 最大连通域面积 / 总前景面积。

        完全连通返回 1.0，碎成很多片时趋近于 1/N。
        """
        m = (mask > 0).astype(np.uint8)
        total = int(m.sum())
        if total == 0:
            return 0.0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return 1.0
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        return float(largest / total)
