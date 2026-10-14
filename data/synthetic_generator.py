# -*- coding: utf-8 -*-
"""
合成测试数据生成（data/synthetic_generator.py）
==============================================
生成三类测试样本并保存到 data/synthetic/：
1. 粉尘干扰图   dust_*.npz    玻璃背景 + 圆形粉尘暗斑（高圆度/低对比/边缘模糊）
2. 真实缺陷图   defect_*.npz  玻璃背景 + 不规则凹坑/崩边（高对比/边缘锐利），
                              并融合 KolektorSDD 数据集的真实缺陷 patch
3. 条纹碎裂mask stripe_*.npz  条纹栅栏背景 + 防伪标被竖向条纹切割成 mask 碎片

背景统一用"渐变灰度 + 高斯噪声"模拟手机盖板玻璃。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, TEST_DATA_CONFIG, REAL_DEFECT_DATASET)  # noqa: E402
from utils import imread_u  # noqa: E402


# --------------------------------------------------------------------- #
# 基础背景：渐变灰度 + 高斯噪声，模拟玻璃盖板
# --------------------------------------------------------------------- #
def _make_glass_background(size, rng):
    """生成渐变灰度玻璃背景 + 高斯噪声。"""
    h, w = size
    base = rng.uniform(185, 210)
    # 竖向轻微渐变
    grad = np.linspace(-8, 8, h, dtype=np.float32)[:, None]
    bg = np.full((h, w), base, dtype=np.float32) + grad
    bg += rng.normal(0, 2.5, (h, w))  # 高斯噪声
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    return bgr


# --------------------------------------------------------------------- #
# 1) 粉尘干扰图
# --------------------------------------------------------------------- #
def generate_dust_image(size=(512, 512), seed=0):
    """生成玻璃背景 + 圆形粉尘暗斑。

    粉尘特征：高圆度(接近1)、低对比度、边缘高斯模糊、内部灰度均匀。
    :return: (image BGR uint8, mask uint8 0/255)
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = _make_glass_background(size, rng).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 单粉尘点（避免两点融合导致圆度下降），位置与半径随机
    cx = rng.integers(70, w - 70)
    cy = rng.integers(70, h - 70)
    radius = rng.uniform(25, 45)
    # 画一个实心圆到 mask
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
    # 在图上叠加一个"暗斑"：用高斯衰减让边缘模糊、内部平滑
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # 中心最暗，向边缘平滑过渡
    dark = np.exp(-(dist ** 2) / (2 * (radius * 0.9) ** 2)) * rng.uniform(45, 70)
    for c in range(3):
        img[:, :, c] -= dark

    img = np.clip(img, 0, 255).astype(np.uint8)
    # mask 也做一点模糊，模拟粉尘边缘弥散
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    return img, mask


# --------------------------------------------------------------------- #
# 2) 真实缺陷图（融合 KolektorSDD 真实缺陷 patch）
# --------------------------------------------------------------------- #
def _load_kolektor_defects():
    """扫描 KolektorSDD，返回 [(img_bgr, defect_mask), ...] 真实缺陷样本列表。"""
    samples = []
    if not os.path.isdir(REAL_DEFECT_DATASET):
        return samples
    for kos in sorted(os.listdir(REAL_DEFECT_DATASET)):
        kp = os.path.join(REAL_DEFECT_DATASET, kos)
        if not os.path.isdir(kp):
            continue
        for f in sorted(os.listdir(kp)):
            if not f.endswith("_label.bmp"):
                continue
            label = imread_u(os.path.join(kp, f), cv2.IMREAD_GRAYSCALE)
            if label is None or (label > 128).sum() < 50:
                continue
            img_path = os.path.join(kp, f.replace("_label.bmp", ".jpg"))
            img = imread_u(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            samples.append((img, (label > 128).astype(np.uint8) * 255))
    return samples


def _make_programmatic_defect(canvas, mask, rng):
    """无真实数据集时程序化生成特征明确的缺陷（划痕 / 崩边 / 凹坑簇）。

    与粉尘（高圆度、内部均匀、边缘模糊）刻意拉开：缺陷形状细长或不规则、
    边缘锐利、内部深浅不一，使特征初筛（圆度 / 灰度方差 / LBP 纹理）能正确保留。
    :param canvas: float32 玻璃背景 (H,W,3)
    :param mask: 空的 uint8 缺陷掩码
    :return: (float32 image, uint8 mask)
    """
    h, w = mask.shape
    dmask = np.zeros((h, w), np.uint8)
    cx, cy = int(rng.integers(120, w - 120)), int(rng.integers(110, h - 110))
    kind = rng.choice(["scratch", "chip"], p=[0.5, 0.5])

    def _rot_xy(lx, ly, ang):
        return (int(cx + lx * np.cos(ang) - ly * np.sin(ang)),
                int(cy + lx * np.sin(ang) + ly * np.cos(ang)))

    if kind == "scratch":
        # 粗、略带弯曲的裂纹（凸包圆度低）；长宽比控制在 3~4.5，
        # 既区别于近圆粉尘，又不会落入"细长栅栏线"（aspect>5）误杀闸门
        thick = int(rng.integers(17, 27))
        length = int(thick * rng.uniform(3.2, 4.4))
        ang = rng.uniform(-0.7, 0.7)
        bend = rng.uniform(-0.08, 0.08) * length
        ts = np.linspace(-length / 2.0, length / 2.0, 11)
        pts = []
        for tt in ts:
            q = tt / (length / 2.0)
            pts.append(_rot_xy(tt, bend * q * q, ang))
        cv2.polylines(dmask, [np.asarray(pts, np.int32)], False, 255, thick, cv2.LINE_8)
        for px, py in (pts[0], pts[-1]):
            cv2.circle(dmask, (px, py), thick // 2, 255, -1)
        # 随机分叉，进一步破坏轮廓规则性
        if rng.random() < 0.5:
            mx, my = pts[len(pts) // 2]
            a2 = ang + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
            bl = int(length * rng.uniform(0.12, 0.25))
            cv2.line(dmask, (mx, my),
                     (int(mx + bl * np.cos(a2)), int(my + bl * np.sin(a2))),
                     255, max(3, thick // 2))
    else:
        # 拉伸、不规则的崩边，沿长轴两端各加尖刺，强制凸包细长（圆度低）
        base = int(rng.integers(40, 58))
        asp = rng.uniform(2.1, 2.9)
        ang = rng.uniform(0, np.pi)
        n = int(rng.integers(9, 14))
        angs = np.sort(rng.uniform(0, 2 * np.pi, n))
        radii = rng.uniform(0.4, 1.0, n)
        pts = []
        for a, rr in zip(angs, radii):
            pts.append(_rot_xy(base * asp * rr * np.cos(a), base * rr * np.sin(a), ang))
        cv2.fillPoly(dmask, [np.asarray(pts, np.int32)], 255)
        for sign in (-1, 1):
            sl = base * asp * rng.uniform(1.25, 1.55)
            tip = _rot_xy(sign * sl, 0.0, ang)
            bp = _rot_xy(sign * base * asp * 0.8, 0.0, ang)
            th = int(rng.uniform(6, 11))
            p1 = (int(bp[0] + th * np.cos(ang + 1.57)), int(bp[1] + th * np.sin(ang + 1.57)))
            p2 = (int(bp[0] - th * np.cos(ang + 1.57)), int(bp[1] - th * np.sin(ang + 1.57)))
            cv2.fillPoly(dmask, [np.asarray([tip, p1, p2], np.int32)], 255)

    # 强压暗、核心深边缘浅、内部深浅不一，提升 ROI 灰度方差与 LBP 纹理复杂度
    dist = cv2.distanceTransform(dmask, cv2.DIST_L2, 5)
    core = np.clip(dist / (dist.max() + 1e-6), 0.0, 1.0)
    base_alpha = rng.uniform(140, 185)
    sub = base_alpha * (0.40 + 0.60 * core)
    sub *= rng.uniform(0.45, 1.60, dmask.shape)
    soft = cv2.GaussianBlur(dmask, (3, 3), 0) / 255.0
    out = canvas.copy()
    for c in range(3):
        out[:, :, c] -= sub * soft
    ys, xs = np.where(dmask > 0)
    if len(ys) > 0:
        nspeck = int(len(ys) * 0.18)
        idx = rng.choice(len(ys), size=min(nspeck, len(ys)), replace=False)
        delta = rng.uniform(-90, 70, len(idx))
        for k, (yi, xi) in enumerate(zip(ys[idx], xs[idx])):
            out[yi, xi] += delta[k]
    return out, dmask


def generate_defect_image(size=(512, 512), seed=0, kolektor_samples=None):
    """生成玻璃背景 + 真实凹坑/崩边缺陷。

    优先叠加 KolektorSDD 的真实缺陷 patch（不规则形状/高对比/边缘锐利）；
    若无数据集则用程序生成的不规则多边形暗斑。
    :return: (image BGR uint8, mask uint8 0/255)
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = _make_glass_background(size, rng).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    placed = False
    if kolektor_samples:
        src_img, src_mask = kolektor_samples[seed % len(kolektor_samples)]
        # 找到缺陷 bbox，裁剪出含缺陷的 patch
        ys, xs = np.where(src_mask > 0)
        if len(xs) > 0:
            pad = 25
            x0, x1 = max(0, xs.min() - pad), min(src_mask.shape[1], xs.max() + pad)
            y0, y1 = max(0, ys.min() - pad), min(src_mask.shape[0], ys.max() + pad)
            patch_img = src_img[y0:y1, x0:x1]
            patch_mask = src_mask[y0:y1, x0:x1]
            # 缩放到合适大小
            target = rng.integers(90, 140)
            scale = target / float(max(patch_img.shape[:2]))
            patch_img = cv2.resize(patch_img, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            patch_mask = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_NEAREST)
            ph, pw = patch_img.shape[:2]
            px = rng.integers(30, w - pw - 30)
            py = rng.integers(30, h - ph - 30)
            # 把金属表面 patch 贴上去（带纹理 -> 高方差/高LBP）
            roi = img[py:py + ph, px:px + pw]
            pm3 = cv2.cvtColor(patch_mask, cv2.COLOR_GRAY2BGR) / 255.0
            # 非缺陷区域用 patch 金属纹理覆盖，缺陷区域进一步压暗
            blended = roi * (1 - pm3 * 0.0)  # 占位
            # 直接用 patch 的灰度纹理替换背景区域
            gray_patch = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
            gray_patch = cv2.normalize(gray_patch, None, 0, 255, cv2.NORM_MINMAX)
            for c in range(3):
                roi[:, :, c] = gray_patch.astype(np.float32)
            # 缺陷区域压暗（强化为锐利暗斑）
            defect_dark = pm3[:, :, 0] * rng.uniform(70, 110)
            for c in range(3):
                roi[:, :, c] -= defect_dark
            img[py:py + ph, px:px + pw] = roi
            mask[py:py + ph, px:px + pw] = np.maximum(
                mask[py:py + ph, px:px + pw], patch_mask)
            placed = True

    if not placed:
        # 无真实数据集时程序化生成缺陷（划痕 / 崩边 / 凹坑簇）
        img, mask = _make_programmatic_defect(img, mask, rng)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


# --------------------------------------------------------------------- #
# 3) 条纹栅栏背景 + 防伪标碎裂 mask
# --------------------------------------------------------------------- #
def generate_stripe_background(size=(512, 512), seed=0):
    """生成条纹栅栏背景，防伪标被竖向条纹切割成多个 mask 碎片。

    :return: (image BGR uint8, fragmented_masks: list[uint8], mark_mask GT uint8)
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = _make_glass_background(size, rng)

    # --- 画一个"防伪标"：圆角矩形/椭圆区域，灰度略暗 ---
    mark_mask = np.zeros((h, w), dtype=np.uint8)
    mx, my = w // 2, h // 2
    mw, mh = rng.integers(140, 200), rng.integers(90, 130)
    cv2.ellipse(mark_mask, (mx, my), (mw // 2, mh // 2), 0, 0, 360, 255, -1)
    # 防伪标整体压暗一点，使其可见
    img = img.astype(np.float32)
    m3 = cv2.cvtColor(mark_mask, cv2.COLOR_GRAY2BGR) / 255.0
    img -= m3 * 35
    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- 竖向条纹栅栏：用背景灰度的细竖条穿过防伪标，把它"切断" ---
    stripe_w = rng.integers(4, 8)
    gap = rng.integers(14, 22)
    bg_gray = int(rng.uniform(195, 205))
    fragments = []
    x = mx - mw // 2 - gap
    while x < mx + mw // 2:
        # 条纹条带（背景色）覆盖 -> 在 mark 上挖掉
        cv2.rectangle(mark_mask, (x, 0), (x + stripe_w, h), 0, -1)
        cv2.rectangle(img, (x, 0), (x + stripe_w, h),
                      (bg_gray, bg_gray, bg_gray), -1)
        x += stripe_w + gap

    # --- 把切剩下的防伪标区域按连通域拆成碎片 mask ---
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mark_mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        frag = (labels == i).astype(np.uint8) * 255
        fragments.append(frag)

    return img, fragments, mark_mask


# --------------------------------------------------------------------- #
# 4) 批量生成并保存
# --------------------------------------------------------------------- #
def generate_all_test_data():
    """按 TEST_DATA_CONFIG 生成全部测试数据并保存为 .npz。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    cfg = TEST_DATA_CONFIG
    size = cfg["image_size"]
    rng_seed = cfg["random_seed"]

    # 如果已有数据则跳过
    if (os.path.exists(os.path.join(DATA_DIR, "dust.npz"))
            and os.path.exists(os.path.join(DATA_DIR, "defect.npz"))
            and os.path.exists(os.path.join(DATA_DIR, "stripe.npz"))
            and len(os.listdir(DATA_DIR)) > 0):
        # 简单校验数量
        dz = np.load(os.path.join(DATA_DIR, "dust.npz"))
        if dz["images"].shape[0] >= cfg["n_dust_interference"]:
            print("[synthetic] 已存在合成数据，跳过生成。")
            return

    kolektor = _load_kolektor_defects()
    print(f"[synthetic] 加载 KolektorSDD 真实缺陷样本: {len(kolektor)} 个")

    # --- 粉尘干扰 ---
    dust_imgs, dust_masks = [], []
    for i in range(cfg["n_dust_interference"]):
        im, mk = generate_dust_image(size, seed=rng_seed + i)
        dust_imgs.append(im)
        dust_masks.append(mk)
    np.savez_compressed(os.path.join(DATA_DIR, "dust.npz"),
                        images=np.stack(dust_imgs),
                        masks=np.stack(dust_masks))
    print(f"[synthetic] 粉尘干扰图: {len(dust_imgs)} 张")

    # --- 真实缺陷 ---
    def_imgs, def_masks = [], []
    for i in range(cfg["n_real_defects"]):
        im, mk = generate_defect_image(size, seed=rng_seed + 1000 + i,
                                       kolektor_samples=kolektor)
        def_imgs.append(im)
        def_masks.append(mk)
    np.savez_compressed(os.path.join(DATA_DIR, "defect.npz"),
                        images=np.stack(def_imgs),
                        masks=np.stack(def_masks))
    print(f"[synthetic] 真实缺陷图: {len(def_imgs)} 张")

    # --- 条纹碎裂 mask ---
    stripe_imgs, stripe_marks, stripe_frags = [], [], []
    for i in range(cfg["n_stripe_masks"]):
        im, frags, mk = generate_stripe_background(size, seed=rng_seed + 2000 + i)
        stripe_imgs.append(im)
        stripe_marks.append(mk)
        stripe_frags.append(np.array(frags, dtype=object))
    np.savez_compressed(
        os.path.join(DATA_DIR, "stripe.npz"),
        images=np.stack(stripe_imgs),
        marks=np.stack(stripe_marks),
        fragments=np.array(stripe_frags, dtype=object),
        allow_pickle=True)
    print(f"[synthetic] 条纹碎裂图: {len(stripe_imgs)} 张")
    print("[synthetic] 全部测试数据生成完毕，保存于:", DATA_DIR)


if __name__ == "__main__":
    generate_all_test_data()
