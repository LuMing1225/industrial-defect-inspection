# -*- coding: utf-8 -*-
"""
自动化测试（tests/test_defect_filter.py）
========================================
验收项：
- test_dust_false_alarm_reduction  粉尘干扰排除率 >= 96%
- test_real_defect_no_miss         真实缺陷漏检率 = 0
- test_stripe_mask_union           SAM 碎片融合后连通度 >= 92%
- test_feature_extraction          圆度/长宽比/灰度方差/LBP 计算正确性
- test_score_formula               评分公式加权正确性
- test_vlm_validator               VLM 校验器接口（真实或 fallback）
- test_filter_sam_latency          单帧 初筛+SAM融合 <= 80ms
- test_end_to_end_pipeline         端到端流水线
"""
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, ACCEPTANCE_CRITERIA  # noqa: E402
from detector.feature_filter import FeatureFilter  # noqa: E402
from segmenter.sam_combiner import SAMCombiner  # noqa: E402
from vlm.validator import VLMValidator  # noqa: E402
from data.synthetic_generator import generate_all_test_data  # noqa: E402


# --------------------------------------------------------------------- #
# 夹具：测试前若 data/synthetic 为空则自动生成
# --------------------------------------------------------------------- #
def _ensure_data():
    needed = ["dust.npz", "defect.npz", "stripe.npz"]
    if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in needed):
        generate_all_test_data()


@pytest.fixture(scope="session")
def data():
    _ensure_data()
    dust = np.load(os.path.join(DATA_DIR, "dust.npz"))
    defect = np.load(os.path.join(DATA_DIR, "defect.npz"))
    stripe = np.load(os.path.join(DATA_DIR, "stripe.npz"), allow_pickle=True)
    return dict(
        dust_imgs=dust["images"], dust_masks=dust["masks"],
        def_imgs=defect["images"], def_masks=defect["masks"],
        stripe_imgs=stripe["images"], stripe_frags=stripe["fragments"],
        stripe_marks=stripe["marks"],
    )


@pytest.fixture(scope="session")
def filter_obj():
    return FeatureFilter()


@pytest.fixture(scope="session")
def combiner_obj():
    return SAMCombiner()


@pytest.fixture(scope="session")
def vlm():
    return VLMValidator()


def _pick_best_box(ff, image, gt_mask):
    """在检测框中选与 GT mask 重叠最大的那个。"""
    boxes = ff.find_contours(image)
    best, best_ov = None, -1.0
    gt_area = max(1, int((gt_mask > 0).sum()))
    for b in boxes:
        x, y, w, h = [int(v) for v in b]
        roi = np.zeros_like(gt_mask)
        roi[y:y + h, x:x + w] = 255
        ov = np.logical_and(roi > 0, gt_mask > 0).sum() / gt_area
        if ov > best_ov:
            best_ov, best = ov, b
    return best


# --------------------------------------------------------------------- #
# 1) 粉尘干扰排除率 >= 96%
# --------------------------------------------------------------------- #
def test_dust_false_alarm_reduction(filter_obj, data):
    """加载 50 张粉尘图，>=96% 应被正确判为干扰项（>=48/50）。"""
    n_filtered = 0
    total = len(data["dust_imgs"])
    for i in range(total):
        im = data["dust_imgs"][i]
        gt = data["dust_masks"][i]
        best = _pick_best_box(filter_obj, im, gt)
        assert best is not None, f"第{i}张粉尘图未检出暗斑"
        feats = filter_obj.extract_features(im, best)
        is_interf, score, stage = filter_obj.is_interference(feats)
        if is_interf:
            n_filtered += 1
    rate = n_filtered / total
    print(f"\n粉尘排除率: {n_filtered}/{total} = {rate:.2%}")
    assert rate >= ACCEPTANCE_CRITERIA["dust_exclusion_rate"], \
        f"粉尘排除率 {rate:.2%} 未达 {ACCEPTANCE_CRITERIA['dust_exclusion_rate']:.0%}"


# --------------------------------------------------------------------- #
# 2) 真实缺陷漏检率 = 0
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("idx", range(100))
def test_real_defect_no_miss(filter_obj, data, idx):
    """加载 100 张真实缺陷图，全部必须保留为真实缺陷（漏检=0）。"""
    im = data["def_imgs"][idx]
    gt = data["def_masks"][idx]
    best = _pick_best_box(filter_obj, im, gt)
    assert best is not None, f"第{idx}张真实缺陷图未检出"
    feats = filter_obj.extract_features(im, best)
    is_interf, score, stage = filter_obj.is_interference(feats)
    assert not is_interf, \
        f"第{idx}张真实缺陷被误判为干扰! score={score:.3f} " \
        f"roundness={feats['roundness']:.3f} var={feats['var_gray']:.3f} " \
        f"lbp={feats['contrast_lbp']:.3f}"


# --------------------------------------------------------------------- #
# 3) 条纹碎裂 mask 融合连通度 >= 92%
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("idx", range(30))
def test_stripe_mask_union(combiner_obj, data, idx):
    """加载 30 张条纹碎裂 mask，融合后连通度 >= 92%。"""
    frags = list(data["stripe_frags"][idx])
    merged = combiner_obj.combine_masks(frags)
    conn = combiner_obj.mask_connectivity(merged)
    assert conn >= ACCEPTANCE_CRITERIA["mask_connectivity"], \
        f"第{idx}张条纹mask融合连通度 {conn:.3f} 未达 0.92"


# --------------------------------------------------------------------- #
# 4) 特征提取正确性
# --------------------------------------------------------------------- #
def test_feature_extraction(filter_obj):
    """已知圆形 -> 圆度接近 1.0；验证各特征量纲合理。"""
    # 构造一个白底黑圆
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, 60, -1)
    feats = filter_obj.extract_features(img, (55, 55, 90, 90))
    # 圆度应接近 1.0
    assert feats["roundness"] > 0.8, f"圆形圆度 {feats['roundness']:.3f} 过低"
    # 长宽比应接近 1
    assert 0.8 < feats["aspect_ratio"] < 1.2
    # 灰度方差在 [0,1]
    assert 0.0 <= feats["var_gray"] <= 1.0
    assert 0.0 <= feats["contrast_lbp"] <= 1.0
    # 面积 > 0
    assert feats["area"] > 0

    # 构造一个细长矩形（长宽比大、圆度低）
    img2 = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.rectangle(img2, (20, 90), (180, 110), 60, -1)
    feats2 = filter_obj.extract_features(img2, (15, 85, 170, 30))
    assert feats2["aspect_ratio"] > 4.0, "细长矩形长宽比应大于4"
    assert feats2["roundness"] < feats["roundness"], \
        "细长形状圆度应低于正圆"


# --------------------------------------------------------------------- #
# 5) 评分公式正确性
# --------------------------------------------------------------------- #
def test_score_formula(filter_obj):
    """验证 Score = w1*R + w2*(1-Var) + w3*LBP 加权正确。"""
    w = filter_obj.weights
    feats = dict(roundness=0.9, var_gray=0.1, contrast_lbp=0.4)
    expected = (w["w1_roundness"] * 0.9
                + w["w2_var_gray"] * (1 - 0.1)
                + w["w3_contrast_lbp"] * 0.4)
    score = filter_obj.compute_score(feats)
    assert abs(score - expected) < 1e-6, f"评分公式错误: {score} != {expected}"
    # 单调性：圆度越高得分越高
    s1 = filter_obj.compute_score(dict(roundness=0.9, var_gray=0.1, contrast_lbp=0.4))
    s2 = filter_obj.compute_score(dict(roundness=0.4, var_gray=0.1, contrast_lbp=0.4))
    assert s1 > s2


# --------------------------------------------------------------------- #
# 6) VLM 校验器接口
# --------------------------------------------------------------------- #
def test_vlm_validator():
    """测试 VLM 校验器：可用则真实调用，不可用则 fallback，返回格式正确。"""
    vlm = VLMValidator()
    # 模糊区间触发判定
    assert vlm.needs_vlm_check(0.6) is True
    assert vlm.needs_vlm_check(0.1) is False
    assert vlm.needs_vlm_check(0.95) is False

    # 构造一个圆形暗斑 crop
    crop = np.full((128, 128, 3), 210, dtype=np.uint8)
    cv2.circle(crop, (64, 64), 30, 120, -1)
    ctx = dict(roundness=0.95, var_gray=0.05)
    result = vlm.validate(crop, context=ctx)
    # 无论走真实 Ollama 还是 fallback，都必须返回约定字段
    assert "label" in result and result["label"] in ("缺陷", "干扰")
    assert "confidence" in result and 0.0 <= result["confidence"] <= 1.0
    assert "reason" in result


# --------------------------------------------------------------------- #
# 7) 初筛 + SAM 融合时延 <= 80ms
# --------------------------------------------------------------------- #
def test_filter_sam_latency(filter_obj, combiner_obj, data):
    """单帧 找连通域+特征初筛+SAM融合 总耗时 <= 80ms。"""
    im = data["dust_imgs"][0]
    times = []
    for _ in range(5):  # 预热+多次取平均
        t0 = time.perf_counter()
        boxes = filter_obj.find_contours(im)
        dets = filter_obj.filter_detections(im, boxes)
        masks = combiner_obj.segment(im)
        if masks:
            combiner_obj.combine_masks(masks)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = float(np.mean(times[1:]))  # 去掉首次预热
    print(f"\n初筛+SAM融合平均耗时: {avg_ms:.1f} ms")
    assert avg_ms <= ACCEPTANCE_CRITERIA["filter_sam_latency_ms"], \
        f"初筛+SAM融合 {avg_ms:.1f}ms 超过 80ms 性能预算"


# --------------------------------------------------------------------- #
# 8) 端到端流水线
# --------------------------------------------------------------------- #
def test_end_to_end_pipeline(filter_obj, combiner_obj, vlm, data):
    """输入图像 -> 找连通域 -> 特征过滤 -> SAM融合 -> VLM终审 -> 输出结果。"""
    im = data["def_imgs"][0]  # 用真实缺陷图走完整流水线
    # 1. 找连通域
    boxes = filter_obj.find_contours(im)
    assert len(boxes) > 0, "端到端：未检出候选区域"
    # 2. 特征过滤
    dets = filter_obj.filter_detections(im, boxes)
    # 3. SAM 融合
    masks = combiner_obj.segment(im)
    if masks:
        merged = combiner_obj.combine_masks(masks)
        conn = combiner_obj.mask_connectivity(merged)
        assert 0.0 <= conn <= 1.0
    # 4. VLM 终审（对保留的缺陷 crop）
    kept = [d for d in dets if not d["is_filtered"]]
    assert len(kept) > 0, "端到端：真实缺陷应被保留"
    x, y, w, h = kept[0]["bbox"]
    crop = im[y:y + h, x:x + w]
    verdict = vlm.validate(crop, context=kept[0]["features"])
    assert verdict["label"] in ("缺陷", "干扰")
