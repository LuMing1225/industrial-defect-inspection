# -*- coding: utf-8 -*-
"""Generate README demo figures from the real pipeline + synthetic samples.

Run:  python scripts/make_demo_figures.py
Outputs PNGs into ../assets/
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as _cfg                                       # noqa: E402
from detector.feature_filter import FeatureFilter           # noqa: E402
from segmenter.sam_combiner import SAMCombiner              # noqa: E402
from data.synthetic_generator import (                      # noqa: E402
    generate_dust_image, generate_defect_image,
    generate_stripe_background, _load_kolektor_defects)

ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

ff = FeatureFilter()
combiner = SAMCombiner()

# Real defect patches (KolektorSDD) if the env var KOLEKTOR_SDD_DIR is set.
KOLE = _load_kolektor_defects()
print("[fig] KolektorSDD real defect patches:", len(KOLE))


def draw_boxes(img, retained, show_filtered=True, show_kept=True):
    canvas = img.copy()
    for r in retained:
        x, y, w, h = r["bbox"]
        if r.get("is_filtered") and not show_filtered:
            continue
        if (not r.get("is_filtered")) and not show_kept:
            continue
        if r.get("is_filtered"):
            # filtered interference -> gray dashed box
            color = (150, 150, 150)
            for dx in range(0, w, 8):  # dashed top/bottom
                cv2.line(canvas, (x + dx, y), (min(x + dx + 4, x + w), y), color, 1)
                cv2.line(canvas, (x + dx, y + h), (min(x + dx + 4, x + w), y + h), color, 1)
            cv2.putText(canvas, "filtered", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # kept real defect -> solid red box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 220), 2)
            cv2.putText(canvas, "defect", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
    return canvas


def find_sample(kind, max_seed=120):
    """Pick a seed with the expected outcome; return (img, res, rep_feats, rep)."""
    last = None
    for seed in range(max_seed):
        if kind == "dust":
            img, _ = generate_dust_image(seed=seed)
            want_filtered = True
        else:
            img, _ = generate_defect_image(seed=seed, kolektor_samples=KOLE)
            want_filtered = False
        boxes = ff.find_contours(img)
        if not boxes:
            continue
        res = ff.filter_detections(img, boxes)
        rep = next((r for r in res if bool(r.get("is_filtered")) == want_filtered),
                   None)
        last = (img, res, rep["features"] if rep else res[0]["features"],
                rep or res[0])
        if rep is not None:
            return last
    return last


# ------------------------------------------------------------------ #
# Figure 1: feature filter  (dust filtered vs real defect kept)
# ------------------------------------------------------------------ #
dust_img, dust_res, dust_f, dust_rep = find_sample("dust")
def_img, def_res, def_f, def_rep = find_sample("defect")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
axes[0].imshow(cv2.cvtColor(
    draw_boxes(dust_img, dust_res, show_kept=False), cv2.COLOR_BGR2RGB))
axes[0].set_title(
    "Dust / water-stain -> filtered\n(roundness=%.2f, score=%.2f)"
    % (dust_f.get("roundness", 0), dust_rep.get("score", 0)), fontsize=11)
axes[1].imshow(cv2.cvtColor(
    draw_boxes(def_img, def_res, show_filtered=False), cv2.COLOR_BGR2RGB))
axes[1].set_title(
    "Real pit / chip -> kept\n(roundness=%.2f, score=%.2f)"
    % (def_f.get("roundness", 0), def_rep.get("score", 0)), fontsize=11)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "feature_filter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[fig] feature_filter.png")


# ------------------------------------------------------------------ #
# Figure 2: SAM fragment fusion (fragmented -> fused convex hull)
# ------------------------------------------------------------------ #
def find_stripe_sample(max_seed=40):
    best = None
    for seed in range(max_seed):
        img, frags, mark = generate_stripe_background(seed=seed)
        masks = combiner.segment(img)
        if len(masks) >= 4:
            fused = combiner.combine_masks(masks, img)
            conn = combiner.mask_connectivity(fused)
            if best is None or len(masks) > len(best[1]):
                best = (img, masks, fused, conn)
    return best


img, masks, fused, conn = find_stripe_sample()
# Cluster-merge the fragments, keep the dominant component (the security mark),
# then convex-hull fill it into one complete outline.
fused = combiner.combine_masks(masks, img)
_nc, _lb, _st, _ = cv2.connectedComponentsWithStats((fused > 0).astype(np.uint8), 8)
if _nc > 1:
    _k = 1 + int(np.argmax(_st[1:, cv2.CC_STAT_AREA]))
    fused = combiner._convex_hull_fill((_lb == _k).astype(np.uint8) * 255)
conn = combiner.mask_connectivity(fused)

# colorize fragmented masks
frag_vis = img.copy()
rng = np.random.default_rng(1)
for m in masks:
    color = tuple(int(v) for v in rng.integers(60, 255, size=3))
    for c in range(3):
        frag_vis[:, :, c] = np.where(m > 0, (0.55 * frag_vis[:, :, c] + 0.45 * color[c]),
                                     frag_vis[:, :, c])

fused_vis = img.copy()
for c in range(3):
    fused_vis[:, :, c] = np.where(fused > 0,
                                  (0.5 * fused_vis[:, :, c] + 0.5 * (60, 220, 90)[c]),
                                  fused_vis[:, :, c])

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
axes[0].imshow(cv2.cvtColor(frag_vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[0].set_title("Fragmented masks (%d pieces, over-segmentation)" % len(masks), fontsize=11)
axes[1].imshow(cv2.cvtColor(fused_vis.astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[1].set_title("Fused by clustering + convex hull\n(connectivity=%.1f%%)"
                  % (conn * 100), fontsize=11)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "mask_fusion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[fig] mask_fusion.png")


# ------------------------------------------------------------------ #
# Figure 3: three-stage pipeline overview (real defect)
# ------------------------------------------------------------------ #
img2, boxes, res = None, None, None
for seed in range(1000, 1130):
    im, _ = generate_defect_image(seed=seed, kolektor_samples=KOLE)
    bx = ff.find_contours(im)
    rr = ff.filter_detections(im, bx)
    if any(not r.get("is_filtered") for r in rr):
        img2, boxes, res = im, bx, rr
        break

# merged bounding box of all kept real defects + a margin ROI
kept = [r for r in res if not r.get("is_filtered")]
xs0 = min(r["bbox"][0] for r in kept)
ys0 = min(r["bbox"][1] for r in kept)
xs1 = max(r["bbox"][0] + r["bbox"][2] for r in kept)
ys1 = max(r["bbox"][1] + r["bbox"][3] for r in kept)
m = 90
cx0, cy0 = max(0, xs0 - m), max(0, ys0 - m)
cx1, cy1 = min(img2.shape[1], xs1 + m), min(img2.shape[0], ys1 + m)

masks2 = combiner.segment(img2)
fused2 = combiner.combine_masks(masks2, img2)

s1 = img2.copy()
for (x, y, w, h) in boxes:
    cv2.rectangle(s1, (x, y), (x + w, y + h), (0, 180, 255), 1)

s2 = img2.copy()
cv2.rectangle(s2, (xs0, ys0), (xs1, ys1), (0, 0, 220), 2)
cv2.putText(s2, "defect", (xs0, max(14, ys0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

s3 = img2.copy()
for c in range(3):
    s3[:, :, c] = np.where(fused2 > 0, (0.5 * s3[:, :, c] + 0.5 * (60, 220, 90)[c]),
                           s3[:, :, c])


def _crop(im):
    return im[cy0:cy1, cx0:cx1]


s1, s2, s3 = _crop(s1), _crop(s2), _crop(s3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
titles = ["Stage 1: candidate detection\n(Otsu + adaptive threshold)",
          "Stage 2: feature filter\n(roundness / variance / LBP)",
          "Stage 3: mask fusion\n(Hausdorff + convex hull)"]
for ax, im, ti in zip(axes, [s1, s2, s3], titles):
    ax.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax.set_title(ti, fontsize=10.5)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "pipeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[fig] pipeline.png")
print("done ->", ASSETS)
