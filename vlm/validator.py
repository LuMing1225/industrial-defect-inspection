# -*- coding: utf-8 -*-
"""
大模型兜底校验模块（vlm/validator.py）
=====================================
当初筛置信度落在模糊区间时，裁切缺陷 ROI 送入本地 Qwen2.5-VL
（Ollama 服务）进行终审，判断是"真实缺陷"还是"粉尘/水渍干扰"。

- Ollama 不可用 / 超时 / 返回异常时，自动退化为基于图像特征的 fallback
  判定（高圆度 -> 干扰），保证产线不阻塞。
- 通信协议：POST {ollama_url}/api/chat, stream=false,
  messages=[{role:system,content:prompt}, {role:user, content:..., images:[b64]}]
"""
import os
import sys
import re
import json
import base64

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VLM_CONFIG  # noqa: E402


class VLMValidator:
    """基于 Ollama Qwen2.5-VL 的视觉兜底校验器。"""

    def __init__(self, config=None):
        cfg = config or VLM_CONFIG
        self.ollama_url = cfg["ollama_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = cfg["timeout"]
        self.system_prompt = cfg["system_prompt"]
        self.trigger_low = cfg["trigger_low"]
        self.trigger_high = cfg["trigger_high"]

    # ------------------------------------------------------------------ #
    # 服务可用性
    # ------------------------------------------------------------------ #
    def is_available(self):
        """检查 Ollama 服务是否运行（GET /api/tags）。"""
        try:
            r = requests.get(self.ollama_url + "/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            data = r.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            # 模型名可能带 :latest 后缀，做包含匹配
            return any(self.model.split(":")[0] in n for n in names)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # 模糊区间触发判定
    # ------------------------------------------------------------------ #
    def needs_vlm_check(self, confidence):
        """置信度落在 [trigger_low, trigger_high] 模糊区间时触发 VLM 终审。"""
        return bool(self.trigger_low <= confidence <= self.trigger_high)

    # ------------------------------------------------------------------ #
    # 主校验
    # ------------------------------------------------------------------ #
    def validate(self, image_crop, context=None):
        """将图片 crop 编码为 base64，调用 VLM 终审。

        :param image_crop: BGR/灰度 numpy 图像（缺陷 ROI）
        :param context: 可选 dict，携带初筛特征（fallback 用）
        :return: dict{label: "缺陷"|"干扰", confidence: float(0-1), reason: str}
        """
        if image_crop is None or image_crop.size == 0:
            return dict(label="缺陷", confidence=0.5,
                        reason="空图像，默认按真实缺陷保留")

        # 缩放到合理尺寸，降低推理时延（VLM 对大图很慢）
        crop = self._resize_for_vlm(image_crop)
        b64 = self._encode_image_b64(crop)

        if self.is_available():
            try:
                return self._call_ollama(b64)
            except Exception as e:  # 超时 / 解析失败等 -> fallback
                return self._fallback_judge(context, note="vlm_error:" + str(e)[:60])
        # 服务不可用 -> 特征 fallback
        return self._fallback_judge(context, note="ollama_unavailable")

    # ------------------------------------------------------------------ #
    # Ollama HTTP 调用
    # ------------------------------------------------------------------ #
    def _call_ollama(self, b64):
        """POST /api/chat 调用 qwen2.5vl，解析返回 JSON。"""
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content":
                 "请判断这张工业盖板图片中的暗斑是真实缺陷还是干扰项。",
                 "images": [b64]},
            ],
            "options": {"temperature": 0.1, "num_predict": 64},
        }
        r = requests.post(self.ollama_url + "/api/chat", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return self._parse_vlm_json(content)

    def _parse_vlm_json(self, text):
        """从 VLM 返回文本中提取 {label, confidence, reason}。"""
        # 抽取第一个 { ... } JSON 块
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                label = str(obj.get("label", "缺陷"))
                conf = float(obj.get("confidence", 0.5))
                reason = str(obj.get("reason", ""))
                return dict(label=self._normalize_label(label),
                            confidence=float(np.clip(conf, 0.0, 1.0)),
                            reason=reason or text[:80])
            except Exception:
                pass
        # 退化：关键字判断
        if "干扰" in text:
            return dict(label="干扰", confidence=0.7, reason=text[:80])
        return dict(label="缺陷", confidence=0.6, reason=text[:80])

    @staticmethod
    def _normalize_label(label):
        if "干" in label or "粉尘" in label or "水渍" in label:
            return "干扰"
        return "缺陷"

    # ------------------------------------------------------------------ #
    # Fallback：基于图像几何特征判定
    # ------------------------------------------------------------------ #
    def _fallback_judge(self, context, note=""):
        """Ollama 不可用时的兜底：高圆度 -> 干扰，否则 -> 缺陷。"""
        ctx = context or {}
        roundness = ctx.get("roundness", 0.5)
        var_gray = ctx.get("var_gray", 0.5)
        # 高圆度 + 低灰度方差 => 像粉尘干扰
        if roundness >= 0.8 and var_gray <= 0.15:
            return dict(label="干扰", confidence=roundness,
                        reason="[fallback]高圆度低方差判定为粉尘 " + note)
        return dict(label="缺陷", confidence=1.0 - roundness,
                    reason="[fallback]形状不规则判定为真实缺陷 " + note)

    # ------------------------------------------------------------------ #
    # 图像编码
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_for_vlm(image_crop, max_side=224):
        """长边缩放到 max_side，控制 VLM 推理时延（越小越快）。"""
        if image_crop.ndim == 2:
            img = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2BGR)
        else:
            img = image_crop
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _encode_image_b64(bgr_img):
        """BGR 图像 -> base64 字符串（jpeg）。"""
        ok, buf = cv2.imencode(".jpg", bgr_img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("utf-8")
