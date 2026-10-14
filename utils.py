# -*- coding: utf-8 -*-
"""
公共工具模块
- 解决 Windows 下 OpenCV 无法直接读取含中文/非 ASCII 路径图片的问题
- 提供中文路径安全的 imread / imwrite / imencode
"""
import os
import cv2
import numpy as np


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """中文路径安全读图：np.fromfile 读字节 -> cv2.imdecode 解码。

    :param path: 图片绝对路径（可含中文）
    :param flags: cv2.IMREAD_* 标志
    :return: 解码后的 numpy 数组（BGR 顺序），失败返回 None
    """
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path, img, ext=".png"):
    """中文路径安全写图：cv2.imencode 编码 -> 写字节。

    :param path: 输出路径（可含中文）
    :param img: 要保存的图像数组
    :param ext: 编码后缀（决定编码格式）
    :return: 是否成功
    """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True
