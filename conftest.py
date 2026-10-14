# -*- coding: utf-8 -*-
"""pytest 共享夹具：确保项目根目录在 sys.path 中。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
