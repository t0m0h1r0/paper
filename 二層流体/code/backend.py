"""
GPU/CPU バックエンド抽象化レイヤー

CuPy が利用可能なら GPU、なければ NumPy にフォールバック。
全モジュールは backend.xp を通じて配列操作を行う。
"""

import importlib


class Backend:
    """配列ライブラリの切り替えを透過的に行うラッパー"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and self._cupy_available()

        if self.use_gpu:
            import cupy as cp
            self.xp = cp
            self.device = 'gpu'
        else:
            import numpy as np
            self.xp = np
            self.device = 'cpu'

    @staticmethod
    def _cupy_available() -> bool:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    def to_host(self, arr):
        """GPU配列 → NumPy 配列（CPU上なら何もしない）"""
        if self.use_gpu:
            return arr.get()
        return arr

    def to_device(self, arr):
        """NumPy 配列 → GPU 配列（CPU上なら何もしない）"""
        if self.use_gpu:
            import cupy as cp
            return cp.asarray(arr)
        return arr

    def __repr__(self):
        return f"Backend(device='{self.device}')"
