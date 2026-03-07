"""
場の変数クラス（2D/3D統一）

ScalarField: スカラー場 (φ, p, ρ, μ, κ)
VectorField: ベクトル場 (u, v, [w])

設計原則:
  - data は GPU 上に1回だけ確保し、以後 in-place 更新
  - CCD 微分値はキャッシュし、data 変更時に自動無効化
  - メモリコピーは copyto で行い、配列再確保しない
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .grid import Grid
    from ..backend import Backend


class ScalarField:
    """
    2D/3D 対応スカラー場

    Attributes:
        data:  GPU 上の配列 shape = grid.shape
        d1:    {axis: 1階微分配列} (CCD計算済みキャッシュ)
        d2:    {axis: 2階微分配列}
        d2_mixed: {(ax1,ax2): 混合偏微分配列}
    """

    def __init__(self, grid: Grid, backend: Backend, name: str = ""):
        self.grid = grid
        self.xp = backend.xp
        self.backend = backend
        self.name = name
        self.ndim = grid.ndim

        # メインデータ（GPU上、確保1回のみ）
        self.data = self.xp.zeros(grid.shape)

        # 微分値キャッシュ（遅延評価）
        self.d1: Dict[int, any] = {}               # ∂f/∂x_ax
        self.d2: Dict[int, any] = {}               # ∂²f/∂x_ax²
        self.d2_mixed: Dict[Tuple[int, int], any] = {}  # ∂²f/∂x_a∂x_b
        self._dirty = True

    def invalidate(self):
        """data が更新されたらキャッシュを無効化"""
        self._dirty = True
        self.d1.clear()
        self.d2.clear()
        self.d2_mixed.clear()

    def fill_from(self, other: ScalarField):
        """in-place コピー（配列再確保なし）"""
        self.xp.copyto(self.data, other.data)
        self.invalidate()

    def ensure_derivatives(self, ccd, axes=None):
        """
        指定軸の1階・2階微分を計算（未計算の場合のみ）

        Args:
            ccd: CCDSolver インスタンス
            axes: 計算する軸のリスト。None なら全軸
        """
        if axes is None:
            axes = list(range(self.ndim))
        for ax in axes:
            if ax not in self.d1:
                d1, d2 = ccd.differentiate(self.data, ax)
                self.d1[ax] = d1
                self.d2[ax] = d2

    def ensure_mixed_derivative(self, ccd, ax1: int, ax2: int):
        """
        混合偏微分 ∂²f/∂x_{ax1}∂x_{ax2} を計算（CCD逐次適用）

        手順:
          1. ax2 方向 CCD → ∂f/∂x_{ax2}
          2. その結果を ax1 方向 CCD → ∂²f/∂x_{ax1}∂x_{ax2}
        """
        key = (ax1, ax2)
        if key not in self.d2_mixed:
            # Step 1: ax2 方向の1階微分
            self.ensure_derivatives(ccd, axes=[ax2])
            df_dax2 = self.d1[ax2]

            # Step 2: df_dax2 を「関数値」として ax1 方向に CCD
            d_mixed, _ = ccd.differentiate(df_dax2, ax1)
            self.d2_mixed[key] = d_mixed

    def __repr__(self):
        return f"ScalarField(name='{self.name}', shape={self.data.shape})"


class VectorField:
    """
    2D/3D 対応ベクトル場

    2D: components = [u, v]     (2成分)
    3D: components = [u, v, w]  (3成分)
    """

    def __init__(self, grid: Grid, backend: Backend, name: str = ""):
        self.ndim = grid.ndim
        self.grid = grid
        self.backend = backend
        self.name = name
        self.components = [
            ScalarField(grid, backend, name=f"{name}_{i}")
            for i in range(grid.ndim)
        ]

    def __getitem__(self, idx: int) -> ScalarField:
        return self.components[idx]

    @property
    def u(self) -> ScalarField:
        return self.components[0]

    @property
    def v(self) -> ScalarField:
        return self.components[1]

    @property
    def w(self) -> ScalarField:
        """3D のみ。2D で呼ぶと IndexError"""
        return self.components[2]

    def invalidate(self):
        for c in self.components:
            c.invalidate()

    def __repr__(self):
        return f"VectorField(name='{self.name}', ndim={self.ndim})"
