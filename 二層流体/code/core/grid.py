"""
コロケート格子（2D/3D統一）

各軸の座標・メトリクス係数を独立に管理。
初期状態は等間隔格子。界面適合格子への更新は後のフェーズで実装。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend import Backend
    from ..config import SimulationConfig


class Grid:
    """
    2D/3D コロケート格子

    Attributes:
        ndim: 空間次元数 (2 or 3)
        N: 各軸のセル数タプル (Nx, Ny) or (Nx, Ny, Nz)
        L: 各軸の領域長タプル
        coords[ax]: 軸 ax の物理座標 (1D配列, 長さ N[ax]+1)
        h[ax]: 軸 ax の格子間隔 (等間隔時はスカラー)
        J[ax]: メトリクス dξ/dx (1D配列)
        dJ_dxi[ax]: ∂J/∂ξ (1D配列)
    """

    def __init__(self, config: SimulationConfig, backend: Backend):
        xp = backend.xp
        self.ndim = config.ndim
        self.N = tuple(config.N)
        self.L = tuple(config.L)
        self.backend = backend

        self.axis_names = ('x', 'y', 'z')[:self.ndim]

        # 各軸の座標とメトリクス
        self.coords = {}
        self.h = {}
        self.J = {}
        self.dJ_dxi = {}

        for ax in range(self.ndim):
            n = self.N[ax]
            length = self.L[ax]
            self.coords[ax] = xp.linspace(0.0, length, n + 1)
            self.h[ax] = length / n
            self.J[ax] = xp.ones(n + 1)            # 等間隔ではJ=N/L
            self.dJ_dxi[ax] = xp.zeros(n + 1)       # 等間隔では0

    @property
    def shape(self):
        """場の配列形状: (Nx+1, Ny+1) or (Nx+1, Ny+1, Nz+1)"""
        return tuple(n + 1 for n in self.N)

    @property
    def dx_min(self) -> float:
        """全軸の最小格子幅"""
        xp = self.backend.xp
        vals = []
        for ax in range(self.ndim):
            dx = self.coords[ax][1:] - self.coords[ax][:-1]
            vals.append(float(xp.min(dx)))
        return min(vals)

    def meshgrid(self):
        """
        格子点座標のメッシュグリッドを返す。

        Returns:
            2D: (X, Y) 各 shape (Nx+1, Ny+1)
            3D: (X, Y, Z) 各 shape (Nx+1, Ny+1, Nz+1)
        """
        xp = self.backend.xp
        coord_arrays = [self.coords[ax] for ax in range(self.ndim)]
        return xp.meshgrid(*coord_arrays, indexing='ij')
