"""
(c) 重力項: -ẑ/Fr²

論文 図1 ラベル (c):
  NS方程式を ρ̃ で除算済みなので ρ̃/ρ̃ = 1 となり単純な定数。
  2D: (0, -1/Fr²)          — y が鉛直方向
  3D: (0, 0, -1/Fr²)       — z が鉛直方向
  最後の軸方向のみ非ゼロ。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import NSTerm

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField
    from ..backend import Backend
    from ..config import SimulationConfig


class GravityTerm(NSTerm):
    """
    (c) 重力項

    ρ̃ で除算済みの定数ベクトル。
    evaluate() では最後の軸成分に -1/Fr² を加算するだけ。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        super().__init__(grid, backend, config)
        self.gravity_value = -1.0 / (config.Fr ** 2)
        self.gravity_axis = self.ndim - 1  # y(2D) or z(3D)

    def evaluate(self, state: dict, out: VectorField, mode: str = 'add'):
        """
        重力項を out に書き込む

        state: 使用しない（定数のため）
        """
        ax = self.gravity_axis
        if mode == 'set':
            for k in range(self.ndim):
                out[k].data[:] = 0.0
            out[ax].data[:] = self.gravity_value
        else:
            out[ax].data += self.gravity_value
        out[ax].invalidate()
