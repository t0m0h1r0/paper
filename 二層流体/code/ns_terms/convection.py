"""
(a) 対流項: -(u·∇)u

論文 図1 ラベル (a):
  離散化: CCD O(h^6) で ∂u_k/∂x_ax を評価
  時間積分: TVD-RK3 陽的（t^n のデータを使用）

  2D: conv_u = u·∂u/∂x + v·∂u/∂y
      conv_v = u·∂v/∂x + v·∂v/∂y

  3D: conv_u = u·∂u/∂x + v·∂u/∂y + w·∂u/∂z
      （v, w 成分も同様）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import NSTerm

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField
    from ..backend import Backend
    from ..config import SimulationConfig


class ConvectionTerm(NSTerm):
    """
    (a) 対流項 -(u·∇)u

    各速度成分 u_k について:
      result_k = -Σ_ax u_ax · (∂u_k/∂x_ax)
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        super().__init__(grid, backend, config)
        self._buf = self.xp.zeros(grid.shape)

    def evaluate(self, state: dict, out: VectorField, mode: str = 'add'):
        """
        対流項を計算して out に書き込む

        state に必要なキー:
          'velocity': VectorField (u^n)
          'ccd': CCDSolver
        """
        xp = self.xp
        vel = state['velocity']
        ccd = state['ccd']
        buf = self._buf

        for k in range(self.ndim):
            # 速度成分 u_k の微分を計算
            vel[k].ensure_derivatives(ccd)

            # conv_k = Σ_ax u_ax · ∂u_k/∂x_ax
            buf[:] = 0.0
            for ax in range(self.ndim):
                buf += vel[ax].data * vel[k].d1[ax]

            # -(u·∇)u: マイナス符号
            if mode == 'set' and k == 0:
                out[k].data[:] = -buf
            elif mode == 'set':
                out[k].data[:] = -buf
            else:
                out[k].data -= buf
            out[k].invalidate()
