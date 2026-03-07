"""
(d) 表面張力項（CSF法）: κ δε(φ) ∇φ / (ρ̃·We)

論文 図1 ラベル (d), §2.3:
  Brackbill et al. (1992) の CSF 法で界面の表面張力を体積力として評価。

  κ^{n+1}   : 曲率（Phase 2 ⑤で計算済み）
  δε(φ^{n+1}): 滑らか化デルタ関数（界面近傍のみ非ゼロ）
  ∇φ^{n+1}  : Level Set 勾配（CCD O(h^6)）
  ρ̃^{n+1}   : 密度（Phase 2 ④で更新済み）

  半陰的: t^{n+1} の φ, ρ, κ を使用
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import NSTerm
from ..levelset.heaviside import delta_smooth

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField
    from ..backend import Backend
    from ..config import SimulationConfig


class SurfaceTensionTerm(NSTerm):
    """
    (d) 表面張力項（CSF法）

    界面近傍でのみ非ゼロとなる体積力。
    曲率 κ と δε の積が界面幅 2ε 内に集中するため、
    遠方での不要な力は自動的にゼロになる。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        super().__init__(grid, backend, config)
        self._delta_buf = self.xp.zeros(grid.shape)

    def evaluate(self, state: dict, out: VectorField, mode: str = 'add'):
        """
        CSF 表面張力を計算して out に書き込む

        state に必要なキー:
          'phi': ScalarField (φ^{n+1}, 微分値キャッシュ済み)
          'kappa': ScalarField (κ^{n+1})
          'rho': ScalarField (ρ̃^{n+1})
          'epsilon': float (界面幅パラメータ)
          'ccd': CCDSolver
          'config': SimulationConfig
        """
        xp = self.xp
        phi = state['phi']
        kappa = state['kappa']
        rho = state['rho']
        We = state['config'].We
        epsilon = state['epsilon']
        ccd = state['ccd']

        # δε(φ) を計算
        delta = self._delta_buf
        delta[:] = delta_smooth(phi.data, epsilon, xp)

        # φ の勾配を確保（Phase 2 で計算済みのはず）
        phi.ensure_derivatives(ccd)

        # 共通係数: κ · δε / (ρ̃ · We)
        # ゼロ割防止: rho が 0 にならないよう保護
        coeff = kappa.data * delta / (xp.maximum(rho.data, 1e-14) * We)

        for k in range(self.ndim):
            # surf_k = coeff · ∂φ/∂x_k
            dphi_k = phi.d1[k]
            force = coeff * dphi_k

            if mode == 'set':
                out[k].data[:] = force
            else:
                out[k].data += force
            out[k].invalidate()
