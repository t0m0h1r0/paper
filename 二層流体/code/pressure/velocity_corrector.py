"""
(f) 速度補正: u^{n+1} = u* - (Δt/ρ̃) ∇p^{n+1}

論文 §7.1, 式(59):
  ∇p は CCD O(h^6) で計算。
  ρ̃ で除算して非圧縮条件 ∇·u^{n+1} = 0 を保証。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField, VectorField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend


class VelocityCorrector:
    """
    (f) 速度補正

    u^{n+1}_k = u*_k - (Δt / ρ̃) · (∂p/∂x_k)

    圧力勾配は CCD O(h^6) で計算。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

    def correct(self, vel_star: VectorField, vel_new: VectorField,
                p: ScalarField, rho: ScalarField,
                ccd: CCDSolver, dt: float):
        """
        速度補正を実行

        Args:
            vel_star: 予測速度 u*
            vel_new: 補正後速度 u^{n+1}（出力先）
            p: 圧力場 p^{n+1}
            rho: 密度場 ρ̃^{n+1}
            ccd: CCD ソルバー
            dt: 時間刻み幅
        """
        xp = self.xp

        # CCD で ∇p を計算
        p.ensure_derivatives(ccd)

        coeff = dt / xp.maximum(rho.data, 1e-14)

        for k in range(self.ndim):
            # u^{n+1}_k = u*_k - (Δt/ρ̃) · ∂p/∂x_k
            vel_new[k].data[:] = vel_star[k].data - coeff * p.d1[k]
            vel_new[k].invalidate()
