"""
⑨ 診断モニター

論文 §10, 表4:
  - 非圧縮性: ||∇·u||_∞ < ε_div
  - 体積保存: ∫Hε dV = const
  - Eikonal品質: || |∇φ| - 1 ||_∞ < ε_eik
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField, VectorField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend


class DiagnosticMonitor:
    """
    ⑨ 収束確認・モニタリング
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self._initial_volume = None

    def check_divergence(self, velocity: VectorField, ccd: CCDSolver) -> float:
        """∇·u の最大値を返す"""
        xp = self.xp
        div = xp.zeros(self.grid.shape)
        for ax in range(self.ndim):
            velocity[ax].ensure_derivatives(ccd)
            div += velocity[ax].d1[ax]

        inner = tuple(slice(2, -2) for _ in range(self.ndim))
        return float(xp.max(xp.abs(div[inner])))

    def check_volume(self, phi: ScalarField, epsilon: float) -> float:
        """∫Hε dV を返す"""
        from ..levelset.heaviside import heaviside_smooth
        xp = self.xp
        H = heaviside_smooth(phi.data, epsilon, xp)
        cell_vol = 1.0
        for ax in range(self.ndim):
            cell_vol *= float(self.grid.h[ax])
        vol = float(xp.sum(H)) * cell_vol

        if self._initial_volume is None:
            self._initial_volume = vol
        return vol

    def volume_error(self, phi: ScalarField, epsilon: float) -> float:
        """体積の相対誤差 (%)"""
        vol = self.check_volume(phi, epsilon)
        if self._initial_volume is not None and self._initial_volume > 0:
            return abs(vol - self._initial_volume) / self._initial_volume * 100.0
        return 0.0

    def check_eikonal(self, phi: ScalarField, ccd: CCDSolver) -> float:
        """|| |∇φ| - 1 ||_∞ を返す"""
        xp = self.xp
        phi.ensure_derivatives(ccd)

        grad_sq = xp.zeros(self.grid.shape)
        for ax in range(self.ndim):
            grad_sq += phi.d1[ax] ** 2
        grad_mag = xp.sqrt(grad_sq)

        inner = tuple(slice(2, -2) for _ in range(self.ndim))
        return float(xp.max(xp.abs(grad_mag[inner] - 1.0)))

    def report(self, time, step, velocity, phi, ccd, epsilon):
        """1行の診断レポートを出力"""
        div = self.check_divergence(velocity, ccd)
        vol_err = self.volume_error(phi, epsilon)
        eik = self.check_eikonal(phi, ccd)
        print(f"  Step {step:>6d}  t={time:.4e}  "
              f"|∇·u|={div:.2e}  vol_err={vol_err:.4f}%  "
              f"|∇φ|-1={eik:.4f}")
