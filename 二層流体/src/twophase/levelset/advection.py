"""
① Level Set 移流

論文 §3.4, 式(16):
  ∂φ/∂t + u·∇φ = 0

空間離散化: CCD O(h^6) で ∇φ を計算
時間積分:   TVD-RK3（3次精度、TVD安定）

L(φ) = -u·∇φ = -Σ_ax u_ax · (∂φ/∂x_ax)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..time_integration.tvd_rk3 import TVDRK3

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField, VectorField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend


class LevelSetAdvection:
    """
    Level Set 移流演算子

    TVD-RK3 + CCD で φ を移流する。
    advance() で φ.data を in-place 更新。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

        # TVD-RK3 積分器
        self._rk3 = TVDRK3(self.xp, grid.shape)

        # 作業バッファ
        self._advection_buf = self.xp.zeros(grid.shape)

    def advance(self, phi: ScalarField, vel: VectorField,
                dt: float, ccd: CCDSolver):
        """
        1タイムステップの Level Set 移流

        ∂φ/∂t = -u·∇φ を TVD-RK3 で解く

        Args:
            phi: Level Set 場（in-place 更新される）
            vel: 速度場 (u, v, [w])
            dt:  時間刻み幅
            ccd: CCD ソルバー
        """
        # 速度データを保持（RK3 の各段で使い回す）
        vel_data = [vel[ax].data for ax in range(self.ndim)]

        def rhs_func(phi_data):
            """
            L(φ) = -Σ_ax u_ax · (∂φ/∂x_ax)

            CCD で ∂φ/∂x_ax を計算し、速度で掛けて足し合わせ
            """
            result = self._advection_buf
            result[:] = 0.0

            for ax in range(self.ndim):
                dphi_dax, _ = ccd.differentiate(phi_data, ax)
                result -= vel_data[ax] * dphi_dax

            return result

        # TVD-RK3 で phi.data を in-place 更新
        self._rk3.advance(phi.data, dt, rhs_func)
        phi.invalidate()
