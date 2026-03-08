"""
⑤ 界面曲率計算（CCD 逐次適用、2D/3D 対応）

論文 §2.4, 式(8):

  2D:
    κ = -(φx²·φyy - 2φxφy·φxy + φy²·φxx) / (φx² + φy²)^{3/2}
    必要微分: φx, φy, φxx, φyy, φxy

  3D:
    κ = -∇·(∇φ/|∇φ|) を展開:
    κ = -[(φy²+φz²)φxx + (φx²+φz²)φyy + (φx²+φy²)φzz
          - 2(φxφyφxy + φxφzφxz + φyφzφyz)]
        / (φx² + φy² + φz²)^{3/2}
    必要微分: φx,φy,φz, φxx,φyy,φzz, φxy,φxz,φyz

全微分値は CCD O(h^6) で統一的に評価。
混合偏微分は CCD 逐次適用で計算。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend


class CurvatureCalculator:
    """
    界面曲率 κ の計算器（2D/3D対応）

    compute() で φ の全微分値を CCD で計算し、
    曲率公式を適用して kappa.data に書き込む。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

    def compute(self, phi: ScalarField, kappa: ScalarField,
                ccd: CCDSolver):
        """
        曲率 κ を計算して kappa.data に書き込む

        Args:
            phi:   Level Set 場（微分値がキャッシュされる）
            kappa: 出力先の曲率場
            ccd:   CCD ソルバー
        """
        # 全軸の1階・2階微分を計算
        phi.ensure_derivatives(ccd)

        # 混合偏微分
        phi.ensure_mixed_derivative(ccd, 0, 1)  # φxy
        if self.ndim == 3:
            phi.ensure_mixed_derivative(ccd, 0, 2)  # φxz
            phi.ensure_mixed_derivative(ccd, 1, 2)  # φyz

        if self.ndim == 2:
            self._compute_2d(phi, kappa)
        else:
            self._compute_3d(phi, kappa)

        kappa.invalidate()

    def _compute_2d(self, phi: ScalarField, kappa: ScalarField):
        """
        2D 曲率（式8）:
          κ = -(φx²·φyy - 2φxφy·φxy + φy²·φxx) / (φx² + φy²)^{3/2}
        """
        xp = self.xp

        px = phi.d1[0]    # ∂φ/∂x
        py = phi.d1[1]    # ∂φ/∂y
        pxx = phi.d2[0]   # ∂²φ/∂x²
        pyy = phi.d2[1]   # ∂²φ/∂y²
        pxy = phi.d2_mixed[(0, 1)]  # compute() で事前計算済み

        num = px ** 2 * pyy - 2.0 * px * py * pxy + py ** 2 * pxx
        den = (px ** 2 + py ** 2) ** 1.5
        den = xp.maximum(den, 1e-12)  # ゼロ割防止

        kappa.data[:] = -num / den

    def _compute_3d(self, phi: ScalarField, kappa: ScalarField):
        """
        3D 曲率:
          κ = -[(φy²+φz²)φxx + (φx²+φz²)φyy + (φx²+φy²)φzz
                - 2(φxφyφxy + φxφzφxz + φyφzφyz)]
              / (φx² + φy² + φz²)^{3/2}
        """
        xp = self.xp

        px, py, pz = phi.d1[0], phi.d1[1], phi.d1[2]
        pxx, pyy, pzz = phi.d2[0], phi.d2[1], phi.d2[2]
        pxy = phi.d2_mixed[(0, 1)]
        pxz = phi.d2_mixed[(0, 2)]
        pyz = phi.d2_mixed[(1, 2)]

        num = ((py ** 2 + pz ** 2) * pxx
               + (px ** 2 + pz ** 2) * pyy
               + (px ** 2 + py ** 2) * pzz
               - 2.0 * (px * py * pxy + px * pz * pxz + py * pz * pyz))

        den = (px ** 2 + py ** 2 + pz ** 2) ** 1.5
        den = xp.maximum(den, 1e-12)

        kappa.data[:] = -num / den
