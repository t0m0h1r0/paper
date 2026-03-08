"""
(e_R) Rhie-Chow 補正速度と RC 発散

論文 §6.3, 式(63):
  u^RC_{i+1/2} = (u*_i + u*_{i+1})/2
    - (Δt/ρ̃_{i+1/2}) [ (p_{i+1}-p_i)/hx - (p^CCD_{x,i} + p^CCD_{x,i+1})/2 ]

角括弧内の構造:
  第1項: 2点直接差分 → チェッカーボード成分を「見える」
  第2項: CCD勾配の平均 → チェッカーボード成分が「見えない」
  差分がチェッカーボード高波数成分のみを選択的に除去する。

式(70) RC発散:
  div^RC = Σ_ax (u^RC_{ax,i+1/2} - u^RC_{ax,i-1/2}) / h_ax

2D: 4面（x±, y±）  |  3D: 6面（x±, y±, z±）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField, VectorField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend


class RhieChowCorrection:
    """
    Rhie-Chow 界面速度と RC 発散の計算器

    face_vel[ax]: 軸 ax 方向の界面速度 u^RC_{i+1/2}
                  格子点 i に i+1/2 面の値を格納
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

        # 各軸の界面速度バッファ
        self.face_vel = {}
        for ax in range(self.ndim):
            self.face_vel[ax] = self.xp.zeros(grid.shape)

        # RC発散バッファ
        self.div_rc = self.xp.zeros(grid.shape)

    def compute_face_velocities(self, vel_star: VectorField,
                                p: ScalarField, rho: ScalarField,
                                ccd: CCDSolver, dt: float):
        """
        全軸の Rhie-Chow 界面速度を計算

        Args:
            vel_star: 予測速度 u*
            p: 圧力場（前タイムステップ）
            rho: 密度場
            ccd: CCDSolver
            dt: 時間刻み幅
        """
        xp = self.xp

        # 圧力のCCD勾配を計算
        p.ensure_derivatives(ccd)

        for ax in range(self.ndim):
            h = float(self.grid.h[ax])
            n = self.grid.shape[ax]
            u_star_ax = vel_star[ax].data
            p_data = p.data
            rho_data = rho.data
            p_grad_ccd = p.d1[ax]  # CCD圧力勾配

            fv = self.face_vel[ax]
            fv[:] = 0.0

            # スライス構築: i と i+1
            sl_i = [slice(None)] * self.ndim
            sl_ip1 = [slice(None)] * self.ndim
            sl_out = [slice(None)] * self.ndim

            sl_i[ax] = slice(0, n - 1)
            sl_ip1[ax] = slice(1, n)
            sl_out[ax] = slice(0, n - 1)

            # ρ̃_{i+1/2} = (ρ_i + ρ_{i+1}) / 2
            rho_face = 0.5 * (rho_data[tuple(sl_i)] + rho_data[tuple(sl_ip1)])
            rho_face = xp.maximum(rho_face, 1e-14)

            # 直接2点差分: (p_{i+1} - p_i) / h
            dp_direct = (p_data[tuple(sl_ip1)] - p_data[tuple(sl_i)]) / h

            # CCD勾配の平均: (p^CCD_i + p^CCD_{i+1}) / 2
            dp_ccd_avg = 0.5 * (p_grad_ccd[tuple(sl_i)]
                                + p_grad_ccd[tuple(sl_ip1)])

            # Rhie-Chow 界面速度（式63）
            fv[tuple(sl_out)] = (
                0.5 * (u_star_ax[tuple(sl_i)] + u_star_ax[tuple(sl_ip1)])
                - (dt / rho_face) * (dp_direct - dp_ccd_avg)
            )

    def compute_divergence(self):
        """
        RC 発散を計算（式70）

        div^RC = Σ_ax (u^RC_{i+1/2} - u^RC_{i-1/2}) / h_ax
        """
        xp = self.xp
        self.div_rc[:] = 0.0

        for ax in range(self.ndim):
            h = float(self.grid.h[ax])
            n = self.grid.shape[ax]
            fv = self.face_vel[ax]

            # u^RC_{i+1/2} は fv[i] に格納
            # u^RC_{i-1/2} は fv[i-1]
            sl_ip = [slice(None)] * self.ndim
            sl_im = [slice(None)] * self.ndim
            sl_out = [slice(None)] * self.ndim

            sl_ip[ax] = slice(1, n - 1)   # fv[i] = u^RC_{i+1/2}
            sl_im[ax] = slice(0, n - 2)   # fv[i-1] = u^RC_{i-1/2}
            sl_out[ax] = slice(1, n - 1)

            self.div_rc[tuple(sl_out)] += (
                fv[tuple(sl_ip)] - fv[tuple(sl_im)]
            ) / h

        return self.div_rc
