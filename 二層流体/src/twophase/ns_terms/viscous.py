"""
(b) 粘性項: ∇·[μ̃(∇u + ∇u^T)] / (ρ̃·Re)

論文 図1 ラベル (b):
  対称応力テンソル τ = μ̃(∇u + ∇u^T) の発散を評価
  離散化: CCD O(h^6)
  時間積分: Crank-Nicolson 半陰的

粘性項の展開（変粘性）:
  ∇·[μ̃(∇u + ∇u^T)] の k成分 =
    Σ_ax ∂/∂x_ax [ μ̃ (∂u_k/∂x_ax + ∂u_ax/∂x_k) ]

  積の微分則で展開:
    = Σ_ax [ (∂μ̃/∂x_ax)(∂u_k/∂x_ax + ∂u_ax/∂x_k)
             + μ̃(∂²u_k/∂x_ax² + ∂²u_ax/∂x_k∂x_ax) ]

  等粘性の場合（μ̃=const）:
    = μ̃ [∇²u_k + ∂/∂x_k(∇·u)]
    非圧縮 ∇·u=0 により = μ̃ ∇²u_k（ラプラシアン）

本実装:
  - evaluate(): 粘性力 V(u)/ρ̃Re を計算
  - evaluate_laplacian(): 等粘性ラプラシアン μ̃∇²u/ρ̃Re（CN用の簡略版）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import NSTerm

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField, ScalarField
    from ..backend import Backend
    from ..config import SimulationConfig


class ViscousTerm(NSTerm):
    """
    (b) 粘性項

    完全変粘性版と等粘性ラプラシアン版の両方を提供。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        super().__init__(grid, backend, config)
        self._buf = self.xp.zeros(grid.shape)
        self._buf2 = self.xp.zeros(grid.shape)

    def evaluate(self, state: dict, out: VectorField, mode: str = 'add'):
        """
        変粘性の完全粘性力を計算: ∇·[μ̃(∇u+∇u^T)] / (ρ̃·Re)

        state に必要なキー:
          'velocity': VectorField (u)
          'rho': ScalarField (ρ̃)
          'mu': ScalarField (μ̃)
          'ccd': CCDSolver
          'config': SimulationConfig
        """
        xp = self.xp
        vel = state['velocity']
        rho = state['rho']
        mu = state['mu']
        ccd = state['ccd']
        Re = state['config'].Re
        buf = self._buf
        buf2 = self._buf2

        # μ̃ の勾配を事前計算
        mu.ensure_derivatives(ccd)

        # 各速度成分の微分を事前計算
        for k in range(self.ndim):
            vel[k].ensure_derivatives(ccd)

        inv_rho_Re = 1.0 / (rho.data * Re)

        for k in range(self.ndim):
            # visc_k = Σ_ax ∂/∂x_ax [ μ̃(∂u_k/∂x_ax + ∂u_ax/∂x_k) ]
            buf[:] = 0.0

            for ax in range(self.ndim):
                # S_k_ax = ∂u_k/∂x_ax + ∂u_ax/∂x_k  （対称歪み速度の成分）
                du_k_dax = vel[k].d1[ax]     # ∂u_k/∂x_ax

                if ax == k:
                    # 対角成分: S_kk = 2 ∂u_k/∂x_k
                    S = 2.0 * du_k_dax
                    # ∂S/∂x_ax = 2 ∂²u_k/∂x_k²
                    dS_dax = 2.0 * vel[k].d2[k]
                else:
                    # 非対角成分: S_k_ax = ∂u_k/∂x_ax + ∂u_ax/∂x_k
                    du_ax_dk = vel[ax].d1[k]  # ∂u_ax/∂x_k
                    S = du_k_dax + du_ax_dk

                    # ∂S/∂x_ax = ∂²u_k/∂x_ax² + ∂²u_ax/(∂x_k ∂x_ax)
                    d2u_k_dax2 = vel[k].d2[ax]
                    # 混合偏微分
                    vel[ax].ensure_mixed_derivative(ccd, ax, k)
                    d2u_ax_dax_dk = vel[ax].d2_mixed[(ax, k)]
                    dS_dax = d2u_k_dax2 + d2u_ax_dax_dk

                # 積の微分: ∂(μ̃·S)/∂x_ax = (∂μ̃/∂x_ax)·S + μ̃·(∂S/∂x_ax)
                dmu_dax = mu.d1[ax]
                buf += dmu_dax * S + mu.data * dS_dax

            # 粘性力/ρ̃Re
            buf *= inv_rho_Re

            if mode == 'set':
                out[k].data[:] = buf
            else:
                out[k].data += buf
            out[k].invalidate()

    def evaluate_laplacian(self, state: dict, out: VectorField,
                           mode: str = 'add'):
        """
        等粘性ラプラシアン版: μ̃·∇²u / (ρ̃·Re)

        非圧縮 ∇·u=0 を利用した簡略形。
        Crank-Nicolson の線形項に使用。
        """
        xp = self.xp
        vel = state['velocity']
        rho = state['rho']
        mu = state['mu']
        ccd = state['ccd']
        Re = state['config'].Re

        inv_rho_Re = 1.0 / (rho.data * Re)

        for k in range(self.ndim):
            vel[k].ensure_derivatives(ccd)

            # ∇²u_k = Σ_ax ∂²u_k/∂x_ax²
            lap = xp.zeros_like(vel[k].data)
            for ax in range(self.ndim):
                lap += vel[k].d2[ax]

            visc = mu.data * lap * inv_rho_Re

            if mode == 'set':
                out[k].data[:] = visc
            else:
                out[k].data += visc
            out[k].invalidate()
