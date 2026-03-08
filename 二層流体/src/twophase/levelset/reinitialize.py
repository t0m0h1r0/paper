"""
② 再初期化

論文 §3.5, 式(17):
  ∂φ/∂τ + sgn(φ₀)(|∇φ|^uw - 1) = 0

仮想時間 τ 方向に 3〜5 反復解くことで |∇φ| = 1 を回復する。

重要:
  - Godunov 上流勾配（φ₀ 符号依存）を使用（式81）
  - 反復回数を増やすと界面位置が微動する副作用がある
  - 仮想時間刻み dτ ≈ 0.5 Δx_min が典型値
  - φ₀ = 0 の点では s=0 なので更新不要
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .godunov import GodunovGradient

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField
    from ..ccd.ccd_solver import CCDSolver
    from ..backend import Backend
    from ..config import SimulationConfig


class Reinitializer:
    """
    Level Set 再初期化演算子

    reinitialize() で φ.data を in-place 更新し、
    |∇φ| ≈ 1 を回復する。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self.reinit_steps = config.reinit_steps

        # Godunov 上流勾配計算器
        self._godunov = GodunovGradient(grid, backend)

        # 作業バッファ
        self._phi0 = self.xp.zeros(grid.shape)      # φ₀ の保存
        self._grad_uw = self.xp.zeros(grid.shape)    # |∇φ|^uw

    def reinitialize(self, phi: ScalarField, ccd: CCDSolver,
                     dtau: float = None):
        """
        再初期化を実行

        ∂φ/∂τ = -sgn(φ₀)(|∇φ|^uw - 1)

        前進オイラー法で仮想時間積分（3〜5反復）

        Args:
            phi: Level Set 場（in-place 更新される）
            ccd: CCD ソルバー（Godunov勾配では不使用だが将来の拡張用）
            dtau: 仮想時間刻み。None なら 0.5*dx_min を使用
        """
        xp = self.xp

        if dtau is None:
            dtau = 0.5 * self.grid.dx_min

        # φ₀ = 再初期化前の φ を保存
        xp.copyto(self._phi0, phi.data)

        # sgn(φ₀) — 数値安定のため滑らか化
        s = self._smoothed_sign(self._phi0, self.grid.dx_min)

        for _ in range(self.reinit_steps):
            # Godunov 上流勾配 |∇φ|^uw
            self._godunov.compute(phi.data, self._phi0, self._grad_uw)

            # ∂φ/∂τ = -s (|∇φ|^uw - 1)
            # 前進オイラー: φ ← φ - dτ · s · (|∇φ|^uw - 1)
            phi.data -= dtau * s * (self._grad_uw - 1.0)

        phi.invalidate()

    def _smoothed_sign(self, phi, dx):
        """
        数値安定化した符号関数

        sgn_ε(φ) = φ / √(φ² + dx²)

        界面(φ=0)近傍で滑らかに遷移する。
        """
        xp = self.xp
        return phi / xp.sqrt(phi ** 2 + dx ** 2)
