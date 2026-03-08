"""
CFL 安定条件（3項の最小値）

論文 §8.4, 式(84):
  Δt ≤ C_CFL · min(
    h_min / |u|_max,          対流 CFL
    ρ_min h_min² / μ_max,     粘性 CFL
    √((ρl+ρg) h³ / (4πσ))    毛管波 CFL
  )
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SimulationConfig


class CFLCalculator:
    """CFL 条件に基づく適応的タイムステップ計算"""

    def __init__(self, config: SimulationConfig):
        self.cfl = config.cfl_number
        self.Re = config.Re
        self.We = config.We
        self.rho_ratio = config.rho_ratio

    def compute(self, velocity, rho, mu, grid, xp) -> float:
        """
        CFL 条件から Δt を計算

        Returns:
            dt: 時間刻み幅
        """
        h_min = grid.dx_min

        # 対流 CFL: h / |u|_max
        u_max = 0.0
        for k in range(grid.ndim):
            u_max = max(u_max, float(xp.max(xp.abs(velocity[k].data))))
        if u_max > 1e-14:
            dt_conv = h_min / u_max
        else:
            dt_conv = 1e10

        # 粘性 CFL: ρ_min h² / μ_max
        rho_min = float(xp.min(rho.data))
        mu_max = float(xp.max(mu.data))
        rho_min = max(rho_min, 1e-14)
        mu_max = max(mu_max, 1e-14)
        dt_visc = rho_min * h_min ** 2 / mu_max

        # 毛管波 CFL: √((ρl+ρg)h³/(4πσ))
        # σ = ρl·U²·L/We (非次元化), ρl=1, ρg=rho_ratio
        # → We が大きいほど σ が小さい → dt_cap が大きい
        if self.We < 1e12:
            dt_cap = math.sqrt(
                (1.0 + self.rho_ratio) * h_min ** 3
                * self.We / (4.0 * math.pi))
        else:
            dt_cap = 1e10

        dt = self.cfl * min(dt_conv, dt_visc, dt_cap)
        return dt
