"""
シミュレーション全パラメータの一元管理

ndim=2 で2次元、ndim=3 で3次元。N, L のタプル長は ndim に一致させる。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SimulationConfig:
    # --- 次元 ---
    ndim: int = 2

    # --- 格子 ---
    N: Tuple[int, ...] = (128, 128)
    L: Tuple[float, ...] = (1.0, 1.0)

    # --- 無次元物性値 ---
    Re: float = 100.0
    Fr: float = 1.0
    We: float = 10.0
    rho_ratio: float = 0.001
    mu_ratio: float = 0.01

    # --- 界面パラメータ ---
    epsilon_factor: float = 1.5
    alpha_grid: float = 3.0
    reinit_steps: int = 4

    # --- 時間積分 ---
    cfl_number: float = 0.3
    t_end: float = 1.0
    cn_viscous: bool = False           # True: Crank-Nicolson半陰的, False: 全項陽的

    # --- ソルバー ---
    bicgstab_tol: float = 1e-10
    bicgstab_maxiter: int = 1000

    # --- 境界条件 ---
    bc_type: str = 'wall'

    # --- GPU ---
    use_gpu: bool = True

    def __post_init__(self):
        assert self.ndim in (2, 3), f"ndim must be 2 or 3, got {self.ndim}"
        assert len(self.N) == self.ndim, f"len(N)={len(self.N)} != ndim={self.ndim}"
        assert len(self.L) == self.ndim, f"len(L)={len(self.L)} != ndim={self.ndim}"
