"""
⑥ Predictor: (a)+(b)+(c)+(d) → u*

論文 §7.1, 式(57),(82):

【陽的モード】 (cn_viscous=False):
  u* = u^n + Δt × [(a) + (b) + (c) + (d)]

【Crank-Nicolson モード】 (cn_viscous=True):
  [I - (Δt/2)·σ·∇²] u* = u^n + Δt·N(u^n) + (Δt/2)·V_full(u^n)/(ρ̃Re)

  σ(x) = Δt·μ̃/(2ρ̃Re), N = (a)+(c)+(d), V_full = CCD完全粘性力

  IMEX分割: ラプラシアンσ∇²を陰的、∇μ交差項を陽的に処理。
  分割誤差 O(Δt²) → CN の2次精度を維持。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .convection import ConvectionTerm
from .viscous import ViscousTerm
from .gravity import GravityTerm
from .surface_tension import SurfaceTensionTerm
from ..core.field import VectorField

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..backend import Backend
    from ..config import SimulationConfig


class Predictor:
    """
    ⑥ Predictor（NS全項の統合）
    config.cn_viscous で陽的/CN半陰的を切り替え。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self.config = config
        self.cn_viscous = config.cn_viscous

        # NS方程式の各項
        self.term_a = ConvectionTerm(grid, backend, config)      # (a) 対流
        self.term_b = ViscousTerm(grid, backend, config)         # (b) 粘性
        self.term_c = GravityTerm(grid, backend, config)         # (c) 重力
        self.term_d = SurfaceTensionTerm(grid, backend, config)  # (d) 表面張力

        # CN用 Helmholtz ソルバー（遅延初期化）
        self._helmholtz = None
        self._visc_explicit = None
        if self.cn_viscous:
            self._init_cn(grid, backend, config)

    def _init_cn(self, grid, backend, config):
        """CN関連オブジェクトの初期化"""
        from .helmholtz import HelmholtzSolver
        self._helmholtz = HelmholtzSolver(grid, backend, config)
        self._visc_explicit = VectorField(grid, backend, "visc_explicit")

    def compute(self, state: dict, vel_star: VectorField, dt: float):
        """u*を計算。cn_viscousに応じて自動切替。"""
        if self.cn_viscous:
            self._compute_cn(state, vel_star, dt)
        else:
            self._compute_explicit(state, vel_star, dt)

    def _compute_explicit(self, state: dict, vel_star: VectorField, dt: float):
        """
        全項陽的: u* = u^n + Δt × [(a)+(b)+(c)+(d)]
        """
        vel = state['velocity']
        self.term_a.evaluate(state, vel_star, mode='set')
        self.term_c.evaluate(state, vel_star, mode='add')
        self.term_d.evaluate(state, vel_star, mode='add')
        self.term_b.evaluate(state, vel_star, mode='add')
        for k in range(self.ndim):
            vel_star[k].data *= dt
            vel_star[k].data += vel[k].data
            vel_star[k].invalidate()

    def _compute_cn(self, state: dict, vel_star: VectorField, dt: float):
        """
        Crank-Nicolson 半陰的 (式82):
          [I - (Δt/2)·σ·∇²] u*_k = u^n_k + Δt·N_k + (Δt/2)·V_k(u^n)

        Step 1: N = (a)+(c)+(d) — 非粘性の陽的項
        Step 2: V_full(u^n)/(ρ̃Re) — CCD完全粘性力
        Step 3: RHS = u^n + Δt·N + (Δt/2)·V_full
        Step 4: Helmholtz行列 [I - σ∇²] 構築
        Step 5: 各成分BiCGSTAB求解
        """
        xp = self.xp
        vel = state['velocity']

        # Step 1: 非粘性項 N = (a)+(c)+(d) → vel_star に一時格納
        self.term_a.evaluate(state, vel_star, mode='set')
        self.term_c.evaluate(state, vel_star, mode='add')
        self.term_d.evaluate(state, vel_star, mode='add')

        # Step 2: V_full(u^n) — 完全変粘性CCD評価
        self.term_b.evaluate(state, self._visc_explicit, mode='set')

        # Step 3: RHS組み立て
        half_dt = 0.5 * dt
        for k in range(self.ndim):
            vel_star[k].data *= dt             # Δt·N_k
            vel_star[k].data += vel[k].data    # + u^n_k
            vel_star[k].data += half_dt * self._visc_explicit[k].data  # + (Δt/2)·V_k
            vel_star[k].invalidate()

        # Step 4: Helmholtz行列構築 [I - σ∇²], σ=dt·μ/(2ρRe)
        self._helmholtz.update_matrix(state['mu'], state['rho'], dt)

        # Step 5: 各速度成分について求解
        for k in range(self.ndim):
            self._helmholtz.solve_component(vel_star[k])

    # 互換インターフェース
    def compute_explicit(self, state, vel_star, dt):
        self._compute_explicit(state, vel_star, dt)

    def compute_rhs_only(self, state, out):
        """RHSのみ計算（テスト用）"""
        self.term_a.evaluate(state, out, mode='set')
        self.term_c.evaluate(state, out, mode='add')
        self.term_d.evaluate(state, out, mode='add')
        self.term_b.evaluate(state, out, mode='add')
