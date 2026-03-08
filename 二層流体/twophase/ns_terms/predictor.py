"""
⑥ Predictor: (a)+(b)+(c)+(d) → u*

論文 §7.1, 式(57):
  (u* - u^n)/Δt = (a) + (b) + (c) + (d)

ここで:
  (a) = -(u^n·∇)u^n           対流（CCD, TVD-RK3 陽的）
  (b) = ∇·[μ̃(∇u)^sym]/(ρ̃Re) 粘性（CCD, Crank-Nicolson 半陰的）
  (c) = -ẑ/Fr²                重力（定数）
  (d) = κδε∇φ/(ρ̃We)          表面張力（半陰的）

Crank-Nicolson 形式（式82）:
  u*/Δt - V(u*)/2ρ̃Re = u^n/Δt + N(u^n) + V(u^n)/2ρ̃Re
  N = (a) + (c) + (d)

現在の実装:
  - 陽的版: u* = u^n + Δt[(a) + (b) + (c) + (d)]
  - 半陰的版(CN): 粘性項のみ半陰的処理（将来拡張用フックあり）
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .convection import ConvectionTerm
from .viscous import ViscousTerm
from .gravity import GravityTerm
from .surface_tension import SurfaceTensionTerm

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField
    from ..backend import Backend
    from ..config import SimulationConfig


class Predictor:
    """
    ⑥ Predictor（NS全項の統合）

    NS方程式の右辺を計算し、u* を生成する。
    コードを読むだけで「どのNS項がどの順で計算されるか」が分かる構造。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self.config = config

        # ─── NS方程式の各項（論文ラベル付き）───
        self.term_a = ConvectionTerm(grid, backend, config)      # (a) 対流
        self.term_b = ViscousTerm(grid, backend, config)         # (b) 粘性
        self.term_c = GravityTerm(grid, backend, config)         # (c) 重力
        self.term_d = SurfaceTensionTerm(grid, backend, config)  # (d) 表面張力

        # RHS 作業バッファ
        self._rhs = [self.xp.zeros(grid.shape) for _ in range(grid.ndim)]

    def compute(self, state: dict, vel_star: VectorField, dt: float):
        """
        u* を計算して vel_star に書き込む

        式: u* = u^n + Δt × [(a) + (b) + (c) + (d)]

        Args:
            state: 場変数辞書
            vel_star: 出力先（in-place書き込み）
            dt: 時間刻み幅
        """
        xp = self.xp
        vel = state['velocity']

        # ── 陽的項をゼロから積算 ──

        # (a) 対流項: -(u^n·∇)u^n（陽的、t^n データ使用）
        self.term_a.evaluate(state, vel_star, mode='set')

        # (c) 重力項: -ẑ/Fr²（定数、ρ̃除算済み）
        self.term_c.evaluate(state, vel_star, mode='add')

        # (d) 表面張力: κδε∇φ/(ρ̃We)（半陰的、t^{n+1} データ）
        #     ※ φ, κ, ρ は Phase 2 で更新済みの t^{n+1} 値を使用
        self.term_d.evaluate(state, vel_star, mode='add')

        # (b) 粘性項: ∇·[μ̃(∇u+∇u^T)]/(ρ̃Re)
        self.term_b.evaluate(state, vel_star, mode='add')

        # ── 時間積分: u* = u^n + Δt × RHS ──
        for k in range(self.ndim):
            vel_star[k].data *= dt
            vel_star[k].data += vel[k].data
            vel_star[k].invalidate()

    def compute_explicit(self, state: dict, vel_star: VectorField, dt: float):
        """
        compute() のエイリアス（全項を陽的に処理する版）
        """
        self.compute(state, vel_star, dt)

    def compute_rhs_only(self, state: dict, out: VectorField):
        """
        RHS のみを計算（時間積分なし）

        u* ではなく dU/dt = (a)+(b)+(c)+(d) を返す。
        テストやCN分離に使用。
        """
        self.term_a.evaluate(state, out, mode='set')
        self.term_c.evaluate(state, out, mode='add')
        self.term_d.evaluate(state, out, mode='add')
        self.term_b.evaluate(state, out, mode='add')
