"""
NS 方程式の項の基底クラス

論文 図1 のラベルに対応:
  (a) ConvectionTerm     : -(u·∇)u
  (b) ViscousTerm        : ∇·[μ̃(∇u+∇u^T)] / (ρ̃·Re)
  (c) GravityTerm        : -ẑ / Fr²
  (d) SurfaceTensionTerm : κδε∇φ / (ρ̃·We)

すべての具象クラスは evaluate() で寄与を VectorField に加算する。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import VectorField
    from ..backend import Backend
    from ..config import SimulationConfig


class NSTerm:
    """
    NS 方程式の1つの項を表す基底クラス

    設計原則:
      - __init__ で作業バッファを事前確保
      - evaluate() 内で新規メモリ確保しない
      - mode='add' で加算、mode='set' で上書き
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self.config = config

    def evaluate(self, state: dict, out: VectorField, mode: str = 'add'):
        """
        この項の寄与を out に書き込む

        Args:
            state: 現在の場変数を保持する辞書
                   {'velocity', 'phi', 'rho', 'mu', 'kappa', 'ccd', 'config', ...}
            out:   結果を書き込む VectorField
            mode:  'add' (加算) or 'set' (上書き)
        """
        raise NotImplementedError
