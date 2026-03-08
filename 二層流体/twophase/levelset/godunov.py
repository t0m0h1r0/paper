"""
Godunov 型上流勾配（φ₀ 符号依存）

論文 §8.3, 式(81):
  s = sgn(φ₀)
  G_α² = max(s·D⁻_α, 0)² + min(s·D⁺_α, 0)²   (α = x,y,z)
  |∇φ|^uw = √(Σ_α G_α²)

重要:
  s の符号で上流方向が反転する。
  φ₀ > 0（液相）: D⁻ が上流 → max(D⁻,0)² + min(D⁺,0)²
  φ₀ < 0（気相）: D⁺ が上流 → min(D⁻,0)² + max(D⁺,0)²

  旧式（s=+1 固定）は気相側で風下参照となり計算破綻を起こす。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..backend import Backend


class GodunovGradient:
    """
    再初期化用の Godunov 型上流勾配計算器（2D/3D対応）

    compute() で |∇φ|^uw を計算する。
    前進差分・後退差分は1次精度（1点ステンシル）。
    論文の記載通り、CCD高精度化は不要。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

        # 作業バッファ
        self._Dp = self.xp.zeros(grid.shape)   # D⁺
        self._Dm = self.xp.zeros(grid.shape)   # D⁻

    def compute(self, phi_data, phi0_data, out_data):
        """
        |∇φ|^uw を計算して out_data に書き込む

        Args:
            phi_data:  現在の φ 配列
            phi0_data: 再初期化前の φ₀ 配列（符号関数の基準）
            out_data:  出力先配列（in-place書き込み）
        """
        xp = self.xp

        # s = sgn(φ₀)（滑らか化は使用せず、厳密符号）
        s = xp.sign(phi0_data)

        grad_sq = xp.zeros_like(phi_data)

        for ax in range(self.ndim):
            h = float(self.grid.h[ax])

            # 前進差分 D⁺ = (φ_{i+1} - φ_i) / h
            Dp = self._forward_diff(phi_data, ax, h)
            # 後退差分 D⁻ = (φ_i - φ_{i-1}) / h
            Dm = self._backward_diff(phi_data, ax, h)

            # G_α² = max(s·D⁻, 0)² + min(s·D⁺, 0)²
            grad_sq += xp.maximum(s * Dm, 0.0) ** 2
            grad_sq += xp.minimum(s * Dp, 0.0) ** 2

        out_data[:] = xp.sqrt(grad_sq)

    def _forward_diff(self, phi, axis, h):
        """
        前進差分 D⁺_α = (φ_{i+1} - φ_i) / h

        境界: 最後の点は片側差分（1次外挿 or ゼロ勾配）
        """
        xp = self.xp
        Dp = xp.zeros_like(phi)
        n = phi.shape[axis]

        # スライス構築
        sl_ip1 = [slice(None)] * phi.ndim
        sl_i = [slice(None)] * phi.ndim

        sl_ip1[axis] = slice(1, n)
        sl_i[axis] = slice(0, n - 1)

        Dp[tuple(sl_i)] = (phi[tuple(sl_ip1)] - phi[tuple(sl_i)]) / h

        # 最後の点: D⁺ = 0（ノイマン的）
        sl_last = [slice(None)] * phi.ndim
        sl_last[axis] = -1
        Dp[tuple(sl_last)] = 0.0

        return Dp

    def _backward_diff(self, phi, axis, h):
        """
        後退差分 D⁻_α = (φ_i - φ_{i-1}) / h

        境界: 最初の点は D⁻ = 0
        """
        xp = self.xp
        Dm = xp.zeros_like(phi)
        n = phi.shape[axis]

        sl_i = [slice(None)] * phi.ndim
        sl_im1 = [slice(None)] * phi.ndim

        sl_i[axis] = slice(1, n)
        sl_im1[axis] = slice(0, n - 1)

        Dm[tuple(sl_i)] = (phi[tuple(sl_i)] - phi[tuple(sl_im1)]) / h

        # 最初の点: D⁻ = 0
        sl_first = [slice(None)] * phi.ndim
        sl_first[axis] = 0
        Dm[tuple(sl_first)] = 0.0

        return Dm
