"""
3次 TVD Runge-Kutta 法（Shu-Osher スキーム）

論文 §8.1, 式(75)-(77):
  q^(1) = q^n + Δt L(q^n)
  q^(2) = (3/4)q^n + (1/4)[q^(1) + Δt L(q^(1))]
  q^{n+1} = (1/3)q^n + (2/3)[q^(2) + Δt L(q^(2))]

各段階が凸結合 → TVD安定性が保証される。

使い方:
  integrator = TVDRK3(xp, shape)
  integrator.advance(q, dt, rhs_func)
  # q.data が in-place で更新される
"""

from __future__ import annotations


class TVDRK3:
    """
    3次 TVD Runge-Kutta 積分器

    rhs_func(data) → 空間離散化演算子 L(q) の出力配列を返す。
    バッファを事前確保し、advance() 内で新規メモリ確保しない。
    """

    def __init__(self, xp, shape):
        """
        Args:
            xp: numpy or cupy
            shape: 場の配列形状
        """
        self.xp = xp
        # 作業バッファ（3段分）
        self._q_n = xp.zeros(shape)     # q^n の保存
        self._q_s1 = xp.zeros(shape)    # q^(1)
        self._q_s2 = xp.zeros(shape)    # q^(2)
        self._Lq = xp.zeros(shape)      # L(q) の出力

    def advance(self, q_data, dt, rhs_func):
        """
        1タイムステップの TVD-RK3 積分（in-place）

        Args:
            q_data: 更新対象の配列（in-place で書き換えられる）
            dt: 時間刻み幅
            rhs_func: callable(data) → L(q) の配列を返す
                      ※戻り値は self._Lq に書き込んでもよい
        """
        xp = self.xp
        q_n = self._q_n
        q_s1 = self._q_s1
        q_s2 = self._q_s2

        # q^n を保存
        xp.copyto(q_n, q_data)

        # ─── Stage 1: q^(1) = q^n + Δt L(q^n) ───
        Lq = rhs_func(q_data)
        xp.copyto(q_s1, q_n)
        q_s1 += dt * Lq

        # ─── Stage 2: q^(2) = (3/4)q^n + (1/4)[q^(1) + Δt L(q^(1))] ───
        Lq = rhs_func(q_s1)
        xp.copyto(q_s2, q_s1)
        q_s2 += dt * Lq
        q_s2 *= 0.25
        q_s2 += 0.75 * q_n

        # ─── Stage 3: q^{n+1} = (1/3)q^n + (2/3)[q^(2) + Δt L(q^(2))] ───
        Lq = rhs_func(q_s2)
        # q_data を直接書き換え
        xp.copyto(q_data, q_s2)
        q_data += dt * Lq
        q_data *= (2.0 / 3.0)
        q_data += (1.0 / 3.0) * q_n
