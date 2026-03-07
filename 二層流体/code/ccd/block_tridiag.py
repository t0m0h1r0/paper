"""
2×2 ブロック三重対角ソルバー

CCD の内点連立系を効率的に解く。
行列係数はバッチ間で共通のため、LU 分解を1回だけ行い、
異なる RHS に対するソルブをバッチ並列で実行する。

ブロック三重対角系:
  D_0 x_0 + U_0 x_1 = b_0
  L_i x_{i-1} + D_i x_i + U_i x_{i+1} = b_i   (i=1..n-2)
  L_{n-1} x_{n-2} + D_{n-1} x_{n-1} = b_{n-1}

各ブロックは 2×2 行列、x_i = [f'_i, f''_i]^T
"""

from __future__ import annotations
import numpy as np


def _inv2x2(M):
    """
    2×2 行列の逆行列（要素演算、バッチ非対応のスカラー版）
    M: shape (2, 2)
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det


def _matmul2x2(A, B):
    """2×2 行列同士の積（明示的に展開）"""
    C = np.empty((2, 2))
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return C


def _matvec2x2_batch(M, v, xp):
    """
    2×2 行列 × バッチベクトルの積
    M: shape (2, 2)
    v: shape (2, batch)
    returns: shape (2, batch)
    """
    out = xp.empty_like(v)
    out[0] = M[0, 0] * v[0] + M[0, 1] * v[1]
    out[1] = M[1, 0] * v[0] + M[1, 1] * v[1]
    return out


class BlockTridiagSolver:
    """
    2×2 ブロック三重対角系のソルバー

    行列係数はバッチ間で共有（CCD の係数は格子間隔のみに依存するため）。
    factorize() で LU 分解を事前計算し、solve() でバッチ並列にソルブ。
    """

    def __init__(self, xp):
        self.xp = xp
        self._factored = False

    def factorize(self, diag, lower, upper):
        """
        ブロック LU 分解を事前計算

        Args:
            diag:  list of (2,2) arrays, length n  — 対角ブロック D_i
            lower: list of (2,2) arrays, length n  — 下対角ブロック L_i
                   lower[0] は未使用（存在しない）
            upper: list of (2,2) arrays, length n  — 上対角ブロック U_i
                   upper[n-1] は未使用
        """
        n = len(diag)
        self._n = n
        self._upper = [np.array(u, dtype=np.float64) for u in upper]

        # 前進掃引: D'_i, W_i を計算
        self._Dp = [None] * n       # 修正対角 D'_i
        self._Dp_inv = [None] * n   # inv(D'_i)
        self._W = [None] * n        # W_i = L_i · inv(D'_{i-1})

        self._Dp[0] = np.array(diag[0], dtype=np.float64)
        self._Dp_inv[0] = _inv2x2(self._Dp[0])

        for i in range(1, n):
            Li = np.array(lower[i], dtype=np.float64)
            Ui_prev = self._upper[i - 1]

            # W_i = L_i · inv(D'_{i-1})
            self._W[i] = _matmul2x2(Li, self._Dp_inv[i - 1])

            # D'_i = D_i - W_i · U_{i-1}
            Di = np.array(diag[i], dtype=np.float64)
            self._Dp[i] = Di - _matmul2x2(self._W[i], Ui_prev)
            self._Dp_inv[i] = _inv2x2(self._Dp[i])

        self._factored = True

    def solve(self, rhs):
        """
        バッチ並列ソルブ

        Args:
            rhs: shape (n, 2, batch) — 右辺ベクトル

        Returns:
            sol: shape (n, 2, batch) — 解ベクトル
        """
        assert self._factored, "Call factorize() first"
        xp = self.xp
        n = self._n

        # 前進代入: b'_i = b_i - W_i · b'_{i-1}
        bp = xp.empty_like(rhs)
        bp[0] = rhs[0].copy()

        for i in range(1, n):
            bp[i] = rhs[i] - _matvec2x2_batch(
                xp.asarray(self._W[i]), bp[i - 1], xp)

        # 後退代入: x_i = inv(D'_i) · (b'_i - U_i · x_{i+1})
        sol = xp.empty_like(rhs)
        sol[n - 1] = _matvec2x2_batch(
            xp.asarray(self._Dp_inv[n - 1]), bp[n - 1], xp)

        for i in range(n - 2, -1, -1):
            tmp = bp[i] - _matvec2x2_batch(
                xp.asarray(self._upper[i]), sol[i + 1], xp)
            sol[i] = _matvec2x2_batch(
                xp.asarray(self._Dp_inv[i]), tmp, xp)

        return sol
