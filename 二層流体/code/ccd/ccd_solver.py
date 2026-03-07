"""
結合コンパクト差分法（CCD）ソルバー

Chu & Fan (1998) の CCD スキームを実装。
3点ステンシルで f' と f'' を同時に O(h^6) 精度で計算する。

CCD 係数（テイラー展開から厳密導出、論文 §5）:
  Eq-I:  α₁ = 7/16,  a₁ = 15/16,  b₁ = 1/16
  Eq-II: β₂ = -1/8,   a₂ = 3,      b₂ = -9/8

ブロック行列（内点 i=1..N-1 の連立系）:
  A11 = tridiag(7/16, 1, 7/16)
  A12 = (h/16) tridiag(+1, 0, -1)     ← 下対角が正、上対角が負
  A21 = (9/(8h)) tridiag(-1, 0, +1)
  A22 = tridiag(-1/8, 1, -1/8)

境界コンパクトスキーム（O(h^5)、論文 §5.7）:
  左境界(i=0):  4点ステンシル f₀,f₁,f₂,f₃ + f'₁,f''₁ との結合
  右境界(i=N):  対称形（h→-h 変換）

バッチ並列:
  2D場に対して x方向CCD を解く場合、y方向の全列を同時処理。
  3D場では (Ny+1)*(Nz+1) 個の独立な1D問題を1回で解く。
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

from .block_tridiag import BlockTridiagSolver

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..backend import Backend


# ═══════════════════════════════════════════════════════════
# CCD 係数（クラス定数）
# ═══════════════════════════════════════════════════════════
ALPHA1 = 7.0 / 16.0    # Eq-I 対角
A1 = 15.0 / 16.0       # Eq-I RHS 関数値係数
B1 = 1.0 / 16.0        # Eq-I RHS f''結合係数

BETA2 = -1.0 / 8.0     # Eq-II 対角
A2 = 3.0               # Eq-II RHS 関数値係数
B2 = -9.0 / 8.0        # Eq-II RHS f'結合係数


class CCDSolver:
    """
    CCD ソルバー（2D/3D 対応、バッチ並列）

    各軸の格子間隔 h に対してブロック三重対角系を事前 LU 分解し、
    differentiate() で任意の場データの微分を高速に計算する。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

        # 各軸の事前分解済みソルバーを構築
        # 'compact': 境界コンパクトスキーム（通常使用）
        # 'prescribed': 指定境界条件（テスト用）
        self._solvers = {}
        for ax in range(self.ndim):
            n_points = grid.N[ax] + 1
            h = float(grid.h[ax])
            self._solvers[ax] = {
                'compact': self._build_solver(n_points, h, absorb_bc=True),
                'prescribed': self._build_solver(n_points, h, absorb_bc=False),
            }

    def _build_solver(self, n_points: int, h: float, absorb_bc: bool = True) -> dict:
        """
        軸1本分の CCD ソルバーを構築

        Returns:
            dict with:
                'solver': 分解済み BlockTridiagSolver
                'h': 格子間隔
                'N': n_points - 1 (セル数)
                'bc_left_coeffs': 左境界の結合係数
                'bc_right_coeffs': 右境界の結合係数
        """
        N = n_points - 1  # インデックス 0..N
        n_interior = N - 1  # 内点数 (i=1..N-1)
        assert n_interior >= 1, f"Need at least 3 points, got {n_points}"

        # ─── 境界スキームの結合係数を計算 ───
        # 左境界: f'₀ = c₁ + M_left @ [f'₁, f''₁]
        #          f''₀ = c₂ + ...
        # c₁, c₂ はデータ依存（solve時に計算）、M_left は定数
        bc_left = self._boundary_coupling_left(h)
        bc_right = self._boundary_coupling_right(h)

        # ─── ブロック三重対角系の構築 ───
        # 各内点 i (i=1..N-1) でのブロック:
        #   x_i = [f'_i, f''_i]^T
        #   L_i x_{i-1} + D_i x_i + U_i x_{i+1} = b_i

        diag = []
        lower = []
        upper = []

        for idx in range(n_interior):
            # 標準の内点ブロック
            D = np.array([[1.0, 0.0],
                          [0.0, 1.0]])
            L = np.array([[ALPHA1, B1 * h],
                          [B2 / h, BETA2]])
            U = np.array([[ALPHA1, -B1 * h],
                          [-B2 / h, BETA2]])

            diag.append(D)
            lower.append(L)
            upper.append(U)

        # ─── 境界吸収: 最初の行を修正 (i=1, idx=0) ───
        if absorb_bc:
            L0 = lower[0]
            M_left = bc_left['M']
            diag[0] = diag[0] + L0 @ M_left
            lower[0] = np.zeros((2, 2))

        # ─── 境界吸収: 最後の行を修正 (i=N-1, idx=n_interior-1) ───
        if absorb_bc and n_interior >= 1:
            U_last = upper[n_interior - 1]
            M_right = bc_right['M']
            diag[n_interior - 1] = diag[n_interior - 1] + U_last @ M_right
            upper[n_interior - 1] = np.zeros((2, 2))

        if not absorb_bc:
            # 指定BC用: 境界結合ブロックをゼロに（RHSに移行するため）
            lower[0] = np.zeros((2, 2))
            if n_interior >= 1:
                upper[n_interior - 1] = np.zeros((2, 2))

        # ─── LU 分解 ───
        solver = BlockTridiagSolver(self.xp)
        solver.factorize(diag, lower, upper)

        return {
            'solver': solver,
            'h': h,
            'N': N,
            'n_interior': n_interior,
            'bc_left': bc_left,
            'bc_right': bc_right,
            'L0_orig': np.array([[ALPHA1, B1 * h],
                                 [B2 / h, BETA2]]),
            'UN_orig': np.array([[ALPHA1, -B1 * h],
                                 [-B2 / h, BETA2]]),
        }

    @staticmethod
    def _boundary_coupling_left(h: float) -> dict:
        """
        左境界 (i=0) コンパクトスキーム（O(h^5)、論文 式(52)(53)）

        Eq-I-bc:
          f'₀ + (3/2)f'₁ - (3h/2)f''₁
          = (1/h)(-23/6·f₀ + 21/4·f₁ - 3/2·f₂ + 1/12·f₃)

        Eq-II-bc:
          f''₀ - 3f''₁ + (23/(3h))f'₀ + (9/h)f'₁
          = (1/h²)(-325/18·f₀ + 39/2·f₁ - 3/2·f₂ + 1/18·f₃)

        f'₀ = R_I - (3/2)f'₁ + (3h/2)f''₁
        f''₀ = [R_II - (23/(3h))R_I] + (5/(2h))f'₁ - (17/2)f''₁

        → [f'₀]  = M @ [f'₁]  + [c₁]
          [f''₀]       [f''₁]    [c₂]

        M は定数、c はデータ依存
        """
        M = np.array([[-3.0 / 2.0, 3.0 * h / 2.0],
                       [5.0 / (2.0 * h), -17.0 / 2.0]])

        # RHS 係数（f値への係数、データ依存部分の構築に使う）
        # R_I = (1/h)(c_I0·f₀ + c_I1·f₁ + c_I2·f₂ + c_I3·f₃)
        c_I = np.array([-23.0 / 6.0, 21.0 / 4.0, -3.0 / 2.0, 1.0 / 12.0]) / h

        # R_II = (1/h²)(c_II0·f₀ + c_II1·f₁ + c_II2·f₂ + c_II3·f₃)
        c_II = np.array([-325.0 / 18.0, 39.0 / 2.0, -3.0 / 2.0, 1.0 / 18.0]) / (h * h)

        return {'M': M, 'c_I': c_I, 'c_II': c_II, 'h': h}

    @staticmethod
    def _boundary_coupling_right(h: float) -> dict:
        """
        右境界 (i=N) コンパクトスキーム（O(h^5)、論文 式(54)(55)）

        h→-h 変換により:
        Eq-I-bc:
          f'_N + (3/2)f'_{N-1} + (3h/2)f''_{N-1}
          = (1/h)(23/6·f_N - 21/4·f_{N-1} + 3/2·f_{N-2} - 1/12·f_{N-3})

        Eq-II-bc:
          f''_N - 3f''_{N-1} - (23/(3h))f'_N - (9/h)f'_{N-1}
          = (1/h²)(-325/18·f_N + 39/2·f_{N-1} - 3/2·f_{N-2} + 1/18·f_{N-3})

        f'_N = R_I_r - (3/2)f'_{N-1} - (3h/2)f''_{N-1}
        f''_N = [R_II_r + (23/(3h))R_I_r] - (5/(2h))f'_{N-1} - (17/2)f''_{N-1}
        """
        M = np.array([[-3.0 / 2.0, -3.0 * h / 2.0],
                       [-5.0 / (2.0 * h), -17.0 / 2.0]])

        c_I = np.array([23.0 / 6.0, -21.0 / 4.0, 3.0 / 2.0, -1.0 / 12.0]) / h
        c_II = np.array([-325.0 / 18.0, 39.0 / 2.0, -3.0 / 2.0, 1.0 / 18.0]) / (h * h)

        return {'M': M, 'c_I': c_I, 'c_II': c_II, 'h': h}

    def differentiate(self, data, axis: int,
                      bc_left: Optional[Tuple] = None,
                      bc_right: Optional[Tuple] = None):
        """
        指定軸方向の1階・2階微分を計算（CCD法）

        Args:
            data: 場の配列 shape = grid.shape
            axis: 微分する軸 (0, 1, or 2)
            bc_left:  (f'₀, f''₀) を指定（テスト用）。None→境界スキーム使用
            bc_right: (f'_N, f''_N) を指定（テスト用）。None→境界スキーム使用

        Returns:
            d1: 1階微分配列 shape = grid.shape
            d2: 2階微分配列 shape = grid.shape
        """
        xp = self.xp
        use_prescribed = (bc_left is not None or bc_right is not None)
        mode_key = 'prescribed' if use_prescribed else 'compact'
        info = self._solvers[axis][mode_key]
        solver = info['solver']
        h = info['h']
        N = info['N']
        n_int = info['n_interior']

        # ─── data を (axis_len, batch) に reshape ───
        f = xp.moveaxis(data, axis, 0)   # 対象軸を先頭に
        orig_shape = f.shape
        n_pts = f.shape[0]               # N+1
        batch_shape = f.shape[1:]
        batch_size = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1
        f = f.reshape(n_pts, batch_size)  # (N+1, batch)

        # ─── 内点 RHS の構築 ───
        # r1_i = (15/(16h)) (f_{i+1} - f_{i-1})
        # r2_i = (3/h²) (f_{i-1} - 2f_i + f_{i+1})
        rhs = xp.zeros((n_int, 2, batch_size))
        for idx in range(n_int):
            i = idx + 1  # 実インデックス
            rhs[idx, 0, :] = (A1 / h) * (f[i + 1] - f[i - 1])
            rhs[idx, 1, :] = (A2 / (h * h)) * (f[i - 1] - 2.0 * f[i] + f[i + 1])

        # ─── 境界値の計算とRHS修正 ───
        if bc_left is not None:
            # 指定値モード（テスト用）
            fp0 = xp.full(batch_size, bc_left[0])
            fpp0 = xp.full(batch_size, bc_left[1])
        else:
            # 境界コンパクトスキーム → データ依存部分 c を計算
            bc_l = info['bc_left']
            c_I, c_II = xp.asarray(bc_l['c_I']), xp.asarray(bc_l['c_II'])
            # R_I = c_I @ f[0:4]
            R_I = c_I[0] * f[0] + c_I[1] * f[1] + c_I[2] * f[2] + c_I[3] * f[3]
            # R_II = c_II @ f[0:4]
            R_II = c_II[0] * f[0] + c_II[1] * f[1] + c_II[2] * f[2] + c_II[3] * f[3]
            # c₁ = R_I, c₂ = R_II - (23/(3h)) R_I
            c1 = R_I
            c2 = R_II - (23.0 / (3.0 * h)) * R_I
            # f'₀, f''₀ は f'₁, f''₁ に依存（吸収済み）
            # RHS修正: b'_0 -= L0_orig @ [c1, c2]
            fp0 = c1    # 後で全解から復元用
            fpp0 = c2

        if bc_right is not None:
            fpN = xp.full(batch_size, bc_right[0])
            fppN = xp.full(batch_size, bc_right[1])
        else:
            bc_r = info['bc_right']
            c_I_r, c_II_r = xp.asarray(bc_r['c_I']), xp.asarray(bc_r['c_II'])
            R_I_r = (c_I_r[0] * f[N] + c_I_r[1] * f[N - 1]
                     + c_I_r[2] * f[N - 2] + c_I_r[3] * f[N - 3])
            R_II_r = (c_II_r[0] * f[N] + c_II_r[1] * f[N - 1]
                      + c_II_r[2] * f[N - 2] + c_II_r[3] * f[N - 3])
            c1_r = R_I_r
            c2_r = R_II_r + (23.0 / (3.0 * h)) * R_I_r
            fpN = c1_r
            fppN = c2_r

        # ─── RHS にデータ依存の境界項を反映 ───
        L0 = xp.asarray(info['L0_orig'])
        UN = xp.asarray(info['UN_orig'])

        if bc_left is not None:
            # 指定値: RHS -= L0 @ [f'₀, f''₀]
            rhs[0, 0, :] -= L0[0, 0] * fp0 + L0[0, 1] * fpp0
            rhs[0, 1, :] -= L0[1, 0] * fp0 + L0[1, 1] * fpp0
        else:
            # 境界スキーム: L0 @ c ベクトルを引く（M部分は行列に吸収済み）
            rhs[0, 0, :] -= L0[0, 0] * fp0 + L0[0, 1] * fpp0
            rhs[0, 1, :] -= L0[1, 0] * fp0 + L0[1, 1] * fpp0

        if bc_right is not None:
            rhs[n_int - 1, 0, :] -= UN[0, 0] * fpN + UN[0, 1] * fppN
            rhs[n_int - 1, 1, :] -= UN[1, 0] * fpN + UN[1, 1] * fppN
        else:
            rhs[n_int - 1, 0, :] -= UN[0, 0] * fpN + UN[0, 1] * fppN
            rhs[n_int - 1, 1, :] -= UN[1, 0] * fpN + UN[1, 1] * fppN

        # ─── ブロック三重対角系をソルブ ───
        sol = solver.solve(rhs)  # shape (n_int, 2, batch)

        # ─── 全解の組み立て ───
        d1_flat = xp.zeros((n_pts, batch_size))
        d2_flat = xp.zeros((n_pts, batch_size))

        # 内点
        for idx in range(n_int):
            d1_flat[idx + 1] = sol[idx, 0, :]
            d2_flat[idx + 1] = sol[idx, 1, :]

        # 境界点
        if bc_left is not None:
            d1_flat[0] = fp0
            d2_flat[0] = fpp0
        else:
            # 復元: [f'₀, f''₀] = M @ [f'₁, f''₁] + [c₁, c₂]
            M_l = xp.asarray(info['bc_left']['M'])
            d1_flat[0] = (M_l[0, 0] * d1_flat[1] + M_l[0, 1] * d2_flat[1]
                          + fp0)
            d2_flat[0] = (M_l[1, 0] * d1_flat[1] + M_l[1, 1] * d2_flat[1]
                          + fpp0)

        if bc_right is not None:
            d1_flat[N] = fpN
            d2_flat[N] = fppN
        else:
            M_r = xp.asarray(info['bc_right']['M'])
            d1_flat[N] = (M_r[0, 0] * d1_flat[N - 1]
                          + M_r[0, 1] * d2_flat[N - 1] + fpN)
            d2_flat[N] = (M_r[1, 0] * d1_flat[N - 1]
                          + M_r[1, 1] * d2_flat[N - 1] + fppN)

        # ─── 元の形状に復元 ───
        d1 = d1_flat.reshape(orig_shape)
        d2 = d2_flat.reshape(orig_shape)
        d1 = xp.moveaxis(d1, 0, axis)
        d2 = xp.moveaxis(d2, 0, axis)

        return d1, d2
