"""
Helmholtz ソルバー（Crank-Nicolson 粘性項用）

論文 §8.2, 式(82):
  [I - (Δt/2)·∇·(σ∇)] u* = RHS

  σ(x) = Δt·μ̃(x) / (2·ρ̃(x)·Re)  — 空間変化する拡散係数

FVM 離散化（PPE と同様の5点/7点ステンシル）:
  ∇·(σ∇u*) ≈ Σ_ax (1/h_ax²)[σ_{+face}(u_{+1}-u) - σ_{-face}(u-u_{-1})]

Helmholtz 行列 A = I - L_σ は:
  - 対角優位（1 + 正の値）→ BiCGSTAB が高速収束
  - SPD に近い → 前処理なしでも十分な場合が多い
  - 構造は PPE と同一（5点/7点）→ 同じ疎行列パターンを再利用

境界条件:
  壁面 Neumann (∂u*/∂n = 0): PPE と同様に対角に吸収
  → 速度の壁面境界条件は Helmholtz 解の後に別途適用
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField
    from ..backend import Backend
    from ..config import SimulationConfig


class HelmholtzSolver:
    """
    [I - L_σ] x = b を解く Helmholtz ソルバー

    L_σ は FVM 変係数ラプラシアン（PPE と同一のステンシル構造）。
    σ = Δt·μ/(2ρRe) は毎ステップ更新される。

    PPEMatrixBuilder と同じインデックスマッピング・非ゼロパターンを使用。
    行列の構築と求解を1クラスに統合（CN用途に特化）。
    """

    def __init__(self, grid: Grid, backend: Backend, config: SimulationConfig):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim
        self.config = config

        # ─── 内部点インデックスマッピング（PPE と同一）───
        shape = grid.shape
        if self.ndim == 2:
            Nx, Ny = shape
            self._inner_shape = (Nx - 2, Ny - 2)
        else:
            Nx, Ny, Nz = shape
            self._inner_shape = (Nx - 2, Ny - 2, Nz - 2)
        self.n_unknowns = 1
        for s in self._inner_shape:
            self.n_unknowns *= s

        # ─── 非ゼロパターンの構築 ───
        self._build_sparsity_pattern()

        # ─── ソルバー設定 ───
        self.tol = config.bicgstab_tol
        self.maxiter = config.bicgstab_maxiter

        # ─── 作業バッファ ───
        self._sol_vec = self.xp.zeros(self.n_unknowns)
        self._rhs_vec = self.xp.zeros(self.n_unknowns)

        # ─── 行列キャッシュ ───
        self.matrix = None
        self._last_iters = 0

    def _ijk_to_row(self, indices):
        """内部点 (i,j[,k]) → 1D 行番号"""
        if self.ndim == 2:
            i, j = indices
            return (i - 1) * self._inner_shape[1] + (j - 1)
        else:
            i, j, k = indices
            nj, nk = self._inner_shape[1], self._inner_shape[2]
            return ((i - 1) * nj + (j - 1)) * nk + (k - 1)

    def _build_sparsity_pattern(self):
        """CSR 非ゼロパターン構築（PPE と同一構造）"""
        shape = self.grid.shape
        rows, cols = [], []

        if self.ndim == 2:
            Nx, Ny = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    row = self._ijk_to_row((i, j))
                    rows.append(row); cols.append(row)  # 対角
                    if i - 1 >= 1:
                        rows.append(row); cols.append(self._ijk_to_row((i-1, j)))
                    if i + 1 < Nx - 1:
                        rows.append(row); cols.append(self._ijk_to_row((i+1, j)))
                    if j - 1 >= 1:
                        rows.append(row); cols.append(self._ijk_to_row((i, j-1)))
                    if j + 1 < Ny - 1:
                        rows.append(row); cols.append(self._ijk_to_row((i, j+1)))
        else:
            Nx, Ny, Nz = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    for k in range(1, Nz - 1):
                        row = self._ijk_to_row((i, j, k))
                        rows.append(row); cols.append(row)
                        if i-1 >= 1:
                            rows.append(row); cols.append(self._ijk_to_row((i-1,j,k)))
                        if i+1 < Nx-1:
                            rows.append(row); cols.append(self._ijk_to_row((i+1,j,k)))
                        if j-1 >= 1:
                            rows.append(row); cols.append(self._ijk_to_row((i,j-1,k)))
                        if j+1 < Ny-1:
                            rows.append(row); cols.append(self._ijk_to_row((i,j+1,k)))
                        if k-1 >= 1:
                            rows.append(row); cols.append(self._ijk_to_row((i,j,k-1)))
                        if k+1 < Nz-1:
                            rows.append(row); cols.append(self._ijk_to_row((i,j,k+1)))

        self._coo_rows = np.array(rows, dtype=np.int32)
        self._coo_cols = np.array(cols, dtype=np.int32)
        self._nnz = len(rows)

    def update_matrix(self, mu: ScalarField, rho: ScalarField, dt: float):
        """
        Helmholtz 行列 [I - L_σ] を構築

        σ(x) = dt · μ̃(x) / (2 · ρ̃(x) · Re)

        FVM 離散化:
          L_σ の (i,j) 行: Σ_ax (1/h_ax²) [σ_{+face}·u_{+1} - (σ_{+face}+σ_{-face})·u + σ_{-face}·u_{-1}]

        Helmholtz 対角: 1 + Σ (σ_+face + σ_-face) / h_ax²  （常に > 1）
        Helmholtz 非対角: -σ_face / h_ax²
        """
        shape = self.grid.shape
        Re = self.config.Re

        # σ = dt·μ/(2ρRe) を numpy 上で計算
        mu_np = self.backend.to_host(mu.data)
        rho_np = self.backend.to_host(rho.data)
        sigma_np = dt * mu_np / (2.0 * np.maximum(rho_np, 1e-14) * Re)

        vals = np.zeros(self._nnz, dtype=np.float64)
        idx = 0

        if self.ndim == 2:
            Nx, Ny = shape
            hx, hy = float(self.grid.h[0]), float(self.grid.h[1])
            inv_hx2, inv_hy2 = 1.0 / (hx * hx), 1.0 / (hy * hy)

            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    # 界面の σ（線形補間）
                    s_xp = 0.5 * (sigma_np[i, j] + sigma_np[i+1, j])
                    s_xm = 0.5 * (sigma_np[i, j] + sigma_np[i-1, j])
                    s_yp = 0.5 * (sigma_np[i, j] + sigma_np[i, j+1])
                    s_ym = 0.5 * (sigma_np[i, j] + sigma_np[i, j-1])

                    # L_σ の非対角係数（符号: L_σ 行列の非対角は正）
                    c_xp = s_xp * inv_hx2
                    c_xm = s_xm * inv_hx2
                    c_yp = s_yp * inv_hy2
                    c_ym = s_ym * inv_hy2

                    # Helmholtz 対角: 1 + Σ c  (I - L_σ の対角)
                    diag = 1.0 + c_xp + c_xm + c_yp + c_ym

                    # Neumann BC 吸収: 境界外の隣接 → 対角から引く
                    if i - 1 < 1:
                        diag -= c_xm; c_xm = 0
                    if i + 1 > Nx - 2:
                        diag -= c_xp; c_xp = 0
                    if j - 1 < 1:
                        diag -= c_ym; c_ym = 0
                    if j + 1 > Ny - 2:
                        diag -= c_yp; c_yp = 0

                    # 行列値の格納（スパシティパターンと同順）
                    vals[idx] = diag; idx += 1
                    if i - 1 >= 1:
                        vals[idx] = -c_xm; idx += 1
                    if i + 1 < Nx - 1:
                        vals[idx] = -c_xp; idx += 1
                    if j - 1 >= 1:
                        vals[idx] = -c_ym; idx += 1
                    if j + 1 < Ny - 1:
                        vals[idx] = -c_yp; idx += 1

        else:  # 3D
            Nx, Ny, Nz = shape
            h = [float(self.grid.h[a]) for a in range(3)]
            inv_h2 = [1.0 / (h[a] * h[a]) for a in range(3)]

            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    for k in range(1, Nz - 1):
                        ijk = (i, j, k)
                        coeffs = {}
                        diag = 1.0  # 恒等行列の寄与

                        for ax in range(3):
                            for sgn, label in [(+1, 'p'), (-1, 'm')]:
                                nb = list(ijk)
                                nb[ax] += sgn
                                nb = tuple(nb)
                                s_face = 0.5 * (sigma_np[ijk] + sigma_np[nb])
                                c = s_face * inv_h2[ax]
                                diag += c
                                lo, hi = 1, shape[ax] - 2
                                if nb[ax] < lo or nb[ax] > hi:
                                    diag -= c  # Neumann 吸収
                                else:
                                    coeffs[(ax, label)] = c

                        vals[idx] = diag; idx += 1
                        if i-1 >= 1 and (0,'m') in coeffs:
                            vals[idx] = -coeffs[(0,'m')]; idx += 1
                        if i+1 < Nx-1 and (0,'p') in coeffs:
                            vals[idx] = -coeffs[(0,'p')]; idx += 1
                        if j-1 >= 1 and (1,'m') in coeffs:
                            vals[idx] = -coeffs[(1,'m')]; idx += 1
                        if j+1 < Ny-1 and (1,'p') in coeffs:
                            vals[idx] = -coeffs[(1,'p')]; idx += 1
                        if k-1 >= 1 and (2,'m') in coeffs:
                            vals[idx] = -coeffs[(2,'m')]; idx += 1
                        if k+1 < Nz-1 and (2,'p') in coeffs:
                            vals[idx] = -coeffs[(2,'p')]; idx += 1

        # CSR 構築
        import scipy.sparse as sp
        coo = sp.coo_matrix(
            (vals[:idx], (self._coo_rows[:idx], self._coo_cols[:idx])),
            shape=(self.n_unknowns, self.n_unknowns))
        self.matrix = coo.tocsr()

    def solve_component(self, field: ScalarField):
        """
        1速度成分の Helmholtz 方程式を in-place で解く

        入力時: field.data に RHS が格納されている
        出力時: field.data に解 u* が書き込まれる（内部点のみ更新）

        Args:
            field: RHS 兼 解の出力先（ScalarField）
        """
        xp = self.xp
        shape = self.grid.shape

        # ─── RHS を 1D ベクトルに gather ───
        rhs = self._rhs_vec
        rhs[:] = 0.0
        if self.ndim == 2:
            Nx, Ny = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    rhs[self._ijk_to_row((i, j))] = field.data[i, j]
        else:
            Nx, Ny, Nz = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    for k in range(1, Nz - 1):
                        rhs[self._ijk_to_row((i,j,k))] = field.data[i,j,k]

        # ─── BiCGSTAB 求解 ───
        import scipy.sparse.linalg as splinalg
        rhs_cpu = self.backend.to_host(rhs)
        x0 = self.backend.to_host(self._sol_vec)

        iters = [0]
        def callback(xk):
            iters[0] += 1

        x, info = splinalg.bicgstab(
            self.matrix, rhs_cpu, x0=x0,
            rtol=self.tol, maxiter=self.maxiter,
            callback=callback)

        self._last_iters = iters[0]
        if info != 0:
            print(f"  [Helmholtz] BiCGSTAB warning: info={info}, iters={iters[0]}")

        sol = self.backend.to_device(x)
        xp.copyto(self._sol_vec, sol)  # 次ステップの初期推定として保存

        # ─── 解を格子場に scatter ───
        if self.ndim == 2:
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    field.data[i, j] = sol[self._ijk_to_row((i, j))]
        else:
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    for k in range(1, Nz - 1):
                        field.data[i,j,k] = sol[self._ijk_to_row((i,j,k))]

        field.invalidate()
