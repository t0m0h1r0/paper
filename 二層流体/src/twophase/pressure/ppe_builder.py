"""
(e_L) FVM 変密度 PPE 行列の構築

論文 §7.4, 式(69) [2D], 式(72) [3D]:

  2D 5点ステンシル:
    (1/hx²)[p_{i+1}/ρ_{i+1/2} - p_i(1/ρ_{i+1/2}+1/ρ_{i-1/2}) + p_{i-1}/ρ_{i-1/2}]
    + (1/hy²)[...]y = div^RC / Δt

  3D 7点ステンシル:
    上記 + (1/hz²)[...]z

CSR疎行列で構築。非ゼロパターンは格子構造で固定。
密度更新時は values 配列のみ in-place 書き換え（行列再構築なし）。
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.grid import Grid
    from ..core.field import ScalarField
    from ..backend import Backend


class PPEMatrixBuilder:
    """
    FVM変密度PPE行列の構築器

    内部点のみで連立系を構築（境界は Neumann dp/dn=0）。
    """

    def __init__(self, grid: Grid, backend: Backend):
        self.grid = grid
        self.backend = backend
        self.xp = backend.xp
        self.ndim = grid.ndim

        # 内部点のインデックスマッピング
        self._build_index_map()
        self._build_sparsity_pattern()

        # 疎行列（初期値は等密度）
        self.matrix = None  # CSR matrix, set by update_coefficients
        self._rhs = self.xp.zeros(self.n_unknowns)

    def _build_index_map(self):
        """内部格子点を1D連番にマッピング"""
        shape = self.grid.shape
        ndim = self.ndim

        # 内部点: 各軸 1..N-1
        if ndim == 2:
            Nx, Ny = shape
            self._inner_ranges = (range(1, Nx - 1), range(1, Ny - 1))
            self._inner_shape = (Nx - 2, Ny - 2)
        else:
            Nx, Ny, Nz = shape
            self._inner_ranges = (range(1, Nx - 1), range(1, Ny - 1),
                                  range(1, Nz - 1))
            self._inner_shape = (Nx - 2, Ny - 2, Nz - 2)

        self.n_unknowns = 1
        for s in self._inner_shape:
            self.n_unknowns *= s

    def _ijk_to_row(self, indices):
        """内部点の (i,j[,k]) → 1D行番号"""
        ndim = self.ndim
        if ndim == 2:
            i, j = indices
            ni, nj = self._inner_shape
            return (i - 1) * nj + (j - 1)
        else:
            i, j, k = indices
            ni, nj, nk = self._inner_shape
            return ((i - 1) * nj + (j - 1)) * nk + (k - 1)

    def _build_sparsity_pattern(self):
        """CSR形式の非ゼロパターンを構築（密度非依存）"""
        ndim = self.ndim
        n = self.n_unknowns
        shape = self.grid.shape

        # ステンシル: 中心 + 2*ndim 個の隣接点
        max_nnz = n * (1 + 2 * ndim)

        rows = []
        cols = []

        if ndim == 2:
            Nx, Ny = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    row = self._ijk_to_row((i, j))
                    # 中心
                    rows.append(row)
                    cols.append(row)
                    # x方向隣接
                    if i - 1 >= 1:
                        rows.append(row)
                        cols.append(self._ijk_to_row((i - 1, j)))
                    if i + 1 < Nx - 1:
                        rows.append(row)
                        cols.append(self._ijk_to_row((i + 1, j)))
                    # y方向隣接
                    if j - 1 >= 1:
                        rows.append(row)
                        cols.append(self._ijk_to_row((i, j - 1)))
                    if j + 1 < Ny - 1:
                        rows.append(row)
                        cols.append(self._ijk_to_row((i, j + 1)))
        else:
            Nx, Ny, Nz = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    for k in range(1, Nz - 1):
                        row = self._ijk_to_row((i, j, k))
                        rows.append(row); cols.append(row)
                        if i-1>=1: rows.append(row); cols.append(self._ijk_to_row((i-1,j,k)))
                        if i+1<Nx-1: rows.append(row); cols.append(self._ijk_to_row((i+1,j,k)))
                        if j-1>=1: rows.append(row); cols.append(self._ijk_to_row((i,j-1,k)))
                        if j+1<Ny-1: rows.append(row); cols.append(self._ijk_to_row((i,j+1,k)))
                        if k-1>=1: rows.append(row); cols.append(self._ijk_to_row((i,j,k-1)))
                        if k+1<Nz-1: rows.append(row); cols.append(self._ijk_to_row((i,j,k+1)))

        self._coo_rows = np.array(rows, dtype=np.int32)
        self._coo_cols = np.array(cols, dtype=np.int32)
        self._nnz = len(rows)

    def update_coefficients(self, rho: ScalarField):
        """
        密度に基づいて PPE 行列係数を更新

        Neumann BC (dp/dn=0): 境界の隣接点は中心に吸収
        """
        xp = self.xp
        ndim = self.ndim
        shape = self.grid.shape
        rho_data = self.backend.to_host(rho.data)  # numpy で構築

        vals = np.zeros(self._nnz, dtype=np.float64)
        idx = 0

        if ndim == 2:
            Nx, Ny = shape
            hx, hy = float(self.grid.h[0]), float(self.grid.h[1])
            inv_hx2 = 1.0 / (hx * hx)
            inv_hy2 = 1.0 / (hy * hy)

            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    # 界面密度
                    rho_xp = 0.5 * (rho_data[i, j] + rho_data[i+1, j])
                    rho_xm = 0.5 * (rho_data[i, j] + rho_data[i-1, j])
                    rho_yp = 0.5 * (rho_data[i, j] + rho_data[i, j+1])
                    rho_ym = 0.5 * (rho_data[i, j] + rho_data[i, j-1])

                    coeff_xp = inv_hx2 / max(rho_xp, 1e-14)
                    coeff_xm = inv_hx2 / max(rho_xm, 1e-14)
                    coeff_yp = inv_hy2 / max(rho_yp, 1e-14)
                    coeff_ym = inv_hy2 / max(rho_ym, 1e-14)

                    diag = -(coeff_xp + coeff_xm + coeff_yp + coeff_ym)

                    # Neumann BC 吸収: 境界外の隣接 → 中心に加算
                    if i - 1 < 1:
                        diag += coeff_xm; coeff_xm = 0
                    if i + 1 > Nx - 2:
                        diag += coeff_xp; coeff_xp = 0
                    if j - 1 < 1:
                        diag += coeff_ym; coeff_ym = 0
                    if j + 1 > Ny - 2:
                        diag += coeff_yp; coeff_yp = 0

                    # 中心
                    vals[idx] = diag; idx += 1
                    # x-
                    if i - 1 >= 1:
                        vals[idx] = coeff_xm; idx += 1
                    # x+
                    if i + 1 < Nx - 1:
                        vals[idx] = coeff_xp; idx += 1
                    # y-
                    if j - 1 >= 1:
                        vals[idx] = coeff_ym; idx += 1
                    # y+
                    if j + 1 < Ny - 1:
                        vals[idx] = coeff_yp; idx += 1
        else:
            Nx, Ny, Nz = shape
            hx, hy, hz = [float(self.grid.h[a]) for a in range(3)]
            inv_h2 = [1.0/(hx*hx), 1.0/(hy*hy), 1.0/(hz*hz)]

            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    for k in range(1, Nz-1):
                        ijk = (i,j,k)
                        coeffs = {}
                        diag = 0.0
                        for ax in range(3):
                            for sgn, delta in [(+1, 'p'), (-1, 'm')]:
                                nb = list(ijk)
                                nb[ax] += sgn
                                nb = tuple(nb)
                                rho_face = 0.5*(rho_data[ijk]+rho_data[nb])
                                c = inv_h2[ax]/max(rho_face, 1e-14)
                                diag -= c
                                key = (ax, delta)
                                lo = 1
                                hi = shape[ax]-2
                                if nb[ax]<lo or nb[ax]>hi:
                                    diag += c
                                else:
                                    coeffs[key] = c
                        vals[idx] = diag; idx += 1
                        if i-1>=1 and (0,'m') in coeffs: vals[idx]=coeffs[(0,'m')]; idx+=1
                        if i+1<Nx-1 and (0,'p') in coeffs: vals[idx]=coeffs[(0,'p')]; idx+=1
                        if j-1>=1 and (1,'m') in coeffs: vals[idx]=coeffs[(1,'m')]; idx+=1
                        if j+1<Ny-1 and (1,'p') in coeffs: vals[idx]=coeffs[(1,'p')]; idx+=1
                        if k-1>=1 and (2,'m') in coeffs: vals[idx]=coeffs[(2,'m')]; idx+=1
                        if k+1<Nz-1 and (2,'p') in coeffs: vals[idx]=coeffs[(2,'p')]; idx+=1

        # ─── 圧力ゲージ固定: row=0 を p=0 に置換 ───
        # 純 Neumann BC では圧力が定数分不定 → 1点を固定
        for k in range(idx):
            if self._coo_rows[k] == 0:
                if self._coo_cols[k] == 0:
                    vals[k] = 1.0
                else:
                    vals[k] = 0.0

        # CSR 構築
        try:
            import cupyx.scipy.sparse as sp
            vals_dev = xp.asarray(vals[:idx])
            rows_dev = xp.asarray(self._coo_rows[:idx].astype(np.int32))
            cols_dev = xp.asarray(self._coo_cols[:idx].astype(np.int32))
            coo = sp.coo_matrix((vals_dev, (rows_dev, cols_dev)),
                                shape=(self.n_unknowns, self.n_unknowns))
            self.matrix = coo.tocsr()
        except ImportError:
            import scipy.sparse as sp
            coo = sp.coo_matrix((vals[:idx],
                                 (self._coo_rows[:idx], self._coo_cols[:idx])),
                                shape=(self.n_unknowns, self.n_unknowns))
            self.matrix = coo.tocsr()

    def build_rhs(self, div_rc, dt):
        """
        PPE 右辺ベクトルを構築: b = div^RC / Δt

        Args:
            div_rc: RC発散の格子配列
            dt: 時間刻み幅

        Returns:
            rhs: 1D ベクトル (n_unknowns,)
        """
        xp = self.xp
        shape = self.grid.shape
        rhs = self._rhs
        rhs[:] = 0.0

        if self.ndim == 2:
            Nx, Ny = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    row = self._ijk_to_row((i, j))
                    rhs[row] = div_rc[i, j] / dt
        else:
            Nx, Ny, Nz = shape
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    for k in range(1, Nz-1):
                        row = self._ijk_to_row((i,j,k))
                        rhs[row] = div_rc[i,j,k] / dt

        # 圧力ゲージ固定: row=0 の RHS を 0 に
        rhs[0] = 0.0

        return rhs

    def scatter_solution(self, sol_vec, p: ScalarField):
        """
        1Dソリューションベクトルを格子場に書き戻す

        Args:
            sol_vec: 1D解ベクトル (n_unknowns,)
            p: 圧力場（in-place書き込み）
        """
        xp = self.xp
        shape = self.grid.shape
        p.data[:] = 0.0

        if self.ndim == 2:
            Nx, Ny = shape
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    row = self._ijk_to_row((i, j))
                    p.data[i, j] = sol_vec[row]
        else:
            Nx, Ny, Nz = shape
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    for k in range(1, Nz-1):
                        row = self._ijk_to_row((i,j,k))
                        p.data[i,j,k] = sol_vec[row]

        p.invalidate()
