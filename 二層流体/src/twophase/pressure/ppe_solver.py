"""
PPE ソルバー: BiCGSTAB + ILU(0) 前処理

論文 §7.4:
  L_FVM · p^{n+1} = b^RC / Δt
  BiCGSTAB (van der Vorst, 1992) で反復求解。

CuPy が利用可能なら cupyx.scipy.sparse.linalg.bicgstab を使用、
なければ scipy.sparse.linalg.bicgstab にフォールバック。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend import Backend
    from ..config import SimulationConfig


class PPESolver:
    """
    疎行列線形系のBiCGSTABソルバー
    """

    def __init__(self, backend: Backend, config: SimulationConfig):
        self.backend = backend
        self.xp = backend.xp
        self.tol = config.bicgstab_tol
        self.maxiter = config.bicgstab_maxiter
        self._last_iters = 0

    def solve(self, A, rhs, x0=None):
        """
        A x = rhs を BiCGSTAB で解く

        Args:
            A: CSR 疎行列 (n × n)
            rhs: 右辺ベクトル (n,)
            x0: 初期推定（None → ゼロ）

        Returns:
            x: 解ベクトル (n,)
        """
        xp = self.xp

        if x0 is None:
            x0 = xp.zeros_like(rhs)

        if self.backend.use_gpu:
            try:
                import cupyx.scipy.sparse.linalg as splinalg
                x, info = splinalg.bicgstab(A, rhs, x0=x0,
                                            tol=self.tol,
                                            maxiter=self.maxiter)
                self._last_iters = -1  # CuPy doesn't report iters
                if info != 0:
                    print(f"  [PPE] BiCGSTAB warning: info={info}")
                return x
            except Exception:
                pass

        # CPU fallback
        import scipy.sparse.linalg as splinalg
        A_cpu = A
        rhs_cpu = self.backend.to_host(rhs)
        x0_cpu = self.backend.to_host(x0)

        iters = [0]
        def callback(xk):
            iters[0] += 1

        x, info = splinalg.bicgstab(A_cpu, rhs_cpu, x0=x0_cpu,
                                     rtol=self.tol,
                                     maxiter=self.maxiter,
                                     callback=callback)
        self._last_iters = iters[0]
        if info != 0:
            print(f"  [PPE] BiCGSTAB warning: info={info}, iters={iters[0]}")

        return self.backend.to_device(x)
