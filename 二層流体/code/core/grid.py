"""
コロケート格子（2D/3D統一）+ 界面適合非一様格子

論文 §4: ω(φ) = 1 + (α-1)δ*(φ) で界面近傍を自動高解像度化
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backend import Backend
    from ..config import SimulationConfig


class Grid:
    def __init__(self, config: SimulationConfig, backend: Backend):
        xp = backend.xp
        self.ndim = config.ndim
        self.N = tuple(config.N)
        self.L = tuple(config.L)
        self.backend = backend
        self.config = config
        self.axis_names = ('x', 'y', 'z')[:self.ndim]
        self.adaptive = False
        self.coords, self.h, self.J, self.dJ_dxi = {}, {}, {}, {}

        for ax in range(self.ndim):
            n, length = self.N[ax], self.L[ax]
            self.coords[ax] = xp.linspace(0.0, length, n + 1)
            self.h[ax] = length / n
            self.J[ax] = xp.ones(n + 1)
            self.dJ_dxi[ax] = xp.zeros(n + 1)

    @property
    def shape(self):
        return tuple(n + 1 for n in self.N)

    @property
    def dx_min(self) -> float:
        xp = self.backend.xp
        return min(float(xp.min(self.coords[ax][1:] - self.coords[ax][:-1]))
                   for ax in range(self.ndim))

    def meshgrid(self):
        xp = self.backend.xp
        return xp.meshgrid(*[self.coords[ax] for ax in range(self.ndim)],
                           indexing='ij')

    def update_from_levelset(self, phi, ccd, alpha=None, max_iter=10, tol=1e-6):
        """③ 界面適合格子の再生成（論文 §4.5）"""
        import numpy as np
        from scipy.interpolate import CubicSpline
        xp = self.backend.xp
        if alpha is None:
            alpha = self.config.alpha_grid
        if alpha <= 1.0:
            return
        self.adaptive = True
        epsilon = self.config.epsilon_factor * self.dx_min

        for ax in range(self.ndim):
            n, length = self.N[ax], self.L[ax]
            phi_1d_np = np.asarray(self.backend.to_host(
                self._extract_axis_profile(phi.data, ax)))
            x_old = np.linspace(0.0, length, n + 1)

            for _ in range(max_iter):
                omega = self._density_func(phi_1d_np, epsilon, alpha)
                dx0 = length / n
                s = np.zeros(n + 1)
                for i in range(1, n + 1):
                    s[i] = s[i-1] + omega[i-1] * dx0
                s_equal = np.linspace(0, s[-1], n + 1)
                cs = CubicSpline(s, x_old)
                x_new = cs(s_equal)
                x_new[0], x_new[-1] = 0.0, length
                # 単調性を保証
                x_new = np.sort(x_new)
                x_new[0], x_new[-1] = 0.0, length
                if np.max(np.abs(x_new - x_old)) < tol:
                    break
                x_old = x_new.copy()

            self.coords[ax] = xp.asarray(x_new)
            dx_arr = x_new[1:] - x_new[:-1]
            self.h[ax] = float(np.min(dx_arr))
            J_np = 1.0 / np.maximum(dx_arr, 1e-14)
            J_full = np.zeros(n + 1)
            J_full[0], J_full[-1] = J_np[0], J_np[-1]
            J_full[1:-1] = 0.5 * (J_np[:-1] + J_np[1:])
            self.J[ax] = xp.asarray(J_full)
            dJ = np.zeros(n + 1)
            dJ[1:-1] = (J_full[2:] - J_full[:-2]) / 2.0
            dJ[0], dJ[-1] = J_full[1]-J_full[0], J_full[-1]-J_full[-2]
            self.dJ_dxi[ax] = xp.asarray(dJ)

    def _extract_axis_profile(self, data, axis):
        xp = self.backend.xp
        slices = []
        for ax in range(self.ndim):
            slices.append(slice(None) if ax == axis else data.shape[ax]//2)
        return data[tuple(slices)]

    @staticmethod
    def _density_func(phi_1d, epsilon, alpha):
        import numpy as np
        delta = np.zeros_like(phi_1d)
        mask = np.abs(phi_1d) <= epsilon
        delta[mask] = (0.5/epsilon)*(1.0+np.cos(math.pi*phi_1d[mask]/epsilon))
        d_max = np.max(delta)
        delta_star = delta / d_max if d_max > 1e-14 else np.zeros_like(delta)
        return 1.0 + (alpha - 1.0) * delta_star
