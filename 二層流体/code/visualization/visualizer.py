"""
可視化モジュール（2D/3D 対応）

2D:
  - φ=0 界面輪郭 + 速度ベクトル
  - 圧力・密度のカラーマップ
  - 曲率分布
  - タイムシリーズアニメーション

3D:
  - φ=0 等値面（marching cubes）
  - 任意断面のカラーマップ
  - 速度ベクトルの断面表示
"""

from __future__ import annotations
import os
import numpy as np
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation import TwoPhaseSimulation


def _to_np(arr):
    """GPU配列をnumpyに安全変換"""
    try:
        return arr.get()
    except AttributeError:
        return np.asarray(arr)


class Visualizer:
    """
    シミュレーション可視化クラス

    sim = TwoPhaseSimulation(config)
    vis = Visualizer(sim)
    vis.snapshot("output.png")
    vis.animate(frames, "bubble.mp4")
    """

    def __init__(self, sim: TwoPhaseSimulation, output_dir: str = "output"):
        self.sim = sim
        self.ndim = sim.grid.ndim
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════
    # 2D 可視化
    # ═══════════════════════════════════════════════════

    def snapshot_2d(self, filename: Optional[str] = None,
                    show: bool = False, title: str = None):
        """
        2D スナップショット（4パネル）:
          左上: φ=0 界面 + 速度ベクトル
          右上: 圧力場
          左下: 密度場
          右下: 曲率場（界面近傍）
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        sim = self.sim
        xp = sim.xp

        X, Y = sim.grid.meshgrid()
        X_np, Y_np = _to_np(X), _to_np(Y)
        phi_np = _to_np(sim.phi.data)
        u_np = _to_np(sim.velocity.u.data)
        v_np = _to_np(sim.velocity.v.data)
        p_np = _to_np(sim.p.data)
        rho_np = _to_np(sim.rho.data)
        kappa_np = _to_np(sim.kappa.data)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if title is None:
            title = f"t = {sim.time:.4e}  (step {sim.step})"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # ─── 左上: 界面 + 速度ベクトル ───
        ax = axes[0, 0]
        speed = np.sqrt(u_np**2 + v_np**2)
        cf = ax.contourf(X_np, Y_np, speed, levels=30, cmap='coolwarm')
        plt.colorbar(cf, ax=ax, label='|u|')
        ax.contour(X_np, Y_np, phi_np, levels=[0], colors='lime',
                   linewidths=2.5)
        # 間引いてベクトル表示
        skip = max(1, min(sim.grid.N) // 16)
        ax.quiver(X_np[::skip, ::skip], Y_np[::skip, ::skip],
                  u_np[::skip, ::skip], v_np[::skip, ::skip],
                  color='white', alpha=0.7, scale_units='xy')
        ax.set_title('Interface (φ=0) + Velocity')
        ax.set_aspect('equal')

        # ─── 右上: 圧力場 ───
        ax = axes[0, 1]
        cf = ax.contourf(X_np, Y_np, p_np, levels=30, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax, label='p')
        ax.contour(X_np, Y_np, phi_np, levels=[0], colors='k',
                   linewidths=1.5)
        ax.set_title('Pressure')
        ax.set_aspect('equal')

        # ─── 左下: 密度場 ───
        ax = axes[1, 0]
        cf = ax.contourf(X_np, Y_np, rho_np, levels=30, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='ρ̃')
        ax.contour(X_np, Y_np, phi_np, levels=[0], colors='r',
                   linewidths=1.5)
        ax.set_title('Density')
        ax.set_aspect('equal')

        # ─── 右下: 曲率（界面近傍のみ） ───
        ax = axes[1, 1]
        eps = sim.epsilon
        mask = np.abs(phi_np) > 3 * eps
        kappa_masked = np.ma.array(kappa_np, mask=mask)
        cf = ax.pcolormesh(X_np, Y_np, kappa_masked, cmap='seismic',
                           shading='auto')
        plt.colorbar(cf, ax=ax, label='κ')
        ax.contour(X_np, Y_np, phi_np, levels=[0], colors='k',
                   linewidths=1.5)
        ax.set_title('Curvature (near interface)')
        ax.set_aspect('equal')

        plt.tight_layout()

        if filename:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ═══════════════════════════════════════════════════
    # 3D 可視化
    # ═══════════════════════════════════════════════════

    def snapshot_3d(self, filename: Optional[str] = None,
                    show: bool = False, title: str = None):
        """
        3D スナップショット:
          - φ=0 等値面（marching cubes）
          - 中央断面の速度大きさ
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        sim = self.sim
        phi_np = _to_np(sim.phi.data)

        fig = plt.figure(figsize=(14, 6))
        if title is None:
            title = f"3D  t = {sim.time:.4e}  (step {sim.step})"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # ─── 左: 等値面 ───
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_isosurface(ax1, phi_np, level=0.0, color='dodgerblue',
                              alpha=0.6)
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.set_title('Interface (φ=0)')

        # ─── 右: 中央 z 断面の速度 ───
        ax2 = fig.add_subplot(122)
        kz = phi_np.shape[2] // 2
        X2d = _to_np(sim.grid.coords[0])
        Y2d = _to_np(sim.grid.coords[1])
        XX, YY = np.meshgrid(X2d, Y2d, indexing='ij')
        u_slice = _to_np(sim.velocity.u.data)[:, :, kz]
        v_slice = _to_np(sim.velocity.v.data)[:, :, kz]
        speed = np.sqrt(u_slice**2 + v_slice**2)

        cf = ax2.contourf(XX, YY, speed, levels=30, cmap='coolwarm')
        plt.colorbar(cf, ax=ax2, label='|u|')
        ax2.contour(XX, YY, phi_np[:, :, kz], levels=[0],
                    colors='lime', linewidths=2)
        if np.max(np.abs(u_slice)) + np.max(np.abs(v_slice)) > 1e-14:
            skip = max(1, min(sim.grid.N[:2]) // 12)
            ax2.quiver(XX[::skip, ::skip], YY[::skip, ::skip],
                       u_slice[::skip, ::skip], v_slice[::skip, ::skip],
                       color='white', alpha=0.6)
        ax2.set_title(f'z-midplane (k={kz})')
        ax2.set_aspect('equal')

        plt.tight_layout()
        if filename:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def _plot_isosurface(self, ax, data_3d, level=0.0, color='blue', alpha=0.5):
        """marching cubes で等値面を描画"""
        from skimage.measure import marching_cubes
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        try:
            verts, faces, _, _ = marching_cubes(data_3d, level=level)
        except (ValueError, RuntimeError):
            return  # 等値面が見つからない

        # 物理座標にスケーリング
        for i, ax_i in enumerate(range(3)):
            if ax_i < self.ndim:
                L = self.sim.grid.L[ax_i]
                N = self.sim.grid.N[ax_i]
                verts[:, i] = verts[:, i] * L / N

        mesh = Poly3DCollection(verts[faces], alpha=alpha,
                                facecolor=color, edgecolor='gray',
                                linewidth=0.1)
        ax.add_collection3d(mesh)

        for i in range(3):
            L = self.sim.grid.L[i] if i < self.ndim else 1.0
            getattr(ax, f'set_{"xyz"[i]}lim')(0, L)

    # ═══════════════════════════════════════════════════
    # 統合インターフェース
    # ═══════════════════════════════════════════════════

    def snapshot(self, filename: Optional[str] = None,
                 show: bool = False, title: str = None):
        """2D/3D に応じた自動スナップショット"""
        if filename is None:
            filename = f"snap_{self.sim.step:06d}.png"
        if self.ndim == 2:
            return self.snapshot_2d(filename, show, title)
        else:
            return self.snapshot_3d(filename, show, title)

    def plot_interface(self, filename: Optional[str] = None,
                       show: bool = False):
        """界面のみのシンプルなプロット"""
        import matplotlib.pyplot as plt
        sim = self.sim
        phi_np = _to_np(sim.phi.data)

        fig, ax = plt.subplots(figsize=(6, 6))
        if self.ndim == 2:
            X, Y = sim.grid.meshgrid()
            ax.contour(_to_np(X), _to_np(Y), phi_np, levels=[0],
                       colors='blue', linewidths=2)
            ax.set_aspect('equal')
            ax.set_title(f'Interface  t={sim.time:.4e}')
            ax.grid(True, alpha=0.3)
        else:
            kz = phi_np.shape[2] // 2
            X2d, Y2d = np.meshgrid(
                _to_np(sim.grid.coords[0]),
                _to_np(sim.grid.coords[1]), indexing='ij')
            ax.contour(X2d, Y2d, phi_np[:, :, kz], levels=[0],
                       colors='blue', linewidths=2)
            ax.set_title(f'Interface z-midplane  t={sim.time:.4e}')
            ax.set_aspect('equal')

        plt.tight_layout()
        if filename:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_grid(self, ax_idx: int = 0, filename: Optional[str] = None):
        """格子点分布の可視化（界面適合格子の確認用）"""
        import matplotlib.pyplot as plt
        sim = self.sim
        coords_np = _to_np(sim.grid.coords[ax_idx])
        dx = coords_np[1:] - coords_np[:-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(coords_np, np.zeros_like(coords_np), '|', ms=10, color='navy')
        ax1.set_title(f'Grid points ({sim.grid.axis_names[ax_idx]}-axis)')
        ax1.set_xlabel(sim.grid.axis_names[ax_idx])

        ax2.plot(coords_np[:-1], dx, 'o-', ms=3, color='navy')
        ax2.set_title(f'Grid spacing Δ{sim.grid.axis_names[ax_idx]}')
        ax2.set_xlabel(sim.grid.axis_names[ax_idx])
        ax2.set_ylabel('Δx')
        ax2.axhline(np.mean(dx), color='red', ls='--', alpha=0.5, label='mean')
        ax2.legend()

        plt.tight_layout()
        if filename:
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, dpi=150)
        plt.close(fig)

    def save_fields_npz(self, filename: Optional[str] = None):
        """全場データをnpzで保存"""
        sim = self.sim
        if filename is None:
            filename = f"fields_{sim.step:06d}.npz"
        path = os.path.join(self.output_dir, filename)

        data = {
            'time': sim.time, 'step': sim.step,
            'phi': _to_np(sim.phi.data),
            'p': _to_np(sim.p.data),
            'rho': _to_np(sim.rho.data),
            'mu': _to_np(sim.mu.data),
            'kappa': _to_np(sim.kappa.data),
        }
        for ax in range(sim.grid.ndim):
            data[f'coords_{ax}'] = _to_np(sim.grid.coords[ax])
            data[f'vel_{ax}'] = _to_np(sim.velocity[ax].data)
        np.savez_compressed(path, **data)
        print(f"  Saved fields: {path}")
