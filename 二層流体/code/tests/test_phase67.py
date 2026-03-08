#!/usr/bin/env python3
"""
Phase 6/7 テスト: 界面適合格子 + 3D検証 + 可視化

テスト項目:
  1. 界面適合格子: 界面近傍の格子密度が α 倍に増加
  2. 3D CCD 精度: sin(z) の微分が O(h^6)
  3. 3D 球の曲率: κ = -2/R
  4. 3D シミュレーション: 気泡の安定性
  5. 2D 可視化出力テスト
  6. 3D 可視化出力テスト
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twophase.backend import Backend
from twophase.config import SimulationConfig
from twophase.core.grid import Grid
from twophase.core.field import ScalarField, VectorField
from twophase.ccd.ccd_solver import CCDSolver
from twophase.levelset.curvature import CurvatureCalculator
from twophase.levelset.heaviside import update_properties
from twophase.simulation import TwoPhaseSimulation
from twophase.visualization.visualizer import Visualizer


def test_adaptive_grid():
    """
    Test 1: 界面適合格子 — 界面近傍の格子密度が増加
    """
    print("=" * 70)
    print("Test 1: Adaptive grid — interface refinement")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        alpha_grid=4.0, epsilon_factor=1.5)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    phi = ScalarField(grid, backend, "phi")
    R = 0.25
    phi.data[:] = xp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - R

    # 初期格子幅（等間隔）
    dx_before = grid.dx_min
    print(f"  等間隔格子: dx_min = {dx_before:.6f}")

    # 適合格子更新
    grid.update_from_levelset(phi, ccd, alpha=4.0)

    dx_after = grid.dx_min
    print(f"  適合格子:   dx_min = {dx_after:.6f}")
    print(f"  細分化率:   {dx_before / dx_after:.2f}x")

    # 界面近傍が細かく、遠方が粗くなっていることを確認
    x_coords = np.asarray(backend.to_host(grid.coords[0]))
    dx_arr = x_coords[1:] - x_coords[:-1]
    dx_max = float(np.max(dx_arr))
    print(f"  dx_max = {dx_max:.6f}, dx_min/dx_max = {dx_after/dx_max:.3f}")

    assert dx_after < dx_before, "Grid not refined"
    assert dx_after / dx_max < 0.5, "Insufficient refinement contrast"
    print("✓ 界面適合格子テスト PASSED\n")
    return True


def test_3d_ccd():
    """
    Test 2: 3D CCD — sin(z) の z方向微分
    """
    print("=" * 70)
    print("Test 2: 3D CCD — differentiation along z-axis")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(
        ndim=3, N=(4, 4, N), L=(1.0, 1.0, np.pi), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    z = grid.coords[2]
    # f(x,y,z) = sin(z)（x,y 非依存）
    f = xp.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            f[i, j, :] = xp.sin(z)

    d1z, d2z = ccd.differentiate(f, axis=2)

    exact_d1z = xp.cos(z)
    err = float(xp.max(xp.abs(d1z[2, 2, :] - exact_d1z)))
    print(f"  N={N}, max|df/dz - cos(z)| = {err:.4e}")
    assert err < 1e-4, f"3D CCD z-error too large: {err}"
    print("✓ 3D CCD テスト PASSED\n")
    return True


def test_3d_sphere_curvature():
    """
    Test 3: 3D 球の曲率 κ = -2/R
    """
    print("=" * 70)
    print("Test 3: 3D sphere curvature (κ = -2/R)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(
        ndim=3, N=(N, N, N), L=(1.0, 1.0, 1.0), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y, Z = grid.meshgrid()
    R = 0.25
    cx, cy, cz = 0.5, 0.5, 0.5
    r = xp.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)

    phi = ScalarField(grid, backend, "phi")
    phi.data[:] = r - R

    kappa = ScalarField(grid, backend, "kappa")
    curv_calc = CurvatureCalculator(grid, backend)
    curv_calc.compute(phi, kappa, ccd)

    exact_kappa = -2.0 / R  # 球の平均曲率

    dx = grid.dx_min
    eps = 3.0 * dx
    mask = xp.abs(phi.data) < eps
    if xp.any(mask):
        k_vals = kappa.data[mask]
        mean_k = float(xp.mean(k_vals))
        max_err = float(xp.max(xp.abs(k_vals - exact_kappa)))
        print(f"  κ_mean = {mean_k:.4f} (exact = {exact_kappa:.1f})")
        print(f"  max|κ - κ_exact| = {max_err:.4e}")
        assert abs(mean_k - exact_kappa) < 1.0, \
            f"Mean curvature wrong: {mean_k} vs {exact_kappa}"

    print("✓ 3D 球曲率テスト PASSED\n")
    return True


def test_3d_simulation():
    """
    Test 4: 3D 気泡シミュレーション — 安定性テスト
    """
    print("=" * 70)
    print("Test 4: 3D simulation — bubble stability (3 steps)")
    print("=" * 70)

    config = SimulationConfig(
        ndim=3, N=(16, 16, 16), L=(1.0, 1.0, 1.0),
        use_gpu=False,
        Re=10.0, Fr=1.0, We=100.0,
        rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=2, alpha_grid=1.0,  # 適合格子なし（速度重視）
        bicgstab_tol=1e-6, bicgstab_maxiter=300)

    sim = TwoPhaseSimulation(config)
    xp = sim.xp
    X, Y, Z = sim.grid.meshgrid()

    R = 0.2
    r = xp.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2)
    sim.phi.data[:] = r - R

    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio, sim.epsilon, xp)
    sim.monitor.check_volume(sim.phi, sim.epsilon)

    for step in range(3):
        dt = sim.step_forward()
        has_nan = bool(xp.any(xp.isnan(sim.velocity[0].data)) or
                       xp.any(xp.isnan(sim.p.data)))
        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        print(f"  Step {step+1}: dt={dt:.4e}, vol_err={vol_err:.3f}%, NaN={has_nan}")
        assert not has_nan, f"NaN at step {step+1}"

    print("✓ 3D シミュレーション安定性テスト PASSED\n")
    return True


def test_visualization_2d():
    """
    Test 5: 2D 可視化出力
    """
    print("=" * 70)
    print("Test 5: 2D visualization output")
    print("=" * 70)

    config = SimulationConfig(
        ndim=2, N=(32, 32), L=(1.0, 1.0), use_gpu=False,
        Re=10.0, Fr=1.0, We=100.0,
        rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=2, alpha_grid=1.0,
        bicgstab_tol=1e-6, bicgstab_maxiter=200)

    sim = TwoPhaseSimulation(config)
    xp = sim.xp
    X, Y = sim.grid.meshgrid()
    sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2) - 0.2

    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio, sim.epsilon, xp)

    # 曲率計算
    from twophase.levelset.curvature import CurvatureCalculator
    curv = CurvatureCalculator(sim.grid, sim.backend)
    curv.compute(sim.phi, sim.kappa, sim.ccd)

    # 2ステップ走らせてから可視化
    for _ in range(2):
        sim.step_forward()

    vis = Visualizer(sim, output_dir="/home/claude/test_output")
    vis.snapshot("test_2d_snapshot.png")
    vis.plot_interface("test_2d_interface.png")
    vis.save_fields_npz("test_2d_fields.npz")

    assert os.path.exists("/home/claude/test_output/test_2d_snapshot.png")
    assert os.path.exists("/home/claude/test_output/test_2d_fields.npz")
    print("✓ 2D 可視化テスト PASSED\n")
    return True


def test_visualization_3d():
    """
    Test 6: 3D 可視化出力
    """
    print("=" * 70)
    print("Test 6: 3D visualization output")
    print("=" * 70)

    config = SimulationConfig(
        ndim=3, N=(16, 16, 16), L=(1.0, 1.0, 1.0), use_gpu=False,
        Re=10.0, Fr=1.0, We=100.0,
        rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=2, alpha_grid=1.0,
        bicgstab_tol=1e-6, bicgstab_maxiter=200)

    sim = TwoPhaseSimulation(config)
    xp = sim.xp
    X, Y, Z = sim.grid.meshgrid()
    sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2) - 0.2

    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio, sim.epsilon, xp)

    vis = Visualizer(sim, output_dir="/home/claude/test_output")
    vis.snapshot("test_3d_snapshot.png")
    vis.save_fields_npz("test_3d_fields.npz")

    assert os.path.exists("/home/claude/test_output/test_3d_snapshot.png")
    assert os.path.exists("/home/claude/test_output/test_3d_fields.npz")
    print("✓ 3D 可視化テスト PASSED\n")
    return True


if __name__ == "__main__":
    all_pass = True
    tests = [
        test_adaptive_grid,
        test_3d_ccd,
        test_3d_sphere_curvature,
        test_3d_simulation,
        test_visualization_2d,
        test_visualization_3d,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("=" * 70)
    if all_pass:
        print("Phase 6/7 全テスト PASSED ✓")
    else:
        print("Phase 6/7 一部テスト FAILED ✗")
    print("=" * 70)
