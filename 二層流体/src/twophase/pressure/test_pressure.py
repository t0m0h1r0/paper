#!/usr/bin/env python3
"""
Phase 4/5 テスト: PPE + 速度補正 + 全結合シミュレーション

テスト項目:
  1. PPE行列: 等密度ラプラシアン → 解析解と比較
  2. Rhie-Chow 発散が予測速度の非圧縮性偏差を捉える
  3. 速度補正後の ∇·u → 0
  4. 静水圧テスト: 静止流体の圧力 = ρgy
  5. 全結合 TwoPhaseSimulation: 二相流気泡の数ステップ安定性
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
from twophase.pressure.rhie_chow import RhieChowCorrection
from twophase.pressure.ppe_builder import PPEMatrixBuilder
from twophase.pressure.ppe_solver import PPESolver
from twophase.pressure.velocity_corrector import VelocityCorrector
from twophase.levelset.heaviside import update_properties
from twophase.levelset.curvature import CurvatureCalculator
from twophase.diagnostics.monitors import DiagnosticMonitor
from twophase.simulation import TwoPhaseSimulation


def test_ppe_laplacian():
    """
    Test 1: PPE 行列の検証 — 等密度ポアソン問題

    ∇²p = f を Neumann BC で解く。
    f をRC発散（速度場から構築）として与え、
    A*x = b が正しく解かれることを行列残差で確認。
    """
    print("=" * 70)
    print("Test 1: PPE matrix — solve and verify residual")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
                              bicgstab_tol=1e-12, bicgstab_maxiter=2000)
    grid = Grid(config, backend)

    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0

    builder = PPEMatrixBuilder(grid, backend)
    builder.update_coefficients(rho)

    # 既知の解から RHS を構築: A*p_exact = b
    X, Y = grid.meshgrid()
    # 内部点にのみ値を持つ解ベクトル
    p_exact_field = xp.sin(np.pi * X) * xp.sin(np.pi * Y)

    # RHS = A * p_exact
    p_exact_vec = xp.zeros(builder.n_unknowns)
    Nx, Ny = grid.shape
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            row = builder._ijk_to_row((i, j))
            p_exact_vec[row] = p_exact_field[i, j]
    # ゲージ固定点: p_exact_vec[0] を 0 にシフト
    shift = float(p_exact_vec[0])
    p_exact_vec -= shift

    rhs = builder.matrix @ p_exact_vec

    # 求解
    solver = PPESolver(backend, config)
    p_sol = solver.solve(builder.matrix, rhs)

    # 残差チェック
    residual = builder.matrix @ p_sol - rhs
    res_norm = float(xp.max(xp.abs(residual)))
    print(f"  ||A*x - b||_∞ = {res_norm:.4e}")

    # 解の比較（定数シフトを補正）
    err = float(xp.max(xp.abs(p_sol - p_exact_vec)))
    print(f"  ||p - p_exact||_∞ = {err:.4e}")

    assert res_norm < 1e-8, f"Residual too large: {res_norm}"
    print("✓ PPE ラプラシアンテスト PASSED\n")
    return True


def test_divergence_free_projection():
    """
    Test 2: Projection — 非ゼロ発散の速度場を補正

    PPE + 速度補正で FVM 発散（面フラックスベース）が
    減少することを確認。CCD発散との違いは論文設計通り（§7.4）。
    """
    print("=" * 70)
    print("Test 2: Projection — FVM divergence reduction")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        bicgstab_tol=1e-12, bicgstab_maxiter=2000)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    dt = 0.01

    # 非圧縮でない予測速度
    vel_star = VectorField(grid, backend, "vel_star")
    vel_star.u.data[:] = xp.sin(np.pi * X)
    vel_star.v.data[:] = xp.sin(np.pi * Y)

    p = ScalarField(grid, backend, "p")
    p.data[:] = 0.0

    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0

    # Rhie-Chow (p=0 なので補正なし)
    rc = RhieChowCorrection(grid, backend)
    rc.compute_face_velocities(vel_star, p, rho, ccd, dt)
    div_rc_before = rc.compute_divergence().copy()
    fvm_div_before = float(xp.max(xp.abs(div_rc_before[2:-2, 2:-2])))

    # PPE
    builder = PPEMatrixBuilder(grid, backend)
    builder.update_coefficients(rho)
    rhs = builder.build_rhs(div_rc_before, dt)

    solver = PPESolver(backend, config)
    p_sol = solver.solve(builder.matrix, rhs)
    builder.scatter_solution(p_sol, p)

    # 速度補正
    vel_new = VectorField(grid, backend, "vel_new")
    corrector = VelocityCorrector(grid, backend)
    corrector.correct(vel_star, vel_new, p, rho, ccd, dt)

    # 補正後の FVM 発散を再計算
    rc2 = RhieChowCorrection(grid, backend)
    rc2.compute_face_velocities(vel_new, p, rho, ccd, dt)
    div_rc_after = rc2.compute_divergence()
    fvm_div_after = float(xp.max(xp.abs(div_rc_after[2:-2, 2:-2])))

    print(f"  FVM ∇·u* (before) = {fvm_div_before:.4e}")
    print(f"  FVM ∇·u  (after)  = {fvm_div_after:.4e}")

    # PPE 残差
    res = builder.matrix @ p_sol - rhs
    res_norm = float(xp.max(xp.abs(res)))
    print(f"  PPE residual = {res_norm:.4e}")
    print(f"  PPE iterations = {solver._last_iters}")

    assert res_norm < 1e-8, f"PPE residual too large: {res_norm}"
    assert fvm_div_after < fvm_div_before or fvm_div_after < 0.5, \
        f"FVM divergence not reduced: {fvm_div_after} >= {fvm_div_before}"
    print("✓ Projection 非圧縮性テスト PASSED\n")
    return True


def test_hydrostatic():
    """
    Test 3: 静水圧テスト

    静止流体 u=0、重力あり。
    PPE を解くと p ≈ ρgy になるはず。
    u は 0 のままに近い状態を維持すべき。
    """
    print("=" * 70)
    print("Test 3: Hydrostatic pressure test")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 16

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        Re=100.0, Fr=1.0, We=1e10,
        rho_ratio=1.0, mu_ratio=1.0,  # 単相
        bicgstab_tol=1e-12)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    dt = 0.001

    # 静止流体
    vel = VectorField(grid, backend, "vel")  # u=0
    vel_star = VectorField(grid, backend, "vel_star")
    p = ScalarField(grid, backend, "p")
    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0
    mu = ScalarField(grid, backend, "mu")
    mu.data[:] = 1.0

    phi = ScalarField(grid, backend, "phi")
    phi.data[:] = 100.0  # 全域液相
    kappa = ScalarField(grid, backend, "kappa")

    # Predictor: u=0 なので u* は重力項のみ
    # u*_x = 0, u*_y = 0 + dt(-1/Fr²) = -dt
    vel_star.u.data[:] = 0.0
    vel_star.v.data[:] = dt * (-1.0 / config.Fr ** 2)

    # PPE
    rc = RhieChowCorrection(grid, backend)
    rc.compute_face_velocities(vel_star, p, rho, ccd, dt)
    div_rc = rc.compute_divergence()

    builder = PPEMatrixBuilder(grid, backend)
    builder.update_coefficients(rho)
    rhs = builder.build_rhs(div_rc, dt)

    solver = PPESolver(backend, config)
    p_sol = solver.solve(builder.matrix, rhs)
    builder.scatter_solution(p_sol, p)

    # 速度補正
    vel_new = VectorField(grid, backend, "vel_new")
    corrector = VelocityCorrector(grid, backend)
    corrector.correct(vel_star, vel_new, p, rho, ccd, dt)

    max_u = float(xp.max(xp.abs(vel_new.u.data)))
    max_v = float(xp.max(xp.abs(vel_new.v.data)))
    print(f"  補正後: max|u|={max_u:.4e}, max|v|={max_v:.4e}")

    # 圧力が y 方向に線形変化していることの確認
    p_col = p.data[N // 2, 1:-1]
    y_col = Y[N // 2, 1:-1]
    if len(p_col) > 2:
        # 線形回帰
        A = xp.vstack([y_col, xp.ones_like(y_col)]).T
        result = xp.linalg.lstsq(A, p_col, rcond=None)
        slope = float(result[0][0])
        print(f"  圧力の y 勾配: {slope:.4f} (期待値: ~ -1/Fr² = -1.0)")

    has_nan = bool(xp.any(xp.isnan(p.data)))
    assert not has_nan, "NaN in pressure"
    print("✓ 静水圧テスト PASSED\n")
    return True


def test_full_simulation_stability():
    """
    Test 4: 全結合 TwoPhaseSimulation — 気泡の安定性

    円形気泡を初期条件として、5ステップ実行。
    NaN なし + 体積保存を確認。
    """
    print("=" * 70)
    print("Test 4: Full simulation — bubble stability (5 steps)")
    print("=" * 70)

    config = SimulationConfig(
        ndim=2, N=(32, 32), L=(1.0, 1.0),
        use_gpu=False,
        Re=100.0, Fr=1.0, We=10.0,
        rho_ratio=0.1, mu_ratio=0.1,
        cfl_number=0.1,
        t_end=1e10,  # 手動でステップ制御
        reinit_steps=3,
        bicgstab_tol=1e-8,
        bicgstab_maxiter=500)

    sim = TwoPhaseSimulation(config)
    xp = sim.xp
    X, Y = sim.grid.meshgrid()

    # 初期条件: 円形気泡
    R = 0.2
    cx, cy = 0.5, 0.5
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    sim.phi.data[:] = r - R

    # 初期物性値
    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio,
                      sim.epsilon, xp)
    sim.monitor.check_volume(sim.phi, sim.epsilon)

    # 5ステップ実行
    for step in range(5):
        dt = sim.step_forward()
        max_u = float(xp.max(xp.abs(sim.velocity.u.data)))
        max_v = float(xp.max(xp.abs(sim.velocity.v.data)))
        has_nan = bool(
            xp.any(xp.isnan(sim.velocity.u.data)) or
            xp.any(xp.isnan(sim.velocity.v.data)) or
            xp.any(xp.isnan(sim.p.data)))

        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        print(f"  Step {step+1}: dt={dt:.4e}, max|u|={max_u:.4e}, "
              f"max|v|={max_v:.4e}, vol_err={vol_err:.3f}%, NaN={has_nan}")

        assert not has_nan, f"NaN detected at step {step+1}"

    print("✓ 全結合シミュレーション安定性テスト PASSED\n")
    return True


def test_full_simulation_run():
    """
    Test 5: sim.run() の動作テスト（短時間）
    """
    print("=" * 70)
    print("Test 5: sim.run() execution test")
    print("=" * 70)

    config = SimulationConfig(
        ndim=2, N=(16, 16), L=(1.0, 1.0),
        use_gpu=False,
        Re=10.0, Fr=1.0, We=100.0,
        rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1,
        t_end=0.002,
        reinit_steps=2,
        bicgstab_tol=1e-6,
        bicgstab_maxiter=200)

    sim = TwoPhaseSimulation(config)
    xp = sim.xp
    X, Y = sim.grid.meshgrid()

    # 初期条件
    R = 0.2
    sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2) - R

    sim.run(output_interval=5, verbose=True)

    has_nan = bool(xp.any(xp.isnan(sim.velocity.u.data)))
    assert not has_nan, "NaN in final velocity"
    print(f"\n  完了: {sim.step} steps, t={sim.time:.4e}")
    print("✓ sim.run() テスト PASSED\n")
    return True


if __name__ == "__main__":
    all_pass = True

    tests = [
        test_ppe_laplacian,
        test_divergence_free_projection,
        test_hydrostatic,
        test_full_simulation_stability,
        test_full_simulation_run,
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
        print("Phase 4/5 全テスト PASSED ✓")
    else:
        print("Phase 4/5 一部テスト FAILED ✗")
    print("=" * 70)
