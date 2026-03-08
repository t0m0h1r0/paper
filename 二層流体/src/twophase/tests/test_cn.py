#!/usr/bin/env python3
"""
Crank-Nicolson 実装の検証テスト

1. Helmholtz 行列の正確性（残差・対角優位性）
2. CN vs Explicit 整合性（小 dt で一致）
3. CN 2次精度収束
4. CN 安定性の優位性（粘性 CFL 超過テスト）
5. 全結合シミュレーション（CN モード）
6. 3D CN テスト
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twophase.backend import Backend
from twophase.config import SimulationConfig
from twophase.core.grid import Grid
from twophase.core.field import ScalarField, VectorField
from twophase.ccd.ccd_solver import CCDSolver
from twophase.ns_terms.predictor import Predictor
from twophase.ns_terms.helmholtz import HelmholtzSolver
from twophase.levelset.heaviside import update_properties
from twophase.simulation import TwoPhaseSimulation

backend = Backend(use_gpu=False)
xp = backend.xp


def make_taylor_green_state(N, config, backend):
    """Taylor-Green 渦の state を構築するヘルパー"""
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)
    X, Y = grid.meshgrid()
    vel = VectorField(grid, backend, "v")
    vel.u.data[:] = xp.sin(X) * xp.cos(Y)
    vel.v.data[:] = -xp.cos(X) * xp.sin(Y)
    rho = ScalarField(grid, backend, "r"); rho.data[:] = 1.0
    mu = ScalarField(grid, backend, "m"); mu.data[:] = 1.0
    phi = ScalarField(grid, backend, "p"); phi.data[:] = 100.0
    kappa = ScalarField(grid, backend, "k"); kappa.data[:] = 0.0
    state = {'velocity': vel, 'phi': phi, 'rho': rho, 'mu': mu,
             'kappa': kappa, 'ccd': ccd, 'config': config,
             'epsilon': 1.5 * grid.dx_min}
    return grid, ccd, state


def test_helmholtz_matrix():
    """Test 1: Helmholtz 行列 [I - σ∇²] の正確性"""
    print("=" * 60)
    print("Test 1: Helmholtz matrix — residual + diagonal dominance")
    print("=" * 60)

    N = 32
    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        Re=1.0, bicgstab_tol=1e-12, bicgstab_maxiter=2000, cn_viscous=True)
    grid = Grid(config, backend)

    mu = ScalarField(grid, backend, "mu"); mu.data[:] = 1.0
    rho = ScalarField(grid, backend, "rho"); rho.data[:] = 1.0
    dt = 0.01

    helmholtz = HelmholtzSolver(grid, backend, config)
    helmholtz.update_matrix(mu, rho, dt)

    # 既知の解ベクトルから残差を確認
    X, Y = grid.meshgrid()
    p_field = xp.sin(np.pi * X) * xp.sin(np.pi * Y)
    p_vec = xp.zeros(helmholtz.n_unknowns)
    Nx, Ny = grid.shape
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            p_vec[helmholtz._ijk_to_row((i, j))] = p_field[i, j]

    b = helmholtz.matrix @ p_vec
    from scipy.sparse.linalg import bicgstab
    x, info = bicgstab(helmholtz.matrix, b, rtol=1e-12, maxiter=2000)
    res_norm = float(np.max(np.abs(helmholtz.matrix @ x - b)))
    sol_err = float(np.max(np.abs(x - p_vec)))

    # 対角優位性
    A_dense = helmholtz.matrix.toarray()
    diag_vals = np.abs(np.diag(A_dense))
    off_diag_sum = np.sum(np.abs(A_dense), axis=1) - diag_vals
    dominance = float(np.min(diag_vals - off_diag_sum))

    print(f"  Residual:           {res_norm:.4e}")
    print(f"  Solution error:     {sol_err:.4e}")
    print(f"  Min diag dominance: {dominance:.4f} (>0 = dominant)")
    print(f"  Min diagonal:       {float(np.min(diag_vals)):.4f} (>1 expected)")

    assert dominance > 0, "Matrix not diagonally dominant"
    assert float(np.min(diag_vals)) > 1.0, "Diagonal should be > 1"
    assert res_norm < 1e-8
    print("✓ Helmholtz matrix PASSED\n")
    return True


def test_helmholtz_variable_coeff():
    """Test 1b: 変係数 Helmholtz（二相流の密度・粘性変化に対応）"""
    print("=" * 60)
    print("Test 1b: Helmholtz with variable σ(x)")
    print("=" * 60)

    N = 32
    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        Re=1.0, bicgstab_tol=1e-12, bicgstab_maxiter=2000, cn_viscous=True)
    grid = Grid(config, backend)

    # 二相流を模擬: ρ, μ が空間変化
    X, Y = grid.meshgrid()
    mu = ScalarField(grid, backend, "mu")
    mu.data[:] = 0.1 + 0.9 * (0.5 + 0.5 * xp.tanh(20 * (X - 0.5)))
    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 0.01 + 0.99 * (0.5 + 0.5 * xp.tanh(20 * (X - 0.5)))
    dt = 0.01

    helmholtz = HelmholtzSolver(grid, backend, config)
    helmholtz.update_matrix(mu, rho, dt)

    # 行列が構築されたことを確認
    assert helmholtz.matrix is not None
    A_dense = helmholtz.matrix.toarray()
    diag_vals = np.abs(np.diag(A_dense))
    assert float(np.min(diag_vals)) > 1.0, "Variable coeff: diagonal should be > 1"

    # solve_component テスト: 適当な RHS を与えて解けるか
    field = ScalarField(grid, backend, "test")
    field.data[:] = xp.sin(2 * np.pi * X) * xp.sin(2 * np.pi * Y)
    helmholtz.solve_component(field)
    assert not np.any(np.isnan(field.data))
    print(f"  Variable coeff solve: no NaN, max|u*| = {float(xp.max(xp.abs(field.data))):.4e}")
    print("✓ Variable coefficient Helmholtz PASSED\n")
    return True


def test_cn_explicit_consistency():
    """Test 2: CN と Explicit が小 dt で一致"""
    print("=" * 60)
    print("Test 2: CN vs Explicit consistency (dt=1e-5)")
    print("=" * 60)

    N = 32
    dt_test = 1e-5
    results = {}
    for cn_flag, label in [(False, "exp"), (True, "cn")]:
        config = SimulationConfig(
            ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi), use_gpu=False,
            Re=1.0, Fr=1e10, We=1e10, cn_viscous=cn_flag,
            bicgstab_tol=1e-14, bicgstab_maxiter=2000)
        grid, ccd, state = make_taylor_green_state(N, config, backend)
        vel_star = VectorField(grid, backend, "vs")
        Predictor(grid, backend, config).compute(state, vel_star, dt_test)
        results[label] = vel_star.u.data.copy()

    inner = slice(3, -3)
    diff = float(xp.max(xp.abs(results["cn"][inner, inner] - results["exp"][inner, inner])))
    scale = float(xp.max(xp.abs(results["exp"][inner, inner])))
    rel = diff / scale

    print(f"  Absolute diff: {diff:.4e}")
    print(f"  Relative diff: {rel:.4e}")
    assert rel < 1e-4, f"Too large: {rel}"
    print("✓ CN/Explicit consistency PASSED\n")
    return True


def test_cn_convergence_order():
    """Test 3: CN の2次精度収束"""
    print("=" * 60)
    print("Test 3: CN second-order convergence in dt")
    print("=" * 60)

    N = 32
    dts = [0.04, 0.02, 0.01, 0.005]
    solutions = []
    inner = slice(3, -3)

    for dt in dts:
        config = SimulationConfig(
            ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi), use_gpu=False,
            Re=1.0, Fr=1e10, We=1e10, cn_viscous=True,
            bicgstab_tol=1e-14, bicgstab_maxiter=2000)
        grid, ccd, state = make_taylor_green_state(N, config, backend)
        vel_star = VectorField(grid, backend, "vs")
        Predictor(grid, backend, config).compute(state, vel_star, dt)
        solutions.append(vel_star.u.data[inner, inner].copy())

    # 最小 dt を参照解として使用
    ref = solutions[-1]
    errs = [float(xp.max(xp.abs(s - ref))) for s in solutions[:-1]]

    print(f"  {'dt':>10} {'err':>14} {'order':>8}")
    print("  " + "-" * 36)
    for i, dt in enumerate(dts[:-1]):
        order = "--" if i == 0 else f"{np.log2(errs[i-1]/errs[i]):.2f}"
        print(f"  {dt:>10.4f} {errs[i]:>14.6e} {order:>8}")

    if len(errs) >= 2 and errs[-1] > 1e-15:
        final_order = np.log2(errs[-2] / errs[-1])
        print(f"\n  Final order: {final_order:.2f} (expected ~2.0)")
        assert final_order > 1.5, f"CN order {final_order:.2f} < 1.5"
    print("✓ CN convergence PASSED\n")
    return True


def test_cn_stability_advantage():
    """Test 4: CN が粘性 CFL 超過でも安定"""
    print("=" * 60)
    print("Test 4: CN stability beyond viscous CFL limit")
    print("=" * 60)

    N = 16
    Re_low = 0.1  # 高粘性
    h = 2 * np.pi / N
    rho_min = 1.0
    mu_max = 1.0
    dt_visc = rho_min * h * h / mu_max  # 粘性 CFL 限界
    dt_test = dt_visc * 5.0  # CFL の 5 倍
    print(f"  Re={Re_low}, h={h:.4f}")
    print(f"  Viscous CFL dt = {dt_visc:.4e}")
    print(f"  Test dt = {dt_test:.4e} ({dt_test/dt_visc:.1f}x CFL)")

    # Explicit: CFL 超過で発散するか確認
    config_exp = SimulationConfig(
        ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi), use_gpu=False,
        Re=Re_low, Fr=1e10, We=1e10, cn_viscous=False,
        bicgstab_tol=1e-10, bicgstab_maxiter=1000)
    grid, ccd, state_exp = make_taylor_green_state(N, config_exp, backend)
    vel_star_exp = VectorField(grid, backend, "vs_exp")

    pred_exp = Predictor(grid, backend, config_exp)
    pred_exp.compute(state_exp, vel_star_exp, dt_test)
    max_u_exp = float(xp.max(xp.abs(vel_star_exp.u.data)))

    # CN: 同じ条件で安定であるべき
    config_cn = SimulationConfig(
        ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi), use_gpu=False,
        Re=Re_low, Fr=1e10, We=1e10, cn_viscous=True,
        bicgstab_tol=1e-10, bicgstab_maxiter=1000)
    grid2, ccd2, state_cn = make_taylor_green_state(N, config_cn, backend)
    vel_star_cn = VectorField(grid2, backend, "vs_cn")

    pred_cn = Predictor(grid2, backend, config_cn)
    pred_cn.compute(state_cn, vel_star_cn, dt_test)
    max_u_cn = float(xp.max(xp.abs(vel_star_cn.u.data)))
    has_nan_cn = bool(xp.any(xp.isnan(vel_star_cn.u.data)))

    print(f"  Explicit max|u*|: {max_u_exp:.4e}")
    print(f"  CN max|u*|:       {max_u_cn:.4e}")
    print(f"  CN NaN:           {has_nan_cn}")
    print(f"  CN/Explicit ratio: {max_u_cn / max(max_u_exp, 1e-30):.4e}")

    # CN は安定（NaN なし、値が有限）
    assert not has_nan_cn, "CN should not produce NaN"
    assert max_u_cn < 10.0, f"CN u* too large: {max_u_cn}"
    # Explicit は CFL 超過で大きな値になりうる
    assert max_u_cn < max_u_exp or max_u_exp < 10.0, "CN should be more stable"
    print("✓ CN stability advantage PASSED\n")
    return True


def test_cn_full_simulation():
    """Test 5: CN モードでの全結合シミュレーション"""
    print("=" * 60)
    print("Test 5: Full simulation with CN (2D bubble, 5 steps)")
    print("=" * 60)

    config = SimulationConfig(
        ndim=2, N=(32, 32), L=(1.0, 1.0), use_gpu=False,
        Re=100.0, Fr=1.0, We=10.0,
        rho_ratio=0.1, mu_ratio=0.1,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=3, alpha_grid=1.0,
        bicgstab_tol=1e-8, bicgstab_maxiter=500,
        cn_viscous=True)

    sim = TwoPhaseSimulation(config)
    X, Y = sim.grid.meshgrid()
    sim.phi.data[:] = xp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) - 0.2
    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio, sim.epsilon, xp)
    sim.monitor.check_volume(sim.phi, sim.epsilon)

    for step in range(5):
        dt = sim.step_forward()
        has_nan = bool(
            xp.any(xp.isnan(sim.velocity.u.data)) or
            xp.any(xp.isnan(sim.p.data)))
        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        max_u = float(xp.max(xp.abs(sim.velocity.u.data)))
        max_v = float(xp.max(xp.abs(sim.velocity.v.data)))
        print(f"  Step {step+1}: dt={dt:.4e}  |u|={max_u:.4e}  "
              f"|v|={max_v:.4e}  vol={vol_err:.3f}%  NaN={has_nan}")
        assert not has_nan, f"NaN at step {step+1}"

    print("✓ CN full simulation PASSED\n")
    return True


def test_cn_3d():
    """Test 6: 3D CN テスト"""
    print("=" * 60)
    print("Test 6: 3D CN simulation (3 steps)")
    print("=" * 60)

    config = SimulationConfig(
        ndim=3, N=(12, 12, 12), L=(1.0, 1.0, 1.0), use_gpu=False,
        Re=10.0, Fr=1.0, We=100.0,
        rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=2, alpha_grid=1.0,
        bicgstab_tol=1e-6, bicgstab_maxiter=300,
        cn_viscous=True)

    sim = TwoPhaseSimulation(config)
    X, Y, Z = sim.grid.meshgrid()
    sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2) - 0.2
    update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                      config.rho_ratio, config.mu_ratio, sim.epsilon, xp)
    sim.monitor.check_volume(sim.phi, sim.epsilon)

    for step in range(3):
        dt = sim.step_forward()
        has_nan = bool(xp.any(xp.isnan(sim.velocity[0].data)) or
                       xp.any(xp.isnan(sim.p.data)))
        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        print(f"  Step {step+1}: dt={dt:.4e}  vol_err={vol_err:.3f}%  NaN={has_nan}")
        assert not has_nan, f"NaN at step {step+1}"

    print("✓ 3D CN simulation PASSED\n")
    return True


def test_cn_explicit_switch():
    """Test 7: cn_viscous フラグでの切り替え確認"""
    print("=" * 60)
    print("Test 7: cn_viscous flag switching")
    print("=" * 60)

    N = 32
    common = dict(ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        Re=100.0, Fr=1.0, We=10.0, rho_ratio=0.5, mu_ratio=0.5,
        cfl_number=0.1, t_end=1e10,
        reinit_steps=2, alpha_grid=1.0, bicgstab_tol=1e-8,
        bicgstab_maxiter=500)

    for cn_flag, label in [(False, "Explicit"), (True, "CN")]:
        config = SimulationConfig(cn_viscous=cn_flag, **common)
        sim = TwoPhaseSimulation(config)
        X, Y = sim.grid.meshgrid()
        sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2) - 0.2
        update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                          config.rho_ratio, config.mu_ratio, sim.epsilon, xp)
        sim.monitor.check_volume(sim.phi, sim.epsilon)
        for _ in range(3):
            sim.step_forward()
        has_nan = bool(xp.any(xp.isnan(sim.velocity.u.data)))
        max_u = float(xp.max(xp.abs(sim.velocity.u.data)))
        print(f"  {label:>10}: {sim.step} steps, max|u|={max_u:.4e}, NaN={has_nan}")
        assert not has_nan, f"{label} NaN"

    print("✓ CN/Explicit switch PASSED\n")
    return True


if __name__ == "__main__":
    tests = [
        test_helmholtz_matrix,
        test_helmholtz_variable_coeff,
        test_cn_explicit_consistency,
        test_cn_convergence_order,
        test_cn_stability_advantage,
        test_cn_full_simulation,
        test_cn_3d,
        test_cn_explicit_switch,
    ]

    all_pass = True
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"✗ {fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("Crank-Nicolson 全テスト PASSED ✓")
    else:
        print("一部テスト FAILED ✗")
    print("=" * 60)
