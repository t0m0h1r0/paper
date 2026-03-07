#!/usr/bin/env python3
"""
Phase 3 テスト: NS方程式の各項 + Predictor

テスト項目:
  1. (a) 対流項: 一様流での自明な検証 + u·∇u の精度
  2. (b) 粘性項: sin(x)sin(y) のラプラシアンが解析解と一致
  3. (c) 重力項: 定数ベクトルの方向と値
  4. (d) 表面張力: 円形界面への CSF 力の方向と大きさ
  5. Predictor 統合: Taylor-Green 渦の初期減衰率
  6. Predictor 統合: 全項結合の安定性テスト
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
from twophase.levelset.heaviside import (
    heaviside_smooth, delta_smooth, update_properties)
from twophase.levelset.curvature import CurvatureCalculator
from twophase.ns_terms.convection import ConvectionTerm
from twophase.ns_terms.viscous import ViscousTerm
from twophase.ns_terms.gravity import GravityTerm
from twophase.ns_terms.surface_tension import SurfaceTensionTerm
from twophase.ns_terms.predictor import Predictor


def make_state(grid, backend, config, ccd,
               vel=None, phi=None, rho=None, mu=None, kappa=None):
    """テスト用の state 辞書を構築するヘルパー"""
    xp = backend.xp
    state = {
        'velocity': vel or VectorField(grid, backend, "vel"),
        'phi': phi or ScalarField(grid, backend, "phi"),
        'rho': rho or ScalarField(grid, backend, "rho"),
        'mu': mu or ScalarField(grid, backend, "mu"),
        'kappa': kappa or ScalarField(grid, backend, "kappa"),
        'ccd': ccd,
        'config': config,
        'epsilon': 1.5 * grid.dx_min,
    }
    # デフォルト: 単相（ρ̃=1, μ̃=1）
    if rho is None:
        state['rho'].data[:] = 1.0
    if mu is None:
        state['mu'].data[:] = 1.0
    return state


def test_convection_uniform():
    """
    Test 1: (a) 一様流 u=(U,0) での対流項 = 0

    一様流の場合 ∂u/∂x = ∂u/∂y = 0 なので -(u·∇)u = 0
    """
    print("=" * 70)
    print("Test 1: (a) Convection — uniform flow gives zero")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0),
                              use_gpu=False, Re=100.0)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    vel = VectorField(grid, backend, "vel")
    vel.u.data[:] = 1.0   # 一様流
    vel.v.data[:] = 0.5

    out = VectorField(grid, backend, "out")
    state = make_state(grid, backend, config, ccd, vel=vel)

    conv = ConvectionTerm(grid, backend, config)
    conv.evaluate(state, out, mode='set')

    err_u = float(xp.max(xp.abs(out.u.data)))
    err_v = float(xp.max(xp.abs(out.v.data)))

    print(f"  max|conv_u| = {err_u:.2e}")
    print(f"  max|conv_v| = {err_v:.2e}")
    assert err_u < 1e-10, f"Convection of uniform flow not zero: {err_u}"
    assert err_v < 1e-10, f"Convection of uniform flow not zero: {err_v}"
    print("✓ 一様流対流 = 0 確認\n")
    return True


def test_convection_accuracy():
    """
    Test 2: (a) 対流項 u·∇u の精度

    u = sin(x)cos(y), v = -cos(x)sin(y) （非圧縮流）
    (u·∇)u の u成分 = u·∂u/∂x + v·∂u/∂y
    = sin(x)cos(y)·cos(x)cos(y) + (-cos(x)sin(y))·(-sin(x)sin(y))
    = sin(x)cos(x)(cos²y + sin²y) = sin(x)cos(x) = sin(2x)/2
    """
    print("=" * 70)
    print("Test 2: (a) Convection accuracy (incompressible test field)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi),
                              use_gpu=False, Re=100.0)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    vel = VectorField(grid, backend, "vel")
    vel.u.data[:] = xp.sin(X) * xp.cos(Y)
    vel.v.data[:] = -xp.cos(X) * xp.sin(Y)

    out = VectorField(grid, backend, "out")
    state = make_state(grid, backend, config, ccd, vel=vel)

    conv = ConvectionTerm(grid, backend, config)
    conv.evaluate(state, out, mode='set')

    # -(u·∇)u の u成分 = -sin(2x)/2
    exact_u = -xp.sin(2.0 * X) / 2.0

    # 内部点で比較（境界は精度が落ちる）
    inner = slice(3, -3)
    err = float(xp.max(xp.abs(out.u.data[inner, inner] - exact_u[inner, inner])))
    print(f"  max|conv_u - exact| = {err:.4e}")
    assert err < 1e-4, f"Convection accuracy too low: {err}"
    print("✓ 対流項精度確認\n")
    return True


def test_viscous_laplacian():
    """
    Test 3: (b) 粘性項 — 等粘性でのラプラシアン検証

    u = sin(x)sin(y), v = 0, μ=1, ρ=1, Re=1
    ∇²u = -2sin(x)sin(y)
    粘性力 = μ∇²u/(ρRe) = -2sin(x)sin(y)
    """
    print("=" * 70)
    print("Test 3: (b) Viscous — Laplacian of sin(x)sin(y)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi),
                              use_gpu=False, Re=1.0)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    vel = VectorField(grid, backend, "vel")
    vel.u.data[:] = xp.sin(X) * xp.sin(Y)
    vel.v.data[:] = 0.0

    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0
    mu = ScalarField(grid, backend, "mu")
    mu.data[:] = 1.0

    out = VectorField(grid, backend, "out")
    state = make_state(grid, backend, config, ccd, vel=vel, rho=rho, mu=mu)

    visc = ViscousTerm(grid, backend, config)
    visc.evaluate_laplacian(state, out, mode='set')

    # 解析解: ∇²(sinx·siny) = -2sinx·siny
    exact_u = -2.0 * xp.sin(X) * xp.sin(Y)  # × μ/(ρRe) = 1

    inner = slice(3, -3)
    err = float(xp.max(xp.abs(out.u.data[inner, inner] - exact_u[inner, inner])))
    print(f"  max|visc_u - exact| = {err:.4e}")
    assert err < 1e-4, f"Viscous Laplacian error too large: {err}"
    print("✓ 粘性ラプラシアン精度確認\n")
    return True


def test_gravity_direction():
    """
    Test 4: (c) 重力項 — 方向と値の確認

    2D: (0, -1/Fr²) where Fr=1 → (0, -1)
    """
    print("=" * 70)
    print("Test 4: (c) Gravity — direction and value")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 8

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0),
                              use_gpu=False, Fr=2.0)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    out = VectorField(grid, backend, "out")
    state = make_state(grid, backend, config, ccd)

    grav = GravityTerm(grid, backend, config)
    grav.evaluate(state, out, mode='set')

    expected_val = -1.0 / (2.0 ** 2)  # -1/Fr² = -0.25

    err_x = float(xp.max(xp.abs(out.u.data)))
    mean_y = float(xp.mean(out.v.data))

    print(f"  u成分: max|g_x| = {err_x:.2e} (expected 0)")
    print(f"  v成分: mean(g_y) = {mean_y:.4f} (expected {expected_val:.4f})")

    assert err_x < 1e-14, f"Gravity x-component not zero: {err_x}"
    assert abs(mean_y - expected_val) < 1e-14, \
        f"Gravity y-component wrong: {mean_y} vs {expected_val}"
    print("✓ 重力項テスト PASSED\n")
    return True


def test_surface_tension_circle():
    """
    Test 5: (d) 表面張力 — 円形界面のCSF力

    φ = r - R（符号付き距離関数）
    κ = -1/R（凹方向）
    δε(φ): 界面近傍のみ非ゼロ
    ∇φ = (x-cx, y-cy)/r（放射方向）

    CSF力 = κδε∇φ/(ρWe) は界面上で半径方向（内向き）
    """
    print("=" * 70)
    print("Test 5: (d) Surface tension — CSF on circle")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0),
                              use_gpu=False, We=1.0, Re=100.0)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    R = 0.25
    cx, cy = 0.5, 0.5
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    eps = 1.5 * grid.dx_min

    phi = ScalarField(grid, backend, "phi")
    phi.data[:] = r - R

    kappa = ScalarField(grid, backend, "kappa")
    curvature_calc = CurvatureCalculator(grid, backend)
    curvature_calc.compute(phi, kappa, ccd)

    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0

    out = VectorField(grid, backend, "out")
    state = make_state(grid, backend, config, ccd,
                       phi=phi, rho=rho, kappa=kappa)

    surf = SurfaceTensionTerm(grid, backend, config)
    surf.evaluate(state, out, mode='set')

    # 界面近傍での力の大きさ
    interface_mask = xp.abs(phi.data) < eps
    if xp.any(interface_mask):
        fx = out.u.data[interface_mask]
        fy = out.v.data[interface_mask]
        force_mag = xp.sqrt(fx ** 2 + fy ** 2)
        max_force = float(xp.max(force_mag))
        mean_force = float(xp.mean(force_mag))
        print(f"  界面近傍の力: mean={mean_force:.4f}, max={max_force:.4f}")

        # 力が界面に集中していることの確認（遠方はゼロ）
        far_mask = xp.abs(phi.data) > 3 * eps
        far_force = xp.sqrt(out.u.data[far_mask] ** 2
                            + out.v.data[far_mask] ** 2)
        max_far = float(xp.max(far_force))
        print(f"  遠方の力: max={max_far:.2e} (should be ~0)")

        assert max_force > 0.1, f"Surface tension force too weak: {max_force}"
        assert max_far < 1e-10, f"Force leaking beyond interface: {max_far}"
    print("✓ CSF 表面張力テスト PASSED\n")
    return True


def test_predictor_taylor_green():
    """
    Test 6: Predictor — Taylor-Green 渦の初期減衰

    u = sin(x)cos(y), v = -cos(x)sin(y)
    非圧縮: ∇·u = 0 ✓

    単相（ρ=1, μ=1）、重力なし、表面張力なしの場合:
      ∂u/∂t = -(u·∇)u + (1/Re)∇²u - ∇p

    Predictor（圧力なし）で u* を計算し、
    解析的な ∂u/∂t と比較する。

    解析解: ∂u/∂t|_{t=0} = (u·∇)u ではない（圧力項も必要）
    → ここでは Predictor の RHS = (a)+(b) が正しいかを検証
    """
    print("=" * 70)
    print("Test 6: Predictor — Taylor-Green decay (single phase)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(2*np.pi, 2*np.pi),
        use_gpu=False, Re=1.0, Fr=1e10, We=1e10)  # 重力・表面張力を無効化
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()

    vel = VectorField(grid, backend, "vel")
    vel.u.data[:] = xp.sin(X) * xp.cos(Y)
    vel.v.data[:] = -xp.cos(X) * xp.sin(Y)

    rho = ScalarField(grid, backend, "rho")
    rho.data[:] = 1.0
    mu = ScalarField(grid, backend, "mu")
    mu.data[:] = 1.0

    # 界面なし（φを大きくしてδε=0にする）
    phi = ScalarField(grid, backend, "phi")
    phi.data[:] = 100.0  # 全域液相
    kappa = ScalarField(grid, backend, "kappa")
    kappa.data[:] = 0.0

    state = make_state(grid, backend, config, ccd,
                       vel=vel, rho=rho, mu=mu, phi=phi, kappa=kappa)

    # Predictor で u* を計算
    vel_star = VectorField(grid, backend, "vel_star")
    predictor = Predictor(grid, backend, config)
    dt = 0.001
    predictor.compute(state, vel_star, dt)

    # u* = u^n + dt * [(a) + (b)]
    # (a) = -(u·∇)u の u成分 = -sin(2x)/2
    # (b) = ∇²u / Re の u成分 = -2sin(x)sin(y) / 1.0
    # → RHS_u = -sin(2x)/2 - 2sin(x)cos(y)
    # u* = sin(x)cos(y) + dt(-sin(2x)/2 - 2sin(x)cos(y))

    # RHS だけ確認
    rhs = VectorField(grid, backend, "rhs")
    predictor.compute_rhs_only(state, rhs)

    exact_rhs_u = (-xp.sin(2*X)/2.0 - 2.0*xp.sin(X)*xp.cos(Y))

    inner = slice(3, -3)
    err = float(xp.max(xp.abs(rhs.u.data[inner, inner]
                               - exact_rhs_u[inner, inner])))
    print(f"  max|RHS_u - exact| = {err:.4e}")

    # u* の確認
    exact_ustar = xp.sin(X)*xp.cos(Y) + dt * exact_rhs_u
    err_star = float(xp.max(xp.abs(vel_star.u.data[inner, inner]
                                    - exact_ustar[inner, inner])))
    print(f"  max|u* - exact_u*| = {err_star:.4e}")

    assert err < 1e-3, f"RHS error too large: {err}"
    assert err_star < 1e-6, f"u* error too large: {err_star}"
    print("✓ Predictor Taylor-Green テスト PASSED\n")
    return True


def test_predictor_stability():
    """
    Test 7: Predictor 全項結合の安定性

    二相流セットアップで数ステップ回して NaN が出ないことを確認
    """
    print("=" * 70)
    print("Test 7: Predictor stability — two-phase setup")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0),
        use_gpu=False, Re=100.0, Fr=1.0, We=10.0,
        rho_ratio=0.1, mu_ratio=0.1)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    R = 0.25
    cx, cy = 0.5, 0.5
    eps = 1.5 * grid.dx_min

    # 初期条件: 円形界面、静止状態
    phi = ScalarField(grid, backend, "phi")
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi.data[:] = r - R

    rho = ScalarField(grid, backend, "rho")
    mu = ScalarField(grid, backend, "mu")
    update_properties(phi.data, rho.data, mu.data,
                      config.rho_ratio, config.mu_ratio, eps, xp)

    kappa = ScalarField(grid, backend, "kappa")
    curvature_calc = CurvatureCalculator(grid, backend)
    curvature_calc.compute(phi, kappa, ccd)

    vel = VectorField(grid, backend, "vel")  # 静止
    vel_star = VectorField(grid, backend, "vel_star")

    state = make_state(grid, backend, config, ccd,
                       vel=vel, phi=phi, rho=rho, mu=mu, kappa=kappa)

    predictor = Predictor(grid, backend, config)

    # 3ステップ実行
    dt = 1e-4
    for step in range(3):
        predictor.compute(state, vel_star, dt)
        # u^n ← u* として更新
        for k in range(grid.ndim):
            vel[k].fill_from(vel_star[k])

        max_u = float(xp.max(xp.abs(vel.u.data)))
        max_v = float(xp.max(xp.abs(vel.v.data)))
        has_nan = bool(xp.any(xp.isnan(vel.u.data)) or
                       xp.any(xp.isnan(vel.v.data)))
        print(f"  Step {step+1}: max|u|={max_u:.4e}, max|v|={max_v:.4e}, "
              f"NaN={has_nan}")

        assert not has_nan, f"NaN detected at step {step+1}"

    print("✓ 二相流 Predictor 安定性テスト PASSED\n")
    return True


if __name__ == "__main__":
    all_pass = True

    tests = [
        test_convection_uniform,
        test_convection_accuracy,
        test_viscous_laplacian,
        test_gravity_direction,
        test_surface_tension_circle,
        test_predictor_taylor_green,
        test_predictor_stability,
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
        print("Phase 3 全テスト PASSED ✓")
    else:
        print("Phase 3 一部テスト FAILED ✗")
    print("=" * 70)
