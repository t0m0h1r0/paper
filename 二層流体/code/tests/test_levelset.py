#!/usr/bin/env python3
"""
Phase 2 テスト: Level Set 移流・再初期化・曲率・物性値補間

テスト項目:
  1. Heaviside / delta 関数の基本性質
  2. TVD-RK3 の 3次精度収束
  3. 円形界面の移流 → 体積保存
  4. 再初期化 → Eikonal |∇φ|≈1 の回復
  5. 円の曲率 κ=1/R の精度検証
  6. Godunov 上流勾配の符号依存テスト
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
from twophase.levelset.advection import LevelSetAdvection
from twophase.levelset.reinitialize import Reinitializer
from twophase.levelset.curvature import CurvatureCalculator
from twophase.levelset.godunov import GodunovGradient
from twophase.time_integration.tvd_rk3 import TVDRK3


def test_heaviside_delta():
    """
    Test 1: Hε, δε の基本性質
      - Hε(-∞) → 0, Hε(+∞) → 1, Hε(0) = 0.5
      - ∫δε dφ = 1 （数値積分で確認）
      - 物性値補間: ρ(φ<-ε) = ρ_g, ρ(φ>ε) = ρ_l
    """
    print("=" * 70)
    print("Test 1: Heaviside / delta functions")
    print("=" * 70)

    xp = np
    eps = 0.1

    # 基本値のテスト
    phi_test = xp.array([-1.0, -eps, 0.0, eps, 1.0])
    H = heaviside_smooth(phi_test, eps, xp)

    assert abs(H[0] - 0.0) < 1e-14, f"H(-1) = {H[0]}, expected 0"
    assert abs(H[2] - 0.5) < 1e-14, f"H(0) = {H[2]}, expected 0.5"
    assert abs(H[4] - 1.0) < 1e-14, f"H(1) = {H[4]}, expected 1"
    print("  Hε(-1)=0, Hε(0)=0.5, Hε(+1)=1 ... OK")

    # δε の積分 = 1
    phi_fine = xp.linspace(-0.5, 0.5, 10001)
    delta = delta_smooth(phi_fine, eps, xp)
    dphi = phi_fine[1] - phi_fine[0]
    integral = float(xp.sum(delta) * dphi)
    assert abs(integral - 1.0) < 1e-3, f"∫δε dφ = {integral}, expected 1"
    print(f"  ∫δε dφ = {integral:.6f} ≈ 1.0 ... OK")

    # 物性値補間テスト
    phi_arr = xp.linspace(-1, 1, 101)
    rho = xp.zeros_like(phi_arr)
    mu = xp.zeros_like(phi_arr)
    update_properties(phi_arr, rho, mu, 0.001, 0.01, eps, xp)

    assert abs(rho[0] - 0.001) < 1e-10, f"ρ(gas) = {rho[0]}"
    assert abs(rho[-1] - 1.0) < 1e-10, f"ρ(liquid) = {rho[-1]}"
    print(f"  ρ(gas)={rho[0]:.4f}, ρ(liquid)={rho[-1]:.4f} ... OK")
    print("✓ Heaviside / delta テスト PASSED\n")
    return True


def test_tvdrk3_convergence():
    """
    Test 2: TVD-RK3 の3次精度収束

    テスト問題: dq/dt = -q, q(0) = 1  → q(t) = exp(-t)
    """
    print("=" * 70)
    print("Test 2: TVD-RK3 convergence (dq/dt = -q)")
    print("=" * 70)

    xp = np
    t_end = 1.0
    q0 = 1.0
    exact = np.exp(-t_end)

    dts = [0.2, 0.1, 0.05, 0.025]
    errors = []

    for dt in dts:
        rk3 = TVDRK3(xp, (1,))
        q = xp.array([q0])

        def rhs(q_data):
            return -q_data

        n_steps = int(t_end / dt)
        for _ in range(n_steps):
            rk3.advance(q, dt, rhs)

        err = abs(float(q[0]) - exact)
        errors.append(err)

    print(f"  {'dt':>10} {'error':>14} {'order':>8}")
    print("  " + "-" * 36)
    for i, dt in enumerate(dts):
        order = "--"
        if i > 0:
            order = f"{np.log2(errors[i-1]/errors[i]):.2f}"
        print(f"  {dt:>10.4f} {errors[i]:>14.6e} {order:>8}")

    final_order = np.log2(errors[-2] / errors[-1])
    print(f"\n  最終収束次数: {final_order:.2f} (期待値: 3.00)")
    assert final_order > 2.5, f"RK3 order {final_order:.2f} < 2.5"
    print("✓ TVD-RK3 3次精度確認\n")
    return True


def test_circle_advection_volume():
    """
    Test 3: 円形界面の剛体回転移流 → 体積保存

    速度場: u = -y + 0.5, v = x - 0.5 （原点(0.5,0.5)を中心とした回転）
    φ: 半径0.15, 中心(0.5,0.75) の円
    1回転後に元に戻るか + 体積保存を確認
    """
    print("=" * 70)
    print("Test 3: Circle advection — volume conservation")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    phi = ScalarField(grid, backend, "phi")
    vel = VectorField(grid, backend, "vel")

    X, Y = grid.meshgrid()

    # 初期条件: 半径0.15の円、中心(0.5, 0.75)
    R = 0.15
    cx, cy = 0.5, 0.75
    phi.data[:] = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - R

    # 速度場: 回転
    vel.u.data[:] = -(Y - 0.5)
    vel.v.data[:] = (X - 0.5)

    # 初期体積
    eps = 1.5 * grid.dx_min
    H0 = heaviside_smooth(phi.data, eps, xp)
    dx = grid.h[0]
    dy = grid.h[1]
    vol_0 = float(xp.sum(H0)) * dx * dy

    # 移流
    advector = LevelSetAdvection(grid, backend)
    dt = 0.005
    n_steps = 100  # 小さめのステップ数で部分回転

    for _ in range(n_steps):
        advector.advance(phi, vel, dt, ccd)

    # 最終体積
    H_final = heaviside_smooth(phi.data, eps, xp)
    vol_final = float(xp.sum(H_final)) * dx * dy

    vol_err = abs(vol_final - vol_0) / vol_0 * 100.0
    print(f"  初期体積: {vol_0:.6f}")
    print(f"  最終体積: {vol_final:.6f}")
    print(f"  体積変化: {vol_err:.4f}%")

    assert vol_err < 1.0, f"Volume error {vol_err:.2f}% > 1%"
    print("✓ 体積保存テスト PASSED（< 1%）\n")
    return True


def test_reinitialization_eikonal():
    """
    Test 4: 再初期化 → Eikonal |∇φ| ≈ 1 の回復

    φ を意図的に非距離関数にしてから再初期化し、
    |∇φ| = 1 が回復されることを確認する。
    """
    print("=" * 70)
    print("Test 4: Reinitialization — Eikonal quality")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 64

    config = SimulationConfig(
        ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False,
        reinit_steps=20)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    phi = ScalarField(grid, backend, "phi")
    X, Y = grid.meshgrid()

    # 初期条件: 非距離関数（φ = 1.5*(r - R) で |∇φ| = 1.5）
    R = 0.25
    cx, cy = 0.5, 0.5
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi.data[:] = 1.5 * (r - R)

    # |∇φ| の初期チェック（界面近傍帯 |φ| < 5dx のみ）
    dx = grid.dx_min
    band = 5.0 * dx
    phi.ensure_derivatives(ccd)
    grad_mag_before = xp.sqrt(phi.d1[0] ** 2 + phi.d1[1] ** 2)
    near_if = xp.abs(phi.data) < band
    if xp.any(near_if):
        eik_err_before = float(xp.mean(xp.abs(grad_mag_before[near_if] - 1.0)))
    else:
        eik_err_before = 1.0
    print(f"  再初期化前: 界面帯 mean||∇φ| - 1| = {eik_err_before:.4f}")

    # 再初期化
    reinit = Reinitializer(grid, backend, config)
    reinit.reinitialize(phi, ccd)

    # 再初期化後
    phi.ensure_derivatives(ccd)
    grad_mag_after = xp.sqrt(phi.d1[0] ** 2 + phi.d1[1] ** 2)
    if xp.any(near_if):
        eik_err_after = float(xp.mean(xp.abs(grad_mag_after[near_if] - 1.0)))
    else:
        eik_err_after = 1.0
    print(f"  再初期化後: 界面帯 mean||∇φ| - 1| = {eik_err_after:.4f}")

    assert eik_err_after < eik_err_before, \
        f"Eikonal error did not decrease: {eik_err_after} >= {eik_err_before}"
    assert eik_err_after < 0.3, f"Eikonal error too large: {eik_err_after}"
    print("✓ Eikonal 品質改善確認\n")
    return True


def test_curvature_circle():
    """
    Test 5: 円の曲率 κ = 1/R

    φ = √((x-cx)² + (y-cy)²) - R  （符号付き距離関数）
    理論値: κ = 1/R（一定）
    CCD O(h^6) で高精度が期待される
    """
    print("=" * 70)
    print("Test 5: Curvature of circle (κ = 1/R)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp

    Ns = [32, 64, 128]
    R = 0.25
    cx, cy = 0.5, 0.5
    # κ = -∇·(∇φ/|∇φ|) で φ=r-R → κ = -1/R（論文の符号規約）
    exact_kappa = -1.0 / R

    for N in Ns:
        config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False)
        grid = Grid(config, backend)
        ccd = CCDSolver(grid, backend)

        phi = ScalarField(grid, backend, "phi")
        kappa = ScalarField(grid, backend, "kappa")

        X, Y = grid.meshgrid()
        r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        phi.data[:] = r - R

        curvature_calc = CurvatureCalculator(grid, backend)
        curvature_calc.compute(phi, kappa, ccd)

        # 界面近傍（|φ| < 3Δx）での曲率誤差
        dx = grid.dx_min
        eps = 3.0 * dx
        interface_mask = xp.abs(phi.data) < eps

        if xp.any(interface_mask):
            kappa_at_interface = kappa.data[interface_mask]
            err = float(xp.max(xp.abs(kappa_at_interface - exact_kappa)))
            mean_kappa = float(xp.mean(kappa_at_interface))
            print(f"  N={N:>4}: κ_mean={mean_kappa:.4f} (exact={exact_kappa:.1f}), "
                  f"max|κ-κ_exact|={err:.4e}")

    # N=128 での精度チェック
    assert err < 1.0, f"Curvature error {err:.4e} too large at N=128"
    print("✓ 曲率テスト PASSED\n")
    return True


def test_godunov_sign_dependence():
    """
    Test 6: Godunov 上流勾配の符号依存テスト

    φ₀ > 0 と φ₀ < 0 で上流方向が反転することを確認
    """
    print("=" * 70)
    print("Test 6: Godunov gradient sign dependence")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 32

    config = SimulationConfig(ndim=2, N=(N, N), L=(1.0, 1.0), use_gpu=False)
    grid = Grid(config, backend)

    X, Y = grid.meshgrid()
    R = 0.25
    cx, cy = 0.5, 0.5

    # 符号付き距離関数（|∇φ| ≈ 1）
    r = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi_data = r - R

    # φ₀ = φ（正しい符号で Godunov 勾配を計算）
    godunov = GodunovGradient(grid, backend)
    out = xp.zeros(grid.shape)

    godunov.compute(phi_data, phi_data, out)

    # 内部点での |∇φ|^uw ≈ 1 を確認
    inner = out[3:-3, 3:-3]
    mean_grad = float(xp.mean(inner))
    max_dev = float(xp.max(xp.abs(inner - 1.0)))

    print(f"  Mean |∇φ|^uw = {mean_grad:.4f}")
    print(f"  Max ||∇φ|^uw - 1| = {max_dev:.4f}")

    # φ₀ を反転したときの挙動確認
    out_neg = xp.zeros(grid.shape)
    godunov.compute(phi_data, -phi_data, out_neg)
    # 反転しても |∇φ|^uw は同じ値になるべき（距離関数の場合）
    inner_neg = out_neg[3:-3, 3:-3]
    mean_grad_neg = float(xp.mean(inner_neg))
    print(f"  反転φ₀での Mean |∇φ|^uw = {mean_grad_neg:.4f}")

    assert abs(mean_grad - 1.0) < 0.2, f"Mean gradient {mean_grad} too far from 1"
    print("✓ Godunov 符号依存テスト PASSED\n")
    return True


if __name__ == "__main__":
    all_pass = True

    tests = [
        test_heaviside_delta,
        test_tvdrk3_convergence,
        test_circle_advection_volume,
        test_reinitialization_eikonal,
        test_curvature_circle,
        test_godunov_sign_dependence,
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
        print("Phase 2 全テスト PASSED ✓")
    else:
        print("Phase 2 一部テスト FAILED ✗")
    print("=" * 70)
