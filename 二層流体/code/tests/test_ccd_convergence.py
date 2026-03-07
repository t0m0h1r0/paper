#!/usr/bin/env python3
"""
CCD 精度収束テスト

テスト関数: f(x) = sin(x), x ∈ [0, π]
  f'(x) = cos(x)
  f''(x) = -sin(x)

検証項目:
  1. 指定境界条件での O(h^6) 収束（1階微分）
  2. 境界コンパクトスキームでの精度
  3. 2D 場でのバッチ並列処理
  4. 多項式完全性テスト（f(x) = x^6）
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twophase.backend import Backend
from twophase.config import SimulationConfig
from twophase.core.grid import Grid
from twophase.ccd.ccd_solver import CCDSolver


def test_sinx_prescribed_bc():
    """
    Test 1: f(x) = sin(x) with prescribed boundary derivatives
    論文 表3 の再現: O(h^6) 収束の確認
    """
    print("=" * 70)
    print("Test 1: sin(x) with prescribed boundary derivatives")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp

    Ns = [10, 20, 40, 80, 160]
    errors_d1 = []
    errors_d2 = []

    for N in Ns:
        config = SimulationConfig(ndim=2, N=(N, 4), L=(np.pi, 1.0), use_gpu=False)
        grid = Grid(config, backend)
        ccd = CCDSolver(grid, backend)

        # f(x) = sin(x) を 2D 場として設定（y方向は定数）
        x = grid.coords[0]
        f_1d = xp.sin(x)
        # 2D に拡張: (Nx+1, Ny+1)
        f_2d = xp.tile(f_1d[:, None], (1, grid.shape[1]))

        # 厳密微分値
        exact_d1 = xp.cos(x)
        exact_d2 = -xp.sin(x)

        # 境界微分値を厳密値として指定
        bc_left = (float(xp.cos(0.0)), float(-xp.sin(0.0)))     # (1, 0)
        bc_right = (float(xp.cos(np.pi)), float(-xp.sin(np.pi)))  # (-1, 0)

        # CCD 実行（x方向、axis=0）
        d1, d2 = ccd.differentiate(f_2d, axis=0,
                                    bc_left=bc_left, bc_right=bc_right)

        # 誤差評価（y=0 の列）
        err_d1 = float(xp.max(xp.abs(d1[:, 0] - exact_d1)))
        err_d2 = float(xp.max(xp.abs(d2[:, 0] - exact_d2)))
        errors_d1.append(err_d1)
        errors_d2.append(err_d2)

    # 収束次数の計算
    print(f"{'N':>6} {'h':>12} {'err_d1':>14} {'order_d1':>10} "
          f"{'err_d2':>14} {'order_d2':>10}")
    print("-" * 70)
    for i, N in enumerate(Ns):
        h = np.pi / N
        order_d1 = "--"
        order_d2 = "--"
        if i > 0:
            order_d1 = f"{np.log2(errors_d1[i-1]/errors_d1[i]):.2f}"
            order_d2 = f"{np.log2(errors_d2[i-1]/errors_d2[i]):.2f}"
        print(f"{N:>6} {h:>12.4e} {errors_d1[i]:>14.4e} {order_d1:>10} "
              f"{errors_d2[i]:>14.4e} {order_d2:>10}")

    # O(h^6) の判定（機械精度到達前の収束率を使用）
    # N=160 では d1 誤差が ~1e-14 で倍精度限界に到達するため
    # N=40→80 の収束率で判定する
    order_d1_check = np.log2(errors_d1[-3] / errors_d1[-2])
    print(f"\n収束次数 (d1, N=40→80): {order_d1_check:.2f} (期待値: 6.00)")
    assert order_d1_check > 5.5, f"d1 convergence order {order_d1_check:.2f} < 5.5"
    print("✓ 1階微分 O(h^6) 収束確認（N=160では倍精度限界に到達）\n")

    return True


def test_sinx_compact_bc():
    """
    Test 2: f(x) = sin(x) with compact boundary scheme
    境界スキーム使用時の精度確認（O(h^5)以上）
    """
    print("=" * 70)
    print("Test 2: sin(x) with compact boundary scheme")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp

    Ns = [10, 20, 40, 80, 160]
    errors_d1 = []
    errors_d2 = []

    for N in Ns:
        config = SimulationConfig(ndim=2, N=(N, 4), L=(np.pi, 1.0), use_gpu=False)
        grid = Grid(config, backend)
        ccd = CCDSolver(grid, backend)

        x = grid.coords[0]
        f_1d = xp.sin(x)
        f_2d = xp.tile(f_1d[:, None], (1, grid.shape[1]))

        exact_d1 = xp.cos(x)
        exact_d2 = -xp.sin(x)

        # 境界スキーム使用（bc=None）
        d1, d2 = ccd.differentiate(f_2d, axis=0)

        err_d1 = float(xp.max(xp.abs(d1[:, 0] - exact_d1)))
        err_d2 = float(xp.max(xp.abs(d2[:, 0] - exact_d2)))
        errors_d1.append(err_d1)
        errors_d2.append(err_d2)

    print(f"{'N':>6} {'h':>12} {'err_d1':>14} {'order_d1':>10} "
          f"{'err_d2':>14} {'order_d2':>10}")
    print("-" * 70)
    for i, N in enumerate(Ns):
        h = np.pi / N
        order_d1 = "--"
        order_d2 = "--"
        if i > 0:
            order_d1 = f"{np.log2(errors_d1[i-1]/errors_d1[i]):.2f}"
            order_d2 = f"{np.log2(errors_d2[i-1]/errors_d2[i]):.2f}"
        print(f"{N:>6} {h:>12.4e} {errors_d1[i]:>14.4e} {order_d1:>10} "
              f"{errors_d2[i]:>14.4e} {order_d2:>10}")

    final_order_d1 = np.log2(errors_d1[-2] / errors_d1[-1])
    final_order_d2 = np.log2(errors_d2[-2] / errors_d2[-1])
    print(f"\n最終収束次数 (d1): {final_order_d1:.2f}")
    print(f"最終収束次数 (d2): {final_order_d2:.2f}")
    assert final_order_d1 > 4.5, f"d1 order {final_order_d1:.2f} < 4.5"
    print("✓ 境界スキーム込みでの精度確認\n")

    return True


def test_polynomial_exactness():
    """
    Test 3: f(x) = x^5 の多項式完全性テスト
    CCD O(h^6) は 5次以下の多項式に対して厳密（打切り誤差 = 0）
    """
    print("=" * 70)
    print("Test 3: Polynomial exactness (f(x) = x^5)")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 20
    L = 2.0

    config = SimulationConfig(ndim=2, N=(N, 4), L=(L, 1.0), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    x = grid.coords[0]
    f_1d = x ** 5
    f_2d = xp.tile(f_1d[:, None], (1, grid.shape[1]))

    exact_d1 = 5.0 * x ** 4
    exact_d2 = 20.0 * x ** 3

    bc_left = (0.0, 0.0)
    bc_right = (5.0 * L**4, 20.0 * L**3)

    d1, d2 = ccd.differentiate(f_2d, axis=0,
                                bc_left=bc_left, bc_right=bc_right)

    err_d1 = float(xp.max(xp.abs(d1[:, 0] - exact_d1)))
    err_d2 = float(xp.max(xp.abs(d2[:, 0] - exact_d2)))

    print(f"  f'  max error: {err_d1:.2e}")
    print(f"  f'' max error: {err_d2:.2e}")

    assert err_d1 < 1e-10, f"d1 error {err_d1:.2e} for polynomial"
    assert err_d2 < 1e-10, f"d2 error {err_d2:.2e} for polynomial"
    print("✓ 多項式完全性テスト OK（機械精度レベル）\n")

    return True


def test_1d_field():
    """
    Test 4: 純粋な1D配列での動作確認
    """
    print("=" * 70)
    print("Test 4: 1D array differentiation")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 40

    config = SimulationConfig(ndim=2, N=(N, 4), L=(np.pi, 1.0), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    x = grid.coords[0]
    f_1d = xp.sin(x)
    # 最小の2D形状 (Nx+1, 1) で確認
    f_2d = f_1d[:, None]

    exact_d1 = xp.cos(x)

    d1, d2 = ccd.differentiate(f_2d, axis=0)

    err = float(xp.max(xp.abs(d1[:, 0] - exact_d1)))
    print(f"  N={N}, max error d1: {err:.4e}")
    assert err < 1e-6, f"Error too large: {err}"
    print("✓ 1D 配列テスト OK\n")

    return True


def test_y_direction():
    """
    Test 5: y方向の微分（axis=1）が正しく動作するか
    f(x,y) = sin(y) → ∂f/∂y = cos(y)
    """
    print("=" * 70)
    print("Test 5: y-direction differentiation")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    Nx, Ny = 4, 40

    config = SimulationConfig(ndim=2, N=(Nx, Ny), L=(1.0, np.pi), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    y = grid.coords[1]
    f_2d = xp.tile(xp.sin(y)[None, :], (Nx + 1, 1))  # (Nx+1, Ny+1)
    exact_d1y = xp.cos(y)

    d1y, d2y = ccd.differentiate(f_2d, axis=1)

    err = float(xp.max(xp.abs(d1y[0, :] - exact_d1y)))
    print(f"  Ny={Ny}, max error d1y: {err:.4e}")
    assert err < 1e-6, f"y-direction error too large: {err}"
    print("✓ y方向微分テスト OK\n")

    return True


def test_mixed_derivative_2d():
    """
    Test 6: 混合偏微分 ∂²f/∂x∂y の検証
    f(x,y) = sin(x)cos(y) → ∂²f/∂x∂y = -cos(x)sin(y)
    """
    print("=" * 70)
    print("Test 6: Mixed partial derivative ∂²f/∂x∂y")
    print("=" * 70)

    backend = Backend(use_gpu=False)
    xp = backend.xp
    N = 40

    config = SimulationConfig(ndim=2, N=(N, N), L=(np.pi, np.pi), use_gpu=False)
    grid = Grid(config, backend)
    ccd = CCDSolver(grid, backend)

    X, Y = grid.meshgrid()
    f = xp.sin(X) * xp.cos(Y)
    exact_dxy = -xp.cos(X) * xp.sin(Y)

    # 逐次適用: y方向CCD → ∂f/∂y, 次にx方向CCD → (∂f/∂y)_x
    d1y, _ = ccd.differentiate(f, axis=1)
    d_xy, _ = ccd.differentiate(d1y, axis=0)

    err = float(xp.max(xp.abs(d_xy - exact_dxy)))
    print(f"  N={N}x{N}, max error ∂²f/∂x∂y: {err:.4e}")
    assert err < 1e-5, f"Mixed derivative error too large: {err}"
    print("✓ 混合偏微分テスト OK\n")

    return True


if __name__ == "__main__":
    all_pass = True

    tests = [
        test_sinx_prescribed_bc,
        test_sinx_compact_bc,
        test_polynomial_exactness,
        test_1d_field,
        test_y_direction,
        test_mixed_derivative_2d,
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
        print("全テスト PASSED ✓")
    else:
        print("一部テスト FAILED ✗")
    print("=" * 70)
