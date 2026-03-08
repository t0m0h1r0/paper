"""
滑らか化ヘヴィサイド関数・デルタ関数

論文 §3.2:
  Hε(φ) — 界面近傍で 0→1 に遷移（式12）
  δε(φ) — dHε/dφ（式13）

物性値補間:
  ρ̃ = ρ̂ + (1 - ρ̂) Hε(φ)    （式14,15）
  μ̃ = μ̂ + (1 - μ̂) Hε(φ)
"""

from __future__ import annotations
import math


def heaviside_smooth(phi, epsilon, xp):
    """
    滑らか化ヘヴィサイド関数 Hε(φ)（式12）

        0                                (φ < -ε)
    H = (1/2)[1 + φ/ε + sin(πφ/ε)/π]  (|φ| ≤ ε)
        1                                (φ > +ε)

    Args:
        phi: Level Set 配列
        epsilon: 界面幅パラメータ
        xp: numpy or cupy

    Returns:
        H: 同形状の配列
    """
    H = xp.zeros_like(phi)
    mask_pos = phi > epsilon
    mask_mid = xp.abs(phi) <= epsilon

    H[mask_pos] = 1.0
    phi_mid = phi[mask_mid]
    H[mask_mid] = 0.5 * (1.0 + phi_mid / epsilon
                         + xp.sin(math.pi * phi_mid / epsilon) / math.pi)
    return H


def delta_smooth(phi, epsilon, xp):
    """
    滑らか化デルタ関数 δε(φ) = dHε/dφ（式13）

         0                                   (|φ| > ε)
    δ = (1/(2ε))[1 + cos(πφ/ε)]            (|φ| ≤ ε)

    Args:
        phi: Level Set 配列
        epsilon: 界面幅パラメータ
        xp: numpy or cupy

    Returns:
        delta: 同形状の配列
    """
    delta = xp.zeros_like(phi)
    mask = xp.abs(phi) <= epsilon
    phi_m = phi[mask]
    delta[mask] = (0.5 / epsilon) * (1.0 + xp.cos(math.pi * phi_m / epsilon))
    return delta


def update_properties(phi_data, rho_data, mu_data,
                      rho_ratio, mu_ratio, epsilon, xp):
    """
    ④ 物性値更新（in-place）

    ρ̃ = ρ̂ + (1 - ρ̂) Hε(φ)
    μ̃ = μ̂ + (1 - μ̂) Hε(φ)

    Args:
        phi_data: φ の配列（GPU上）
        rho_data: ρ の配列（in-place書き込み先）
        mu_data:  μ の配列（in-place書き込み先）
        rho_ratio: ρ̂ = ρ_g/ρ_l
        mu_ratio:  μ̂ = μ_g/μ_l
        epsilon: 界面幅
        xp: numpy or cupy
    """
    H = heaviside_smooth(phi_data, epsilon, xp)
    rho_data[:] = rho_ratio + (1.0 - rho_ratio) * H
    mu_data[:] = mu_ratio + (1.0 - mu_ratio) * H
