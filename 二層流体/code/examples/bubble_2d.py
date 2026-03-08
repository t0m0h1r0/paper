#!/usr/bin/env python3
"""
2D 気泡上昇シミュレーション + 可視化デモ

円形の軽い気泡（φ<0: 気相）が浮力で上昇する。
表面張力が界面を維持し、重力が気泡を押し上げる。

使い方:
  python examples/bubble_2d.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twophase import TwoPhaseSimulation, SimulationConfig
from twophase.levelset.heaviside import update_properties
from twophase.visualization import Visualizer

config = SimulationConfig(
    ndim=2,
    N=(64, 128),
    L=(1.0, 2.0),
    Re=50.0,
    Fr=1.0,
    We=20.0,
    rho_ratio=0.1,
    mu_ratio=0.1,
    cfl_number=0.2,
    t_end=0.05,
    reinit_steps=3,
    alpha_grid=1.0,
    epsilon_factor=1.5,
    bicgstab_tol=1e-8,
    bicgstab_maxiter=500,
    use_gpu=False,
)

sim = TwoPhaseSimulation(config)
xp = sim.xp
X, Y = sim.grid.meshgrid()

# 初期条件: 半径0.15の円形気泡, 中心(0.5, 0.5)
R = 0.15
cx, cy = 0.5, 0.5
sim.phi.data[:] = xp.sqrt((X - cx)**2 + (Y - cy)**2) - R

# 物性値初期化
update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                  config.rho_ratio, config.mu_ratio, sim.epsilon, xp)

vis = Visualizer(sim, output_dir="output_2d")

# 初期状態を保存
from twophase.levelset.curvature import CurvatureCalculator
curv = CurvatureCalculator(sim.grid, sim.backend)
curv.compute(sim.phi, sim.kappa, sim.ccd)
vis.snapshot("frame_0000.png", title="t=0 (initial)")

# シミュレーション実行
frame = 1
output_interval = 20
print(f"\n2D Bubble Rising: N={config.N}, Re={config.Re}, "
      f"Fr={config.Fr}, We={config.We}")
print(f"  ρ_ratio={config.rho_ratio}, μ_ratio={config.mu_ratio}")
print("-" * 60)

while sim.time < config.t_end:
    dt = sim.step_forward()

    if sim.step % output_interval == 0:
        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        div = sim.monitor.check_divergence(sim.velocity, sim.ccd)
        max_u = float(xp.max(xp.abs(sim.velocity.u.data)))
        max_v = float(xp.max(xp.abs(sim.velocity.v.data)))
        print(f"  Step {sim.step:>5d}  t={sim.time:.4e}  dt={dt:.2e}  "
              f"|u|={max_u:.3e}  |v|={max_v:.3e}  "
              f"vol_err={vol_err:.3f}%  div={div:.2e}")

        vis.snapshot(f"frame_{frame:04d}.png")
        frame += 1

print(f"\nDone: {sim.step} steps, t={sim.time:.4e}")
print(f"Output in: output_2d/")
