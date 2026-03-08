#!/usr/bin/env python3
"""
3D 気泡上昇シミュレーション + 可視化デモ

球形の軽い気泡が浮力で上昇する。
等値面 + 中央断面の速度場を可視化。

使い方:
  python examples/bubble_3d.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twophase import TwoPhaseSimulation, SimulationConfig
from twophase.levelset.heaviside import update_properties
from twophase.visualization import Visualizer

config = SimulationConfig(
    ndim=3,
    N=(24, 24, 48),
    L=(1.0, 1.0, 2.0),
    Re=20.0,
    Fr=1.0,
    We=50.0,
    rho_ratio=0.5,
    mu_ratio=0.5,
    cfl_number=0.15,
    t_end=0.02,
    reinit_steps=2,
    alpha_grid=1.0,
    epsilon_factor=1.5,
    bicgstab_tol=1e-6,
    bicgstab_maxiter=300,
    use_gpu=False,
)

sim = TwoPhaseSimulation(config)
xp = sim.xp
X, Y, Z = sim.grid.meshgrid()

# 初期条件: 半径0.15の球形気泡, 中心(0.5, 0.5, 0.5)
R = 0.15
sim.phi.data[:] = xp.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2) - R

update_properties(sim.phi.data, sim.rho.data, sim.mu.data,
                  config.rho_ratio, config.mu_ratio, sim.epsilon, xp)

vis = Visualizer(sim, output_dir="output_3d")
vis.snapshot("frame_0000.png", title="t=0 (initial)")

frame = 1
output_interval = 10
print(f"\n3D Bubble Rising: N={config.N}, Re={config.Re}")
print("-" * 60)

while sim.time < config.t_end:
    dt = sim.step_forward()

    if sim.step % output_interval == 0:
        vol_err = sim.monitor.volume_error(sim.phi, sim.epsilon)
        print(f"  Step {sim.step:>5d}  t={sim.time:.4e}  "
              f"vol_err={vol_err:.3f}%")
        vis.snapshot(f"frame_{frame:04d}.png")
        frame += 1

print(f"\nDone: {sim.step} steps, t={sim.time:.4e}")
print(f"Output in: output_3d/")
