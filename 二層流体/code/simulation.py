"""
TwoPhaseSimulation: 5フェーズ統合

論文 §9 / 図1 の完全計算フロー:

  Phase 1: Level Set 更新（①②）
  Phase 2: 格子・物性・曲率更新（③④⑤）
  Phase 3: 運動量予測 Predictor（⑥）
  Phase 4: 圧力 Poisson 求解（⑦）
  Phase 5: 速度補正・収束確認（⑧⑨）
"""

from __future__ import annotations

from .config import SimulationConfig
from .backend import Backend
from .core.grid import Grid
from .core.field import ScalarField, VectorField
from .ccd.ccd_solver import CCDSolver
from .levelset.advection import LevelSetAdvection
from .levelset.reinitialize import Reinitializer
from .levelset.curvature import CurvatureCalculator
from .levelset.heaviside import update_properties
from .ns_terms.predictor import Predictor
from .pressure.rhie_chow import RhieChowCorrection
from .pressure.ppe_builder import PPEMatrixBuilder
from .pressure.ppe_solver import PPESolver
from .pressure.velocity_corrector import VelocityCorrector
from .time_integration.cfl import CFLCalculator
from .diagnostics.monitors import DiagnosticMonitor


class TwoPhaseSimulation:
    """
    二次元/三次元 気液二相流ソルバー

    全場変数を GPU 上に保持し、5フェーズのタイムステップを繰り返す。
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.backend = Backend(config.use_gpu)
        self.xp = self.backend.xp

        # ─── コア ───
        self.grid = Grid(config, self.backend)
        self.ccd = CCDSolver(self.grid, self.backend)

        # ─── 場変数（全て GPU 上に常駐）───
        self.phi   = ScalarField(self.grid, self.backend, "phi")
        self.velocity = VectorField(self.grid, self.backend, "vel")
        self.p     = ScalarField(self.grid, self.backend, "pressure")
        self.rho   = ScalarField(self.grid, self.backend, "density")
        self.mu    = ScalarField(self.grid, self.backend, "viscosity")
        self.kappa = ScalarField(self.grid, self.backend, "curvature")

        # 作業用
        self.vel_star = VectorField(self.grid, self.backend, "vel_star")
        self.epsilon = config.epsilon_factor * self.grid.dx_min

        # ─── Phase 1: Level Set ───
        self.ls_advect = LevelSetAdvection(self.grid, self.backend)
        self.ls_reinit = Reinitializer(self.grid, self.backend, config)

        # ─── Phase 2: 曲率 ───
        self.curvature_calc = CurvatureCalculator(self.grid, self.backend)

        # ─── Phase 3: Predictor ───
        self.predictor = Predictor(self.grid, self.backend, config)

        # ─── Phase 4: 圧力 ───
        self.rhie_chow = RhieChowCorrection(self.grid, self.backend)
        self.ppe_builder = PPEMatrixBuilder(self.grid, self.backend)
        self.ppe_solver = PPESolver(self.backend, config)

        # ─── Phase 5: 速度補正 ───
        self.vel_corrector = VelocityCorrector(self.grid, self.backend)

        # ─── 診断・CFL ───
        self.monitor = DiagnosticMonitor(self.grid, self.backend)
        self.cfl_calc = CFLCalculator(config)

        # ─── 状態 ───
        self.time = 0.0
        self.step = 0

        # 初期物性値
        self.rho.data[:] = 1.0
        self.mu.data[:] = 1.0

    def _make_state(self) -> dict:
        """NS各項に渡す state 辞書を構築"""
        return {
            'velocity': self.velocity,
            'phi': self.phi,
            'rho': self.rho,
            'mu': self.mu,
            'kappa': self.kappa,
            'ccd': self.ccd,
            'config': self.config,
            'epsilon': self.epsilon,
        }

    def step_forward(self):
        """
        1タイムステップ: t^n → t^{n+1}

        5フェーズ構成（論文 図1 下段）
        """
        xp = self.xp
        cfg = self.config

        # ─── Δt 決定（CFL条件、式84）───
        dt = self.cfl_calc.compute(
            self.velocity, self.rho, self.mu, self.grid, xp)

        # ═══════════════════════════════════════════
        # Phase 1: Level Set 更新  ①②
        # ═══════════════════════════════════════════
        # ① LS移流: ∂φ/∂t + u^n·∇φ = 0 → φ*
        self.ls_advect.advance(self.phi, self.velocity, dt, self.ccd)

        # ② 再初期化: sgn(φ₀)(|∇φ|^uw - 1) = 0 → φ^{n+1}
        self.ls_reinit.reinitialize(self.phi, self.ccd)

        # ═══════════════════════════════════════════
        # Phase 2: 格子・物性値・曲率更新  ③④⑤
        # ═══════════════════════════════════════════
        # ③ グリッド更新（界面適合格子、α > 1 の場合のみ）
        if self.config.alpha_grid > 1.0:
            self.grid.update_from_levelset(self.phi, self.ccd)

        # ④ 物性値更新: ρ̃, μ̃
        update_properties(self.phi.data, self.rho.data, self.mu.data,
                          cfg.rho_ratio, cfg.mu_ratio, self.epsilon, xp)

        # ⑤ 曲率計算: CCD逐次適用 → κ^{n+1}
        self.curvature_calc.compute(self.phi, self.kappa, self.ccd)

        # ═══════════════════════════════════════════
        # Phase 3: 運動量予測 Predictor  ⑥
        #   (a)対流 + (b)粘性 + (c)重力 + (d)表面張力 → u*
        # ═══════════════════════════════════════════
        state = self._make_state()
        self.predictor.compute(state, self.vel_star, dt)

        # ═══════════════════════════════════════════
        # Phase 4: 圧力 Poisson 求解  ⑦
        #   (e_L) FVM変密度PPE + (e_R) Rhie-Chow発散
        # ═══════════════════════════════════════════
        # Rhie-Chow 界面速度
        self.rhie_chow.compute_face_velocities(
            self.vel_star, self.p, self.rho, self.ccd, dt)
        div_rc = self.rhie_chow.compute_divergence()

        # PPE 行列更新 + 右辺構築
        self.ppe_builder.update_coefficients(self.rho)
        rhs = self.ppe_builder.build_rhs(div_rc, dt)

        # 圧力の初期推定（前ステップの値を使用）
        p_vec = self.ppe_builder.build_rhs(self.p.data * 0, 1.0)  # dummy
        # 1D 解ベクトル初期値
        p_init = xp.zeros(self.ppe_builder.n_unknowns)

        # BiCGSTAB 求解
        p_sol = self.ppe_solver.solve(self.ppe_builder.matrix, rhs, p_init)
        self.ppe_builder.scatter_solution(p_sol, self.p)

        # ═══════════════════════════════════════════
        # Phase 5: 速度補正 + 収束確認  ⑧⑨
        # ═══════════════════════════════════════════
        # ⑧ 速度補正: u^{n+1} = u* - (Δt/ρ̃)∇p
        self.vel_corrector.correct(
            self.vel_star, self.velocity, self.p,
            self.rho, self.ccd, dt)

        self.time += dt
        self.step += 1
        return dt

    def run(self, output_interval: int = 100, verbose: bool = True):
        """
        メインループ: t=0 → t_end

        Args:
            output_interval: 診断出力の間隔（ステップ数）
            verbose: 診断レポートを表示するか
        """
        if verbose:
            print(f"Two-Phase Simulation: ndim={self.config.ndim}, "
                  f"N={self.config.N}, t_end={self.config.t_end}")
            print(f"  Re={self.config.Re}, Fr={self.config.Fr}, "
                  f"We={self.config.We}")
            print(f"  ρ̂={self.config.rho_ratio}, μ̂={self.config.mu_ratio}")
            print(f"  Backend: {self.backend}")

        # 初期物性値
        update_properties(self.phi.data, self.rho.data, self.mu.data,
                          self.config.rho_ratio, self.config.mu_ratio,
                          self.epsilon, self.xp)
        self.monitor.check_volume(self.phi, self.epsilon)  # 初期体積を記録

        while self.time < self.config.t_end:
            dt = self.step_forward()

            if verbose and self.step % output_interval == 0:
                self.monitor.report(
                    self.time, self.step, self.velocity,
                    self.phi, self.ccd, self.epsilon)

        if verbose:
            print(f"\nSimulation complete: {self.step} steps, t={self.time:.4e}")
