# 二次元／三次元 気液二相流ソルバー — 全体設計書 v2

## 0. 設計変更サマリ（v1 → v2）

| 変更点 | 内容 |
|--------|------|
| **2D/3D 統一** | `ndim` パラメータで 2D/3D を透過的に切り替え。配列形状・演算子・PPEステンシルを自動調整 |
| **NS項ラベル対応** | 論文 図1 の (a)対流, (b)粘性, (c)重力, (d)表面張力, (e)PPE, (f)速度補正 を直接コード構造に反映 |
| **5フェーズ構成** | タイムステップを Phase 1〜5 に明確分割。コード構造が論文のフローと 1:1 対応 |

---

## 1. 設計思想

### 1.1 NS方程式の各項がそのままコードに見える構造

論文の Predictor 式（式57）を**コードレベルで再現**する：

```
∂u/∂t = -(u·∇)u     + ∇·[μ̃(∇u)^sym]/(ρ̃·Re) - ẑ/Fr² + κδε∇φ/(ρ̃·We)
         ─────────     ──────────────────────   ──────   ──────────────
         (a) 対流項     (b) 粘性項               (c)重力  (d) 表面張力
```

これを直接反映したコード構造：

```python
class NSTerms:
    """NS方程式の各項を個別に計算する演算子群"""
    convection: ConvectionTerm     # (a) -u·∇u          — CCD O(h^6), TVD-RK3 陽的
    viscous:    ViscousTerm        # (b) ∇·[μ(∇u)^sym]  — CCD O(h^6), Crank-Nicolson 半陰的
    gravity:    GravityTerm        # (c) -ẑ/Fr²         — 定数（ρ̃で除算済み）
    surface:    SurfaceTensionTerm # (d) κδε∇φ/We       — CCD曲率, 半陰的(t^{n+1}値)

class PressureStep:
    """PPE + 速度補正"""
    ppe:        PPESolver          # (e_L)(e_R) FVM変密度PPE + Rhie-Chow
    correction: VelocityCorrector  # (f) u^{n+1} = u* - (Δt/ρ̃)∇p
```

### 1.2 2D/3D 統一の仕組み

次元の違いを**配列の最終軸の有無**のみで吸収する：

```
2D: field.shape = (Nx+1, Ny+1)        速度 = (u, v)
3D: field.shape = (Nx+1, Ny+1, Nz+1)  速度 = (u, v, w)
```

CCD 演算子は常に「1D問題をバッチ並列で解く」構造のため、次元数に関わらず同一コードで動作。
PPE は 2D→5点ステンシル、3D→7点ステンシル を `ndim` で自動構築。

---

## 2. パッケージ構成

```
twophase/
├── __init__.py
├── config.py                 # SimulationConfig — 全パラメータ一元管理
├── backend.py                # Backend — CuPy/NumPy 抽象化
│
├── core/
│   ├── __init__.py
│   ├── field.py              # ScalarField, VectorField（2D/3D自動対応）
│   ├── grid.py               # Grid（ndim対応、メトリクス管理）
│   └── boundary.py           # BoundaryCondition（壁/周期/流入出）
│
├── ccd/
│   ├── __init__.py
│   ├── ccd_solver.py         # CCDSolver — 1D CCD のバッチ並列実行
│   ├── block_tridiag.py      # ブロック三重対角ソルバー（GPU並列）
│   └── derivatives.py        # DerivativeCalculator — 全微分値の統合計算
│
├── ns_terms/                  ★ 論文の (a)(b)(c)(d) に直接対応
│   ├── __init__.py
│   ├── base.py               # NSTerm 基底クラス
│   ├── convection.py         # (a) ConvectionTerm    — u·∇u
│   ├── viscous.py            # (b) ViscousTerm       — ∇·[μ(∇u+∇u^T)]
│   ├── gravity.py            # (c) GravityTerm       — -ẑ/Fr²
│   ├── surface_tension.py    # (d) SurfaceTensionTerm — κδε∇φ/We
│   └── predictor.py          # Predictor — (a)+(b)+(c)+(d) を統合して u* を計算
│
├── pressure/                  ★ 論文の (e_L)(e_R)(f) に直接対応
│   ├── __init__.py
│   ├── rhie_chow.py          # (e_R) RhieChowCorrection — 界面速度・RC発散
│   ├── ppe_builder.py        # (e_L) PPEMatrixBuilder — FVM 5点/7点疎行列
│   ├── ppe_solver.py         # (e_L) PPESolver — BiCGSTAB + ILU(0)
│   └── velocity_corrector.py # (f)   VelocityCorrector — u^{n+1} = u* - (Δt/ρ̃)∇p
│
├── levelset/                  ★ 論文の Phase 1,2 に対応
│   ├── __init__.py
│   ├── advection.py          # ① LS移流 — TVD-RK3
│   ├── reinitialize.py       # ② 再初期化 — Godunov上流勾配(sgn依存)
│   ├── godunov.py            # Godunov型上流勾配（2D/3D対応、式81）
│   ├── heaviside.py          # Hε, δε — 滑らか化関数
│   └── curvature.py          # ⑤ 曲率計算 — CCD逐次適用（2D/3D対応）
│
├── time_integration/
│   ├── __init__.py
│   ├── tvd_rk3.py            # TVD Runge-Kutta 3次（汎用）
│   └── cfl.py                # CFL条件（3項の最小値、式84）
│
├── simulation.py             # TwoPhaseSimulation — 5フェーズ統合
│
├── diagnostics/
│   ├── __init__.py
│   ├── monitors.py           # ∇·u, 体積保存, Eikonal 監視
│   └── io.py                 # HDF5/VTK 出力
│
└── tests/
    ├── test_ccd_convergence.py
    ├── test_levelset.py
    ├── test_hydrostatic.py
    └── test_capillary.py
```

---

## 3. コアデータ構造

### 3.1 SimulationConfig（`config.py`）

```python
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class SimulationConfig:
    # --- 次元 ---
    ndim: int = 2                     # 2 or 3

    # --- 格子 ---
    N: Tuple[int, ...] = (128, 128)   # 2D: (Nx,Ny), 3D: (Nx,Ny,Nz)
    L: Tuple[float, ...] = (1.0, 1.0) # 計算領域サイズ

    # --- 無次元物性値 ---
    Re: float = 100.0                  # Reynolds数
    Fr: float = 1.0                    # Froude数
    We: float = 10.0                   # Weber数
    rho_ratio: float = 0.001           # ρ̂ = ρ_g / ρ_l
    mu_ratio: float = 0.01             # μ̂ = μ_g / μ_l

    # --- 界面パラメータ ---
    epsilon_factor: float = 1.5        # ε = factor × Δx_min
    alpha_grid: float = 3.0            # 界面近傍格子密度倍率
    reinit_steps: int = 4              # 再初期化反復回数

    # --- 時間積分 ---
    cfl_number: float = 0.3
    t_end: float = 1.0

    # --- ソルバー ---
    bicgstab_tol: float = 1e-10
    bicgstab_maxiter: int = 1000

    # --- 境界条件 ---
    bc_type: str = 'wall'              # 'wall', 'periodic', 'inflow_outflow'

    # --- GPU ---
    use_gpu: bool = True

    def __post_init__(self):
        assert self.ndim in (2, 3)
        assert len(self.N) == self.ndim
        assert len(self.L) == self.ndim
```

### 3.2 Grid（`core/grid.py`）— 2D/3D統一

```python
class Grid:
    """
    コロケート格子 + 界面適合非一様格子
    ndim=2: (x, y) 方向  |  ndim=3: (x, y, z) 方向

    各軸のメトリクスを独立に管理:
      coords[axis]      : 物理座標 1D配列
      J[axis]           : メトリクス dξ/dx (1D配列)
      dJ_dxi[axis]      : ∂J/∂ξ (1D配列)
    """
    def __init__(self, config: SimulationConfig, backend: Backend):
        self.ndim = config.ndim
        self.N = config.N                 # (Nx, Ny) or (Nx, Ny, Nz)
        self.L = config.L
        xp = backend.xp

        # 軸名マッピング
        self.axis_names = ('x', 'y', 'z')[:self.ndim]

        # 各軸の座標・メトリクス
        self.coords = {}    # axis_idx -> 1D array
        self.J = {}         # axis_idx -> 1D array (dξ/dx)
        self.dJ_dxi = {}    # axis_idx -> 1D array (∂J/∂ξ)

        for ax in range(self.ndim):
            n = self.N[ax]
            self.coords[ax] = xp.linspace(0, self.L[ax], n + 1)
            self.J[ax] = xp.ones(n + 1)
            self.dJ_dxi[ax] = xp.zeros(n + 1)

    @property
    def shape(self):
        """場の配列形状: (Nx+1, Ny+1) or (Nx+1, Ny+1, Nz+1)"""
        return tuple(n + 1 for n in self.N)

    @property
    def dx_min(self):
        """全軸の最小格子幅"""
        return min(float((self.coords[ax][1:] - self.coords[ax][:-1]).min())
                   for ax in range(self.ndim))

    def update_from_levelset(self, phi, ccd):
        """界面適合格子の再生成（各軸独立にin-place更新）"""
        ...
```

### 3.3 ScalarField / VectorField（`core/field.py`）

```python
class ScalarField:
    """
    2D/3D対応スカラー場
    - data: GPU上の配列 shape = grid.shape
    - 微分値キャッシュ（遅延評価）
    """
    def __init__(self, grid: Grid, backend: Backend, name: str = ""):
        self.grid = grid
        self.xp = backend.xp
        self.name = name
        self.ndim = grid.ndim

        # メインデータ（GPU上、確保は1回のみ）
        self.data = self.xp.zeros(grid.shape)

        # CCD微分値キャッシュ（遅延評価）
        # d1[ax]: ∂f/∂x_ax,  d2[ax]: ∂²f/∂x_ax²
        # d2_mixed[(ax1,ax2)]: ∂²f/∂x_ax1∂x_ax2
        self.d1 = {}
        self.d2 = {}
        self.d2_mixed = {}
        self._dirty = True

    def invalidate(self):
        self._dirty = True

    def fill_from(self, other: 'ScalarField'):
        """in-placeコピー（配列再確保なし）"""
        self.xp.copyto(self.data, other.data)
        self.invalidate()


class VectorField:
    """
    2D: (u, v) の2成分  |  3D: (u, v, w) の3成分

    components[0] = u (x方向)
    components[1] = v (y方向)
    components[2] = w (z方向)  ← 3Dのみ
    """
    def __init__(self, grid: Grid, backend: Backend, name: str = ""):
        self.ndim = grid.ndim
        self.components = [
            ScalarField(grid, backend, name=f"{name}_{ax}")
            for ax in range(grid.ndim)
        ]

    def __getitem__(self, idx):
        return self.components[idx]

    @property
    def u(self): return self.components[0]
    @property
    def v(self): return self.components[1]
    @property
    def w(self): return self.components[2]  # 3D only

    def invalidate(self):
        for c in self.components:
            c.invalidate()
```

---

## 4. NS方程式の項別演算子（`ns_terms/`）

### 4.0 基底クラス

```python
class NSTerm:
    """NS方程式の1つの項を表す基底クラス

    すべての具象クラスは evaluate() で「その項の寄与」を
    出力バッファに in-place で加算する。

    論文との対応:
      (a) ConvectionTerm     : -u·∇u
      (b) ViscousTerm        : ∇·[μ̃(∇u+∇u^T)] / (ρ̃·Re)
      (c) GravityTerm        : -ẑ / Fr²
      (d) SurfaceTensionTerm : κδε∇φ / (ρ̃·We)
    """
    def __init__(self, grid, backend, config):
        self.grid = grid
        self.xp = backend.xp
        self.ndim = grid.ndim
        # 作業バッファを __init__ で確保

    def evaluate(self, state, out: VectorField, mode='add'):
        """
        state: 現在の全場変数を保持する辞書/オブジェクト
        out:   結果を加算する VectorField
        mode:  'add' (加算) or 'set' (上書き)
        """
        raise NotImplementedError
```

### 4.1 (a) ConvectionTerm（`ns_terms/convection.py`）

```python
class ConvectionTerm(NSTerm):
    """
    (a) 対流項: -(u·∇)u

    離散化: CCD O(h^6) で ∂u/∂x, ∂u/∂y, (∂u/∂z)
    時間積分: TVD-RK3（陽的）
    時刻: t^n のデータを使用
    """
    def evaluate(self, state, out, mode='add'):
        """
        各速度成分 u_k について:
          conv_k = Σ_ax u_ax · (∂u_k/∂x_ax)

        2D: conv_u = u·∂u/∂x + v·∂u/∂y
            conv_v = u·∂v/∂x + v·∂v/∂y

        3D: conv_u = u·∂u/∂x + v·∂u/∂y + w·∂u/∂z
            conv_v = u·∂v/∂x + v·∂v/∂y + w·∂v/∂z
            conv_w = u·∂w/∂x + v·∂w/∂y + w·∂w/∂z
        """
        vel = state.velocity   # VectorField
        ccd = state.ccd

        for k in range(self.ndim):     # 速度成分ループ
            uk = vel[k]
            uk.ensure_derivatives(ccd)  # CCD で ∂u_k/∂x_ax を計算

            conv = self._buf           # 事前確保バッファ
            conv[:] = 0
            for ax in range(self.ndim):
                # conv += u_ax * ∂u_k/∂x_ax
                conv += vel[ax].data * uk.d1[ax]

            if mode == 'add':
                out[k].data -= conv    # マイナス符号（-(u·∇)u）
            else:
                out[k].data[:] = -conv
            out[k].invalidate()
```

### 4.2 (b) ViscousTerm（`ns_terms/viscous.py`）

```python
class ViscousTerm(NSTerm):
    """
    (b) 粘性項: (1/ρ̃Re) ∇·[μ̃(∇u + ∇u^T)]

    対称応力テンソル (∇u)^sym の発散を展開:
    2D u成分: ∂(2μ̃·∂u/∂x)/∂x + ∂(μ̃(∂u/∂y + ∂v/∂x))/∂y
    3D u成分: ∂(2μ̃·∂u/∂x)/∂x + ∂(μ̃(∂u/∂y + ∂v/∂x))/∂y + ∂(μ̃(∂u/∂z + ∂w/∂x))/∂z

    離散化: CCD O(h^6)
    時間積分: Crank-Nicolson 半陰的
    """
    def evaluate(self, state, out, mode='add'):
        """変粘性・変密度の粘性力を計算"""
        ...

    def build_cn_matrix(self, state, dt):
        """
        Crank-Nicolson の線形系:
          [I - (Δt/2ρ̃Re)·V] u* = u^n + (Δt/2ρ̃Re)·V(u^n) + Δt·N(u^n)
        行列を構築して返す
        """
        ...
```

### 4.3 (c) GravityTerm（`ns_terms/gravity.py`）

```python
class GravityTerm(NSTerm):
    """
    (c) 重力項: -ẑ/Fr²

    NS方程式を ρ̃ で除算済みなので、
    ρ̃/ρ̃ = 1 となり単純な定数ベクトル。
    2D: (0, -1/Fr²)
    3D: (0, 0, -1/Fr²)

    最後の軸（y:2D, z:3D）方向のみ非ゼロ
    """
    def __init__(self, grid, backend, config):
        super().__init__(grid, backend, config)
        self.gravity_value = -1.0 / (config.Fr ** 2)
        self.gravity_axis = self.ndim - 1  # y(2D) or z(3D)

    def evaluate(self, state, out, mode='add'):
        ax = self.gravity_axis
        if mode == 'add':
            out[ax].data += self.gravity_value
        else:
            for k in range(self.ndim):
                out[k].data[:] = 0
            out[ax].data[:] = self.gravity_value
        out[ax].invalidate()
```

### 4.4 (d) SurfaceTensionTerm（`ns_terms/surface_tension.py`）

```python
class SurfaceTensionTerm(NSTerm):
    """
    (d) 表面張力項（CSF法）: κδε(∇φ) / (ρ̃·We)

    κ^{n+1} : 曲率（CCD逐次適用で計算済み、Phase 2で先行評価）
    δε(φ^{n+1}) : 滑らか化デルタ関数
    ∇φ^{n+1} : Level Set勾配（CCD）
    ρ̃^{n+1} : 密度（Phase 2で更新済み）

    半陰的: t^{n+1} の φ, ρ, κ を使用
    """
    def evaluate(self, state, out, mode='add'):
        """
        各速度成分 k について:
          surf_k = κ · δε · (∂φ/∂x_k) / (ρ̃ · We)
        """
        phi = state.phi
        kappa = state.kappa
        rho = state.rho
        We = state.config.We
        eps = state.epsilon

        delta = heaviside_delta(phi.data, eps, self.xp)  # δε(φ)

        for k in range(self.ndim):
            # φ の k方向微分（CCD計算済み）
            dphi_k = phi.d1[k]

            coeff = kappa.data * delta * dphi_k / (rho.data * We)

            if mode == 'add':
                out[k].data += coeff
            else:
                out[k].data[:] = coeff
            out[k].invalidate()
```

### 4.5 Predictor 統合（`ns_terms/predictor.py`）

```python
class Predictor:
    """
    ⑥ Predictor: (a)+(b)+(c)+(d) → u*

    論文 式(57):
      (u* - u^n)/Δt = (a) + (b) + (c) + (d)

    Crank-Nicolson 形式（式82）:
      u*/Δt - V(u*)/2ρ̃Re = u^n/Δt + N(u^n) + V(u^n)/2ρ̃Re
      ここで N = (a) + (c) + (d)（非線形・陽的部分）
    """
    def __init__(self, grid, backend, config):
        # ─── NS方程式の各項（論文ラベル付き）───
        self.term_a = ConvectionTerm(grid, backend, config)      # (a) 対流
        self.term_b = ViscousTerm(grid, backend, config)         # (b) 粘性
        self.term_c = GravityTerm(grid, backend, config)         # (c) 重力
        self.term_d = SurfaceTensionTerm(grid, backend, config)  # (d) 表面張力

        # 作業バッファ
        self._rhs = VectorField(grid, backend, "predictor_rhs")

    def compute(self, state, vel_star: VectorField, dt: float):
        """
        u* を計算して vel_star に書き込む

        Step 1: 陽的項を集約
          rhs = (a) + (c) + (d) + V(u^n)/(2ρ̃Re) + u^n/Δt

        Step 2: Crank-Nicolson 線形系を解く
          [I/(Δt) - V/(2ρ̃Re)] u* = rhs
        """
        rhs = self._rhs

        # ── (a) 対流項: -u^n·∇u^n（陽的、t^n データ使用）──
        self.term_a.evaluate(state, rhs, mode='set')

        # ── (c) 重力項: -ẑ/Fr²（定数）──
        self.term_c.evaluate(state, rhs, mode='add')

        # ── (d) 表面張力: κδε∇φ/(ρ̃We)（半陰的、t^{n+1} データ）──
        self.term_d.evaluate(state, rhs, mode='add')

        # ── (b) 粘性項: Crank-Nicolson 処理 ──
        # rhs += V(u^n)/(2ρ̃Re) + u^n/Δt
        # → [I/Δt - V/(2ρ̃Re)] u* = rhs を解く
        self.term_b.apply_crank_nicolson(state, rhs, vel_star, dt)
```

---

## 5. 圧力ステップ（`pressure/`）

### 5.1 Rhie-Chow 補正（`pressure/rhie_chow.py`）

```python
class RhieChowCorrection:
    """
    (e_R) Rhie-Chow 界面速度と RC発散の計算

    式(63):
      u^RC_{i+1/2} = (u*_i + u*_{i+1})/2
        - (Δt/ρ̃_{i+1/2}) [  (p_{i+1}-p_i)/hx
                            - (p^CCD_x,i + p^CCD_x,i+1)/2  ]

    2D: x,y 界面（4面）
    3D: x,y,z 界面（6面）

    式(70): RC発散
      div^RC = (u^RC_{i+1/2} - u^RC_{i-1/2})/hx
             + (v^RC_{j+1/2} - v^RC_{j-1/2})/hy
             + (w^RC_{k+1/2} - w^RC_{k-1/2})/hz   [3Dのみ]
    """
    def __init__(self, grid, backend):
        self.ndim = grid.ndim
        # 各軸の界面速度バッファ（+面、-面）
        # 形状: grid.shape（セル中心と同じ形状で管理、i+1/2 のデータを i に格納）
        self.face_vel = {}
        for ax in range(self.ndim):
            self.face_vel[ax] = backend.xp.zeros(grid.shape)

    def compute_face_velocities(self, vel_star, p, rho, ccd, dt):
        """全軸の界面速度 u^RC を計算（in-place）"""
        for ax in range(self.ndim):
            self._compute_face_vel_axis(ax, vel_star, p, rho, ccd, dt)

    def compute_divergence(self, out):
        """RC発散を out に書き込み（PPE右辺用）"""
        ...
```

### 5.2 PPE 疎行列構築（`pressure/ppe_builder.py`）

```python
class PPEMatrixBuilder:
    """
    (e_L) FVM変密度PPE行列の構築

    2D: 5点ステンシル（式69）
      (1/hx²)[p_{i+1}/ρ_{i+1/2} - p_i/ρ_{i+1/2} - p_i/ρ_{i-1/2} + p_{i-1}/ρ_{i-1/2}]
      + (1/hy²)[同様にy方向]

    3D: 7点ステンシル（式72）
      + (1/hz²)[同様にz方向]

    CSR疎行列で構築。密度更新時は values のみ in-place 書き換え。
    """
    def __init__(self, grid, backend):
        self.ndim = grid.ndim
        self._build_sparsity_pattern(grid)  # 非ゼロパターンは固定

    def update_coefficients(self, rho: ScalarField):
        """密度変化に伴い行列係数のみ更新（パターン不変）"""
        ...

    def _build_sparsity_pattern(self, grid):
        """CSR形式の row_ptr, col_idx を構築"""
        ...
```

---

## 6. Level Set（`levelset/`）

### 6.1 曲率計算（`levelset/curvature.py`）— 2D/3D 対応

```python
class CurvatureCalculator:
    """
    ⑤ 界面曲率 κ の計算

    2D（式8）:
      κ = -(φx²·φyy - 2φxφy·φxy + φy²·φxx) / (φx² + φy²)^{3/2}
      必要微分: φx, φy, φxx, φyy, φxy

    3D:
      κ = -∇·(∇φ/|∇φ|) を展開
      必要微分: φx,φy,φz, φxx,φyy,φzz, φxy,φxz,φyz
      = -(φy²+φz²)φxx + (φx²+φz²)φyy + (φx²+φy²)φzz
        - 2(φxφyφxy + φxφzφxz + φyφzφyz)
        ─────────────────────────────────────
              (φx² + φy² + φz²)^{3/2}

    CCD逐次適用で全微分値を O(h^6) で計算
    """
    def compute(self, phi: ScalarField, kappa: ScalarField, ccd):
        """κ を計算して kappa.data に書き込み"""
        phi.ensure_all_derivatives(ccd)  # d1, d2, d2_mixed すべて計算

        if self.ndim == 2:
            self._compute_2d(phi, kappa)
        else:
            self._compute_3d(phi, kappa)

    def _compute_2d(self, phi, kappa):
        px, py = phi.d1[0], phi.d1[1]
        pxx, pyy = phi.d2[0], phi.d2[1]
        pxy = phi.d2_mixed[(0, 1)]

        num = px**2 * pyy - 2*px*py*pxy + py**2 * pxx
        den = (px**2 + py**2) ** 1.5
        den = self.xp.maximum(den, 1e-12)  # ゼロ割防止
        kappa.data[:] = -num / den

    def _compute_3d(self, phi, kappa):
        px, py, pz = phi.d1[0], phi.d1[1], phi.d1[2]
        pxx, pyy, pzz = phi.d2[0], phi.d2[1], phi.d2[2]
        pxy = phi.d2_mixed[(0, 1)]
        pxz = phi.d2_mixed[(0, 2)]
        pyz = phi.d2_mixed[(1, 2)]

        num = ((py**2 + pz**2)*pxx + (px**2 + pz**2)*pyy + (px**2 + py**2)*pzz
               - 2*(px*py*pxy + px*pz*pxz + py*pz*pyz))
        den = (px**2 + py**2 + pz**2) ** 1.5
        den = self.xp.maximum(den, 1e-12)
        kappa.data[:] = -num / den
```

### 6.2 Godunov 上流勾配（`levelset/godunov.py`）— φ₀符号依存

```python
class GodunovGradient:
    """
    Godunov型上流勾配（φ₀符号依存、式81）

    s = sgn(φ₀)

    G_α² = max(s·D⁻_α, 0)² + min(s·D⁺_α, 0)²   (α = x,y,z)

    |∇φ|^uw = √(Σ_α G_α²)

    2D: α ∈ {x, y}
    3D: α ∈ {x, y, z}
    """
    def compute(self, phi: ScalarField, phi0: ScalarField, out: ScalarField):
        xp = self.xp
        s = xp.sign(phi0.data)  # sgn(φ₀)

        grad_sq = xp.zeros_like(phi.data)
        for ax in range(self.ndim):
            Dp = self._forward_diff(phi.data, ax)   # D⁺
            Dm = self._backward_diff(phi.data, ax)   # D⁻

            grad_sq += xp.maximum(s * Dm, 0) ** 2 + xp.minimum(s * Dp, 0) ** 2

        out.data[:] = xp.sqrt(grad_sq)
```

---

## 7. 5フェーズ構成のメインループ（`simulation.py`）

```python
class SimulationState:
    """全場変数を保持（演算子間のデータ受け渡し）"""
    def __init__(self, grid, backend, config):
        self.config = config
        self.ccd = CCDSolver(grid, backend)

        # ─── 場変数（GPU上に常駐） ───
        self.phi   = ScalarField(grid, backend, "phi")        # Level Set
        self.velocity = VectorField(grid, backend, "vel")     # u = (u,v) or (u,v,w)
        self.p     = ScalarField(grid, backend, "pressure")
        self.rho   = ScalarField(grid, backend, "density")
        self.mu    = ScalarField(grid, backend, "viscosity")
        self.kappa = ScalarField(grid, backend, "curvature")

        # 作業用
        self.vel_star = VectorField(grid, backend, "vel_star")
        self.epsilon = config.epsilon_factor * grid.dx_min


class TwoPhaseSimulation:
    """
    論文 §9 / 図1 の完全計算フロー

    5フェーズ構成:
      Phase 1: Level Set 更新（①②）
      Phase 2: 格子・物性・曲率更新（③④⑤）
      Phase 3: 運動量予測 Predictor（⑥）
      Phase 4: 圧力 Poisson 求解（⑦）
      Phase 5: 速度補正・収束確認（⑧⑨）
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.backend = Backend(config.use_gpu)
        self.grid = Grid(config, self.backend)
        self.state = SimulationState(self.grid, self.backend, config)

        # ─── Phase 1: Level Set ───
        self.ls_advect = LevelSetAdvection(self.grid, self.backend, config)
        self.ls_reinit = Reinitializer(self.grid, self.backend, config)

        # ─── Phase 2: 格子・物性・曲率 ───
        self.curvature = CurvatureCalculator(self.grid, self.backend)

        # ─── Phase 3: Predictor（NS各項を集約） ───
        self.predictor = Predictor(self.grid, self.backend, config)

        # ─── Phase 4: 圧力 ───
        self.rhie_chow = RhieChowCorrection(self.grid, self.backend)
        self.ppe_builder = PPEMatrixBuilder(self.grid, self.backend)
        self.ppe_solver = PPESolver(self.backend, config)

        # ─── Phase 5: 速度補正 ───
        self.vel_corrector = VelocityCorrector(self.grid, self.backend)

        # ─── 診断 ───
        self.monitor = DiagnosticMonitor(self.grid, self.backend)
        self.cfl = CFLCalculator(config)

        self.time = 0.0
        self.step = 0

    def step_forward(self):
        """1タイムステップ: t^n → t^{n+1}"""
        s = self.state
        dt = self.cfl.compute(s.velocity, s.rho, s.mu, self.grid)

        # ═══════════════════════════════════════════════════
        # Phase 1: Level Set 更新（界面追跡）  ①②
        # ═══════════════════════════════════════════════════
        # ① LS移流: ∂φ/∂t + u^n·∇φ = 0 → φ*
        self.ls_advect.advance(s.phi, s.velocity, dt, s.ccd)

        # ② 再初期化: sgn(φ₀)(|∇φ|^uw - 1) = 0 → φ^{n+1}
        self.ls_reinit.reinitialize(s.phi, s.ccd)

        # ═══════════════════════════════════════════════════
        # Phase 2: 格子・物性値・曲率更新  ③④⑤
        # ═══════════════════════════════════════════════════
        # ③ グリッド更新: φ^{n+1} → ω, Jx, Jy, (Jz)
        self.grid.update_from_levelset(s.phi, s.ccd)

        # ④ 物性値更新: ρ̃ = ρ̂ + (1-ρ̂)Hε(φ),  μ̃ = 同様
        update_properties(s.phi, s.rho, s.mu,
                          self.config, self.backend)

        # ⑤ 曲率計算: CCD逐次適用 → κ^{n+1}
        self.curvature.compute(s.phi, s.kappa, s.ccd)

        # ═══════════════════════════════════════════════════
        # Phase 3: 運動量予測 Predictor  ⑥
        #   (a)対流 + (b)粘性 + (c)重力 + (d)表面張力 → u*
        # ═══════════════════════════════════════════════════
        self.predictor.compute(s, s.vel_star, dt)

        # ═══════════════════════════════════════════════════
        # Phase 4: 圧力 Poisson 求解  ⑦
        #   (e_L) FVM変密度PPE  +  (e_R) Rhie-Chow発散
        # ═══════════════════════════════════════════════════
        # Rhie-Chow 界面速度 → RC発散（PPE右辺）
        self.rhie_chow.compute_face_velocities(
            s.vel_star, s.p, s.rho, s.ccd, dt)

        # PPE行列係数をρ^{n+1}で更新
        self.ppe_builder.update_coefficients(s.rho)

        # BiCGSTAB + ILU(0) で p^{n+1} を求解
        self.ppe_solver.solve(
            self.ppe_builder.matrix,
            self.rhie_chow, s.p, dt)

        # ═══════════════════════════════════════════════════
        # Phase 5: 速度補正 + 収束確認  ⑧⑨
        #   (f) u^{n+1} = u* - (Δt/ρ̃)∇p^{n+1}
        # ═══════════════════════════════════════════════════
        # ⑧ 速度補正（CCD圧力勾配 + Rhie-Chow界面速度）
        self.vel_corrector.correct(
            s.vel_star, s.velocity, s.p, s.rho, s.ccd, dt)

        # ⑨ 収束確認
        self.monitor.check_all(s)

        self.time += dt
        self.step += 1

    def run(self, output_interval=100):
        while self.time < self.config.t_end:
            self.step_forward()
            if self.step % output_interval == 0:
                self.monitor.report(self.time, self.step)
```

---

## 8. CCD ソルバーの 2D/3D バッチ並列設計

```
CCDSolver.solve_along_axis(data, axis, out_d1, out_d2)
                                  │
                          ┌───────┴────────┐
                          │ 軸方向に配列を   │
                          │ 転置して並べる   │
                          └───────┬────────┘
                                  │
                          ┌───────┴────────┐
                          │  バッチ並列      │
                          │  ブロック三重対角 │
                          │  ソルバー       │
                          └───────┬────────┘
                                  │
          2D例: x方向CCD            3D例: x方向CCD
          data[Nx+1, Ny+1]          data[Nx+1, Ny+1, Nz+1]
          → reshape (Nx+1, Ny+1)    → reshape (Nx+1, (Ny+1)*(Nz+1))
          バッチ数 = Ny+1            バッチ数 = (Ny+1)*(Nz+1)
          1カーネル呼び出し          1カーネル呼び出し

混合偏微分 (∂²f/∂x∂y) の計算:
  Step 1: y方向CCD → ∂f/∂y  [全x列を同時バッチ処理]
  Step 2: ∂f/∂y を「値」として x方向CCD → (∂f/∂y)^(1)_x = ∂²f/∂x∂y

3Dで必要な混合偏微分:
  φxy: y→x逐次  |  φxz: z→x逐次  |  φyz: z→y逐次
```

---

## 9. メモリレイアウトとGPU高速化

### 9.1 メモリ配置

```
┌──────────────── GPU メモリ ────────────────────┐
│                                                │
│ 【場変数】（確保1回、in-place更新のみ）          │
│  phi.data     [Nx+1, Ny+1, (Nz+1)]             │
│  vel[0].data  [Nx+1, Ny+1, (Nz+1)]  = u       │
│  vel[1].data  [Nx+1, Ny+1, (Nz+1)]  = v       │
│  vel[2].data  [Nx+1, Ny+1, (Nz+1)]  = w (3D)  │
│  p.data       [Nx+1, Ny+1, (Nz+1)]             │
│  rho.data     [Nx+1, Ny+1, (Nz+1)]             │
│  mu.data      [Nx+1, Ny+1, (Nz+1)]             │
│  kappa.data   [Nx+1, Ny+1, (Nz+1)]             │
│                                                │
│ 【微分値キャッシュ】（遅延評価、変更時に無効化） │
│  phi.d1[0..ndim-1]   各方向1階微分              │
│  phi.d2[0..ndim-1]   各方向2階微分              │
│  phi.d2_mixed[...]   混合偏微分                 │
│                                                │
│ 【演算子内バッファ】（__init__で確保、再利用）    │
│  Predictor._rhs                                 │
│  RhieChow.face_vel[0..ndim-1]                   │
│  PPE._rhs_vec, _sol_vec                         │
│  CCD._rhs_buf, _sol_buf                         │
│                                                │
│ 【疎行列】                                      │
│  PPE L_FVM  (CSR, パターン固定、値のみ更新)      │
│                                                │
│ GPU → CPU 転送: 診断スカラー値のみ（5〜10個）   │
└────────────────────────────────────────────────┘
```

### 9.2 カーネル融合例

```python
# ④ 物性値更新: Hε→ρ,μ を1カーネルで計算
update_properties_kernel = cp.ElementwiseKernel(
    'float64 phi, float64 eps, float64 rr, float64 mr',
    'float64 rho, float64 mu',
    '''
    double H;
    if (phi < -eps) H = 0.0;
    else if (phi > eps) H = 1.0;
    else H = 0.5*(1.0 + phi/eps + sin(M_PI*phi/eps)/M_PI);
    rho = rr + (1.0 - rr) * H;
    mu  = mr + (1.0 - mr) * H;
    ''',
    'fused_update_properties'
)
```

---

## 10. 論文→コード 対応表

| 論文ラベル | 数式 | コードクラス | モジュール |
|-----------|------|-------------|-----------|
| **(a) 対流** | $-\mathbf{u}\cdot\nabla\mathbf{u}$ | `ConvectionTerm` | `ns_terms/convection.py` |
| **(b) 粘性** | $\nabla\cdot[\tilde\mu(\nabla\mathbf{u})^{\rm sym}]/(\tilde\rho Re)$ | `ViscousTerm` | `ns_terms/viscous.py` |
| **(c) 重力** | $-\hat{\mathbf{z}}/Fr^2$ | `GravityTerm` | `ns_terms/gravity.py` |
| **(d) 表面張力** | $\kappa\delta_\varepsilon\nabla\phi/(\tilde\rho We)$ | `SurfaceTensionTerm` | `ns_terms/surface_tension.py` |
| **(e_L) PPE左辺** | FVM変密度ラプラシアン | `PPEMatrixBuilder` | `pressure/ppe_builder.py` |
| **(e_R) PPE右辺** | Rhie-Chow発散 | `RhieChowCorrection` | `pressure/rhie_chow.py` |
| **(f) 速度補正** | $\mathbf{u}^{n+1}=\mathbf{u}^*-(\Delta t/\tilde\rho)\nabla p$ | `VelocityCorrector` | `pressure/velocity_corrector.py` |
| **① LS移流** | $\partial\phi/\partial t + \mathbf{u}\cdot\nabla\phi = 0$ | `LevelSetAdvection` | `levelset/advection.py` |
| **② 再初期化** | $\partial\phi/\partial\tau + {\rm sgn}(\phi_0)(|\nabla\phi|^{\rm uw}-1)=0$ | `Reinitializer` | `levelset/reinitialize.py` |
| **③ グリッド更新** | $\phi\to\omega,J$ | `Grid.update_from_levelset` | `core/grid.py` |
| **④ 物性値更新** | $\tilde\rho=\hat\rho+(1-\hat\rho)H_\varepsilon$ | `update_properties` | `levelset/heaviside.py` |
| **⑤ 曲率計算** | $\kappa=-\nabla\cdot(\nabla\phi/|\nabla\phi|)$ | `CurvatureCalculator` | `levelset/curvature.py` |

---

## 11. 実装ロードマップ

| Phase | 内容 | 検証テスト | 2D/3D |
|-------|------|-----------|-------|
| **1** | `Backend` + `Grid` + `ScalarField` + `CCDSolver` | sin(x) 収束 O(h^6) | 2D先行 |
| **2** | `LevelSetAdvection` + `Reinitializer` + `GodunovGradient` | 円/球の移流・体積保存 | 2D/3D |
| **3** | `CurvatureCalculator` | 円/球の曲率=1/R の検証 | 2D/3D |
| **4** | `Predictor`（(a)(b)(c)(d) 全NS項） | Lid-driven cavity（単相） | 2D |
| **5** | `PPE` + `RhieChow` + `VelocityCorrector` | 非圧縮性 ∇·u → 0 | 2D |
| **6** | 全結合 `TwoPhaseSimulation` | 静水圧テスト、気泡上昇 | 2D |
| **7** | 3D 完全対応 + 界面適合格子 | 3D気泡上昇 | 3D |
| **8** | GPU最適化（カスタムCUDAカーネル） | ベンチマーク | 2D/3D |

---

## 12. 使用例

```python
from twophase import SimulationConfig, TwoPhaseSimulation

# ── 2D 気泡上昇 ──
config_2d = SimulationConfig(
    ndim=2, N=(256, 512), L=(1.0, 2.0),
    Re=100, Fr=1.0, We=10.0,
    rho_ratio=0.001, mu_ratio=0.01,
    cfl_number=0.3, t_end=5.0
)
sim_2d = TwoPhaseSimulation(config_2d)
# 初期条件: 円形気泡
sim_2d.state.phi.data = xp.sqrt((X-0.5)**2 + (Y-0.5)**2) - 0.25
sim_2d.run()

# ── 3D 気泡上昇 ──
config_3d = SimulationConfig(
    ndim=3, N=(64, 64, 128), L=(1.0, 1.0, 2.0),
    Re=100, Fr=1.0, We=10.0,
    rho_ratio=0.001, mu_ratio=0.01,
    cfl_number=0.3, t_end=2.0
)
sim_3d = TwoPhaseSimulation(config_3d)
# 初期条件: 球形気泡
sim_3d.state.phi.data = xp.sqrt((X-0.5)**2 + (Y-0.5)**2 + (Z-0.5)**2) - 0.25
sim_3d.run()
```
