# Newton cloth_franka 環境構築 & CUDA Graph Capture エラー修正

## 概要

Newton物理エンジン (https://github.com/MY-CODE-1981/newton) を Isaac Sim 4.5.0 環境にセットアップし、
`cloth_franka` サンプル（Frankaロボットによる布の折り畳みシミュレーション）を実行可能にした。

## 環境

- OS: Ubuntu (Linux 5.15.0-139-generic)
- GPU: NVIDIA RTX 6000 Ada Generation (48GB VRAM)
- CUDA Driver: 535.183.01 (CUDA 12.2)
- Warp: 1.12.0.dev20260127 (内蔵 CUDA Toolkit 12.9)
- Python: 3.11.13 (uv管理)
- Newton: 0.2.0

## セットアップ手順

```bash
cd /home/initial/.local/share/ov/pkg/isaac-sim-4.5.0/cloth_folding_data_collection-main_20260206_newton/
git clone https://github.com/MY-CODE-1981/newton.git
cd newton
uv sync --extra examples
```

## 実行方法

```bash
# GUI付き
uv run -m newton.examples cloth_franka

# ヘッドレス（フレーム数指定必須、--viewer null を使う）
uv run -m newton.examples cloth_franka --viewer null --num-frames 100
```

> **注意**: `--headless` はGLビューワーがフレーム数で終了しないため無限ループする。
> ヘッドレス実行には必ず `--viewer null` を使用すること。

## 発生した問題

### CUDA error 900: operation not permitted when stream is capturing

**症状**: 初回実行時に以下のエラーで落ちる。

```
Warp CUDA error 900: operation not permitted when stream is capturing
Exception: Failed to load CUDA module 'newton._src.geometry.contact_reduction_global'
```

**原因**: `Example.__init__` の末尾で CUDA Graph Capture (`self.capture()`) が呼ばれる。
この中で `simulate()` が実行されるが、`simulate()` 内の `model.collide()` は初期化時とは
**異なるパラメータ** (`soft_contact_margin=self.cloth_body_contact_margin`) で呼ばれる。
これにより `contact_reduction_global` モジュールの新しい変種（ハッシュ `fe4de47`）の
JITコンパイルが発生するが、CUDA Graph Capture 中は新しいモジュールのロードが禁止されているため
エラーとなる。

同様に、VBDソルバーの `step()` も初めて呼ばれるカーネルが複数あり、capture中にJITが発生する。

**修正**: Graph Capture の前に、`simulate()` が使う全コードパスを1回実行して
JITコンパイルを事前に済ませる（ウォームアップ）。

```python
# Warm up: trigger JIT compilation of all kernel variants that
# simulate() will need, so they are ready before graph capture.
if self.add_cloth:
    if self.collision_pipeline is not None:
        self.collision_pipeline.soft_contact_margin = self.cloth_body_contact_margin
    self.model.collide(
        self.state_0,
        collision_pipeline=self.collision_pipeline,
        soft_contact_margin=self.cloth_body_contact_margin,
    )
    self.cloth_solver.rebuild_bvh(self.state_0)
    self.cloth_solver.step(
        self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
    )
    # Reset states after warmup
    self.state_0 = self.model.state()
    self.state_1 = self.model.state()
    newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
    self.sim_time = 0.0

# graph capture (now safe - all kernels pre-compiled)
if self.add_cloth:
    self.capture()
```

## 試行錯誤で得た知見

### Graph Capture 無効化は非推奨

Graph Capture を `self.graph = None` で無効化すると動作はするが、以下の問題が発生する:

1. **sim_time 二重加算**: `simulate()` 内で `sim_time += sim_dt` × substeps 分加算され、
   さらに `step()` で `sim_time += frame_dt` が加算される。Graph Capture 時は
   `simulate()` 内のPython文が実行されないため問題ないが、無効化すると2倍速で進む。
   → ロボットのキーポーズ遷移が狂い、布を正しく把持できない。

2. **パフォーマンス低下**: GPU Graph なしではCPU-GPU同期が毎substep発生し遅くなる。

### 接触パラメータ調整の試行

| 試行 | 結果 |
|------|------|
| `soft_contact_ke` 200→500, `kd` 2e-3→1e-2 | 把持力は改善するが、リリース時に布がグリッパーに粘着する（kd増加が原因） |
| `cloth_particle_radius` 0.008→0.012 | 衝突半径が大きすぎてグリッパーが閉じられない |
| `cloth_body_contact_margin` 0.01→0.03 | 効果薄。根本的に貫通は防げない |
| `soft_contact_ke` 2000, substeps 30 | 貫通は減るが完全には防げず、計算コスト大幅増 |
| `robot_friction` 1.0 を shape_material_mu に適用 | 把持力は改善するが貫通問題は解決しない |

**結論**: ペナルティベースの接触モデルでは、運動学的に駆動されるグリッパーの貫通を
完全に防ぐことはできない。Graph Capture を正しく動作させて元のパラメータで使うのが最善。

## 修正ファイル

- `newton/examples/cloth/example_cloth_franka.py`
  - ウォームアップ処理追加（L289-L305付近）
  - `step()` の sim_time 更新を graph/non-graph で分岐（L531-L537付近）

## 初回実行時の注意

初回実行時は約90個のWarpカーネルのJITコンパイルが発生し、5-10分かかる。
2回目以降は `~/.cache/warp/` にキャッシュされ、数秒で起動する。
キャッシュを消した場合 (`rm -rf ~/.cache/warp/`) は再度コンパイルが必要。
