"""05: 手法選択の意思決定順序 — 速度/精度/サイズのトレードオフを順番に詰める。

最適化は「全部盛り」ではなく*順番*が大事。コストの低い・壊れにくい手から試し、
各段階で必ず計測(速度・サイズ・精度)して、目標に届いたら止める。やみくもに
量子化や pruning から始めると、精度を壊した割に速くならない事故が起きる。

推奨の意思決定順序(上から順に試す):
  0. 計測ファースト          : まず現状を p50/p99・スループット・精度・サイズで把握。
  1. eval() + inference_mode : 勾配・Dropout/BN 学習挙動を切る。タダで効く・必須。
  2. スレッド数の調整        : torch.set_num_threads / ORT intra_op を実機で探索。
  3. bf16 autocast(+compile): CPU は bf16。compile はツールチェインがあれば。
  4. ONNX Runtime(fp32)    : グラフ最適化+専用カーネル。多くの場合 eager より速い。
  5. ONNX 動的量子化(int8) : サイズ ~1/4。Linear/MatMul 主体で速度も伸びる(CNN は要実測)。
  6. 静的 PTQ(キャリブレーション): Conv 主体でも効かせたいとき。精度を測りながら。
  7. QAT / 構造化 pruning    : ここまでで足りない時の最後の手段(学習コスト高)。
  8. エッジ専用ランタイム     : OpenVINO/CoreML/LiteRT/TensorRT で最終仕上げ。

このスクリプトは『目標(latency/size/精度)』を入力に、上の順序で
「次に試すべき手」を判定する小さな意思決定エンジンを実演する。

実行:
  uv run python lectures/37_runtime_edge_optimization/05_decision_order.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_lab import ensure_output_dir  # noqa: E402


@dataclass
class Target:
    """最適化のゴール。"""

    max_latency_ms: float
    max_size_mb: float
    min_top1_keep: float  # fp32 基準に対する許容精度保持率(例 0.99)


@dataclass
class State:
    """現状のベンチ値(各ステップ後に更新していく想定)。"""

    latency_ms: float
    size_mb: float
    top1_keep: float
    eval_mode: bool = False
    onnx: bool = False
    quantized: bool = False


def next_action(state: State, target: Target) -> str:
    """現状と目標から「次に試すべき1手」を早期 return で決める(意思決定順序の核)。"""
    # まず壊れていないか(精度)を最優先で守る。割っていたら緩める方向。
    if state.top1_keep < target.min_top1_keep:
        if state.quantized:
            return "精度が目標未満。量子化を緩める(per-channel/静的PTQ/QAT へ)か量子化を外す。"
        return "精度が目標未満。最適化を一段戻して原因(層/演算)を特定する。"

    met_latency = state.latency_ms <= target.max_latency_ms
    met_size = state.size_mb <= target.max_size_mb
    if met_latency and met_size:
        return "目標達成 🎉 これ以上は触らない(過剰最適化は精度/保守性を損なう)。"

    # 効果が大きく壊れにくい手から順に提案する(早期 return で1手に絞る)。
    if not state.eval_mode:
        return "STEP1: model.eval()+inference_mode を入れる(タダ・必須)。"
    if not met_latency and not state.onnx:
        return "STEP4: ONNX Runtime(fp32, graph最適化ALL)へ。多くの場合まず速くなる。"
    if not met_size and not state.quantized:
        return "STEP5: ONNX 動的量子化(int8)。サイズ ~1/4。CNN は速度を実測で確認。"
    if not met_latency and state.onnx and not state.quantized:
        return "STEP5/6: 量子化(動的→静的PTQ)。速度が伸びるか精度とセットで実測。"
    return "STEP7+: QAT/構造化pruning や エッジ専用ランタイム(OpenVINO/CoreML)を検討。"


def run_scenario(name: str, states: list[State], target: Target) -> None:
    """1シナリオを順に流し、各段階での推奨アクションを表示する。"""
    print(f"\n### シナリオ: {name}")
    print(f"  目標: latency<={target.max_latency_ms}ms, size<={target.max_size_mb}MB, "
          f"精度保持>={target.min_top1_keep:.0%}")
    for i, st in enumerate(states):
        ok_l = "OK" if st.latency_ms <= target.max_latency_ms else "未"
        ok_s = "OK" if st.size_mb <= target.max_size_mb else "未"
        ok_a = "OK" if st.top1_keep >= target.min_top1_keep else "割れ"
        print(f"  [{i}] lat={st.latency_ms:5.1f}ms({ok_l}) size={st.size_mb:5.1f}MB({ok_s}) "
              f"精度保持={st.top1_keep:.1%}({ok_a})")
        print(f"       → 次の一手: {next_action(st, target)}")


def main() -> None:
    ensure_output_dir()
    print("手法選択の意思決定順序(コスト低→高・壊れにくい順に試し、目標到達で止める)")

    # シナリオ A: レイテンシ削減が目的(精度は守りたい)。
    target_a = Target(max_latency_ms=6.0, max_size_mb=50.0, min_top1_keep=0.99)
    states_a = [
        State(latency_ms=9.8, size_mb=46.8, top1_keep=1.00, eval_mode=False),
        State(latency_ms=9.0, size_mb=46.8, top1_keep=1.00, eval_mode=True),
        State(latency_ms=5.8, size_mb=46.7, top1_keep=1.00, eval_mode=True, onnx=True),
    ]
    run_scenario("レイテンシ最優先(サーバ CPU 推論)", states_a, target_a)

    # シナリオ B: サイズ最優先(エッジ配布)。量子化で精度が割れる例を含む。
    target_b = Target(max_latency_ms=30.0, max_size_mb=15.0, min_top1_keep=0.98)
    states_b = [
        State(latency_ms=9.8, size_mb=46.8, top1_keep=1.00, eval_mode=True),
        State(latency_ms=16.5, size_mb=11.7, top1_keep=0.995, eval_mode=True, onnx=True, quantized=True),
        State(latency_ms=12.0, size_mb=12.0, top1_keep=0.965, eval_mode=True, onnx=True, quantized=True),
    ]
    run_scenario("サイズ最優先(エッジ配布)", states_b, target_b)

    print("\n[まとめ]")
    print("  - 順序が命: eval→スレッド→bf16/compile→ONNX→量子化→(静的PTQ/QAT/pruning)→エッジ専用。")
    print("  - 各段で必ず計測。『精度保持』を最優先のガードにして、割れたら一段戻す。")
    print("  - 目標を満たしたら止める。過剰最適化は精度・保守性・移植性を削る。")


if __name__ == "__main__":
    main()
