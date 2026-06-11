"""第35回 演習問題（量子化と枝刈り / 三角関係で選ぶ圧縮）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/35_quantization_pruning/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/35_quantization_pruning/exercises_solutions.py

テーマ（易→難）:
  1 affine scale/zero_point   2 量子化→逆量子化   3 往復MAE
  4 per-channel scale         5 重みスパース率     6 非構造化(magnitude)枝刈り
  7 int8 で節約できるバイト数 8 構造化枝刈りの残す行 9 手法選択の意思決定

採点は決定的な合成テンソルだけで行うので、モデル DL も学習も不要で速く再現的。
実モデルへの適用(量子化/プルーニング)は 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

import torch


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_affine_scale_zero_point(x_min: float, x_max: float) -> tuple[float, int]:
    """演習1: 非対称(affine) int8(uint8, qmin=0/qmax=255) の scale と zero_point を返す。

      scale = (x_max - x_min) / (qmax - qmin)
      zero_point = round(qmin - x_min/scale) を [qmin, qmax] にクランプ
    x_max==x_min の退化時は scale=1.0, zero_point=0 を返すこと。
    """
    # TODO: qmin=0, qmax=255 で scale と zero_point を計算して返す
    raise NotImplementedError


def ex2_quantize_dequantize(x: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """演習2: x を uint8(0..255) に量子化→逆量子化した復元テンソル x̂ を返す。

      q   = clamp(round(x/scale) + zero_point, 0, 255)
      x̂ = scale * (q - zero_point)
    """
    # TODO: 量子化してから逆量子化した値を返す
    raise NotImplementedError


def ex3_roundtrip_mae(x: torch.Tensor, num_bits: int) -> float:
    """演習3: 対称(symmetric) num_bits 量子化の往復 平均絶対誤差(MAE) を返す。

      qmax = 2**(num_bits-1) - 1, zero_point=0
      scale = max(|x|) / qmax
      q = clamp(round(x/scale), -qmax-1, qmax) → x̂ = scale*q → mean(|x - x̂|)
    """
    # TODO: 対称量子化で往復させ、平均絶対誤差(float)を返す
    raise NotImplementedError


def ex4_per_channel_scales(w: torch.Tensor) -> torch.Tensor:
    """演習4: 重み行列 w(out,in) の per-channel(行ごと) 対称int8 scale を返す（形 (out,)）。

      各行 i について scale_i = max(|w[i]|) / 127。max(|w[i]|)==0 の行は scale=1.0。
    """
    # TODO: 行ごとに max(|w|)/127 を計算して (out,) のテンソルで返す
    raise NotImplementedError


def ex5_weight_sparsity(weight: torch.Tensor) -> float:
    """演習5: 重みテンソルのスパース率（値が 0 の要素の割合, 0〜1）を返す。"""
    # TODO: (weight==0) の割合を float で返す
    raise NotImplementedError


def ex6_magnitude_prune(weight: torch.Tensor, amount: float) -> torch.Tensor:
    """演習6: |w| が小さい順に amount 割合を 0 にした、**同じ形**のテンソルを返す。

    これが非構造化(unstructured)枝刈りの中身。形・dtype は変えない（密のまま）。
    amount=0 なら原本のコピー、amount>=1 なら全 0。閾値は |w| の amount 分位点を使う。
    同値境界の都合で実際の 0 の割合は概ね amount になればよい。
    """
    # TODO: |w| の amount 分位点を閾値に、それ以下を 0 にした同形テンソルを返す
    raise NotImplementedError


def ex7_int8_bytes_saved(num_params: int) -> int:
    """演習7: num_params 個の重みを fp32→int8 にしたとき節約できるバイト数を返す。

    fp32 = 4 byte/要素、int8 = 1 byte/要素。差は 1 要素あたり 3 byte。
    """
    # TODO: num_params * (4 - 1) を返す
    raise NotImplementedError


def ex8_structured_keep_rows(weight: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """演習8: 構造化枝刈りで「残す出力ニューロン(行)」の index を**昇順**で返す。

      各行の L2 ノルムで採点し、上位 keep = max(1, round(out*keep_frac)) 行を残す。
      返り値は long テンソル（昇順ソート済み）。
    """
    # TODO: 行ごと L2 ノルム → topk(keep) → 昇順ソートした index を返す
    raise NotImplementedError


def ex9_recommend_method(model_kind: str, has_calibration: bool, accuracy_drop_ok: bool) -> str:
    """演習9: 状況に応じた量子化手法を返す（"dynamic" / "static" / "qat" のいずれか）。

    ルール:
      - model_kind=="linear"（Linear/Transformer 主体）→ "dynamic"（即効・キャリブ不要）
      - それ以外(cnn)で、キャリブデータがあり精度劣化を許容できる → "static"
      - 上記以外（精度劣化を許容できない、またはキャリブデータが無い）→ "qat"
    """
    # TODO: 上のルールに従って手法名を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全PASSなら True。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def c1():
        s, zp = funcs["ex1"](-2.0, 6.0)  # 範囲幅8 → scale=8/255, 0.0 が zp=64 に対応
        ok = abs(s - 8 / 255) < 1e-6 and zp == 64
        s0, zp0 = funcs["ex1"](3.0, 3.0)  # 退化
        return (ok and s0 == 1.0 and zp0 == 0, f"scale={s:.5f} zp={zp}")

    def c2():
        x = torch.tensor([0.0, 1.0, 2.0, 6.0])
        xhat = funcs["ex2"](x, 6.0 / 255.0, 0)
        return (xhat.shape == x.shape and float((x - xhat).abs().max()) < 0.05,
                f"max|x-x̂|={(x - xhat).abs().max():.4f}")

    def c3():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2048, generator=g)
        mae8 = funcs["ex3"](x, 8)
        mae2 = funcs["ex3"](x, 2)
        return (mae8 < 0.02 and mae2 > mae8 * 5, f"MAE int8={mae8:.4f} int2={mae2:.4f}")

    def c4():
        w = torch.tensor([[0.0, 0.0, 0.0], [1.27, -0.5, 0.3], [254.0, 0.0, 0.0]])
        s = funcs["ex4"](w)
        return (s.shape == (3,) and abs(float(s[0]) - 1.0) < 1e-6
                and abs(float(s[1]) - 1.27 / 127) < 1e-6 and abs(float(s[2]) - 2.0) < 1e-6,
                f"scales={[round(float(v), 4) for v in s]}")

    def c5():
        w = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        return (abs(funcs["ex5"](w) - 0.75) < 1e-9, f"sparsity={funcs['ex5'](w):.3f}")

    def c6():
        g = torch.Generator().manual_seed(1)
        w = torch.randn(64, 64, generator=g)
        out = funcs["ex6"](w, 0.5)
        sp = float((out == 0).float().mean())
        # 形は不変、0 の割合は概ね 0.5、生き残ったのは大きい重み
        kept_min = out[out != 0].abs().min() if (out != 0).any() else torch.tensor(0.0)
        zeroed_max = w[out == 0].abs().max()
        return (out.shape == w.shape and abs(sp - 0.5) < 0.05 and kept_min >= zeroed_max - 1e-6,
                f"shape={tuple(out.shape)} sparsity={sp:.3f}")

    def c7():
        return (funcs["ex7"](1_000_000) == 3_000_000, f"saved={funcs['ex7'](1_000_000)} bytes")

    def c8():
        # 行ノルムが 行index と共に単調増加する重み → 上位 keep を残す
        w = torch.stack([torch.full((4,), float(i)) for i in range(10)])  # row i のノルム ∝ i
        idx = funcs["ex8"](w, 0.3)  # 10*0.3=3 → 残すのはノルム最大の行 7,8,9
        return (idx.tolist() == [7, 8, 9], f"keep={idx.tolist()}")

    def c9():
        f = funcs["ex9"]
        cases = {
            ("linear", False, True): "dynamic",
            ("cnn", True, True): "static",
            ("cnn", True, False): "qat",
            ("cnn", False, True): "qat",
        }
        got = {k: f(*k) for k in cases}
        return (got == cases, f"decisions={list(got.values())}")

    check("ex1_affine_scale_zero_point", c1)
    check("ex2_quantize_dequantize", c2)
    check("ex3_roundtrip_mae", c3)
    check("ex4_per_channel_scales", c4)
    check("ex5_weight_sparsity", c5)
    check("ex6_magnitude_prune", c6)
    check("ex7_int8_bytes_saved", c7)
    check("ex8_structured_keep_rows", c8)
    check("ex9_recommend_method", c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:30s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
    return {
        "ex1": ex1_affine_scale_zero_point,
        "ex2": ex2_quantize_dequantize,
        "ex3": ex3_roundtrip_mae,
        "ex4": ex4_per_channel_scales,
        "ex5": ex5_weight_sparsity,
        "ex6": ex6_magnitude_prune,
        "ex7": ex7_int8_bytes_saved,
        "ex8": ex8_structured_keep_rows,
        "ex9": ex9_recommend_method,
    }


if __name__ == "__main__":
    grade(current_funcs())
