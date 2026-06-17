"""01: SigLIP と CLIP の違い — sigmoid 損失 vs softmax 対照損失。

学ぶこと:
  - CLIP と SigLIP はどちらも「画像とテキストを同じ空間へ写す」対照学習モデルだが、
    損失設計が違う:
      * CLIP   … バッチ内で「正しい対だけ 1、他は 0」を softmax で当てる（相互排他）。
                 → 推論では logits_per_image に softmax を掛け、ラベル間で和が 1 の確率にする。
      * SigLIP … 各 (画像, テキスト) 対を独立に「対応する/しない」の2値で判定（sigmoid）。
                 → 推論では sigmoid を掛け、各ラベルが独立に 0〜1 の確率になる（和は 1 でない）。
  - この違いは「確率の読み方」に直結する。softmax は “候補の中でどれ” を、sigmoid は
    “各ラベルが当てはまるか” を答える。ゼロショット分類の出力解釈で混同しやすい最重要点。

実行:
  uv run python lectures/33_multimodal_embeddings/01_siglip_vs_clip.py
結果は lectures/33_multimodal_embeddings/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from siglip_lab import (  # noqa: E402
    MMEncoder,
    ensure_output_dir,
    make_shape_image,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    """行ごとの softmax（ラベル方向で和が 1）。CLIP の確率化に使う。"""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """要素ごとの sigmoid（各ラベル独立で 0〜1）。SigLIP の確率化に使う。"""
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    out = ensure_output_dir()

    # === 1. 題材の画像とラベルを用意 =====================================
    # 「赤い円」を 1 枚。ラベルには “当たり”(赤い円) と “近いが違う”(赤い四角・青い円) と
    #  “全然違う”(緑の三角) を混ぜ、確率の付き方の違いを観察する。
    image = make_shape_image("red", "circle")
    labels = [
        "a photo of a red circle",  # 正解
        "a photo of a red square",  # 色は合うが形が違う
        "a photo of a blue circle",  # 形は合うが色が違う
        "a photo of a green triangle",  # 全然違う
    ]
    print(f"[1] 画像 = 赤い円 / ラベル {len(labels)} 件で CLIP と SigLIP を比較\n")

    # === 2. CLIP: logits → softmax（相互排他の確率） ====================
    clip = MMEncoder("clip")
    clip_logits = clip.logits_per_image([image], labels)  # (1, 4)
    clip_softmax = _softmax(clip_logits)[0]
    clip_sigmoid = _sigmoid(clip_logits)[0]  # 参考: CLIP に sigmoid を当てると解釈が壊れる例

    # === 3. SigLIP: logits → sigmoid（独立な確率） ======================
    siglip = MMEncoder("siglip")
    siglip_logits = siglip.logits_per_image([image], labels)  # (1, 4)
    siglip_sigmoid = _sigmoid(siglip_logits)[0]
    siglip_softmax = _softmax(siglip_logits)[0]  # 参考: SigLIP に softmax は本来不適

    # === 4. 並べて表示 ===================================================
    print("[2] CLIP（正準: softmax）— ラベル間で和が 1 の『どれ？』確率")
    _print_row(labels, clip_softmax, total=True)
    print("\n[3] SigLIP（正準: sigmoid）— 各ラベル独立の『当てはまる？』確率")
    _print_row(labels, siglip_sigmoid, total=True)

    print("\n[4] わざと取り違えた場合（確率解釈が壊れる例）:")
    print("    CLIP に sigmoid を当てる → 和が 1 にならず『候補内での比較』が消える:")
    _print_row(labels, clip_sigmoid, total=True, indent="      ")
    print("    SigLIP に softmax を当てる → 独立判定を無理に1へ押し込め、自信過剰になりがち:")
    _print_row(labels, siglip_softmax, total=True, indent="      ")

    # === 5. 図に保存 =====================================================
    _save_bars(out, labels, clip_softmax, siglip_sigmoid, image)
    print("\n[5] 図を保存:", (out / "01_sigmoid_vs_softmax.png").name)

    # === 6. まとめ =======================================================
    print("\n[まとめ]")
    print("  - CLIP=softmax: 『提示した候補の中でどれが最も近いか』を answer する（相互排他）。")
    print("  - SigLIP=sigmoid: 『各ラベルが当てはまるか』を独立に answer する（多ラベル可・和≠1）。")
    print("  - sigmoid なら『どれも当てはまらない(全部低い)』『複数当てはまる』も自然に表せる。")


def _print_row(labels: list[str], probs: np.ndarray, total: bool = False, indent: str = "    ") -> None:
    """ラベルと確率を整列表示する小ヘルパ。"""
    for lab, p in zip(labels, probs):
        bar = "#" * int(round(p * 20))
        print(f"{indent}{p:5.3f} |{bar:<20}| {lab}")
    if total:
        print(f"{indent}（合計 = {float(np.sum(probs)):.3f}）")


def _save_bars(
    out: pathlib.Path, labels, clip_softmax: np.ndarray, siglip_sigmoid: np.ndarray, image
) -> None:
    """CLIP(softmax) と SigLIP(sigmoid) の確率を横並び棒グラフ＋クエリ画像で保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [lab.replace("a photo of a ", "") for lab in labels]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 2, 2]})
    axes[0].imshow(image)
    axes[0].set_title("query: red circle", fontsize=10)
    axes[0].axis("off")
    axes[1].barh(y, clip_softmax, color="tab:blue")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(short, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title("CLIP softmax (sum=1, mutually exclusive)")
    axes[2].barh(y, siglip_sigmoid, color="tab:green")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(short, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_title("SigLIP sigmoid (independent per label)")
    fig.tight_layout()
    fig.savefig(out / "01_sigmoid_vs_softmax.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
