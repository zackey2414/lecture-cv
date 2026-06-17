"""05: ImageBind の発想 — 音声・画像・テキストを「1つの空間」に束ねる（概念デモ）。

なぜ概念デモか:
  ImageBind は画像・テキスト・音声・深度・IMU など複数モダリティを『1つの埋め込み空間』
  に束ねるモデルで、音声クエリで画像を引く（音→画像）クロスモーダル検索ができる。ただし
  公式 PyPI が無く git + torchaudio 依存が重いため、本講座では実行経路には入れない
  （= 重い/壊れやすい依存を必須にしない方針）。代わりに「束ねるとは何が嬉しいのか」を
  トイ空間で忠実に再現して体感する。最後に、本物の ImageBind を試すための雛形を
  try/except でガードして添える（未導入なら自動スキップ＝exit 0）。

学ぶこと:
  - ImageBind の本質は「各モダリティの encoder が、同じ概念を “共通アンカー” の近くへ
    写すよう学習されている」こと。共有空間さえできれば、音 → 画像 は cos 類似度で引ける。
  - その「共有空間があればクロスモーダル検索が回る」というメカニズムを、概念ごとの
    アンカー + モダリティ別ノイズで再現し、Recall@k で評価する（04 と同じ指標）。

実行:
  uv run python lectures/33_multimodal_embeddings/05_imagebind_concept.py
結果は lectures/33_multimodal_embeddings/outputs/ に保存される。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from siglip_lab import (  # noqa: E402
    average_precision_at_k,
    build_ip_index,
    ensure_output_dir,
    make_trimodal_toy,
    mean,
    recall_at_k,
    save_wav,
    search_index,
    synth_tone,
)

# ImageBind の代表ユースケースは「犬の鳴き声 → 犬の画像」のような “音 → 画像”。
# それに合わせ、特徴的な音を持つ6概念を用意する（音は具体物として可視化・wav 保存）。
# 周波数は概念ごとに変え、波形を見分けられるようにする。
CONCEPTS = ["dog", "bird", "bell", "drum", "rain", "engine"]
BASE_FREQ = {"dog": 220.0, "bird": 880.0, "bell": 1320.0, "drum": 110.0, "rain": 1760.0, "engine": 165.0}


def main() -> None:
    out = ensure_output_dir()

    # === 1. 「音」を具体的に作る（可視化・wav 保存用） ==================
    # トイ埋め込みの正体はアンカーだが、まず “音声クエリ” を実体として用意して
    # 「何を検索しているか」を直感的にする。概念ごとに違う周波数の合成音。
    waves = {c: synth_tone(BASE_FREQ[c], dur=0.4, seed=i) for i, c in enumerate(CONCEPTS)}
    for c, w in waves.items():
        save_wav(out / f"05_audio_{c}.wav", w)
    _save_waveforms(out, waves)
    print(f"[1] 概念 {CONCEPTS} の合成音を作成し wav と波形図を保存")

    # === 2. 三モーダルの「共有空間」をトイで構成 ========================
    # image/text/audio それぞれが、同じ概念アンカーの近くへ写る（=束ねられた）状態を作る。
    toy = make_trimodal_toy(CONCEPTS, dim=64, noise=0.18, seed=0)
    print(f"\n[2] 共有空間(64次元)に image/text/audio を配置（概念={len(CONCEPTS)}）")
    print("    各モダリティの埋め込みは『共通アンカー + モダリティ別ノイズ』で構成")

    # アンカー同士の cos（概念が違えばほぼ無相関＝直交に近い）を確認。
    a = toy["anchor"]
    cross = a @ a.T
    print(f"    アンカー間 cos（非対角の最大）= {float(cross[~np.eye(len(a), dtype=bool)].max()):.3f}")

    # === 3. 音 → 画像 クロスモーダル検索 ================================
    # 画像を index 化し、音声埋め込みをクエリにして最近傍を引く。共有空間なので引ける。
    image_index = build_ip_index(toy["image"])
    scores, ids = search_index(image_index, toy["audio"], k=len(CONCEPTS))
    print("\n[3] 音声クエリ → 画像 検索（top1 が同じ概念なら束ね成功）:")
    for qi, c in enumerate(CONCEPTS):
        top = CONCEPTS[int(ids[qi][0])]
        mark = "OK" if top == c else "  "
        print(f"  [{mark}] audio[{c}] -> image[{top}]  (cos={scores[qi][0]:.3f})")

    # === 4. Recall@k / mAP で評価（04 と同じ関数を流用） ================
    # 正解は「同じ概念の画像」=この toy では1概念1画像なので relevant は1件ずつ。
    recalls, aps = [], []
    for qi in range(len(CONCEPTS)):
        ranked = [int(j) for j in ids[qi].tolist()]
        relevant = {qi}  # audio[qi] の正解画像は image[qi]
        recalls.append(recall_at_k(ranked, relevant, k=1))
        aps.append(average_precision_at_k(ranked, relevant, k=len(CONCEPTS)))
    print(f"\n[4] 音→画像 Recall@1 = {mean(recalls):.3f} / mAP = {mean(aps):.3f}")

    # おまけ: テキスト→画像 も同じ空間で引ける（束ねの汎用性）。
    txt_index = build_ip_index(toy["image"])
    _s, tids = search_index(txt_index, toy["text"], k=1)
    txt_ok = sum(int(tids[qi][0]) == qi for qi in range(len(CONCEPTS)))
    print(f"    （参考）テキスト→画像 top-1 一致 = {txt_ok}/{len(CONCEPTS)}")

    # === 5. 本物の ImageBind を試す雛形（未導入なら自動スキップ） =======
    _try_real_imagebind()

    print("\n[まとめ]")
    print("  - クロスモーダル検索は『どのモダリティも同じ空間に写っている』ことが全て。")
    print("  - 共有空間さえあれば、音→画像 も テキスト→画像 も同じ内積検索で実現できる。")
    print("  - 本物の ImageBind は git 依存で重いので任意。本デモは束ねの“効能”を再現したもの。")


def _save_waveforms(out: pathlib.Path, waves: dict[str, np.ndarray]) -> None:
    """概念ごとの合成音の波形（先頭部分）を1枚に並べて保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(waves), 1, figsize=(8, 1.5 * len(waves)))
    axes = np.atleast_1d(axes)
    for ax, (c, w) in zip(axes, waves.items()):
        ax.plot(w[:400], color="tab:purple", linewidth=0.8)
        ax.set_ylabel(c, fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("sample (first 400)")
    axes[0].set_title("Synthetic audio queries (one tone per concept)")
    fig.tight_layout()
    fig.savefig(out / "05_audio_waveforms.png", dpi=110)
    plt.close(fig)


def _try_real_imagebind() -> None:
    """本物の ImageBind を import できれば一言案内、無ければ静かにスキップする。

    重い git 依存（imagebind, torchaudio, ffmpeg）を必須にしないためのガード。
    """
    try:
        import imagebind  # type: ignore  # noqa: F401

        print("\n[5] 本物の ImageBind が import できました（任意・実験的）。")
        print("    imagebind_model.imagebind_huge(pretrained=True) と")
        print("    data.load_and_transform_audio_data / ModalityType.AUDIO で音→画像が試せます。")
    except Exception:
        print("\n[5] 本物の ImageBind は未導入のためスキップ（概念デモのみで完結）。")
        print("    試すなら: uv add --group imagebind \\")
        print('              "imagebind @ git+https://github.com/facebookresearch/ImageBind.git" torchaudio')


if __name__ == "__main__":
    main()
