"""03_action_topk_eval.py — clip-level の top-1/top-5 と「前処理を壊す」実験。

本章のいちばん大事なメッセージ:
  「行動認識はフレームサンプリング(clip_len/frame_rate)と専用正規化を誤ると、
   出力が無意味になる」。それを数値で示す。

評価の考え方（合成データなので工夫が要る）:
  合成クリップには本物の Kinetics ラベルが無い。そこで
  「正しい前処理で得た予測(top-1)」を各クリップの基準ラベル(pseudo-GT)とする。
  そのうえで前処理を色々に崩し、pseudo-GT に対する top-1/top-5 一致率を測る。
  正しい前処理なら 100%（定義上）、崩すほど一致率は落ちる。
  これは「絶対精度」ではなく「前処理感度（robustness）」の定量化である点に注意。

モデルは軽い r3d_18 を使う（CPU で多数クリップ×多数前処理を回すため）。

実行: uv run python lectures/29_video_action_recognition/03_action_topk_eval.py
"""

from __future__ import annotations

import numpy as np
import torch

import action_helpers as H

# r3d_18 と無関係な ImageNet 正規化（わざと間違える実験用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_variant_tensor(clip_rgb: np.ndarray, variant: str, clip_len: int = 16) -> torch.Tensor:
    """前処理バリアントごとに (1,C,T,H,W) を作る。

    correct       : 正しい（等間隔サンプリング＋Kinetics 正規化）
    no_normalize  : 正規化を省く（よくある事故）
    imagenet_norm : 画像用 ImageNet の mean/std を誤用
    static_frame  : 中央フレームを clip_len 枚に複製（時間情報を消す）
    front_window  : 先頭だけを連続サンプリング（動きの一部しか見ない）
    """
    total = clip_rgb.shape[0]
    if variant == "correct":
        idx = H.uniform_indices(total, clip_len)
        return H.preprocess_for_r3d(clip_rgb, idx, normalize=True)
    if variant == "no_normalize":
        idx = H.uniform_indices(total, clip_len)
        return H.preprocess_for_r3d(clip_rgb, idx, normalize=False)
    if variant == "imagenet_norm":
        idx = H.uniform_indices(total, clip_len)
        return H.preprocess_for_r3d(clip_rgb, idx, normalize=True,
                                    mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if variant == "static_frame":
        mid = total // 2
        idx = np.full(clip_len, mid, dtype=int)  # 同じフレームを並べる→動きゼロ
        return H.preprocess_for_r3d(clip_rgb, idx, normalize=True)
    if variant == "front_window":
        idx = H.strided_indices(total, clip_len, frame_rate=1, start=0)  # 先頭から詰めて取る
        return H.preprocess_for_r3d(clip_rgb, idx, normalize=True)
    raise ValueError(variant)


def logits_for_variant(model, clips, variant: str) -> np.ndarray:
    """全クリップを 1 つの前処理バリアントで推論し、logits (N, 400) を返す。"""
    rows = []
    with torch.inference_mode():
        for clip in clips:
            x = build_variant_tensor(clip, variant)
            rows.append(model(x)[0].numpy())
    return np.stack(rows)


def compact_confusion(pseudo_gt: np.ndarray, pred: np.ndarray):
    """pseudo-GT に出現するクラスだけで小さな混同行列を作る（列に "other" を足す）。

    行=pseudo-GT のラベル、列=[出現ラベル..., "other"]。崩れた前処理の予測が
    どれだけ「別物(other)」へ散るかが一目で分かる。
    """
    classes = sorted(set(pseudo_gt.tolist()))  # 出現する Kinetics クラスID
    cls_to_row = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n + 1), dtype=int)  # 最後の列が "other"
    for g, p in zip(pseudo_gt, pred):
        r = cls_to_row[int(g)]
        c = cls_to_row.get(int(p), n)  # 出現ラベルに無ければ other
        cm[r, c] += 1
    return cm, classes


def main() -> None:
    H.limit_cpu_threads(4)
    print("=" * 72)
    print("行動認識の top-1/top-5 評価 ＋ 前処理を壊す実験（r3d_18）")
    print("=" * 72)

    try:
        model, categories = H.load_r3d()
    except Exception as e:
        print(f"[案内] r3d_18 をロードできませんでした（オフライン？）: {type(e).__name__}: {e}")
        return

    # 1) 小さな検証セット（6 種の動き × 2 seed = 12 クリップ。32 フレームでサンプリング差を出す）
    clips, _kinds = H.make_clip_dataset(num_frames=32, seeds_per_kind=2)
    print(f"検証セット: {len(clips)} クリップ（各 32 フレーム, {len(H.ACTION_KINDS)} 種の動き）")

    # 2) 正しい前処理で pseudo-GT を作る
    correct_logits = logits_for_variant(model, clips, "correct")
    pseudo_gt = correct_logits.argmax(1)  # 各クリップの基準ラベル(Kinetics クラスID)
    distinct = sorted(set(pseudo_gt.tolist()))
    print(f"pseudo-GT（正しい前処理の top-1）の異なるクラス数: {len(distinct)}")
    print("  例:", [categories[c] for c in distinct[:6]])

    # 3) 各前処理バリアントで top-1/top-5 一致率を測る
    variants = ["correct", "no_normalize", "imagenet_norm", "static_frame", "front_window"]
    results = {}
    for v in variants:
        logits = logits_for_variant(model, clips, v) if v != "correct" else correct_logits
        t1 = H.topk_accuracy(logits, pseudo_gt, k=1)
        t5 = H.topk_accuracy(logits, pseudo_gt, k=5)
        results[v] = {"top1": t1, "top5": t5}
        print(f"  {v:<14s} top-1={t1:5.3f}  top-5={t5:5.3f}")

    print("\n→ correct は定義上 1.000。正規化やサンプリングを崩すほど一致率が落ちる"
          "（= 前処理を誤ると出力が壊れる）。")

    # 4) 棒グラフ（top-1 と top-5 を横並び）
    _save_grouped_bar(variants, results, H.OUT_DIR / "03_topk_by_variant.png")

    # 5) 代表的に崩れたバリアントの混同行列（no_normalize）
    bad_logits = logits_for_variant(model, clips, "no_normalize")
    cm, classes = compact_confusion(pseudo_gt, bad_logits.argmax(1))
    labels = [categories[c][:14] for c in classes] + ["other"]
    # 行ラベルは出現クラスのみ、列はそれ＋other。save_confusion は正方前提なので簡易描画する。
    _save_rect_confusion(cm, [categories[c][:14] for c in classes], labels,
                         H.OUT_DIR / "03_confusion_no_normalize.png",
                         "confusion (pseudo-GT vs no_normalize pred)")

    H.save_json(
        {
            "note": "pseudo-GT = 正しい前処理での top-1。値は pseudo-GT に対する一致率。",
            "num_clips": len(clips),
            "num_distinct_pseudo_gt": len(distinct),
            "results": results,
        },
        H.OUT_DIR / "03_action_eval.json",
    )
    print("\n保存: 03_topk_by_variant.png / 03_confusion_no_normalize.png / 03_action_eval.json")


def _save_grouped_bar(variants, results, path) -> None:
    """top-1 と top-5 を変種ごとに並べた棒グラフ。"""
    import matplotlib.pyplot as plt

    x = np.arange(len(variants))
    w = 0.38
    t1 = [results[v]["top1"] for v in variants]
    t5 = [results[v]["top5"] for v in variants]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, t1, w, label="top-1", color="#4c72b0")
    ax.bar(x + w / 2, t5, w, label="top-5", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("agreement vs pseudo-GT")
    ax.set_ylim(0, 1.05)
    ax.set_title("preprocessing sensitivity of action recognition (r3d_18)", fontsize=10)
    ax.legend()
    for xi, v in zip(x - w / 2, t1):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w / 2, t5):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_rect_confusion(cm, row_labels, col_labels, path, title) -> None:
    """非正方の混同行列（列に other を含む）を描画する。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(col_labels), 1.5 + 0.7 * len(row_labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("pseudo-GT")
    ax.set_title(title, fontsize=9)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
