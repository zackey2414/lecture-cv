"""mini_project.py — 章末統合: 「mp4 → フレーム抽出 → 行動推定 → 評価」一気通貫。

本章の要素を 1 本のパイプラインに統合する:
  ステージA【動画 I/O】 合成行動クリップを mp4 に書き出し、cv2.VideoCapture で読み戻す
                        （torchvision read_video 廃止の代替経路をそのまま実演）。
  ステージB【基準作り】 正しい前処理(r3d_18)で各クリップの pseudo-GT(top-1) を決める。
  ステージC【サンプリング掃引】 clip_len と frame_rate(stride) を変えながら
                        top-1/top-5 一致率を測り、「短すぎ/粗すぎるサンプリングで崩れる」を定量化。
  ステージD【別モデル(任意)】 VideoMAE が使えれば数クリップで r3d_18 と top-1 を突き合わせる。
  ステージE【レポート】 サマリ図(03 とは別観点=サンプリング掃引)と総合 json を出力。

設計方針: どこかのモデルが落ちても（オフライン等）パイプライン全体は止めず、
できた範囲でレポートを出す。だから必ず exit 0 になる（実運用パイプラインの堅牢性の最小形）。

実行: uv run python lectures/29_video_action_recognition/mini_project.py
"""

from __future__ import annotations

import numpy as np
import torch

import action_helpers as H


def stage_a_video_io(clips):
    """各クリップを mp4 に書き出し、cv2 で読み戻したフレーム列のリストを返す。"""
    print("\n[ステージA] mp4 書き出し → cv2.VideoCapture 読み戻し")
    frames_list = []
    for i, clip in enumerate(clips):
        mp4 = H.write_clip_mp4(clip, H.OUT_DIR / f"mini_clip_{i:02d}.mp4", fps=8.0)
        frames_list.append(H.read_clip_mp4(mp4))
    print(f"  {len(clips)} クリップを mp4 経由でラウンドトリップ（各 {clips[0].shape[0]} フレーム）")
    return frames_list


def stage_b_pseudo_gt(model, frames_list):
    """正しい前処理での top-1 を pseudo-GT として返す。"""
    print("\n[ステージB] 正しい前処理で pseudo-GT(基準ラベル) を作成")
    logits = []
    with torch.inference_mode():
        for frames in frames_list:
            idx = H.uniform_indices(frames.shape[0], 16)
            x = H.preprocess_for_r3d(frames, idx, normalize=True)
            logits.append(model(x)[0].numpy())
    logits = np.stack(logits)
    pseudo_gt = logits.argmax(1)
    print(f"  pseudo-GT の異なるクラス数: {len(set(pseudo_gt.tolist()))}")
    return pseudo_gt


def stage_c_sampling_sweep(model, frames_list, pseudo_gt, categories):
    """clip_len と frame_rate を掃引して top-1/top-5 一致率を測る。"""
    print("\n[ステージC] フレームサンプリング掃引（clip_len / frame_rate）")

    # (1) clip_len を変える（uniform サンプリング, 正規化は正しい）
    clip_len_grid = [2, 4, 8, 16]
    by_clip_len = {}
    for cl in clip_len_grid:
        logits = []
        with torch.inference_mode():
            for frames in frames_list:
                idx = H.uniform_indices(frames.shape[0], cl)
                # r3d_18 は時間方向を畳み込むので clip_len は可変でも動く（時間長が変わるだけ）
                x = H.preprocess_for_r3d(frames, idx, normalize=True)
                logits.append(model(x)[0].numpy())
        logits = np.stack(logits)
        by_clip_len[cl] = {
            "top1": H.topk_accuracy(logits, pseudo_gt, 1),
            "top5": H.topk_accuracy(logits, pseudo_gt, 5),
        }
        print(f"  clip_len={cl:2d}: top-1={by_clip_len[cl]['top1']:.3f} "
              f"top-5={by_clip_len[cl]['top5']:.3f}")

    # (2) frame_rate(stride) を変える（clip_len=16 固定, strided サンプリング）
    rate_grid = [1, 2, 4]
    by_rate = {}
    for r in rate_grid:
        logits = []
        with torch.inference_mode():
            for frames in frames_list:
                idx = H.strided_indices(frames.shape[0], 16, frame_rate=r)
                x = H.preprocess_for_r3d(frames, idx, normalize=True)
                logits.append(model(x)[0].numpy())
        logits = np.stack(logits)
        by_rate[r] = {
            "top1": H.topk_accuracy(logits, pseudo_gt, 1),
            "top5": H.topk_accuracy(logits, pseudo_gt, 5),
        }
        print(f"  frame_rate={r}: top-1={by_rate[r]['top1']:.3f} top-5={by_rate[r]['top5']:.3f}")

    return by_clip_len, by_rate


def stage_d_videomae(frames_list, pseudo_gt, categories):
    """任意: VideoMAE が使えれば数クリップで r3d_18 の pseudo-GT と top-1 を突き合わせる。"""
    print("\n[ステージD] VideoMAE との突き合わせ（任意・重いので 3 クリップだけ）")
    try:
        model, processor, id2label, _ = H.load_videomae()
    except Exception as e:
        print(f"  VideoMAE は使用不可（{type(e).__name__}）。このステージはスキップ。")
        return None
    n = min(3, len(frames_list))
    rows = []
    with torch.inference_mode():
        for i in range(n):
            frames = frames_list[i]
            idx = H.uniform_indices(frames.shape[0], model.config.num_frames)
            inputs = processor([frames[j] for j in idx], return_tensors="pt")
            pred = int(model(**inputs).logits.argmax(1))
            rows.append({
                "clip": i,
                "r3d18_pseudo_gt": categories[int(pseudo_gt[i])],
                "videomae_top1": id2label[pred],
            })
            print(f"  clip{i}: r3d18='{rows[-1]['r3d18_pseudo_gt']}' "
                  f"vs videomae='{rows[-1]['videomae_top1']}'")
    print("  （異なるモデルは別クラスを出しうる＝合成データに正解が無いことの裏返し）")
    return rows


def main() -> None:
    H.limit_cpu_threads(4)
    print("=" * 72)
    print("章末ミニプロジェクト: mp4 → フレーム抽出 → 行動推定 → 評価")
    print("=" * 72)

    try:
        model, categories = H.load_r3d()
    except Exception as e:
        print(f"[案内] r3d_18 をロードできませんでした（オフライン？）: {type(e).__name__}: {e}")
        print("       初回はネットワークが必要です。重みは ~/.cache/torch/hub に保存されます。")
        return

    # データセット（6 種 × 1 seed = 6 クリップ。各 32 フレームでサンプリング差を出す）
    clips, _kinds = H.make_clip_dataset(num_frames=32, seeds_per_kind=1)
    print(f"合成データセット: {len(clips)} クリップ（各 32 フレーム, {len(H.ACTION_KINDS)} 種）")

    frames_list = stage_a_video_io(clips)
    pseudo_gt = stage_b_pseudo_gt(model, frames_list)
    by_clip_len, by_rate = stage_c_sampling_sweep(model, frames_list, pseudo_gt, categories)
    videomae_rows = stage_d_videomae(frames_list, pseudo_gt, categories)

    # ステージE: サマリ図 + 混同行列 + レポート
    print("\n[ステージE] サマリ図とレポートを保存")
    _save_summary(by_clip_len, by_rate, H.OUT_DIR / "mini_summary.png")

    # 一番粗いサンプリング(clip_len=2)の混同行列を出して「崩れ」を可視化
    logits2 = []
    with torch.inference_mode():
        for frames in frames_list:
            idx = H.uniform_indices(frames.shape[0], 2)
            logits2.append(model(H.preprocess_for_r3d(frames, idx, normalize=True))[0].numpy())
    pred2 = np.stack(logits2).argmax(1)
    _save_compact_confusion(pseudo_gt, pred2, categories,
                            H.OUT_DIR / "mini_confusion_cliplen2.png")

    H.save_json(
        {
            "pipeline": "mp4 -> cv2.VideoCapture -> sample -> r3d_18 -> top-k",
            "num_clips": len(clips),
            "pseudo_gt": [categories[int(c)] for c in pseudo_gt],
            "top_k_by_clip_len": {str(k): v for k, v in by_clip_len.items()},
            "top_k_by_frame_rate": {str(k): v for k, v in by_rate.items()},
            "videomae_vs_r3d18": videomae_rows,
        },
        H.OUT_DIR / "mini_report.json",
    )
    print("\n完了。保存物:")
    print("  lectures/29_video_action_recognition/outputs/mini_summary.png         （サンプリング掃引）")
    print("  lectures/29_video_action_recognition/outputs/mini_confusion_cliplen2.png（粗サンプリングの崩れ）")
    print("  lectures/29_video_action_recognition/outputs/mini_report.json          （総合レポート）")
    print("  lectures/29_video_action_recognition/outputs/mini_clip_XX.mp4          （各クリップの動画）")


def _save_summary(by_clip_len, by_rate, path) -> None:
    """サンプリング掃引(clip_len / frame_rate)の top-1/top-5 を 2 枚並べる。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    # 左: clip_len
    cls = sorted(by_clip_len)
    axes[0].plot(cls, [by_clip_len[c]["top1"] for c in cls], "o-", label="top-1", color="#4c72b0")
    axes[0].plot(cls, [by_clip_len[c]["top5"] for c in cls], "s--", label="top-5", color="#dd8452")
    axes[0].set_xlabel("clip_len (frames)")
    axes[0].set_ylabel("agreement vs pseudo-GT")
    axes[0].set_title("effect of clip_len", fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(cls)
    axes[0].legend()
    # 右: frame_rate
    rs = sorted(by_rate)
    axes[1].plot(rs, [by_rate[r]["top1"] for r in rs], "o-", label="top-1", color="#4c72b0")
    axes[1].plot(rs, [by_rate[r]["top5"] for r in rs], "s--", label="top-5", color="#dd8452")
    axes[1].set_xlabel("frame_rate (stride)")
    axes[1].set_title("effect of frame_rate", fontsize=10)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(rs)
    axes[1].legend()
    fig.suptitle("mini project: frame-sampling sweep (r3d_18)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_compact_confusion(pseudo_gt, pred, categories, path) -> None:
    """pseudo-GT 出現クラス + other の小さな混同行列。"""
    import matplotlib.pyplot as plt

    classes = sorted(set(pseudo_gt.tolist()))
    cls_to_row = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n + 1), dtype=int)
    for g, p in zip(pseudo_gt, pred):
        cm[cls_to_row[int(g)], cls_to_row.get(int(p), n)] += 1
    row_labels = [categories[c][:14] for c in classes]
    col_labels = row_labels + ["other"]
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(col_labels), 1.5 + 0.7 * n))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("predicted (clip_len=2)")
    ax.set_ylabel("pseudo-GT (clip_len=16)")
    ax.set_title("too-short sampling scatters predictions", fontsize=9)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(n):
        for j in range(n + 1):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
