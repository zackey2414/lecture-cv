"""01: PaDiM を自前実装して「正常だけで」キズ/異物を検出する。

学ぶこと:
  - 異常検知の基本姿勢 = **正常画像だけで分布を覚え、そこから外れたものを異常**とする。
    （欠陥は珍しく・多様で集めにくいので「異常を学習する」のは現実的でない。）
  - PaDiM の中身を手で組む:
      1. 学習済み resnet18 の中間特徴(layer1..3)を、各空間位置のベクトル(埋め込み)にする。
      2. チャネルをランダムに 100 本だけ間引く（共分散を扱える次元に落とす）。
      3. 正常画像だけを使い、各位置ごとに多変量ガウシアン（平均・共分散）を推定。
      4. テスト画像の各位置を、その位置のガウシアンからの **マハラノビス距離** で採点。
  - 異常マップ（位置別スコア）を画像に重ね、欠陥がどこかを可視化する。
  - image-level / pixel-level の AUROC を確認し、しきい値で異常/正常を切る。

実行:
  uv run python lectures/32_anomaly_iqa/01_padim_anomaly.py
出力:
  lectures/32_anomaly_iqa/outputs/01_anomaly_maps.png   … 入力・GT・異常マップ・重ね合わせ
  lectures/32_anomaly_iqa/outputs/01_score_hist.png     … 正常/異常の image-level スコア分布
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless でも保存できるよう GUI を使わないバックエンド
import matplotlib.pyplot as plt
import numpy as np

import anomaly_iqa_lab as lab


def build_padim(ds: dict, n_dims: int = 100):
    """正常画像で PaDiM を学習し、テスト画像の異常マップ・image スコアを返す。"""
    ext = lab.FeatureExtractor(seed=0)  # resnet18（重みDL失敗時はランダム初期化で完走）
    print(f"  特徴抽出器: resnet18 ({ext.mode})  入力 {lab.IMG_SIZE}px")
    emb_train = ext.embed(ds["train"])  # (N, Hc, Wc, C)
    emb_test = ext.embed(ds["test"])
    hc, wc, c = emb_train.shape[1:]
    print(f"  埋め込み: 位置 {hc}x{wc}  連結チャネル {c} → ランダム {n_dims} 次元に削減")

    idx = lab.select_dims(c, n_dims, seed=0)  # PaDiM の次元削減
    model = lab.padim_fit(emb_train[..., idx])  # 位置別ガウシアン（正常のみ）
    raw_maps = lab.padim_score(emb_test[..., idx], model)  # (N, Hc, Wc) マハラノビス距離
    maps = lab.upsample_maps(raw_maps)  # 画像サイズへ拡大 + 平滑化
    scores = lab.image_scores(maps)  # 画像ごとの最大スコア
    return maps, scores


def plot_anomaly_maps(ds: dict, maps: np.ndarray, path: str, n_show: int = 4) -> None:
    """異常画像を数枚選び、入力 / GT / 異常マップ / 重ね合わせ を 1 枚に並べる。"""
    anom_idx = np.where(ds["labels"] == 1)[0][:n_show]
    vmax = float(maps[anom_idx].max())  # 表示スケールを異常側に合わせる
    fig, axes = plt.subplots(len(anom_idx), 4, figsize=(11, 3 * len(anom_idx)))
    col = ["input", "ground truth", "anomaly map", "overlay"]
    for r, i in enumerate(anom_idx):
        img = ds["test"][i]
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(ds["masks"][i], cmap="gray")
        axes[r, 2].imshow(maps[i], cmap="jet", vmin=0, vmax=vmax)
        axes[r, 3].imshow(img)
        axes[r, 3].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        for k in range(4):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
            if r == 0:
                axes[r, k].set_title(col[k])
    fig.suptitle("PaDiM anomaly maps (trained on normal images only)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_score_hist(scores: np.ndarray, labels: np.ndarray, path: str) -> float:
    """正常/異常の image-level スコア分布を描き、両者を分ける素朴なしきい値を返す。"""
    normal = scores[labels == 0]
    anom = scores[labels == 1]
    # 単純な分離しきい値: 正常の最大と異常の最小の中点（重なりが無ければ完全分離）。
    thr = float((normal.max() + anom.min()) / 2.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 20)
    ax.hist(normal, bins=bins, alpha=0.6, label="normal", color="tab:blue")
    ax.hist(anom, bins=bins, alpha=0.6, label="anomalous", color="tab:red")
    ax.axvline(thr, color="k", ls="--", label=f"threshold={thr:.2f}")
    ax.set_xlabel("image-level anomaly score (max of map)")
    ax.set_ylabel("count")
    ax.set_title("Image-level score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return thr


def main() -> None:
    out = lab.ensure_out_dir()
    print("=" * 64)
    print("01 PaDiM 異常検知（正常だけで学習 → マハラノビス距離で採点）")

    ds = lab.make_dataset()  # 正常(train) + 正常/異常(test) + 画素GTマスク
    print(
        f"  学習 正常 {len(ds['train'])} 枚 / 評価 {len(ds['test'])} 枚（異常率 {ds['labels'].mean():.2f}）"
    )

    maps, scores = build_padim(ds)

    # 評価（詳しい AUPR/PRO は 02 で扱う。ここは AUROC とスコア分布まで）。
    img_auroc = lab.image_auroc(scores, ds["labels"])
    px_auroc = lab.pixel_auroc(maps, ds["masks"])
    print(f"  image-level AUROC = {img_auroc:.3f}（画像が異常か）")
    print(f"  pixel-level AUROC = {px_auroc:.3f}（欠陥がどこか）")

    map_path = os.path.join(out, "01_anomaly_maps.png")
    hist_path = os.path.join(out, "01_score_hist.png")
    plot_anomaly_maps(ds, maps, map_path)
    thr = plot_score_hist(scores, ds["labels"], hist_path)

    # しきい値で異常判定したときの正解率（合成データは分離が容易で満点になりやすい）。
    pred = (scores >= thr).astype(int)
    acc = float((pred == ds["labels"]).mean())
    print(f"  しきい値 {thr:.2f} での image 判定 正解率 = {acc:.3f}")
    print(f"  保存: {map_path}")
    print(f"  保存: {hist_path}")
    print("  → 欠陥を一切学習せず『正常からの逸脱』だけで検出できている点が肝。")


if __name__ == "__main__":
    main()
