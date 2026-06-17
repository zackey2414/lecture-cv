"""章末ミニプロジェクト: 製造ラインの「外観検査 + 品質ゲート」を 1 本に統合する。

このスクリプトは本章の全工程を統合した完成形です。現実の検査ラインを模した流れ:

  1. 撮影品質ゲート（無参照IQA）: ボケた撮影は検査前に弾く。鮮鋭度(varLaplacian)が
     正常品の分布から大きく外れる画像は「再撮影」に回す（不良判定の前段）。
  2. 異常検知（PaDiM・自前実装）: 正常品だけで覚えた分布から、欠陥品とその位置を検出。
  3. 評価: image-level / pixel-level の AUROC・AUPR・PRO を一括算出。
  4. しきい値の決定: image スコアのしきい値を、正常品の分位点（例 99 パーセンタイル）から
     データドリブンに決める（“正常の上澄み”を異常側に倒さない実務的な決め方）。
  5. 出力: 合否マップ・スコア分布・指標 JSON を保存。

★頑健性: resnet18 の重み DL に失敗してもランダム初期化にフォールバックし、必ず exit 0。

実行:
  uv run python lectures/32_anomaly_iqa/mini_project.py
出力:
  lectures/32_anomaly_iqa/outputs/mini_inspection.png … 検出結果（合否・異常マップ）
  lectures/32_anomaly_iqa/outputs/mini_summary.json    … 品質ゲート結果と全評価指標
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import anomaly_iqa_lab as lab


def quality_gate(train_imgs: np.ndarray, test_imgs: np.ndarray, k: float = 3.0) -> dict:
    """無参照IQA(鮮鋭度)で撮影品質ゲートを作る。

    正常品の varLaplacian 分布から下限 (mean - k*std) を決め、それ未満の画像を「ボケ＝要再撮影」
    として弾く。返り値: 下限しきい値と、各テスト画像の合否(True=検査に回す)。
    """
    sharp_train = np.array([lab.variance_of_laplacian(im) for im in train_imgs])
    lo = float(sharp_train.mean() - k * sharp_train.std())
    sharp_test = np.array([lab.variance_of_laplacian(im) for im in test_imgs])
    passed = sharp_test >= lo
    return {"low_threshold": lo, "sharpness_test": sharp_test, "passed": passed}


def detect(ds: dict) -> tuple[np.ndarray, np.ndarray]:
    """PaDiM で異常マップ(N,H,W) と image スコア(N,) を作る。"""
    ext = lab.FeatureExtractor(seed=0)
    print(f"  特徴抽出器: resnet18 ({ext.mode})")
    etr, ete = ext.embed(ds["train"]), ext.embed(ds["test"])
    idx = lab.select_dims(etr.shape[-1], 100, seed=0)
    model = lab.padim_fit(etr[..., idx])
    maps = lab.upsample_maps(lab.padim_score(ete[..., idx], model))
    return maps, lab.image_scores(maps)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, q: float = 99.0) -> float:
    """正常品スコアの q パーセンタイルをしきい値にする（ラベル無しでも決められる実務的手法）。"""
    normal = scores[labels == 0]
    return float(np.percentile(normal, q))


def plot_inspection(ds: dict, maps: np.ndarray, scores: np.ndarray, thr: float, path: str) -> None:
    """テスト画像の一部について、入力＋異常マップ＋合否ラベルを並べる。"""
    show = list(np.where(ds["labels"] == 0)[0][:3]) + list(np.where(ds["labels"] == 1)[0][:5])
    vmax = float(maps.max())
    fig, axes = plt.subplots(2, len(show), figsize=(2.4 * len(show), 5))
    for c, i in enumerate(show):
        pred = scores[i] >= thr
        gt = bool(ds["labels"][i])
        ok = pred == gt
        axes[0, c].imshow(ds["test"][i])
        axes[0, c].set_title(
            f"{'NG' if pred else 'OK'} / gt={'NG' if gt else 'OK'}",
            color=("green" if ok else "red"),
            fontsize=9,
        )
        axes[1, c].imshow(maps[i], cmap="jet", vmin=0, vmax=vmax)
        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("anomaly map")
    fig.suptitle(f"Inspection result (threshold={thr:.2f}; top=decision, bottom=anomaly map)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    out = lab.ensure_out_dir()
    print("=" * 64)
    print("ミニプロジェクト: 外観検査ライン（品質ゲート → 異常検知 → 評価）")

    ds = lab.make_dataset()
    print(
        f"  正常 {len(ds['train'])} 枚で学習 / 評価 {len(ds['test'])} 枚（異常率 {ds['labels'].mean():.2f}）"
    )

    # 1) 撮影品質ゲート（無参照IQA）。デモのため評価セットに 1 枚ボケ画像を混ぜて弾けるか見る。
    blurred_probe = lab.cv2.GaussianBlur(ds["test"][0], (0, 0), sigmaX=4.0)
    gate = quality_gate(ds["train"], np.concatenate([ds["test"], blurred_probe[None]]))
    n_rejected = int((~gate["passed"]).sum())
    print(
        f"  品質ゲート: 鮮鋭度下限={gate['low_threshold']:.4f} → 要再撮影 {n_rejected} 枚（ボケ画像を弾けたか確認）"
    )

    # 2) 異常検知（PaDiM）
    maps, scores = detect(ds)

    # 3) 評価（image / pixel の AUROC・AUPR・PRO）
    metrics = {
        "image_auroc": lab.image_auroc(scores, ds["labels"]),
        "image_aupr": lab.image_aupr(scores, ds["labels"]),
        "pixel_auroc": lab.pixel_auroc(maps, ds["masks"]),
        "pixel_aupr": lab.pixel_aupr(maps, ds["masks"]),
        "pro_auc": lab.pro_auc(maps, ds["masks"]),
    }
    print(
        f"  評価: img AUROC={metrics['image_auroc']:.3f}/AUPR={metrics['image_aupr']:.3f} | "
        f"px AUROC={metrics['pixel_auroc']:.3f}/AUPR={metrics['pixel_aupr']:.3f} | PRO={metrics['pro_auc']:.3f}"
    )

    # 4) しきい値（正常品スコアの 99 パーセンタイル）→ 合否判定
    thr = choose_threshold(scores, ds["labels"], q=99.0)
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (ds["labels"] == 1)).sum())
    fp = int(((pred == 1) & (ds["labels"] == 0)).sum())
    fn = int(((pred == 0) & (ds["labels"] == 1)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    print(
        f"  しきい値 {thr:.2f}（正常99%点）→ 不良検出 recall={recall:.3f} precision={precision:.3f} (FP={fp})"
    )

    # 5) 出力
    img_path = os.path.join(out, "mini_inspection.png")
    json_path = os.path.join(out, "mini_summary.json")
    plot_inspection(ds, maps, scores, thr, img_path)
    summary = {
        "n_train": int(len(ds["train"])),
        "n_test": int(len(ds["test"])),
        "quality_gate": {"low_threshold": gate["low_threshold"], "n_rejected": n_rejected},
        "threshold": thr,
        "recall": recall,
        "precision": precision,
        "metrics": metrics,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  保存: {img_path}")
    print(f"  保存: {json_path}")
    print(
        "  完成: 品質ゲート(無参照IQA) → 異常検知(PaDiM) → 評価(AUROC/AUPR/PRO) → 合否判定 を一気通貫。"
    )


if __name__ == "__main__":
    main()
