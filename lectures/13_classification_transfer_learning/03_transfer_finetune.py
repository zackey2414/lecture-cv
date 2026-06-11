"""第13回 / 03: 転移学習 — 事前学習バックボーンを凍結し、新しいヘッドだけ学習する。

このモジュールの成果物。少データ(合成図形 9クラス)に対して
  1. ImageNet 学習済み ResNet-18 を「特徴抽出器」として凍結 (requires_grad_(False))
  2. 1000クラスの分類層を外し、9クラス用の nn.Linear ヘッドに付け替える
  3. ヘッドだけを数十イテレーション学習する（CPU で数十秒以内）
  4. 学習前(乱数ヘッド)と学習後で accuracy を比較し、混同行列・top-1/top-5・macro-F1 を出す

なぜ凍結＋ヘッド付け替えで効くのか:
  ImageNet で鍛えた畳み込み特徴は「色・形・テクスチャ」といった汎用的な視覚特徴を既に捉えている。
  新しいタスクのクラスがその特徴空間で線形分離できるなら、最後の線形層を少し学習するだけで高精度になる。
  ゼロから CNN を学習するのに比べ、必要データも計算も桁違いに少なくて済む — これが転移学習の威力。

高速化の工夫:
  バックボーンは凍結されており出力は変わらないので、全画像の特徴を「一度だけ」前計算してキャッシュし、
  以降は軽い nn.Linear だけを学習する。これで CPU でも一瞬で学習が終わる（実務でも定石の最適化）。

実行:
  uv run python lectures/13_classification_transfer_learning/03_transfer_finetune.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.metrics import classification_report
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)
from torchvision.models import ResNet18_Weights, resnet18

from dl_helpers import (
    SHAPE_CLASSES,
    ensure_out_dir,
    get_device,
    make_dataset,
    save_prediction_panel,
    set_seed,
)

NUM_CLASSES = len(SHAPE_CLASSES)  # 9
INPUT_SIZE = 128  # 224 より小さくして CPU を軽くする（resnet は適応プールで可変入力に対応）
TRAIN_PER_CLASS = 14
TEST_PER_CLASS = 8
TRAIN_STEPS = 30
LR = 1e-2


def build_transform() -> T.Compose:
    """ImageNet 学習済みモデルに合わせた前処理（resize → 0-1 → 正規化）。

    重みに付属する weights.transforms() は 224px 固定なので、ここでは学習を軽くするため
    INPUT_SIZE にリサイズしつつ、平均/分散は ImageNet の値を使う（手打ちズレ防止のため定数で明示）。
    """
    return T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToImage(),  # PIL → tensor(uint8, CHW)
            T.ToDtype(torch.float32, scale=True),  # 0-255 → 0-1
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_frozen_backbone(device: torch.device) -> nn.Module:
    """ImageNet 学習済み ResNet-18 を特徴抽出器(512次元)にして凍結する。"""
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()  # 1000クラス分類層を外す → 出力が penultimate(512)になる
    # 全パラメータの勾配計算を止める。これが「凍結＝特徴抽出器として固定」。
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone.to(device).eval()


def extract_features(
    backbone: nn.Module, images, transform: T.Compose, device: torch.device, batch: int = 32
) -> torch.Tensor:
    """凍結バックボーンで全画像の特徴(512次元)を前計算してキャッシュする。

    注: ここは inference_mode ではなく no_grad を使う。inference_mode で作ったテンソルは
    「推論専用テンソル」になり、後段でヘッドを学習する際の autograd 計算に入力できない
    （Inference tensors cannot be saved for backward）。no_grad なら通常テンソルとして扱える。
    """
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            xs = torch.stack([transform(im) for im in images[i : i + batch]]).to(device)
            feats.append(backbone(xs).cpu())
    return torch.cat(feats, dim=0)  # (N, 512)


def evaluate(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """torchmetrics で top-1 / top-5 accuracy・macro-F1・混同行列を計算する。

    指標の定義:
      - top-1 accuracy: argmax の予測が正解と一致した割合
      - top-5 accuracy: 上位5予測のどれかに正解が含まれる割合（クラス数9なので易しめ）
      - macro-F1      : クラスごとの F1 を単純平均（不均衡に強い。各クラスを平等に評価）
      - 混同行列      : 行=正解クラス, 列=予測クラス。対角が大きいほど良い
    top_k を使う指標は argmax 後のラベルではなく logits(スコア列)を渡す必要がある点に注意。
    """
    acc1 = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro")
    acc5 = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro", top_k=5)
    f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    cm = MulticlassConfusionMatrix(num_classes=NUM_CLASSES)
    return {
        "top1": float(acc1(logits, targets)),
        "top5": float(acc5(logits, targets)),
        "macro_f1": float(f1(logits, targets)),
        "confusion": cm(logits, targets).numpy(),
    }


def save_confusion(cm: np.ndarray, path: pathlib.Path, title: str) -> None:
    """混同行列をヒートマップとして保存（クラス名は ASCII なので豆腐にならない）。"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(SHAPE_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SHAPE_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    # セルに件数を書き込む。
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    set_seed(0)
    out_dir = ensure_out_dir()
    device = get_device()
    print(f"[info] device = {device} / classes = {NUM_CLASSES}")

    # ---------- 1. データ生成 ----------
    train_imgs, train_y = make_dataset(TRAIN_PER_CLASS, size=96, seed=1, noise_std=18.0)
    test_imgs, test_y = make_dataset(TEST_PER_CLASS, size=96, seed=2, noise_std=18.0)
    print(f"[data] train={len(train_imgs)}  test={len(test_imgs)} (合成図形 色×形 9クラス)")

    # ---------- 2. 凍結バックボーンで特徴を前計算 ----------
    transform = build_transform()
    backbone = build_frozen_backbone(device)
    n_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[backbone] resnet18 frozen, trainable params = {n_trainable} (0なら完全凍結)")

    f_train = extract_features(backbone, train_imgs, transform, device)
    f_test = extract_features(backbone, test_imgs, transform, device)
    y_train = torch.from_numpy(train_y)
    y_test = torch.from_numpy(test_y)
    print(f"[features] train {tuple(f_train.shape)}  test {tuple(f_test.shape)} (キャッシュ済み)")

    # ---------- 3. 新しい分類ヘッド(nn.Linear)を用意 ----------
    head = nn.Linear(512, NUM_CLASSES)  # 512次元の特徴 → 9クラス
    # 【ベースライン】学習前(乱数初期化)のヘッドで評価 → ほぼ偶然(1/9≈0.11)になるはず。
    with torch.inference_mode():
        base = evaluate(head(f_test), y_test)
    print(f"\n[baseline] 学習前(乱数ヘッド): top1={base['top1']:.3f}  macro-F1={base['macro_f1']:.3f}")

    # ---------- 4. ヘッドだけを学習 ----------
    # 凍結バックボーンの特徴は固定なので、学習対象は head のパラメータだけ。
    # 補足: バックボーンも一緒に微調整する「全体ファインチューニング」では、
    #       optimizer に param_groups を渡し backbone に小さい lr、head に大きい lr を与える
    #       （= "学習率の段差"）。事前学習特徴を壊さずに微修正するための定石。
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    head.train()
    for step in range(TRAIN_STEPS):
        optimizer.zero_grad()
        loss = loss_fn(head(f_train), y_train)  # 全データを1バッチで（データが小さいので十分）
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1:>2d}/{TRAIN_STEPS}  loss={loss.item():.4f}")
    head.eval()

    # ---------- 5. 学習後の評価 ----------
    with torch.inference_mode():
        test_logits = head(f_test)
    tuned = evaluate(test_logits, y_test)
    print(
        f"\n[finetuned] 学習後: top1={tuned['top1']:.3f}  top5={tuned['top5']:.3f}  "
        f"macro-F1={tuned['macro_f1']:.3f}"
    )
    print(
        f"[compare]  top-1 accuracy: {base['top1']:.3f} (学習前) -> {tuned['top1']:.3f} (学習後)  "
        "← 凍結特徴 + 線形ヘッドだけで大きく改善"
    )

    # sklearn のレポートでクラス別 precision/recall/F1 も確認（14回への布石）。
    pred = test_logits.argmax(-1).numpy()
    report = classification_report(
        y_test.numpy(), pred, target_names=SHAPE_CLASSES, zero_division=0
    )
    print("\n[classification_report]\n" + report)

    # ---------- 6. 可視化と保存 ----------
    save_confusion(
        tuned["confusion"],
        out_dir / "03_confusion_matrix.png",
        f"Confusion matrix (finetuned)  top1={tuned['top1']:.2f}",
    )

    # テスト画像のサンプルに「正解/予測」を付けて並べる（誤りには * を付ける）。
    sample_idx = list(range(0, len(test_imgs), max(1, len(test_imgs) // 8)))[:8]
    sample_imgs = [test_imgs[i] for i in sample_idx]
    sample_titles = []
    for i in sample_idx:
        t, p = SHAPE_CLASSES[int(test_y[i])], SHAPE_CLASSES[int(pred[i])]
        mark = "" if t == p else " *"
        sample_titles.append(f"true: {t}\npred: {p}{mark}")
    save_prediction_panel(
        sample_imgs,
        sample_titles,
        out_dir / "03_test_predictions.png",
        suptitle="Finetuned head predictions ( * = wrong )",
    )

    metrics = {
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "classes": SHAPE_CLASSES,
        "train_size": len(train_imgs),
        "test_size": len(test_imgs),
        "train_steps": TRAIN_STEPS,
        "baseline": {k: round(v, 4) for k, v in base.items() if k != "confusion"},
        "finetuned": {k: round(v, 4) for k, v in tuned.items() if k != "confusion"},
        "confusion_finetuned": tuned["confusion"].tolist(),
    }
    (out_dir / "03_transfer_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[done] saved confusion matrix / predictions / metrics to {out_dir}")


if __name__ == "__main__":
    main()
