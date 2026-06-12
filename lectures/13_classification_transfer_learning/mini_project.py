"""mini_project.py — 章末ミニプロジェクト: バックボーン比べ＋転移学習＋検索の総合課題。

この回の学び（CNN/ViT の埋め込み取り出し・凍結＋線形ヘッドの転移学習・複数指標での評価・
コサイン類似度）を 1 本のパイプラインに統合する「完成形」。合成図形 9 クラス(色×形)を題材に、
3 つの事前学習バックボーンを「特徴抽出器」として横並びでベンチマークする。

    バックボーン:
      - resnet18_tv   : torchvision ResNet-18, fc=Identity → penultimate 512 次元
      - resnet18_timm : timm ResNet-18, num_classes=0 → pooled 512 次元
      - vit_tiny_cls  : HuggingFace ViT-tiny(AutoModel) の CLS トークン 192 次元

    各バックボーンについて以下を一気通貫で実施:
      ① 凍結特徴の前計算: 学習用/テスト用/「強ノイズ」テスト用の画像を埋め込みに変換
         （inference_mode ではなく no_grad。後でヘッド学習に使えるようにするため）
      ② 線形プローブ: 凍結特徴の上に nn.Linear だけを学習（= 転移学習の本質）
      ③ 評価: top-1 / macro-F1 をクリーンとノイズ強のテストで計測（分布シフトへの頑健性）
      ④ 学習不要ベースライン: 最近傍重心(nearest-centroid, コサイン類似)での分類精度
         → 「埋め込みが既にクラスを分離していれば、学習ゼロでもそこそこ当たる」を体感
      ⑤ 画像検索デモ: ベスト backbone の埋め込みで、クエリ画像に近い学習画像を上位 3 件提示

外部データもネットも GPU も不要（合成画像のみ・CPU 完結、モデル重みは初回のみ DL してキャッシュ）。
実運用ではこの「凍結特徴 → 線形プローブ／重心分類／近傍検索」がそのまま少データ分類の定石になる。

実行:
    uv run python lectures/13_classification_transfer_learning/mini_project.py
結果（図・JSON）は outputs/13_classification_transfer_learning/ に保存される（画面表示はしない）。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless 環境向け。pyplot より前に Agg を指定する。

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import transformers
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.models import ResNet18_Weights, resnet18

# 同じフォルダの dl_helpers を確実に import できるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import timm  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402

from dl_helpers import (  # noqa: E402
    SHAPE_CLASSES,
    ensure_out_dir,
    get_device,
    make_dataset,
    set_seed,
)

transformers.logging.set_verbosity_error()

NUM_CLASSES = len(SHAPE_CLASSES)  # 9
VIT_ID = "WinKawaks/vit-tiny-patch16-224"

# CPU で数十秒に収まる規模。少データ転移学習の設定を模す。
TRAIN_PER_CLASS = 10
TEST_PER_CLASS = 6
TRAIN_NOISE = 18.0   # 学習/クリーンテストのノイズ量
HARD_NOISE = 42.0    # 「強ノイズ」テスト（分布シフトを模す）
PROBE_STEPS = 60
PROBE_LR = 1e-2
BATCH = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# 埋め込み器（3 つのバックボーンを統一インターフェースで包む）
# =====================================================================

def _imagenet_transform(size: int) -> T.Compose:
    """ImageNet 学習済み CNN 用の前処理（resize → 0-1 → 正規化）。手打ちズレ防止に定数で明示。"""
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToImage(),                              # PIL → uint8 CHW tensor
            T.ToDtype(torch.float32, scale=True),     # 0-255 → 0-1
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_embedder(name: str, device: torch.device):
    """バックボーン名から (embed_batch, dim) を返す。

    embed_batch(list[PIL]) -> torch.Tensor (n, dim)（CPU 上, 勾配なし）。
    どの backbone も「画像 → 前処理 → 凍結モデル → 特徴ベクトル」という同じ型に揃える。
    特徴抽出は no_grad で行う（inference_mode で作ると後段の線形ヘッド学習で autograd に使えない）。
    """
    if name == "resnet18_tv":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()  # 1000 クラス分類層を外す → 512 次元 penultimate が出力に
        model = model.to(device).eval()
        tf = _imagenet_transform(160)

        def embed(imgs):
            x = torch.stack([tf(im) for im in imgs]).to(device)
            with torch.no_grad():
                return model(x).cpu()

        return embed, 512

    if name == "resnet18_timm":
        model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)  # モデル固有の前処理を自動生成
        tf = timm.data.create_transform(**cfg)

        def embed(imgs):
            x = torch.stack([tf(im) for im in imgs]).to(device)
            with torch.no_grad():
                return model(x).cpu()

        return embed, 512

    if name == "vit_tiny_cls":
        processor = AutoImageProcessor.from_pretrained(VIT_ID)  # fast 実装(torchvision backend)
        model = AutoModel.from_pretrained(VIT_ID).to(device).eval()

        def embed(imgs):
            inputs = processor(images=list(imgs), return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            return out.last_hidden_state[:, 0].cpu()  # CLS トークン(学習済み)を埋め込みに使う

        return embed, 192

    raise ValueError(f"unknown backbone: {name}")


def extract_features(embed, images: list, batch: int = BATCH) -> torch.Tensor:
    """全画像をミニバッチで埋め込みに変換し (N, dim) を返す。"""
    feats = [embed(images[i : i + batch]) for i in range(0, len(images), batch)]
    return torch.cat(feats, dim=0)


# =====================================================================
# 評価ロジック（線形プローブ・最近傍重心・指標）
# =====================================================================

def train_linear_probe(
    f_train: torch.Tensor, y_train: torch.Tensor, dim: int, steps: int = PROBE_STEPS
) -> nn.Linear:
    """凍結特徴の上に nn.Linear ヘッドだけを学習する（転移学習の本質部分）。"""
    head = nn.Linear(dim, NUM_CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()
    head.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(head(f_train), y_train)  # 小データなので全件 1 バッチで十分
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def score(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """torchmetrics で top-1 accuracy と macro-F1 を返す。"""
    acc = MulticlassAccuracy(num_classes=NUM_CLASSES, average="micro")
    f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    return float(acc(logits, targets)), float(f1(logits, targets))


def nearest_centroid_accuracy(
    f_train: torch.Tensor, y_train: torch.Tensor, f_test: torch.Tensor, y_test: torch.Tensor
) -> float:
    """学習不要ベースライン: クラス重心へのコサイン類似で最近傍分類した正解率。

    各クラスの学習特徴の平均(重心)を作り、L2 正規化してクエリとの内積=コサイン類似を取り、
    最も近い重心のクラスを予測する。埋め込みが既にクラスを分離していれば学習ゼロでも当たる。
    """
    dim = f_train.shape[1]
    centroids = torch.zeros(NUM_CLASSES, dim)
    for c in range(NUM_CLASSES):
        mask = y_train == c
        if mask.any():
            centroids[c] = f_train[mask].mean(dim=0)
    cen = F.normalize(centroids, p=2, dim=-1)
    q = F.normalize(f_test, p=2, dim=-1)
    pred = (q @ cen.t()).argmax(dim=1)  # (M,) 最も似た重心のクラス
    return float((pred == y_test).float().mean().item())


# =====================================================================
# 可視化
# =====================================================================

def save_comparison_bar(results: dict, path: pathlib.Path) -> None:
    """バックボーン横並び比較の棒グラフ（クリーン/ノイズ強の線形プローブ＋重心分類）。"""
    names = list(results.keys())
    clean = [results[n]["probe_top1"] for n in names]
    noisy = [results[n]["probe_hard_top1"] for n in names]
    centroid = [results[n]["centroid_top1"] for n in names]

    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(x - width, clean, width, label="linear probe (clean)", color="#3b78c3")
    ax.bar(x, noisy, width, label="linear probe (hard noise)", color="#e08a1e")
    ax.bar(x + width, centroid, width, label="nearest-centroid (no training)", color="#5aa469")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Backbone shootout: frozen features for 9-class shapes")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for xi, vals in zip(x, zip(clean, noisy, centroid)):
        for dx, v in zip((-width, 0, width), vals):
            ax.text(xi + dx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_confusion(cm: np.ndarray, path: pathlib.Path, title: str) -> None:
    """混同行列ヒートマップ（クラス名は ASCII なので豆腐にならない）。"""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(SHAPE_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SHAPE_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
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


def save_retrieval(
    query_imgs: list, neighbor_rows: list[list], path: pathlib.Path, backbone: str
) -> None:
    """画像検索の結果パネル。各行 = [クエリ画像] + 上位 3 近傍(タイトルに cos と正誤)。"""
    n_rows = len(query_imgs)
    n_cols = 1 + len(neighbor_rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.6 * n_rows))
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        axes[r, 0].imshow(query_imgs[r])
        axes[r, 0].set_title("query", fontsize=9)
        axes[r, 0].axis("off")
        for c, (img, ttl) in enumerate(neighbor_rows[r], start=1):
            axes[r, c].imshow(img)
            axes[r, c].set_title(ttl, fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(f"Image retrieval by cosine similarity  [{backbone}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# メイン
# =====================================================================

def main() -> None:
    set_seed(0)
    out_dir = ensure_out_dir()
    device = get_device()
    print("=" * 64)
    print("章末ミニプロジェクト: バックボーン比べ × 転移学習 × 画像検索")
    print(f"  device = {device} / classes = {NUM_CLASSES}")

    # ---------- 1. データ生成（学習 / クリーンテスト / 強ノイズテスト） ----------
    train_imgs, train_y = make_dataset(TRAIN_PER_CLASS, size=96, seed=1, noise_std=TRAIN_NOISE)
    test_imgs, test_y = make_dataset(TEST_PER_CLASS, size=96, seed=2, noise_std=TRAIN_NOISE)
    hard_imgs, hard_y = make_dataset(TEST_PER_CLASS, size=96, seed=3, noise_std=HARD_NOISE)
    y_train = torch.from_numpy(train_y)
    y_test = torch.from_numpy(test_y)
    y_hard = torch.from_numpy(hard_y)
    print(
        f"  data: train={len(train_imgs)}  test={len(test_imgs)}  "
        f"hard(noise={HARD_NOISE:.0f})={len(hard_imgs)}"
    )

    # ---------- 2. 各バックボーンをベンチマーク ----------
    backbones = ["resnet18_tv", "resnet18_timm", "vit_tiny_cls"]
    results: dict = {}
    cache: dict = {}  # 検索デモ用に best backbone の特徴を保持する
    print("\n  backbone        dim  extract(s)  probe_top1  probe_F1  hard_top1  centroid")
    for name in backbones:
        embed, dim = build_embedder(name, device)
        t0 = time.time()
        f_train = extract_features(embed, train_imgs)
        f_test = extract_features(embed, test_imgs)
        f_hard = extract_features(embed, hard_imgs)
        extract_sec = time.time() - t0

        head = train_linear_probe(f_train, y_train, dim)
        with torch.inference_mode():
            logits_test = head(f_test)
            logits_hard = head(f_hard)
        top1, mf1 = score(logits_test, y_test)
        hard_top1, _ = score(logits_hard, y_hard)
        centroid_top1 = nearest_centroid_accuracy(f_train, y_train, f_test, y_test)

        results[name] = {
            "embed_dim": dim,
            "extract_sec": round(extract_sec, 2),
            "probe_top1": round(top1, 4),
            "probe_macro_f1": round(mf1, 4),
            "probe_hard_top1": round(hard_top1, 4),
            "centroid_top1": round(centroid_top1, 4),
        }
        cache[name] = {
            "f_train": f_train, "f_test": f_test, "logits_test": logits_test,
        }
        print(
            f"  {name:14s} {dim:4d} {extract_sec:10.2f}  {top1:10.3f} {mf1:9.3f} "
            f"{hard_top1:10.3f} {centroid_top1:9.3f}"
        )

    # ---------- 3. ベスト backbone を選ぶ（probe_top1 → macro_f1 → 抽出が速い順） ----------
    best = max(
        results,
        key=lambda n: (
            results[n]["probe_top1"],
            results[n]["probe_macro_f1"],
            -results[n]["extract_sec"],
        ),
    )
    print(f"\n  → best backbone (by clean probe top-1): {best}")

    # ---------- 4. 比較棒グラフ ----------
    save_comparison_bar(results, out_dir / "mini_project_backbone_comparison.png")

    # ---------- 5. ベスト backbone の混同行列（線形プローブ・クリーンテスト） ----------
    pred_best = cache[best]["logits_test"].argmax(-1).numpy()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(test_y, pred_best):
        cm[int(t), int(p)] += 1
    save_confusion(
        cm,
        out_dir / "mini_project_confusion_best.png",
        f"Confusion (linear probe, {best})  top1={results[best]['probe_top1']:.2f}",
    )

    # ---------- 6. 画像検索デモ（ベスト backbone の埋め込みでコサイン近傍検索） ----------
    f_train_b = F.normalize(cache[best]["f_train"], p=2, dim=-1)
    f_test_b = F.normalize(cache[best]["f_test"], p=2, dim=-1)
    sims = f_test_b @ f_train_b.t()  # (n_test, n_train) コサイン類似度
    # クラスをまたいで 4 件クエリを選ぶ（テスト集合の等間隔サンプル）。
    q_idx = list(range(0, len(test_imgs), max(1, len(test_imgs) // 4)))[:4]
    query_imgs, neighbor_rows = [], []
    retrieval_hits = 0
    for qi in q_idx:
        query_imgs.append(test_imgs[qi])
        top3 = torch.topk(sims[qi], k=3).indices.tolist()
        row = []
        for rank, ti in enumerate(top3):
            cls = SHAPE_CLASSES[int(train_y[ti])]
            correct = int(train_y[ti]) == int(test_y[qi])
            if rank == 0 and correct:
                retrieval_hits += 1
            mark = "OK" if correct else "x"
            row.append((train_imgs[ti], f"{cls}\ncos={sims[qi, ti]:.2f} [{mark}]"))
        neighbor_rows.append(row)
    save_retrieval(query_imgs, neighbor_rows, out_dir / "mini_project_retrieval.png", best)
    retrieval_at1 = retrieval_hits / len(q_idx)
    print(
        f"  検索デモ: クエリ {len(q_idx)} 件中 top-1 近傍がクラス一致 = "
        f"{retrieval_hits}/{len(q_idx)} ({retrieval_at1:.2f})"
    )

    # ---------- 7. 指標 JSON ----------
    metrics = {
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "classes": SHAPE_CLASSES,
        "data": {
            "train": len(train_imgs),
            "test": len(test_imgs),
            "hard_noise_std": HARD_NOISE,
        },
        "backbones": results,
        "best_backbone": best,
        "retrieval_precision_at1": round(retrieval_at1, 4),
        "confusion_best": cm.tolist(),
    }
    json_path = out_dir / "mini_project_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  保存物:")
    print("    - 比較棒グラフ : mini_project_backbone_comparison.png")
    print("    - 混同行列     : mini_project_confusion_best.png")
    print("    - 検索デモ     : mini_project_retrieval.png")
    print(f"    - 指標 JSON    : {json_path.name}")
    print("\n完成: 凍結特徴 → 線形プローブ / 重心分類 / 近傍検索 を 3 バックボーンで横並び評価できた。")
    print("発展: backbone を増やす・hard noise を上げる・実写真を data/ に置いて頑健性を測る。")


if __name__ == "__main__":
    main()
