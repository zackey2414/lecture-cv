"""use_case.py — 「自分のカテゴリで画像分類器」を作る実践ツール（フォルダ＝クラス方式）。

==============================================================================
これは何か（mini_project.py との違い）
==============================================================================
本章の `mini_project.py` は「3 つのバックボーンを横並びでベンチマークする教材」でした。
こちらの `use_case.py` は、その学びを **現実の小さなツール** に落とし込んだものです。

  ねらい:  data/13_classification_transfer_learning/<クラス名>/ に画像を放り込むだけで、
           事前学習バックボーンの「凍結特徴 + 線形プローブ（= nn.Linear 1 枚）」で
           あなた専用の画像分類器を **少数枚から** 学習し、保存して、推論まで行う。

  Teachable-Machine 的なツールの最小実装です。学習結果（線形ヘッドの重み + クラス名）を
  outputs/ に「モデル成果物(.pt)」として保存し、それを読み直して新しい画像を分類します
  （= 一度学習すれば再利用できる、という実運用の形）。mini_project が「比較実験」なら、
  こちらは「あなたのデータで動く完成品の出発点」です。

==============================================================================
データの置き方（実データ優先・無ければ合成で必ず完走）
==============================================================================
実データを使う場合は、クラスごとにサブフォルダを作って画像を入れてください:

    data/13_classification_transfer_learning/
      ├── cat/        ← フォルダ名がそのままクラス名になる
      │     ├── img001.jpg
      │     └── img002.png
      ├── dog/
      │     ├── ...
      └── _inbox/     ← (任意) ラベル無しの「分類したい画像」を置く場所
            └── unknown01.jpg

  ・クラスは 2 つ以上、各クラス 3〜5 枚もあれば線形プローブは動きます（少データでOK）。
  ・`_inbox/`（_ で始まるフォルダ）はクラスとして扱わず、推論デモの入力に使います。
  ・実データが見つからない（クラスが 2 つ未満）ときは、dl_helpers の合成図形で
    「フォルダ＝クラス」のデータを擬似生成して必ず完走します（exit 0）。

==============================================================================
拡張アイデア（ここから練習を広げる）
==============================================================================
  1. クラスを増やす  : data/ にフォルダを足すだけで再学習できる（コード変更不要）。
  2. バックボーン交換: BACKBONE を timm の `mobilenetv3_small_100` 等に差し替えて速度/精度比較。
  3. データ拡張      : 学習画像に torchvision.transforms.v2 の回転/色ジッタを足して頑健化。
  4. しきい値で棄却  : 推論の softmax 信頼度が低いときは "unknown" を返す「オープンセット」化。
  5. 増分学習        : 保存した .pt を読み込み、新クラスのフォルダを足して線形層だけ再学習。
  6. API 化          : `load_classifier()` + `predict()` をそのまま FastAPI などから呼ぶ。

==============================================================================
実行
==============================================================================
    uv run python lectures/13_classification_transfer_learning/use_case.py

結果（学習済みモデル .pt・予測パネル・混同行列・指標 JSON）は
lectures/13_classification_transfer_learning/outputs/ に保存されます（画面表示はしません）。
CPU 完結。事前学習 ResNet-18 の重みのみ初回にダウンロードしてキャッシュします。
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境で固まらないよう、pyplot より前に Agg を指定

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

# 同じフォルダの dl_helpers を確実に import できるようにする（番号付きスクリプトは import しない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dl_helpers import (  # noqa: E402
    DATA_DIR,
    OUT_DIR,
    ensure_out_dir,
    get_device,
    make_shape_image,
    set_seed,
)

# ---- ツール設定（少データ・CPU で数十秒に収まる規模） --------------------------
BACKBONE = "resnet18_tv"        # 凍結特徴の抽出器。512 次元 penultimate を使う
EMBED_DIM = 512
IMG_SIZE = 160                  # ImageNet 前処理の入力サイズ（小さめでCPUを軽く）
BATCH = 16
PROBE_STEPS = 150               # 線形ヘッドの学習ステップ数（小データなので全件1バッチ）
PROBE_LR = 1e-2
VAL_RATIO = 0.3                 # クラスごとに検証へ回す割合（最低1枚は学習に残す）

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INBOX_DIRNAME = "_inbox"        # ラベル無しの推論対象を置くフォルダ名

# 合成フォールバック用の「あなたのカテゴリ」（dl_helpers の図形クラスから選ぶ）。
SYNTH_CLASSES = ["red_circle", "green_square", "blue_triangle", "red_square"]
SYNTH_PER_CLASS = 12


# =====================================================================
# 1. データ読み込み（実データ＝フォルダ／クラス、無ければ合成）
# =====================================================================

def scan_class_dirs(root: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """root 直下の「クラス名フォルダ → 画像パス一覧」を集める。

    `.` や `_` で始まるフォルダ（例: _inbox）はクラスとして扱わない。
    画像を1枚も含まないフォルダも無視する。
    """
    classes: dict[str, list[pathlib.Path]] = {}
    if not root.is_dir():
        return classes
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if files:
            classes[d.name] = files
    return classes


def load_real_dataset(
    class_dirs: dict[str, list[pathlib.Path]],
) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """フォルダ＝クラスの実画像を読み込んで (images, labels, class_names) を返す。"""
    class_names = sorted(class_dirs)
    images: list[Image.Image] = []
    labels: list[int] = []
    for idx, name in enumerate(class_names):
        for path in class_dirs[name]:
            try:
                images.append(Image.open(path).convert("RGB"))
                labels.append(idx)
            except (OSError, ValueError):
                # 壊れた画像はスキップ（実データではよくある）。落とさず続ける。
                print(f"  [warn] 読み込み失敗のためスキップ: {path}")
    return images, np.asarray(labels, dtype=np.int64), class_names


def make_synthetic_dataset(
    per_class: int = SYNTH_PER_CLASS, seed: int = 0
) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """実データが無いときのフォールバック。合成図形で「フォルダ＝クラス」を擬似再現する。"""
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    for idx, name in enumerate(SYNTH_CLASSES):
        for _ in range(per_class):
            images.append(make_shape_image(name, size=120, rng=rng, noise_std=14.0))
            labels.append(idx)
    return images, np.asarray(labels, dtype=np.int64), list(SYNTH_CLASSES)


def split_per_class(
    labels: np.ndarray, val_ratio: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """クラスごとに train / val のインデックスへ層化分割する（各クラス最低1枚は学習に残す）。"""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        # 1枚しかないクラスは全部 train（検証へは回さない）。
        n_val = max(1, round(len(idx) * val_ratio)) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


# =====================================================================
# 2. 凍結特徴の抽出器（ResNet-18 penultimate 512 次元）
# =====================================================================

def build_embedder(device: torch.device):
    """事前学習 ResNet-18 の fc を Identity に差し替え、画像 → 512 次元特徴の関数を返す。

    embed(list[PIL]) -> Tensor (n, 512)（CPU 上・勾配なし）。
    特徴は no_grad で取り出す（inference_mode で作ると後段のヘッド学習で autograd に使えない）。
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToImage(),                            # PIL → uint8 CHW tensor
            T.ToDtype(torch.float32, scale=True),   # 0-255 → 0-1
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def embed(imgs: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([tf(im) for im in imgs]).to(device)
        with torch.no_grad():
            return model(x).cpu()

    return embed


def extract_features(embed, images: list[Image.Image]) -> torch.Tensor:
    """全画像をミニバッチで埋め込みに変換して (N, EMBED_DIM) を返す。"""
    feats = [embed(images[i : i + BATCH]) for i in range(0, len(images), BATCH)]
    return torch.cat(feats, dim=0)


# =====================================================================
# 3. 線形プローブの学習・保存・読み込み・推論（= ツールの中核）
# =====================================================================

def train_linear_probe(
    features: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> nn.Linear:
    """凍結特徴の上に nn.Linear ヘッドだけを学習する（転移学習の本質部分）。"""
    head = nn.Linear(features.shape[1], n_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()
    head.train()
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad()
        loss = loss_fn(head(features), labels)  # 小データなので全件を1バッチで
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def save_classifier(head: nn.Linear, class_names: list[str], path: pathlib.Path) -> None:
    """学習済み線形ヘッド＋クラス名＋構成をモデル成果物(.pt)として保存する。"""
    torch.save(
        {
            "backbone": BACKBONE,
            "embed_dim": head.in_features,
            "img_size": IMG_SIZE,
            "classes": class_names,
            "head_state": head.state_dict(),
        },
        path,
    )


def load_classifier(path: pathlib.Path, device: torch.device):
    """保存した .pt を読み直し、(embed, head, class_names) を返す（再利用の実演）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    head = nn.Linear(ckpt["embed_dim"], len(ckpt["classes"]))
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    embed = build_embedder(device)
    return embed, head, ckpt["classes"]


def predict(
    embed, head: nn.Linear, class_names: list[str], images: list[Image.Image]
) -> tuple[list[str], list[float]]:
    """画像リストを分類し、(予測ラベル名, 信頼度=softmax最大値) を返す。"""
    feats = extract_features(embed, images)
    with torch.inference_mode():
        probs = torch.softmax(head(feats), dim=1)
    conf, idx = probs.max(dim=1)
    preds = [class_names[i] for i in idx.tolist()]
    return preds, conf.tolist()


# =====================================================================
# 4. 評価・可視化
# =====================================================================

def evaluate(
    head: nn.Linear, f_val: torch.Tensor, y_val: torch.Tensor, n_classes: int
) -> tuple[float, np.ndarray]:
    """検証特徴に対する top-1 accuracy と混同行列を返す。"""
    with torch.inference_mode():
        pred = head(f_val).argmax(dim=1)
    acc = float((pred == y_val).float().mean().item())
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_val.tolist(), pred.tolist()):
        cm[t, p] += 1
    return acc, cm


def _ascii(label: str, fallback_idx: int) -> str:
    """図のタイトル用に非ASCII（日本語フォルダ名など）を安全化する。豆腐(□)を避ける。"""
    safe = label.encode("ascii", "ignore").decode("ascii").strip()
    return safe if safe else f"class{fallback_idx}"


def save_confusion(cm: np.ndarray, class_names: list[str], path: pathlib.Path) -> None:
    """混同行列のヒートマップを保存する（軸ラベルは ASCII 化）。"""
    n = len(class_names)
    labels = [_ascii(c, i) for i, c in enumerate(class_names)]
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.3 * n + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Custom classifier confusion (linear probe)")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_prediction_panel(
    images: list[Image.Image], titles: list[str], path: pathlib.Path, suptitle: str
) -> None:
    """推論結果（画像＋予測ラベル/信頼度）を並べたパネルを保存する。"""
    n = len(images)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5. メイン（データ取得 → 学習 → 保存 → 読み直して推論）
# =====================================================================

def main() -> None:
    set_seed(0)
    out_dir = ensure_out_dir()
    device = get_device()
    print("=" * 64)
    print("実践ユースケース: 自分のカテゴリで画像分類器（凍結特徴 + 線形プローブ）")
    print(f"  device = {device}")

    # ---------- 1. データ取得（実データ優先・無ければ合成） ----------
    class_dirs = scan_class_dirs(DATA_DIR)
    if len(class_dirs) >= 2:
        images, labels, class_names = load_real_dataset(class_dirs)
        source = "real"
        print(f"  データ源: 実データ {DATA_DIR}")
    else:
        images, labels, class_names = make_synthetic_dataset()
        source = "synthetic"
        print("  データ源: 合成図形（data/ にクラス別フォルダを置くと実データに切り替わります）")
    n_classes = len(class_names)
    print(f"  クラス({n_classes}): {class_names}")
    print(f"  画像枚数: {len(images)}")

    # ---------- 2. train / val 分割（クラスごとに層化） ----------
    train_idx, val_idx = split_per_class(labels, VAL_RATIO, seed=0)
    train_imgs = [images[i] for i in train_idx]
    y_train = torch.from_numpy(labels[train_idx])
    val_imgs = [images[i] for i in val_idx]
    y_val = torch.from_numpy(labels[val_idx]) if len(val_idx) else None
    print(f"  分割: train={len(train_imgs)}  val={len(val_idx)}")

    # ---------- 3. 凍結特徴の抽出 → 線形プローブ学習 ----------
    embed = build_embedder(device)
    f_train = extract_features(embed, train_imgs)
    head = train_linear_probe(f_train, y_train, n_classes)

    # ---------- 4. 検証で評価（val が空＝各クラス1枚なら学習データで代用） ----------
    if y_val is not None and len(val_idx):
        f_val = extract_features(embed, val_imgs)
        val_acc, cm = evaluate(head, f_val, y_val, n_classes)
        eval_split = "val"
    else:
        val_acc, cm = evaluate(head, f_train, y_train, n_classes)
        eval_split = "train (val が無いため学習データで代用)"
    print(f"  top-1 accuracy ({eval_split}) = {val_acc:.3f}")
    save_confusion(cm, class_names, out_dir / "use_case_confusion.png")

    # ---------- 5. モデル成果物を保存 → 読み直して再利用（実運用の形） ----------
    artifact_path = out_dir / "use_case_classifier.pt"
    save_classifier(head, class_names, artifact_path)
    print(f"  学習済みモデルを保存: {artifact_path.name}")
    embed2, head2, classes2 = load_classifier(artifact_path, device)  # 保存物だけで推論できる実証

    # ---------- 6. 推論デモ（data/.../_inbox 優先・無ければ val/train から） ----------
    inbox_dir = DATA_DIR / INBOX_DIRNAME
    inbox_files = (
        sorted(p for p in inbox_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if inbox_dir.is_dir()
        else []
    )
    if inbox_files:
        query_imgs = [Image.open(p).convert("RGB") for p in inbox_files[:8]]
        query_src = f"_inbox ({len(query_imgs)} 枚のラベル無し画像)"
    else:
        pool = val_imgs if val_imgs else train_imgs
        query_imgs = pool[:8]
        query_src = "検証画像（_inbox が無いため）"

    preds, confs = predict(embed2, head2, classes2, query_imgs)
    titles = [f"{_ascii(p, 0)}\nconf={c:.2f}" for p, c in zip(preds, confs)]
    save_prediction_panel(
        query_imgs, titles, out_dir / "use_case_predictions.png",
        "Predictions by your custom classifier",
    )
    print(f"  推論デモ ({query_src}):")
    for p, c in zip(preds, confs):
        print(f"    - 予測: {p:16s} (信頼度 {c:.2f})")

    # ---------- 7. 指標 JSON ----------
    metrics = {
        "source": source,
        "backbone": BACKBONE,
        "classes": class_names,
        "num_images": len(images),
        "train_size": len(train_imgs),
        "val_size": int(len(val_idx)),
        "eval_split": eval_split,
        "top1_accuracy": round(val_acc, 4),
        "confusion": cm.tolist(),
        "artifact": artifact_path.name,
        "query_predictions": [
            {"pred": p, "confidence": round(c, 4)} for p, c in zip(preds, confs)
        ],
    }
    (out_dir / "use_case_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n  保存物 (lectures/13_classification_transfer_learning/outputs/):")
    print("    - 学習済みモデル : use_case_classifier.pt")
    print("    - 混同行列       : use_case_confusion.png")
    print("    - 推論パネル     : use_case_predictions.png")
    print("    - 指標 JSON      : use_case_metrics.json")
    print("\n完成: フォルダ＝クラスの画像から、凍結特徴＋線形プローブで自前分類器を学習・保存・推論できた。")
    if source == "synthetic":
        print(f"次の一歩: {DATA_DIR}/<クラス名>/ に実画像を置いて再実行すると、あなたのデータで動きます。")


if __name__ == "__main__":
    main()
