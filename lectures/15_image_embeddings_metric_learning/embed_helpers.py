"""共有ヘルパ: device 判定 / 合成ラベル付きデータセット / コサイン kNN / Recall@k。

このモジュール内で自己完結する小道具を集めたファイル。各スクリプトは
`import embed_helpers as eh` で利用する（同ディレクトリに置いてあるので、
`uv run python lectures/15_image_embeddings_metric_learning/01_xxx.py` のように
直接実行すれば sys.path にスクリプトのディレクトリが入り、そのまま import できる）。

ここに置くのは「モデルに依存しない」汎用部品だけ:
  - device の自動判定（cpu / mps / cuda）
  - 合成のラベル付き画像データセット（外部データ無しで毎回同じものを再現）
  - L2 正規化・コサイン類似度・kNN 分類精度・Recall@k
埋め込みの「抽出」そのもの（ViT/ResNet/timm の使い方）は、学習目的なので
各スクリプト側に厚いコメント付きで書く（ここには持ち込まない）。
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image

MODULE_ID = "15_image_embeddings_metric_learning"
OUT_DIR = os.path.join("outputs", MODULE_ID)


# ---------------------------------------------------------------------------
# device 判定（全スクリプト共通の定石）
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """cuda → mps → cpu の優先順で利用可能な device を返す。

    本講座は CPU 前提だが、Apple Silicon(mps) や GPU(cuda) があれば自動で使う。
    重要: モデルと入力テンソルは必ず同じ device に載せる（片方だけ .to() すると
    RuntimeError）。各スクリプトでは model.to(device) と inputs.to(device) を必ず揃える。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 合成ラベル付きデータセット（外部データ無しで動く・乱数シードで再現可能）
# ---------------------------------------------------------------------------
# 「形 × 色」で見分けやすい 6 クラス。学習済みバックボーン（ImageNet）でも
# 視覚的に似た画像は近いベクトルに写るので、同クラスがクラスタを作る。
CLASS_SPECS: list[tuple[str, tuple[int, int, int]]] = [
    ("red_circle", (60, 60, 220)),  # 色は BGR（cv2 で描くため）
    ("green_square", (70, 200, 80)),
    ("blue_triangle", (220, 120, 40)),
    ("yellow_ring", (40, 210, 230)),
    ("magenta_cross", (200, 60, 200)),
    ("cyan_diamond", (210, 200, 40)),
]


def _draw_shape(name: str, color: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """1 枚の合成画像（112x112, BGR uint8）を作る。位置・大きさ・色・ノイズを揺らす。

    同じクラスでも見た目をばらつかせることで、kNN/Recall@k が「丸暗記」ではなく
    本当に埋め込みの近さで効いているかを試せるようにする。
    """
    h = w = 112
    # 背景はうっすらノイズの乗ったグレー（真っ黒だと不自然に簡単になりすぎる）
    bg = int(rng.integers(40, 90))
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    img = cv2.add(img, rng.integers(0, 25, (h, w, 3), dtype=np.uint8))

    # クラスごとに少しだけ色を揺らす（照明差のつもり）
    jitter = rng.integers(-25, 25, size=3)
    col = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(color, jitter))

    cx = int(rng.integers(45, 67))  # 中心位置を少し動かす
    cy = int(rng.integers(45, 67))
    r = int(rng.integers(26, 36))  # 大きさを揺らす

    if name == "red_circle":
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
    elif name == "green_square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
    elif name == "blue_triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    elif name == "yellow_ring":
        cv2.circle(img, (cx, cy), r, col, max(4, r // 4), cv2.LINE_AA)  # 中空のリング
    elif name == "magenta_cross":
        t = max(5, r // 3)
        cv2.rectangle(img, (cx - r, cy - t), (cx + r, cy + t), col, -1)
        cv2.rectangle(img, (cx - t, cy - r), (cx + t, cy + r), col, -1)
    elif name == "cyan_diamond":
        pts = np.array([[cx, cy - r], [cx - r, cy], [cx, cy + r], [cx + r, cy]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)

    # 全体に軽い回転をかけて、さらにバリエーションを増やす
    ang = float(rng.uniform(-20, 20))
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, rot, (w, h), borderValue=(bg, bg, bg))
    return img


def make_dataset(
    n_per_class: int = 20, seed: int = 0
) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """合成データセットを作る。返り値は (PIL画像リスト, ラベル配列, クラス名リスト)。

    PIL.Image(RGB) で返すのは、HuggingFace の ImageProcessor がそのまま受け取れる形だから。
    cv2 は BGR なので、PIL に渡す前に必ず RGB へ変換する（BGR→RGB を忘れると色が反転する）。
    """
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    class_names = [name for name, _ in CLASS_SPECS]
    for cls_idx, (name, color) in enumerate(CLASS_SPECS):
        for _ in range(n_per_class):
            bgr = _draw_shape(name, color, rng)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # ★ BGR→RGB
            images.append(Image.fromarray(rgb))
            labels.append(cls_idx)
    return images, np.asarray(labels, dtype=np.int64), class_names


# ---------------------------------------------------------------------------
# 埋め込みの評価（コサイン類似度ベース）
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """各行ベクトルを L2 ノルム 1 に正規化する。

    正規化後は「内積 = コサイン類似度」になる。コサイン類似度は向きだけを見るので、
    ベクトルの長さ（明るさやコントラストで変わりやすい）に振り回されず、検索/分類が安定する。
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(行ごと) と b(行ごと) のコサイン類似度行列 (len(a), len(b)) を返す。"""
    return l2_normalize(a) @ l2_normalize(b).T


def knn_accuracy(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
    k: int = 5,
) -> float:
    """kNN 分類精度。各クエリの「コサイン最近傍 k 件」を多数決し、正解ラベルと比較する。

    手順:
      1. クエリ×ギャラリーのコサイン類似度行列を作る（大きいほど近い）。
      2. 各クエリ行で類似度上位 k 件のギャラリー添字を取る。
      3. その k 件のラベルを多数決し、クエリの予測ラベルとする。
      4. 予測 == 正解 の割合（accuracy）を返す。
    埋め込みが良いほど「同クラスが近くに集まる」ので精度が上がる、という素直な指標。
    """
    sim = cosine_sim_matrix(query, gallery)  # (nq, ng) 大きいほど近い
    topk = np.argsort(-sim, axis=1)[:, :k]  # 各行で類似度降順の上位 k 添字
    neigh_labels = gallery_labels[topk]  # (nq, k)
    preds = np.array(
        [np.bincount(row).argmax() for row in neigh_labels], dtype=np.int64
    )  # 多数決
    return float((preds == query_labels).mean())


def recall_at_k(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Recall@k = 各クエリの上位 k 件に「同クラス」が 1 件でも入る割合。

    kNN 分類（多数決で 1 ラベルに当てる）とは別の、検索向けの指標。
    『欲しい仲間が上位 k に出てくるか』だけを問うので、しきい値は緩め。
    """
    sim = cosine_sim_matrix(query, gallery)
    topk = np.argsort(-sim, axis=1)[:, :k]
    neigh_labels = gallery_labels[topk]  # (nq, k)
    hit = (neigh_labels == query_labels[:, None]).any(axis=1)  # 上位kに同クラスがあるか
    return float(hit.mean())
