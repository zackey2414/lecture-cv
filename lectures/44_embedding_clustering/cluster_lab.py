"""第44回 共通の道具箱（埋め込みのクラスタリング）。

このファイルは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務は単一責務で次の5つに絞ってあります:

  1. device の自動判定（cuda > mps > cpu。既定は cpu）と出力ディレクトリの用意
  2. 合成データの生成（画像コレクション＝色×形で既知グループ／テキスト集合＝トピック別）
  3. CLIP 埋め込みの取り出し（画像・テキストを同じ 512 次元空間へ。**L2 正規化込み**）
  4. クラスタリングの評価（教師なし silhouette ／ 正解ありの purity/NMI/homogeneity ほか）
  5. 共通の可視化（2D 散布図・クラスタ別アルバム）

なぜ合成データか:
  検索（16/17）とは違い、本章のゴールは「ラベル無しで自動グルーピングできること」です。
  ところが教材としては『答え合わせ』もしたい。そこで色×形で“既知グループ”を持つ合成画像と、
  トピックの分かったテキストを使い、ラベルを **クラスタリングには渡さず** 評価だけに使います。
  CLIP は「赤い丸」「青い三角」のような色＋形の概念を強く捉えるので、外部データ無しでも
  クラスタが綺麗に分かれ、silhouette や NMI に意味のある差が出ます。

実行（このファイル単体でスモークテスト）:
  uv run python lectures/44_embedding_clustering/cluster_lab.py
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# transformers / huggingface_hub のログを静かにする（教材出力を読みやすく）。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

MODULE_ID = "44_embedding_clustering"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

# CPU でも現実的な「最速の定番」。patch16 や large は CPU では重い。
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# 既知グループ（色×形）。それぞれ色も形も違うので CLIP 埋め込みが綺麗に分かれる。
# 色は (R, G, B)。cv2 で描くが imread/imwrite を経由しないので BGR 混乱は起きない。
GROUPS: list[tuple[str, tuple[int, int, int], str]] = [
    ("red circle", (210, 45, 45), "circle"),
    ("green square", (45, 175, 70), "square"),
    ("blue triangle", (50, 90, 215), "triangle"),
    ("yellow diamond", (235, 200, 35), "diamond"),
    ("magenta cross", (200, 60, 190), "cross"),
    ("cyan star", (40, 195, 210), "star"),
]

# テキスト集合。4 トピック（動物・乗り物・食べ物・天気）。ラベルは評価専用。
TEXT_TOPICS: dict[str, list[str]] = {
    "animals": [
        "a cat sleeping on a sofa",
        "a dog running in the park",
        "a horse grazing in a field",
        "a small bird singing on a branch",
        "an elephant walking near a river",
    ],
    "vehicles": [
        "a red sports car on the highway",
        "an airplane taking off from the runway",
        "a sailboat crossing the blue sea",
        "a motorcycle parked on the street",
        "a train arriving at the station",
    ],
    "food": [
        "a slice of pepperoni pizza",
        "a bowl of fresh green salad",
        "a cup of hot coffee with milk",
        "a plate of sushi rolls",
        "a chocolate cake on a table",
    ],
    "weather": [
        "a sunny day with a clear blue sky",
        "heavy rain falling on the city",
        "snow covering the mountain peaks",
        "thick fog over a quiet lake",
        "a thunderstorm with bright lightning",
    ],
}


# ---------------------------------------------------------------------------
# device / 出力ディレクトリ
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず揃える（.to(device) を両方に）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_out_dir() -> pathlib.Path:
    """lectures/<module_id>/outputs/ を作って返す（既にあってもエラーにしない）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# ---------------------------------------------------------------------------
# 合成データ（画像コレクション）
# ---------------------------------------------------------------------------
def _draw_shape(
    shape: str, color: tuple[int, int, int], rng: np.random.Generator, size: int = 96
) -> np.ndarray:
    """1 枚の合成画像（size 角・RGB uint8）を描く。位置・大きさ・色・回転を揺らす。

    同じグループでも見た目をばらつかせることで、クラスタリングが『丸暗記』ではなく
    本当に埋め込みの近さで効いているかを試せるようにする。
    """
    bg = int(rng.integers(235, 250))  # 明るい灰色の背景（少し揺らす）
    img = np.full((size, size, 3), bg, dtype=np.uint8)

    # グループ色を少しだけ揺らす（照明差のつもり）。RGB のまま描く。
    jit = rng.integers(-20, 20, size=3)
    col = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(color, jit))

    cx = int(rng.integers(size * 0.44, size * 0.56))  # 中心を少し動かす
    cy = int(rng.integers(size * 0.44, size * 0.56))
    r = int(rng.integers(size * 0.26, size * 0.34))  # 大きさを揺らす

    if shape == "circle":
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    elif shape == "diamond":
        pts = np.array([[cx, cy - r], [cx - r, cy], [cx, cy + r], [cx + r, cy]], np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    elif shape == "cross":
        t = max(4, r // 3)
        cv2.rectangle(img, (cx - r, cy - t), (cx + r, cy + t), col, -1)
        cv2.rectangle(img, (cx - t, cy - r), (cx + t, cy + r), col, -1)
    elif shape == "star":
        pts = _star_points(cx, cy, r, r * 0.45)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    else:
        raise ValueError(f"未知の形: {shape}")

    # 全体に軽い回転をかけてさらにバリエーションを増やす。
    ang = float(rng.uniform(-25, 25))
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
    img = cv2.warpAffine(img, rot, (size, size), borderValue=(bg, bg, bg))
    return img


def _star_points(cx: int, cy: int, r_out: float, r_in: float) -> np.ndarray:
    """5 角星の 10 頂点（外側・内側を交互）を int32 の (10, 2) で返す。"""
    pts = []
    for i in range(10):
        ang = -np.pi / 2 + i * np.pi / 5  # 頂点を上から時計回りに
        rad = r_out if i % 2 == 0 else r_in
        pts.append([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
    return np.array(pts, np.int32)


def make_image_collection(
    n_per_group: int = 6, seed: int = 0
) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """色×形の合成画像コレクションを作る。

    返り値:
      images: PIL.Image(RGB) のリスト（CLIP の ImageProcessor がそのまま受け取れる形）
      labels: 既知グループの ID 配列（int64）。**クラスタリングには渡さず評価専用**
      names:  グループ名のリスト（labels の値に対応）
    """
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    names = [g[0] for g in GROUPS]
    for gid, (_, color, shape) in enumerate(GROUPS):
        for _ in range(n_per_group):
            rgb = _draw_shape(shape, color, rng)
            images.append(Image.fromarray(rgb))  # RGB のまま PIL へ
            labels.append(gid)
    return images, np.asarray(labels, dtype=np.int64), names


def make_text_collection() -> tuple[list[str], np.ndarray, list[str]]:
    """トピック別テキスト集合を作る。返り値は (文リスト, トピックID配列, トピック名)。"""
    texts: list[str] = []
    labels: list[int] = []
    names = list(TEXT_TOPICS.keys())
    for tid, key in enumerate(names):
        for sentence in TEXT_TOPICS[key]:
            texts.append(sentence)
            labels.append(tid)
    return texts, np.asarray(labels, dtype=np.int64), names


# ---------------------------------------------------------------------------
# CLIP 埋め込み（画像・テキストを同じ空間へ・L2 正規化込み）
# ---------------------------------------------------------------------------
_CLIP_CACHE: dict[str, object] = {}


def load_clip(device: torch.device):
    """CLIP モデルとプロセッサを返す（eval + device 済み・モジュール内でキャッシュ）。

    transformers v5 の正準。get_image_features / get_text_features は
    BaseModelOutputWithPooling を返し、射影後ベクトルは .pooler_output に入る（後述）。
    """
    if "model" not in _CLIP_CACHE:
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _CLIP_CACHE["model"] = model
        _CLIP_CACHE["proc"] = proc
    return _CLIP_CACHE["model"], _CLIP_CACHE["proc"]


def embed_images(
    images: list[Image.Image], device: torch.device, batch_size: int = 16
) -> np.ndarray:
    """画像コレクションを CLIP で埋め込み、**L2 正規化した** (N, 512) float32 を返す。

    ★クラスタリングの大前提: 埋め込みは必ず L2 正規化する。正規化後は内積＝コサイン類似度に
    なり、ベクトルの長さ（明るさ等で揺れる）に振り回されず、距離ベースの k-means/DBSCAN/
    Agglomerative が安定する。get_image_features の戻り値は **未正規化** なので、ここで正規化を
    必ず挟む（16 章で確認した非対称: forward の logits は正規化済みだが get_*_features は違う）。
    """
    model, proc = load_clip(device)
    feats: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        with torch.inference_mode():  # 推論は勾配を切る（CPU で特に効く）
            out = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats.append(out.pooler_output.float().cpu().numpy())
    return l2_normalize(np.concatenate(feats, axis=0))


def embed_texts(texts: list[str], device: torch.device) -> np.ndarray:
    """テキスト集合を CLIP で埋め込み、**L2 正規化した** (N, 512) float32 を返す。

    画像と同じ 512 次元・同じ空間に乗る（クロスモーダル）。padding=True を忘れると
    複数文で長さ不一致エラーになる点に注意。
    """
    model, proc = load_clip(device)
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        out = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    return l2_normalize(out.pooler_output.float().cpu().numpy())


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """各行ベクトルを L2 ノルム 1 に正規化する（内積＝コサイン類似度になる）。"""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


# ---------------------------------------------------------------------------
# クラスタリングの評価
# ---------------------------------------------------------------------------
def purity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """purity = 各クラスタを『多数派の真ラベル』に割り当てた時の正解率。

    直感的だが、クラスタを増やすほど上がる（極端には 1 点 1 クラスタで purity=1）。
    NMI など『増やすほど得しない』指標と併読する。ノイズ(-1) も 1 クラスタ扱いで数える。
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total = len(labels_true)
    if total == 0:
        return 0.0
    # 真ラベルを 0..C-1 に詰め直してから bincount（負値や飛び番でも安全に）。
    _, true_idx = np.unique(labels_true, return_inverse=True)
    correct = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        correct += np.bincount(true_idx[mask]).max()
    return float(correct / total)


def silhouette_safe(emb: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """silhouette_score を安全に計算する（クラスタ 1 個 or 全ノイズなら nan）。

    silhouette は『自分のクラスタに近く、隣のクラスタから遠いほど 1 に近い』教師なし指標。
    正解ラベル不要で k 選びにも使える。クラスタが 2 個以上ないと定義できない点に注意。
    """
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    mask = labels != -1  # DBSCAN のノイズは除いて評価する
    if mask.sum() < 2:
        return float("nan")
    uniq = np.unique(labels[mask])
    if len(uniq) < 2 or len(uniq) >= mask.sum():
        return float("nan")
    return float(silhouette_score(emb[mask], labels[mask], metric=metric))


def clustering_report(
    labels_pred: np.ndarray, emb: np.ndarray, labels_true: np.ndarray | None = None
) -> dict:
    """クラスタリング結果を 1 つの dict にまとめる。

    教師なし指標（silhouette）は常に計算。labels_true があれば purity/NMI/homogeneity/
    completeness も足す。n_clusters はノイズ(-1)を除いた数、n_noise はノイズ点数。
    """
    labels_pred = np.asarray(labels_pred)
    n_clusters = len(set(labels_pred.tolist()) - {-1})
    n_noise = int(np.sum(labels_pred == -1))
    rep: dict = {
        "n_clusters_found": n_clusters,
        "n_noise": n_noise,
        "silhouette": silhouette_safe(emb, labels_pred),
    }
    if labels_true is not None:
        from sklearn.metrics import (
            completeness_score,
            homogeneity_score,
            normalized_mutual_info_score,
        )

        rep["purity"] = purity_score(labels_true, labels_pred)
        rep["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        rep["homogeneity"] = float(homogeneity_score(labels_true, labels_pred))
        rep["completeness"] = float(completeness_score(labels_true, labels_pred))
    return rep


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
def scatter_2d(
    xy: np.ndarray,
    pred: np.ndarray,
    labels_true: np.ndarray,
    title: str,
    path: pathlib.Path,
) -> None:
    """2D 座標 xy を『色＝予測クラスタ / 形＝真グループ』で散布図にして保存する。

    同じ色（予測）の点がまとまり、別グループ（形）が離れていれば、埋め込み空間でグループが
    綺麗に分離していることが目で分かる。図中の文字は CJK 豆腐を避けるため ASCII。
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for gid in np.unique(labels_true):
        m = labels_true == gid
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            c=pred[m],
            cmap="tab10",
            vmin=0,
            vmax=9,
            marker=markers[int(gid) % len(markers)],
            s=70,
            edgecolors="k",
            linewidths=0.4,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_cluster_album(
    images: list[Image.Image], pred: np.ndarray, path: pathlib.Path, max_per_row: int = 8
) -> None:
    """予測クラスタごとに画像を 1 行に並べた『アルバム』を作る（行＝自動グルーピング結果）。

    ノイズ(-1) は最終行にまとめ、各行の左端にクラスタ ID を書く。
    """
    cell = 64
    pad = 4
    label_w = 56
    clusters = sorted(set(pred.tolist()), key=lambda c: (c == -1, c))
    rows = []
    for c in clusters:
        idxs = np.where(pred == c)[0][:max_per_row]
        row = np.full((cell, label_w, 3), 245, dtype=np.uint8)
        tag = "noise" if c == -1 else f"#{c}"
        cv2.putText(row, tag, (3, cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for i in idxs:
            rgb = np.array(images[i].resize((cell, cell)))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # imwrite は BGR を期待
            row = np.hstack([row, np.full((cell, pad, 3), 245, np.uint8), bgr])
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((cell, width - r.shape[1], 3), 245, np.uint8)]) for r in rows]
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, np.full((pad, width, 3), 200, np.uint8), r])
    cv2.imwrite(str(path), canvas)


# ---------------------------------------------------------------------------
# 単体スモークテスト
# ---------------------------------------------------------------------------
def _selftest() -> None:
    out = ensure_out_dir()
    device = get_device()
    print(f"device={device}")

    images, labels, names = make_image_collection(n_per_group=4, seed=0)
    print(f"画像: {len(images)} 枚 / {len(names)} グループ {names}")
    emb = embed_images(images, device)
    print(f"埋め込み: shape={emb.shape}  各行ノルム≈{np.linalg.norm(emb, axis=1).mean():.3f}")

    from sklearn.cluster import KMeans

    pred = KMeans(n_clusters=len(names), n_init=10, random_state=0).fit_predict(emb)
    rep = clustering_report(pred, emb, labels)
    print(f"KMeans 評価: {rep}")

    texts, tlabels, tnames = make_text_collection()
    temb = embed_texts(texts, device)
    print(f"テキスト: {len(texts)} 文 / {len(tnames)} トピック  埋め込み shape={temb.shape}")
    print(f"スモークテスト OK（outputs は {out}）")


if __name__ == "__main__":
    _selftest()
