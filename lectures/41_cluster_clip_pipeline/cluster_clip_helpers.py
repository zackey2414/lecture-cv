"""41_cluster_clip_pipeline 共通ヘルパ — 我々の小さな `adaptive_cluster_clip` パッケージ。

この章は講座の総仕上げ（capstone）。Cluster-CLIP（手法）の
Split → Build → Search → Stream を、CPU で完走する小型版として再構築する。
番号付きスクリプト（01〜04）と mini_project はここの関数を呼ぶだけの薄いドライバにし、
「Cluster-CLIP の中身」はこのファイルに集約する（= 参考実装の scripts/ と src/ の関係）。

参考実装との対応:
  - load_clip / dense_clip_embeddings / cluster_agglomerative
        … src/adaptive_cluster_clip/build/models.py
  - run_split                … src/adaptive_cluster_clip/split/processor.py
  - run_build(consumer+indexer+db_writer を1関数に統合)
        … build/consumer.py + build/indexer.py + build/db_writer.py
  - search_index / overlay_cluster_mask / save_search_gallery
        … search/engine.py + search/visualizer.py
  - run_stream_pipeline / HistogramDeltaSampler / _capture/_consumer/_writer
        … stream/pipeline.py + stream/capture.py + stream/consumer.py + stream/writer.py

設計の勘所（参考実装と同一）:
  - CLIP は ViT-B-32 を force_quick_gelu=True でロードする。
  - 前処理は CenterCrop を排し、正方形に強制 Resize する（端を切らない＝dense 特徴の品質に直結）。
  - dense 特徴は visual transformer のパッチトークン列（CLS を除く）を空間 (H/ps, W/ps, D) に並べ替える。
  - コサイン類似は「L2 正規化 + 内積(IndexFlatIP)」。FAISS 入力は float32・C 連続を徹底。
  - すべて CPU・model.eval() + inference_mode()。学習はしない（推論 + クラスタリング + 検索）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# 画面のない環境でも図を保存できるよう Agg を明示（imshow は呼ばない）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import open_clip  # noqa: E402
import faiss  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.feature_extraction.image import grid_to_graph  # noqa: E402
from torchvision.transforms import Compose, Resize, ToTensor, Normalize  # noqa: E402


# ---------------------------------------------------------------------------
# 定数（モジュール全体で共有）
# ---------------------------------------------------------------------------
MODULE_ID = "41_cluster_clip_pipeline"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMG_SIZE = 224          # 正方形に強制リサイズするターゲット
PATCH_GRID = 7          # ViT-B-32: 224 / 32 = 7 → 7x7=49 パッチ
EMBED_DIM = 512         # ViT-B-32 の射影後次元（text/image/region で共通）
DB_NAME = "metadata.db"
INDEX_NAME = "index.faiss"
POISON_PILL = "__POISON_PILL__"   # ストリームのプロセス終了シグナル

# CLIP 専用の正規化統計（ImageNet とは別物。Normalize 層が取れない時のフォールバック用）
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 合成「動画」に登場する色つきオブジェクト（BGR）。各フレームを横に流して動画らしくする。
# これらが「領域」として検出される対象（小物体・複数物体の retrieval を体験するため）。
OBJECT_COLORS_BGR = {
    "red": (40, 40, 210),
    "green": (60, 175, 60),
    "blue": (210, 90, 40),
    "yellow": (40, 210, 220),
}


# ---------------------------------------------------------------------------
# パス・乱数・図の保存
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """リポジトリ直下を返す（lectures/<id>/ から 2 つ上）。"""
    return Path(__file__).resolve().parents[2]


def outputs_dir(sub: str | None = None) -> Path:
    """成果物保存先 outputs/41_cluster_clip_pipeline[/sub] を作って返す。"""
    d = repo_root() / "outputs" / MODULE_ID
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int = 0) -> None:
    """再現性のため numpy / torch の乱数を固定する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """本講座は CPU 前提。参考実装と同じく cuda 可用ならそちらを使う書き方にしておく。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_fig(fig, name: str, sub: str | None = None) -> Path:
    """図を outputs に保存してパスを返す（タイトルは ASCII でフォント警告回避）。"""
    path = outputs_dir(sub) / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 合成「動画」フレーム（cv2 で色つきオブジェクトを描く）
# ---------------------------------------------------------------------------
# 合成「動画」を 4 つのショット（場面）に分け、各ショットで登場オブジェクトを入れ替える。
# こうするとショット境界で色ヒストグラムが大きく変わり、適応サンプリング（ヒストグラム差分）が
# 「場面の変わり目だけキーフレームとして残す」様子を実演できる。ショット内は位置ドリフトのみ。
_SHOTS = [["red", "green"], ["green", "blue"], ["blue", "yellow"], ["yellow", "red"]]


def synth_frame(idx: int, n_frames: int, h: int = 180, w: int = 240) -> np.ndarray:
    """1 フレーム（BGR uint8, h x w x 3）を描く。

    薄い空色の背景に、ショットごとに決まる 2 色の矩形オブジェクトを配置する。
    ショット内ではフレーム番号 idx に応じて少しずつ横にドリフトさせる（= 動画らしさ）。
    背景 + 複数オブジェクトという構図なので、領域（dense）クラスタリングが意味を持つ。
    """
    img = np.full((h, w, 3), (245, 235, 220), dtype=np.uint8)  # 薄い空色（BGR）の背景

    shot = min(int(idx / max(1, n_frames) * len(_SHOTS)), len(_SHOTS) - 1)
    names = _SHOTS[shot]
    drift = (idx * 4) % 16  # ショット内の小さな横移動（位置は変わるが色構成は不変）

    # ショットごとに 2 色の大きなパネルを左右に描く。面積を大きく取ることで、
    # ショット境界での色入れ替えが色ヒストグラムを大きく動かす（=適応サンプリングが効く）。
    # 背景の縁を残すので「背景クラスタ」も生まれ、領域分割としても自然になる。
    boxes = [(18 + drift, 28, 112 + drift, 152), (128 + drift, 28, 222 + drift, 152)]
    for name, (x1, y1, x2, y2) in zip(names, boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_COLORS_BGR[name], -1)
    return img


def write_synth_frames(frames_dir: Path, n_frames: int, video_name: str = "synthA") -> list[Path]:
    """合成フレームを連番 JPEG として書き出し、パス一覧を返す（Split の出力相当）。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_frames):
        frame = synth_frame(i, n_frames)
        # 参考実装の命名: {video}_frame{n:010d}.jpg を模す（桁は短縮）。
        p = frames_dir / f"{video_name}_frame{i:06d}.jpg"
        cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        paths.append(p)
    return paths


def read_frame_paths(frames_dir: Path) -> list[Path]:
    """フレームディレクトリの JPEG を番号順に返す。"""
    return sorted(frames_dir.glob("*.jpg"))


# ---------------------------------------------------------------------------
# CLIP モデルのロードとカスタム前処理（build/models.py::load_clip_model 相当）
# ---------------------------------------------------------------------------
def load_clip(device: torch.device | None = None):
    """open_clip の ViT-B-32 を force_quick_gelu でロードし、(model, preprocess, tokenizer) を返す。

    前処理は CenterCrop を排し、アスペクト比を無視して正方形に強制 Resize する。
    これは「画像の端を切らない＝小物体や周辺領域の dense 特徴を捨てない」ための重要な選択。
    """
    device = device or pick_device()
    model, _, base_preprocess = open_clip.create_model_and_transforms(
        model_name=CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
        device=device,
        force_quick_gelu=True,  # ★ openai 重みは quick-gelu。指定しないと活性化がズレる
    )
    model.eval()

    # 元 preprocess から Normalize（CLIP 専用統計）だけ抜き出す。無ければ既定値。
    norm_layer = next(
        (t for t in base_preprocess.transforms if isinstance(t, Normalize)),
        Normalize(CLIP_MEAN, CLIP_STD),
    )
    preprocess = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor(), norm_layer])

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# dense CLIP 特徴（build/models.py::dense_clip_embeddings_vit 相当）
# ---------------------------------------------------------------------------
@torch.inference_mode()
def dense_clip_embeddings(model, images: torch.Tensor) -> torch.Tensor:
    """ViT の visual encoder を手で展開し、パッチ単位の dense 特徴 [B, C, H, W] を返す。

    通常の encode_image は CLS トークンだけを使う（= 画像全体 1 ベクトル）。
    ここでは transformer 出力のパッチトークン列（CLS を除く）を空間グリッドに並べ替えて、
    「領域ごとの埋め込み」を得る。これが Cluster-CLIP の心臓部。
    """
    visual = model.visual
    x = visual.conv1(images)                  # [B, width, gh, gw]  パッチ埋め込み
    B, width, gh, gw = x.shape
    x = x.reshape(B, width, gh * gw).permute(0, 2, 1).contiguous()  # [B, gh*gw, width]

    # CLS トークンを先頭に連結し、位置埋め込みを足す
    cls = visual.class_embedding.to(dtype=x.dtype, device=x.device)
    cls = cls.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    x = visual.ln_pre(x)

    # transformer は (seq, batch, dim) 並びを期待する
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)

    # 共有埋め込み空間へ射影（proj は Parameter）。text 埋め込みと同じ空間に揃える。
    proj = getattr(visual, "proj", None)
    if proj is not None:
        x = x @ proj if isinstance(proj, torch.Tensor) else proj(x)

    x = x[:, 1:, :]                            # ★ CLS を捨ててパッチトークンだけ残す
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # パッチごとに L2 正規化

    # [B, gh*gw, C] → [B, C, gh, gw]
    return x.permute(0, 2, 1).reshape(B, -1, gh, gw).contiguous()


# ---------------------------------------------------------------------------
# 空間連結クラスタリング（build/models.py::cluster_agglomerative 相当）
# ---------------------------------------------------------------------------
def cluster_agglomerative(
    feature_map: torch.Tensor | np.ndarray,
    n_clusters: int,
    connectivity: bool = True,
    linkage: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """dense 特徴 [C, H, W] を「空間的に隣り合うパッチのみ連結」した凝集型クラスタリングで領域分割する。

    Returns:
      reps: [k, C] float32, L2 正規化済みのクラスタ代表ベクトル（各クラスタ平均）
      cmap: [H, W] int32, 各パッチがどのクラスタに属するかのラベルマップ

    connectivity=True のとき grid_to_graph で「隣接パッチ間のみ」併合を許す。
    これにより、意味も位置も近いパッチがまとまり、画像が連続した領域に分かれる。
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()
    C, H, W = feature_map.shape
    # [C, H, W] → [H*W, C]（各パッチが 1 サンプル）
    Y = np.ascontiguousarray(
        feature_map.transpose(1, 2, 0).reshape(H * W, C), dtype=np.float32
    )

    conn = grid_to_graph(n_x=H, n_y=W) if connectivity else None
    n_clusters = min(n_clusters, H * W)
    ag = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",  # 正規化済みベクトルなのでユークリッド ≒ コサイン
        connectivity=conn,
        compute_distances=False,
    )
    labels = ag.fit_predict(Y)

    # 各クラスタの平均 → L2 正規化で代表ベクトルにする
    global_mean = Y.mean(axis=0)
    reps = np.empty((n_clusters, C), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        v = Y[mask].mean(axis=0) if mask.any() else global_mean
        reps[c] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    cmap = labels.reshape(H, W).astype(np.int32)
    return reps, cmap


# ---------------------------------------------------------------------------
# テキスト / 画像エンコーダ
# ---------------------------------------------------------------------------
@torch.inference_mode()
def encode_text(model, tokenizer, queries: list[str], device: torch.device | None = None) -> np.ndarray:
    """テキストクエリを CLIP text encoder でベクトル化し、L2 正規化して [n, C] float32 を返す。"""
    device = device or pick_device()
    tokens = tokenizer(queries).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return np.ascontiguousarray(feats.cpu().numpy(), dtype=np.float32)


@torch.inference_mode()
def encode_image_global(model, preprocess, pil_images, device: torch.device | None = None) -> np.ndarray:
    """画像全体の CLIP 埋め込み（CLS ベース）を [n, C] float32 で返す。領域検索と対比するためのベースライン。"""
    device = device or pick_device()
    batch = torch.stack([preprocess(im.convert("RGB")) for im in pil_images]).to(device)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return np.ascontiguousarray(feats.cpu().numpy(), dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化（float32, C 連続を保証）。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-12)


# ---------------------------------------------------------------------------
# FAISS / SQLite（build/indexer.py + db_writer.py 相当）
# ---------------------------------------------------------------------------
def build_faiss_idmap(vectors: np.ndarray, ids: np.ndarray) -> faiss.Index:
    """L2 正規化済みベクトルを IndexFlatIP(+IDMap) に登録して返す（内積＝コサイン類似）。"""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    dim = vectors.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, ids)
    return index


def create_tables(conn: sqlite3.Connection) -> None:
    """Frames（フレームのメタデータ）と VectorMapping（faiss_id ↔ フレーム/クラスタ）を作る。

    参考実装は Segments（適応サンプリングの単位）も持つが、ここでは簡略化して 2 表にする。
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Frames (
            frame_id    TEXT PRIMARY KEY,
            video_name  TEXT NOT NULL,
            image_path  TEXT NOT NULL,
            cmap_path   TEXT NOT NULL,
            vector_path TEXT NOT NULL,
            k_generated INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS VectorMapping (
            faiss_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id    TEXT NOT NULL,
            cluster_idx INTEGER NOT NULL,
            FOREIGN KEY (frame_id) REFERENCES Frames(frame_id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Split ステージ
# ---------------------------------------------------------------------------
def run_split(frames_dir: Path, n_frames: int, video_name: str = "synthA") -> list[Path]:
    """合成動画を連番フレームに「分割」する（参考実装 split/processor.py の縮小版）。

    実リポは VideoCapture で mp4 を読みフレームを JPEG 保存する。ここでは入力動画も合成する。
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = read_frame_paths(frames_dir)
    if len(existing) >= n_frames:
        return existing[:n_frames]
    return write_synth_frames(frames_dir, n_frames, video_name=video_name)


# ---------------------------------------------------------------------------
# Build ステージ（consumer + indexer + db_writer を 1 関数に統合）
# ---------------------------------------------------------------------------
def run_build(
    build_dir: Path,
    frame_paths: list[Path],
    model,
    preprocess,
    *,
    n_clusters: int = 6,
    batch_size: int = 8,
    connectivity: bool = True,
    device: torch.device | None = None,
    video_name: str = "synthA",
) -> dict[str, Any]:
    """フレーム群を dense CLIP → 領域クラスタ → 代表ベクトル化し、FAISS + SQLite を構築する。

    生成物（参考実装の Build カプセルに対応）:
      build_dir/cluster_maps/*.npy   各フレームのクラスタラベルマップ [H, W]
      build_dir/vectors/*.npy        各フレームのクラスタ代表ベクトル [k, C]
      build_dir/index.faiss          全代表ベクトルの IndexFlatIP(+IDMap)
      build_dir/metadata.db          Frames / VectorMapping
    """
    import shutil

    from PIL import Image

    device = device or pick_device()
    build_dir.mkdir(parents=True, exist_ok=True)
    cmap_dir = build_dir / "cluster_maps"
    vec_dir = build_dir / "vectors"
    # 冪等性: 古い npy を消してから作り直す
    for d in (cmap_dir, vec_dir):
        if d.exists():
            shutil.rmtree(d)
    cmap_dir.mkdir(exist_ok=True)
    vec_dir.mkdir(exist_ok=True)

    db_path = build_dir / DB_NAME
    if db_path.exists():
        db_path.unlink()  # 冪等性: 作り直す
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cur = conn.cursor()

    all_vectors: list[np.ndarray] = []
    all_ids: list[int] = []
    total_clusters = 0

    # バッチ単位で dense 特徴を計算し、フレームごとにクラスタリングして保存する
    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start : start + batch_size]
        pil_batch = [Image.open(p).convert("RGB") for p in chunk]
        tensor = torch.stack([preprocess(im) for im in pil_batch]).to(device)
        feats = dense_clip_embeddings(model, tensor).cpu()  # [b, C, H, W]

        for i, fpath in enumerate(chunk):
            reps, cmap = cluster_agglomerative(
                feats[i], n_clusters=n_clusters, connectivity=connectivity
            )
            frame_id = f"{video_name}_frame{start + i:06d}"

            cmap_path = cmap_dir / f"{frame_id}_cmap.npy"
            vec_path = vec_dir / f"{frame_id}_vecs.npy"
            np.save(cmap_path, cmap)
            np.save(vec_path, reps)

            cur.execute(
                "INSERT INTO Frames (frame_id, video_name, image_path, cmap_path, vector_path, k_generated)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (frame_id, video_name, str(fpath), str(cmap_path), str(vec_path), reps.shape[0]),
            )
            # クラスタ 1 つずつに faiss_id を採番し、同じ順序でベクトルを積む（ID とベクトルの整合）
            for c in range(reps.shape[0]):
                cur.execute(
                    "INSERT INTO VectorMapping (frame_id, cluster_idx) VALUES (?, ?)",
                    (frame_id, c),
                )
                all_ids.append(int(cur.lastrowid))
            all_vectors.append(reps)
            total_clusters += reps.shape[0]

    conn.commit()

    # 全代表ベクトルを 1 つの FAISS インデックスに登録して書き出す
    vectors = np.vstack(all_vectors)
    ids = np.asarray(all_ids, dtype=np.int64)
    index = build_faiss_idmap(vectors, ids)
    faiss.write_index(index, str(build_dir / INDEX_NAME))
    conn.close()

    return {
        "build_dir": build_dir,
        "n_frames": len(frame_paths),
        "n_vectors": int(index.ntotal),
        "dim": int(vectors.shape[1]),
        "total_clusters": total_clusters,
    }


# ---------------------------------------------------------------------------
# Search ステージ（search/engine.py 相当）
# ---------------------------------------------------------------------------
def search_index(build_dir: Path, query_vec: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    """クエリベクトル [1, C] で FAISS を検索し、SQLite からメタデータを引いて結果を返す。

    返す各 dict: faiss_id / score / frame_id / video_name / image_path / cmap_path / cluster_idx
    """
    query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]

    index = faiss.read_index(str(build_dir / INDEX_NAME))
    distances, indices = index.search(query_vec, top_k)

    # faiss_id → スコア。-1（該当なし）は捨てる。
    score_map: dict[int, float] = {}
    for rank in range(indices.shape[1]):
        fid = int(indices[0, rank])
        if fid == -1:
            continue
        score_map[fid] = float(distances[0, rank])
    if not score_map:
        return []

    conn = sqlite3.connect(build_dir / DB_NAME)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(score_map))
    cur.execute(
        f"""
        SELECT v.faiss_id, v.cluster_idx, f.frame_id, f.video_name, f.image_path, f.cmap_path
        FROM VectorMapping v JOIN Frames f ON v.frame_id = f.frame_id
        WHERE v.faiss_id IN ({placeholders})
        """,
        list(score_map.keys()),
    )
    results = []
    for fid, cluster_idx, frame_id, video_name, image_path, cmap_path in cur.fetchall():
        results.append(
            {
                "faiss_id": fid,
                "score": score_map[fid],
                "cluster_idx": cluster_idx,
                "frame_id": frame_id,
                "video_name": video_name,
                "image_path": image_path,
                "cmap_path": cmap_path,
            }
        )
    conn.close()
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def region_vector(build_dir: Path, frame_id: str, cluster_idx: int) -> np.ndarray:
    """Build 済みカプセルから、ある frame の cluster_idx の代表ベクトル [1, C] を取り出す。"""
    conn = sqlite3.connect(build_dir / DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT vector_path FROM Frames WHERE frame_id = ?", (frame_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"frame_id が見つかりません: {frame_id}")
    reps = np.load(row[0]).astype(np.float32)
    return np.ascontiguousarray(reps[cluster_idx][None, :], dtype=np.float32)


def pick_region_by_color(frame_bgr: np.ndarray, cmap: np.ndarray, target_bgr: tuple[int, int, int]) -> int:
    """cmap の各クラスタについて元画像での平均色を求め、target_bgr に最も近いクラスタ index を返す。

    「フレーム 0 の赤い領域」のような“画像クエリ”を作るのに使う（領域→領域検索のデモ用）。
    """
    h, w = frame_bgr.shape[:2]
    cmap_u = cv2.resize(cmap.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    target = np.asarray(target_bgr, dtype=np.float32)
    best_c, best_d = 0, np.inf
    for c in np.unique(cmap_u):
        mask = cmap_u == c
        mean_bgr = frame_bgr[mask].reshape(-1, 3).mean(axis=0)
        d = float(np.linalg.norm(mean_bgr - target))
        if d < best_d:
            best_d, best_c = d, int(c)
    return best_c


# ---------------------------------------------------------------------------
# 可視化（search/visualizer.py::_create_masked_image 相当）
# ---------------------------------------------------------------------------
def overlay_cluster_mask(
    frame_bgr: np.ndarray,
    cmap: np.ndarray,
    cluster_idx: int,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """元フレームに、該当クラスタ領域を半透明色で重畳した BGR 画像を返す。"""
    h, w = frame_bgr.shape[:2]
    # クラスタラベルを最近傍補間で元解像度へ拡大（ラベルを壊さない）
    cmap_u = cv2.resize(cmap.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cmap_u == cluster_idx

    color_layer = np.zeros_like(frame_bgr)
    color_layer[mask] = color
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_layer, alpha, 0)
    out = np.where(mask[..., None], blended, frame_bgr)
    return out.astype(np.uint8)


def colorize_cmap(cmap: np.ndarray, h: int, w: int) -> np.ndarray:
    """クラスタラベルマップを色分け BGR 画像（h x w）にする（領域分割の可視化用）。"""
    cmap_u = cv2.resize(cmap.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = int(cmap.max()) + 1
    palette = (np.asarray(plt.cm.tab20(np.linspace(0, 1, max(k, 1)))[:, :3]) * 255).astype(np.uint8)
    out = palette[np.clip(cmap_u, 0, k - 1)]  # RGB
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def save_search_gallery(query_label: str, results: list[dict[str, Any]], out_name: str) -> Path:
    """検索結果（上位 k 件）を、ヒット領域のマスクを重畳したギャラリー図として保存する。"""
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        frame = cv2.imread(res["image_path"])
        cmap = np.load(res["cmap_path"])
        masked = overlay_cluster_mask(frame, cmap, res["cluster_idx"])
        ax.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        ax.set_title(f"score {res['score']:.3f}\n{res['frame_id'][-9:]} cl{res['cluster_idx']}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(f"Region search: {query_label}", fontsize=11)
    return save_fig(fig, out_name)


# ---------------------------------------------------------------------------
# Stream: サンプラー（stream/capture.py::HistogramDeltaSampler 相当）
# ---------------------------------------------------------------------------
class HistogramDeltaSampler:
    """ヒストグラム交差の差分が閾値を超えたフレームだけ通すサンプラー（適応サンプリングの最小版）。"""

    def __init__(self, threshold: float = 0.25) -> None:
        self._threshold = threshold
        self._prev_hist: np.ndarray | None = None

    def should_keep(self, frame_idx: int, frame_bgr: np.ndarray) -> bool:
        hist = self._compute_hist(frame_bgr)
        if self._prev_hist is None:
            self._prev_hist = hist
            return True
        inter = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_INTERSECT))
        delta = max(0.0, 1.0 - inter)
        if delta >= self._threshold:
            self._prev_hist = hist
            return True
        return False

    @staticmethod
    def _compute_hist(frame_bgr: np.ndarray) -> np.ndarray:
        """3 チャネルのヒストグラムを連結し、全体の和が 1 になるよう L1 正規化する。

        和=1 の確率分布にしておくと、HISTCMP_INTERSECT が [0, 1] に収まり
        delta = 1 - intersection を素直に「変化量」として使える。
        （各チャネルを別々に L2 正規化すると intersection が 1 を超え、差分が壊れる。）
        """
        hists = [cv2.calcHist([frame_bgr], [ch], None, [64], [0, 256]) for ch in range(3)]
        hist = np.vstack(hists).astype(np.float32)
        hist /= float(hist.sum()) + 1e-8
        return hist


# ---------------------------------------------------------------------------
# Stream: 3 プロセス（capture / consumer / writer）と オーケストレータ
# ---------------------------------------------------------------------------
def _capture_proc(config: dict, task_queue, stats_path: Path, ready_event) -> None:
    """[CPU] 合成フレームを生成し task_queue に投入する。キュー満杯なら即ドロップ（put_nowait）。"""
    import json
    import queue as _queue
    import time

    # Consumer のモデルロード（数秒）が終わってから配信を始める＝起動待ちをドロップに数えない。
    ready_event.wait(timeout=120.0)

    n_frames = config["n_frames"]
    video_name = config.get("video_name", "stream")
    sampler_type = config.get("sampler", "passthrough")
    base_k = config.get("n_clusters", 6)
    fps_cap = config.get("fps_cap", 12.0)
    min_interval = 1.0 / fps_cap if fps_cap > 0 else 0.0

    sampler: Any
    if sampler_type == "histogram_delta":
        sampler = HistogramDeltaSampler(threshold=config.get("threshold", 0.05))
    else:
        sampler = None  # passthrough

    from PIL import Image

    kept = dropped = sampled_out = 0
    for i in range(n_frames):
        t_loop = time.time()
        frame_bgr = synth_frame(i, n_frames)
        # 適応サンプリング: 前フレームと似すぎていれば送らない（無駄な推論を省く）
        if sampler is not None and not sampler.should_keep(i, frame_bgr):
            sampled_out += 1
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            task = {
                "video_name": video_name,
                "frame_id": f"{video_name}_frame{i:06d}",
                "image": Image.fromarray(rgb),
                "target_k": base_k,
            }
            try:
                task_queue.put_nowait(task)  # リアルタイム優先: 満杯なら捨てる
                kept += 1
            except _queue.Full:
                dropped += 1
        # fps_cap に合わせて待つ（実時間ストリームを模す）
        elapsed = time.time() - t_loop
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    task_queue.put(POISON_PILL)  # Consumer に終了を通知
    stats_path.write_text(
        json.dumps({"total": n_frames, "kept": kept, "dropped": dropped, "sampled_out": sampled_out})
    )


def _consumer_proc(build_dir: Path, config: dict, task_queue, result_queue, ready_event) -> None:
    """[CPU バウンド] CLIP 推論 + クラスタリングを担当し、結果メタを result_queue に流す。"""
    import queue as _queue

    torch.set_num_threads(1)  # fork/spawn 下での OpenMP 競合を避ける
    device = torch.device("cpu")
    model, preprocess, _ = load_clip(device)
    ready_event.set()  # モデルロード完了を Capture に知らせる（起動待ちを計測から除く）
    batch_size = config.get("batch_size", 4)
    n_clusters = config.get("n_clusters", 6)
    cmap_dir = build_dir / "cluster_maps"
    vec_dir = build_dir / "vectors"
    cmap_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    def flush(batch: list[dict]) -> None:
        if not batch:
            return
        tensor = torch.stack([preprocess(it["image"]) for it in batch]).to(device)
        feats = dense_clip_embeddings(model, tensor).cpu()
        for i, it in enumerate(batch):
            reps, cmap = cluster_agglomerative(feats[i], n_clusters=n_clusters)
            fid = it["frame_id"]
            cmap_path = cmap_dir / f"{fid}_cmap.npy"
            vec_path = vec_dir / f"{fid}_vecs.npy"
            np.save(cmap_path, cmap)
            np.save(vec_path, reps)
            result_queue.put(
                {
                    "frame_id": fid,
                    "video_name": it["video_name"],
                    "image_path": "",  # ストリームはメモリ上画像なのでファイルなし
                    "cmap_path": str(cmap_path),
                    "vector_path": str(vec_path),
                    "k_generated": int(reps.shape[0]),
                }
            )

    batch: list[dict] = []
    while True:
        try:
            item = task_queue.get(timeout=2.0)
        except _queue.Empty:
            flush(batch)
            batch = []
            continue
        if item == POISON_PILL:
            flush(batch)
            return
        batch.append(item)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []


def _writer_proc(build_dir: Path, result_queue) -> None:
    """[CPU] 受信したメタを SQLite に書き、FAISS をインクリメンタルに構築して常時検索可能にする。"""
    import queue as _queue

    conn = sqlite3.connect(build_dir / DB_NAME)
    create_tables(conn)
    cur = conn.cursor()
    index: faiss.Index | None = None
    total = 0

    while True:
        try:
            item = result_queue.get(timeout=5.0)
        except _queue.Empty:
            continue
        if item == POISON_PILL:
            break
        cur.execute(
            "INSERT OR REPLACE INTO Frames (frame_id, video_name, image_path, cmap_path, vector_path, k_generated)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                item["frame_id"], item["video_name"], item["image_path"],
                item["cmap_path"], item["vector_path"], item["k_generated"],
            ),
        )
        reps = np.load(item["vector_path"]).astype(np.float32)
        if index is None:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(reps.shape[1]))
        ids = []
        for c in range(reps.shape[0]):
            cur.execute(
                "INSERT INTO VectorMapping (frame_id, cluster_idx) VALUES (?, ?)",
                (item["frame_id"], c),
            )
            ids.append(int(cur.lastrowid))
        index.add_with_ids(np.ascontiguousarray(reps), np.asarray(ids, dtype=np.int64))
        conn.commit()
        total += 1

    if index is not None and index.ntotal > 0:
        faiss.write_index(index, str(build_dir / INDEX_NAME))
    conn.close()


def run_stream_pipeline(build_dir: Path, n_frames: int = 16, **cfg) -> dict[str, Any]:
    """capture → task_queue → consumer → result_queue → writer の 3 プロセスを起動して回す。

    「取得（CPU 軽い）」と「推論（CPU 重い）」を別プロセスに分離し、キュー満杯時はフレームを
    ドロップしてリアルタイム性を保つ、というストリーム設計の最小実装。
    """
    import json
    import multiprocessing as mp

    import shutil

    build_dir.mkdir(parents=True, exist_ok=True)
    # 冪等性: 既存の DB/index/npy を消してから作り直す（古い npy が処理数の集計を汚さないように）
    for f in (build_dir / DB_NAME, build_dir / INDEX_NAME):
        if f.exists():
            f.unlink()
    for d in (build_dir / "vectors", build_dir / "cluster_maps"):
        if d.exists():
            shutil.rmtree(d)

    config = {
        "n_frames": n_frames,
        "n_clusters": cfg.get("n_clusters", 6),
        "batch_size": cfg.get("batch_size", 4),
        "video_name": cfg.get("video_name", "stream"),
        "sampler": cfg.get("sampler", "passthrough"),
        "threshold": cfg.get("threshold", 0.05),
        "fps_cap": cfg.get("fps_cap", 12.0),
    }
    queue_size = cfg.get("queue_size", 8)
    stats_path = build_dir / "capture_stats.json"

    ctx = mp.get_context("spawn")  # torch を載せるので spawn で各子プロセスを独立起動
    task_queue = ctx.Queue(maxsize=queue_size)
    result_queue = ctx.Queue()
    ready_event = ctx.Event()  # Consumer のモデルロード完了通知

    writer = ctx.Process(target=_writer_proc, args=(build_dir, result_queue), name="writer")
    consumer = ctx.Process(
        target=_consumer_proc, args=(build_dir, config, task_queue, result_queue, ready_event), name="consumer"
    )
    capture = ctx.Process(target=_capture_proc, args=(config, task_queue, stats_path, ready_event), name="capture")

    import time

    t_start = time.time()
    writer.start()
    consumer.start()
    capture.start()

    ready_event.wait(timeout=120.0)  # Consumer のモデルロード完了を待つ
    t_stream0 = time.time()          # ここから「実時間ストリーム」の計測を始める

    capture.join()       # 全フレーム投入 + POISON 送出で終了
    consumer.join()      # POISON 受領 → 残バッチを flush して終了
    result_queue.put(POISON_PILL)  # Writer に終了を通知
    writer.join()
    elapsed = time.time() - t_stream0
    warmup = t_stream0 - t_start

    default_stats = {"total": n_frames, "kept": 0, "dropped": 0, "sampled_out": 0}
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else default_stats
    processed = sum(1 for _ in (build_dir / "vectors").glob("*.npy")) if (build_dir / "vectors").exists() else 0
    return {
        "build_dir": build_dir,
        "warmup_sec": warmup,
        "elapsed_sec": elapsed,
        "effective_fps": processed / elapsed if elapsed > 0 else 0.0,
        "total": stats["total"],
        "sampled_out": stats.get("sampled_out", 0),
        "captured": stats["kept"],
        "dropped": stats["dropped"],
        "processed": processed,
    }
