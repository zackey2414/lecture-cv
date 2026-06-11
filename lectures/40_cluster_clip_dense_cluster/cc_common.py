"""40_cluster_clip_dense_cluster 共通の道具箱（Cluster-CLIP ミニ実装）。

このモジュールは 01〜05 のスクリプト・mini_project.py・exercises 系が import する
再利用部品をまとめる。設計の元ネタは参照リポ cluster-clip の
  - build/models.py … dense CLIP 特徴抽出 + 空間連結クラスタリング
  - build/indexer.py … FAISS IndexIDMap(IndexFlatIP) 構築
  - build/db_writer.py … SQLite メタデータ
  - search/engine.py … テキスト→FAISS→SQLite の領域検索
を CPU だけで動く小型版に落としたもの。

方針:
  - 入力画像は cv2 が扱う **BGR uint8 (H,W,C)**。CLIP に渡す直前で RGB へ変換する。
  - dense 特徴は L2 正規化済みの [C, H, W]（パッチを空間に並べたマップ）。
  - FAISS に渡す配列は必ず float32・C連続（np.ascontiguousarray）。
  - 重み DL ができない環境でも教材が動くよう、CLIP のロードに失敗したら
    決定論的なフォールバック特徴抽出器に切り替える（本番では常に CLIP を使う）。

実行はリポジトリルートから:
  uv run python lectures/40_cluster_clip_dense_cluster/01_dense_vs_global_clip.py
"""

from __future__ import annotations

import hashlib
import pathlib
from collections import deque

import cv2
import numpy as np

MODULE_ID = "40_cluster_clip_dense_cluster"

# CLIP の前処理統計（ImageNet とは別物。ここを間違えると静かに精度が落ちる）。
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# =====================================================================
# 出力先 / デバイス
# =====================================================================
def output_dir() -> pathlib.Path:
    """outputs/40_cluster_clip_dense_cluster/ を作って返す（無ければ作成）。"""
    # lectures/40_.../cc_common.py から見てリポジトリルートは 2 つ上。
    root = pathlib.Path(__file__).resolve().parents[2]
    out = root / "outputs" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


def pick_device() -> str:
    """cuda が使えれば cuda、無ければ cpu。本講座は CPU 前提だが定石として残す。"""
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 数値ユーティリティ
# =====================================================================
def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """指定軸で L2 正規化（ノルム 0 でも落ちないよう eps を足す）。"""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def as_faiss(x: np.ndarray) -> np.ndarray:
    """FAISS に渡せる形（float32・C連続）に整える。これを怠ると落ちる/誤動作する。"""
    return np.ascontiguousarray(x, dtype=np.float32)


# =====================================================================
# 合成シーン（小物体・複数物体を含む BGR 画像）
# =====================================================================
def make_scene(seed: int = 0, size: int = 224) -> tuple[np.ndarray, list[dict]]:
    """複数の色付き物体を置いた合成シーンを BGR uint8 で返す。

    「画像全体 1 ベクトル」では小物体が埋もれる、を実感させるための題材。
    戻り値: (img_bgr, regions)。regions は各物体の
            {"label": テキスト, "bbox": (x1,y1,x2,y2), "color_bgr": (b,g,r)}。
    """
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), dtype=np.uint8)

    # --- 背景: 上半分=空(水色)、下半分=地面(緑) のグラデーション ---
    for y in range(size):
        if y < size // 2:
            img[y, :] = (230 - y // 3, 170 - y // 6, 90)  # 空(BGR)
        else:
            img[y, :] = (60, 140 + (y - size // 2) // 4, 70)  # 地面(BGR)

    regions: list[dict] = []

    def _put(x1, y1, x2, y2, color, label, shape="rect"):
        if shape == "rect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
        else:  # circle
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), (x2 - x1) // 2, color, -1)
        regions.append({"label": label, "bbox": (x1, y1, x2, y2), "color_bgr": color})

    s = size / 224.0  # 解像度スケール
    # 大きめの赤い車（横長の矩形 + 黒い窓帯）
    _put(int(20 * s), int(140 * s), int(95 * s), int(185 * s), (40, 40, 200), "a red car")
    cv2.rectangle(img, (int(30 * s), int(148 * s)), (int(85 * s), int(160 * s)), (60, 30, 30), -1)
    # 青い四角い看板（中サイズ）
    _put(int(120 * s), int(40 * s), int(165 * s), int(85 * s), (200, 120, 40), "a blue sign")
    # 小さい黄色いボール（小物体: dense 検索の主役）
    _put(int(180 * s), int(150 * s), int(205 * s), int(175 * s), (40, 210, 230), "a yellow ball", "circle")
    # 緑の木（細長い矩形）
    _put(int(150 * s), int(95 * s), int(175 * s), int(150 * s), (40, 110, 30), "a green tree")

    # わずかなノイズで平坦さを崩す（CLIP の特徴を少し安定させる）
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, regions


def make_frame_sequence(
    n: int = 8, seed: int = 0, size: int = 224
) -> list[tuple[np.ndarray, list[dict]]]:
    """小物体（黄色いボール）が左→右へ動く連番フレーム列を返す（Split/Stream 用）。"""
    frames = []
    for i in range(n):
        img, regions = make_scene(seed=seed + i, size=size)
        # ボールを水平移動させて「動画らしさ」を出す
        cx = int((20 + 170 * i / max(n - 1, 1)) * size / 224.0)
        cy = int(165 * size / 224.0)
        cv2.circle(img, (cx, cy), int(13 * size / 224.0), (40, 210, 230), -1)
        regions = [r for r in regions if r["label"] != "a yellow ball"]
        regions.append(
            {
                "label": "a yellow ball",
                "bbox": (cx - 13, cy - 13, cx + 13, cy + 13),
                "color_bgr": (40, 210, 230),
            }
        )
        frames.append((img, regions))
    return frames


def make_block_feature_map(
    grid: int = 7, dim: int = 16, n_blocks: int = 2, seed: int = 0
) -> np.ndarray:
    """テスト用の合成 dense 特徴マップ [dim, grid, grid] を作る。

    grid を縦方向に n_blocks 個の帯に割り、帯ごとに別の基底ベクトル + ノイズ。
    クラスタリングや FAISS/SQLite の機構を、CLIP を呼ばずに高速・決定論的に試すための部品。
    """
    rng = np.random.default_rng(seed)
    bases = l2n(rng.standard_normal((n_blocks, dim)).astype("float32"), axis=1)
    fmap = np.zeros((dim, grid, grid), dtype="float32")
    for r in range(grid):
        b = min(r * n_blocks // grid, n_blocks - 1)
        for c in range(grid):
            v = bases[b] + 0.05 * rng.standard_normal(dim).astype("float32")
            fmap[:, r, c] = v / (np.linalg.norm(v) + 1e-8)
    return fmap


# =====================================================================
# 空間連結クラスタリング（Cluster-CLIP の中核）
# =====================================================================
def grid_connectivity(h: int, w: int):
    """h×w グリッドの 4 近傍隣接グラフ（scipy 疎行列）を返す。

    sklearn の grid_to_graph は「隣り合うパッチ同士だけ」をエッジで結ぶ。
    AgglomerativeClustering にこれを渡すと、空間的に離れたパッチは決して
    同じクラスタに併合されない＝必ず連結した領域になる。
    """
    from sklearn.feature_extraction.image import grid_to_graph

    return grid_to_graph(n_x=h, n_y=w)


def cluster_regions(
    feat_map: np.ndarray,
    n_clusters: int,
    connectivity: bool = True,
    linkage: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """dense 特徴マップ [C,H,W] を「意味の似た連結領域」に分割する。

    手順（参照リポ build/models.py: cluster_agglomerative と同じ勘所）:
      1. [C,H,W] を [H*W, C] に並べ替える（各行=1パッチの特徴）。
      2. 空間連結（grid_to_graph）付き AgglomerativeClustering で領域分割。
      3. 各クラスタの平均→L2 正規化を「クラスタ代表ベクトル」にする。

    戻り値: (reps[k,C] float32, label_map[H,W] int)。
    """
    from sklearn.cluster import AgglomerativeClustering

    c, h, w = feat_map.shape
    k = min(n_clusters, h * w)
    # [C,H,W] -> [H*W, C]。permute して連続化してから reshape。
    y = np.ascontiguousarray(feat_map.transpose(1, 2, 0)).reshape(h * w, c).astype("float32")

    conn = grid_connectivity(h, w) if connectivity else None
    ag = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric="euclidean",
        connectivity=conn,
    )
    labels = ag.fit_predict(y)

    global_mean = y.mean(axis=0)
    reps = np.empty((k, c), dtype="float32")
    for cid in range(k):
        mask = labels == cid
        v = y[mask].mean(axis=0) if mask.any() else global_mean
        reps[cid] = v / (np.linalg.norm(v) + 1e-8)
    return reps, labels.reshape(h, w).astype(np.int32)


def count_components(label_map: np.ndarray, cluster_id: int) -> int:
    """label_map 上で cluster_id のパッチが幾つの 4 連結成分に分かれているか数える。

    空間連結クラスタリングが効いていれば、どのクラスタも必ず 1 連結成分になる。
    """
    mask = label_map == cluster_id
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            comps += 1
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    return comps


def total_components(label_map: np.ndarray) -> int:
    """全クラスタの連結成分数の合計。クラスタ数と一致すれば「全領域が連結」。"""
    return sum(count_components(label_map, cid) for cid in np.unique(label_map))


# =====================================================================
# FAISS（領域代表ベクトルのインデックス）
# =====================================================================
def build_ip_idmap(vectors: np.ndarray, ids: np.ndarray):
    """IndexIDMap(IndexFlatIP) を作って add_with_ids する。

    コサイン類似度は「L2 正規化済みベクトル + 内積(IP)」で得る。
    任意の int64 ID（=SQLite の faiss_id）を割り当てられるよう IndexIDMap を使う。
    """
    import faiss

    dim = vectors.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(as_faiss(vectors), np.ascontiguousarray(ids, dtype=np.int64))
    return index


# =====================================================================
# 可視化
# =====================================================================
_PALETTE = np.array(
    [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
        [153, 153, 153], [102, 194, 165], [252, 141, 98], [141, 160, 203],
    ],
    dtype=np.uint8,
)


def label_map_to_color(label_map: np.ndarray) -> np.ndarray:
    """クラスタ ID マップ [H,W] を色分け RGB 画像にする。"""
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        out[label_map == cid] = _PALETTE[int(cid) % len(_PALETTE)]
    return out


def upscale_nearest(arr2d: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """2D ラベル/マスクを最近傍補間で画像サイズへ拡大（境界をぼかさない）。"""
    h, w = size_hw
    return cv2.resize(arr2d.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_region(
    img_rgb: np.ndarray, mask_bool: np.ndarray, color=(255, 0, 0), alpha: float = 0.55
) -> np.ndarray:
    """マスク領域に半透明の色を重ねた RGB 画像を返す（検索ヒット領域の強調）。"""
    out = img_rgb.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask_bool] = color
    blended = cv2.addWeighted(out, 1 - alpha, color_layer, alpha, 0)
    out[mask_bool] = blended[mask_bool]
    return out


def features_to_rgb(feat_map: np.ndarray) -> np.ndarray:
    """dense 特徴 [C,H,W] を PCA 上位3成分→RGB に落として可視化する。

    「隣り合うパッチの特徴が似ている＝似た色」で、特徴マップの空間構造が見える。
    """
    c, h, w = feat_map.shape
    y = feat_map.transpose(1, 2, 0).reshape(h * w, c).astype("float64")
    y = y - y.mean(axis=0, keepdims=True)
    # SVD で上位3主成分（sklearn 不要の軽量版）
    _, _, vt = np.linalg.svd(y, full_matrices=False)
    proj = y @ vt[:3].T  # [H*W, 3]
    proj -= proj.min(axis=0, keepdims=True)
    proj /= proj.max(axis=0, keepdims=True) + 1e-8
    return (proj.reshape(h, w, 3) * 255).astype(np.uint8)


# =====================================================================
# dense CLIP エンコーダ（ViT を手で展開してパッチ特徴を取り出す）
# =====================================================================
def dense_vit_tokens(visual, images):
    """open_clip の ViT visual encoder を手で forward 展開し、パッチ特徴 [B,C,gh,gw] を返す。

    CLIP の通常 API は「画像全体 1 ベクトル（CLS 由来）」しか返さない。
    領域検索のためにはパッチ単位の特徴が要るので、transformer を自前で通して
    最終トークン列を取り出し、CLS を捨てて空間 (gh,gw) に並べ替える。
    （参照リポ build/models.py: dense_clip_embeddings_vit と同じ手順。）
    """
    import torch

    x = visual.conv1(images)  # [B, width, gh, gw] : パッチ埋め込み
    b, width, gh, gw = x.shape
    x = x.reshape(b, width, gh * gw).permute(0, 2, 1).contiguous()  # [B, gh*gw, width]

    cls = visual.class_embedding.to(dtype=x.dtype, device=x.device)
    cls = cls.unsqueeze(0).unsqueeze(0).expand(b, 1, -1)
    x = torch.cat([cls, x], dim=1)  # 先頭に CLS を付与
    x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device).unsqueeze(0)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # [seq, B, width] : transformer は seq-first
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # [B, seq, width]
    x = visual.ln_post(x)

    proj = getattr(visual, "proj", None)
    if proj is not None:
        x = x @ proj  # 共通潜在空間へ射影（テキストと同じ次元へ）

    x = x[:, 1:, :]  # ★ CLS を捨ててパッチトークンだけ残す
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # パッチごとに L2 正規化
    return x.permute(0, 2, 1).reshape(b, -1, gh, gw).contiguous()  # [B, C, gh, gw]


class DenseEncoder:
    """dense CLIP 特徴とテキスト埋め込みを返す統一インターフェース。

    backend="clip"     : open_clip ViT-B-32 (force_quick_gelu, 正方形 Resize 前処理)
    backend="fallback" : 重み DL 不可時の決定論フォールバック（機構確認用）
    """

    def __init__(self, backend, model=None, preprocess=None, tokenizer=None, proj=None,
                 grid=7, feat_dim=512, device="cpu"):
        self.backend = backend
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._proj = proj
        self.grid = grid
        self.feat_dim = feat_dim
        self.device = device

    # ---- 前処理: BGR uint8 -> CLIP 入力テンソル [3,H,W] ----
    def preprocess_bgr(self, img_bgr: np.ndarray):
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self.backend == "clip":
            return self._preprocess(pil)
        # フォールバックも CLIP と同じ「正方形 Resize + 正規化」を踏襲
        import torch

        arr = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0
        arr = (arr - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    # ---- dense 特徴: バッチテンソル [B,3,224,224] -> np [B,C,gh,gw] ----
    def dense(self, images) -> np.ndarray:
        import torch

        with torch.inference_mode():
            if self.backend == "clip":
                feat = dense_vit_tokens(self._model.visual, images)
                return feat.cpu().numpy().astype("float32")
            # フォールバック: パッチを切り出して固定射影 -> L2 正規化
            b = images.shape[0]
            ps = 224 // self.grid
            patches = images[:, :, : ps * self.grid, : ps * self.grid]
            patches = patches.unfold(2, ps, ps).unfold(3, ps, ps)  # [B,3,gh,gw,ps,ps]
            g = self.grid
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, g * g, 3 * ps * ps)
            feat = patches.cpu().numpy().astype("float32") @ self._proj  # [B, g*g, C]
            feat = l2n(feat, axis=-1)
            return feat.transpose(0, 2, 1).reshape(b, self.feat_dim, g, g)

    # ---- 画像全体 1 ベクトル（CLS 由来の従来 CLIP 埋め込み） ----
    def global_embedding(self, images) -> np.ndarray:
        """画像全体を 1 本に潰した埋め込み [B,C]（L2 正規化済み）。

        これが「従来の CLIP 画像検索」が使うベクトル。位置情報を持たない。
        """
        import torch

        with torch.inference_mode():
            if self.backend == "clip":
                g = self._model.encode_image(images)
                g = g / g.norm(dim=-1, keepdim=True)
                return g.cpu().numpy().astype("float32")
            # フォールバック: dense パッチの平均を全体ベクトルの代用にする
            d = self.dense(images)  # [B,C,gh,gw]
            b, c = d.shape[0], d.shape[1]
            return l2n(d.reshape(b, c, -1).mean(axis=2), axis=-1)

    # ---- テキスト埋め込み: list[str] -> np [n,C]（L2 正規化済み） ----
    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        if self.backend == "clip":
            tok = self._tokenizer(texts).to(self.device)
            with torch.inference_mode():
                tf = self._model.encode_text(tok)
                tf = tf / tf.norm(dim=-1, keepdim=True)
            return tf.cpu().numpy().astype("float32")
        # フォールバック: 文字列ハッシュから決定論的な単位ベクトル（意味は持たない）
        out = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
            v = np.random.default_rng(h).standard_normal(self.feat_dim).astype("float32")
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out)


def load_encoder(verbose: bool = True) -> DenseEncoder:
    """DenseEncoder をロードする。CLIP が使えなければフォールバックへ自動切替。"""
    device = pick_device()
    try:
        import open_clip
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device, force_quick_gelu=True
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # ★ CenterCrop を排し、アスペクト比を無視して正方形へ強制 Resize する。
        #   端を切らないことが dense 特徴（領域分割）の品質に直結する。
        img_size = model.visual.image_size
        target = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        norm = next(
            (t for t in preprocess.transforms if isinstance(t, Normalize)),
            Normalize(CLIP_MEAN, CLIP_STD),
        )
        custom = Compose([Resize(target), ToTensor(), norm])

        patch = model.visual.conv1.kernel_size[0]
        grid = target[0] // patch
        feat_dim = model.visual.proj.shape[1] if model.visual.proj is not None else 512
        if verbose:
            print(f"[encoder] open_clip ViT-B-32 backend (grid={grid}x{grid}, dim={feat_dim}, dev={device})")
        return DenseEncoder("clip", model=model, preprocess=custom, tokenizer=tokenizer,
                            grid=grid, feat_dim=feat_dim, device=device)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[encoder] CLIP ロード不可 -> フォールバック特徴抽出器を使用: {type(e).__name__}: {e}")
        rng = np.random.default_rng(0)
        proj = (rng.standard_normal((3 * 32 * 32, 512)) * 0.02).astype("float32")
        return DenseEncoder("fallback", proj=proj, grid=7, feat_dim=512, device="cpu")


if __name__ == "__main__":
    # 簡易セルフテスト（ライブラリ配線の確認）
    enc = load_encoder()
    img, regions = make_scene(0)
    print("scene", img.shape, "regions", [r["label"] for r in regions])
    import torch

    x = torch.stack([enc.preprocess_bgr(img)])
    fmap = enc.dense(x)[0]
    print("dense feat", fmap.shape)
    reps, lab = cluster_regions(fmap, n_clusters=5)
    print("reps", reps.shape, "label_map\n", lab)
    print("total connected components == n_clusters ?", total_components(lab), len(np.unique(lab)))
