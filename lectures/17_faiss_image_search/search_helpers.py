"""17_faiss_image_search 共通の道具箱。

各スクリプト(01〜04)と exercises.py が import して使う再利用部品をまとめる。
方針:
  - このファイルの「トップレベル」では重い依存(torch / transformers)を import しない。
    そうしないと FAISS だけ学ぶ 01/03/04 まで毎回 torch を読み込んでしまい遅い。
    画像エンコーダ(CLIP)が要るのは 02 だけなので、torch/transformers は
    Embedder のメソッド内で「遅延 import」する。
  - 数値配列の約束: FAISS に渡す配列は必ず float32 かつ C連続(row-major)。
    np.ascontiguousarray(x, dtype=np.float32) を徹底する。

実行はプロジェクトルートから:
  uv run python lectures/17_faiss_image_search/01_flat_ip_cosine.py
"""

from __future__ import annotations

import pathlib

import numpy as np

MODULE_ID = "17_faiss_image_search"


# =====================================================================
# 出力先 / デバイス
# =====================================================================


def output_dir() -> pathlib.Path:
    """lectures/17_faiss_image_search/outputs/ を作って返す（無ければ作成）。"""
    # 出力はスクリプト隣の outputs/ に置く(CWD非依存・__file__基準)。
    out = pathlib.Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_dir() -> pathlib.Path:
    """data/17_faiss_image_search/（実画像を置く人向けの入力先）。"""
    root = pathlib.Path(__file__).resolve().parents[2]
    return root / "data" / MODULE_ID


def pick_device() -> str:
    """cuda > mps > cpu の順に使えるデバイス名を返す（全回共通の定石）。

    torch を遅延 import する。torch 未導入でも 'cpu' を返して落とさない。
    """
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================================
# FAISS 用の数値ユーティリティ
# =====================================================================


def as_faiss_array(x: np.ndarray) -> np.ndarray:
    """FAISS が要求する float32・C連続の配列に変換して返す。

    入力が float64 だったり、スライス/転置でメモリが非連続だと search が
    落ちる・誤動作するので、index に渡す直前に必ずこれを通す。
    """
    return np.ascontiguousarray(x, dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """各行を L2 ノルム1に正規化した「コピー」を返す（float32, C連続）。

    コサイン類似度を内積(IndexFlatIP)で計算するための前処理。
    注意: faiss.normalize_L2(x) は in-place で元配列を破壊する。学習用に挙動を
    切り分けたいので、ここでは破壊しない純粋関数版を用意する。
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # 0除算ガード（零ベクトル対策）
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def recall_at_k(gt_ids: np.ndarray, ann_ids: np.ndarray, k: int) -> float:
    """Recall@k を「集合一致」で自前計算する。

    gt_ids : 厳密検索(IndexFlat)で得た正解の近傍ID列 (n_query, k_gt)
    ann_ids: ANN(IVF/HNSW/PQ)が返した近傍ID列          (n_query, k_ann)
    各クエリで「上位k の正解のうち、ANN の上位k に何個入っていたか / k」を平均する。
    -1（近傍が足りないときの埋め草）は intersect1d で自然に無視される。
    """
    recalls = []
    for gt_row, ann_row in zip(gt_ids, ann_ids):
        gt_set = gt_row[:k]
        ann_set = ann_row[:k]
        hit = np.intersect1d(gt_set, ann_set).size
        recalls.append(hit / k)
    return float(np.mean(recalls))


def average_precision_at_k(ranked_labels: np.ndarray, query_label: int, k: int) -> float:
    """1クエリ分の Average Precision@k（ラベル一致を正解とする検索評価）。

    ranked_labels: 類似度降順に並べた候補のラベル列（クエリ自身は除いておく）。
    query_label  : クエリのラベル。これと一致する候補が「正例」。
    AP = (各正例位置での precision の平均)。正例が k 内に1つも無ければ 0。
    """
    hits = 0
    score = 0.0
    for rank, lab in enumerate(ranked_labels[:k], start=1):
        if lab == query_label:
            hits += 1
            score += hits / rank  # この順位までの precision
    if hits == 0:
        return 0.0
    return score / hits


# =====================================================================
# 合成データ生成（FAISS 学習用のベクトル群）
# =====================================================================


def make_clustered_vectors(
    n: int,
    d: int,
    n_clusters: int,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """ガウス混合でクラスタ構造のあるベクトル群を作る（正規化はしない）。

    返り値: (vectors[float32, (n, d)], labels[int64, (n,)])
    クラスタ中心の周りに noise の散らばりで点を撒く。labels はどのクラスタ由来かで、
    「同じラベル=関連あり」とみなす検索評価(Recall/mAP)の正解に使える。
    正規化は呼び出し側で l2_normalize して行う（normalize の意味を明示するため）。
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n)
    vectors = centers[labels] + noise * rng.standard_normal((n, d)).astype(np.float32)
    return as_faiss_array(vectors), labels.astype(np.int64)


# =====================================================================
# 合成画像（画像検索デモ用）
# =====================================================================

# (図形, 色名, RGB) の組。色×形でカテゴリを作る。
_COLORS = [
    ("red", (220, 30, 30)),
    ("green", (30, 170, 60)),
    ("blue", (40, 70, 220)),
    ("yellow", (230, 200, 20)),
]
_SHAPES = ["circle", "square", "triangle"]


def _draw_shape(shape: str, rgb: tuple[int, int, int]):
    """単色の図形を描いた 96x96 の PIL.Image(RGB) を返す。"""
    from PIL import Image, ImageDraw  # 遅延 import（Pillow は main 依存だが対称性のため）

    img = Image.new("RGB", (96, 96), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([16, 16, 80, 80], fill=rgb)
    elif shape == "square":
        draw.rectangle([16, 16, 80, 80], fill=rgb)
    else:  # triangle
        draw.polygon([(48, 14), (16, 82), (80, 82)], fill=rgb)
    return img


def make_shape_images(repeat: int = 2):
    """色×形の合成画像群と、その日本語ラベルを返す。

    返り値: (images[list[PIL.Image]], labels[list[str]])
    repeat 回だけ少しずつ位置/色をずらして複製し、検索対象を水増しする。
    「赤い円」を探したら別の「赤い円」が上位に来る、という意味のある検索になる。
    """
    from PIL import ImageDraw

    images = []
    labels = []
    jp_color = {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄"}
    jp_shape = {"circle": "円", "square": "四角", "triangle": "三角"}
    for r in range(repeat):
        for cname, rgb in _COLORS:
            for shape in _SHAPES:
                # 複製ごとに色を少し揺らし、背景に薄いノイズを足して「別個体」にする。
                jitter = tuple(int(np.clip(c + (r * 12 - 6), 0, 255)) for c in rgb)
                img = _draw_shape(shape, jitter)
                # 位置の揺らぎ（小さな点を描いて個体差を出す）
                d = ImageDraw.Draw(img)
                d.point([(r * 3 + 2, r * 3 + 2)], fill=(0, 0, 0))
                images.append(img)
                labels.append(f"{jp_color[cname]}い{jp_shape[shape]}")
    return images, labels


def images_from_data_or_synthetic(repeat: int = 2):
    """data/17_faiss_image_search/ に画像があればそれを、無ければ合成画像を使う。

    実画像で試したい人向けの導線。data ディレクトリに *.png/*.jpg を置くと、
    そのファイル名(拡張子なし)をラベルとして読み込む。
    返り値: (images[list[PIL.Image]], labels[list[str]], source[str])
    """
    from PIL import Image

    ddir = data_dir()
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in ddir.glob("*") if p.suffix.lower() in exts) if ddir.exists() else []
    if files:
        images = [Image.open(p).convert("RGB") for p in files]
        labels = [p.stem for p in files]
        return images, labels, f"data/{MODULE_ID}/ の実画像 {len(files)} 枚"
    images, labels = make_shape_images(repeat=repeat)
    return images, labels, "合成画像（色×形）"


# =====================================================================
# 画像/テキスト エンコーダ（CLIP、無ければ色記述子にフォールバック）
# =====================================================================


class Embedder:
    """画像とテキストを同じ次元のベクトルに変換するエンコーダ。

    既定では HuggingFace の openai/clip-vit-base-patch32 を使う。画像とテキストを
    共有埋め込み空間に写すので、テキスト→画像 検索（マルチモーダル）ができる。
    モデルを取得できない（オフライン等）場合は「色記述子」へ自動フォールバックし、
    スクリプトが exit 0 で完走できるようにする（テキスト検索は CLIP のときのみ可能）。

    重要(transformers v5): get_image_features / get_text_features の戻り値は
    BaseModelOutputWithPooling オブジェクトに変わった。射影後ベクトルは
    .pooler_output に入っており、しかも L2 正規化されていない。検索の前に
    自分で正規化する必要がある（この未正規化が最頻出の落とし穴）。
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, force_fallback: bool = False) -> None:
        self.backend = "fallback-color"
        self.dim = 16 * 16 * 3  # 色記述子の次元
        self.supports_text = False
        self._model = None
        self._processor = None
        self._device = "cpu"
        if not force_fallback:
            self._try_load_clip()

    def _try_load_clip(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel

            self._device = pick_device()
            self._model = CLIPModel.from_pretrained(self.MODEL_ID).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.dim = int(self._model.config.projection_dim)  # 512
            self.backend = f"CLIP({self.MODEL_ID})"
            self.supports_text = True
            self._torch = torch
        except Exception as e:  # noqa: BLE001
            # ネット不通・依存不足など。色記述子に落として続行する。
            print(
                f"  [Embedder] CLIP を使えませんでした → 色記述子にフォールバック: {type(e).__name__}"
            )

    # --- 画像 ---------------------------------------------------------
    def encode_images(self, images) -> np.ndarray:
        """画像リスト → (n, dim) の「未正規化」float32 ベクトル。"""
        if self._model is not None:
            return self._encode_images_clip(images)
        return self._encode_images_color(images)

    def _encode_images_clip(self, images) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():  # 推論は勾配を切る（CPUでも必須）
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            # v5: 戻り値は BaseModelOutputWithPooling。射影ベクトルは pooler_output。
            out = self._model.get_image_features(**inputs)
            feats = out.pooler_output  # (n, 512)（未正規化）
        return as_faiss_array(feats.cpu().numpy())

    @staticmethod
    def _encode_images_color(images) -> np.ndarray:
        """フォールバック: 16x16 に縮小した RGB を平坦化した色記述子。

        単色図形なら『赤い円どうし』が近くなる、ナイーブだが意味のある特徴。
        """
        from PIL import Image

        vecs = []
        for im in images:
            small = im.convert("RGB").resize((16, 16), Image.BILINEAR)
            vecs.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
        return as_faiss_array(np.stack(vecs))

    # --- テキスト -----------------------------------------------------
    def encode_texts(self, texts) -> np.ndarray | None:
        """テキストリスト → (n, dim) の未正規化ベクトル。CLIP 以外では None。"""
        if not self.supports_text:
            return None
        torch = self._torch
        with torch.inference_mode():
            # 複数文をまとめるので padding=True（長さ不一致エラーの定番回避）。
            inputs = self._processor(text=list(texts), return_tensors="pt", padding=True).to(
                self._device
            )
            out = self._model.get_text_features(**inputs)
            feats = out.pooler_output
        return as_faiss_array(feats.cpu().numpy())
