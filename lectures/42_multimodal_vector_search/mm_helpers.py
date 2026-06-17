"""第42回 共通の道具箱（マルチモーダル・ベクトル検索 / CLIP + FAISS）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務をはっきり4つに絞ってあります（単一責務）:

  1. device 判定 / 出力ディレクトリ / matplotlib(Agg) の共通設定
  2. 合成コレクション（色 × 形）の画像とテキストの生成（ネット不要・決定的）
  3. CLIP のロードと「画像・テキストを同じ512次元空間へ」写す埋め込み関数
  4. MultimodalSearchEngine — 画像もテキストも1つの FAISS index に載せて
     id↔メタデータ付きで検索・永続化できる統一エンジン

なぜ合成データか:
  CLIP は「赤い円」「青い四角」のような色＋形の概念をゼロショットで強く捉えます。
  なので大量の実画像を DL しなくても "a photo of a red circle" → 赤い円 の検索が
  きちんと決まり、教材として意味のある類似度・Recall が出ます（README 参照）。
  ネットへ出るのは CLIP の重み DL のときだけ。入力は全て合成で完結します。

実画像で試したい人へ:
  data/42_multimodal_vector_search/ に画像を置くと build_collection より優先されます。
"""

from __future__ import annotations

import os
import pathlib

# --- HuggingFace / tokenizers のログを静かにして、教材の標準出力を読みやすくする ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも savefig が動くバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# 0. 定数・パス（プロジェクト規約に合わせて lectures/<module_id>/outputs/ に集約）
# ---------------------------------------------------------------------------
MODULE_ID = "42_multimodal_vector_search"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID

# CPU でも現実的な「最速の定番」。large / patch16 は CPU では重い。
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM = 512  # このモデルの画像・テキスト共通の埋め込み次元

# 合成コレクションの語彙。色は (R, G, B)。PIL/RGB のまま扱い BGR 混乱を避ける。
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 180, 60),
    "blue": (30, 60, 220),
    "yellow": (240, 210, 30),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")


# ---------------------------------------------------------------------------
# 1. device / 出力 / 図
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で device を返す（本講座は CPU 前提だが自動で活用）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """lectures/<module_id>/outputs/ を作って返す（既存でもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# 2. 合成コレクション（色 × 形）
# ---------------------------------------------------------------------------
def make_shape_image(
    color_name: str, shape_name: str, size: int = 224, bg: int = 245, jitter: int = 0, seed: int = 0
) -> Image.Image:
    """指定した色・形の合成画像（PIL.Image, RGB）を1枚作る。

    jitter>0 なら中心を ±jitter だけランダムにずらし、同じ (色,形) でも
    少しずつ違う画像を作る（画像→画像検索や Recall 評価に「複数枚」が要るため）。
    cv2 は描画にだけ使い、配列は最初から RGB として書くので BGR 変換は不要。
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), bg, dtype=np.uint8)  # 明るい灰色の背景
    color = COLORS[color_name]
    off = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
    c = size // 2 + off
    if shape_name == "circle":
        cv2.circle(img, (c, c), int(size * 0.30), color, thickness=-1)
    elif shape_name == "square":
        r = int(size * 0.28)
        cv2.rectangle(img, (c - r, c - r), (c + r, c + r), color, thickness=-1)
    elif shape_name == "triangle":
        pts = np.array(
            [[c, int(size * 0.16)], [int(size * 0.18), int(size * 0.82)], [int(size * 0.82), int(size * 0.82)]],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(f"未知の形: {shape_name}")
    return Image.fromarray(img)


def build_collection(variants: int = 4) -> tuple[list[Image.Image], list[dict]]:
    """色 × 形 × variants 枚 の合成画像コレクションを作る（既定 4×3×4 = 48枚）。

    返り値:
      images: PIL.Image のリスト
      metas:  各画像のメタ情報 dict のリスト
              {"modality": "image", "color", "shape", "text"(人間可読の説明)}
    検索の「正解」は color/shape の一致でこの metas から判定する。
    """
    images: list[Image.Image] = []
    metas: list[dict] = []
    for color_name in COLORS:
        for shape_name in SHAPES:
            for v in range(variants):
                seed = abs(hash((color_name, shape_name, v))) % 100_000
                images.append(make_shape_image(color_name, shape_name, jitter=18, seed=seed))
                metas.append(
                    {
                        "modality": "image",
                        "color": color_name,
                        "shape": shape_name,
                        "text": f"{color_name} {shape_name} #{v}",
                    }
                )
    return images, metas


def build_caption_set() -> tuple[list[str], list[dict]]:
    """色 × 形 の説明文（12本）とメタ情報を作る。テキスト側のDB / クエリ語彙。

    返り値:
      captions: "a photo of a red circle" のような文のリスト
      metas:    {"modality": "text", "color", "shape", "text"} のリスト
    """
    captions: list[str] = []
    metas: list[dict] = []
    for color_name in COLORS:
        for shape_name in SHAPES:
            cap = f"a photo of a {color_name} {shape_name}"
            captions.append(cap)
            metas.append({"modality": "text", "color": color_name, "shape": shape_name, "text": cap})
    return captions, metas


def load_user_images_or_none() -> tuple[list[Image.Image], list[dict]] | None:
    """data/42_multimodal_vector_search/ に実画像があれば (images, metas) を返す。無ければ None。

    実画像時の color/shape は不明なので "?" を入れ、text にはファイル名を使う。
    （実画像では Recall 評価の正解定義はユーザ自身が決める想定。検索自体は動く。）
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not DATA_DIR.is_dir():
        return None
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
    if not files:
        return None
    images = [Image.open(p).convert("RGB") for p in files]
    metas = [{"modality": "image", "color": "?", "shape": "?", "text": p.stem} for p in files]
    return images, metas


# ---------------------------------------------------------------------------
# 3. CLIP のロードと埋め込み（画像・テキストを同じ空間へ）
# ---------------------------------------------------------------------------
def load_clip(device: torch.device):
    """CLIP モデル（eval + device 済み）とプロセッサを返す。

    transformers v5 の正準: CLIPModel / CLIPProcessor（AutoModel/AutoProcessor でも可）。
    重み DL は初回のみネット接続が要る（以後はキャッシュ）。
    """
    from transformers import CLIPModel, CLIPProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def _pooled(features) -> torch.Tensor:
    """get_*_features の戻り値からベクトル本体 (B, D) を取り出す。

    ★transformers v5 の重要点:
      get_image_features / get_text_features は『テンソル』ではなく
      BaseModelOutputWithPooling を返す。射影後の埋め込みは .pooler_output に入る。
      旧版（テンソルを直接返す）との両対応にしておくと壊れにくい。
    """
    return features.pooler_output if hasattr(features, "pooler_output") else features


def embed_images(model, processor, images: list[Image.Image], device: torch.device, batch_size: int = 16) -> np.ndarray:
    """画像のリストを CLIP で埋め込み、(N, 512) float32 の ndarray を返す（未正規化）。

    正規化はわざと「しない」。FAISS 側で faiss.normalize_L2 する設計に統一するため。
    推論は torch.inference_mode()（勾配・autograd を完全に切って軽く・速く）。
    """
    vecs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            out = model.get_image_features(pixel_values=inputs["pixel_values"])
            vecs.append(_pooled(out).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(vecs, axis=0), dtype=np.float32)


def embed_texts(model, processor, texts: list[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    """テキストのリストを CLIP で埋め込み、(N, 512) float32 の ndarray を返す（未正規化）。

    画像と「同じ」空間に写るので、テキスト埋め込みで画像 index を引ける（クロスモーダル）。
    """
    vecs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True).to(device)
            out = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            vecs.append(_pooled(out).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(vecs, axis=0), dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. FAISS の小道具
# ---------------------------------------------------------------------------
def as_faiss_array(x: np.ndarray) -> np.ndarray:
    """FAISS 仕様（float32・C連続）に変換して返す。index へ渡す前に必ず通す。"""
    return np.ascontiguousarray(x, dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """各行を L2 正規化した「新しい」配列を返す（非破壊）。

    faiss.normalize_L2 は in-place で元配列を壊すので、学習用にこちらを使う。
    正規化後に内積を取ると = コサイン類似度になる。
    """
    x = as_faiss_array(x)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return as_faiss_array(x / np.maximum(norm, 1e-12))


# ---------------------------------------------------------------------------
# 5. 評価指標（retrieval 用）
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """1クエリの Recall@k = (上位k件に入った正解数) / (正解の総数)。

    relevant_ids が空のクエリは 0.0 を返す（0除算回避）。-1（無効ID）は自然に無視される。
    """
    if not relevant_ids:
        return 0.0
    topk = [i for i in retrieved_ids[:k] if i != -1]
    hit = len(set(topk) & relevant_ids)
    return hit / len(relevant_ids)


def average_precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """1クエリの AP@k（順位を考慮した適合率の平均）。mAP はこれを全クエリ平均したもの。

    上位から見て、正解を引くたびに「その時点の precision」を足し、正解総数(上限k)で割る。
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for rank, idx in enumerate(retrieved_ids[:k], start=1):
        if idx in relevant_ids:
            hits += 1
            score += hits / rank
    denom = min(len(relevant_ids), k)
    return score / denom if denom > 0 else 0.0


def mean_at_k(per_query: list[float]) -> float:
    """クエリごとの指標値リストを平均する（mean Recall / mAP の最終段）。"""
    return float(np.mean(per_query)) if per_query else 0.0


# ---------------------------------------------------------------------------
# 6. 統一マルチモーダル検索エンジン（画像もテキストも1つの index に）
# ---------------------------------------------------------------------------
class MultimodalSearchEngine:
    """画像・テキストを同じ FAISS index に載せ、id↔メタデータ付きで検索する統一エンジン。

    設計のキモ:
      - CLIP のおかげで画像とテキストは「同じ512次元空間」にある。だから両方を
        1つの IndexIDMap(IndexFlatIP) に入れておけば、どのモダリティのクエリでも
        どのモダリティの結果でも引ける（画像→画像 / テキスト→画像 / … が統一される）。
      - FAISS は連番の内部 ID しか持たないので、画像パスや説明文などのメタデータは
        Python 側の辞書(self.meta: id→dict)で別管理し、save 時に JSON へ書き出す。
      - 追加されるベクトルは内部で必ず L2 正規化する（内積=コサインにするため）。
    """

    def __init__(self, dim: int = CLIP_DIM, model_id: str = CLIP_MODEL_ID) -> None:
        self.dim = dim
        self.model_id = model_id
        # IndexIDMap で「任意の int64 ID」を割り当て可能にする（Flat だと 0..N-1 固定）。
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.meta: dict[int, dict] = {}  # id -> メタ情報
        self._next_id = 0

    # --- 追加 ---------------------------------------------------------------
    def add(self, embeddings: np.ndarray, metas: list[dict]) -> list[int]:
        """埋め込み(N, dim)とメタ情報(N件)を追加し、割り当てた ID のリストを返す。

        ID は内部カウンタで自動採番する。次元不一致は静かにバグるので最初に弾く。
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"次元不一致: index={self.dim} だが embeddings={embeddings.shape[1]}")
        if len(embeddings) != len(metas):
            raise ValueError("embeddings と metas の件数が一致しません")
        ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
        self._next_id += len(embeddings)
        xb = l2_normalize(embeddings)  # 内積=コサインにするため正規化してから add
        self.index.add_with_ids(xb, ids)
        for i, m in zip(ids.tolist(), metas):
            self.meta[i] = dict(m)
        return ids.tolist()

    # --- 検索 ---------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 5, modality: str | None = None) -> list[list[dict]]:
        """クエリ埋め込み(Q, dim)で上位k件を引く。modality を指定すると結果を絞る。

        返り値: クエリごとの結果リスト。各要素は {"id","score","meta"} の dict。
        modality 絞り込みは「多めに取って Python で後フィルタ」で実装（教材の簡明さ優先）。
        実運用で厳密に絞るなら faiss の IDSelector を使う（README 参照）。
        """
        if query.ndim == 1:
            query = query[None, :]
        xq = l2_normalize(query)
        # modality 指定時は、フィルタで件数が減るぶん多めに取る。
        fetch = k if modality is None else min(k * 5 + 10, max(self.index.ntotal, 1))
        scores, ids = self.index.search(xq, fetch)
        results: list[list[dict]] = []
        for row_scores, row_ids in zip(scores, ids):
            hits: list[dict] = []
            for s, i in zip(row_scores.tolist(), row_ids.tolist()):
                if i == -1:  # 近傍が足りないと -1。ID として使うとクラッシュするのでガード。
                    continue
                m = self.meta.get(int(i))
                if m is None:
                    continue
                if modality is not None and m.get("modality") != modality:
                    continue
                hits.append({"id": int(i), "score": float(s), "meta": m})
                if len(hits) >= k:
                    break
            results.append(hits)
        return results

    # --- 永続化 -------------------------------------------------------------
    def save(self, stem: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        """index 本体(.faiss) と メタデータ(.meta.json) をセットで保存する。

        ★よくある罠: write_index は index 本体しか保存しない。ID→メタの対応表は
        別ファイルに必ず一緒に書き出し、バージョンを整合させること。
        """
        stem = pathlib.Path(stem)
        idx_path = stem.with_suffix(".faiss")
        meta_path = stem.with_suffix(".meta.json")
        faiss.write_index(self.index, str(idx_path))
        import json

        payload = {
            "dim": self.dim,
            "model_id": self.model_id,
            "next_id": self._next_id,
            # JSON のキーは文字列になるので、ロード時に int へ戻す。
            "meta": {str(k): v for k, v in self.meta.items()},
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return idx_path, meta_path

    @classmethod
    def load(cls, stem: pathlib.Path) -> "MultimodalSearchEngine":
        """save で書いた .faiss と .meta.json から復元する。"""
        import json

        stem = pathlib.Path(stem)
        payload = json.loads(stem.with_suffix(".meta.json").read_text(encoding="utf-8"))
        eng = cls(dim=int(payload["dim"]), model_id=payload["model_id"])
        eng.index = faiss.read_index(str(stem.with_suffix(".faiss")))
        eng.meta = {int(k): v for k, v in payload["meta"].items()}
        eng._next_id = int(payload["next_id"])
        return eng


# ---------------------------------------------------------------------------
# 7. 可視化の小道具
# ---------------------------------------------------------------------------
def save_query_results_figure(
    query_title: str, query_image: Image.Image | None, result_images: list, result_titles: list[str],
    path: pathlib.Path,
) -> None:
    """1クエリの検索結果（左にクエリ、右に上位ヒット画像）を1枚にまとめて保存する。"""
    n = len(result_images)
    cols = n + (1 if query_image is not None else 0)
    cols = max(cols, 1)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.3, 2.8))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    col = 0
    if query_image is not None:
        axes[col].imshow(query_image)
        axes[col].set_title(f"QUERY\n{query_title}", fontsize=9, color="darkred")
        col += 1
    for im, t in zip(result_images, result_titles):
        axes[col].imshow(im)
        axes[col].set_title(t, fontsize=8)
        col += 1
    if query_image is None:
        fig.suptitle(f"QUERY (text): {query_title}", fontsize=10, color="darkred")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認（CLIP は使わず軽い部分だけ）。
    out = ensure_output_dir()
    dev = pick_device()
    images, metas = build_collection(variants=2)
    caps, cmetas = build_caption_set()
    # CLIP を使わない決定的ベクトルでエンジンの形だけ確認する。
    rng = np.random.default_rng(0)
    eng = MultimodalSearchEngine(dim=8)
    fake = rng.standard_normal((len(metas), 8)).astype(np.float32)
    eng.add(fake, metas)
    res = eng.search(fake[0], k=3)
    print(f"device={dev} images={len(images)} captions={len(caps)} ntotal={eng.index.ntotal}")
    print(f"スモーク検索 top1 id={res[0][0]['id']} score={res[0][0]['score']:.3f}")
    print(f"out={out}")
