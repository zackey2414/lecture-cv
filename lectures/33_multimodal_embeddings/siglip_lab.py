"""第33回 共通の道具箱（マルチモーダル埋め込み / SigLIP・SigLIP2・CLIP）。

このファイルは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務をはっきり絞ってあります（単一責務）:

  1. device 判定 / 出力ディレクトリ / matplotlib(Agg) などの共通設定
  2. 合成コレクション（色 × 形）の画像と、多言語キャプションの生成（ネット不要・決定的）
  3. MMEncoder — CLIP / SigLIP / SigLIP2 を同じ作法で扱い、画像・テキストを
     同一空間のベクトルへ写す薄いラッパ（モデルごとの前処理差を内部で吸収）
  4. FAISS の小道具（内積=コサインの index）と検索評価指標（Recall@k / AP@k）
  5. ImageBind の「束ねる」原理を体感するためのトイ三モーダル空間ジェネレータ

なぜ合成データか:
  SigLIP/CLIP は「赤い円」「青い四角」のような “色 × 形” の概念をゼロショットで
  強く捉えます。実画像を大量に DL しなくても "a red circle" → 赤い円 の検索が
  きちんと決まり、Recall も意味のある値になります。ネットへ出るのはモデル重みの
  DL のときだけで、入力は全て合成で完結します。

実画像で試したい人へ:
  data/33_multimodal_embeddings/ に画像を置くと build_collection より優先されます。

参照ライブラリ+版: torch 2.12+cpu / transformers 5.11 / faiss-cpu / numpy (2026-06)。
"""

from __future__ import annotations

import os

# --- HuggingFace / tokenizers のログを静かにして、教材の標準出力を読みやすくする ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib  # noqa: E402
import wave  # noqa: E402

import cv2  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも savefig が動くバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# CPU を1スレッドが食い潰さないよう、ほどほどに制限（環境が小さくても安定して速い）。
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

# ---------------------------------------------------------------------------
# 0. 定数・パス（プロジェクト規約に合わせて lectures/<module_id>/outputs/ に集約）
# ---------------------------------------------------------------------------
MODULE_ID = "33_multimodal_embeddings"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID

# 3つの代表モデル。いずれも base サイズで CPU でも数秒〜十数秒で動く。
#   clip    : softmax 対照損失。テキスト側は英語中心。比較対象。
#   siglip  : sigmoid 損失。英語中心。本モジュールの主役。
#   siglip2 : sigmoid 損失 + 多言語トークナイザ(Gemma)。多言語検索の強化に使う。
MODEL_IDS: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "siglip2": "google/siglip2-base-patch16-224",
}
# それぞれの「画像・テキスト共通の埋め込み次元」。CLIP は 512、SigLIP 系は 768。
# → 次元が違うので CLIP と SigLIP は同じ FAISS index に混ぜられない（別空間）。
MODEL_DIMS: dict[str, int] = {"clip": 512, "siglip": 768, "siglip2": 768}

# 合成コレクションの語彙。色は (R, G, B)。PIL/RGB のまま扱い BGR 混乱を避ける。
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 70, 220),
    "yellow": (240, 205, 30),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")

# 多言語キャプションの語彙。SigLIP2 が多言語に強いことを「同じ画像」で確かめるための辞書。
# 文法は厳密でなくてよい（モデルは概念を捉える）。色名・形名 + テンプレートで機械生成する。
CAPTION_VOCAB: dict[str, dict] = {
    "en": {
        "color": {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow"},
        "shape": {"circle": "circle", "square": "square", "triangle": "triangle"},
        "template": "a photo of a {color} {shape}",
    },
    "ja": {
        "color": {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄"},
        "shape": {"circle": "円", "square": "四角形", "triangle": "三角形"},
        "template": "{color}色の{shape}の写真",
    },
    "fr": {
        "color": {"red": "rouge", "green": "vert", "blue": "bleu", "yellow": "jaune"},
        "shape": {"circle": "cercle", "square": "carré", "triangle": "triangle"},
        "template": "une photo d'un {shape} {color}",
    },
    "es": {
        "color": {"red": "rojo", "green": "verde", "blue": "azul", "yellow": "amarillo"},
        "shape": {"circle": "círculo", "square": "cuadrado", "triangle": "triángulo"},
        "template": "una foto de un {shape} {color}",
    },
}


# ---------------------------------------------------------------------------
# 1. device / 出力 / 乱数
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
# 2. 合成コレクション（色 × 形）と多言語キャプション
# ---------------------------------------------------------------------------
def make_shape_image(
    color_name: str, shape_name: str, size: int = 224, bg: int = 245, jitter: int = 0, seed: int = 0
) -> Image.Image:
    """指定した色・形の合成画像（PIL.Image, RGB）を1枚作る。

    jitter>0 なら中心を ±jitter だけランダムにずらし、同じ (色,形) でも少しずつ違う
    画像にする（Recall 評価には「同じ概念の別バリアント」が複数必要なため）。
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
            [
                [c, int(size * 0.16)],
                [int(size * 0.18), int(size * 0.82)],
                [int(size * 0.82), int(size * 0.82)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(f"未知の形: {shape_name}")
    return Image.fromarray(img)


def build_collection(variants: int = 3) -> tuple[list[Image.Image], list[dict]]:
    """色 × 形 × variants 枚 の合成画像コレクションを作る（既定 4×3×3 = 36枚）。

    返り値:
      images: PIL.Image のリスト
      metas:  各画像のメタ情報 dict（{"modality","color","shape","concept","text"}）
    検索の「正解」は concept = "color shape"（例 "red circle"）の一致で判定する。
    """
    images: list[Image.Image] = []
    metas: list[dict] = []
    # seed は決定的に採番する（Python の hash() はプロセスごとにランダム化され再現しないため）。
    counter = 0
    for color_name in COLORS:
        for shape_name in SHAPES:
            for v in range(variants):
                seed = counter * 7 + v  # 概念ごとに散らばりつつ、実行間で完全に再現する
                counter += 1
                images.append(make_shape_image(color_name, shape_name, jitter=18, seed=seed))
                metas.append(
                    {
                        "modality": "image",
                        "color": color_name,
                        "shape": shape_name,
                        "concept": f"{color_name} {shape_name}",
                        "text": f"{color_name} {shape_name} #{v}",
                    }
                )
    return images, metas


def build_caption(color_name: str, shape_name: str, lang: str = "en") -> str:
    """1つの (色, 形) を指定言語のキャプション文にする（CAPTION_VOCAB を使用）。"""
    voc = CAPTION_VOCAB[lang]
    return voc["template"].format(color=voc["color"][color_name], shape=voc["shape"][shape_name])


def build_concept_captions(lang: str = "en") -> tuple[list[str], list[dict]]:
    """全 (色 × 形) = 12 概念のキャプション文と、その正解メタを指定言語で作る。

    返り値:
      captions: 文のリスト（12本）
      metas:    {"modality":"text","color","shape","concept","lang","text"} のリスト
    """
    captions: list[str] = []
    metas: list[dict] = []
    for color_name in COLORS:
        for shape_name in SHAPES:
            cap = build_caption(color_name, shape_name, lang)
            captions.append(cap)
            metas.append(
                {
                    "modality": "text",
                    "color": color_name,
                    "shape": shape_name,
                    "concept": f"{color_name} {shape_name}",
                    "lang": lang,
                    "text": cap,
                }
            )
    return captions, metas


def load_user_images_or_none() -> tuple[list[Image.Image], list[dict]] | None:
    """data/33_multimodal_embeddings/ に実画像があれば (images, metas) を返す。無ければ None。

    実画像の color/shape は不明なので "?" を入れ、concept にはファイル名 stem を使う。
    （実画像では検索自体は動くが、Recall の正解定義はユーザ側で決める想定。）
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not DATA_DIR.is_dir():
        return None
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
    if not files:
        return None
    images = [Image.open(p).convert("RGB") for p in files]
    metas = [
        {"modality": "image", "color": "?", "shape": "?", "concept": p.stem, "text": p.stem}
        for p in files
    ]
    return images, metas


# ---------------------------------------------------------------------------
# 3. MMEncoder — CLIP / SigLIP / SigLIP2 を「同じ作法」で扱う薄いラッパ
# ---------------------------------------------------------------------------
def _pooled(features) -> torch.Tensor:
    """get_*_features の戻り値からベクトル本体 (B, D) を取り出す。

    ★transformers v5 の重要点:
      get_image_features / get_text_features は『テンソル』ではなく
      BaseModelOutputWithPooling を返すことがある。射影後の埋め込みは .pooler_output。
      旧版（テンソル直返し）との両対応にしておくと壊れにくい。
    """
    return features.pooler_output if hasattr(features, "pooler_output") else features


class MMEncoder:
    """画像・テキストを同一空間のベクトルへ写すエンコーダ（CLIP/SigLIP/SigLIP2 共通 API）。

    モデルごとの前処理の差を内部で吸収するのが役目:
      - SigLIP / SigLIP2 はテキストを padding="max_length"（既定 64 トークン）で詰める。
        テキスト特徴は input_ids だけで取り（attention_mask は使わない）。
      - CLIP は padding=True（バッチ内最長）で詰め、input_ids + attention_mask を渡す。
    埋め込みはあえて「正規化しない」で返す。正規化は FAISS 側 / 呼び出し側で行い、
    「未正規化を直接コサインに使う」初学者の罠を一箇所に閉じ込める設計にしている。
    """

    def __init__(self, kind: str = "siglip") -> None:
        if kind not in MODEL_IDS:
            raise ValueError(f"kind は {list(MODEL_IDS)} のいずれか。受け取った値: {kind}")
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.kind = kind
        self.model_id = MODEL_IDS[kind]
        self.dim = MODEL_DIMS[kind]
        self.device = pick_device()
        self.is_siglip = kind in ("siglip", "siglip2")
        # 初回のみネット接続で重みを DL（以後はキャッシュ）。eval + device 済みにしておく。
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    # --- 前処理（モデル差の吸収） -----------------------------------------
    def _encode_text_inputs(self, texts: list[str]):
        if self.is_siglip:
            # SigLIP は固定長(max_length)で詰めるのが正準。attention_mask は返らない。
            return self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
        return self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)

    # --- 埋め込み ---------------------------------------------------------
    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """画像リストを (N, dim) float32 の未正規化 ndarray にする。"""
        vecs: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                vecs.append(_pooled(out).cpu().numpy())
        return np.ascontiguousarray(np.concatenate(vecs, axis=0), dtype=np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """テキストリストを (N, dim) float32 の未正規化 ndarray にする。

        画像と「同じ」空間に写るので、テキスト埋め込みで画像 index を引ける（クロスモーダル）。
        """
        vecs: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._encode_text_inputs(batch)
                if self.is_siglip:
                    out = self.model.get_text_features(input_ids=inputs["input_ids"])
                else:
                    out = self.model.get_text_features(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                    )
                vecs.append(_pooled(out).cpu().numpy())
        return np.ascontiguousarray(np.concatenate(vecs, axis=0), dtype=np.float32)

    # --- ゼロショット分類のための生ロジット --------------------------------
    def logits_per_image(self, images: list[Image.Image], labels: list[str]) -> np.ndarray:
        """model(**inputs).logits_per_image を (N_img, N_label) の ndarray で返す。

        この生ロジットに対して、CLIP は softmax、SigLIP は sigmoid を掛けて確率にする
        （損失設計の違い＝確率解釈の違い。01 で実演する）。
        """
        if self.is_siglip:
            inputs = self.processor(
                images=images, text=labels, padding="max_length", return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                images=images, text=labels, padding=True, return_tensors="pt"
            ).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        return out.logits_per_image.cpu().numpy()


# ---------------------------------------------------------------------------
# 4. FAISS の小道具（内積 = コサイン）
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """各行を L2 正規化した「新しい」配列を返す（非破壊）。

    faiss.normalize_L2 は in-place で元配列を壊すので、安全なこちらを使う。
    正規化後に内積を取ると = コサイン類似度になる。
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norm, 1e-12), dtype=np.float32)


def build_ip_index(vectors: np.ndarray) -> faiss.Index:
    """ベクトル(N, dim)を L2 正規化して IndexFlatIP に載せて返す（内積=コサイン）。"""
    xb = l2_normalize(vectors)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    return index


def search_index(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """クエリ(Q, dim)を正規化して上位k件を引く。(scores, ids) を返す。"""
    if queries.ndim == 1:
        queries = queries[None, :]
    xq = l2_normalize(queries)
    k = min(k, index.ntotal)
    return index.search(xq, k)


# ---------------------------------------------------------------------------
# 5. 検索評価指標（Recall@k / AP@k / mean）
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

    上位から見て、正解を引くたびに「その時点の precision」を足し、min(正解数, k) で割る。
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


def mean(values: list[float]) -> float:
    """値リストの平均（mean Recall / mAP の最終段）。空なら 0.0。"""
    return float(np.mean(values)) if values else 0.0


def evaluate_text_to_image(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    text_metas: list[dict],
    image_metas: list[dict],
    k: int = 5,
) -> dict:
    """テキスト→画像検索の Recall@k と mAP@k をまとめて計算する。

    正解は concept（"red circle" など）の一致で定義。返り値は集計と per-query の生値。
    """
    index = build_ip_index(image_emb)
    _scores, ids = search_index(index, text_emb, k=index.ntotal)  # 全件ランキング
    recalls: list[float] = []
    aps: list[float] = []
    for qi, tmeta in enumerate(text_metas):
        relevant = {j for j, m in enumerate(image_metas) if m["concept"] == tmeta["concept"]}
        ranked = [int(i) for i in ids[qi].tolist()]
        recalls.append(recall_at_k(ranked, relevant, k))
        aps.append(average_precision_at_k(ranked, relevant, k))
    return {
        "k": k,
        "recall_at_k": mean(recalls),
        "map_at_k": mean(aps),
        "per_query_recall": recalls,
        "per_query_ap": aps,
    }


# ---------------------------------------------------------------------------
# 6. ImageBind の「束ねる」原理を体感するトイ三モーダル空間
# ---------------------------------------------------------------------------
def make_trimodal_toy(
    concept_names: list[str], dim: int = 64, noise: float = 0.18, seed: int = 0
) -> dict[str, np.ndarray]:
    """音声・画像・テキストを「1つの空間に束ねた」状態を再現するトイ埋め込みを作る。

    ImageBind の本質は「各モダリティの encoder が、同じ概念を “共通アンカー” の近くへ
    写すよう学習されている」こと。ここではその学習結果を、概念ごとのランダムな単位ベクトル
    （アンカー）＋モダリティ固有の小さなノイズ、として直接構成する。実モデルの重みは使わない
    が、『共有空間があれば 音→画像 が cos 類似度で引ける』というメカニズムは忠実に再現する。

    返り値: {"anchor","image","text","audio"} それぞれ (n_concept, dim) の正規化済み配列。
    """
    rng = np.random.default_rng(seed)
    n = len(concept_names)
    anchor = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))

    def encode(noise_scale: float) -> np.ndarray:
        # アンカー + モダリティ固有ノイズ → 正規化。各モダリティで別ノイズを引く。
        return l2_normalize(anchor + noise_scale * rng.standard_normal((n, dim)).astype(np.float32))

    return {
        "anchor": anchor,
        "image": encode(noise),
        "text": encode(noise),
        "audio": encode(noise),
    }


def synth_tone(freq: float, dur: float = 0.4, sr: int = 8000, seed: int = 0) -> np.ndarray:
    """概念ごとに違う「音」を作るための単純な合成波形（float32, [-1,1]）。

    教材で『音声クエリ』を具体物として見せる（波形を可視化・wav 保存する）ためのもの。
    トイ埋め込みの正体はアンカーなので、この波形自体は埋め込みには使わない（あくまで可視化用）。
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    wave_arr = 0.6 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 2 * freq * t)
    wave_arr = wave_arr + 0.02 * rng.standard_normal(t.shape)  # ほんの少しノイズ
    return wave_arr.astype(np.float32)


def save_wav(path: pathlib.Path, wave_arr: np.ndarray, sr: int = 8000) -> None:
    """float32 [-1,1] の波形を 16bit PCM の .wav として保存する（標準ライブラリ wave）。"""
    pcm = np.clip(wave_arr, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


# ---------------------------------------------------------------------------
# 7. 可視化の小道具
# ---------------------------------------------------------------------------
def save_retrieval_figure(
    query_title: str,
    query_image: Image.Image | None,
    result_images: list,
    result_titles: list[str],
    path: pathlib.Path,
) -> None:
    """1クエリの検索結果（左にクエリ、右に上位ヒット画像）を1枚にまとめて保存する。"""
    n = len(result_images)
    cols = max(n + (1 if query_image is not None else 0), 1)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.3, 2.9))
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


def save_heatmap(matrix: np.ndarray, row_labels: list[str], title: str, path: pathlib.Path) -> None:
    """類似度行列をヒートマップに保存する（行ラベル付き）。"""
    fig, ax = plt.subplots(figsize=(11, max(3.0, len(row_labels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("image index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認（モデルは使わず軽い部分だけ）。
    out = ensure_output_dir()
    dev = pick_device()
    images, metas = build_collection(variants=2)
    caps_en, _ = build_concept_captions("en")
    caps_ja, _ = build_concept_captions("ja")
    toy = make_trimodal_toy([m["concept"] for m in metas[:6]], dim=32)
    # トイ空間で「音→画像」最近傍が自分自身（同概念）になることを確認。
    sim = toy["audio"] @ toy["image"].T
    top1 = int(np.argmax(sim[0]))
    print(f"device={dev} images={len(images)} en[0]={caps_en[0]!r} ja[0]={caps_ja[0]!r}")
    print(f"toy audio->image top1 index = {top1}（0 になれば束ね成功）")
    print(f"out={out}")
