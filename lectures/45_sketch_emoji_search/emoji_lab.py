"""第45回 共通の道具箱（手書きスケッチで絵文字を検索する SBIR）。

このモジュールは「読み物」ではなく、各スクリプト（01〜04 / mini_project /
exercises）が import して使う再利用部品です。スケッチベース画像検索（SBIR）の
パイプライン全体を、単一責務の小さな関数に割って提供します。

提供する道具（責務ごと）:
  1. パス／デバイス        : output_dir / data_dir / pick_device
  2. FAISS 用の数値処理     : as_faiss_array / l2_normalize
  3. 絵文字コレクションの用意 : build_emoji_collection（data→フォント→合成 の優先順）
  4. 合成の顔（絵文字/線画） : draw_synthetic_emoji / make_synthetic_sketch
  5. スタイル変換           : style_transform（color/gray/edge/invert＝ドメインギャップ実験用）
  6. スケッチ前処理         : preprocess_sketch（白背景・黒線・正方化・グレースケール）
  7. CLIP 埋め込み          : EmojiEmbedder（取得不可なら画素記述子へ自動フォールバック）
  8. FAISS 索引            : build_index / save_index / load_index / search_topn
  9. 可視化                : save_gallery / save_result_panel

設計上の肝（なぜグレースケール？）:
  スケッチは「白背景に黒線」の線画、絵文字は「色つきの塗り」です。この見た目の差
  （ドメインギャップ）が CLIP 埋め込みの距離を不必要に開かせます。そこで絵文字側を
  グレースケール化（さらに発展としてエッジ抽出/反転）してクエリの様式に寄せると、
  検索の当たりが良くなります。本講座の既定の索引は「絵文字をグレースケール化して
  埋め込む」方針です（04_eval_domaingap.py で効果を定量化します）。

実行はプロジェクトルートから:
  uv run python lectures/45_sketch_emoji_search/emoji_lab.py   # 単体スモークテスト
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import faiss
import numpy as np

# transformers / huggingface_hub のログを静かにして教材の出力を読みやすくする。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODULE_ID = "45_sketch_emoji_search"
# CPU でも現実的な「最速の定番」。patch16 や large は CPU では重い。
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# 作業キャンバスの一辺（描画・前処理の基準サイズ。CLIP 側でさらに 224 に正規化される）。
CANVAS = 224


# =====================================================================
# パス / デバイス
# =====================================================================


def _root() -> pathlib.Path:
    """リポジトリルート（このファイルから3つ上: lectures/45_.../emoji_lab.py）。"""
    return pathlib.Path(__file__).resolve().parents[2]


def output_dir() -> pathlib.Path:
    """outputs/45_sketch_emoji_search/ を作って返す（無ければ作成）。"""
    out = _root() / "outputs" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_dir() -> pathlib.Path:
    """data/45_sketch_emoji_search/（実データの置き場）。無くても呼び出し側が判断する。"""
    return _root() / "data" / MODULE_ID


def pick_device() -> str:
    """cuda > mps > cpu の順に使えるデバイス名を返す（全回共通の定石）。

    torch を遅延 import し、未導入でも 'cpu' を返して落とさない。
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

    float64 や非連続（スライス/転置）のまま search に渡すと落ちる・誤動作するので、
    index に渡す直前に必ずこれを通す。
    """
    return np.ascontiguousarray(x, dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """各行を L2 ノルム1に正規化した「コピー」を返す（float32・C連続）。

    コサイン類似度を内積（IndexFlatIP）で計算するための前処理。
    注意: faiss.normalize_L2(x) は in-place で元配列を破壊する。学習用に挙動を
    切り分けたいので、ここでは破壊しない純粋関数版を用意する。
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # 0除算ガード（零ベクトル対策）
    return np.ascontiguousarray(x / norms, dtype=np.float32)


# =====================================================================
# 合成の顔 — 絵文字（塗り）と スケッチ（線画）で「同じ幾何」を共有する
# =====================================================================
#
# 表情ごとに (名前, 目, 口, 眉, 追加) を定義し、emoji/sketch の両モードで同じ
# パーツを描く。こうすると「happy のスケッチ」と「happy の絵文字」が幾何的に対応し、
# 04_eval_domaingap.py の Recall 評価で “正解” がはっきりする。

CENTER = (CANVAS // 2, CANVAS // 2)  # 顔の中心
FACE_R = 92  # 顔の半径
EYE_Y = 96  # 目の高さ
EYE_DX = 34  # 目の左右オフセット（左=center-DX, 右=center+DX）
MOUTH_Y = 150  # 口のおおよその高さ

# (name, eye, mouth, brow, extra)。すべて (eye,mouth,brow,extra) が一意になるようにしてある。
EXPRESSIONS: list[tuple[str, str, str, str, str]] = [
    ("happy", "dot", "smile", "none", "none"),
    ("laugh", "closed", "grin", "none", "none"),
    ("sad", "dot", "frown", "none", "none"),
    ("cry", "dot", "frown", "none", "tear"),
    ("neutral", "dot", "line", "none", "none"),
    ("surprised", "wide", "o", "raised", "none"),
    ("angry", "dot", "frown", "angry", "none"),
    ("wink", "wink", "smile", "none", "none"),
    ("smirk", "dot", "smirk", "none", "none"),
    ("sleepy", "sleepy", "line", "none", "none"),
    ("worried", "dot", "smallfrown", "raised", "none"),
    ("confused", "dot", "wavy", "one_raised", "none"),
    ("grin", "dot", "grin", "none", "none"),
    ("cool", "closed", "smile", "none", "none"),
    ("shock", "wide", "o", "none", "none"),
    ("sad_open", "wide", "frown", "raised", "none"),
]


def _draw_eyes(draw, style: str, color) -> None:
    """目を描く（パーツ単位の単一責務）。"""
    lx, rx, y = CENTER[0] - EYE_DX, CENTER[0] + EYE_DX, EYE_Y
    if style == "dot":
        for x in (lx, rx):
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=color)
    elif style == "wide":  # 見開き（驚き）
        for x in (lx, rx):
            draw.ellipse([x - 14, y - 17, x + 14, y + 17], fill=color)
    elif style == "closed":  # 閉じた笑い目 ^ ^（上向きの弧）
        for x in (lx, rx):
            draw.arc([x - 15, y - 8, x + 15, y + 14], start=200, end=340, fill=color, width=5)
    elif style == "wink":  # 左は閉じ（横線）、右は点
        draw.line([lx - 13, y, lx + 13, y], fill=color, width=5)
        draw.ellipse([rx - 10, y - 10, rx + 10, y + 10], fill=color)
    elif style == "sleepy":  # 半目（下向きの弧）
        for x in (lx, rx):
            draw.arc([x - 15, y - 12, x + 15, y + 10], start=20, end=160, fill=color, width=5)
    else:  # 未知スタイルは点でフォールバック
        for x in (lx, rx):
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=color)


def _draw_brows(draw, style: str, color) -> None:
    """眉を描く（怒り・驚き・困惑などの印象を決める）。"""
    if style == "none":
        return
    lx, rx, y = CENTER[0] - EYE_DX, CENTER[0] + EYE_DX, EYE_Y - 26
    if style == "angry":  # 内側下がりのハの字（怒り）
        draw.line([lx - 15, y - 6, lx + 13, y + 8], fill=color, width=5)
        draw.line([rx + 15, y - 6, rx - 13, y + 8], fill=color, width=5)
    elif style == "raised":  # 上がり眉（驚き/不安）
        for x in (lx, rx):
            draw.arc([x - 15, y - 2, x + 15, y + 22], start=200, end=340, fill=color, width=4)
    elif style == "one_raised":  # 片眉上げ（困惑）
        draw.arc([lx - 15, y - 2, lx + 15, y + 22], start=200, end=340, fill=color, width=4)
        draw.line([rx - 15, y + 6, rx + 15, y + 6], fill=color, width=4)


def _draw_mouth(draw, style: str, color) -> None:
    """口を描く（表情の主役。弧の向き=笑い/悲しみ）。

    PIL の arc は y 軸下向き・時計回りで角度を測る。start=0,end=180 は楕円の下半分
    （= U 字の笑顔）、start=180,end=360 は上半分（= ∩ の困り顔）になる。
    """
    cx, y = CENTER[0], MOUTH_Y
    if style == "smile":
        draw.arc([cx - 38, y - 30, cx + 38, y + 22], start=0, end=180, fill=color, width=6)
    elif style == "frown":
        draw.arc([cx - 38, y - 4, cx + 38, y + 48], start=180, end=360, fill=color, width=6)
    elif style == "smallfrown":
        draw.arc([cx - 22, y - 2, cx + 22, y + 26], start=180, end=360, fill=color, width=5)
    elif style == "line":
        draw.line([cx - 30, y + 6, cx + 30, y + 6], fill=color, width=6)
    elif style == "o":  # 開いた口（塗り）
        draw.ellipse([cx - 18, y - 12, cx + 18, y + 30], fill=color)
    elif style == "grin":  # 大きな歯見せ笑い（下半楕円の塗り）
        draw.chord([cx - 44, y - 34, cx + 44, y + 30], start=0, end=180, fill=color)
    elif style == "smirk":  # 片方上がりの短い線
        draw.line([cx - 6, y + 8, cx + 30, y - 2], fill=color, width=6)
    elif style == "wavy":  # 波線（困惑）
        pts = [(cx - 30, y + 4), (cx - 15, y - 6), (cx, y + 4), (cx + 15, y - 6), (cx + 30, y + 4)]
        draw.line(pts, fill=color, width=5, joint="curve")


def _draw_extra(draw, extra: str, color) -> None:
    """付帯要素（涙など）。emoji/sketch どちらも黒で描き、対応を崩さない。"""
    if extra == "tear":
        x = CENTER[0] + EYE_DX  # 右目の下に涙
        draw.ellipse([x - 6, EYE_Y + 14, x + 6, EYE_Y + 40], fill=color)


def _draw_face(spec: tuple[str, str, str, str, str], mode: str):
    """表情スペックから顔を1枚描く。mode='emoji'（塗り）/ 'sketch'（線画）。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    d = ImageDraw.Draw(img)
    cx, cy = CENTER
    if mode == "emoji":
        # 黄色い塗りの円＋黒い表情パーツ（典型的な絵文字の見た目）。
        d.ellipse([cx - FACE_R, cy - FACE_R, cx + FACE_R, cy + FACE_R],
                  fill=(255, 205, 40), outline=(90, 70, 0), width=3)
        feat = (40, 30, 0)
    else:  # sketch: 白背景に黒線（手書きの様式に寄せる）
        d.ellipse([cx - FACE_R, cy - FACE_R, cx + FACE_R, cy + FACE_R], outline=(0, 0, 0), width=5)
        feat = (0, 0, 0)
    _, eye, mouth, brow, extra = spec
    _draw_brows(d, brow, feat)
    _draw_eyes(d, eye, feat)
    _draw_mouth(d, mouth, feat)
    _draw_extra(d, extra, feat)
    return img


def draw_synthetic_emoji(spec: tuple[str, str, str, str, str]):
    """合成絵文字（黄円＋表情）を1枚返す（PIL.Image, RGB）。"""
    return _draw_face(spec, "emoji")


def make_synthetic_sketch(spec: tuple[str, str, str, str, str] | None = None):
    """合成スケッチ（白背景・黒線の顔の線画）を1枚返す。

    spec を省略すると happy（先頭の表情）の線画を返す。display 無し環境での
    手書き入力の代用（フォールバック）に使う。
    """
    if spec is None:
        spec = EXPRESSIONS[0]
    return _draw_face(spec, "sketch")


def build_synthetic_pairs():
    """評価用: 表情ごとの (合成絵文字, 合成スケッチ, ラベル) を対応づけて返す。

    絵文字（塗り）とスケッチ（線画）は独立に描いた別レンダリングなので、
    両者の間には本物のドメインギャップがある。これを使うと「スケッチ→絵文字検索」の
    Recall を、正解ラベルつきで誠実に測れる。
    """
    emojis = [draw_synthetic_emoji(s) for s in EXPRESSIONS]
    sketches = [make_synthetic_sketch(s) for s in EXPRESSIONS]
    labels = [s[0] for s in EXPRESSIONS]
    return emojis, sketches, labels


# =====================================================================
# 絵文字フォントでの描画（顔絵文字を PIL で1枚ずつラスタライズ）
# =====================================================================

# 多様な表情の顔絵文字。フォントにグリフが無いものは描画時に自動スキップする。
EMOJI_CHARS: list[tuple[str, str]] = [
    ("\U0001F600", "grinning"), ("\U0001F603", "smiling"), ("\U0001F604", "grin_eyes"),
    ("\U0001F601", "beaming"), ("\U0001F605", "sweat_smile"), ("\U0001F602", "tears_joy"),
    ("\U0001F642", "slight_smile"), ("\U0001F609", "winking"), ("\U0001F60A", "smiling_eyes"),
    ("\U0001F60D", "heart_eyes"), ("\U0001F61C", "tongue_wink"), ("\U0001F914", "thinking"),
    ("\U0001F610", "neutral"), ("\U0001F611", "expressionless"), ("\U0001F644", "rolling_eyes"),
    ("\U0001F60F", "smirk"), ("\U0001F623", "persevere"), ("\U0001F62E", "open_mouth"),
    ("\U0001F62A", "sleepy"), ("\U0001F634", "sleeping"), ("\U0001F621", "angry"),
    ("\U0001F620", "mad"), ("\U0001F622", "crying"), ("\U0001F62D", "sob"),
    ("\U0001F631", "scream"), ("\U0001F633", "flushed"),
]

# よく入っている絵文字フォントの代表パス（Linux / macOS）。
_EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf",
    "/usr/share/fonts/truetype/ancient-scripts/Symbola.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]


def _find_emoji_font() -> str | None:
    """絵文字フォントのパスを1つ見つけて返す（無ければ None）。

    既知パス→fc-list 探索の順。色絵文字（CBDT 等）・モノクロ絵文字のどちらでもよい。
    """
    for p in _EMOJI_FONT_CANDIDATES:
        if pathlib.Path(p).exists():
            return p
    try:
        out = subprocess.run(["fc-list"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return None
    for line in out.splitlines():
        path = line.split(":", 1)[0].strip()
        low = (line.lower())
        if path.endswith((".ttf", ".otf", ".ttc")) and ("emoji" in low or "symbola" in low):
            return path
    return None


def _render_emoji_glyph(char: str, font_path: str, target: int = CANVAS):
    """1文字の絵文字を白背景に描いて PIL.Image(RGB) を返す。描けなければ None。

    色絵文字フォント（例 Noto Color Emoji）は埋め込みビットマップの「特定サイズ」
    でしか開けない（例: 109px のみ）。複数サイズを試して最初に開けたものを使い、
    最後に target にリサイズする。embedded_color が使えない旧 Pillow も考慮する。
    """
    from PIL import Image, ImageDraw, ImageFont

    for size in (109, 128, 136, 160, 96, 64, 48, 40, 32):
        try:
            font = ImageFont.truetype(font_path, size=size)
        except Exception:
            continue  # このサイズでは開けない（次のサイズへ）
        canvas = size + 24
        img = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        use_color = True
        try:
            bbox = d.textbbox((0, 0), char, font=font, embedded_color=True)
        except TypeError:
            bbox = d.textbbox((0, 0), char, font=font)
            use_color = False
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox, oy = (canvas - w) // 2 - bbox[0], (canvas - h) // 2 - bbox[1]
        try:
            if use_color:
                d.text((ox, oy), char, font=font, embedded_color=True)
            else:
                d.text((ox, oy), char, font=font, fill=(0, 0, 0, 255))
        except Exception:
            continue
        rgb = img.convert("RGB").resize((target, target))
        # グリフが無いと真っ白になる。非白画素が極端に少なければ「描けていない」とみなす。
        if int((np.asarray(rgb.convert("L")) < 250).sum()) < 80:
            continue
        return rgb
    return None


# =====================================================================
# 絵文字コレクションの用意（優先順位: data/ → フォント → 合成）
# =====================================================================


def build_emoji_collection():
    """顔絵文字の集合を (images[RGB], labels[str], source[str]) で返す。

    優先順位:
      (1) data/45_sketch_emoji_search/emoji/*.png があればそれを採用
      (2) 絵文字フォントで顔絵文字を描画
      (3) どちらも不可なら合成絵文字（黄円＋表情ちがい）
    どの経路でも必ず完走する。グレースケール化は呼び出し側（索引構築時）で行う。
    """
    from PIL import Image

    # (1) ユーザ提供の絵文字画像
    edir = data_dir() / "emoji"
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if edir.is_dir():
        files = sorted(p for p in edir.iterdir() if p.suffix.lower() in exts)
        if files:
            imgs = [Image.open(p).convert("RGB").resize((CANVAS, CANVAS)) for p in files]
            labels = [p.stem for p in files]
            return imgs, labels, f"data/{MODULE_ID}/emoji/ の画像 {len(files)} 枚"

    # (2) 絵文字フォント描画
    font = _find_emoji_font()
    if font:
        imgs, labels = [], []
        for ch, name in EMOJI_CHARS:
            glyph = _render_emoji_glyph(ch, font)
            if glyph is not None:
                imgs.append(glyph)
                labels.append(name)
        if len(imgs) >= 8:  # 最低限の多様性が確保できたときだけ採用
            return imgs, labels, f"絵文字フォント描画（{pathlib.Path(font).name}）{len(imgs)} 種"

    # (3) 合成絵文字（フォントが無い/描けない環境のフォールバック）
    imgs = [draw_synthetic_emoji(s) for s in EXPRESSIONS]
    labels = [s[0] for s in EXPRESSIONS]
    return imgs, labels, f"合成絵文字 {len(imgs)} 種"


# =====================================================================
# スタイル変換（ドメインギャップ緩和の実験用）
# =====================================================================


def to_grayscale_rgb(img):
    """グレースケール化して 3ch に戻す（CLIP は 3ch 入力を期待するため）。"""
    import cv2
    from PIL import Image

    a = np.asarray(img.convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))


def to_edge_rgb(img):
    """Canny エッジで「白背景に黒線」の線画にする（スケッチ様式へ最も寄せる）。"""
    import cv2
    from PIL import Image

    a = np.asarray(img.convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 60, 160)  # エッジ=255(白), 背景=0(黒)
    line = 255 - edges  # 反転して「白地に黒線」へ
    return Image.fromarray(cv2.cvtColor(line, cv2.COLOR_GRAY2RGB))


def to_inverted_rgb(img):
    """明暗反転したグレースケール（白黒の極性をスケッチに合わせる実験用）。"""
    import cv2
    from PIL import Image

    g = np.asarray(img.convert("L"))
    inv = 255 - g
    return Image.fromarray(cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB))


def style_transform(img, style: str):
    """絵文字に施すスタイル変換を1か所に集約する。

    style: 'color'（無加工）/ 'gray'（既定の方針）/ 'edge'（線画化）/ 'invert'（反転）。
    """
    if style == "color":
        return img.convert("RGB")
    if style == "gray":
        return to_grayscale_rgb(img)
    if style == "edge":
        return to_edge_rgb(img)
    if style == "invert":
        return to_inverted_rgb(img)
    raise ValueError(f"未知のスタイル: {style}")


# =====================================================================
# スケッチ前処理（白背景・黒線・正方化・グレースケール）
# =====================================================================


def _square_pad(arr: np.ndarray, fill: int = 255) -> np.ndarray:
    """長辺に合わせて正方形に白パディングする（縦横比を保つ）。"""
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s), fill, dtype=arr.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = arr
    return canvas


def preprocess_sketch(img):
    """手書きスケッチを CLIP に入れる前の標準形に整える（PIL.Image, RGB)。

    手順（単一責務の積み重ね）:
      1. グレースケール化
      2. 背景が暗ければ反転（白背景・黒線に統一）
      3. 長辺基準で正方化（白パディング）
      4. CANVAS にリサイズして 3ch グレースケールで返す
    """
    import cv2
    from PIL import Image

    g = np.asarray(img.convert("L"))
    if g.mean() < 110:  # 背景が暗い＝黒地に白線で描かれた → 反転して白地に黒線へ
        g = 255 - g
    g = _square_pad(g, fill=255)
    g = cv2.resize(g, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))


# =====================================================================
# CLIP 埋め込み（取得不可なら画素記述子へフォールバック）
# =====================================================================


class EmojiEmbedder:
    """画像（絵文字・スケッチ）を同次元のベクトルに変換するエンコーダ。

    既定は openai/clip-vit-base-patch32 の画像エンコーダ。絵文字もスケッチも
    同じ get_image_features を通すので、共有空間でのコサイン類似度で検索できる。
    モデルを取得できない（オフライン等）場合は「画素記述子」へ自動フォールバックし、
    スクリプトが exit 0 で完走できるようにする（精度は落ちるが配線は確認できる）。

    重要(transformers v5): get_image_features の戻り値は BaseModelOutputWithPooling
    オブジェクトで、射影後ベクトルは .pooler_output に入っており、しかも L2 正規化
    されていない。検索の前に自分で正規化する（最頻出の落とし穴）。
    """

    MODEL_ID = CLIP_MODEL_ID

    def __init__(self, force_fallback: bool = False) -> None:
        self.backend = "fallback-pixels"
        self.dim = 24 * 24  # 画素記述子の次元
        self._model = None
        self._proc = None
        self._device = "cpu"
        self._torch = None
        if not force_fallback:
            self._try_load_clip()

    def _try_load_clip(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            self._device = pick_device()
            self._model = CLIPModel.from_pretrained(self.MODEL_ID).to(self._device).eval()
            self._proc = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.dim = int(self._model.config.projection_dim)  # 512
            self.backend = f"CLIP({self.MODEL_ID})"
            self._torch = torch
        except Exception as e:  # noqa: BLE001
            print(f"  [EmojiEmbedder] CLIP を使えません → 画素記述子にフォールバック: {type(e).__name__}")

    def encode_images(self, images) -> np.ndarray:
        """PIL 画像のリスト → (n, dim) の「未正規化」float32 ベクトル。"""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            return self._encode_clip(images)
        return self._encode_pixels(images)

    def _encode_clip(self, images) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():  # 推論は勾配を切る（CPUでも必須）
            inputs = self._proc(images=[im.convert("RGB") for im in images],
                                return_tensors="pt").to(self._device)
            out = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = out.pooler_output  # (n, 512) 未正規化
        return as_faiss_array(feats.cpu().numpy())

    @staticmethod
    def _encode_pixels(images) -> np.ndarray:
        """フォールバック: 24x24 グレースケールを平坦化した素朴な記述子。

        線画どうし（スケッチとエッジ化絵文字）は近く、塗り絵文字とは遠い、という
        ドメインギャップ自体は素朴に再現できるので、教材の配線確認には十分。
        """
        vecs = []
        for im in images:
            g = im.convert("L").resize((24, 24))
            vecs.append(np.asarray(g, dtype=np.float32).reshape(-1) / 255.0)
        return as_faiss_array(np.stack(vecs))


# =====================================================================
# FAISS 索引（コサイン＝正規化＋IndexFlatIP、IDMap で任意ID）
# =====================================================================


def build_index(vecs: np.ndarray, ids) -> faiss.Index:
    """未正規化ベクトルから IndexIDMap2(IndexFlatIP) を作って返す（add 済み）。

    コサイン類似度にするため内部で L2 正規化してから IP インデックスに add する。
    """
    vn = l2_normalize(vecs)
    d = vn.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
    index.add_with_ids(vn, np.asarray(ids, dtype=np.int64))
    return index


def save_index(index: faiss.Index, meta: dict, index_path: pathlib.Path, meta_path: pathlib.Path) -> None:
    """index 本体（.faiss）とメタデータ（.json）をセットで永続化する。

    write_index は index しか保存しない。ID→ラベル等のメタは別ファイルに保存し、
    必ずセットで整合させる（片方だけ更新すると ID から中身が引けなくなる）。
    """
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: pathlib.Path, meta_path: pathlib.Path):
    """保存した索引とメタデータを読み戻す（index, meta dict）。"""
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


def search_topn(index: faiss.Index, query_vec: np.ndarray, n: int):
    """クエリ（未正規化 1本 or 複数）で上位 n 件を引き、(scores, ids) を返す（1本想定）。

    返り値はそれぞれ長さ n の配列（1本クエリ前提で先頭行を返す）。
    """
    q = l2_normalize(query_vec)  # (1, d) になる
    scores, ids = index.search(as_faiss_array(q), n)
    return scores[0], ids[0]


# =====================================================================
# 可視化（matplotlib=Agg、図タイトルは ASCII）
# =====================================================================


def save_gallery(images, titles, path: pathlib.Path, ncol: int = 6, suptitle: str = "Emoji collection") -> None:
    """絵文字（またはスケッチ）コレクションをタイトル付きグリッドで保存する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.9))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (im, title) in enumerate(zip(images, titles)):
        axes[i].imshow(im, cmap="gray")
        axes[i].set_title(str(title), fontsize=8)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_result_panel(query_img, result_imgs, scores, labels, path: pathlib.Path,
                      title: str = "Sketch -> emoji search (left: query)") -> None:
    """左にクエリ・スケッチ、右に上位 N 件の絵文字（スコア＋ラベル）を並べて保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result_imgs)
    fig, axes = plt.subplots(1, n + 1, figsize=(1.9 * (n + 1), 2.5))
    axes = np.array(axes).reshape(-1)
    axes[0].imshow(query_img, cmap="gray")
    axes[0].set_title("query sketch", fontsize=9)
    axes[0].axis("off")
    for ax, im, sc, lab in zip(axes[1:], result_imgs, scores, labels):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"{lab}\ncos={sc:+.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
# スモークテスト（単体実行でも exit 0 を確認）
# =====================================================================

if __name__ == "__main__":
    out = output_dir()
    imgs, labels, source = build_emoji_collection()
    gray = [to_grayscale_rgb(im) for im in imgs]
    save_gallery(gray, labels, out / "00_smoke_gallery.png",
                 suptitle="smoke: emoji collection (grayscale)")  # 図タイトルは ASCII
    emb = EmojiEmbedder()
    feats = emb.encode_images(gray)
    index = build_index(feats, np.arange(len(gray)))
    sk = preprocess_sketch(make_synthetic_sketch())
    qf = emb.encode_images([sk])
    sc, ids = search_topn(index, qf, min(3, len(gray)))
    print(f"source={source}  n={len(gray)}  backend={emb.backend}  dim={emb.dim}")
    print(f"device={pick_device()}  top3 ids={ids.tolist()}  scores={[round(float(s), 3) for s in sc]}")
    print(f"saved={out / '00_smoke_gallery.png'}")
