"""第25回 共通の道具箱（VQA / 軽量VLM / グラウンディング）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務はだいたい次の5つに絞ってあります（単一責務・最小限）:

  1. device の自動判定（cuda > mps > cpu。本講座は cpu 前提）
  2. 合成シーン画像の生成（色 × 形を既知の位置に配置 → 正解付きの VQA を作れる）
  3. 小さな VQA ベンチマーク（画像・質問・正解・人間10人の回答・グラウンディング箱）の構築
  4. 評価の核となる純計算（VQAv2 流の answer 正規化 / accuracy / IoU / 点-in-箱）
  5. モデルのロード（ViLT / BLIP-VQA / SmolVLM2）。失敗しても呼び出し側が握りつぶせるよう
     例外メッセージを分かりやすくして投げる（各スクリプトは exit 0 を守る）

なぜ合成画像か: VQA/VLM の評価で大事なのは「正解（ground truth）が手元にあること」。
色×形を決まった座標に描けば、「左の図形は何色か」「図形は何個か」の正解を
コードで自動生成できる。データセットを大量DLせずに評価の仕組みを最後まで作り切れる。
（実画像で試したい人は data/25_vqa_vlm/ に画像を置けば load_user_images() が拾う。）

ネットに出るのは Hugging Face からのモデル重みDLのみ。画像はすべてローカル生成。
"""

from __future__ import annotations

import os
import pathlib
import re

import cv2
import numpy as np
import torch
from PIL import Image

# --- HF / tokenizers のログや進捗バーを静かにして、教材の出力を読みやすくする ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # GUI の無い環境（Docker/CI）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# 出力は規約どおり lectures/<module_id>/outputs/ に集約する。
MODULE_ID = "25_vqa_vlm"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID

# --- CPU で現実的な VQA / VLM モデル（衝突する重い依存を増やさない選択） ---
# ViLT: 「3129語の固定回答」から選ぶ分類型 VQA（生成しない＝速い・決定的）。
VILT_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
# BLIP-VQA: 短い回答を「生成」する Encoder-Decoder 型 VQA。
BLIP_VQA_MODEL_ID = "Salesforce/blip-vqa-base"
# SmolVLM2-256M: チャット形式（apply_chat_template）で動く超小型 VLM。命令に従える。
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Instruct"

# 合成シーンで使う色（PIL/numpy は RGB。cv2 の描画に (R,G,B) をそのまま渡し BGR 混乱を避ける）。
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (240, 205, 30),
}
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")


# =====================================================================
# 1) device / 出力ディレクトリ
# =====================================================================
def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提。手元に GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず .to(device) で揃える。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """lectures/<module_id>/outputs/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# =====================================================================
# 2) 合成シーン生成（正解つき）
# =====================================================================
def _draw_shape(
    img: np.ndarray, shape: str, box: tuple[int, int, int, int], color: tuple[int, int, int]
) -> None:
    """box=(x1,y1,x2,y2) の矩形に内接する形を、塗りつぶしで描く（in-place）。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = min(x2 - x1, y2 - y1) // 2
    if shape == "circle":
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
    elif shape == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness=-1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    else:
        raise ValueError(f"未知の形: {shape}")


def make_scene(
    objects: list[tuple[str, str, tuple[int, int, int, int]]], size: int = 384, bg: int = 245
) -> Image.Image:
    """オブジェクト列 [(color, shape, box), ...] から合成シーン（PIL.Image, RGB）を作る。

    box は (x1,y1,x2,y2) の絶対座標。背景は明るい灰色。cv2 で描くが配列は
    最初から RGB として扱う（imread/imwrite を通さないので BGR 変換は不要）。
    """
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    for color, shape, box in objects:
        _draw_shape(img, shape, box, COLORS[color])
    return Image.fromarray(img)  # RGB のまま PIL へ


def load_user_images() -> list[tuple[str, Image.Image]] | None:
    """data/25_vqa_vlm/ に実画像があれば [(name, PIL), ...] を返す。無ければ None。"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not DATA_DIR.is_dir():
        return None
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
    if not files:
        return None
    return [(p.stem, Image.open(p).convert("RGB")) for p in files]


# =====================================================================
# 3) 小さな VQA ベンチマーク（画像・質問・正解・人間10人の回答・GT箱）
# =====================================================================
def build_benchmark() -> list[dict]:
    """評価用の小さな VQA データセットを返す（決定的・ネット不要）。

    各要素 dict:
      scene_id      : シーン名
      image         : PIL.Image（合成）
      question      : 質問文（英語。VQAモデルは英語で学習されている）
      gt_answer     : 設計上の正解（1語）
      human_answers : 人間10人ぶんの回答（VQAv2 流。表記ゆれを含む）
      target        : グラウンディング対象の (色, 形)（無ければ None）
      gt_box        : target の正解バウンディングボックス (x1,y1,x2,y2)（無ければ None）

    人間回答は「赤」を crimson/maroon と言う人がいる等、現実の表記ゆれを模す。
    これにより VQAv2 accuracy = min(一致数/3, 1) の意味が体感できる。
    """
    s = 384
    # シーンA: 左に赤い円、右に青い四角。
    scene_a_objs = [
        ("red", "circle", (40, 150, 170, 280)),
        ("blue", "square", (220, 150, 340, 270)),
    ]
    img_a = make_scene(scene_a_objs, s)
    # シーンB: 緑の三角・黄の円・赤の四角（3個）。
    scene_b_objs = [
        ("green", "triangle", (30, 60, 150, 180)),
        ("yellow", "circle", (160, 200, 270, 310)),
        ("red", "square", (280, 60, 360, 140)),
    ]
    img_b = make_scene(scene_b_objs, s)
    # シーンC: 大きな赤い円ひとつ。
    scene_c_objs = [("red", "circle", (90, 90, 300, 300))]
    img_c = make_scene(scene_c_objs, s)

    bench: list[dict] = [
        {
            "scene_id": "A",
            "image": img_a,
            "question": "What color is the circle?",
            "gt_answer": "red",
            "human_answers": ["red"] * 7 + ["crimson", "maroon", "red"],
            "target": ("red", "circle"),
            "gt_box": (40, 150, 170, 280),
        },
        {
            "scene_id": "A",
            "image": img_a,
            "question": "How many shapes are in the image?",
            "gt_answer": "2",
            "human_answers": ["2"] * 8 + ["two", "2"],
            "target": None,
            "gt_box": None,
        },
        {
            "scene_id": "A",
            "image": img_a,
            "question": "Is there a square in the image?",
            "gt_answer": "yes",
            "human_answers": ["yes"] * 9 + ["yep"],
            "target": ("blue", "square"),
            "gt_box": (220, 150, 340, 270),
        },
        {
            "scene_id": "B",
            "image": img_b,
            "question": "How many shapes are in the image?",
            "gt_answer": "3",
            "human_answers": ["3"] * 7 + ["three", "3", "3"],
            "target": None,
            "gt_box": None,
        },
        {
            "scene_id": "B",
            "image": img_b,
            "question": "What color is the triangle?",
            "gt_answer": "green",
            "human_answers": ["green"] * 8 + ["green", "olive"],
            "target": ("green", "triangle"),
            "gt_box": (30, 60, 150, 180),
        },
        {
            "scene_id": "C",
            "image": img_c,
            "question": "What color is the circle?",
            "gt_answer": "red",
            "human_answers": ["red"] * 9 + ["red"],
            "target": ("red", "circle"),
            "gt_box": (90, 90, 300, 300),
        },
    ]
    return bench


# =====================================================================
# 4) 評価の純計算（VQAv2 流の正規化 / accuracy / IoU / 点-in-箱）
# =====================================================================
# VQAv2 公式評価を簡略化したもの（教材として要点を残す）。
_ARTICLES = {"a", "an", "the"}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_CONTRACTIONS = {"dont": "do not", "isnt": "is not", "arent": "are not", "cant": "can not"}


def normalize_answer(ans: str) -> str:
    """VQAv2 流に回答文字列を正規化する（簡略版）。

    手順: 小文字化 → 余分な空白除去 → 句読点除去 → 数詞を数字へ → 冠詞(a/an/the)除去。
    これで "Red." と "red" と "a red" を同一視できる。公式評価はさらに細かい
    （小数点の保持・コンマ処理など）が、教材では本質だけを残す。
    """
    text = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)  # 句読点・記号を空白へ
    words = []
    for w in text.split():
        w = _CONTRACTIONS.get(w, w)
        for token in w.split():  # 展開した縮約が複数語になる場合に対応
            token = _NUMBER_MAP.get(token, token)
            if token in _ARTICLES:
                continue
            words.append(token)
    return " ".join(words)


def vqa_accuracy_simple(pred: str, human_answers: list[str]) -> float:
    """簡易版 VQA accuracy = min(予測と一致した人間の数 / 3, 1)。

    3人以上が同じ答えなら満点。両者とも normalize_answer してから比較する。
    """
    p = normalize_answer(pred)
    agree = sum(1 for h in human_answers if normalize_answer(h) == p)
    return min(agree / 3.0, 1.0)


def vqa_accuracy_vqav2(pred: str, human_answers: list[str]) -> float:
    """公式 VQAv2 accuracy（10人の leave-one-out 平均）。

    各人を1人ずつ「抜いた」残り9人に対して min(一致数/3, 1) を求め、10通りを平均する。
    これにより「自分が答えた人」を二重に数えないようにしている。簡易版より少しだけ厳しい。
    """
    p = normalize_answer(pred)
    hn = [normalize_answer(h) for h in human_answers]
    n = len(hn)
    if n == 0:
        return 0.0
    accs = []
    for i in range(n):
        others = hn[:i] + hn[i + 1 :]
        agree = sum(1 for h in others if h == p)
        accs.append(min(agree / 3.0, 1.0))
    return sum(accs) / n


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """2つの (x1,y1,x2,y2) 箱の IoU（Intersection over Union）を返す。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(pt: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    """点 (x,y) が箱 (x1,y1,x2,y2) の内側（境界含む）にあるかを返す。"""
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def build_chat_messages(image: Image.Image, question: str) -> list[dict]:
    """チャット VLM 用のメッセージ（画像を content に埋め込む正準形）を作る。

    ★最重要: 画像は別引数ではなく content の中に {"type":"image","image":PIL} として入れる。
    transformers v5 の apply_chat_template はこの形を受け取って <image> トークンへ展開する。
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


# =====================================================================
# 5) モデルのロード（失敗時は分かりやすい例外を投げる。呼び出し側が握りつぶす）
# =====================================================================
def load_vilt(device: torch.device):
    """ViLT VQA（分類型）の (processor, model) を返す。eval + device 済み。"""
    from transformers import ViltForQuestionAnswering, ViltProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = ViltProcessor.from_pretrained(VILT_MODEL_ID)
    model = ViltForQuestionAnswering.from_pretrained(VILT_MODEL_ID).to(device).eval()
    return processor, model


def load_blip_vqa(device: torch.device):
    """BLIP-VQA（生成型）の (processor, model) を返す。eval + device 済み。"""
    from transformers import BlipForQuestionAnswering, BlipProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(BLIP_VQA_MODEL_ID).to(device).eval()
    return processor, model


def load_smolvlm(device: torch.device):
    """SmolVLM2-256M（チャット型 VLM）の (processor, model) を返す。

    注意: SmolVLM のプロセッサは num2words に依存する。未導入なら ImportError を投げるので
    呼び出し側で握りつぶし、`uv add num2words` を案内すること（本リポの hf 群には含めていない）。
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(SMOLVLM_MODEL_ID, dtype=torch.float32)
    model = model.to(device).eval()
    return processor, model


def hint_for_model_error(err: Exception) -> str:
    """モデルロード/推論の失敗理由から、学習者向けの一言ヒントを作る。"""
    msg = str(err).lower()
    if "num2words" in msg:
        return "SmolVLM のプロセッサに num2words が必要です → `uv add num2words` を実行してください。"
    if "connection" in msg or "offline" in msg or "resolve" in msg or "max retries" in msg:
        return "ネットに出られないようです。初回はモデルDLのためにネット接続が必要です。"
    return "モデルのロード/推論に失敗しました（メモリ不足や一時的なDL失敗の可能性）。"


def save_qa_panel(
    image: Image.Image, lines: list[str], path: pathlib.Path, title: str = ""
) -> None:
    """画像の横に Q&A テキストを並べた1枚絵を保存する（結果の目視確認用）。"""
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax_img.imshow(image)  # PIL は RGB なので matplotlib にそのまま渡せる
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=11)
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する（モデルDL不要の純計算）。
    out = ensure_output_dir()
    bench = build_benchmark()
    print(f"device={pick_device()}  benchmark items={len(bench)}")
    # 正規化と accuracy の動作確認。
    print("normalize('a Red.') =", repr(normalize_answer("a Red.")))
    print("simple acc(red, 7xred...) =", vqa_accuracy_simple("red", bench[0]["human_answers"]))
    print("vqav2 acc(red, 7xred...) =", round(vqa_accuracy_vqav2("red", bench[0]["human_answers"]), 3))
    # ベンチの先頭シーンを保存。
    bench[0]["image"].save(out / "00_scene_example.png")
    print(f"saved: {out/'00_scene_example.png'}")
