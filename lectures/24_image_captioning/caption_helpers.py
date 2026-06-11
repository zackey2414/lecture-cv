"""第24回 共通の道具箱（画像キャプション生成）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務は次の4つに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。本講座の既定は cpu）
  2. 合成シーン画像の生成（色・形でそれっぽい風景を描く。ネット不要・決定的）
     ＋ 各シーンの「参照キャプション」（BLEU/CIDEr の正解として使う）
  3. キャプションモデルのロードと推論（BLIP / GIT / ViT-GPT2 を1つの窓口で）
  4. 出力ディレクトリの用意と matplotlib(Agg) の共通設定

★transformers v5 の要点（落とし穴）:
  - キャプションモデルの正準 Auto クラスは `AutoModelForImageTextToText`。
    旧 `AutoModelForVision2Seq` は v5 で削除された（import するとエラー）。
  - BLIP / GIT は `AutoProcessor` で「画像＋テキストをまとめて」前処理できる。
  - ところが ViT-GPT2 の `AutoProcessor` は中身が GPT2Tokenizer だけで、
    画像を渡せない。よって ViT-GPT2 は `AutoImageProcessor` ＋ `AutoTokenizer` を
    別々に使う（VisionEncoderDecoder の作法）。この差をこの道具箱で吸収する。
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field

# --- HuggingFace / トークナイザのログや進捗バーを静かにする（教材の出力を読みやすく） ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2  # noqa: E402  図形描画ツールとして使う（imshow は呼ばない＝headless 安全）
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# 結果は outputs/<module_id>/ に集約する（プロジェクト規約）。
MODULE_ID = "24_image_captioning"
OUTPUT_DIR = pathlib.Path("outputs") / MODULE_ID

# --- CPU で現実的な小型キャプションモデル（既定） ---------------------------------
# kind="combined": AutoProcessor が画像＋テキストをまとめて処理できる（BLIP / GIT）
# kind="encdec":   ViT-GPT2 のように ImageProcessor と Tokenizer を分けて使う
MODELS: dict[str, dict[str, str]] = {
    "blip": {"id": "Salesforce/blip-image-captioning-base", "kind": "combined"},
    "git": {"id": "microsoft/git-base", "kind": "combined"},
    "vitgpt2": {"id": "nlpconnect/vit-gpt2-image-captioning", "kind": "encdec"},
}
# 参照不要メトリクス CLIPScore 用の小型 CLIP（module 16 と同じ・キャッシュ共有）。
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず両方 .to(device) で揃える。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir() -> pathlib.Path:
    """outputs/<module_id>/ を作って返す（既にあってもエラーにしない）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# =====================================================================
# 1) 合成シーン画像 ＋ 参照キャプション
# =====================================================================
# なぜ合成画像か:
#   ネットへ出るのは「モデル重みの DL」だけに限りたい（再現性・オフライン耐性）。
#   そこで cv2 で色と形を描き、「夕焼け」「赤い車」「木のある草原」のような
#   素朴な風景を作る。各シーンには人間が書いた参照キャプションを添えておき、
#   BLEU / ROUGE / CIDEr の「正解」として使う。
#   （モデルの出力は汎用的になりがちなので、メトリクスの値そのものより
#     "どう計算し・パラメータでどう動くか" を学ぶことに主眼を置く。）


def _draw_sunset(size: int) -> np.ndarray:
    """夕焼けの海: 上=オレンジ空 + 黄色い太陽、下=青い海。RGB の uint8 配列。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # 空（縦グラデーション: 上はオレンジ、地平線付近は黄みがかる）
    for y in range(size):
        t = y / size
        img[y, :] = (255, int(120 + 120 * t), int(40 + 60 * t))
    # 海（下半分を青で塗る）
    sea_top = int(size * 0.62)
    img[sea_top:, :] = (30, 80, 170)
    # 太陽（黄色い円）
    cv2.circle(img, (int(size * 0.5), int(size * 0.5)), int(size * 0.12), (255, 230, 90), -1)
    return img


def _draw_red_car(size: int) -> np.ndarray:
    """赤い車が道路にいる風景。"""
    img = np.full((size, size, 3), (200, 215, 235), dtype=np.uint8)  # 薄い空
    road_top = int(size * 0.62)
    img[road_top:, :] = (90, 90, 95)  # 灰色の道路
    # 車体（赤い長方形）と屋根（一回り小さい赤い長方形）
    x0, y0, x1, y1 = int(size * 0.22), int(size * 0.42), int(size * 0.78), int(size * 0.66)
    cv2.rectangle(img, (x0, y0), (x1, y1), (210, 40, 40), -1)
    cv2.rectangle(
        img,
        (int(size * 0.30), int(size * 0.34)),
        (int(size * 0.66), int(size * 0.46)),
        (210, 40, 40),
        -1,
    )
    # 窓
    cv2.rectangle(
        img,
        (int(size * 0.34), int(size * 0.36)),
        (int(size * 0.62), int(size * 0.45)),
        (150, 200, 230),
        -1,
    )
    # タイヤ（黒い円2つ）
    cv2.circle(img, (int(size * 0.34), y1), int(size * 0.06), (20, 20, 20), -1)
    cv2.circle(img, (int(size * 0.66), y1), int(size * 0.06), (20, 20, 20), -1)
    return img


def _draw_tree(size: int) -> np.ndarray:
    """青空の下、草原に立つ1本の木。"""
    img = np.full((size, size, 3), (140, 200, 245), dtype=np.uint8)  # 青空
    grass_top = int(size * 0.60)
    img[grass_top:, :] = (70, 170, 70)  # 緑の草原
    # 幹（茶色の長方形）
    cx = int(size * 0.5)
    cv2.rectangle(
        img,
        (cx - int(size * 0.03), int(size * 0.45)),
        (cx + int(size * 0.03), grass_top + 5),
        (110, 70, 30),
        -1,
    )
    # 樹冠（緑の円）
    cv2.circle(img, (cx, int(size * 0.38)), int(size * 0.17), (40, 140, 50), -1)
    return img


# シーン名 -> (描画関数, 参照キャプションのリスト)
SCENES: dict[str, tuple] = {
    "sunset": (
        _draw_sunset,
        [
            "a sunset over the ocean",
            "the sun is setting over the sea",
            "an orange sky above the blue water",
        ],
    ),
    "red_car": (
        _draw_red_car,
        [
            "a red car on the road",
            "a red vehicle parked on a gray street",
            "a small red car on a road",
        ],
    ),
    "tree": (
        _draw_tree,
        [
            "a green tree in a grassy field",
            "a tree standing on green grass under a blue sky",
            "a single tree in a field",
        ],
    ),
}


def make_scene(name: str, size: int = 384) -> Image.Image:
    """シーン名から合成画像（PIL.Image, RGB）を1枚作る。

    cv2 で描画するが配列は最初から RGB として扱い、cv2.imread/imwrite を
    経由しないので「BGR→RGB 変換」は不要（色タプルを (R,G,B) で渡している）。
    """
    if name not in SCENES:
        raise ValueError(f"未知のシーン: {name}（候補: {list(SCENES)}）")
    draw_fn, _ = SCENES[name]
    arr = draw_fn(size)
    return Image.fromarray(arr)  # RGB のまま PIL へ


def build_gallery(size: int = 384) -> tuple[list[Image.Image], list[str], list[list[str]]]:
    """全シーンの (画像, 名前, 参照キャプション群) を返す。

    返り値:
      images:     PIL.Image のリスト
      names:      シーン名のリスト（images と同じ並び）
      references: 各画像の参照キャプションのリスト（BLEU/CIDEr の正解）
    """
    images, names, references = [], [], []
    for name, (_, refs) in SCENES.items():
        images.append(make_scene(name, size))
        names.append(name)
        references.append(list(refs))
    return images, names, references


def load_user_or_synthetic(size: int = 384) -> tuple[list[Image.Image], list[str], list[list[str]]]:
    """実画像があればそれを、無ければ合成ギャラリーを返す（実画像への導線）。

    data/24_image_captioning/ に .png/.jpg を置くと、それを読み込んで使う
    （参照キャプションは無いので references は空リストになる＝CLIPScore のみ可）。
    無ければ合成シーンにフォールバックするので、引数なしでも必ず動く。
    """
    data_dir = pathlib.Path("data") / MODULE_ID
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if data_dir.is_dir():
        files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in exts)
        if files:
            images = [Image.open(p).convert("RGB") for p in files]
            return images, [p.stem for p in files], [[] for _ in files]
    return build_gallery(size)


# =====================================================================
# 2) キャプションモデルのロードと推論（BLIP / GIT / ViT-GPT2 を統一窓口で）
# =====================================================================


@dataclass
class Captioner:
    """キャプションモデルの「持ち手」。kind に応じて中身が変わる。

    - combined（BLIP/GIT）: processor 1つで画像＋テキストを処理。
    - encdec（ViT-GPT2）:   image_processor と tokenizer を別々に持つ。
    どちらでも caption() / decode() の使い勝手は同じになるよう吸収する。
    """

    key: str
    kind: str
    model: object
    device: torch.device
    processor: object = None  # combined 用
    image_processor: object = None  # encdec 用
    tokenizer: object = None  # encdec 用
    meta: dict = field(default_factory=dict)


def load_captioner(key: str, device: torch.device | None = None) -> Captioner:
    """キャプションモデルを key（"blip"/"git"/"vitgpt2"）でロードする。

    すべて `AutoModelForImageTextToText` で読み込めるのが v5 の利点（旧
    `AutoModelForVision2Seq` は廃止）。前処理側だけ kind で出し分ける。
    モデルは eval() ＋ device 済みで返す。
    """
    if key not in MODELS:
        raise ValueError(f"未知のモデル key: {key}（候補: {list(MODELS)}）")
    if device is None:
        device = pick_device()

    from transformers import AutoModelForImageTextToText
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    spec = MODELS[key]
    model = AutoModelForImageTextToText.from_pretrained(spec["id"]).to(device).eval()

    if spec["kind"] == "combined":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(spec["id"])
        return Captioner(
            key=key,
            kind="combined",
            model=model,
            device=device,
            processor=processor,
            meta={"id": spec["id"]},
        )
    # encdec（ViT-GPT2）: 画像と文字を別プロセッサで扱う
    from transformers import AutoImageProcessor, AutoTokenizer

    image_processor = AutoImageProcessor.from_pretrained(spec["id"])
    tokenizer = AutoTokenizer.from_pretrained(spec["id"])
    return Captioner(
        key=key,
        kind="encdec",
        model=model,
        device=device,
        image_processor=image_processor,
        tokenizer=tokenizer,
        meta={"id": spec["id"]},
    )


def caption_images(
    cap: Captioner,
    images: list[Image.Image],
    text: str | None = None,
    **gen_kwargs,
) -> list[str]:
    """画像のリストにキャプションを付けて返す（バッチ推論 + batch_decode）。

    - text を与えると「条件付きキャプション」（接頭辞）になる（BLIP/GIT のみ有効。
      ViT-GPT2 は条件付けに対応しないので text は無視される）。
    - gen_kwargs は model.generate にそのまま渡す（num_beams / max_new_tokens /
      do_sample / temperature / repetition_penalty など）。
    - 推論は必ず eval()（ロード時に設定済み）＋ inference_mode で行う。
    """
    gen_kwargs.setdefault("max_new_tokens", 20)

    if cap.kind == "combined":
        if text:
            inputs = cap.processor(
                images=images, text=[text] * len(images), return_tensors="pt", padding=True
            )
        else:
            inputs = cap.processor(images=images, return_tensors="pt")
        inputs = inputs.to(cap.device)
        with torch.inference_mode():
            out = cap.model.generate(**inputs, **gen_kwargs)
        return [t.strip() for t in cap.processor.batch_decode(out, skip_special_tokens=True)]

    # encdec（ViT-GPT2）
    pixel_values = cap.image_processor(images=images, return_tensors="pt").pixel_values.to(
        cap.device
    )
    with torch.inference_mode():
        out = cap.model.generate(pixel_values, **gen_kwargs)
    return [t.strip() for t in cap.tokenizer.batch_decode(out, skip_special_tokens=True)]


# =====================================================================
# 3) 可視化ユーティリティ
# =====================================================================


def save_caption_grid(
    images: list[Image.Image],
    captions: list[str],
    path: pathlib.Path,
    title: str = "",
    ncol: int = 3,
) -> None:
    """画像とその生成キャプションをグリッドで保存する（結果確認用）。"""
    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (im, cap_text) in enumerate(zip(images, captions)):
        axes[i].imshow(im)  # PIL は RGB なので matplotlib にそのまま渡せる
        wrapped = "\n".join(_wrap(cap_text, width=28))
        axes[i].set_title(wrapped, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else (0, 0, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _wrap(text: str, width: int) -> list[str]:
    """簡易な単語折り返し（タイトル表示用、外部依存なし）。"""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する（モデル DL は無し）。
    out = ensure_output_dir()
    dev = pick_device()
    images, names, refs = build_gallery()
    save_caption_grid(images, names, out / "00_gallery.png", title="synthetic scenes (24)")
    print(f"device={dev}  scenes={names}")
    print(f"saved: {out / '00_gallery.png'}")
