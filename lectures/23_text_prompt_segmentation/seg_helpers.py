"""第23回 共通の道具箱（テキストプロンプト/参照セグメンテーション）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務を次の5つに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。本講座は cpu 前提）
  2. 合成シーン（複数オブジェクト）の生成 ＋ 各オブジェクトの正解マスク(GT)
  3. CLIPSeg / SAM / Grounding DINO のロード（eval + device 済み）
  4. 参照セグメの評価指標（IoU / Dice）を numpy で計算
  5. マスク可視化（オーバレイ・ヒートマップ）と出力ディレクトリの用意

なぜ合成シーンか:
  CLIPSeg は PhraseCut 等の「文で領域を指す」参照セグメで学習されており、
  色＋形（"a red circle"）のような単純概念は合成画像でも反応します。GT マスクは
  自分で描いたので画素単位で厳密に分かり、IoU/Dice の定義確認や閾値スイープに最適です。
  ただし合成画像は実写よりテクスチャが乏しく、Grounding DINO の検出が振るわない
  こともあります（その場合でもスクリプトは必ず exit 0 になるよう実装してあります）。
  実写で試したい人は data/23_text_prompt_segmentation/ に画像を置いてください。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# --- transformers / huggingface_hub のログを静かにする（教材の出力を読みやすく） ---
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# プロジェクト規約: 結果は outputs/<module_id>/ に集約する。
MODULE_ID = "23_text_prompt_segmentation"
OUTPUT_DIR = pathlib.Path("outputs") / MODULE_ID

# 使用モデル（いずれも CPU で現実的な小型）。
CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"  # テキスト条件付きセグメ（軽量）
SAM_MODEL_ID = "facebook/sam-vit-base"  # プロンプト型セグメ（点/ボックス）
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # オープン語彙の box 検出

# 合成シーンに置くオブジェクト。color は (R, G, B)。prompt は CLIPSeg/GDINO に渡す参照文。
# cv2 で描くが配列は最初から RGB として解釈する（imread/imwrite を通さないので BGR 混乱なし）。
_SCENE = [
    {"name": "red circle", "prompt": "a red circle", "color": (215, 40, 40)},
    {"name": "blue square", "prompt": "a blue square", "color": (40, 70, 210)},
    {"name": "green triangle", "prompt": "a green triangle", "color": (40, 170, 70)},
]
SCENE_SIZE = 384  # 正方画像の一辺


def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、GPU/Apple Silicon があれば自動で使う。
    重要: モデルと入力テンソルの device は必ず揃える（.to(device) を両方に）。
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


def _draw_shape(canvas: np.ndarray, name: str, color: tuple[int, int, int]) -> None:
    """canvas（HxWx3 の RGB 配列）に1つの図形を塗りつぶしで描く（内部関数）。

    座標は SCENE_SIZE=384 を基準にした固定配置。各図形は重ならない位置に置く
    （参照セグメは「どの領域か」を文で一意に指したいので、被らせない）。
    """
    s = SCENE_SIZE
    if name == "red circle":
        cv2.circle(canvas, (int(s * 0.28), int(s * 0.32)), int(s * 0.16), color, thickness=-1)
    elif name == "blue square":
        x0, y0 = int(s * 0.56), int(s * 0.14)
        x1, y1 = int(s * 0.88), int(s * 0.46)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness=-1)
    elif name == "green triangle":
        pts = np.array(
            [
                [int(s * 0.50), int(s * 0.56)],
                [int(s * 0.34), int(s * 0.88)],
                [int(s * 0.66), int(s * 0.88)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [pts], color)
    else:
        raise ValueError(f"未知の図形: {name}")


def build_scene() -> tuple[Image.Image, dict[str, np.ndarray], list[dict]]:
    """合成シーン（複数オブジェクト）と各オブジェクトの正解マスク(GT)を作る。

    返り値:
      image: PIL.Image（RGB, SCENE_SIZE×SCENE_SIZE）
      gt:    {オブジェクト名: bool マスク (H, W)} の辞書（画素単位の厳密な正解）
      scene: [{name, prompt, color}, ...]（プロンプトと正解名の対応表）

    GT は「各図形だけを別キャンバスに描いて画素>0 を取る」ことで厳密に得る。
    こうして得た GT があるから、CLIPSeg のしきい値スイープで IoU 最大点を探せる。
    """
    bg = 235  # 明るい灰色の背景
    img = np.full((SCENE_SIZE, SCENE_SIZE, 3), bg, dtype=np.uint8)
    gt: dict[str, np.ndarray] = {}
    for obj in _SCENE:
        # 1) 本番キャンバスに描画
        _draw_shape(img, obj["name"], obj["color"])
        # 2) GT 用に「その図形だけ」を黒地に描いてマスク化
        solo = np.zeros((SCENE_SIZE, SCENE_SIZE, 3), dtype=np.uint8)
        _draw_shape(solo, obj["name"], (255, 255, 255))
        gt[obj["name"]] = solo[..., 0] > 127  # bool マスク (H, W)
    return Image.fromarray(img), gt, [dict(o) for o in _SCENE]


def load_user_or_scene() -> tuple[Image.Image, dict[str, np.ndarray] | None, list[dict]]:
    """実画像があればそれを、無ければ合成シーンを返す（実画像への導線）。

    data/23_text_prompt_segmentation/ に image.png を置くと、それを使う。GT が無い
    場合は gt=None を返し、評価系は IoU を出さず可視化のみに切り替える。プロンプトは
    data/prompts.txt（1行1プロンプト）を読む。無ければ合成シーンへフォールバック
    するので、引数なしでも必ず動く。
    """
    data_dir = pathlib.Path("data") / MODULE_ID
    img_path = data_dir / "image.png"
    if img_path.is_file():
        image = Image.open(img_path).convert("RGB")
        prompts_file = data_dir / "prompts.txt"
        if prompts_file.is_file():
            prompts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
        else:
            prompts = ["the main object"]
        scene = [{"name": p, "prompt": p, "color": (200, 60, 60)} for p in prompts]
        return image, None, scene  # 実画像は GT なし（評価は可視化のみ）
    return build_scene()


# ---------------------------------------------------------------------------
# モデルのロード（いずれも eval + device 済みで返す）
# ---------------------------------------------------------------------------
def load_clipseg(device: torch.device):
    """CLIPSeg（テキスト条件付きセグメ）をロードして返す。

    transformers v5 の正準: 専用クラス CLIPSegProcessor / CLIPSegForImageSegmentation。
    出力 outputs.logits は (プロンプト数, 352, 352) のロジット（後段で sigmoid+閾値）。
    """
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_ID).to(device).eval()
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_ID)
    return model, processor


def load_sam(device: torch.device):
    """SAM（プロンプト型セグメ）をロードして返す。点/ボックスのプロンプトを受ける。"""
    from transformers import SamModel, SamProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = SamModel.from_pretrained(SAM_MODEL_ID).to(device).eval()
    processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    return model, processor


def load_gdino(device: torch.device):
    """Grounding DINO（オープン語彙の box 検出）をロードして返す。"""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    return model, processor


# ---------------------------------------------------------------------------
# CLIPSeg の中核: ロジット → 原寸 → sigmoid → 確率マップ
# ---------------------------------------------------------------------------
def clipseg_probs(model, processor, image: Image.Image, prompts: list[str], device) -> np.ndarray:
    """CLIPSeg で各プロンプトの「確率マップ」(P, H, W) を返す（0〜1, 原寸）。

    手順（参照セグメの核）:
      1. processor で画像＋複数プロンプトを前処理（padding=True で長さを揃える）。
      2. model(**inputs).logits は (P, 352, 352) のロジット。
      3. 原寸 (H, W) へ bilinear 補間してから sigmoid（順序は逆でも実質同じだが、
         先に補間→sigmoid の方が境界が滑らか）。
    返り値は numpy の float32 (P, H, W)。閾値処理は呼び出し側で行う（単一責務）。
    """
    w, h = image.size  # PIL は (W, H)
    inputs = processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits  # (P, 352, 352)
    if logits.ndim == 2:  # 念のため: 単一プロンプトで (352,352) が返る版への保険
        logits = logits.unsqueeze(0)
    # (P,352,352) -> (P,1,352,352) にして原寸へ補間 -> (P,H,W)
    up = torch.nn.functional.interpolate(
        logits.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(1)
    probs = torch.sigmoid(up).cpu().numpy().astype(np.float32)
    return probs


# ---------------------------------------------------------------------------
# 評価指標（参照セグメは前景マスクどうしの一致を見る）
# ---------------------------------------------------------------------------
def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """2値マスクの IoU = |∩| / |∪|。両方空なら 1.0（完全一致扱い）。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """2値マスクの Dice = 2|∩| / (|P|+|G|)（= F1）。両方空なら 1.0。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float(np.logical_and(p, g).sum()) / float(denom)


def best_threshold(prob: np.ndarray, gt: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """確率マップ prob を thresholds で2値化し、IoU 最大のしきい値と IoU を返す。

    返り値: (最良しきい値, そのときの IoU)。参照セグメで「どこで切るか」を学ぶための関数。
    """
    best_t, best_v = float(thresholds[0]), -1.0
    for t in thresholds:
        v = mask_iou(prob >= t, gt)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t, best_v


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """画像にマスク領域を半透明色で重ねた RGB 配列 (H, W, 3) を返す。"""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    m = mask.astype(bool)
    for c in range(3):
        overlay[..., c][m] = (1 - alpha) * base[..., c][m] + alpha * color[c]
    return overlay.clip(0, 255).astype(np.uint8)


def save_panel(image: Image.Image, prob_maps, prompts, masks, path, gt=None) -> None:
    """プロンプトごとに [元画像 / 確率ヒートマップ / マスクオーバレイ] を縦に並べて保存。

    gt（{name: mask}）を渡すと各行の IoU をタイトルに添える。可視化専用（評価値は別関数）。
    """
    n = len(prompts)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.0 * n))
    axes = np.array(axes).reshape(n, 3)
    for r, prompt in enumerate(prompts):
        axes[r, 0].imshow(image)
        axes[r, 0].set_title("input", fontsize=9)
        im = axes[r, 1].imshow(prob_maps[r], cmap="jet", vmin=0, vmax=1)
        title = f"CLIPSeg prob: '{prompt}'"
        axes[r, 1].set_title(title, fontsize=9)
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)
        ov = overlay_mask(image, masks[r])
        sub = "mask overlay"
        if gt is not None and prompt in gt:
            sub += f"  IoU={mask_iou(masks[r], gt[prompt]):.3f}"
        axes[r, 2].imshow(ov)
        axes[r, 2].set_title(sub, fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になること＋シーン図の保存を確認する。
    out = ensure_output_dir()
    dev = pick_device()
    image, gt, scene = build_scene()
    # シーンと GT マスクを並べて保存（モデル不要・即終了）。
    names = [o["name"] for o in scene]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.0 * (len(names) + 1), 3.0))
    axes[0].imshow(image)
    axes[0].set_title("synthetic scene", fontsize=9)
    axes[0].axis("off")
    for i, name in enumerate(names):
        axes[i + 1].imshow(gt[name], cmap="gray")
        axes[i + 1].set_title(f"GT: {name}", fontsize=9)
        axes[i + 1].axis("off")
    fig.tight_layout()
    p = out / "00_scene_and_gt.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"device={dev}  objects={names}  saved={p}")
