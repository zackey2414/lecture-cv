"""第18回 共通の道具箱（物体検出 入門）。

このモジュールは「読み物」ではなく、各スクリプトが import して使う再利用部品です。
責務は次の5つに絞ってあります（単一責務）:

  1. device の自動判定（cuda > mps > cpu。本講座の既定は cpu）
  2. 合成シーン画像の生成（人物・車らしき形。ネット不要・決定的）
  3. 入力画像のロード（data/<module_id>/ に実写があれば優先、無ければ合成）
  4. 検出の共通後処理（score 閾値フィルタ → クラス別 NMS）
  5. 可視化（draw_bounding_boxes は uint8 画像テンソルを要求）と図の保存

なぜ合成画像か:
  本講座のルール上、ネットへ出てよいのは「モデル重みのDLのみ」で、入力画像は
  合成生成する。COCO で学習した検出器は写実的な写真向けに最適化されているため、
  抽象的な合成シーンでは検出が乏しい（スコアが低い・少数しか出ない）ことがある。
  それでもスクリプトは必ず exit 0 になるよう設計してあり、`data/18_object_detection_intro/`
  に実写 .jpg/.png を置けば、そのまま実用的な検出結果が得られる（load_input_image 参照）。
"""

from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
import torch
from PIL import Image

# --- transformers / huggingface_hub のログを静かにする（教材の出力を読みやすく） ---
# 重み DL の進捗バーや未認証警告は学習の本質ではないので抑制する。
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 環境（GUI なし）でも図を保存できるバックエンド
import matplotlib.pyplot as plt  # noqa: E402

# プロジェクトの規約に合わせ、結果は lectures/<module_id>/outputs/ に集約する。
MODULE_ID = "18_object_detection_intro"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
DATA_DIR = pathlib.Path("data") / MODULE_ID


def pick_device() -> torch.device:
    """cuda > mps > cpu の優先で torch.device を返す（全回共通の定石）。

    本講座は CPU 前提だが、手元に GPU/Apple Silicon があれば自動で使えるようにする。
    重要: モデルと入力テンソルの device は必ず揃える（.to(device) を両方に）。
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


def make_scene_image(width: int = 520, height: int = 360) -> Image.Image:
    """人物・車らしき形を置いた合成シーン（PIL.Image, RGB）を1枚作る。

    COCO 学習済み検出器は「縦長の胴体＋丸い頭」「横長の箱＋車輪」のような単純図形にも
    反応する。ただし写実的な写真ではないため、付くラベルは person/car とは限らず
    "stop sign" など的外れになりがち。これは「合成画像では検出器が当てにならない」
    という大事な学び（README 参照）で、パイプライン（前処理→推論→閾値→NMS→可視化）の
    動作確認には十分。実用的な結果が欲しければ data/<module_id>/ に実写を置く。

    cv2 を描画ツールとして使うが、配列は最初から RGB として解釈する
    （色タプルを (R,G,B) で渡し、cv2.imread/imwrite を経由しないので BGR 混乱は起きない）。
    """
    img = np.full((height, width, 3), 235, dtype=np.uint8)  # 明るい灰色の背景
    # 地面（下 1/5 を少し濃い灰色に）
    cv2.rectangle(img, (0, int(height * 0.8)), (width, height), (205, 205, 205), thickness=-1)

    def draw_person(cx: int, top: int, color: tuple[int, int, int],
                    body_w: int, body_h: int, head_r: int) -> None:
        """縦長の胴体＋丸い頭で人物らしき figure を描く。"""
        cv2.circle(img, (cx, top + head_r), head_r, color, thickness=-1)  # 頭
        cv2.rectangle(
            img,
            (cx - body_w // 2, top + 2 * head_r),
            (cx + body_w // 2, top + 2 * head_r + body_h),
            color,
            thickness=-1,
        )  # 胴体

    def draw_car(cx: int, cy: int, color: tuple[int, int, int]) -> None:
        """横長の車体＋2つの車輪で車らしき figure を描く。"""
        cv2.rectangle(img, (cx - 70, cy - 26), (cx + 70, cy + 16), color, thickness=-1)  # 車体下部
        cv2.rectangle(img, (cx - 44, cy - 46), (cx + 34, cy - 22), color, thickness=-1)  # 車体上部（窓側）
        cv2.circle(img, (cx - 44, cy + 16), 16, (35, 35, 35), thickness=-1)  # 前輪
        cv2.circle(img, (cx + 44, cy + 16), 16, (35, 35, 35), thickness=-1)  # 後輪

    draw_person(110, 40, (60, 70, 170), body_w=70, body_h=200, head_r=34)  # 大きい青い人物
    draw_person(250, 70, (150, 80, 70), body_w=56, body_h=150, head_r=26)  # 小さめの赤い人物
    draw_car(410, 250, (50, 120, 90))  # 緑っぽい車
    return Image.fromarray(img)  # RGB のまま PIL へ


def load_input_image() -> tuple[Image.Image, str]:
    """入力画像を1枚返す。data/<module_id>/ に実写があれば優先、無ければ合成シーン。

    返り値: (PIL.Image[RGB], source)  source は "data:<filename>" か "synthetic"。
    実写を置けば検出が実用的になることを体感できるよう、必ずこの導線を用意する。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if DATA_DIR.is_dir():
        files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
        if files:
            return Image.open(files[0]).convert("RGB"), f"data:{files[0].name}"
    return make_scene_image(), "synthetic"


def to_uint8_chw(image: Image.Image) -> torch.Tensor:
    """PIL(RGB) を draw_bounding_boxes が要求する uint8 CHW テンソルに変換する。

    ★落とし穴: draw_bounding_boxes は float のままだと例外/真っ黒になる。必ず uint8。
    """
    arr = np.array(image, dtype=np.uint8)  # HWC, RGB（np.array はコピーを作るので書き込み可）
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8


def apply_threshold_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """検出の共通後処理: score 閾値で間引いてから「クラス別」NMS を掛ける。

    - score_thresh: これ未満の確信度の箱を捨てる（生の検出は低スコアの箱だらけ）。
    - iou_thresh:   重なりが大きい同一クラスの箱を1つに統合する閾値。
    - batched_nms は idxs（=クラス）ごとに独立に NMS する。普通の nms を全クラス
      まとめて掛けると、隣接する別クラス（人と車）が誤って潰れるので batched を使う。

    返り値: 閾値・NMS 後の (boxes(xyxy), scores, labels)。空になることもある。
    """
    from torchvision.ops import batched_nms

    keep_mask = scores >= score_thresh
    boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = batched_nms(boxes, scores, labels, iou_thresh)
    return boxes[keep], scores[keep], labels[keep]


def draw_detections(
    image: Image.Image,
    boxes: torch.Tensor,
    names: list[str],
    scores: torch.Tensor,
    color: str = "red",
) -> Image.Image:
    """検出結果を画像に重ね書きして PIL(RGB) で返す（draw_bounding_boxes 利用）。

    - boxes: (N,4) xyxy（絶対座標, float）。names/scores は長さ N。
    - 検出ゼロのときは元画像をそのまま返す（draw_bounding_boxes は空 box で描けない）。
    """
    from torchvision.utils import draw_bounding_boxes

    canvas = to_uint8_chw(image)
    if boxes is None or len(boxes) == 0:
        return image
    box_t = torch.as_tensor(boxes, dtype=torch.float32)
    text = [f"{n} {float(s):.2f}" for n, s in zip(names, scores)]
    drawn = draw_bounding_boxes(canvas, box_t, labels=text, colors=color, width=2)
    return Image.fromarray(drawn.permute(1, 2, 0).contiguous().numpy())


def save_panel(items: list[tuple[Image.Image, str]], path: pathlib.Path, suptitle: str) -> None:
    """(画像, タイトル) のリストを横並びパネルにして保存する（定性確認用）。"""
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0))
    axes = np.atleast_1d(axes)
    for ax, (im, title) in zip(axes, items):
        ax.imshow(im)  # PIL は RGB なので matplotlib にそのまま渡せる
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # スモークテスト: 単体実行でも exit 0 になることを確認する。
    out = ensure_output_dir()
    dev = pick_device()
    image, source = load_input_image()
    image.save(out / "00_scene.png")
    print(f"device={dev}  source={source}  size={image.size}  saved={out / '00_scene.png'}")
