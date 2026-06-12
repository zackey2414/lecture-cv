"""章末ミニプロジェクト: 色ベースの簡易物体検出 → 可視化 → モデル入力前処理を統合する。

この回で学んだ部品を「1 枚の画像を受け取り、結果の図と総合レポートを返す」1 本の
パイプラインへ統合する総合課題。後段の検出・分類の章で必要になる「前処理＋可視化」の
雛形そのものなので、ここで手に馴染ませておく。

統合する要素:
  (1) 色空間変換と HSV マスク（preprocess.extract_color）で「色で物体を抜き出す」
  (2) マスクから外接矩形 bbox を numpy の軸順 [y, x] で求める（軸順の罠の総点検）
  (3) バウンディングボックス＋ラベル帯＋白文字を描く（検出可視化の定番）
  (4) 検出物体ごとに正方形レターボックスへ整形（分類モデル入力の作り方）
  (5) EXIF Orientation 正規化のラウンドトリップ検証（写真前処理の入口）
  (6) cv2(BGR) ↔ PIL(RGB) ↔ numpy の相互変換が閉じることを確認

設計方針: 各関数は1つのことだけ・早期 return。どの段が落ちても全体は止めず、
できた範囲でレポートを出す（必ず exit 0）。合成データで完走し、GPU もネットも不要。
実画像で試したい場合は data/sample.jpg を置けば自動でそちらが使われる。

実行: uv run python lectures/03_image_transforms/mini_project.py
出力: outputs/03_image_transforms/ に mini_*.png と mini_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

# matplotlib は import より前に Agg バックエンドへ固定する（headless 安全）。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cv_helpers import get_sample_bgr, output_dir  # noqa: E402
from preprocess import (  # noqa: E402
    center_crop_square,
    extract_color,
    resize_keep_aspect,
    resize_to_square,
)

# EXIF の Orientation タグ番号。6 = 「表示時に時計回り90度回す必要あり」。
_ORIENTATION_TAG = 0x0112

# 色名 → 描画色（BGR）。可視化の見やすさのため、その色に近い濃いめの BGR を割り当てる。
_DRAW_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 220),
    "yellow": (0, 200, 220),
    "green": (0, 200, 0),
    "blue": (220, 0, 0),
}


# ---------------------------------------------------------------------------
# 部品（各関数は単一責務）
# ---------------------------------------------------------------------------
def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """2値マスク(0/255)から、白画素を囲む最小の外接矩形 (x0, y0, x1, y1) を返す。

    np.where は (行=y, 列=x) の順でインデックスを返す——この章で繰り返した軸順そのもの。
    白画素が無ければ None を返す（呼び出し側で None チェックする前提）。
    """
    ys, xs = np.where(mask == 255)   # ys=行(y), xs=列(x) の順で返る
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_label_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    """矩形＋「文字が読める」ラベル帯＋白文字を描く（検出可視化の定番。img はその場で書き換え）。"""
    # 1) バウンディングボックス本体。座標は (x, y) の2点、color は BGR。
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
    # 2) 文字寸法を測ってから、その大きさに合わせた塗りつぶし帯を上辺に置く。
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    top = max(0, y1 - th - baseline - 4)   # 画像上端からはみ出さないようにクランプ
    cv2.rectangle(img, (x1, top), (x1 + tw + 4, y1), color, thickness=-1)
    # 3) 帯の上に白文字（putText の基準点は文字の左下＝ベースライン）。
    cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), font, scale,
                (255, 255, 255), thick, lineType=cv2.LINE_AA)


def detect_colored_objects(bgr: np.ndarray) -> list[dict]:
    """色ごとに HSV マスク → 外接矩形を取り、検出結果のリストを返す。

    各要素: {"color", "bbox":(x0,y0,x1,y1), "area_px"}。
    「色で物体を選り分ける」というこの章の中心技能を、検出の形にまとめた段。
    """
    detections: list[dict] = []
    for color in ("red", "yellow", "green", "blue"):
        mask = extract_color(bgr, color)            # HSV inRange + 軽い開閉（赤は2区間合成）
        bb = bbox_from_mask(mask)
        if bb is None:                              # その色が画像に無ければスキップ
            continue
        detections.append({
            "color": color,
            "bbox": list(bb),
            "area_px": int((mask == 255).sum()),
        })
    return detections


def make_model_inputs(bgr: np.ndarray, detections: list[dict], size: int = 128) -> list[dict]:
    """検出物体ごとに bbox を切り出し、正方形レターボックスでモデル入力サイズへ整形する。

    クロップは numpy スライス [y0:y1, x0:x1]（行=y, 列=x）。歪ませないために
    resize_to_square（アスペクト比保持＋余白）を使う——分類モデル入力作りの定番。
    """
    crops: list[dict] = []
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        patch = bgr[y0:y1 + 1, x0:x1 + 1]           # スライスは [y, x] の順！
        if patch.size == 0:
            continue
        square = resize_to_square(patch, size, pad_color=(30, 30, 30))
        crops.append({"color": det["color"], "image": square})
    return crops


def verify_exif_roundtrip(bgr: np.ndarray, out_dir: pathlib.Path) -> dict:
    """わざと Orientation=6 を付けた JPEG を作り、exif_transpose で正規化できるか検証する。

    素朴に開くと size はそのまま・向きタグも残る。exif_transpose 後は画素が回り、
    (W,H) が入れ替わって向きタグが消える——写真前処理の最重要パターン。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    exif = pil.getexif()
    exif[_ORIENTATION_TAG] = 6
    path = out_dir / "mini_with_exif.jpg"
    pil.save(path, format="JPEG", exif=exif)

    naive = Image.open(path)                                  # 画素は回らない（横倒し）
    fixed = ImageOps.exif_transpose(Image.open(path))        # 画素を実際に回す
    naive.convert("RGB").save(out_dir / "mini_exif_naive.png")
    fixed.convert("RGB").save(out_dir / "mini_exif_fixed.png")
    return {
        "naive_size": list(naive.size),
        "fixed_size": list(fixed.size),
        "naive_orientation": naive.getexif().get(_ORIENTATION_TAG),
        "fixed_orientation": fixed.getexif().get(_ORIENTATION_TAG),
        # 正規化成功 = (W,H) が入れ替わり、向きタグが消えた
        "normalized": list(naive.size) != list(fixed.size)
        and fixed.getexif().get(_ORIENTATION_TAG) is None,
    }


def check_interconversion(bgr: np.ndarray) -> bool:
    """cv2(BGR) → RGB → PIL → numpy → cv2(BGR) と1周して元に戻るかを返す。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(Image.fromarray(rgb))
    back = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bool(np.array_equal(bgr, back))


# ---------------------------------------------------------------------------
# 可視化（1 枚の総合パネル図にまとめる）
# ---------------------------------------------------------------------------
def save_summary_panel(
    scene_bgr: np.ndarray,
    detected_bgr: np.ndarray,
    crops: list[dict],
    out_path: pathlib.Path,
) -> None:
    """元シーン・検出結果・整形済みパッチ2枚を 1 枚に並べて保存する（タイトルは ASCII）。"""
    panels: list[tuple[str, np.ndarray]] = [
        ("input scene", scene_bgr),
        ("color detection", detected_bgr),
    ]
    for det in crops[:2]:
        panels.append((f"letterbox: {det['color']}", det["image"]))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))   # BGR→RGB を忘れない
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Mini project: color detect -> visualize -> preprocess")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# パイプライン本体
# ---------------------------------------------------------------------------
def main() -> None:
    out = output_dir()
    print("=== 第3回 章末ミニプロジェクト: 色検出 → 可視化 → 前処理 ===\n")

    # 入力（data/sample.jpg があればそれ、無ければ合成シーン）。
    scene = get_sample_bgr()
    print(f"[入力] shape={scene.shape}（H={scene.shape[0]}, W={scene.shape[1]}）")

    # (1)(2) 色で物体を検出する。
    detections = detect_colored_objects(scene)
    print(f"\n[検出] {len(detections)} 個の色物体を検出")
    for det in detections:
        print(f"  {det['color']:6s} bbox={det['bbox']} area={det['area_px']}px")

    # (3) 検出結果を可視化する。
    vis = scene.copy()
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        label = f"{det['color']} {det['area_px']}px"
        draw_label_box(vis, x0, y0, x1, y1, label, _DRAW_COLOR_BGR[det["color"]])
    cv2.imwrite(str(out / "mini_detections.png"), vis)

    # (4) 検出物体をモデル入力サイズへ整形する。
    crops = make_model_inputs(scene, detections, size=128)
    for det in crops:
        cv2.imwrite(str(out / f"mini_crop_{det['color']}.png"), det["image"])
    print(f"\n[前処理] {len(crops)} 個のパッチを 128x128 レターボックスへ整形")

    # 参考: シーン全体の前処理3方式の shape も記録する。
    keep = resize_keep_aspect(scene, long_side=256)
    square = resize_to_square(scene, 256, pad_color=(0, 0, 0))
    ccrop = center_crop_square(scene)
    print(f"  resize_keep_aspect(256) -> {keep.shape}")
    print(f"  resize_to_square(256)   -> {square.shape}")
    print(f"  center_crop_square()    -> {ccrop.shape}")

    # (5) EXIF 正規化の検証。
    exif = verify_exif_roundtrip(scene, out)
    print(f"\n[EXIF] naive={exif['naive_size']} -> fixed={exif['fixed_size']} "
          f"normalized={exif['normalized']}")

    # (6) 相互変換ラウンドトリップ。
    roundtrip_ok = check_interconversion(scene)
    print(f"[相互変換] cv2->PIL->numpy->cv2 一致 = {roundtrip_ok}")

    # 総合パネル図。
    save_summary_panel(scene, vis, crops, out / "mini_summary.png")

    # 総合レポート（JSON）。後段の自動評価でも読めるよう機械可読で残す。
    report = {
        "module": "03_image_transforms",
        "input_shape": list(scene.shape),
        "num_detections": len(detections),
        "detections": detections,
        "preprocess_shapes": {
            "resize_keep_aspect_256": list(keep.shape),
            "resize_to_square_256": list(square.shape),
            "center_crop_square": list(ccrop.shape),
        },
        "exif": exif,
        "interconversion_roundtrip_ok": roundtrip_ok,
    }
    report_path = out / "mini_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n完了。outputs/03_image_transforms/ を確認してください。")
    print(f"  - mini_detections.png : 色検出＋ラベルの可視化")
    print(f"  - mini_summary.png    : 入力→検出→前処理の総合パネル")
    print(f"  - mini_exif_naive.png / mini_exif_fixed.png : EXIF 向きの違い")
    print(f"  - mini_report.json    : 総合レポート（機械可読）")


if __name__ == "__main__":
    # ミニプロジェクトは「落ちても exit 0」。想定外でもレポートまで辿り着くことを優先する。
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 途中で例外が発生しましたが、終了コードは 0 にします: {type(e).__name__}: {e}")
