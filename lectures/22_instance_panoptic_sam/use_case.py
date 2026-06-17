"""実践ユースケース: クリック切り抜きツール（SAM の点プロンプトで透明背景 PNG を作る）。

このスクリプトは「学んだ SAM をそのまま“現実の小ツール”にする」ことを目的にした、
動く出発点です。テーマは——

    画像のある一点をクリック（=点プロンプト）したら、その“もの”だけを切り出して
    背景を透明にした PNG（RGBA）を書き出す「カットアウトツール」。

商品写真の背景抜き、アイコン素材づくり、資料に貼る切り抜き画像など、現場で日常的に
発生する「この物体だけ欲しい」を、クラスを知らない SAM のプロンプト1点で解決します。

──────────────────────────────────────────────────────────────────────
mini_project.py との違い（ここが大事）
──────────────────────────────────────────────────────────────────────
  - mini_project.py は「SAM で切り出した結果を mask AP / PQ で“採点”する」評価・
    ベンチ寄りの総合課題です（指標を学ぶのが主目的）。
  - 本 use_case.py は採点しません。代わりに「点 → マスク → 透明 PNG 保存」という
    “納品物（透明背景の切り抜き素材）”を作る、現実の小ツールとして完結します。
    拡張して自分の作業に使えるよう、最小限かつ素直な構成にしてあります。

──────────────────────────────────────────────────────────────────────
実データの置き方（実写優先・合成フォールバック）
──────────────────────────────────────────────────────────────────────
  data/22_instance_panoptic_sam/ に .png / .jpg などを 1 枚置くと、その先頭画像を
  入力に使います（seg_helpers.load_user_or_synthetic が自動で拾います）。画像が
  無ければ合成シーン（赤い円・青い四角・黄色い円）にフォールバックし、必ず完走
  （exit 0）します。SAM はクラス非依存なので、合成図形でも実写でも“指した領域”を
  きれいに切れます（このモジュールで唯一、合成でも絵になるのが SAM）。
  ※ ネットに出るのは SAM 重みの初回ダウンロード（数百MB）だけ。入力はローカル。
  ※ 万一 SAM をロードできない環境では、合成なら GT マスク／実写なら中央楕円の
     “暫定マスク”へ自動フォールバックして exit 0 します（実写で実用にするには
     SAM 重みのDLが必要、という旨をログに出します）。

──────────────────────────────────────────────────────────────────────
クリック位置の決め方
──────────────────────────────────────────────────────────────────────
  実ツールではマウスのクリック座標を渡しますが、CPU・headless で自動実行できるよう、
  既定では「クリック座標を自動で用意」します:
    - 合成シーン: 各 GT 物体の重心を“クリック点”とみなす（=3個の切り抜きができる）
    - 実写       : 画像中央を1点クリックしたとみなす
  自分の画像で狙った物体を切りたいときは、下の CLICK_POINTS に (x, y) を書けば
  その座標を“クリック”として扱います（複数書けば複数枚を切り出します）。

──────────────────────────────────────────────────────────────────────
拡張アイデア（練習に）
──────────────────────────────────────────────────────────────────────
  - 背景点（input_labels=0）を足して「ここは要らない」を SAM に伝え、切り抜きを精緻化。
  - 箱プロンプト（input_boxes）に切り替え、ドラッグ枠での切り抜きにする。
  - アルファのフチを少しぼかす（フェザー）／収縮させて“緑のにじみ”を消す。
  - data/ 内の全画像をループしてバッチ切り抜き（素材一括生成）にする。
  - CLICK_POINTS をコマンドライン引数（argparse）から受け取る簡易 CLI 化。
  - 切り抜きを別背景に合成（透過 PNG → 任意背景へ貼り付け）して合成写真を作る。

実行: uv run python lectures/22_instance_panoptic_sam/use_case.py
出力: lectures/22_instance_panoptic_sam/outputs/
        use_case_cutout_01.png ...   切り抜き本体（透明背景 RGBA・物体ごとに1枚）
        use_case_preview.png         入力＋クリック点／マスク／透明合成プレビュー
        use_case_cutouts.json        生成した切り抜きのメタ情報（座標・面積・bbox 等）
"""

from __future__ import annotations

import json
import pathlib
import sys

# 同じフォルダの seg_helpers.py を import できるようにする（番号付きスクリプトは import 不可）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cv2  # noqa: E402  クリック点のマーカー描画に使う（headless 版）
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import seg_helpers as H  # noqa: E402

# --- 設定（ここを書き換えて自分のツールに育てる）------------------------------
SAM_MODEL_ID = "facebook/sam-vit-base"  # CPU 既定。さらに軽くするなら Zigeng/SlimSAM-uniform-77
# クリック座標を直接指定したいときに (x, y) を列挙する。None なら自動決定（docstring 参照）。
CLICK_POINTS: list[tuple[float, float]] | None = None
PAD = 12  # 切り抜き PNG の bbox に付ける余白（px）。物体の周囲を少し残すと貼りやすい。


# =====================================================================
# 1) SAM のロードと「1点 → ベストマスク」
# =====================================================================

def load_sam(device: torch.device):
    """SAM 本体とプロセッサを返す。失敗時は None（→暫定マスクへフォールバック）。"""
    try:
        from transformers import SamModel, SamProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
        model = SamModel.from_pretrained(SAM_MODEL_ID).to(device).eval()
        return model, processor
    except Exception as e:  # noqa: BLE001  ネット不通・環境差はフォールバックで吸収して exit 0
        print(f"  [warn] SAM をロードできませんでした（{type(e).__name__}）。暫定マスクで続行します。")
        return None


def sam_mask_at_point(model, processor, image, point, device) -> tuple[np.ndarray, float]:
    """点 (x, y) を前景（label=1）として SAM に渡し、最良の1マスク(bool H,W)と品質を返す。

    SAM は曖昧性を考慮して1プロンプトにつき3マスクを返すので、iou_scores（モデルが
    自己申告する品質。真の IoU ではない）が最大の1枚を採る。pred_masks は 256x256 の
    低解像なので、post_process_masks で必ず原寸へ戻す（戻さないとマスクがズレる）。
    """
    inputs = processor(
        image,
        input_points=[[[float(point[0]), float(point[1])]]],
        input_labels=[[1]],  # 1=前景 / 0=背景（取り違え注意）
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        out = model(**inputs)
    masks = processor.post_process_masks(  # ★低解像→原寸（必須）
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]  # (3, H, W) bool
    scores = out.iou_scores[0, 0].cpu()  # (3,) 予測品質
    best = int(torch.argmax(scores))
    return masks[best].numpy().astype(bool), float(scores[best])


def fallback_mask(image: Image.Image, point, owner) -> np.ndarray:
    """SAM が使えない時の暫定マスク（必ず何か返して exit 0 を守る）。

    合成シーンならクリック点が乗っている GT インスタンスをそのまま使う（=正確）。
    実写は GT が無いので、クリック点を中心にした楕円で“それらしく”切る（要 SAM 重みで実用化）。
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if owner is not None:
        return owner["mask"].astype(bool)
    cx, cy = point
    yy, xx = np.ogrid[:h, :w]
    ry, rx = h * 0.32, w * 0.32  # 画像の3割程度を覆う中央楕円
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


# =====================================================================
# 2) クリック座標の決定と、マスク → 透明 PNG への変換
# =====================================================================

def decide_clicks(image: Image.Image, instances) -> tuple[list[tuple[float, float]], list[dict | None]]:
    """クリック座標の一覧と、その点の“正体”（合成時の GT インスタンス or None）を返す。"""
    if CLICK_POINTS:
        # 手動指定。owner は合成シーンなら点が乗る GT を引く（実写は None）。
        owners = [_instance_at(instances, x, y) if instances else None for x, y in CLICK_POINTS]
        return [(float(x), float(y)) for x, y in CLICK_POINTS], owners
    if instances:
        # 合成: 各 GT 物体の重心を“クリック”とみなす → 物体ごとに1枚切り抜く。
        points, owners = [], []
        for inst in instances:
            ys, xs = np.where(inst["mask"])
            points.append((float(xs.mean()), float(ys.mean())))
            owners.append(inst)
        return points, owners
    # 実写で座標未指定: 画像中央を1点クリックしたとみなす。
    w, h = image.size
    return [(w / 2.0, h / 2.0)], [None]


def _instance_at(instances, x: float, y: float):
    """点 (x, y) を含む GT インスタンスを返す（無ければ None）。"""
    xi, yi = int(round(x)), int(round(y))
    for inst in instances:
        m = inst["mask"]
        if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
            return inst
    return None


def cutout_rgba(rgb: np.ndarray, mask: np.ndarray, pad: int) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """マスクをアルファに使い、bbox+余白でトリミングした RGBA 切り抜きを返す。

    アルファ = マスク内 255 / マスク外 0。bbox で切り詰めることで、貼り付けやすい
    “物体ぴったり”の透過 PNG になる（背景の無駄な余白を持ち回らない）。
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None  # 空マスク（呼び出し側でスキップ）
    h, w = mask.shape
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_alpha = (mask[y0:y1, x0:x1].astype(np.uint8)) * 255  # bool → 0/255
    rgba = np.dstack([crop_rgb, crop_alpha]).astype(np.uint8)  # (h, w, 4)
    return rgba, (x0, y0, x1, y1)


# =====================================================================
# 3) プレビュー用の可視化（透明が“見える”ようにチェッカー背景へ合成）
# =====================================================================

def _checkerboard(h: int, w: int, tile: int = 16) -> np.ndarray:
    """透過の確認用チェッカー柄（薄灰/濃灰）の RGB を作る。"""
    yy, xx = np.mgrid[:h, :w]
    pat = (((yy // tile) + (xx // tile)) % 2)[..., None]
    return np.where(pat == 0, 235, 200).astype(np.uint8).repeat(3, axis=2)


def _composite_on_checker(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """全画面サイズの (rgb, alpha) をチェッカー背景に重ねた RGB を返す（透過の見える化）。"""
    bg = _checkerboard(*alpha.shape)
    a = alpha.astype(np.float32)[..., None] / 255.0
    return (a * rgb + (1.0 - a) * bg).clip(0, 255).astype(np.uint8)


def _draw_clicks(rgb: np.ndarray, points) -> np.ndarray:
    """入力画像にクリック点のマーカー（白丸＋赤十字）を描いた RGB を返す。"""
    canvas = rgb.copy()
    for x, y in points:
        c = (int(x), int(y))
        cv2.circle(canvas, c, 9, (255, 255, 255), thickness=2)   # 白丸（RGB 配列なので白）
        cv2.drawMarker(canvas, c, (220, 50, 50), cv2.MARKER_CROSS, markerSize=18, thickness=2)  # 赤十字
    return canvas


def _overlay_masks(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """複数マスクを色分けして半透明に重ねた RGB を返す（どこを切ったかの確認）。"""
    over = rgb.astype(np.float32).copy()
    colors = H.PALETTE[1:1 + len(masks)] if masks else H.PALETTE[1:2]
    for m, col in zip(masks, colors):
        over[m] = 0.45 * over[m] + 0.55 * col
    return over.clip(0, 255).astype(np.uint8)


# =====================================================================
# main
# =====================================================================

def main() -> None:
    device = H.pick_device()
    out_dir = H.ensure_output_dir()

    # --- 入力（実写優先・合成フォールバック）---
    image, instances = H.load_user_or_synthetic()
    source = "synthetic(scene+GT)" if instances else "data/(real image)"
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    print(f"[use_case] device={device}  source={source}  size={w}x{h}")

    # --- クリック座標を決める ---
    points, owners = decide_clicks(image, instances)
    print(f"  クリック点 {len(points)} 個: {[(round(x), round(y)) for x, y in points]}")

    # --- SAM をロード（無ければ暫定マスク）---
    sam = load_sam(device)
    used_sam = sam is not None
    print(f"  予測器: {'SAM ' + SAM_MODEL_ID if used_sam else 'fallback(暫定マスク)'}")

    # --- 各クリック → マスク → 透明 PNG 保存 ---
    chosen_masks: list[np.ndarray] = []
    cutouts_meta: list[dict] = []
    for i, (point, owner) in enumerate(zip(points, owners), start=1):
        if used_sam:
            mask, quality = sam_mask_at_point(sam[0], sam[1], image, point, device)
        else:
            mask, quality = fallback_mask(image, point, owner), None

        rgba, bbox = cutout_rgba(rgb, mask, PAD)
        if rgba is None:
            print(f"  [skip] click#{i} はマスクが空でした（点を物体の内側に置いてください）。")
            continue

        png_path = out_dir / f"use_case_cutout_{i:02d}.png"
        Image.fromarray(rgba, mode="RGBA").save(png_path)  # 透明背景 PNG として保存

        chosen_masks.append(mask)
        cutouts_meta.append({
            "file": png_path.name,
            "click_xy": [round(point[0], 1), round(point[1], 1)],
            "label": (owner["label"] if owner is not None else None),  # 合成時のみ正体が分かる
            "pred_quality": (round(quality, 4) if quality is not None else None),
            "mask_area_px": int(mask.sum()),
            "bbox_xyxy": list(bbox),
            "cutout_size_wh": [int(rgba.shape[1]), int(rgba.shape[0])],
        })
        q_txt = f"quality={quality:.3f}" if quality is not None else "quality=NA(fallback)"
        print(f"  click#{i} {(round(point[0]), round(point[1]))} -> {png_path.name}"
              f"  area={int(mask.sum())}px  {q_txt}")

    # --- プレビュー図（入力＋クリック / 切ったマスク / 透明合成）---
    if chosen_masks:
        combined_alpha = np.zeros((h, w), dtype=np.uint8)
        for m in chosen_masks:
            combined_alpha[m] = 255
        panels = [
            ("input + clicks", _draw_clicks(rgb, points)),
            (f"selected masks (n={len(chosen_masks)})", _overlay_masks(rgb, chosen_masks)),
            ("cutout (alpha on checker)", _composite_on_checker(rgb, combined_alpha)),
        ]
    else:
        # 全クリックで空マスク（実写で点が外れた等）でも図は必ず1枚出す。
        panels = [("input + clicks", _draw_clicks(rgb, points))]
    preview_path = out_dir / "use_case_preview.png"
    H.save_side_by_side(panels, preview_path, suptitle="SAM click-cutout tool")

    # --- メタ情報 JSON ---
    manifest = {
        "tool": "SAM click-cutout (transparent PNG)",
        "model": SAM_MODEL_ID,
        "predictor": "SAM" if used_sam else "fallback",
        "source": source,
        "device": str(device),
        "image_size_wh": [w, h],
        "pad_px": PAD,
        "num_cutouts": len(cutouts_meta),
        "cutouts": cutouts_meta,
    }
    json_path = out_dir / "use_case_cutouts.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  saved: {preview_path.name}, {json_path.name}"
          + (f", {len(cutouts_meta)} cutout PNG(s)" if cutouts_meta else ""))
    if not used_sam:
        print("  ※ 実写で実用にするには SAM 重みのDL（初回のみ・ネット必要）が要ります。"
              "合成なら GT マスクで正確に切れています。")


if __name__ == "__main__":
    main()
