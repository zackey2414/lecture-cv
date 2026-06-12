"""use_case: テキストで領域を選んで加工する「ことばで選んで編集」ツール。

このスクリプトは『文（例: "the cat"）で対象を指し、その領域だけを加工する』現実の
小道具です。CLIPSeg でテキスト条件付きのマスクを作り、同じマスクから次の3つの編集を
一度に出力します（mini_project の「手法比較ベンチ」とは別物の、完結した実ツール）:

  1) ぼかす (blur)   … 個人情報（顔・ナンバープレート・名札・看板）を文で指定して匿名化
  2) 消す  (remove)  … 不要物を cv2.inpaint で周囲から塗り消す（簡易オブジェクト除去）
  3) 抜く  (cutout)  … 対象を背景透過 PNG(RGBA) として切り出す（ステッカー/素材作り）

────────────────────────────────────────────────────────────────────────
■ 実データ優先・合成フォールバック（実写を置けばそのまま実用ツールになる）
  - data/23_text_prompt_segmentation/image.png があれば、それを入力に使います。
    対象プロンプトは data/23_text_prompt_segmentation/prompts.txt の1行目（無ければ
    "the main object"）。環境変数 USECASE_PROMPT を指定すればそれで上書きできます。
  - 画像が無ければ seg_helpers の合成シーン（赤い円・青い四角・緑の三角）に自動で
    フォールバックし、ネットもデータも無しで必ず exit 0 で完走します。
  - 合成画像は実写よりテクスチャが乏しく CLIPSeg の確信度が低めです。閾値に1画素も
    届かない場合は「確率の上位だけを選ぶ」救済（fallback）を入れ、ツールとして“必ず
    何かを選ぶ”ようにしています。検出/セグメは合成だと反応が弱くても問題ありません。
    実写を data/ に置けば、そのまま匿名化/除去/切り出しの実用ツールになります。

■ mini_project.py との違い（重要）
  - mini_project.py … CLIPSeg@0.5 / CLIPSeg@best / Grounded-SAM の3手法を GT との
                       IoU で競わせる「評価寄りのベンチ」。GT マスクと数値が主役。
  - use_case.py(本ファイル) … 評価ではなく『マスクを作った後に何ができるか』を示す
                       現実の編集ツール。GT も IoU も不要で、1枚の入力から
                       blur/remove/cutout の3成果物を出して終わる。

■ 拡張アイデア（ここを出発点に練習を広げる）
  - 閾値・ぼかし強度・inpaint 半径を argparse で引数化し、CLI ツールに仕立てる。
  - マスクを SAM(seg_helpers.load_sam) の input_boxes/points で境界リファインしてから
    加工する（Grounded-SAM の流用で、ぼかし・切り出しの輪郭が鋭くなる）。
  - remove を cv2.inpaint から拡散モデル(diffusers)のインペイントに差し替える。
  - prompts.txt に複数行書いて、複数対象を一括で匿名化するバッチ処理にする。
  - 動画の各フレームへ適用し、「文で指定した対象だけモザイク」の動画を作る。

実行: uv run python lectures/23_text_prompt_segmentation/use_case.py
出力: outputs/23_text_prompt_segmentation/use_case_blur.png / use_case_removed.png /
      use_case_cutout.png（透過PNG）/ use_case_panel.png / use_case_report.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# digit 始まりのスクリプト(01_*.py 等)は import できないが、seg_helpers は import 可。
# device 判定・入力ロード・CLIPSeg 後処理・可視化はすべて道具箱から借用する（再発明しない）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cv2  # noqa: E402  （ぼかし・inpaint・透過合成に使う。配列は終始 RGB のまま扱う）
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  （seg_helpers が Agg を設定済み）
import seg_helpers as H  # noqa: E402


# ---------------------------------------------------------------------------
# マスク作り（CLIPSeg の確率マップ → 2値マスク → 羽根付きα）
# ---------------------------------------------------------------------------
def binarize_with_fallback(prob: np.ndarray, threshold: float) -> tuple[np.ndarray, bool]:
    """確率マップを2値化する。空になったら上位確率で救済し、必ず非空にして返す。

    返り値: (bool マスク (H, W), 救済を使ったか)。
    合成画像では CLIPSeg の確信度が全体的に低く、threshold(=0.5) に1画素も届かない
    ことがある。ツールとして「何も選べない」のは不便なので、その場合は確率の最大値の
    半分を閾値にして“最も反応した領域”を選ぶ。実写では通常 threshold でそのまま選べる。
    """
    mask = prob >= threshold
    if mask.any():
        return mask, False
    pmax = float(prob.max())
    if pmax <= 0.0:
        # 完全に無反応（実用上ほぼ起きない）。空マスクのまま返し、各編集は no-op になる。
        return mask, True
    return prob >= (pmax * 0.5), True


def feather_alpha(mask: np.ndarray, sigma: float) -> np.ndarray:
    """bool マスクの境界をぼかして、0〜1 の滑らかな合成用αを作る。

    硬い 0/1 マスクで合成するとぼかし/切り出しの輪郭がギザギザになる。ガウシアンで
    境界を羽根付き(feather)にすると、自然な合成になる（αは透過の度合いそのもの）。
    """
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3つの編集オペレーション（どれも入力 RGB(uint8) を壊さず新しい配列を返す）
# ---------------------------------------------------------------------------
def op_blur(rgb: np.ndarray, alpha: np.ndarray, blur_sigma: float) -> np.ndarray:
    """選択領域だけをぼかして匿名化する。α合成で「マスク内=ぼかし／外=元画像」。"""
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    a = alpha[..., None]  # (H,W,1) にして3チャンネルへブロードキャスト
    out = a * blurred.astype(np.float32) + (1.0 - a) * rgb.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def op_remove(rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """選択領域を cv2.inpaint で周囲から塗り消す（簡易オブジェクト除去）。

    inpaint のマスクは「塗り直す画素=非ゼロ」の 8bit 1ch。境界の塗り残しを減らすため
    少し dilate(膨張)してから渡す。INPAINT_TELEA は高速な Fast Marching 法。
    cv2.inpaint はチャンネルを独立に処理するので RGB/BGR の別は結果に影響しない。
    """
    m = (mask.astype(np.uint8)) * 255
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.inpaint(rgb, m, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def op_cutout(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """選択領域を背景透過で切り出した RGBA(uint8, H×W×4) を返す。αが透明度。"""
    a = (alpha * 255.0).clip(0, 255).astype(np.uint8)
    return np.dstack([rgb, a])  # (H,W,3)+(H,W) -> (H,W,4)


# ---------------------------------------------------------------------------
# 可視化補助（透過の見え方を確認するための市松模様合成）
# ---------------------------------------------------------------------------
def composite_on_checker(rgba: np.ndarray, square: int = 18) -> np.ndarray:
    """RGBA を市松模様の背景に重ねた RGB を返す（透明部分が分かるようにする）。"""
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    checker = ((yy // square) + (xx // square)) % 2
    board = np.where(checker[..., None] == 0, 205, 245).astype(np.float32)
    board = np.repeat(board, 3, axis=2)
    a = (rgba[..., 3:4].astype(np.float32)) / 255.0
    out = a * rgba[..., :3].astype(np.float32) + (1.0 - a) * board
    return out.clip(0, 255).astype(np.uint8)


def save_panel(
    image: Image.Image,
    mask: np.ndarray,
    blurred: np.ndarray,
    removed: np.ndarray,
    cutout_rgba: np.ndarray,
    prompt: str,
    path: pathlib.Path,
) -> None:
    """[input / selected mask / blur / remove / cutout] を横1列に並べて保存（ASCIIタイトル）。"""
    panels = [
        ("input", np.asarray(image.convert("RGB"))),
        ("CLIPSeg mask overlay", H.overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5)),
        ("edit: blur region", blurred),
        ("edit: remove (inpaint)", removed),
        ("edit: cutout (RGBA)", composite_on_checker(cutout_rgba)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    # 図全体のタイトルは ASCII のみ（CJK は豆腐□になるため、prompt は ascii 化して載せる）。
    safe_prompt = prompt.encode("ascii", "replace").decode("ascii")
    fig.suptitle(f"text-prompted editing  prompt='{safe_prompt}'", fontsize=11)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # --- 1) 入力（実画像があれば優先、無ければ合成シーン）---
    image, _gt, scene = H.load_user_or_scene()
    rgb = np.asarray(image.convert("RGB"))  # (H, W, 3) uint8, RGB のまま
    h, w = rgb.shape[:2]

    # 対象プロンプト: 環境変数 > prompts.txt(=scene[0]) の優先。1つだけ加工する。
    target_prompt = os.environ.get("USECASE_PROMPT", "").strip() or scene[0]["prompt"]

    # 入力ソースの判定（レポート用）。data/ に実画像があれば "real"。
    data_img = pathlib.Path("data") / H.MODULE_ID / "image.png"
    source = "real (data/)" if data_img.is_file() else "synthetic (fallback)"

    # --- 2) CLIPSeg で対象プロンプトの確率マップを得る（道具箱の中核関数を借用）---
    model, processor = H.load_clipseg(device)
    prob = H.clipseg_probs(model, processor, image, [target_prompt], device)[0]  # (H, W) 0〜1

    # --- 3) 確率マップ → 2値マスク（空なら上位確率で救済）→ 羽根付きα ---
    threshold = 0.5
    mask, used_fallback = binarize_with_fallback(prob, threshold)
    feather_sigma = max(1.0, min(h, w) * 0.008)
    alpha = feather_alpha(mask, feather_sigma)

    # --- 4) 3つの編集を一度に作る ---
    blur_sigma = max(3.0, min(h, w) / 30.0)  # 画像サイズに比例した自然なぼかし量
    inpaint_radius = 3
    blurred = op_blur(rgb, alpha, blur_sigma)
    removed = op_remove(rgb, mask, inpaint_radius)
    cutout_rgba = op_cutout(rgb, alpha)

    # --- 5) 保存（編集結果・透過PNG・パネル図）---
    blur_path = out / "use_case_blur.png"
    removed_path = out / "use_case_removed.png"
    cutout_path = out / "use_case_cutout.png"
    panel_path = out / "use_case_panel.png"
    Image.fromarray(blurred).save(blur_path)
    Image.fromarray(removed).save(removed_path)
    Image.fromarray(cutout_rgba, mode="RGBA").save(cutout_path)  # 背景透過 PNG
    save_panel(image, mask, blurred, removed, cutout_rgba, target_prompt, panel_path)

    # --- 6) レポート JSON ---
    mask_pixels = int(mask.sum())
    report = {
        "prompt": target_prompt,
        "input_source": source,
        "image_size": [int(w), int(h)],
        "threshold": threshold,
        "used_topk_fallback": used_fallback,  # 閾値で選べず上位確率で救済したか
        "prob_max": round(float(prob.max()), 4),
        "mask_pixels": mask_pixels,
        "mask_coverage": round(mask_pixels / float(h * w), 4),
        "params": {
            "feather_sigma": round(float(feather_sigma), 3),
            "blur_sigma": round(float(blur_sigma), 3),
            "inpaint_radius": inpaint_radius,
        },
        "outputs": {
            "blur": str(blur_path),
            "remove": str(removed_path),
            "cutout_rgba": str(cutout_path),
            "panel": str(panel_path),
        },
    }
    report_path = out / "use_case_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 7) コンソール要約 ---
    print(f"device = {device}  input = {source}")
    print(f"prompt = '{target_prompt}'  prob[max] = {report['prob_max']:.2f}")
    note = "（閾値で選べず上位確率で救済）" if used_fallback else ""
    print(f"selected pixels = {mask_pixels}  coverage = {report['mask_coverage']:.1%} {note}")
    print("edits saved:")
    print(f"  blur   -> {blur_path}")
    print(f"  remove -> {removed_path}")
    print(f"  cutout -> {cutout_path}  (transparent RGBA)")
    print(f"  panel  -> {panel_path}")
    print(f"  report -> {report_path}")
    if source.startswith("synthetic"):
        print("\nヒント: data/23_text_prompt_segmentation/image.png と prompts.txt を置くと、")
        print("       実写に対する匿名化/除去/切り出しの実用ツールとして動きます。")


if __name__ == "__main__":
    main()
