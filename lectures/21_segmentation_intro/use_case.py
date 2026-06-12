"""実践ユースケース: 空・背景の置き換え（セマンティックセグメ × 合成編集ツール）。

この章で学んだ「画像 → クラスマップ」を、評価ではなく **現実の画像編集ツール** に
落とし込んだ動く出発点です。やることは写真スタジオの定番作業「空の差し替え」:

    1) SegFormer（ADE20K, 150クラス）で画像をセグメンテーションし、
    2) 特定クラス（既定は "sky"）のマスクだけを取り出し、
    3) その領域を「別の空画像」や「単色」に差し替えて合成し、
    4) おまけに『空を抜いた透過PNG（前景の切り抜き）』も書き出す。

mini_project.py との違い:
    mini_project.py は mIoU/Dice を測る「評価ベンチマーク（数値レポート）」でした。
    こちらはマスクを“編集に使う”小ツールで、指標は出しません。マスクの羽化（境界の
    ぼかし）やアルファ合成といった、実務の画像編集で効くテクニックに寄せてあります。

実データ優先・合成フォールバック:
    - 入力画像:        data/21_segmentation_intro/ に .png/.jpg を置くと自動で使われます
                       （無ければ seg_helpers の合成シーンにフォールバック）。
    - 差し替え用の空:  data/21_segmentation_intro/replacement/ に .png/.jpg を置くと、
                       その画像を「新しい空」として使います（無ければ夕焼けを自前生成）。
    合成シーンでも上部に青空グラデーションがあるので、SegFormer は "sky" を拾えます。
    ただし合成は反応が弱いこともあり、その場合でも必ず exit 0 になります。実写を置けば
    そのまま実用的な「空入れ替え」ツールとして機能します。

頑健性（必ず exit 0）:
    SegFormer の重みDLに失敗（オフライン等）しても、上部＋青っぽい画素という素朴な
    ヒューリスティックで空マスクを推定し、ツール自体は最後まで完走します。
    ネット接続を使うのはモデル重みのダウンロード時だけです。

拡張して練習するアイデア:
    - TARGET_LABELS を "sky" 以外（"wall" / "building" / "grass" / "tree" など ADE20K の
      クラス名）に変えると、空以外の“背景”置き換えツールになる。
    - 差し替え画像の色味を元写真の明るさに合わせる color grading（馴染ませ）を足す。
    - 地平線付近にグラデーションのブレンド帯を作り、継ぎ目をさらに自然にする。
    - data/ のフォルダを総なめしてバッチ処理する CLI に拡張する。
    - 第23回の CLIPSeg を使えば「文（the sky）」でターゲット領域を指定できる。

実行:   uv run python lectures/21_segmentation_intro/use_case.py
出力:   outputs/21_segmentation_intro/use_case_*.png / use_case_sky_replace.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_helpers as H  # noqa: E402  同じフォルダの道具箱（device 判定・合成シーン・図保存など）

# 差し替えたいクラス（ADE20K のラベル名）。空の置換が既定。
# 例: ("wall", "building") にすれば屋内の壁を、("grass",) なら芝生を差し替える背景ツールになる。
TARGET_LABELS: tuple[str, ...] = ("sky",)

# ベタ塗り置換に使う色（RGB）。晴天ブルー。data/ の差し替え画像が無くても“色置換”は常に出る。
REPLACE_COLOR: tuple[int, int, int] = (96, 175, 235)

# ユーザ提供の「差し替え用の空」を探すフォルダ。
REPLACEMENT_DIR = H.DATA_DIR / "replacement"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# =====================================================================
# 1) ターゲット領域（空）のマスクを得る
# =====================================================================
def _segformer_target_mask(image: Image.Image, device, target_labels: tuple[str, ...]):
    """SegFormer の pipeline で target_labels に一致するクラスの和集合マスクを作る。

    pipeline は [{'label', 'mask'(PIL L 画像)}, ...] を返す（セマンティックでは score=None）。
    各マスクは互いに重ならないので、対象ラベルのマスクを OR で束ねれば良い。
    """
    seg = H.load_segformer_pipeline(device)
    outputs = seg(image)
    w, h = image.size  # PIL は (幅, 高さ)
    mask = np.zeros((h, w), dtype=bool)
    labels_present: list[str] = []
    for o in outputs:
        labels_present.append(o["label"])
        if o["label"] in target_labels:
            mask |= np.asarray(o["mask"]) > 127  # L 画像（前景255）→ bool
    return mask, labels_present


def _heuristic_sky_mask(rgb: np.ndarray) -> np.ndarray:
    """重みDLに失敗したときの素朴な空マスク（上部かつ明るい青っぽい画素）。

    精度は出ないが「ツールが必ず完走する」ためのフォールバック。実写＋モデルが本命。
    """
    h, w, _ = rgb.shape
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    bright = (r + g + b) / 3.0 > 120  # 空は概して明るい
    bluish = (b >= r) & (b > 90)      # 赤より青が強い
    upper = np.zeros((h, w), dtype=bool)
    upper[: int(h * 0.6), :] = True   # 画面上 60% を空候補にする位置事前分布
    return upper & bright & bluish


def get_target_mask(image: Image.Image, device, target_labels: tuple[str, ...]):
    """空マスクを返す。SegFormer 優先、失敗時はヒューリスティックに切り替える。

    返り値: (mask(bool, H×W), method(str), labels_present(list))
    """
    try:
        mask, labels_present = _segformer_target_mask(image, device, target_labels)
        return mask, "segformer_ade20k", labels_present
    except Exception as e:  # noqa: BLE001  オフライン等でも教材は exit 0 を守る
        rgb = np.asarray(image)
        mask = _heuristic_sky_mask(rgb)
        return mask, f"heuristic_fallback({type(e).__name__})", []


# =====================================================================
# 2) マスクの羽化（境界をぼかして自然に合成する）
# =====================================================================
def feather_alpha(mask_bool: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """bool マスクを [0,1] のソフトなアルファに変換する（境界の継ぎ目を目立たせない）。

    硬いマスクのまま貼り替えると輪郭がギザギザになる。GaussianBlur で羽化すると、
    元画像と差し替え画像が境界でなめらかに混ざり、合成が一気に“それっぽく”なる。
    """
    h, w = mask_bool.shape
    if sigma is None:
        sigma = max(2.0, min(h, w) * 0.01)  # 画像サイズに応じた控えめな羽化幅
    a = mask_bool.astype(np.float32)
    a = cv2.GaussianBlur(a, ksize=(0, 0), sigmaX=sigma)
    return np.clip(a, 0.0, 1.0)


# =====================================================================
# 3) 差し替え用の「新しい空」を用意する（実画像優先 / 無ければ合成）
# =====================================================================
def load_replacement_image(h: int, w: int):
    """data/<id>/replacement/ にある画像を新しい空として読み込み、入力サイズへ合わせる。

    無ければ (None, None) を返す（呼び出し側で合成の夕焼けにフォールバック）。
    """
    if REPLACEMENT_DIR.is_dir():
        files = sorted(p for p in REPLACEMENT_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
        if files:
            im = Image.open(files[0]).convert("RGB").resize((w, h))
            return np.asarray(im), files[0].name
    return None, None


def make_sunset_sky(h: int, w: int) -> np.ndarray:
    """夕焼けの空（縦グラデ＋にじむ太陽）を合成する (H,W,3) uint8。差し替えの既定。"""
    top = np.array([45, 25, 80], dtype=np.float32)    # 上空: 藍色
    bottom = np.array([255, 170, 90], dtype=np.float32)  # 地平線: 暖色オレンジ
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    sky = (1.0 - ys) * top + ys * bottom  # 上→下の縦グラデーション
    sky = np.repeat(sky, w, axis=1)
    # 地平線寄りの中央に、ふんわり光る太陽を加える
    glow = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glow, (int(w * 0.5), int(h * 0.80)), int(min(h, w) * 0.10), 1.0, thickness=-1)
    glow = cv2.GaussianBlur(glow, ksize=(0, 0), sigmaX=min(h, w) * 0.06)
    sun_color = np.array([255, 245, 200], dtype=np.float32)
    sky += (sun_color - sky) * np.clip(glow, 0, 1)[..., None] * 0.85
    return np.clip(sky, 0, 255).astype(np.uint8)


def make_solid_sky(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    """単色で塗りつぶした差し替え画像 (H,W,3) uint8。"""
    sky = np.empty((h, w, 3), dtype=np.uint8)
    sky[:] = color
    return sky


# =====================================================================
# 4) 合成（アルファブレンド）と透過切り抜き
# =====================================================================
def composite(rgb: np.ndarray, replacement: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha=1 の画素を replacement に、alpha=0 の画素を元画像に。境界は連続的に混ぜる。"""
    a = alpha[..., None]  # (H,W,1) にして 3ch へブロードキャスト
    out = a * replacement.astype(np.float32) + (1.0 - a) * rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_cutout_rgba(rgb: np.ndarray, sky_alpha: np.ndarray) -> np.ndarray:
    """空を透明にして前景だけ残した RGBA 画像 (H,W,4) uint8 を作る（切り抜き出力）。

    前景アルファ = 1 - 空アルファ。空の画素ほど透明（α→0）になる。
    """
    fg_alpha = np.clip(1.0 - sky_alpha, 0.0, 1.0)
    a = (fg_alpha * 255).astype(np.uint8)
    return np.dstack([rgb, a])


# =====================================================================
# メイン
# =====================================================================
def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()

    # 入力画像（実写優先 / 無ければ合成シーン）
    image, src = H.load_user_or_synthetic_image()
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]

    # 1) 空マスク → 2) 羽化してソフトなアルファに
    mask, method, labels_present = get_target_mask(image, device, TARGET_LABELS)
    alpha = feather_alpha(mask)
    sky_ratio = float(mask.mean())  # 空が画面に占める割合

    # 3) 差し替え用の空を用意（ユーザ画像があればそれ、無ければ夕焼け）。色置換は常に作る。
    user_sky, user_name = load_replacement_image(h, w)
    sunset = make_sunset_sky(h, w)
    solid = make_solid_sky(h, w, REPLACE_COLOR)
    replacement_sky = user_sky if user_sky is not None else sunset
    replacement_name = user_name if user_name is not None else "synthetic_sunset"

    # 4) 合成（単色置換 / 空画像置換）と透過切り抜き
    comp_solid = composite(rgb, solid, alpha)
    comp_image = composite(rgb, replacement_sky, alpha)
    cutout_rgba = make_cutout_rgba(rgb, alpha)

    # --- 可視化パネル（入力 / 空アルファ / 色置換 / 画像置換）---
    mask_vis = np.repeat((np.clip(alpha, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    panel = [rgb, mask_vis, comp_solid, comp_image]
    titles = [
        f"input ({src})",
        f"sky alpha ({sky_ratio:.1%})",
        "replace: solid color",
        f"replace: {replacement_name}",
    ]
    panel_path = out / "use_case_sky_replace.png"
    H.save_panel(panel, titles, panel_path, ncol=4)

    # --- 透過切り抜き（前景のみ・空は透明）を RGBA PNG で保存 ---
    cutout_path = out / "use_case_cutout.png"
    Image.fromarray(cutout_rgba, mode="RGBA").save(cutout_path)

    # --- レポート JSON ---
    summary = {
        "input": src,
        "device": str(device),
        "segmentation_method": method,
        "target_labels": list(TARGET_LABELS),
        "labels_present": labels_present,
        "sky_pixel_ratio": round(sky_ratio, 4),
        "replacement_used": replacement_name,
        "outputs": [panel_path.name, cutout_path.name],
    }
    json_path = out / "use_case_sky_replace.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- コンソール ---
    print("=== 空・背景の置き換えツール ===")
    print(f"input={src}  device={device}  method={method}")
    print(f"対象クラス {TARGET_LABELS} の占有率 = {sky_ratio:.1%}")
    if labels_present:
        print(f"SegFormer が付けたラベル: {labels_present}")
    if sky_ratio < 0.005:
        print("注意: 対象クラスがほぼ検出されませんでした。"
              "合成画像では反応が弱いことがあります。data/ に実写を置くと実用的になります。")
    print(f"差し替えに使った空: {replacement_name}"
          + ("（data/.../replacement/ の画像）" if user_sky is not None else "（自前生成の夕焼け）"))
    print(f"saved: {panel_path}, {cutout_path}, {json_path}")


if __name__ == "__main__":
    main()
