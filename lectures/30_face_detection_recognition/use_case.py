"""use_case.py — 顔の自動ぼかし（匿名化）ツール

この回の「実践ユースケース」: 画像中の顔を **検出 → 自動でぼかし**、プライバシーを
保護した画像を書き出す **匿名化(anonymization)ツール** です。SNS 投稿前の通行人の顔
消し、街頭カメラ映像の公開用マスキング、データセット公開時の被写体保護、ストリート
ビュー的サービスの顔ぼかし —— どれも「**顔検出 × ぼかし合成**」の組み合わせで作れます。

────────────────────────────────────────────────────────────────────────
mini_project.py との違い（役割分担）
────────────────────────────────────────────────────────────────────────
- `mini_project.py` は「検出 → 整列 → 埋め込み → クラスタリング → **人物アルバム＋
  認識/クラスタ品質の評価**」を通す統合課題で、**「誰が誰か」を当てる**のが目的です。
- 本 `use_case.py` は逆に **「誰か分からなくする」** のが目的の、現実の小ツールです。
  認識も埋め込みも使わず、**検出した顔領域を読めなくする**ことだけに集中します。
  検出器は本章の Haar カスケードをそのまま借りつつ、その上に「顔の矩形を
  ガウシアンぼかし／モザイク／黒塗りで潰す」という匿名化ロジックだけを薄く足しています。
  そのまま実運用のプライバシー保護ツールへ拡張できます。

────────────────────────────────────────────────────────────────────────
データ配置（実データ優先・合成フォールバック）
────────────────────────────────────────────────────────────────────────
- `data/30_face_detection_recognition/` に **画像**（*.jpg / *.jpeg / *.png / *.bmp）を
  置くと、それを実入力として匿名化します。
  例: `data/30_face_detection_recognition/street.jpg` を置けば、その写真の顔を
  自動でぼかした `outputs/30_face_detection_recognition/use_case_anon_street.png` が出ます。
- 置かなければ **合成の集合写真**（複数の顔を 1 枚に並べたもの）で必ず完走します（exit 0）。
- ★重要な注意: Haar は **実写の顔**で学習されているため、**合成顔は検出されない**ことが
  あります（=ドメインギャップ）。本ツールは合成シーンで検出が弱いときだけ、顔の **正解位置**
  に自動フォールバックして匿名化を必ず実演します（exit 0 を保証）。**実写を `data/` に
  置けば、検出 → ぼかしが本来の挙動（検出ドリブン）で動きます。**

────────────────────────────────────────────────────────────────────────
匿名化の方式（METHOD で切替）
────────────────────────────────────────────────────────────────────────
- "blur"    : ガウシアンぼかし。顔の存在感は残しつつ個人を識別不能にする（最も自然）。
- "pixelate": モザイク（粗いブロック化）。「加工した」ことが一目で分かる定番表現。
- "box"     : 黒塗り。最も強力（情報を完全に消す）だが見た目は最も不自然。
- 既定は楕円マスクで顔の輪郭に沿って合成するので、矩形の角に背景が残らず自然に仕上がります。

────────────────────────────────────────────────────────────────────────
拡張アイデア（練習用）
────────────────────────────────────────────────────────────────────────
- 検出を OpenCV DNN(res10 SSD) や MediaPipe / RetinaFace に差し替えて、横顔・小顔・
  逆光に強くする（Haar の取りこぼしを減らす）。`01_face_detection.py` の DNN 案内を参照。
- 動画対応: `cv2.VideoCapture` で 1 フレームずつ読み、同じ匿名化を適用して `VideoWriter`
  で書き出す（防犯カメラ映像の公開用マスキング）。
- 「特定の人物だけ残す（または特定の人物だけ消す）」: 本章の `FaceEncoder` で埋め込みを
  取り、登録済みベクトルとのコサイン類似で本人だけ除外（同意者は残す等）に拡張。
- 顔以外（ナンバープレート・名札・画面の文字）も検出して同時にぼかす総合匿名化へ拡張。
- 元の解像度に戻せないよう、ぼかし強度・モザイク粒度を「復元困難」な水準に固定する
  （弱いモザイクは超解像で部分的に復元され得るため、強度の検証も練習になります）。

実行:
    uv run python lectures/30_face_detection_recognition/use_case.py

出力（outputs/30_face_detection_recognition/ に保存）:
    use_case_anonymized.png   : 元画像 vs 匿名化後 の Before/After モンタージュ
    use_case_methods.png      : 同じ顔に blur / pixelate / box を適用した方式比較（matplotlib）
    use_case_anon_<name>.png  : （実写を置いた場合）各入力の匿名化後フル画像
    use_case_report.json      : 入力ごとの検出数・匿名化数・方式などの機械可読レポート
"""

from __future__ import annotations

import glob
import json
import os

# headless 前提: matplotlib は必ず Agg バックエンド（GUI ウィンドウを開かない）。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import face_lab as fl  # noqa: E402  # 合成顔データ生成・パス定数を借用（番号付きスクリプトは import 不可）

# ----------------------------------------------------------------------------
# 設定（ここを変えるだけで挙動を調整できる）
# ----------------------------------------------------------------------------
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

METHOD = "blur"        # "blur" | "pixelate" | "box"。匿名化の方式。
USE_ELLIPSE = True     # True で顔の楕円輪郭に沿って合成（矩形の角に背景を残さない）。
EXPAND = 0.18          # 検出枠を上下左右にこの比率だけ広げる（額・あごまで確実に覆う）。
BLUR_STRENGTH = 0.55   # ガウシアンぼかしの強さ（顔サイズに対するカーネル比。大きいほど強い）。
PIXELATE_BLOCKS = 8    # モザイクのブロック数（小さいほど粗い＝強い匿名化）。
MIN_MATCH_RATIO = 0.5  # 合成シーンで検出がこの割合未満なら「正解位置」へフォールバック。
MAX_REAL_IMAGES = 12   # 実写はこの枚数まで処理（モンタージュが巨大化しないように）。


# ----------------------------------------------------------------------------
# 顔検出（本章の Haar カスケードをそのまま使う・重み不要で即動く）
# ----------------------------------------------------------------------------
def load_cascade() -> cv2.CascadeClassifier:
    """OpenCV 同梱の正面顔 Haar カスケードを読み込む。"""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"カスケードを読み込めません: {path}")
    return cascade


def detect_faces(cascade: cv2.CascadeClassifier, bgr: np.ndarray) -> np.ndarray:
    """グレースケール化 → equalizeHist → detectMultiScale。返り値は (N,4)=[x,y,w,h]。"""
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40)
    )
    return np.asarray(faces).reshape(-1, 4)


# ----------------------------------------------------------------------------
# 匿名化の中身（顔 1 個ぶんの矩形を「読めなく」する 3 方式）
# ----------------------------------------------------------------------------
def _gaussian_blur(roi: np.ndarray) -> np.ndarray:
    """顔サイズに比例した強いガウシアンぼかし。カーネルは必ず奇数にする。"""
    h, w = roi.shape[:2]
    k = max(3, int(min(h, w) * BLUR_STRENGTH))
    k = k + 1 if k % 2 == 0 else k  # GaussianBlur のカーネルは奇数必須
    return cv2.GaussianBlur(roi, (k, k), 0)


def _pixelate(roi: np.ndarray) -> np.ndarray:
    """いったん極小に縮小 → 最近傍で元サイズへ拡大してモザイク（ブロック化）にする。"""
    h, w = roi.shape[:2]
    bw = max(1, min(PIXELATE_BLOCKS, w))
    bh = max(1, min(PIXELATE_BLOCKS, h))
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _blackbox(roi: np.ndarray) -> np.ndarray:
    """黒で塗り潰す（情報を完全に消す最も強い匿名化）。"""
    return np.zeros_like(roi)


_METHODS = {"blur": _gaussian_blur, "pixelate": _pixelate, "box": _blackbox}


def anonymize_face(
    bgr: np.ndarray, box, method: str = METHOD, ellipse: bool = USE_ELLIPSE, expand: float = EXPAND
) -> None:
    """画像 bgr の box(=[x,y,w,h]) の顔を **その場で** 匿名化する（破壊的更新）。

    手順:
      1. 検出枠を expand だけ広げ、画像の範囲にクランプ（額・あごまで覆う）。
      2. 切り出した ROI に方式関数（ぼかし/モザイク/黒塗り）を適用。
      3. ellipse=True なら顔の楕円マスクで元 ROI と α 合成し、矩形の角に背景を残さず
         自然に仕上げる。マスクの縁を少しぼかして境界を目立たなくする。
    """
    x, y, w, h = (int(v) for v in box)
    # --- 1. 枠を広げてクランプ ---
    pad_x, pad_y = int(w * expand), int(h * expand)
    H, W = bgr.shape[:2]
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return  # 画面外などの保険
    roi = bgr[y0:y1, x0:x1]

    # --- 2. 方式を適用 ---
    anon = _METHODS.get(method, _gaussian_blur)(roi)

    # --- 3. 楕円マスクで自然に合成（または矩形ごと置換）---
    if not ellipse:
        bgr[y0:y1, x0:x1] = anon
        return
    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), np.uint8)
    cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
    # 縁を少しぼかして境界（ハードエッジ）を目立たなくする
    feather = max(3, (min(rh, rw) // 8) | 1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = roi.astype(np.float32) * (1.0 - alpha) + anon.astype(np.float32) * alpha
    bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


def anonymize_image(
    bgr: np.ndarray, boxes: np.ndarray, method: str = METHOD
) -> np.ndarray:
    """元画像を壊さずコピーし、全 boxes を匿名化した画像を返す。"""
    out = bgr.copy()
    for box in boxes:
        anonymize_face(out, box, method=method)
    return out


# ----------------------------------------------------------------------------
# 入力シーンの用意（実データ優先 → 合成フォールバック）
# ----------------------------------------------------------------------------
def load_scenes():
    """処理対象を (name, bgr, gt_boxes_or_None, is_real) のリストで返す。

    - data/30_.../ に画像があれば、それぞれを実写シーンとして返す（gt_boxes=None）。
    - 無ければ合成の集合写真 1 枚を返す（gt_boxes=各顔の正解位置, is_real=False）。
    """
    files: list[str] = []
    for pat in IMAGE_EXTS:
        files += glob.glob(os.path.join(fl.DATA_DIR, pat))
    files = sorted(files)[:MAX_REAL_IMAGES]
    if files:
        scenes = []
        for f in files:
            bgr = cv2.imread(f)
            if bgr is None:  # 読み込み失敗（非対応形式・壊れ）はスキップ
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            scenes.append((name, bgr, None, True))
        if scenes:
            return scenes
        print(f"[警告] {fl.DATA_DIR}/ の画像を読めませんでした。合成にフォールバックします。")

    # --- 合成フォールバック: 6 人 × 3 枚 = 18 顔を 1 枚に並べた集合写真 ---
    images, _, _ = fl.make_face_dataset(n_ids=6, n_per_id=3, seed=7)
    canvas, gt_boxes = fl.compose_group_photo(images, cols=6, return_boxes=True)
    return [("synthetic_scene", canvas, gt_boxes, False)]


def resolve_boxes(cascade, name, bgr, gt_boxes, is_real):
    """そのシーンで実際に匿名化する顔の矩形を決める。

    実写は **検出結果をそのまま** 使う（検出ドリブン＝ツール本来の挙動）。
    合成は検出が弱い（ドメインギャップ）ことがあるので、検出数が正解の半分未満なら
    正解位置にフォールバックして匿名化を必ず実演する（exit 0 を保証）。
    返り値は (boxes, n_detected, used_fallback)。
    """
    det = detect_faces(cascade, bgr)
    if is_real or gt_boxes is None:
        return det, len(det), False
    used_fallback = len(det) < MIN_MATCH_RATIO * len(gt_boxes)
    if used_fallback:
        print(
            f"  [{name}] 合成顔は検出が弱い（{len(det)}/{len(gt_boxes)}）= ドメインギャップ。"
            "正解位置にフォールバックして匿名化します。"
        )
        return gt_boxes, len(det), True
    return det, len(det), False


# ----------------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------------
def _label_bar(width: int, text: str) -> np.ndarray:
    """モンタージュ上部に貼る ASCII ラベル帯（headless で安全に焼き込む）。"""
    bar = np.full((24, width, 3), 245, np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def save_before_after(scenes_vis, path: str) -> None:
    """各シーンの「元画像｜匿名化後」を縦に積んだ Before/After モンタージュを保存。"""
    rows = []
    target_w = max(orig.shape[1] for _, orig, _ in scenes_vis)
    target_w = min(target_w, 560)  # 大きすぎる実写は適度に縮小
    for name, orig, anon in scenes_vis:
        pair = []
        for tag, img in (("original", orig), (f"anonymized ({METHOD})", anon)):
            scale = target_w / img.shape[1]
            r = cv2.resize(img, (target_w, int(round(img.shape[0] * scale))))
            pair.append(np.vstack([_label_bar(target_w, f"{name}: {tag}"), r]))
        # 元と匿名化後を縦に並べ、シーン間に区切り帯を入れる
        rows.append(np.vstack(pair))
    sep = np.full((8, target_w, 3), 180, np.uint8)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.vstack([canvas, sep, r])
    cv2.imwrite(path, canvas)


def save_methods_figure(face_crop: np.ndarray, path: str) -> None:
    """同じ顔に 3 方式（blur/pixelate/box）を当てた比較図（matplotlib, Agg, ASCII タイトル）。"""
    panels = [("original", face_crop)]
    for m, fn in _METHODS.items():
        panels.append((m, fn(face_crop)))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2(BGR)→RGB を忘れない
        ax.set_title(title)  # ASCII タイトル（日本語の豆腐化を回避）
        ax.axis("off")
    fig.suptitle("Face anonymization methods")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)
    cascade = load_cascade()

    print("=" * 68)
    print("顔の自動ぼかし（匿名化）ツール")
    print(f"  方式: METHOD={METHOD}  楕円マスク={USE_ELLIPSE}  枠拡張={EXPAND}")

    scenes = load_scenes()
    is_real = scenes[0][3]
    print(f"  入力: {'実写 ' + str(len(scenes)) + ' 枚' if is_real else '合成集合写真 1 枚'}")

    scenes_vis = []     # Before/After モンタージュ用 (name, orig, anon)
    report_imgs = []    # JSON 用
    methods_crop = None  # 方式比較図に使う代表的な顔の切り出し
    total_anon = 0

    for name, bgr, gt_boxes, real in scenes:
        boxes, n_det, used_fb = resolve_boxes(cascade, name, bgr, gt_boxes, real)
        anon = anonymize_image(bgr, boxes, method=METHOD)
        total_anon += len(boxes)

        print(f"  [{name}] 検出={n_det}  匿名化={len(boxes)}"
              + ("  (正解位置フォールバック)" if used_fb else ""))

        scenes_vis.append((name, bgr, anon))
        report_imgs.append({
            "name": name,
            "is_real": bool(real),
            "faces_detected": int(n_det),
            "faces_anonymized": int(len(boxes)),
            "used_fallback_gt": bool(used_fb),
        })

        # 実写は各匿名化フル画像も個別保存（そのまま公開に使える成果物）
        if real:
            outp = os.path.join(fl.OUT_DIR, f"use_case_anon_{name}.png")
            cv2.imwrite(outp, anon)

        # 方式比較図用に、最初に見つかった顔を 1 つ切り出しておく
        if methods_crop is None and len(boxes) > 0:
            x, y, w, h = (int(v) for v in boxes[0])
            pad = int(max(w, h) * 0.15)
            H, W = bgr.shape[:2]
            cx0, cy0 = max(0, x - pad), max(0, y - pad)
            cx1, cy1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = bgr[cy0:cy1, cx0:cx1]
            if crop.size > 0:
                methods_crop = crop.copy()

    # --- 可視化・レポート出力 ---
    ba_path = os.path.join(fl.OUT_DIR, "use_case_anonymized.png")
    save_before_after(scenes_vis, ba_path)

    methods_path = None
    if methods_crop is not None:
        methods_path = os.path.join(fl.OUT_DIR, "use_case_methods.png")
        save_methods_figure(methods_crop, methods_path)

    report = {
        "tool": "face_anonymizer",
        "method": METHOD,
        "ellipse": USE_ELLIPSE,
        "expand": EXPAND,
        "is_real_input": bool(is_real),
        "num_images": len(scenes),
        "total_faces_anonymized": int(total_anon),
        "images": report_imgs,
    }
    report_path = os.path.join(fl.OUT_DIR, "use_case_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- サマリ ---
    print(f"\n  匿名化した顔の合計: {total_anon} 個")
    print(f"  Before/After : {ba_path}")
    if methods_path:
        print(f"  方式比較図   : {methods_path}")
    print(f"  レポート JSON: {report_path}")
    if not is_real:
        print("  ※ 合成入力で完走。data/30_face_detection_recognition/ に実写を置けば、")
        print("    検出 → ぼかしが本来の検出ドリブンで動きます（Haar は実写で当たりやすい）。")
    print("\n発展: 検出を DNN/MediaPipe に、対象を動画・ナンバープレートへ広げると実用度が上がります。")


if __name__ == "__main__":
    main()
