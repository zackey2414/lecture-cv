"""01: 顔検出 — OpenCV Haar カスケード（＋任意で DNN 顔検出）。

学ぶこと:
  - `cv2.CascadeClassifier` で Haar カスケード分類器を読み込み、`detectMultiScale` で
    画像から顔の矩形を得る、という古典顔検出の基本フロー。
  - 主要パラメータ `scaleFactor` / `minNeighbors` / `minSize` が「見逃し ↔ 誤検出」の
    トレードオフをどう動かすかを、パラメータ掃引で体感する。
  - ★ドメインギャップ: Haar は実写の顔で学習されている。合成顔は検出される場合もされない
    場合もある（本講座の合成集合写真ではそこそこ当たるが、誤検出や取りこぼしも出る）。
    実写を data/30_face_detection_recognition/ に置けばそちらでも検出を試す。
  - DNN 顔検出（res10 SSD）は精度・頑健性が高いが、別途モデル重みが要る。重みが無ければ
    案内だけ出して **必ず exit 0** を保つ（import/ファイル存在を try/except でガード）。

実行:
  uv run python lectures/30_face_detection_recognition/01_face_detection.py
結果は lectures/30_face_detection_recognition/outputs/ に保存（画面表示はしない・headless）。
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

import face_lab as fl


def load_face_cascade() -> cv2.CascadeClassifier:
    """OpenCV 同梱の正面顔 Haar カスケードを読み込む（重み不要・即動く）。"""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():  # 読み込み失敗はここで気づけるようにする
        raise RuntimeError(f"カスケードを読み込めません: {path}")
    return cascade


def detect_faces(
    cascade: cv2.CascadeClassifier,
    bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: int = 40,
) -> np.ndarray:
    """グレースケール化して detectMultiScale を呼ぶ。返り値は (N,4) の [x,y,w,h]。

    Haar はグレースケールの明暗パターン（暗い目・明るい額…）で顔を探すので、必ず
    グレースケールへ変換してから渡す。ヒストグラム平坦化で明暗差を強調すると安定する。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,  # 画像ピラミッドの縮小率。1.05=細かい/遅い, 1.3=粗い/速い
        minNeighbors=min_neighbors,  # 近接検出の最低数。大きいほど誤検出↓・見逃し↑
        minSize=(min_size, min_size),  # これより小さい顔は無視
    )
    return np.asarray(faces).reshape(-1, 4)


def draw_boxes(bgr: np.ndarray, faces: np.ndarray, color=(0, 200, 0)) -> np.ndarray:
    """検出矩形を描いた BGR 画像のコピーを返す。"""
    out = bgr.copy()
    for x, y, w, h in faces:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    return out


def sweep_parameters(cascade: cv2.CascadeClassifier, bgr: np.ndarray) -> None:
    """scaleFactor × minNeighbors を掃引し、検出数の変化を表で見せる。"""
    print("  scaleFactor / minNeighbors を掃引した検出数（顔は合計 24 枚）:")
    print("            mN=2   mN=3   mN=5")
    for sf in (1.05, 1.1, 1.2):
        counts = [len(detect_faces(cascade, bgr, sf, mn)) for mn in (2, 3, 5)]
        print(f"    sf={sf:<4}  {counts[0]:>4}   {counts[1]:>4}   {counts[2]:>4}")
    print("  → minNeighbors を上げると誤検出は減るが顔も取りこぼす。scaleFactor を下げると")
    print("    細かく探して当たりやすいが遅くなる。唯一の正解は無く、用途で決める。")


def try_dnn_face_detection(bgr: np.ndarray) -> np.ndarray | None:
    """任意: OpenCV DNN(res10 SSD) 顔検出。モデル重みが data/ に無ければ None を返す。

    重み（res10_300x300_ssd_iter_140000.caffemodel と deploy.prototxt）は別途入手が必要。
    無くても落とさず案内だけ出す（必ず exit 0 を保つための設計）。
    """
    proto = os.path.join(fl.DATA_DIR, "deploy.prototxt")
    model = os.path.join(fl.DATA_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    if not (os.path.exists(proto) and os.path.exists(model)):
        print("\n[任意] OpenCV DNN 顔検出（res10 SSD）はモデル重みが見つからないためスキップ。")
        print(f"  使う場合は次の 2 ファイルを {fl.DATA_DIR}/ に置いてください:")
        print("    - deploy.prototxt / res10_300x300_ssd_iter_140000.caffemodel")
        print("  DNN 検出は照明・角度・小顔に強く、実写では Haar より頑健です。")
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(proto, model)
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        det = net.forward()  # (1,1,N,7): [_, _, conf, x1,y1,x2,y2]（座標は 0..1 正規化）
        boxes = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        print(f"\n[任意] DNN 顔検出: {len(boxes)} 件（conf>=0.5）")
        return np.asarray(boxes).reshape(-1, 4)
    except Exception as e:  # noqa: BLE001
        print(f"\n[任意] DNN 顔検出に失敗（{type(e).__name__}）。スキップします。")
        return None


def detect_on_real_images(cascade: cv2.CascadeClassifier) -> None:
    """data/30_.../ に実写があれば、そこでも顔検出を試す（無ければ何もしない）。"""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: list[str] = []
    for p in patterns:
        files += glob.glob(os.path.join(fl.DATA_DIR, p))
    if not files:
        print(f"\n[実写] {fl.DATA_DIR}/ に画像が無いのでスキップ（置けば自動で検出します）。")
        return
    print(f"\n[実写] {len(files)} 枚で顔検出を実行:")
    for f in sorted(files):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        faces = detect_faces(cascade, bgr)
        out = draw_boxes(bgr, faces, color=(0, 165, 255))
        name = "real_" + os.path.splitext(os.path.basename(f))[0] + "_detected.png"
        cv2.imwrite(os.path.join(fl.OUT_DIR, name), out)
        print(f"    {os.path.basename(f)}: {len(faces)} 件 → {name}")


def main() -> None:
    os.makedirs(fl.OUT_DIR, exist_ok=True)

    # 合成の集合写真を作る（8 人 × 3 枚 = 24 顔を格子に並べる）
    images, labels, _ = fl.make_face_dataset(n_ids=8, n_per_id=3, seed=0)
    canvas = fl.compose_group_photo(images, cols=6)
    cv2.imwrite(os.path.join(fl.OUT_DIR, "01_group_photo.png"), canvas)
    print(f"合成集合写真: {canvas.shape[1]}x{canvas.shape[0]} に 24 顔を配置")

    cascade = load_face_cascade()
    print("\n[Haar] 正面顔カスケードで検出")
    faces = detect_faces(cascade, canvas, scale_factor=1.05, min_neighbors=5, min_size=40)
    print(f"  既定設定(sf=1.05, mN=5)の検出数: {len(faces)} 件 / 顔は 24 枚")
    annotated = draw_boxes(canvas, faces)
    cv2.imwrite(os.path.join(fl.OUT_DIR, "01_haar_detected.png"), annotated)

    print()
    sweep_parameters(cascade, canvas)

    print("\n[ドメインギャップ] Haar は実写の顔で学習されている。合成顔は当たることもあるが、")
    print("  誤検出・取りこぼしが出やすい。検出の良し悪しは『学習データと入力のドメイン一致』が")
    print(
        "  支配的で、合成 → 実写の乖離（domain gap）が精度を決める。実写を data/ に置いて確かめよう。"
    )

    # 任意の DNN 顔検出（重みがあれば実行、無ければ案内のみ）
    dnn_boxes = try_dnn_face_detection(canvas)
    if dnn_boxes is not None and len(dnn_boxes):
        cv2.imwrite(
            os.path.join(fl.OUT_DIR, "01_dnn_detected.png"),
            draw_boxes(canvas, dnn_boxes, color=(255, 120, 0)),
        )

    # 実写があればそこでも検出
    detect_on_real_images(cascade)

    print(f"\n完了。lectures/{fl.MODULE_ID}/outputs/01_*.png を確認してください。")


if __name__ == "__main__":
    main()
