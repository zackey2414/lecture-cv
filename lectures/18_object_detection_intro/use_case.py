"""use_case: 物体・人数カウンター — 画像中の指定クラスを数えて「注記」を焼き込む小ツール。

これは「動く出発点」です。第18回で学んだ検出パイプライン（4点セット →[0,1] float CHW
リスト入力 → eval+inference_mode → score 閾値 → クラス別 NMS → 可視化）を、評価ベンチ
ではなく **現実の小ツール** に仕立て直したものです。具体的には:

  data/18_object_detection_intro/ にある画像（複数可）について、指定クラス
  （既定 person / car）の **検出数を数え**、画像に箱と「person: 3 / car: 1 / TOTAL: 4」の
  ような **カウントのバナーを焼き込み**、フォルダ全体の合計を JSON に書き出します。

  来店人数のざっくり計測、駐車場の在車台数チェック、棚の物品個数の目視補助、イベント会場の
  混雑度モニタなど、「写真を見て “何が何個あるか” を機械に数えさせたい」場面の出発点です。

────────────────────────────────────────────────────────────────────────
data の置き方（実写を入れると一気に実用になる）
  mkdir -p data/18_object_detection_intro
  cp ~/Pictures/*.jpg data/18_object_detection_intro/      # 人や車が写った普通の写真でOK
  uv run python lectures/18_object_detection_intro/use_case.py
  → data/ にある画像を **全部** 処理して、1枚ずつ注記画像を保存し、合計を集計します。
  → data/ が空なら合成シーン1枚にフォールバックして必ず exit 0 します（合成は検出が弱く
    カウントが 0 になりがちですが、それは故障ではなく「ドメインギャップ＝正常」。実写を置けば
    person 0.99 のような妥当な検出になり、そのまま実用カウンタになります）。

調整（環境変数。コードを書き換えずに挙動を変えられる）
  USE_CASE_CLASSES="person,car,bus"  数えたいクラス（COCO 名・カンマ区切り。既定 person,car）
  USE_CASE_SCORE=0.5                  これ未満の確信度は数えない（実写は 0.5 前後が定番）
  例) USE_CASE_CLASSES="person" USE_CASE_SCORE=0.5 \
        uv run python lectures/18_object_detection_intro/use_case.py

mini_project.py との違い（役割を取り違えないこと）
  - mini_project.py … 「検出パイプライン × 自作 mAP@0.5」を torchmetrics で検算する **評価ベンチ**。
    出力は PR 曲線や mAP の数値で、目的は “検出器の良し悪しを測る” こと。
  - use_case.py（これ）… 測るのではなく **数えて注記する現実の小ツール**。出力は業務に渡せる
    「カウント表（JSON）＋注記を焼き込んだ画像」。アノテーションも mAP も要らず、
    フォルダに写真を放り込めばカウント結果が返る、という単機能の完結ツールに振ってあります。

拡張アイデア（ここから練習を広げる）
  - 動画対応: cv2.VideoCapture で N フレームおきに切り出し、時系列のカウント推移を CSV/折れ線に。
  - ROI カウント: 関心領域（駐車枠・レジ前など）の矩形を決め、その中に中心が入る箱だけ数える。
  - 混雑アラート: TOTAL がしきい値を超えたら "CROWDED" を注記/ログ出力（簡易の人数オーバー検知）。
  - 精度を上げる: ssdlite320 → fasterrcnn_resnet50_fpn や RT-DETR(PekingU/rtdetr_v2_r18vd) に差し替え
    （小物体・遠景の人に強くなる。CPU では遅くなるトレードオフ）。
  - 集計の充実: クラス別カウントを棒グラフ化、複数フォルダ（日付別）で推移比較、CSV エクスポート。

実行: uv run python lectures/18_object_detection_intro/use_case.py
出力: lectures/18_object_detection_intro/outputs/
  - use_case_count_<n>.png   … 入力ごとの注記画像（箱＋カウントのバナー）
  - use_case_overview.png     … 全入力を並べたサムネイル（注記入り）
  - use_case_counts.json      … 入力別カウントと、フォルダ全体の合計
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# 同フォルダの共通道具箱（detection_helpers）を import できるようにパスを通す。
# ※ 01〜03 のような数字始まりスクリプトは import 不可なので、共通部品は helper から借りる。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import detection_helpers as H  # noqa: E402  device 判定 / 合成画像 / 後処理 / 可視化 / 保存

# CPU スレッドを物理コア数程度まで使うと推論が安定する（任意）。
torch.set_num_threads(min(8, torch.get_num_threads()))

# 既定値（環境変数で上書き可能）。合成画像は検出が弱いので閾値は低めの 0.30 を既定にする。
DEFAULT_CLASSES = "person,car"
DEFAULT_SCORE = 0.30
NMS_IOU = 0.50
MAX_IMAGES = 8  # data/ に大量に置かれても CPU で完走できるよう処理枚数の上限を設ける。


def parse_target_classes() -> list[str]:
    """数えたいクラス名のリストを環境変数 USE_CASE_CLASSES から取得する（既定 person,car）。

    COCO のクラス名（小文字）で指定する。例: "person,car,bus,truck"。
    空要素・前後空白は捨てる。指定が空なら既定に戻す。
    """
    raw = os.environ.get("USE_CASE_CLASSES", DEFAULT_CLASSES)
    classes = [c.strip() for c in raw.split(",") if c.strip()]
    return classes or DEFAULT_CLASSES.split(",")


def parse_score_thresh() -> float:
    """score 閾値を環境変数 USE_CASE_SCORE から取得する（不正値なら既定にフォールバック）。"""
    try:
        return float(os.environ.get("USE_CASE_SCORE", DEFAULT_SCORE))
    except ValueError:
        return DEFAULT_SCORE


def load_all_inputs() -> list[tuple[Image.Image, str]]:
    """data/<module_id>/ の画像を全部（最大 MAX_IMAGES 枚）読む。無ければ合成1枚。

    helper の load_input_image() は「先頭1枚」だが、カウンタは複数枚をまとめて集計したい
    （監視フレーム群・棚の連続写真など）ので、ここではフォルダ全体を対象にする。
    返り値: [(PIL.Image[RGB], source名), ...]  source は "data:<file>" か "synthetic"。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if H.DATA_DIR.is_dir():
        files = sorted(p for p in H.DATA_DIR.iterdir() if p.suffix.lower() in exts)
        if files:
            return [(Image.open(p).convert("RGB"), f"data:{p.name}") for p in files[:MAX_IMAGES]]
    # 実写が無ければ合成シーン1枚で必ず完走する（COCO 検出器は合成に弱くカウント0になりがち＝正常）。
    return [(H.make_scene_image(), "synthetic")]


def build_counter_model(device: torch.device):
    """カウンタ用の検出器（torchvision の最軽量 ssdlite320）を 4点セットで用意する。

    返り値: (weights, model)。model は eval + device 済み。
    CPU で速い ssdlite を既定にする。精度が欲しければ fasterrcnn 等へ差し替え（docstring 参照）。
    """
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT  # ① COCO 学習済み重み
    model = ssdlite320_mobilenet_v3_large(weights=weights)   # ② 本体
    model.to(device).eval()                                  # ★推論モードへ（必須）
    return weights, model


def count_one(model, weights, image: Image.Image, device: torch.device,
              targets: list[str], score_thresh: float) -> dict:
    """1枚を検出し、指定クラスだけに絞ってカウントする。

    返り値:
      counts   … {クラス名: 個数}（targets の順、検出ゼロのクラスも 0 で含む）
      total    … 指定クラスの合計個数
      _boxes / _names / _scores … 注記描画に使う、指定クラスに絞った検出（json には出さない）
    """
    preproc = weights.transforms()           # PIL → [0,1] float CHW（正規化はモデル内部）
    categories = weights.meta["categories"]  # COCO 91クラス（index0 = __background__）

    x = preproc(image).to(device)
    with torch.inference_mode():             # ★勾配・メモリを止める
        raw = model([x])[0]                  # ★入力は「テンソルのリスト」。出力は {boxes,labels,scores}

    # 共通後処理: score 閾値 → クラス別 NMS（detection_helpers を借用）。
    boxes, scores, labels = H.apply_threshold_nms(
        raw["boxes"].cpu(), raw["scores"].cpu(), raw["labels"].cpu(),
        score_thresh=score_thresh, iou_thresh=NMS_IOU,
    )

    # label(int) → クラス名（index0=背景なので categories[int(label)] で引く）。
    all_names = [categories[int(i)] for i in labels]

    # 指定クラスだけ残す。targets に無いクラス（COCO 全部）は数えない。
    target_set = set(targets)
    keep_idx = [i for i, n in enumerate(all_names) if n in target_set]
    kept_boxes = boxes[keep_idx] if keep_idx else boxes[:0]
    kept_scores = scores[keep_idx] if keep_idx else scores[:0]
    kept_names = [all_names[i] for i in keep_idx]

    # targets の順番でカウント（検出ゼロのクラスも 0 として明示する＝レポートとして読みやすい）。
    counts = {name: kept_names.count(name) for name in targets}
    total = sum(counts.values())
    return {
        "counts": counts,
        "total": total,
        "_boxes": kept_boxes,
        "_names": kept_names,
        "_scores": kept_scores,
    }


def draw_count_banner(image: Image.Image, counts: dict[str, int], total: int) -> Image.Image:
    """画像の左上に「カウントの注記バナー」を半透明で焼き込んで返す（imshow は使わない）。

    クラス名は COCO の英語名（ASCII）なので、headless の DejaVu フォントでも豆腐にならない。
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")          # RGBA で描くと半透明の下地が描ける
    font = ImageFont.load_default()             # 追加フォント不要（環境非依存）

    lines = [f"{name}: {n}" for name, n in counts.items()]
    lines.append(f"TOTAL: {total}")

    # 文字サイズを測ってバナーの大きさを決める（textbbox は (l,t,r,b) を返す）。
    pad, gap = 6, 3
    widths, heights = [], []
    for ln in lines:
        l, t, r, b = draw.textbbox((0, 0), ln, font=font)
        widths.append(r - l)
        heights.append(b - t)
    box_w = max(widths) + pad * 2
    box_h = sum(heights) + gap * (len(lines) - 1) + pad * 2

    # 半透明の黒い下地 → その上に白文字（混雑時は赤）で重ねる。
    draw.rectangle([4, 4, 4 + box_w, 4 + box_h], fill=(0, 0, 0, 160))
    y = 4 + pad
    for ln, h in zip(lines, heights):
        color = (255, 80, 80) if ln.startswith("TOTAL") and total > 0 else (255, 255, 255)
        draw.text((4 + pad, y), ln, fill=color, font=font)
        y += h + gap
    return img


def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    targets = parse_target_classes()
    score_thresh = parse_score_thresh()
    inputs = load_all_inputs()

    print(f"device = {device}   targets = {targets}   score_thresh = {score_thresh}")
    print(f"inputs = {len(inputs)} 枚  ({', '.join(s for _img, s in inputs)})\n")

    # 検出器のロード（重み DL 不可などでも全体を落とさず、空集計で exit 0 にする）。
    try:
        weights, model = build_counter_model(device)
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 検出器をロードできませんでした（{type(e).__name__}: {e}）。")
        print("       ネット未接続で重みが未キャッシュの可能性。空の集計を書き出して終了します。")
        (out / "use_case_counts.json").write_text(
            json.dumps({"error": f"{type(e).__name__}: {e}", "targets": targets},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    panel_items: list[tuple[Image.Image, str]] = []
    per_image: list[dict] = []
    grand_total: dict[str, int] = {name: 0 for name in targets}

    for idx, (image, source) in enumerate(inputs, start=1):
        res = count_one(model, weights, image, device, targets, score_thresh)

        # 注記画像 = 「箱（指定クラスのみ）」＋「カウントのバナー」。
        boxed = H.draw_detections(image, res["_boxes"], res["_names"], res["_scores"], color="red")
        annotated = draw_count_banner(boxed, res["counts"], res["total"])
        annotated.save(out / f"use_case_count_{idx}.png")

        # overview パネル用に縮小版とタイトル（ASCII）を貯める。
        count_str = " ".join(f"{k}={v}" for k, v in res["counts"].items())
        panel_items.append((annotated, f"#{idx} {source}\n{count_str} total={res['total']}"))

        # 集計を更新。
        for name, n in res["counts"].items():
            grand_total[name] += n
        per_image.append({"index": idx, "source": source,
                          "counts": res["counts"], "total": res["total"]})

        print(f"[#{idx}] {source}: {count_str}  total={res['total']}")

    # 全入力を1枚に並べた俯瞰図（注記入り）。
    H.save_panel(panel_items, out / "use_case_overview.png",
                 "Object / People Counter (use case): boxes + count banner per image")

    summary = {
        "model": "torchvision/ssdlite320_mobilenet_v3_large",
        "targets": targets,
        "score_thresh": score_thresh,
        "nms_iou": NMS_IOU,
        "n_images": len(inputs),
        "per_image": per_image,
        "grand_total": grand_total,                       # クラス別の合計
        "grand_total_all": sum(grand_total.values()),     # 全クラス合計
        "note": "合成画像は検出が弱くカウント0になりがち（ドメインギャップ＝正常）。"
                "data/18_object_detection_intro/ に実写を置くと実用カウンタになる。",
    }
    (out / "use_case_counts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n合計（{len(inputs)}枚）: " +
          ", ".join(f"{k}={v}" for k, v in grand_total.items()) +
          f"  全クラス合計={sum(grand_total.values())}")
    print("ポイント:")
    print(" - 検出を指定クラスに絞って数え、画像にカウントを焼き込む単機能ツール（mAPは測らない）。")
    print(" - 合成だとカウント0になりがち＝正常。data/18_object_detection_intro/ に実写を置くと実用に。")
    print(f"saved: {out / 'use_case_overview.png'}, {out / 'use_case_counts.json'}")


if __name__ == "__main__":
    main()
