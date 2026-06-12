"""use_case.py — 任意物体ファインダー（自由文で物を探して枠表示する実用最小ツール）。

【何をするツールか】
  コマンドラインに『a yellow umbrella』『a person with a backpack』のような
  自由文（任意ラベル）を渡すと、1枚の画像からその物体を検出して枠で囲み、
  「見つかった / 見つからない」を一覧で返す小さなオープン語彙検出ツールです。
  固定クラス（COCO 80種）の検出器と違い、探す対象を実行時に文章で差し替えられるのが肝。
  本章 §2〜§4 の「候補ラベルを渡す → post_process_grounded_object_detection で box を取る」
  をそのまま実用の入口にした形で、防犯カメラの『不審物探し』や写真整理の『写ってる物探し』の
  最小自作版にあたります。

【実データで使うには（ここが本題）】
  data/20_open_vocabulary_detection/ に自分の .png/.jpg を1枚以上置くだけで、
  その最初の1枚が検出対象になります（合成シーンは使われなくなります）。
  画像が無いときは色×形の合成シーン（赤い円・青い四角・緑の三角・黄色い円）に
  自動フォールバックして必ず完走します（exit 0）。合成は『色・形』の語にはよく反応しますが、
  『umbrella』『person』のような実世界の語は当然ヒットしません（=「見つからない」を体験する用）。
  実写を data/ に置けば、自由文で本物の物体を探す実用ツールとして動きます。

【mini_project.py との違い】
  mini_project.py は章の総合課題（OWL-ViT vs OWLv2 を AP@0.5・mAP@[.5:.95]・F1 で
  厳密比較する評価レポート＝ベンチ寄りの完成形）。
  こちらは『現実の小ツール』として完結し、自由文クエリを CLI から渡して物を探すだけ。
  評価やスイープは無く、コピーして自分の物体ファインダーの出発点にできる形にしてあります。

【使い方】
  # 既定のデモクエリ（合成シーンに有る語＋無い語）で動かす
  uv run python lectures/20_open_vocabulary_detection/use_case.py
  # 自分の探したい物を自由文で渡す（複数可・スペースを含むフレーズは引用符で）
  uv run python lectures/20_open_vocabulary_detection/use_case.py "a red circle" "a yellow umbrella"

  # 使う検出器を選ぶ（環境変数）。既定は精度重視の OWLv2。
  OVD_MODEL=owlvit ...   # CPU で軽く速く試す（スコア低め）
  OVD_MODEL=owlv2  ...   # 精度重視（既定。スコア高め・やや重い）
  OVD_MODEL=gdino  ...   # Grounding DINO（自由文フレーズ向き。キャプション形式に整形して渡す）
  # 検出のしきい値を上書き（小さいほど拾う・過検出も増える）
  OVD_THRESHOLD=0.15 ...

【出力（すべて outputs/20_open_vocabulary_detection/ に保存）】
  use_case_finder.png    検出ボックス（クエリ名＋スコア付き）を重ねた注釈画像
  use_case_finder.json   クエリごとの「見つかった/件数/最高スコア/box」一覧

【拡張アイデア（練習）】
  - 複数画像フォルダを総当りし、「指定物が写っている画像」を検索する画像検索ツールへ。
  - 検出 box を SAM に input_boxes として渡し、領域をマスク切り出し（Grounded-SAM・第23回）。
  - 計数（カウント）機能: 同じラベルの box 数を数えて在庫・人数カウントにする。
  - しきい値を自分のデータで較正（§8 の F1 最大点）。固定値ではなく検証データで決める。
  - 連番画像/動画フレームに回し、指定物が現れたフレームだけ通知するアラート化。
  - プロンプト工夫: "a {x}" / "a photo of a {x}" の言い回しで拾い方が変わる（§発展・第16回）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# digit 始まりの番号付きスクリプト（01_*.py 等）は import 不可だが、ovd_helpers は import できる。
# 同じディレクトリを import パスへ追加して道具箱（device 判定・モデルロード・検出ラッパ・描画）を借りる。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ovd_helpers as H  # noqa: E402

# 引数なしのときのデモクエリ。合成シーンに『有る』語2つ＋『無い』語1つを混ぜ、
# 「見つかる」「見つからない」の両方を1回で体験できるようにしてある。
DEMO_QUERIES = [
    "a red circle",       # 合成シーンに有る → 見つかる
    "a green triangle",   # 合成シーンに有る → 見つかる
    "a yellow umbrella",  # 合成シーンに無い → 見つからない（実写を置けば探せる対象）
]

# モデル選択（環境変数 OVD_MODEL）。既定は精度重視の OWLv2。
# 値ごとに (種別, モデルID, 既定しきい値) を引く。owlvit は速さ重視・スコア低めなので閾値も低め。
_MODEL_TABLE = {
    "owlvit": ("owl", H.OWLVIT_ID, 0.1),
    "owlv2": ("owl", H.OWLV2_ID, 0.2),
    "gdino": ("gdino", H.GDINO_ID, 0.3),
}


def resolve_model() -> tuple[str, str, float]:
    """環境変数 OVD_MODEL / OVD_THRESHOLD から (種別, モデルID, しきい値) を決める。

    未知の値が来たら既定の owlv2 にフォールバックする（タイポで落とさない）。
    """
    key = os.environ.get("OVD_MODEL", "owlv2").strip().lower()
    kind, model_id, default_thr = _MODEL_TABLE.get(key, _MODEL_TABLE["owlv2"])
    thr_env = os.environ.get("OVD_THRESHOLD")
    threshold = float(thr_env) if thr_env else default_thr
    return kind, model_id, threshold


def find_image_files() -> list[pathlib.Path]:
    """data/<module_id>/ にある画像ファイル一覧を返す（無ければ空リスト）。

    実データを使っているか合成にフォールバックしたかを利用者に明示するため自前で確認する。
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not H.DATA_DIR.is_dir():
        return []
    return sorted(p for p in H.DATA_DIR.iterdir() if p.suffix.lower() in exts)


def _normalize(text: str) -> str:
    """ラベル文字列を比較用に正規化（小文字・末尾ピリオド除去・冠詞 a/an/the を落とす）。

    OWL は候補ラベルそのものを返すが、Grounding DINO は "red circle" のような断片を返すため、
    冠詞を落として『包含』でクエリへ対応づけられるようにしておく。
    """
    s = text.strip().lower().rstrip(".")
    for article in ("a ", "an ", "the "):
        if s.startswith(article):
            s = s[len(article):]
            break
    return s.strip()


def assign_to_query(label: str, queries: list[str]) -> str | None:
    """検出ラベルを、最もよく対応するクエリへ割り当てる（無ければ None）。

    完全一致を最優先し、無ければ正規化後の包含関係（断片対応）で拾う。
    """
    nl = _normalize(label)
    norm_q = [(_normalize(q), q) for q in queries]
    for nq, q in norm_q:  # まず完全一致
        if nl == nq:
            return q
    for nq, q in norm_q:  # 次に包含（gdino の断片ラベル用）
        if nl and (nl in nq or nq in nl):
            return q
    return None


def detect(kind: str, model_id: str, image, queries: list[str], threshold: float):
    """選んだ検出器で1枚の画像を検出し、(boxes(N,4 xyxy), scores(N,), labels(list[str])) を返す。

    OWL は候補ラベルのリストをそのまま渡す。Grounding DINO は『小文字＋ピリオド区切り』の
    キャプションに整形してから渡す（書式が崩れると検出が安定しない・§5）。
    """
    processor, model = H.load_zero_shot_detector(model_id)
    if kind == "gdino":
        caption = H.labels_to_caption(queries)  # 例: "a red circle. a yellow umbrella."
        boxes, scores, text_labels = H.detect_gdino(
            processor, model, image, caption, box_threshold=threshold, text_threshold=0.25
        )
    else:  # owl（OWL-ViT / OWLv2 共通）
        boxes, scores, text_labels = H.detect_owl(
            processor, model, image, queries, threshold=threshold
        )
    return boxes, scores, text_labels


def summarize(queries, boxes, scores, text_labels) -> tuple[list[dict], list[str]]:
    """検出をクエリ単位に集計し、(クエリ別レポート, 描画用の表示ラベル) を返す。

    各クエリについて「見つかったか・件数・最高スコア・box」を作る。
    表示ラベルは、検出を割り当てたクエリ名（断片より読みやすい）に揃える。
    """
    # クエリ別バケツに検出を仕分ける。
    buckets: dict[str, list[int]] = {q: [] for q in queries}
    display_labels = list(text_labels)
    for i, lab in enumerate(text_labels):
        q = assign_to_query(lab, queries)
        if q is not None:
            buckets[q].append(i)
            display_labels[i] = q  # 図には割り当て先のクエリ名を出す

    report = []
    for q in queries:
        idx = buckets[q]
        if idx:
            best = max(idx, key=lambda i: float(scores[i]))
            report.append({
                "query": q,
                "found": True,
                "count": len(idx),
                "best_score": round(float(scores[best]), 3),
                "best_box": [round(float(v), 1) for v in boxes[best]],
            })
        else:
            report.append({"query": q, "found": False, "count": 0,
                           "best_score": 0.0, "best_box": None})
    return report, display_labels


def main(argv: list[str]) -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    kind, model_id, threshold = resolve_model()

    # --- 1) 探す対象画像（実データ優先・合成フォールバック）---
    files = find_image_files()
    image, _gt_boxes, _gt_labels = H.load_user_or_synthetic()
    source = (f"data/{H.MODULE_ID}/（実画像 {len(files)} 枚の先頭）"
              if files else "合成シーン（色×形・実世界の語はヒットしない）")

    # --- 2) クエリ（引数があればそれ、無ければデモ）---
    queries = argv[1:] if len(argv) > 1 else DEMO_QUERIES

    print(f"device={device}  model={model_id}  threshold={threshold}")
    print(f"source={source}  size={image.width}x{image.height}")
    print(f"queries={queries}")

    # --- 3) 検出（重み DL や環境要因で失敗しても講座が止まらないよう包む）---
    try:
        boxes, scores, text_labels = detect(kind, model_id, image, queries, threshold)
        ok = True
        err = None
    except Exception as e:  # noqa: BLE001
        # モデルをロード/推論できない環境でも、空の結果を出して必ず exit 0 にする。
        print(f"[フォールバック] 検出器を実行できませんでした: {type(e).__name__}: {e}")
        print("注釈なし画像とメモだけ保存して終了します（exit 0）。重い場合は GPU 推奨。")
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        text_labels = []
        ok = False
        err = f"{type(e).__name__}: {e}"

    # --- 4) クエリ単位に集計 ---
    report, display_labels = summarize(queries, boxes, scores, text_labels)

    # --- 5) 注釈画像を保存（道具箱の描画を再利用。GT 線は出さない実用表示）---
    title = f"object finder ({os.environ.get('OVD_MODEL', 'owlv2')}, threshold={threshold})"
    H.draw_detections(image, boxes, display_labels, scores, title, out / "use_case_finder.png")

    # --- 6) コンソール出力 + JSON 保存 ---
    print("\n--- 検索結果 ---")
    for r in report:
        if r["found"]:
            print(f"  ✓ {r['query']!r:24s} {r['count']} 件 / 最高スコア {r['best_score']:.3f}")
        else:
            print(f"  ✗ {r['query']!r:24s} 見つかりませんでした")
    n_found = sum(1 for r in report if r["found"])
    print(f"\n{len(queries)} 件中 {n_found} 件で物体を発見。")
    if not files:
        print("ヒント: data/20_open_vocabulary_detection/ に実写を置くと、自由文で本物の物体を探せます。")

    payload = {
        "ok": ok,
        "error": err,
        "model_id": model_id,
        "model_kind": kind,
        "threshold": threshold,
        "source": source,
        "image_size": [image.width, image.height],
        "used_synthetic_scene": not files,
        "queries": report,
        "n_found": n_found,
    }
    (out / "use_case_finder.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {out / 'use_case_finder.png'}, {out / 'use_case_finder.json'}")


if __name__ == "__main__":
    main(sys.argv)
