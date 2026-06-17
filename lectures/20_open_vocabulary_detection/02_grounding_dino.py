"""02: Grounding DINO で『キャプション』による検出と、box/text 閾値の効き方を見る。

OWL-ViT/OWLv2 が『候補ラベルのリスト』を受け取るのに対し、Grounding DINO は
1本の『キャプション』を受け取る。書式が厳密で、

    小文字 + 各物体をピリオドで区切る   例: "a red circle. a blue square."

を守らないと検出が安定しない（これが最頻の落とし穴）。さらに閾値が2種類ある:
  - box_threshold : 検出ボックスの確信度（小さいほど過検出）
  - text_threshold: ボックスをどの語に結びつけるかの確信度
デフォルトのままだと過検出/未検出になりやすいので、両方を調整して挙動を体感する。

なお Grounding DINO は OWL より重い（timm 依存・初回 DL も大きめ）。CPU でも tiny なら
数秒で動くが、環境次第で失敗しても講座が止まらないよう、ロード/推論は try/except で
包み、ダメなら『概念紹介のみ』にフォールバックして必ず exit 0 にする。

実行: uv run python lectures/20_open_vocabulary_detection/02_grounding_dino.py
出力: lectures/20_open_vocabulary_detection/outputs/02_*.png / 02_gdino_results.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


import ovd_helpers as H  # noqa: E402

# 概念紹介（フォールバック時にも必ず出す要点）。
_CONCEPT = [
    "Grounding DINO はテキストと画像を early-fusion でクロスアテンションし、",
    "キャプション中の任意フレーズに対応する物体を検出する（参照表現にも強い）。",
    "入力キャプションは『小文字＋ピリオド区切り』が必須: 'a cat. a remote control.'",
    "後処理は post_process_grounded_object_detection(input_ids=, threshold=box, text_threshold=)。",
    "box_threshold↑で過検出を抑え、text_threshold↑で語との対応を厳しくする。",
]


def run_grounding_dino(image, caption: str, gt_boxes, gt_labels, out) -> dict:
    """Grounding DINO をロードし、2 種類の閾値設定で検出して図と要約を作る。"""
    processor, model = H.load_zero_shot_detector(H.GDINO_ID)

    # 閾値を変えて『緩い設定（過検出寄り）』と『厳しい設定』を比較する。
    settings = [
        {"name": "loose", "box": 0.15, "text": 0.15},
        {"name": "strict", "box": 0.35, "text": 0.25},
    ]
    runs = []
    for st in settings:
        boxes, scores, text_labels = H.detect_gdino(
            processor,
            model,
            image,
            caption,
            box_threshold=st["box"],
            text_threshold=st["text"],
        )
        H.draw_detections(
            image,
            boxes,
            text_labels,
            scores,
            f"grounding-dino ({st['name']}: box>={st['box']}, text>={st['text']})",
            out / f"02_gdino_{st['name']}.png",
            gt_boxes=gt_boxes,
        )
        runs.append(
            {
                "setting": st["name"],
                "box_threshold": st["box"],
                "text_threshold": st["text"],
                "n_detections": int(len(boxes)),
                "detections": [
                    {"label": lab, "score": round(float(sc), 3)}
                    for lab, sc in sorted(zip(text_labels, scores), key=lambda x: -x[1])
                ],
            }
        )
        print(
            f"[grounding-dino / {st['name']}] box>={st['box']} text>={st['text']}  "
            f"-> {len(boxes)} 検出"
        )
        for d in runs[-1]["detections"]:
            print(f"    {d['score']:.3f}  {d['label']}")

    return {"ok": True, "model_id": H.GDINO_ID, "caption": caption, "runs": runs}


def main() -> None:
    out = H.ensure_output_dir()
    image, gt_boxes, gt_labels = H.load_user_or_synthetic()

    # 候補ラベル -> Grounding DINO 用キャプションへ整形（小文字＋ピリオド区切り）。
    labels = H.scene_candidate_labels()
    caption = H.labels_to_caption(labels)
    print(f"device = {H.pick_device()}")
    print(f"caption（小文字＋ピリオド区切り）= {caption!r}\n")

    # 重い/失敗しうるので包む。失敗しても概念紹介を出して exit 0。
    try:
        payload = run_grounding_dino(image, caption, gt_boxes, gt_labels, out)
    except Exception as e:  # noqa: BLE001
        print(f"[フォールバック] Grounding DINO を実行できませんでした: {type(e).__name__}: {e}")
        print("概念だけ確認して終了します（exit 0）。重い場合は GPU 推奨。")
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}", "caption": caption}

    print("\n--- Grounding DINO の要点 ---")
    for line in _CONCEPT:
        print("  " + line)
    print("\nOWL との違い: OWL は候補ラベルの『リスト』、Grounding DINO は1本の『キャプション』。")
    print("緩い設定では過検出（同じ物体に複数 / 関係ない語が当たる）が起きやすい。")

    payload["concept"] = _CONCEPT
    (out / "02_gdino_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nsaved: {out / '02_gdino_results.json'}"
        + (
            f", {out / '02_gdino_loose.png'}, {out / '02_gdino_strict.png'}"
            if payload.get("ok")
            else ""
        )
    )


if __name__ == "__main__":
    main()
