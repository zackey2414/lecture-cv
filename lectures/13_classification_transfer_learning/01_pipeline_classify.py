"""第13回 / 01: HuggingFace `pipeline` で画像分類を最短体験する。

このスクリプトの狙い:
  - まず「動く」体験を最短で得る。前処理→推論→後処理を pipeline が1行で面倒みてくれる。
  - そのうえで「pipeline の中身（processor + model）」を 02 で分解する、という流れにつなげる。

ポイント（transformers 5.x の正準API）:
  - pipeline("image-classification", model=..., top_k=k) で上位 k 件のラベルとスコアが返る。
  - device は torch.device を渡せば cpu/mps/cuda を切り替えられる（既定は CPU）。
  - 合成図形は ImageNet の正解ラベルを持たないので、出力ラベルは「それっぽい何か」になる。
    意味のある予測を見たい人は data/13_classification_transfer_learning/ に写真を置けば、
    自動でそちらが使われる（load_demo_images の導線）。学ぶべきは"仕組み"であって、
    合成画像のラベルの正しさではない点に注意。

実行:
  uv run python lectures/13_classification_transfer_learning/01_pipeline_classify.py
"""

from __future__ import annotations

import json
import pathlib
import sys

# 同じフォルダの dl_helpers を import できるようにする。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import transformers
from transformers import pipeline

from dl_helpers import ensure_out_dir, get_device, load_demo_images, save_prediction_panel

# モデルのロード時に出る冗長な情報ログを抑える（教材の出力を読みやすく保つ）。
transformers.logging.set_verbosity_error()

# CPU でも軽い小型モデルを明示指定する。pipeline の既定モデル(google/vit-base...)は
# CPU には少し重いので、ここでは ResNet-18(約45MB)を使う。
MODEL_ID = "microsoft/resnet-18"
TOP_K = 5


def main() -> None:
    out_dir = ensure_out_dir()
    device = get_device()
    print(f"[info] device = {device} / model = {MODEL_ID}")

    # --- pipeline を1つ作るだけで前処理〜後処理まで完結する ---
    # device に torch.device を渡せる（CPU なら "cpu"）。top_k で上位何件返すか決める。
    clf = pipeline(
        task="image-classification",
        model=MODEL_ID,
        device=device,
        top_k=TOP_K,
    )

    # デモ画像を用意（data/ に写真があればそれを、無ければ合成図形を使う）。
    images, names = load_demo_images(n=4)

    # --- 1枚ずつ分類し、上位 TOP_K を表示 ---
    panel_titles: list[str] = []
    records: list[dict] = []
    for img, name in zip(images, names):
        results = clf(img)  # [{'label': str, 'score': float}, ...] が score 降順で返る
        top1 = results[0]
        print(f"\n[{name}]")
        for rank, r in enumerate(results, start=1):
            print(f"  top{rank}: {r['label']:<35s} score={r['score']:.3f}")

        # パネル用タイトルは ASCII のみ（CJK は図で豆腐になるため、英語ラベルをそのまま使う）。
        # 英語ラベルが長いので先頭の語だけ表示。
        short = top1["label"].split(",")[0]
        panel_titles.append(f"{short}\n({top1['score']:.2f})")
        records.append(
            {
                "input": name,
                "top1_label": top1["label"],
                "top1_score": round(float(top1["score"]), 4),
                "topk": [
                    {"label": r["label"], "score": round(float(r["score"]), 4)} for r in results
                ],
            }
        )

    # --- 結果を画像パネルと JSON に保存 ---
    panel_path = out_dir / "01_pipeline_predictions.png"
    save_prediction_panel(
        images,
        panel_titles,
        panel_path,
        suptitle=f"pipeline(image-classification) top-1  [{MODEL_ID}]",
    )
    json_path = out_dir / "01_pipeline_predictions.json"
    json_path.write_text(
        json.dumps({"model": MODEL_ID, "device": str(device), "results": records}, indent=2),
        encoding="utf-8",
    )

    print(f"\n[done] saved panel -> {panel_path}")
    print(f"[done] saved json  -> {json_path}")
    print(
        "\nヒント: 合成図形だと ImageNet ラベルは"
        "それっぽい別物になります。意味のある予測を見たい場合は\n"
        f"        data/{pathlib.Path('13_classification_transfer_learning')}/ に写真を置いて再実行してください。"
    )


if __name__ == "__main__":
    main()
