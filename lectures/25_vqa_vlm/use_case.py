"""use_case.py — 画像Q&Aアシスタント（画像と複数の質問を入れると答えを返す対話ツール）

この回の「実践ユースケース」: 1 枚の画像に対して **複数の質問** をまとめて投げ、
それぞれに短い回答を返す小さな Q&A アシスタントです。VQA モデル（BLIP-VQA）を
「回答エンジン」にして、画像 + 質問リスト → 回答リスト という現実のツールとして完結させます。
商品写真への問い合わせ（「色は？」「個数は？」）、書類・標識の確認、アクセシビリティ用の
画像説明、チャットボットの画像理解バックエンド —— どれも「画像 × 自由な質問」で作れます。

────────────────────────────────────────────────────────────────────────
mini_project.py との違い（役割分担）
────────────────────────────────────────────────────────────────────────
- `mini_project.py` は「VQAv2 accuracy / IoU でモデルを **採点** する」評価寄りの統合課題で、
  正解（ground truth）を持つベンチに対してスコアを出すのが主眼です（レポートカード）。
- 本 `use_case.py` は採点ではなく **アプリの出力（質問への答え）** を作るのが目的の
  **現実の小ツール** です。正解は不要で、任意の画像に任意の質問を投げて答えを得ます。
  回答エンジンは BLIP-VQA を借用し、その上に「複数質問をループして 1 枚絵 + JSON にまとめる」
  というアプリ固有ロジックだけを薄く足しています。そのまま問い合わせ窓口に拡張できます。

────────────────────────────────────────────────────────────────────────
データ配置（実データ優先・合成フォールバック）
────────────────────────────────────────────────────────────────────────
- `data/25_vqa_vlm/` に **画像**（*.png / *.jpg / *.jpeg / *.bmp / *.webp）を置くと、
  それを実入力として使います（例: `data/25_vqa_vlm/product.jpg`）。
- さらに `data/25_vqa_vlm/questions.txt` を置くと、1 行 1 問でカスタム質問を使えます
  （`#` 始まりはコメント）。無ければ用途に応じた既定の質問セットを使います。
- 画像が無ければ **合成シーン**（色 × 形を既知の位置に配置）で必ず完走します（exit 0）。
- 回答エンジンの BLIP-VQA が（ネット未接続・モデル未DL 等で）使えない場合は、合成シーンの
  既知メタデータを使った **オフラインのルールベース回答** に自動で切り替えて完走します。

────────────────────────────────────────────────────────────────────────
拡張アイデア（練習用）
────────────────────────────────────────────────────────────────────────
- 回答エンジンを **チャット VLM**（SmolVLM2-256M。`uv add num2words` 後に
  `vqa_helpers.load_smolvlm` + `apply_chat_template`）へ差し替え、推論や説明文も返す。
- 回答に **信頼度** を付ける（ViLT に切り替えれば softmax 確率が取れる）。
- 「一語で答えて」等の **プロンプト整形** で出力を構造化し、yes/no や数値を抽出しやすくする。
- 質問を固定テンプレ化して **バッチ問い合わせ**（CSV 入出力）にし、商品台帳の自動タグ付けにする。
- グラウンディング（「赤い円はどこ？」→ 箱/点）を足したいなら moondream2 の detect/point か
  オープン語彙検出（第20回）と組み合わせる。

実行:
    uv run python lectures/25_vqa_vlm/use_case.py

出力（outputs/25_vqa_vlm/ に保存）:
    use_case_qa_XX.png : 画像の横に Q&A を並べたパネル（画像ごとに 1 枚）
    use_case_qa.json   : 全画像・全質問・回答（機械可読）
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# headless 前提: 図は必ず Agg バックエンド（GUI を開かない）。
import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

# 同ディレクトリの再利用部品（vqa_helpers）を import する。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vqa_helpers as H  # noqa: E402


# ----------------------------------------------------------------------
# 既定の質問セット（data/ に questions.txt が無いとき使う）
# ----------------------------------------------------------------------
# 実画像向け: どんな写真にも当てられる汎用的な問い。
DEFAULT_QUESTIONS_REAL = [
    "What is in the image?",
    "What is the main object?",
    "What color is the main object?",
    "How many objects are there?",
]
# 合成シーン向け: 色・数・有無を問う、評価しやすい問い。
DEFAULT_QUESTIONS_SYNTH = [
    "What color is the circle?",
    "How many shapes are in the image?",
    "Is there a square in the image?",
    "What is in the image?",
]


# ----------------------------------------------------------------------
# 入力の用意（実画像優先 → 合成フォールバック）
# ----------------------------------------------------------------------
def _synthetic_scenes() -> list[dict]:
    """既知メタデータ付きの合成シーンを返す。

    各要素: {name, image(PIL), objects=[(color, shape, box), ...]}。
    objects を持っているので、モデルが無くてもオフラインで答えを作れる。
    """
    objs_pair = [
        ("red", "circle", (40, 150, 170, 280)),
        ("blue", "square", (220, 150, 340, 270)),
    ]
    objs_shapes = [
        ("green", "triangle", (30, 60, 150, 180)),
        ("yellow", "circle", (160, 200, 270, 310)),
        ("red", "square", (280, 60, 360, 140)),
    ]
    return [
        {"name": "pair", "image": H.make_scene(objs_pair, 384), "objects": objs_pair},
        {"name": "shapes", "image": H.make_scene(objs_shapes, 384), "objects": objs_shapes},
    ]


def load_questions(default: list[str]) -> tuple[list[str], bool]:
    """data/25_vqa_vlm/questions.txt があればそれを、無ければ default を返す。"""
    qfile = H.DATA_DIR / "questions.txt"
    if qfile.is_file():
        qs = [
            ln.strip()
            for ln in qfile.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if qs:
            return qs, True
    return default, False


def load_items() -> tuple[list[dict], list[str], bool, str]:
    """(items, questions, is_synthetic, source) を返す。

    items は {name, image, objects} のリスト（実画像は objects=None）。
    """
    users = H.load_user_images()
    if users:
        items = [{"name": name, "image": img, "objects": None} for name, img in users]
        questions, _ = load_questions(DEFAULT_QUESTIONS_REAL)
        return items, questions, False, f"data/25_vqa_vlm/ の実画像 {len(items)} 枚"
    items = _synthetic_scenes()
    questions, _ = load_questions(DEFAULT_QUESTIONS_SYNTH)
    return items, questions, True, "合成シーン（data/25_vqa_vlm/ に画像を置けば実画像で動く）"


# ----------------------------------------------------------------------
# 回答エンジン: BLIP-VQA（生成型）→ 失敗時はオフラインのルールベース
# ----------------------------------------------------------------------
def _offline_answer(question: str, objects: list | None) -> str:
    """モデルが無いときの簡易回答（合成シーンの既知メタデータを使う）。

    実画像（objects=None）は答えようがないので「モデルが必要」と返す。
    合成シーンでは色・数・有無の典型的な問いだけルールで答える。
    """
    if objects is None:
        return "(needs a VQA model)"
    q = question.lower()
    colors = [c for c, _s, _b in objects]
    shapes = [s for _c, s, _b in objects]
    # 「いくつ？」→ オブジェクト数
    if "how many" in q:
        return str(len(objects))
    # 「<形>の色は？」→ その形を持つオブジェクトの色
    if "color" in q or "colour" in q:
        for sh in H.SHAPES:
            if sh in q:
                for c, s, _b in objects:
                    if s == sh:
                        return c
                return "none"
        return colors[0] if colors else "none"
    # 「<色/形>はある？」→ yes / no
    if "is there" in q or "are there" in q or "does" in q:
        for c in H.COLORS:
            if c in q:
                return "yes" if c in colors else "no"
        for sh in H.SHAPES:
            if sh in q:
                return "yes" if sh in shapes else "no"
        return "yes" if objects else "no"
    # 「何がある？」→ 「<色> <形>」を列挙
    if "what" in q:
        return ", ".join(f"{c} {s}" for c, s, _b in objects)
    return "n/a"


def build_answerer(device: torch.device):
    """(answer_fn, engine_name) を返す。answer_fn(image, question, objects) -> str。

    まず BLIP-VQA（短い自由回答を生成）を試し、ロードに失敗したらオフラインへ。
    どちらでも同じ呼び出し方（image, question, objects）にして main を単純に保つ。
    """
    try:
        processor, model = H.load_blip_vqa(device)

        def answer(image, question, _objects):
            enc = processor(image, question, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=12)
            return processor.decode(out[0], skip_special_tokens=True).strip()

        return answer, "BLIP-VQA"
    except Exception as e:  # noqa: BLE001  未導入 / DL 失敗でも exit 0 で続行
        print(f"  VQA モデルをスキップ: {H.hint_for_model_error(e)}")
        print("  → オフラインのルールベース回答で続行します（合成シーンのみ実質回答）。")

        def answer(_image, question, objects):
            return _offline_answer(question, objects)

        return answer, "offline-rule"


# ----------------------------------------------------------------------
# メイン: 画像ごとに複数質問を回し、パネル + JSON にまとめる
# ----------------------------------------------------------------------
def main() -> None:
    out = H.ensure_output_dir()
    device = H.pick_device()
    items, questions, is_synth, source = load_items()
    answer, engine = build_answerer(device)

    print("=" * 64)
    print("画像Q&Aアシスタント（画像 × 複数質問 → 回答）")
    print(f"  入力 : {source}")
    print(f"  質問 : {len(questions)} 問 / 画像 {len(items)} 枚")
    print(f"  エンジン: {engine}  (device={device})")

    images_out = []
    for i, item in enumerate(items):
        name, image, objects = item["name"], item["image"], item["objects"]
        print(f"\n[{i:02d}] {name}")
        qa = []
        for q in questions:
            a = answer(image, q, objects)
            qa.append({"question": q, "answer": a})
            print(f"   Q: {q}")
            print(f"   A: {a}")

        # 画像の横に Q&A を並べたパネルを保存（目視確認用）。タイトルは ASCII。
        lines: list[str] = []
        for pair in qa:
            lines.append(f"Q: {pair['question']}")
            lines.append(f"A: {pair['answer']}")
            lines.append("")
        panel_path = out / f"use_case_qa_{i:02d}.png"
        H.save_qa_panel(image, lines, panel_path, title=f"QA assistant (engine={engine})")

        images_out.append(
            {"name": name, "panel": str(panel_path), "qa": qa}
        )

    # 機械可読な結果 JSON。
    result = {
        "source": source,
        "engine": engine,
        "num_images": len(items),
        "num_questions": len(questions),
        "questions": questions,
        "images": images_out,
    }
    json_path = out / "use_case_qa.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "-" * 64)
    print(f"  パネル : {out}/use_case_qa_XX.png（画像ごと {len(items)} 枚）")
    print(f"  回答JSON: {json_path}")
    if is_synth:
        print("  ※ 合成入力で完走。data/25_vqa_vlm/ に画像を置けば実画像のQ&Aになります。")
    if engine == "offline-rule":
        print("  ※ オフライン回答で完走。ネット接続後は BLIP-VQA が自由回答を返します。")
    print("\n発展: 回答エンジンを SmolVLM2 等のチャットVLMに替えると、説明や推論も返せます。")


if __name__ == "__main__":
    main()
