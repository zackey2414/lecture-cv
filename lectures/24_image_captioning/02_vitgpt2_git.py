"""02: ViT-GPT2 / GIT と「生成パラメータ」で出力を制御する。

この回のねらいは2つ:
  A. BLIP 以外の代表モデルを動かし、アーキテクチャの違いを体感する
       - ViT-GPT2 : VisionEncoderDecoderModel（ViT エンコーダ + GPT-2 デコーダ）
                    画像プロセッサとトークナイザを別々に使う点が BLIP/GIT と違う
       - GIT      : GitForCausalLM（画像トークンを言語モデルの接頭辞にする方式）
  B. model.generate の探索戦略を切り替えて、出力がどう変わるかを観察する
       - greedy（num_beams=1, do_sample=False）  : 毎回同じ・無難・反復しがち
       - beam search（num_beams=4）              : 複数候補を保持し尤度最大を選ぶ
       - sampling（do_sample=True, temperature） : 多様だが不安定（seed 固定で再現）
       - repetition_penalty                      : 同じ語の繰り返しを抑える
       - max_new_tokens                          : 長さの上限

実行: uv run python lectures/24_image_captioning/02_vitgpt2_git.py
出力: lectures/24_image_captioning/outputs/02_*.png / 02_generation_params.json
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import caption_helpers as H  # noqa: E402


def compare_models() -> dict:
    """3モデル（blip/git/vitgpt2）で同じギャラリーにキャプションを付けて比較する。"""
    device = H.pick_device()
    images, names, _ = H.build_gallery()

    table: dict[str, list[str]] = {}
    for key in ("blip", "git", "vitgpt2"):
        cap = H.load_captioner(key, device)
        # バッチ推論 + batch_decode（複数画像を一度に処理する正準パターン）
        captions = H.caption_images(cap, images, num_beams=4, max_new_tokens=20)
        table[key] = captions

    rows = []
    for i, name in enumerate(names):
        rows.append({"image": name, **{k: table[k][i] for k in table}})
    return {"device": str(device), "models": {k: H.MODELS[k]["id"] for k in table}, "rows": rows}


def generation_param_study() -> dict:
    """1枚の画像に対し、生成パラメータを変えて出力の変化を観察する。"""
    device = H.pick_device()
    images, names, _ = H.build_gallery()
    # 反復が出やすいよう、情報の少ない「夕焼け」を題材にする
    idx = names.index("sunset")
    image = images[idx]

    cap = H.load_captioner("vitgpt2", device)  # GPT-2 デコーダは反復が出やすく教材向き

    settings = {
        "greedy": dict(num_beams=1, do_sample=False, max_new_tokens=30),
        "beam4": dict(num_beams=4, do_sample=False, max_new_tokens=30),
        "beam4_reppen": dict(
            num_beams=4, do_sample=False, max_new_tokens=30, repetition_penalty=1.5
        ),
        "max_new_tokens=8": dict(num_beams=1, do_sample=False, max_new_tokens=8),
    }
    out: dict[str, str] = {}
    for label, kwargs in settings.items():
        out[label] = H.caption_images(cap, [image], **kwargs)[0]

    # サンプリングは乱数なので seed を固定して再現性を確保する。
    sampled: list[str] = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        s = H.caption_images(
            cap, [image], do_sample=True, temperature=1.3, top_p=0.9, num_beams=1, max_new_tokens=30
        )[0]
        sampled.append(s)

    return {
        "image": names[idx],
        "model": H.MODELS["vitgpt2"]["id"],
        "deterministic": out,
        "sampling_temperature1.3": sampled,
    }


def main() -> None:
    out = H.ensure_output_dir()

    comp = compare_models()
    study = generation_param_study()

    # 図1: 3モデル比較グリッド（BLIP のキャプションを表示）
    images, names, _ = H.build_gallery()
    blip_caps = [r["blip"] for r in comp["rows"]]
    H.save_caption_grid(
        images,
        [f"{n}:\n{c}" for n, c in zip(names, blip_caps)],
        out / "02_models_blip.png",
        title="BLIP captions (compare baseline)",
    )

    (out / "02_generation_params.json").write_text(
        json.dumps({"model_compare": comp, "param_study": study}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # コンソール: モデル比較
    print(f"device = {comp['device']}")
    print("\n=== 3モデル比較（num_beams=4, max_new_tokens=20） ===")
    print(f"{'image':8s} | {'blip':40s} | {'git':30s} | vitgpt2")
    for r in comp["rows"]:
        print(f"{r['image']:8s} | {r['blip'][:40]:40s} | {r['git'][:30]:30s} | {r['vitgpt2']}")

    # コンソール: 生成パラメータ
    print(f"\n=== 生成パラメータの効果（題材: {study['image']} / {study['model']}） ===")
    for label, text in study["deterministic"].items():
        print(f"  [{label:18s}] {text}")
    print("  [sampling t=1.3   ] (seed 0/1/2 で毎回変わる＝多様性)")
    for i, s in enumerate(study["sampling_temperature1.3"]):
        print(f"      seed{i}: {s}")
    print("\n観察ポイント: greedy は無難で反復しがち / beam は尤度最大の安定文 /")
    print("repetition_penalty で繰り返しが減る / max_new_tokens で長さが変わる /")
    print("sampling は多様だが品質が揺れる（seed 固定で再現性を担保）。")
    print(f"\nsaved: {out / '02_models_blip.png'}, {out / '02_generation_params.json'}")


if __name__ == "__main__":
    main()
