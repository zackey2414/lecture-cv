"""章末ミニプロジェクト: キャプション・ベンチマーク。

この回の要素を1本に統合する総合課題:
  - 複数の小型モデル（BLIP / GIT / ViT-GPT2）で合成ギャラリーにキャプションを付け、
  - 生成パラメータ（greedy / beam）を切り替え、
  - 参照あり（BLEU・ROUGE-L・CIDEr）＋参照なし（CLIPScore）で採点し、
  - CIDEr 降順の「リーダーボード」を作って図・JSON・テキストに出力する。

「どのモデル × どの探索戦略が、この題材で一番マシか」を一望できる比較表を作るのが
ゴール。実データ（data/24_image_captioning/）があればそれを優先的に使う。

実行: uv run python lectures/24_image_captioning/mini_project.py
出力: lectures/24_image_captioning/outputs/mini_*.png / mini_leaderboard.json / mini_report.txt
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import caption_helpers as H  # noqa: E402
import caption_metrics as M  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 比較する (モデル, 生成設定) の組み合わせ。CPU 時間を抑えるため設定は2種類に絞る。
SETTINGS = {
    "greedy": dict(num_beams=1, do_sample=False, max_new_tokens=20),
    "beam4": dict(num_beams=4, do_sample=False, max_new_tokens=20),
}
MODEL_KEYS = ("blip", "git", "vitgpt2")


def run_benchmark() -> dict:
    """全 (モデル×設定) でキャプションを生成し、全指標で採点する。"""
    device = H.pick_device()
    images, names, references = H.load_user_or_synthetic()
    has_refs = all(len(r) > 0 for r in references)  # 実画像投入時は参照が無いことがある

    entries: list[dict] = []
    captions_table: dict[str, list[str]] = {}

    for key in MODEL_KEYS:
        cap = H.load_captioner(key, device)  # モデルは設定間で使い回す（ロードは1回）
        for setting_name, gen_kwargs in SETTINGS.items():
            captions = H.caption_images(cap, images, **gen_kwargs)
            tag = f"{key}/{setting_name}"
            captions_table[tag] = captions

            # 参照なし: CLIPScore（必ず計算できる）
            clip_scores = M.clip_score_pairs(images, captions, device)
            entry = {
                "model": key,
                "setting": setting_name,
                "tag": tag,
                "CLIPScore_mean": round(float(np.mean(clip_scores)), 3),
                "captions": captions,
            }
            # 参照あり: 参照キャプションがある時だけ
            if has_refs:
                entry["corpus_BLEU"] = round(M.corpus_bleu(captions, references), 4)
                entry["ROUGE_L"] = round(M.rouge_l(captions, references), 4)
                cider_mean, _ = M.compute_cider(captions, references)
                entry["CIDEr_mean"] = round(cider_mean, 4)
            entries.append(entry)

    # 並べ替え: 参照があれば CIDEr、無ければ CLIPScore を主キーにする
    sort_key = "CIDEr_mean" if has_refs else "CLIPScore_mean"
    entries.sort(key=lambda e: e.get(sort_key, 0.0), reverse=True)
    return {
        "device": str(device),
        "has_references": has_refs,
        "primary_metric": sort_key,
        "names": names,
        "references": references,
        "entries": entries,
        "captions_table": captions_table,
        "_images": images,  # 図の描画にだけ使う（JSON 保存前に取り除く）
    }


def plot_leaderboard(result: dict, path: pathlib.Path) -> None:
    """各 (モデル×設定) の主指標を横棒のリーダーボードで保存する。"""
    entries = result["entries"]
    metric = result["primary_metric"]
    tags = [e["tag"] for e in entries]
    vals = [e.get(metric, 0.0) for e in entries]
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(tags) + 1.5))
    colors = ["#2ca02c" if i == 0 else "#9ecae1" for i in range(len(tags))]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.invert_yaxis()  # 1位を上に
    ax.set_xlabel(metric)
    ax.set_title(f"Caption leaderboard (sorted by {metric})")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_best_captions(result: dict, path: pathlib.Path) -> None:
    """1位の (モデル×設定) のキャプションを画像グリッドで保存する。"""
    best = result["entries"][0]
    images = result["_images"]
    names = result["names"]
    caps = [f"{n}:\n{c}" for n, c in zip(names, best["captions"])]
    H.save_caption_grid(images, caps, path, title=f"Best: {best['tag']}")


def write_report(result: dict, path: pathlib.Path) -> None:
    """人が読むテキストの順位表を書き出す。"""
    lines = [
        "# Caption Benchmark Report",
        f"device = {result['device']}",
        f"primary metric = {result['primary_metric']}",
        f"references available = {result['has_references']}",
        "",
    ]
    if result["has_references"]:
        header = f"{'rank':>4s}  {'tag':16s} {'BLEU':>6s} {'ROUGE_L':>8s} {'CIDEr':>7s} {'CLIPScore':>10s}"
        lines.append(header)
        for i, e in enumerate(result["entries"], 1):
            lines.append(
                f"{i:>4d}  {e['tag']:16s} {e.get('corpus_BLEU', 0):6.3f} "
                f"{e.get('ROUGE_L', 0):8.3f} {e.get('CIDEr_mean', 0):7.3f} "
                f"{e['CLIPScore_mean']:10.2f}"
            )
    else:
        lines.append(f"{'rank':>4s}  {'tag':16s} {'CLIPScore':>10s}")
        for i, e in enumerate(result["entries"], 1):
            lines.append(f"{i:>4d}  {e['tag']:16s} {e['CLIPScore_mean']:10.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out = H.ensure_output_dir()
    result = run_benchmark()

    plot_leaderboard(result, out / "mini_leaderboard.png")
    plot_best_captions(result, out / "mini_best_captions.png")
    write_report(result, out / "mini_report.txt")

    # JSON は画像オブジェクトを除いてから保存する
    images = result.pop("_images")
    (out / "mini_leaderboard.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # コンソール要約
    print(f"device = {result['device']}  primary = {result['primary_metric']}")
    print(f"画像 {len(images)} 枚 × モデル {len(MODEL_KEYS)} × 設定 {len(SETTINGS)} を採点")
    print("\n=== リーダーボード ===")
    best = result["entries"][0]
    for i, e in enumerate(result["entries"], 1):
        mark = " <= BEST" if e is best else ""
        extra = (
            f"BLEU={e.get('corpus_BLEU', 0):.3f} CIDEr={e.get('CIDEr_mean', 0):.3f} "
            if result["has_references"]
            else ""
        )
        print(f"  {i}. {e['tag']:16s} {extra}CLIP={e['CLIPScore_mean']:.2f}{mark}")
    print(f"\nBEST = {best['tag']} のキャプション:")
    for n, c in zip(result["names"], best["captions"]):
        print(f"  [{n}] {c}")
    print(f"\nsaved: {out / 'mini_leaderboard.png'}, {out / 'mini_best_captions.png'},")
    print(f"       {out / 'mini_leaderboard.json'}, {out / 'mini_report.txt'}")


if __name__ == "__main__":
    main()
