"""use_case.py — 実践ユースケース:「動画ライブラリの行動タグ付けツール」。

================================================================================
これは何か（現実の小ツール）
--------------------------------------------------------------------------------
フォルダに溜まった動画たちを 1 本ずつ開き、r3d_18(Kinetics-400) で
「上位 N 個の行動ラベル」を自動で振り分け、検索できる **タグ・カタログ**
(JSON / CSV) として書き出す小さな道具です。動画アセット管理・ざっくり分類・
"このフォルダにはどんな行動の動画が多いか" の俯瞰、といった現実の用途に直結します。

最終成果物（lectures/29_video_action_recognition/outputs/ に保存）:
  - use_case_tags.json   … 各動画 → 上位 N タグ(確率つき)のカタログ
  - use_case_tags.csv    … 表計算/検索しやすいフラットなタグ表（1 行 = 1 動画）
  - use_case_gallery.png … 各動画の代表フレーム + 主タグを並べたサムネ一覧
  - use_case_tagfreq.png … ライブラリ全体での主タグ出現頻度（何が多いか）
最後にカタログを 1 つのタグで横断検索するデモも実行します
（＝タグ付けの「うれしさ」＝あとから探せること、を体感する）。

================================================================================
実データの置き方（実データ優先・合成フォールバック）
--------------------------------------------------------------------------------
data/29_video_action_recognition/ に手持ちの動画
(.mp4 / .avi / .mov / .mkv / .webm) を置くと、それを優先してタグ付けします。

    lecture-cv/
      └─ data/
         └─ 29_video_action_recognition/
            ├─ jog.mp4
            ├─ wave.mov
            └─ ...

データが 1 本も無ければ、合成「行動クリップ」(動く図形) を数本 mp4 に書き出し、
それを実際に cv2.VideoCapture でデコードしてタグ付けします
（＝ネットもデータも無しで必ず exit 0。実運用と同じ "フォルダを舐める" 経路を通る）。
※合成クリップに本物の Kinetics ラベルは無いので、付くタグは「例示」です。
  実写を data/ に置けば、タグの中身も意味を持ち始めます。

================================================================================
mini_project.py との違い（役割分担）
--------------------------------------------------------------------------------
  - mini_project.py … 教材の総まとめ。合成データセットに対して
      clip_len / frame_rate を掃引し「前処理を崩すと予測がどれだけ壊れるか」を
      pseudo-GT 基準の top-1/top-5 で “定量化” する学習用パイプライン。
  - use_case.py（これ）… 学習の話は脇に置き、**現実のフォルダ**を入力に
      「動画 → 上位 N タグ → 検索できるカタログ(JSON/CSV)」を作る “実アプリの出発点”。
      評価指標ではなく「あとから探せる成果物」を作るのが目的。

================================================================================
拡張アイデア（ここから練習を広げる）
--------------------------------------------------------------------------------
  1) しきい値: prob が TAG_THRESHOLD 未満のタグは「自信なし」として捨てる。
  2) multi-clip TTA: 1 本の動画から時間窓を複数取り、logits を平均して頑健化。
  3) モデル差し替え: r3d_18 → H.load_videomae() で VideoMAE に。精度は上がるが重い。
  4) 検索 UI: use_case_tags.csv を pandas/SQLite に読み込み、タグで全文検索や集計。
  5) サムネ生成: 主タグごとに代表フレームを切り出してプレビュー画像を量産。
  6) しきい値 + 人手確認: 低信頼のものだけ人にレビューさせる "human-in-the-loop"。

実行:
    uv run python lectures/29_video_action_recognition/use_case.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless でも図を保存できるよう、画面を持たないバックエンドに固定
import matplotlib.pyplot as plt
import numpy as np
import torch

import action_helpers as H

# --- 設定 ---------------------------------------------------------------------
# 実データ置き場（無ければ合成にフォールバック）。OUT_DIR と同じ流儀でルートから辿る。
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "29_video_action_recognition"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")  # 認識する動画拡張子
NUM_FRAMES = 16  # r3d_18 に渡すサンプリング枚数（標準クリップ長）
TOP_N = 3  # 各動画に付ける上位タグ数（"行動ラベル(上位N)" の N）
MAX_SCAN = 600  # 1 本あたり走査する最大フレーム数（長尺動画のメモリ暴発を防ぐ安全弁）
MAX_GALLERY = 8  # サムネ一覧に載せる最大本数（見やすさのため）


# ---------------------------------------------------------------------------
# 1) 入力動画の発見（実データ優先）/ 無ければ合成ライブラリを用意
# ---------------------------------------------------------------------------
def discover_videos(data_dir: Path) -> list[Path]:
    """data_dir 直下から動画ファイルを拡張子で拾って返す（名前順）。"""
    if not data_dir.exists():
        return []
    found = [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    return found


def make_synthetic_library(out_dir: Path, n_videos: int = 6) -> list[Path]:
    """合成「行動クリップ」を mp4 に書き出して “疑似ライブラリ” を作る。

    実データが無いときのフォールバック。ポイントは、ここで mp4 を実ファイルとして
    書き出し、後段で本物の動画と同じく cv2.VideoCapture でデコードして読むこと。
    ＝ツールの本流（フォルダを舐めてデコード）を合成でもそのまま通す。
    """
    lib_dir = out_dir / "use_case_synthetic"
    lib_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # ACTION_KINDS（円が動く/四角が落ちる/回転…）を一通り使い、運動の異なる動画を並べる
    for k in range(min(n_videos, len(H.ACTION_KINDS))):
        # 本数ぶんフレーム数を少し変え、長さの違う動画が混ざる現実に寄せる
        num_frames = 24 + 4 * k
        clip = H.make_action_clip(kind=k, num_frames=num_frames, seed=k)
        path = lib_dir / f"sample_{k:02d}_{H.ACTION_KINDS[k]}.mp4"
        H.write_clip_mp4(clip, path, fps=8.0)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 2) 動画 → サンプリング済みフレーム列（メモリ安全な実動画リーダ）
# ---------------------------------------------------------------------------
def sample_video_frames(path: Path, num_frames: int = NUM_FRAMES, max_scan: int = MAX_SCAN):
    """動画から num_frames 枚を等間隔サンプリングして (T,H,W,3) RGB で返す。

    長尺動画でも全フレームを抱え込まないよう、総フレーム数が分かる場合は
    cap.grab() で安く読み飛ばし、必要なフレームだけ cap.retrieve() でデコードする
    （H.read_clip_mp4 は全フレームを読むので、不定長/長尺の実動画にはこちらが安全）。
    総数が不明なストリーム的入力では、max_scan までを順次読んでから等間隔に間引く。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq: list[np.ndarray] = []

    if total and total > 0:
        # --- 総数が分かる場合: 目標インデックスだけを retrieve（高速・省メモリ） ---
        total = min(total, max_scan)
        targets = sorted(set(int(i) for i in H.uniform_indices(total, num_frames)))
        target_set = set(targets)
        max_t = targets[-1]
        i = 0
        while i <= max_t:
            if not cap.grab():  # デコードせずに 1 フレーム進める（安い）
                break
            if i in target_set:
                ok, frame = cap.retrieve()  # ここで初めてデコード
                if not ok:
                    break
                seq.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
    else:
        # --- 総数が不明（ストリーム等）: max_scan まで読んで後で間引く ---
        while len(seq) < max_scan:
            ok, frame = cap.read()
            if not ok:
                break
            seq.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not seq:
        raise RuntimeError(f"フレームを 1 枚も読めませんでした: {path}")

    arr = np.stack(seq, axis=0)
    # 端数や読み落としで枚数が num_frames と違っても、もう一度等間隔で揃える（必ず固定長に）
    if arr.shape[0] != num_frames:
        arr = arr[H.uniform_indices(arr.shape[0], num_frames)]
    return arr


# ---------------------------------------------------------------------------
# 3) フレーム列 → 上位 N タグ（r3d_18 推論）
# ---------------------------------------------------------------------------
def tag_video(model, categories, frames: np.ndarray, top_n: int = TOP_N):
    """(T,H,W,3) RGB を r3d_18 に通して上位 top_n の (タグ, 確率) を返す。"""
    # 正しい専用正規化で (1,C,T,H,W) に整形（前処理を誤ると無意味な出力になる章の肝）
    x = H.preprocess_for_r3d(frames, indices=None, normalize=True)
    with torch.inference_mode():
        logits = model(x)
    return H.topk_from_logits(logits, categories, k=top_n)


# ---------------------------------------------------------------------------
# 4) カタログ構築 / 保存 / 検索
# ---------------------------------------------------------------------------
def build_catalog(model, categories, videos: list[Path]) -> list[dict]:
    """各動画をタグ付けして、検索可能なカタログ(辞書のリスト)を作る。

    1 本でも失敗したら飛ばして次へ進む（実運用ツールの最低限の堅牢性）。
    各エントリは代表フレーム(rep_frame)も持ち、後段のサムネ一覧で使う。
    """
    catalog: list[dict] = []
    for path in videos:
        try:
            frames = sample_video_frames(path)
            tags = tag_video(model, categories, frames, TOP_N)
        except Exception as e:  # 壊れた動画やデコード失敗はスキップして継続
            print(f"  [skip] {path.name}: {type(e).__name__}: {e}")
            continue
        rep = frames[frames.shape[0] // 2]  # 中央フレームを代表サムネに
        entry = {
            "video": path.name,
            "path": str(path),
            "frames_used": int(frames.shape[0]),
            "primary_tag": tags[0][0],
            "primary_prob": round(tags[0][1], 4),
            "top_tags": [{"tag": t, "prob": round(p, 4)} for t, p in tags],
            "_rep_frame": rep,  # 図示用（json/csv には出さない）
        }
        catalog.append(entry)
        tag_str = ", ".join(f"{t}({p:.2f})" for t, p in tags)
        print(f"  {path.name:<34s} -> {tag_str}")
    return catalog


def save_catalog_json(catalog: list[dict], path: Path) -> None:
    """カタログを JSON で保存（_rep_frame は numpy なので除いて書く）。"""
    serializable = [{k: v for k, v in e.items() if k != "_rep_frame"} for e in catalog]
    H.save_json({"num_videos": len(serializable), "top_n": TOP_N, "items": serializable}, path)


def save_catalog_csv(catalog: list[dict], path: Path) -> None:
    """カタログを CSV(ワイド形式: 1 行 = 1 動画) で保存。表計算/検索に向く。"""
    # 列: video, frames_used, tag_1, prob_1, ..., tag_N, prob_N
    header = ["video", "frames_used"]
    for r in range(1, TOP_N + 1):
        header += [f"tag_{r}", f"prob_{r}"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for e in catalog:
            row = [e["video"], e["frames_used"]]
            for t in e["top_tags"]:
                row += [t["tag"], t["prob"]]
            # top_tags が TOP_N 未満でも列数を合わせる（クラス数 < N の保険）
            while len(row) < len(header):
                row += ["", ""]
            writer.writerow(row)


def search_by_tag(catalog: list[dict], query: str) -> list[dict]:
    """タグ名の部分一致でカタログを横断検索（タグ付けの実利＝あとから探せる、のデモ）。

    primary だけでなく上位 N タグのどれかに query を含めばヒット扱いにする。
    """
    q = query.lower()
    hits = []
    for e in catalog:
        if any(q in t["tag"].lower() for t in e["top_tags"]):
            hits.append(e)
    return hits


# ---------------------------------------------------------------------------
# 5) 可視化（サムネ一覧 + タグ頻度）
# ---------------------------------------------------------------------------
def save_gallery(catalog: list[dict], path: Path) -> None:
    """各動画の代表フレームを並べ、主タグ(ASCII)を見出しにしたサムネ一覧を保存。"""
    items = catalog[:MAX_GALLERY]
    if not items:
        return
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, e in zip(axes, items):
        ax.imshow(e["_rep_frame"])
        # Kinetics ラベルは英語(ASCII)なので豆腐にならない。確率も併記。
        ax.set_title(f"{e['primary_tag']}\n({e['primary_prob']:.2f})", fontsize=8)
        ax.set_xlabel(e["video"][:22], fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:  # 余ったマスは消す
        ax.axis("off")
    fig.suptitle("auto action tags (r3d_18) - representative frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_tag_frequency(catalog: list[dict], path: Path) -> None:
    """ライブラリ全体の主タグ出現頻度を棒グラフで保存（何の行動が多いかの俯瞰）。"""
    counts = Counter(e["primary_tag"] for e in catalog)
    if not counts:
        return
    names, values = zip(*counts.most_common())
    # save_bar は値を 0..1 想定ではないので ylim を頻度に合わせて使う
    H.save_bar(
        [n[:18] for n in names],
        list(values),
        path,
        title="primary action tag frequency in library",
        ylabel="num videos",
        ylim=(0, max(values) + 1),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    H.limit_cpu_threads(4)
    print("=" * 72)
    print("実践ユースケース: 動画ライブラリの行動タグ付けツール (r3d_18)")
    print("=" * 72)

    # 1) モデルロード（初回のみ ~127MB DL）。失敗しても案内して exit 0。
    try:
        model, categories = H.load_r3d()
    except Exception as e:
        print(f"[案内] r3d_18 をロードできませんでした（オフライン？）: {type(e).__name__}: {e}")
        print("       初回はネットワークが必要です。重みは ~/.cache/torch/hub に保存されます。")
        print("       （タグ付けにはモデルが必須のため、ここで終了します。exit 0）")
        return
    print(f"モデル: r3d_18 / クラス数: {len(categories)} / device: {H.get_device()}")

    # 2) 入力動画を集める（実データ優先 → 無ければ合成ライブラリ）
    videos = discover_videos(DATA_DIR)
    if videos:
        print(f"\n実データを使用: {DATA_DIR} に {len(videos)} 本の動画")
    else:
        print(f"\n実データ無し（{DATA_DIR} 空/不在）→ 合成ライブラリを生成してタグ付け")
        videos = make_synthetic_library(H.OUT_DIR)
        print(f"  合成 {len(videos)} 本を {H.OUT_DIR / 'use_case_synthetic'} に書き出し")

    # 3) フォルダを舐めてタグ・カタログを構築
    print("\n[タグ付け] 各動画 → cv2 デコード → サンプリング → r3d_18 → 上位タグ")
    catalog = build_catalog(model, categories, videos)
    if not catalog:
        print("\nタグ付けできる動画がありませんでした。data/ に動画を置いて再実行してください。")
        return

    # 4) カタログを保存（JSON / CSV）
    json_path = H.OUT_DIR / "use_case_tags.json"
    csv_path = H.OUT_DIR / "use_case_tags.csv"
    save_catalog_json(catalog, json_path)
    save_catalog_csv(catalog, csv_path)

    # 5) 可視化（サムネ一覧 + タグ頻度）
    save_gallery(catalog, H.OUT_DIR / "use_case_gallery.png")
    save_tag_frequency(catalog, H.OUT_DIR / "use_case_tagfreq.png")

    # 6) "あとから探せる" デモ: ライブラリで最頻の主タグの語で横断検索
    top_tag = Counter(e["primary_tag"] for e in catalog).most_common(1)[0][0]
    query = top_tag.split()[0]  # タグの先頭語で部分一致検索
    hits = search_by_tag(catalog, query)
    print(f"\n[検索デモ] タグに '{query}' を含む動画: {len(hits)} 本")
    for e in hits[:5]:
        print(f"  - {e['video']} (primary={e['primary_tag']}, {e['primary_prob']:.2f})")

    print("\n完了。保存物:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {H.OUT_DIR / 'use_case_gallery.png'}")
    print(f"  {H.OUT_DIR / 'use_case_tagfreq.png'}")
    print("\nヒント: data/29_video_action_recognition/ に実写動画を置くと、タグが意味を持ちます。")


if __name__ == "__main__":
    main()
