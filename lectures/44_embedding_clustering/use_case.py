"""実践ユースケース: 写真の自動フォルダ分け（auto-organizer）。

このスクリプトは「散らかった画像フォルダを、中身（概念）ごとのサブフォルダに自動で
仕分けする小ツール」の“動く出発点”です。本章で学んだ
  埋め込み → L2 正規化 → クラスタリング → 評価 → 可視化
の骨格を、そのまま現実の作業（写真整理）に落とし込みます。mini_project.py が
「画像＋テキストの統合デモ（ベンチ寄り）」だったのに対し、こちらは
**実在の画像フォルダを受け取り、サブフォルダ分けの“提案”を出す**ことだけに集中した実ツールです。

------------------------------------------------------------------------
★ 自分の写真で実運用するには（ここが本番）★
  data/44_embedding_clustering/ に自分の画像（jpg/png/webp ...）を置くだけ。
  サブフォルダに分かれていても再帰的に拾います。画像が見つかればそれを使い、
  無ければ合成データ（色×形の6グループ）で必ず完走します（exit 0）。

  実行例:
    # 1) 提案だけ（ドライラン。何も移動しない。既定）
    uv run python lectures/44_embedding_clustering/use_case.py
    # 2) クラスタ数を自分で指定（例: 8 個のフォルダに分けたい）
    uv run python lectures/44_embedding_clustering/use_case.py --k 8
    # 3) 提案どおりに実際にコピーして整理する（元ファイルは消さない・コピー）
    uv run python lectures/44_embedding_clustering/use_case.py --apply

  出力（すべて lectures/44_embedding_clustering/outputs/ 配下）:
    use_case_organizer.png       … クラスタ別アルバム（行＝提案サブフォルダ）
    use_case_organizer_plan.json … 仕分け計画（どのファイルをどのフォルダへ）
    organized/<folder>/...       … --apply のときだけ。実際にコピーした結果

------------------------------------------------------------------------
★ 拡張アイデア（練習）★
  - フォルダ名の付け方を賢く: PHOTO_VOCAB を自分の用途（旅行/料理/家族 ...）に書き換える。
  - クラスタ数 k の決め方: 今は silhouette 最大で自動選択。エルボー法や固定 k と比べる。
  - 重複・ニアデュープ検出: 同一クラスタ内でコサイン類似が極端に高いペアを「重複候補」に。
  - 手法の差し替え: KMeans を AgglomerativeClustering(distance_threshold=...) に変えて
    「k を決めずに」整理する（§5）。DBSCAN なら「どの束にも入らない外れ写真」を弾ける。
  - 大規模化: 数万枚なら MiniBatchKMeans＋FAISS（vector グループ）へ（README §発展）。
  - GUI/CLI 化: plan.json を読んで実際のフォルダ操作を行う別ツールに分離する。

設計メモ:
  - CPU・headless（matplotlib=Agg、cv2.imshow は使わない）。図中文字は ASCII（CJK 豆腐回避）。
  - 共通処理（device 判定・合成データ・CLIP 埋め込み・正規化）は道具箱 cluster_lab.py を借用。
    番号付きスクリプト（01_*.py 等）は import できないが、cluster_lab.py は import 可能。
  - モデルは本章と同じ小型 CLIP（openai/clip-vit-base-patch32）。ネットは初回の重みDLのみ。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

import matplotlib

matplotlib.use("Agg")  # headless でも図を保存できるバックエンド（GUI 不要）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

import cluster_lab as cl  # noqa: E402  道具箱（埋め込み・正規化・device 判定）

# 読み込む画像拡張子（小文字で比較）。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
# CPU で現実的な上限。これを超える分は先頭から間引く（拡張時は増やしてよい）。
MAX_IMAGES = 200

# 実写真フォルダ用の語彙: (フォルダ名, CLIP に渡すプロンプト)。
# クラスタの中心ベクトルに最も近いプロンプトを選び、その「フォルダ名」を提案名にする。
# ★自分の用途に合わせて書き換えるのが一番効く拡張ポイント。
PHOTO_VOCAB: list[tuple[str, str]] = [
    ("people", "a photo of a person or people"),
    ("animals", "a photo of an animal or a pet"),
    ("food", "a photo of food or a meal"),
    ("nature", "a photo of nature, mountains or landscape"),
    ("buildings", "a photo of a building or architecture"),
    ("street", "a photo of a city street"),
    ("vehicles", "a photo of a car, train or vehicle"),
    ("plants", "a photo of flowers or plants"),
    ("water", "a photo of the sea, a lake or a river"),
    ("documents", "a screenshot, document or text"),
    ("indoor", "an indoor room scene"),
    ("sky", "a photo of the sky or clouds"),
]


# ---------------------------------------------------------------------------
# 入力データ: 実フォルダ優先・無ければ合成にフォールバック
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """フォルダ名に使える安全な文字列へ（小文字・英数とアンダースコアのみ）。"""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "cluster"


def load_real_images(
    data_dir: pathlib.Path, max_n: int = MAX_IMAGES
) -> tuple[list[Image.Image], list[str]]:
    """data_dir 配下（再帰）の画像を読み込む。読めないファイルは黙ってスキップ。

    返り値: (PIL.Image[RGB] のリスト, 元ファイルパス文字列のリスト)。
    1 枚も無ければ ([], []) を返し、呼び出し側が合成にフォールバックする。
    """
    if not data_dir.exists():
        return [], []
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    images: list[Image.Image] = []
    names: list[str] = []
    for p in paths[:max_n]:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:  # 壊れた画像・非画像は飛ばす（実運用では普通に混ざる）
            continue
        images.append(img)
        names.append(str(p))
    return images, names


def load_inputs(
    data_dir: pathlib.Path,
) -> tuple[list[Image.Image], list[str], list[tuple[str, str]], str]:
    """整理対象の画像・表示名・語彙・データ種別を決める。

    実データがあれば実写真語彙(PHOTO_VOCAB)、無ければ合成データ＋合成語彙(色×形)。
    語彙はクラスタの自動命名（どのサブフォルダ名にするか）に使う。
    """
    images, names = load_real_images(data_dir)
    if images:
        return images, names, PHOTO_VOCAB, str(data_dir)

    # --- 合成フォールバック: 道具箱の色×形 6 グループを“バラバラの1フォルダ”に見立てる ---
    syn_images, _labels, _group_names = cl.make_image_collection(n_per_group=6, seed=0)
    names = [f"synthetic_{i:04d}.png" for i in range(len(syn_images))]
    # 合成語彙はグループ名そのもの（CLIP が色＋形を強く捉えるので命名が当たる）。
    vocab = [(slugify(g[0]), f"a photo of a {g[0]}") for g in cl.GROUPS]
    return syn_images, names, vocab, "synthetic(色×形6グループ)"


# ---------------------------------------------------------------------------
# クラスタ数 k の自動選択（silhouette 最大）
# ---------------------------------------------------------------------------
def choose_k(emb: np.ndarray, k_max: int = 10) -> tuple[int, list[int], list[float]]:
    """silhouette が最大になる k を 2..k_max で選ぶ（§4）。

    枚数が少ないときは安全に縮め、k を取れないときは 1（全部 1 フォルダ）を返す。
    """
    n = len(emb)
    k_hi = min(k_max, n - 1)
    if k_hi < 2:
        return 1, [], []
    ks, sils = [], []
    for k in range(2, k_hi + 1):
        pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        ks.append(k)
        sils.append(float(silhouette_score(emb, pred, metric="cosine")))
    best_k = int(ks[int(np.argmax(sils))])
    return best_k, ks, sils


# ---------------------------------------------------------------------------
# クラスタの自動命名（中心ベクトル × CLIP テキスト語彙）
# ---------------------------------------------------------------------------
def name_clusters(
    emb: np.ndarray,
    pred: np.ndarray,
    vocab: list[tuple[str, str]],
    device,
) -> dict[int, tuple[str, float]]:
    """各クラスタに「最も近い語彙ラベル」を割り当てる（ゼロショット命名）。

    手順: クラスタ内埋め込みの平均→再 L2 正規化＝中心ベクトル。語彙プロンプトも
    CLIP で埋め込み（同じ 512 次元空間）、中心とのコサイン類似が最大のラベルを採用。
    返り値: {cluster_id: (folder_name, confidence)}。同名衝突は _<id> で一意化。
    """
    folder_names = [v[0] for v in vocab]
    prompts = [v[1] for v in vocab]
    vocab_emb = cl.embed_texts(prompts, device)  # (V, 512) 正規化済み

    result: dict[int, tuple[str, float]] = {}
    used: set[str] = set()
    for c in sorted(set(pred.tolist())):
        members = emb[pred == c]
        center = cl.l2_normalize(members.mean(axis=0, keepdims=True))[0]  # (512,)
        sims = vocab_emb @ center  # コサイン類似（両者正規化済みなので内積でよい）
        best = int(np.argmax(sims))
        name = folder_names[best]
        if name in used:  # 別クラスタが同じラベルを引いたら id を付けて区別
            name = f"{name}_{c}"
        used.add(name)
        result[c] = (name, float(sims[best]))
    return result


# ---------------------------------------------------------------------------
# 可視化: クラスタ別アルバム（行＝提案サブフォルダ）
# ---------------------------------------------------------------------------
def save_album(
    images: list[Image.Image],
    pred: np.ndarray,
    cluster_names: dict[int, tuple[str, float]],
    path: pathlib.Path,
    max_per_row: int = 8,
) -> None:
    """予測クラスタごとに代表画像を 1 行に並べ、左に提案フォルダ名を書いて保存する。

    図中の文字は ASCII（CJK 豆腐回避）。行ラベルは "folder (nXX)" 形式。
    """
    clusters = sorted(set(pred.tolist()))
    nrows = len(clusters)
    fig, axes = plt.subplots(
        nrows, max_per_row, figsize=(max_per_row * 1.3, nrows * 1.45), squeeze=False
    )
    for r, c in enumerate(clusters):
        idxs = np.where(pred == c)[0][:max_per_row]
        folder, conf = cluster_names[c]
        count = int(np.sum(pred == c))
        for col in range(max_per_row):
            ax = axes[r][col]
            if col < len(idxs):
                ax.imshow(images[idxs[col]])
            ax.set_xticks([])
            ax.set_yticks([])
            if col > 0 or col >= len(idxs):
                ax.axis("off")
        # 左端セルに提案フォルダ名を縦書き風（横ラベル）で添える。
        axes[r][0].axis("on")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
        axes[r][0].set_ylabel(
            f"{folder}\n(n={count}, s={conf:.2f})",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=44,
        )
    fig.suptitle("Auto-organizer: proposed subfolders (row = cluster)", fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 仕分け計画 → JSON ／ --apply での実コピー
# ---------------------------------------------------------------------------
def build_plan(
    pred: np.ndarray,
    file_names: list[str],
    cluster_names: dict[int, tuple[str, float]],
    source: str,
    silhouette: float,
) -> dict:
    """「どのファイルをどのフォルダへ」をまとめた仕分け計画を dict で返す。"""
    clusters = []
    for c in sorted(set(pred.tolist())):
        idxs = np.where(pred == c)[0]
        folder, conf = cluster_names[c]
        clusters.append(
            {
                "cluster_id": int(c),
                "folder": folder,
                "confidence": round(conf, 4),
                "n_files": int(len(idxs)),
                "files": [file_names[i] for i in idxs],
            }
        )
    return {
        "source": source,
        "n_images": int(len(file_names)),
        "n_clusters": len(clusters),
        "silhouette": None if np.isnan(silhouette) else round(float(silhouette), 4),
        "clusters": clusters,
    }


def apply_plan(
    plan: dict,
    images: list[Image.Image],
    file_names: list[str],
    out_root: pathlib.Path,
) -> int:
    """計画どおりに organized/<folder>/ へコピー（元は消さない）。コピー枚数を返す。

    実ファイルがあればそれをコピー、合成データならメモリ上の PIL を書き出す。
    """
    name_to_idx = {n: i for i, n in enumerate(file_names)}
    copied = 0
    for cluster in plan["clusters"]:
        dst_dir = out_root / "organized" / cluster["folder"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in cluster["files"]:
            src = pathlib.Path(fname)
            if src.exists():  # 実ファイル → そのままコピー（中身を保持）
                shutil.copy2(src, dst_dir / src.name)
            else:  # 合成データ → PIL を PNG 保存
                images[name_to_idx[fname]].save(dst_dir / pathlib.Path(fname).name)
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="写真の自動フォルダ分け（auto-organizer）")
    parser.add_argument(
        "--data-dir",
        default=f"data/{cl.MODULE_ID}",
        help="整理したい画像フォルダ（既定: data/44_embedding_clustering/。無ければ合成）",
    )
    parser.add_argument(
        "--k", type=int, default=0, help="サブフォルダ数を固定（0=silhouette で自動選択）"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="提案どおりに lectures/.../outputs/organized/ へ実際にコピーする（既定はドライラン）",
    )
    args = parser.parse_args()

    out = cl.ensure_out_dir()
    device = cl.get_device()
    data_dir = pathlib.Path(args.data_dir)

    # --- 1) 入力（実データ優先・合成フォールバック）---
    images, file_names, vocab, source = load_inputs(data_dir)
    print(f"device={device}  source={source}  images={len(images)} 枚")
    if source.startswith("synthetic"):
        print(f"（data/{cl.MODULE_ID}/ に自分の画像を置くと実運用になります）")

    # --- 2) 埋め込み（CLIP・L2 正規化済み 512 次元）---
    emb = cl.embed_images(images, device)
    print(f"埋め込み: shape={emb.shape}  各行ノルム≈{np.linalg.norm(emb, axis=1).mean():.3f}")

    # --- 3) クラスタ数 k を決める（指定 or 自動）---
    if args.k and args.k >= 1:
        k = min(args.k, len(images))
        print(f"k={k}（ユーザ指定）")
    else:
        k, ks, sils = choose_k(emb)
        if ks:
            print(f"k={k}（silhouette 自動選択: " + ", ".join(f"k{a}={b:.2f}" for a, b in zip(ks, sils)) + ")")
        else:
            print(f"k={k}（枚数が少ないため単一フォルダ）")

    # --- 4) クラスタリング ---
    if k <= 1:
        pred = np.zeros(len(images), dtype=int)  # 全部まとめて 1 フォルダ
        sil = float("nan")
    else:
        pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        sil = float(silhouette_score(emb, pred, metric="cosine"))
    print(f"クラスタリング完了: {len(set(pred.tolist()))} クラスタ / silhouette={sil:.3f}")

    # --- 5) クラスタの自動命名（→ 提案サブフォルダ名）---
    cluster_names = name_clusters(emb, pred, vocab, device)

    # --- 6) 計画を JSON・アルバムを PNG に保存 ---
    plan = build_plan(pred, file_names, cluster_names, source, sil)
    plan_path = out / "use_case_organizer_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    album_path = out / "use_case_organizer.png"
    save_album(images, pred, cluster_names, album_path)

    print("\n=== 仕分け提案（ドライラン）===")
    for c in plan["clusters"]:
        print(f"  [{c['folder']:<14}] {c['n_files']:>3} 枚  (confidence={c['confidence']:.2f})")

    # --- 7) --apply なら実際にコピー ---
    if args.apply:
        copied = apply_plan(plan, images, file_names, out)
        print(f"\n--apply: {copied} 枚を {out / 'organized'}/<folder>/ へコピーしました（元は保持）。")
    else:
        print("\n（--apply を付けると lectures/.../outputs/organized/ へ実際にコピーします）")

    print(f"\n完了。計画: {plan_path}  アルバム: {album_path}")


if __name__ == "__main__":
    main()
