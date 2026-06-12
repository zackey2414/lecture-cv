"""実践ユースケース: 近重複(near-duplicate)画像ファインダー。

フォルダの中にある「重複・そっくりな画像」を、埋め込みのコサイン類似度で見つけて
グループにまとめる小ツールの【動く出発点】です。写真整理で「同じ写真の連写・JPEG 再保存・
リサイズ・明るさ調整だけ違うほぼ同じ画像」を炙り出し、各グループから 1 枚だけ残す——
といった現実の作業に直結します。章末ミニプロジェクト(mini_project.py)が「圧縮スイープの
ベンチ」だったのに対し、こちらは"そのまま使える小道具"として完結させてあります。

このモジュールで学んだ流れ（01 取り出す → 02 測る）をそのまま使います:
  画像 → resnet18 で 512 次元の埋め込み → L2 正規化 → コサイン類似度
  → しきい値以上のペアを Union-Find で連結 → 連結成分が「近重複グループ」。

━━━ 自分のデータで実運用にする ━━━
  data/15_image_embeddings_metric_learning/ に画像(.jpg/.png/...)を置くだけで、
  合成データではなく"あなたのフォルダ"を対象に動きます（2 枚以上あれば自動でそちらを使用）。
  画像が無ければ、近重複を仕込んだ合成データで必ず完走します（exit 0）。

実行:
  uv run python lectures/15_image_embeddings_metric_learning/use_case.py
結果は outputs/15_image_embeddings_metric_learning/ に保存（画面表示はしない）:
  - use_case_duplicate_groups.png : 見つかった近重複グループのモンタージュ
  - use_case_similarity_hist.png  : 各画像の最近傍コサイン分布（しきい値チューニング用）
  - use_case_groups.json          : グループ・メンバー・類似度・設定値

━━━ 練習(拡張)アイデア ━━━
  - SIM_THRESHOLD を上下させ、誤検出(別物を重複扱い)と取りこぼしのバランスを調整する。
    use_case_similarity_hist.png の「谷」を見て自動で閾値を決める実装に発展させても良い。
  - 埋め込みを CLIP(16 回)に差し替えると「意味的に似た画像(構図違いの同一被写体)」まで拾える。
    resnet18 は見た目寄り、CLIP は意味寄り——用途で選ぶ。
  - 各グループの代表 1 枚（最大解像度など）を残し、他を別フォルダへ退避する整理アクションを足す。
  - pHash(知覚ハッシュ)を前段の粗いフィルタにして、埋め込み計算を減らす二段構えにする。
  - 画像が数万枚に増えたら、総当たり O(N^2) をやめて FAISS(17 回)で近傍だけ引く。
"""

from __future__ import annotations

import io
import json
import os

import matplotlib

matplotlib.use("Agg")  # 画面の無い環境でも図を保存できるよう、必ず pyplot より前に設定する
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image, ImageEnhance, ImageFilter

import embed_helpers as eh

# --- 設定（ここを触って挙動を調整できる）-----------------------------------
DATA_DIR = os.path.join("data", eh.MODULE_ID)  # ここに自分の画像を置けば実運用になる
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # 読み込む拡張子
MAX_IMAGES = 200  # CPU で現実的な上限（総当たりは O(N^2) なので増やしすぎ注意）
SIM_THRESHOLD = 0.90  # この値以上のコサイン類似ペアを「近重複」とみなす（要チューニング）
MODEL_NAME = "resnet18"  # 小型・CPU で軽い。CLIP に差し替えれば意味的な近重複も拾える
SEED = 0


# ---------------------------------------------------------------------------
# 1. 入力画像の用意（実データ優先・無ければ合成データへフォールバック）
# ---------------------------------------------------------------------------
def load_real_images(data_dir: str) -> list[tuple[str, Image.Image]]:
    """data_dir 直下の画像ファイルを (表示名, PIL.RGB) のリストで返す。無ければ空リスト。"""
    if not os.path.isdir(data_dir):
        return []
    names = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(IMG_EXTS))
    items: list[tuple[str, Image.Image]] = []
    for name in names[:MAX_IMAGES]:
        path = os.path.join(data_dir, name)
        try:
            items.append((name, Image.open(path).convert("RGB")))  # RGB に揃える
        except Exception as e:  # 壊れたファイルは黙って飛ばす（実運用では普通に起きる）
            print(f"  skip {name}: {e}")
    return items


def _perturb(img: Image.Image, kinds: tuple[str, ...], rng: np.random.Generator) -> Image.Image:
    """1 枚の画像に「軽い」加工を重ねて near-duplicate（そっくり画像）を作る。

    実世界で near-duplicate を生む典型操作を模擬する: 明るさ調整・わずかなボケ・
    JPEG 再圧縮・小さな回転・リサイズ往復。いずれも内容は保つので埋め込みは近いまま。
    """
    out = img
    for kind in kinds:
        if kind == "bright":
            out = ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.90, 1.10)))
        elif kind == "contrast":
            out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.92, 1.08)))
        elif kind == "blur":
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.6))))
        elif kind == "jpeg":  # JPEG 再保存で生じる圧縮ノイズ（連写・SNS 再投稿で頻発）
            buf = io.BytesIO()
            out.save(buf, format="JPEG", quality=int(rng.integers(82, 94)))
            buf.seek(0)
            out = Image.open(buf).convert("RGB")
        elif kind == "rotate":  # ごく小さな回転（手ブレ・スキャンのズレ）
            out = out.rotate(float(rng.uniform(-2.5, 2.5)), resample=Image.BILINEAR, fillcolor=(64, 64, 64))
        elif kind == "resize":  # 縮小→拡大の往復で解像度が落ちた版
            w, h = out.size
            small = out.resize((max(8, int(w * 0.82)), max(8, int(h * 0.82))), Image.BILINEAR)
            out = small.resize((w, h), Image.BILINEAR)
    return out


def make_synthetic_dataset(seed: int) -> tuple[list[tuple[str, Image.Image]], list[int]]:
    """近重複を「仕込んだ」合成データを作る。返り値は (items, planted_group)。

    手順:
      - embed_helpers の合成画像から、クラスごとに 1 枚ずつ"ユニークな原画像(base)"を取る。
      - 一部の base には _perturb で 1〜2 枚の near-duplicate を作って植え込む。
      - 残りの base は単独（重複なし）。最後にシャッフルして実フォルダ風に並べる。
    planted_group[i] は画像 i がどの base 由来か（=本来の正解グループ）。
    """
    rng = np.random.default_rng(seed)
    imgs, labels, names = eh.make_dataset(n_per_class=3, seed=7)
    # 各クラスの先頭 1 枚を「別物の原画像」として採用（6 クラス → 6 base）
    bases: list[Image.Image] = []
    for c in range(len(names)):
        idx = int(np.where(labels == c)[0][0])
        bases.append(imgs[idx])

    # base ごとに作る near-duplicate の枚数（0 なら単独画像）と、使う加工の組み合わせ
    plan = [2, 2, 2, 1, 0, 0]  # base0..2 は 2 枚ずつ複製、base3 は 1 枚、base4,5 は単独
    variant_kinds = [
        ("bright", "jpeg"),
        ("blur", "resize"),
        ("contrast", "rotate"),
        ("jpeg", "bright"),
    ]

    items: list[tuple[str, Image.Image]] = []
    planted: list[int] = []
    for b, (base, n_dup) in enumerate(zip(bases, plan)):
        items.append((f"base{b:02d}_orig", base))  # 原画像そのもの
        planted.append(b)
        for v in range(n_dup):
            dup = _perturb(base, variant_kinds[v % len(variant_kinds)], rng)
            items.append((f"base{b:02d}_dup{v}", dup))  # 近重複（base 由来）
            planted.append(b)

    order = rng.permutation(len(items))  # 実フォルダのように順序をシャッフル
    items = [items[i] for i in order]
    planted = [planted[i] for i in order]
    return items, planted


# ---------------------------------------------------------------------------
# 2. 埋め込み抽出（digit 始まりスクリプトは import 不可なので自己完結で書く）
# ---------------------------------------------------------------------------
@torch.inference_mode()  # 推論のみ。勾配を切って CPU のメモリ・時間を節約
def extract_embeddings(images: list[Image.Image], device: torch.device) -> np.ndarray:
    """timm resnet18(num_classes=0) のプール済み 512 次元埋め込みを (N,512) で返す（未正規化）。"""
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    # 前処理は手打ちせず、モデル固有の入力サイズ/平均/分散を自動生成する（ズレ事故を防ぐ）
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    feats: list[np.ndarray] = []
    for i in range(0, len(images), 16):  # ミニバッチでメモリを抑える
        batch = torch.stack([transform(im) for im in images[i : i + 16]]).to(device)
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# 3. 近重複グループの検出（Union-Find = 連結成分）
# ---------------------------------------------------------------------------
def find_duplicate_groups(emb_norm: np.ndarray, threshold: float) -> tuple[list[list[int]], np.ndarray]:
    """L2 正規化済み埋め込みから、コサイン類似度 >= threshold のペアを連結してグループ化。

    返り値は (グループのリスト, コサイン類似度行列)。
    グループは「2 枚以上が連結した連結成分」だけを返す（単独画像は重複ではないので除く）。
    """
    n = len(emb_norm)
    sim = emb_norm @ emb_norm.T  # 正規化済みなので内積 = コサイン類似度（対角は ~1.0）

    parent = list(range(n))  # Union-Find（素集合データ構造）

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 経路圧縮
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 上三角だけ走査し、しきい値以上のペアを連結する
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # メンバー 2 枚以上のグループのみ。大きい順 → 添字順で安定ソート
    dup_groups = sorted((sorted(m) for m in groups.values() if len(m) >= 2), key=lambda m: (-len(m), m[0]))
    return dup_groups, sim


# ---------------------------------------------------------------------------
# 4. 可視化・保存
# ---------------------------------------------------------------------------
def save_group_montage(
    items: list[tuple[str, Image.Image]],
    groups: list[list[int]],
    sim: np.ndarray,
    path: str,
    max_groups: int = 6,
    max_members: int = 6,
) -> None:
    """検出した近重複グループを「1 行=1 グループ」で並べたモンタージュを保存する。

    各行の先頭が代表画像。後続には代表とのコサイン類似度を ASCII タイトルで添える。
    グループが無い場合でも「見つからなかった」旨の図を必ず 1 枚出す（後工程が落ちないように）。
    """
    if not groups:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.text(0.5, 0.5, "no near-duplicate groups found\n(try lowering SIM_THRESHOLD)",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return

    shown = groups[:max_groups]
    ncols = min(max_members, max(len(g) for g in shown))
    fig, axes = plt.subplots(len(shown), ncols, figsize=(1.7 * ncols, 1.9 * len(shown)), squeeze=False)
    for r, g in enumerate(shown):
        rep = g[0]  # 代表（添字最小）
        members = g[:max_members]
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(members):
                mi = members[c]
                ax.imshow(np.asarray(items[mi][1]))
                tag = "rep" if c == 0 else f"cos={sim[rep, mi]:.2f}"
                ax.set_title(f"{tag}\n{items[mi][0][:14]}", fontsize=7)
                for sp in ax.spines.values():
                    sp.set_color("green")  # 同一グループは緑枠で囲う
                    sp.set_linewidth(2.0)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(f"group {r}\n(n={len(g)})", fontsize=8)
    fig.suptitle("Near-duplicate groups (cosine on resnet18 embeddings)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_similarity_hist(sim: np.ndarray, threshold: float, path: str) -> None:
    """各画像の「最近傍（自分以外で最も似た 1 枚）コサイン類似度」のヒストグラム。

    しきい値チューニングの道具。重複ペアは右側の山、無関係な画像は左側の山に分かれ、
    その「谷」あたりに threshold を置くのが定石。縦の破線が現在の SIM_THRESHOLD。
    """
    s = sim.copy()
    np.fill_diagonal(s, -1.0)  # 自分自身(=1.0)を除外
    nn_sim = s.max(axis=1)  # 各行の最近傍類似度

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(nn_sim, bins=20, range=(-0.2, 1.0), color="#4C78A8", edgecolor="white")
    ax.axvline(threshold, ls="--", color="red", label=f"threshold = {threshold:.2f}")
    ax.set_xlabel("nearest-neighbor cosine similarity")
    ax.set_ylabel("number of images")
    ax.set_title("Tune the threshold at the valley between the two humps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def group_stats(group: list[int], sim: np.ndarray) -> tuple[float, float]:
    """グループ内ペアのコサイン類似度の (最小, 平均) を返す（重複の"固さ"の目安）。"""
    vals = [float(sim[a, b]) for i, a in enumerate(group) for b in group[i + 1 :]]
    return (min(vals), float(np.mean(vals))) if vals else (1.0, 1.0)


# ---------------------------------------------------------------------------
# 5. パイプライン本体
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(eh.OUT_DIR, exist_ok=True)
    device = eh.get_device()
    print(f"device = {device}")

    # --- (1) 入力の用意: 実データがあれば使い、無ければ合成データへフォールバック ---
    real = load_real_images(DATA_DIR)
    if len(real) >= 2:
        items, planted, mode = real, None, "real"
        print(f"[1] 実データを使用: {DATA_DIR} から {len(items)} 枚")
    else:
        items, planted = make_synthetic_dataset(SEED)
        mode = "synthetic"
        print(f"[1] 実データ({DATA_DIR})が無いので合成データを使用: {len(items)} 枚")
        print(f"    → 自分の画像を {DATA_DIR}/ に置けば実運用になります。")

    images = [im for _, im in items]

    # --- (2) 埋め込み抽出 → L2 正規化 --------------------------------------
    print(f"[2] {MODEL_NAME} で埋め込み抽出（初回はモデルDLあり）…")
    emb = extract_embeddings(images, device)
    emb_norm = eh.l2_normalize(emb)  # 正規化後は内積 = コサイン類似度
    print(f"    埋め込み: {emb.shape} → L2 正規化済み")

    # --- (3) 近重複グループ検出 -------------------------------------------
    groups, sim = find_duplicate_groups(emb_norm, SIM_THRESHOLD)
    print(f"[3] しきい値 cos>= {SIM_THRESHOLD}: 近重複グループ {len(groups)} 個を検出")
    for r, g in enumerate(groups):
        mn, mean = group_stats(g, sim)
        member_names = ", ".join(items[i][0] for i in g)
        print(f"    group {r} (n={len(g)}, min_cos={mn:.3f}, mean_cos={mean:.3f}): {member_names}")
    if not groups:
        print("    （0 個。SIM_THRESHOLD を下げると拾いやすくなります）")

    # 合成データのときは「仕込んだ正解」とどれだけ一致したか軽くチェック（exit 0 は不変）
    sanity = None
    if mode == "synthetic" and planted is not None:
        planted_arr = np.asarray(planted)
        planted_groups = sum(1 for c in np.unique(planted_arr) if (planted_arr == c).sum() >= 2)
        recovered = 0
        for g in groups:  # グループ内が単一 base 由来で、その base の全枚数を回収できたら成功
            srcs = {planted[i] for i in g}
            if len(srcs) == 1:
                src = next(iter(srcs))
                if len(g) == int((planted_arr == src).sum()):
                    recovered += 1
        sanity = {"planted_groups": planted_groups, "fully_recovered": recovered}
        print(f"    [sanity] 仕込んだ重複グループ {planted_groups} 個中 {recovered} 個を完全回収")

    # --- (4) 図と JSON を保存 --------------------------------------------
    montage_path = os.path.join(eh.OUT_DIR, "use_case_duplicate_groups.png")
    hist_path = os.path.join(eh.OUT_DIR, "use_case_similarity_hist.png")
    json_path = os.path.join(eh.OUT_DIR, "use_case_groups.json")
    save_group_montage(items, groups, sim, montage_path)
    save_similarity_hist(sim, SIM_THRESHOLD, hist_path)

    payload = {
        "tool": "near-duplicate image finder (cosine on resnet18 embeddings)",
        "mode": mode,
        "data_dir": DATA_DIR,
        "model": f"timm {MODEL_NAME} (GAP 512d)",
        "sim_threshold": SIM_THRESHOLD,
        "n_images": int(len(items)),
        "n_groups": int(len(groups)),
        "groups": [
            {
                "group_id": r,
                "size": len(g),
                "min_cosine": round(group_stats(g, sim)[0], 4),
                "mean_cosine": round(group_stats(g, sim)[1], 4),
                "members": [{"index": int(i), "name": items[i][0]} for i in g],
            }
            for r, g in enumerate(groups)
        ],
    }
    if sanity is not None:
        payload["synthetic_sanity_check"] = sanity
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[4] 成果物を保存:")
    for p in (montage_path, hist_path, json_path):
        print(f"    {p}")
    print("\n完了。use_case_similarity_hist.png の谷を見て SIM_THRESHOLD を調整し、")
    print("自分の画像を data/15_image_embeddings_metric_learning/ に置いて試してください。")


if __name__ == "__main__":
    main()
