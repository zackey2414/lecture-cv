"""第13回 演習問題（画像分類と転移学習）。

使い方:
  1. 各 exN_*() の中の TODO を実装する（最初は NotImplementedError が出るが、採点ランナーが
     拾うのでプロセスは落ちず FAIL 表示になるだけ）。
  2. 自己採点:
         uv run python lectures/13_classification_transfer_learning/exercises.py
  3. どうしても分からない時だけ模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/13_classification_transfer_learning/exercises.py
     完全な解答は exercises_solutions.py にもある（実行すると全 PASS）。

9 問は本モジュールの核を易→難で 1 つずつ抜き出したもの:
  ex1: device 自動判定 (cuda → mps → cpu)                       … dl_helpers.get_device
  ex2: ImageProcessor がやること（0-1化 → ImageNet 正規化）      … 02
  ex3: 転移学習モデルの組み立て（バックボーン凍結 + 新ヘッド）   … 03
  ex4: top-k accuracy の定義どおりの計算                          … 03 / 14
  ex5: コサイン類似度（L2正規化してから内積）                    … 02 / 15
  ex6: top-k のラベル名をスコア降順で返す（id2label 後処理）     … 01 / 03
  ex7: 混同行列を numpy で組む（row=正解, col=予測）             … 03 / 14
  ex8: 混同行列から macro recall（クラス平均の再現率）           … 03 / 14
  ex9: 最近傍重心分類（学習不要の転移ベースライン, コサイン）    … 02 / mini_project
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_pick_device(cuda_available: bool, mps_available: bool) -> str:
    """演習1: 利用可能性から使うべき device 名を返す。

    優先順位は cuda → mps → cpu。
      - cuda_available が True なら "cuda"
      - そうでなく mps_available が True なら "mps"
      - どちらも False なら "cpu"
    """
    # TODO: 早期 return で3分岐を書く
    raise NotImplementedError


def ex2_imagenet_normalize(pixels_uint8: np.ndarray) -> np.ndarray:
    """演習2: ImageProcessor の中身を手で再現する。

    入力: (H, W, 3) の uint8 画像（値域 0-255、チャンネル順は RGB）。
    手順:
      1. float32 にして 255 で割る（0-1 に rescale）。
      2. (3, H, W) に転置する（CHW 形式。深層モデルの入力レイアウト）。
      3. チャンネルごとに (x - mean) / std で正規化（mean/std は IMAGENET_MEAN/STD）。
    返り値: (3, H, W) の float32 配列。
    """
    # TODO: rescale → 転置 → 正規化（チャンネル軸のブロードキャストに注意）
    raise NotImplementedError


def ex3_make_transfer_model(
    backbone: nn.Module, in_features: int, num_classes: int
) -> nn.Module:
    """演習3: 転移学習モデルを組み立てる（特徴抽出器を凍結し、新しいヘッドを付ける）。

    手順:
      1. backbone の全パラメータを requires_grad_(False) で凍結する。
      2. head = nn.Linear(in_features, num_classes) を作る（新規なので requires_grad=True）。
      3. nn.Sequential(backbone, head) を返す。
    （こうすると学習対象は head だけになる。）
    """
    # TODO: 凍結 → ヘッド作成 → Sequential で連結して返す
    raise NotImplementedError


def ex4_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """演習4: top-k accuracy を定義どおり計算する。

    logits: (N, C) のスコア、targets: (N,) の正解クラス。
    各サンプルについて「スコア上位 k クラスのどれかに正解が含まれていれば当たり」。
    当たった割合(0.0〜1.0)を float で返す。N==0 のときは 0.0。
    ヒント: torch.topk(logits, k, dim=1).indices と targets.unsqueeze(1) の一致を見る。
    """
    # TODO: 上位 k クラスに正解が入っている割合を返す
    raise NotImplementedError


def ex5_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """演習5: コサイン類似度を「L2正規化してから内積」で計算する。

    a, b: 同じ長さの1次元ベクトル。
    手順: それぞれを L2 正規化（torch.nn.functional.normalize, p=2）してから内積を取り float で返す。
    （CLIP の get_*_features 等は未正規化なので、この正規化を忘れると検索が崩れる、が本モジュールの教訓。）
    """
    # TODO: F.normalize で正規化 → 内積
    raise NotImplementedError


def ex6_topk_labels(logits: torch.Tensor, labels: list[str], k: int) -> list[str]:
    """演習6: 上位 k クラスの「ラベル名」をスコア降順で返す（pipeline 風の後処理）。

    logits: (C,) の1次元スコア、labels: 長さ C のラベル名リスト（index→名前）、k: int。
    手順: スコアが高い順に上位 k 個のインデックスを取り、labels で名前に変換してリストで返す。
      例) logits=[0.1, 5.0, 2.0, 3.0], labels=["a","b","c","d"], k=2 → ["b", "d"]
    （生の argmax インデックスのままでは人が読めない。id2label でラベル名に直す、が本モジュールの教訓。）
    ヒント: torch.topk(logits, k).indices を Python の int リストにして labels を引く。
    """
    # TODO: 上位 k の index → ラベル名のリスト
    raise NotImplementedError


def ex7_confusion_matrix(
    preds: np.ndarray, targets: np.ndarray, num_classes: int
) -> np.ndarray:
    """演習7: 混同行列を numpy で組む（row=正解クラス, col=予測クラス）。

    preds, targets: 同じ長さの整数配列（クラスインデックス 0..num_classes-1）。
    返り値: (num_classes, num_classes) の int64 配列 cm。
            cm[t, p] = 「正解が t で予測が p だったサンプル数」。
    （対角 cm[i, i] が大きいほど良い。どのクラスをどのクラスと取り違えたかが読める。）
    ヒント: zip(targets, preds) で1件ずつ cm[t, p] += 1。
    """
    # TODO: ゼロ行列を作り、(正解, 予測) のペアごとに加算
    raise NotImplementedError


def ex8_macro_recall(confusion: np.ndarray) -> float:
    """演習8: 混同行列から macro recall（クラスごとの再現率の単純平均）を計算する。

    confusion: (C, C) の混同行列（row=正解, col=予測。ex7 と同じ向き）。
    各クラス i の recall = confusion[i, i] / (i 行の総和)。
    行和が 0 のクラス（そのクラスの正解サンプルが無い）は平均から除外する。
    有効なクラスが 1 つも無ければ 0.0 を返す。返り値は float。
    （macro 平均は各クラスを平等に扱うので、クラス不均衡に強い。）
    """
    # TODO: 各クラスの recall を集めて（行和0は除外して）平均
    raise NotImplementedError


def ex9_nearest_centroid(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    query_emb: torch.Tensor,
) -> torch.Tensor:
    """演習9: 最近傍重心分類（学習不要の転移ベースライン, コサイン類似度）。

    train_emb: (N, D) の学習埋め込み、train_labels: (N,) の long、query_emb: (M, D)。
    手順:
      1. クラスごとに train_emb の平均ベクトル（重心 centroid）を計算 → (num_classes, D)。
         （あるクラスのサンプルが無ければ、その重心はゼロベクトルのままでよい）
      2. 重心とクエリをそれぞれ L2 正規化する（F.normalize, p=2, dim=-1）。
      3. クエリ×重心のコサイン類似度 (M, num_classes) を計算し、argmax で各クエリのクラスを返す。
    返り値: (M,) の long tensor（予測クラス）。
    （埋め込みが既にクラスを分離していれば、線形ヘッドを学習しなくてもそこそこ当たる。）
    """
    # TODO: 重心を作る → 正規化 → コサイン類似度 → argmax
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー
# =====================================================================

def _grade() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        cases = {
            (True, True): "cuda",
            (True, False): "cuda",
            (False, True): "mps",
            (False, False): "cpu",
        }
        ok = all(ex1_pick_device(c, m) == exp for (c, m), exp in cases.items())
        return ok, "cuda→mps→cpu の優先順位"
    check("ex1_pick_device", _c1)

    def _c2():
        px = (np.arange(2 * 2 * 3).reshape(2, 2, 3) % 255).astype(np.uint8)
        got = ex2_imagenet_normalize(px)
        ref = _sol_ex2(px)
        ok = (
            isinstance(got, np.ndarray)
            and got.shape == (3, 2, 2)
            and got.dtype == np.float32
            and np.allclose(got, ref, atol=1e-5)
        )
        return ok, "0-1化 + CHW + ImageNet 正規化"
    check("ex2_imagenet_normalize", _c2)

    def _c3():
        backbone = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 4))
        model = ex3_make_transfer_model(backbone, in_features=4, num_classes=3)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        # 学習対象は head(nn.Linear(4,3)) のみ = 4*3 + 3 = 15。backbone は全て凍結。
        ok = isinstance(model, nn.Module) and trainable == 15 and frozen > 0
        return ok, f"trainable={trainable} (head=15), frozen={frozen}"
    check("ex3_make_transfer_model", _c3)

    def _c4():
        logits = torch.tensor(
            [
                [3.0, 1.0, 0.0, 2.0],  # 正解0 → top1 で当たり
                [0.0, 1.0, 2.0, 0.5],  # 正解1 → top1(=2)外れ, top2(=2,1)で当たり
                [1.0, 0.0, 0.0, 0.0],  # 正解3 → top2(=0,1)外れ
            ]
        )
        targets = torch.tensor([0, 1, 3])
        a1 = ex4_topk_accuracy(logits, targets, k=1)  # 1/3
        a2 = ex4_topk_accuracy(logits, targets, k=2)  # 2/3
        empty = ex4_topk_accuracy(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), k=1)
        ok = abs(a1 - 1 / 3) < 1e-6 and abs(a2 - 2 / 3) < 1e-6 and empty == 0.0
        return ok, f"top1={a1:.3f}, top2={a2:.3f}"
    check("ex4_topk_accuracy", _c4)

    def _c5():
        a = torch.tensor([3.0, 0.0, 0.0])
        b = torch.tensor([10.0, 0.0, 0.0])  # 向き同じ・長さ違い → cos=1
        c = torch.tensor([0.0, 5.0, 0.0])  # 直交 → cos=0
        s_same = ex5_cosine_sim(a, b)
        s_orth = ex5_cosine_sim(a, c)
        ok = abs(s_same - 1.0) < 1e-6 and abs(s_orth - 0.0) < 1e-6
        return ok, f"same={s_same:.3f}, orth={s_orth:.3f}"
    check("ex5_cosine_sim", _c5)

    def _c6():
        logits = torch.tensor([0.1, 5.0, 2.0, 3.0])
        labels = ["a", "b", "c", "d"]
        got2 = ex6_topk_labels(logits, labels, 2)
        got1 = ex6_topk_labels(logits, labels, 1)
        ok = (
            isinstance(got2, list)
            and got2 == ["b", "d"]   # スコア降順: 5.0(b) → 3.0(d)
            and got1 == ["b"]
        )
        return ok, f"top2={got2}"
    check("ex6_topk_labels", _c6)

    def _c7():
        preds = np.array([0, 1, 2, 1, 0])
        targets = np.array([0, 1, 1, 1, 2])
        cm = ex7_confusion_matrix(preds, targets, 3)
        ref = _sol_ex7(preds, targets, 3)
        ok = (
            isinstance(cm, np.ndarray)
            and cm.shape == (3, 3)
            and int(cm.sum()) == 5
            and np.array_equal(cm, ref)
        )
        return ok, f"diag={[int(cm[i, i]) for i in range(3)] if isinstance(cm, np.ndarray) else '?'}"
    check("ex7_confusion_matrix", _c7)

    def _c8():
        cm = np.array([[8, 2, 0], [0, 10, 0], [1, 1, 8]])  # 各行和=10 → recall 0.8,1.0,0.8
        mr = ex8_macro_recall(cm)
        cm2 = np.array([[5, 0], [0, 0]])  # 行1は正解サンプル無し → 除外 → class0 のみ recall=1.0
        mr2 = ex8_macro_recall(cm2)
        ok = abs(mr - (0.8 + 1.0 + 0.8) / 3) < 1e-6 and abs(mr2 - 1.0) < 1e-6
        return ok, f"macro_recall={mr:.4f}, empty_row_skip={mr2:.3f}"
    check("ex8_macro_recall", _c8)

    def _c9():
        # クラス0は (1,0) 付近、クラス1は (0,1) 付近に分離した埋め込み。
        train_emb = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        train_labels = torch.tensor([0, 0, 1, 1])
        query = torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.8, 0.2]])  # → 0, 1, 0
        pred = ex9_nearest_centroid(train_emb, train_labels, 2, query)
        ref = _sol_ex9(train_emb, train_labels, 2, query)
        ok = (
            isinstance(pred, torch.Tensor)
            and pred.shape == (3,)
            and torch.equal(pred.long(), torch.tensor([0, 1, 0]))
            and torch.equal(pred.long(), ref.long())
        )
        return ok, f"pred={pred.tolist() if isinstance(pred, torch.Tensor) else '?'}"
    check("ex9_nearest_centroid", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:24s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のとき本体へ差し替えて実行）。まず自力で。
# =====================================================================

def _sol_ex1(cuda_available: bool, mps_available: bool) -> str:
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _sol_ex2(pixels_uint8: np.ndarray) -> np.ndarray:
    x = pixels_uint8.astype(np.float32) / 255.0  # 0-1
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    mean = IMAGENET_MEAN.reshape(3, 1, 1)
    std = IMAGENET_STD.reshape(3, 1, 1)
    return ((x - mean) / std).astype(np.float32)


def _sol_ex3(backbone: nn.Module, in_features: int, num_classes: int) -> nn.Module:
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = nn.Linear(in_features, num_classes)
    return nn.Sequential(backbone, head)


def _sol_ex4(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.shape[0] == 0:
        return 0.0
    topk = logits.topk(k, dim=1).indices  # (N, k)
    hit = (topk == targets.unsqueeze(1)).any(dim=1)  # (N,)
    return float(hit.float().mean().item())


def _sol_ex5(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = torch.nn.functional.normalize(b, p=2, dim=-1)
    return float((a * b).sum().item())


def _sol_ex6(logits: torch.Tensor, labels: list[str], k: int) -> list[str]:
    idx = torch.topk(logits, k).indices.tolist()  # スコア降順の index
    return [labels[i] for i in idx]


def _sol_ex7(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1  # 行=正解, 列=予測
    return cm


def _sol_ex8(confusion: np.ndarray) -> float:
    cm = np.asarray(confusion, dtype=np.float64)
    row_sums = cm.sum(axis=1)
    recalls = [cm[i, i] / row_sums[i] for i in range(cm.shape[0]) if row_sums[i] > 0]
    return float(np.mean(recalls)) if recalls else 0.0


def _sol_ex9(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    query_emb: torch.Tensor,
) -> torch.Tensor:
    dim = train_emb.shape[1]
    centroids = torch.zeros(num_classes, dim)
    for c in range(num_classes):
        mask = train_labels == c
        if mask.any():
            centroids[c] = train_emb[mask].mean(dim=0)
    cen = torch.nn.functional.normalize(centroids, p=2, dim=-1)
    q = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
    sims = q @ cen.t()  # (M, num_classes)
    return sims.argmax(dim=1)


def _install_solutions() -> None:
    """模範解答で TODO 関数を差し替える（教材検証・答え合わせ用）。"""
    g = globals()
    g["ex1_pick_device"] = _sol_ex1
    g["ex2_imagenet_normalize"] = _sol_ex2
    g["ex3_make_transfer_model"] = _sol_ex3
    g["ex4_topk_accuracy"] = _sol_ex4
    g["ex5_cosine_sim"] = _sol_ex5
    g["ex6_topk_labels"] = _sol_ex6
    g["ex7_confusion_matrix"] = _sol_ex7
    g["ex8_macro_recall"] = _sol_ex8
    g["ex9_nearest_centroid"] = _sol_ex9


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
