"""第13回 演習問題（画像分類と転移学習）。

使い方:
  1. 各 exN_*() の中の TODO を実装する（最初は NotImplementedError が出るが、採点ランナーが
     拾うのでプロセスは落ちず FAIL 表示になるだけ）。
  2. 自己採点:
         uv run python lectures/13_classification_transfer_learning/exercises.py
  3. どうしても分からない時だけ模範解答の挙動を見る:
         SHOW_SOLUTION=1 uv run python lectures/13_classification_transfer_learning/exercises.py

5問は本モジュールの核を1つずつ抜き出したもの:
  ex1: device 自動判定 (cuda → mps → cpu)                       … dl_helpers.get_device
  ex2: ImageProcessor がやること（0-1化 → ImageNet 正規化）      … 02
  ex3: 転移学習モデルの組み立て（バックボーン凍結 + 新ヘッド）   … 03
  ex4: top-k accuracy の定義どおりの計算                          … 03 / 14
  ex5: コサイン類似度（L2正規化してから内積）                    … 02 / 15
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


def _install_solutions() -> None:
    g = globals()
    g["ex1_pick_device"] = _sol_ex1
    g["ex2_imagenet_normalize"] = _sol_ex2
    g["ex3_make_transfer_model"] = _sol_ex3
    g["ex4_topk_accuracy"] = _sol_ex4
    g["ex5_cosine_sim"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
