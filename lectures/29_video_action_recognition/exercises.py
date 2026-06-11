"""第29回 演習問題（動画理解・行動認識 ― サンプリング/正規化/評価のコア）。

使い方:
  1. 各関数の TODO を自分で実装する（最初は NotImplementedError だが、採点ランナーが
     拾うのでプロセスは落ちず FAIL と表示されるだけ＝必ず exit 0）。
  2. 自己採点:  uv run python lectures/29_video_action_recognition/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 模範解答（実行すると全 PASS）:
        uv run python lectures/29_video_action_recognition/exercises_solutions.py

この8問は本モジュールの核（モデル DL 不要・純計算）を易→難で1つずつ抜き出したもの:
  ex1: 等間隔フレームサンプリング（uniform）                                  … helpers/03
  ex2: clip_len + frame_rate(stride) サンプリング（クランプ付き）              … helpers/03
  ex3: クリップの専用正規化（(x-mean)/std, チャンネル別）                     … 02/03
  ex4: (T,H,W,C) → (1,C,T,H,W) への並べ替え（3D CNN の入力形）               … 02
  ex5: logits → softmax → top-k のインデックス取り出し                        … 01/02
  ex6: top-k accuracy（gt が上位 k に入る割合）                               … 03/helpers
  ex7: 混同行列（行=正解, 列=予測）                                           … 03
  ex8: 前処理を崩したときの top-k 一致率（pseudo-GT に対する robustness）      … 03
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_uniform_indices(total: int, num_frames: int) -> np.ndarray:
    """演習1: 全長 total から num_frames 枚を等間隔で選ぶインデックス(int 配列)を返す。

    - np.linspace(0, total-1, num=num_frames) で端から端まで均等に取り、四捨五入して int 化。
    - total < num_frames でも端点が重複しつつ必ず num_frames 個返る（落ちない）。
    例: ex1_uniform_indices(32, 8) -> [0, 4, 9, 13, 18, 22, 27, 31]
        ex1_uniform_indices(4, 8)  -> [0, 0, 1, 1, 2, 2, 3, 3]
    """
    # TODO: np.linspace → np.round → astype(int) した配列を返す
    raise NotImplementedError


def ex2_strided_indices(
    total: int, num_frames: int, frame_rate: int = 1, start: int | None = None
) -> np.ndarray:
    """演習2: clip_len(=num_frames) + frame_rate(=ストライド) 方式のサンプリング。

    - start から frame_rate おきに num_frames 枚: idx = start + arange(num_frames)*frame_rate。
    - 末尾を超えたら最終フレーム total-1 にクランプ（np.clip）。
    - start 未指定なら中央寄せ: span=(num_frames-1)*frame_rate+1; start=max(0,(total-span)//2)。
    例: ex2_strided_indices(32, 8, 4)            -> [1, 5, 9, 13, 17, 21, 25, 29]
        ex2_strided_indices(10, 8, 3, start=0)  -> [0, 3, 6, 9, 9, 9, 9, 9]
    """
    # TODO: start を決め、strided なインデックスを作り、0..total-1 にクランプして返す
    raise NotImplementedError


def ex3_normalize_clip(clip01: np.ndarray, mean, std) -> np.ndarray:
    """演習3: [0,1] スケールのクリップをチャンネル別に正規化する: (x - mean) / std。

    - clip01 は最後の軸がチャンネル(C=3)の配列（例: (T,H,W,3)）。
    - mean, std は長さ 3 のシーケンス。最後の軸に沿ってブロードキャストして適用する。
    - r3d_18 は ImageNet とは別の mean/std を使う点が重要（誤ると無意味な出力）。
    例: clip=0.5 一様, mean=std=(0.5,0.5,0.5)/(0.25,..) -> 全要素 0.0
    """
    # TODO: numpy のブロードキャストで (clip01 - mean) / std を返す
    raise NotImplementedError


def ex4_clip_to_ncthw(clip_thwc: np.ndarray) -> np.ndarray:
    """演習4: (T,H,W,C) のクリップを 3D CNN 入力の (1, C, T, H, W) へ並べ替える。

    - 軸の入れ替え: (T,H,W,C) -> (C,T,H,W) は np.transpose(clip,(3,0,1,2))。
    - 先頭にバッチ次元を足す: np.expand_dims(..., 0) で (1,C,T,H,W)。
    - （torch なら tensor.permute(3,0,1,2).unsqueeze(0) と等価。ここでは numpy で。）
    例: 入力 (4,8,10,3) -> 出力 shape (1,3,4,8,10)
    """
    # TODO: transpose で (C,T,H,W) にして、先頭にバッチ次元を足した配列を返す
    raise NotImplementedError


def ex5_softmax_topk(logits: np.ndarray, k: int = 5):
    """演習5: 1 次元 logits から softmax を取り、上位 k の (indices, probs) を返す。

    - softmax: 数値安定のため max を引いてから exp し、総和で割る。
    - 上位 k: 確率の降順に並べた最初の k 個の (クラスインデックス配列, 確率配列)。
    - 返り値はタプル (indices: np.ndarray[int], probs: np.ndarray[float])。
    例: logits=[1,3,2,0], k=2 -> indices=[1,2], probs≈[0.644,0.237]
    """
    # TODO: softmax → argsort 降順 → 上位 k の indices と probs を返す
    raise NotImplementedError


def ex6_topk_accuracy(logits: np.ndarray, gt_indices: np.ndarray, k: int = 1) -> float:
    """演習6: top-k accuracy = gt が上位 k 予測に入っているサンプルの割合。

    - logits: (N, C)、gt_indices: (N,)。各行で上位 k のクラスを取り、gt が含まれれば的中。
    - 的中数 / サンプル数 を float で返す。N=0 なら 0.0。
    - ヒント: np.argsort(-logits, axis=1)[:, :k] で各行の上位 k インデックス。
    例: logits=[[0.1,0.9,0.0],[0.3,0.1,0.6]], gt=[1,0], k=1 -> 0.5
    """
    # TODO: 各行の上位 k に gt が入る割合を返す
    raise NotImplementedError


def ex7_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """演習7: 混同行列 (num_classes, num_classes) を作る。行=正解, 列=予測。

    - cm[g, p] に「正解 g・予測 p」の件数を足し込む。dtype は int。
    例: gt=[0,1,1], pred=[0,1,0], C=2 -> [[1,0],[1,1]]
    """
    # TODO: ゼロ行列を作り、(g,p) ごとに +1 する
    raise NotImplementedError


def ex8_topk_agreement_vs_pseudo_gt(
    correct_logits: np.ndarray, variant_logits: np.ndarray, k: int = 1
) -> float:
    """演習8: 「正しい前処理の top-1」を pseudo-GT として、別前処理の top-k 一致率を返す。

    - pseudo_gt = correct_logits の各行 argmax（正しい前処理での予測ラベル）。
    - その pseudo_gt に対する variant_logits の top-k accuracy を返す（ex6 を使う）。
    - これは「前処理を変えたときに予測がどれだけ保たれるか（robustness）」の指標。
    例: correct=[[0,1],[1,0]] -> pseudo_gt=[1,0];
        variant=[[0,1],[0,1]] の top-1 一致率 -> 0.5
    """
    # TODO: pseudo_gt を作り、variant_logits の top-k accuracy を返す
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
            results.append((name, False, "未実装（TODO を埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    def _c1():
        a = ex1_uniform_indices(32, 8)
        b = ex1_uniform_indices(4, 8)
        ok = (list(a) == [0, 4, 9, 13, 18, 22, 27, 31]
              and list(b) == [0, 0, 1, 1, 2, 2, 3, 3]
              and np.asarray(a).dtype.kind == "i")
        return ok, f"{a.tolist()}"
    check("ex1_uniform_indices", _c1)

    def _c2():
        a = ex2_strided_indices(32, 8, 4)
        b = ex2_strided_indices(10, 8, 3, start=0)
        ok = (list(a) == [1, 5, 9, 13, 17, 21, 25, 29]
              and list(b) == [0, 3, 6, 9, 9, 9, 9, 9])
        return ok, f"center={a.tolist()}, clamp={b.tolist()}"
    check("ex2_strided_indices", _c2)

    def _c3():
        clip = np.full((2, 2, 3), 0.5, dtype=np.float64)
        out = ex3_normalize_clip(clip, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ok = out.shape == clip.shape and np.allclose(out, 0.0)
        # チャンネル別に効くか（mean を変えて確認）
        out2 = ex3_normalize_clip(np.ones((1, 1, 3)), (0.0, 1.0, 2.0), (1.0, 1.0, 1.0))
        ok = ok and np.allclose(out2.ravel(), [1.0, 0.0, -1.0])
        return ok, f"zeros_ok, per-ch={out2.ravel().tolist()}"
    check("ex3_normalize_clip", _c3)

    def _c4():
        clip = np.zeros((4, 8, 10, 3))
        clip[0, 0, 0, 1] = 7.0  # T=0,H=0,W=0,C=1 に印
        out = ex4_clip_to_ncthw(clip)
        ok = out.shape == (1, 3, 4, 8, 10) and out[0, 1, 0, 0, 0] == 7.0
        return ok, f"shape={out.shape}"
    check("ex4_clip_to_ncthw", _c4)

    def _c5():
        idx, probs = ex5_softmax_topk(np.array([1.0, 3.0, 2.0, 0.0]), k=2)
        ok = list(idx) == [1, 2] and abs(float(probs[0]) - 0.6439) < 1e-3 and probs[0] >= probs[1]
        return ok, f"idx={idx.tolist()}, probs={np.round(probs,3).tolist()}"
    check("ex5_softmax_topk", _c5)

    def _c6():
        logits = np.array([[0.1, 0.9, 0.0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        a = ex6_topk_accuracy(logits, np.array([1, 0, 1]), k=1)
        b = ex6_topk_accuracy(logits, np.array([1, 0, 1]), k=2)
        ok = abs(a - 2 / 3) < 1e-9 and abs(b - 1.0) < 1e-9
        return ok, f"top1={a:.3f}, top2={b:.3f}"
    check("ex6_topk_accuracy", _c6)

    def _c7():
        cm = ex7_confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]), 2)
        ok = cm.tolist() == [[1, 0], [1, 1]]
        return ok, f"{cm.tolist()}"
    check("ex7_confusion_matrix", _c7)

    def _c8():
        correct = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])  # pseudo_gt=[1,0,1]
        variant = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # pred=[1,1,1]
        v = ex8_topk_agreement_vs_pseudo_gt(correct, variant, k=1)
        ok = abs(v - 2 / 3) < 1e-9
        return ok, f"agreement={v:.3f}"
    check("ex8_topk_agreement_vs_pseudo_gt", _c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:32s} {detail}")
    print("\nALL PASS 🎉" if all_ok
          else "\nまだ未達の演習があります。TODO を埋めましょう（解答: exercises_solutions.py）。")


if __name__ == "__main__":
    _grade()
