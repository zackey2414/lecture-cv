"""exercises.py — 30_face_detection_recognition 演習（10 問・易→難・自己採点つき）

各 TODO 関数を実装して保存し、実行すると自動採点されます:

    uv run python lectures/30_face_detection_recognition/exercises.py

- 未実装でも **exit 0** で終了します（PASS/FAIL の一覧が出るだけ）。
- 模範解答は exercises_solutions.py（実行すると全 PASS）。
- ねらい: L2 正規化 → コサイン類似/距離 → 照合ペア → FAR/TAR → EER → TAR@FAR →
  rank-1 識別 → purity → 検出の IoU まで、顔の検出・認識・クラスタリング評価の核を
  ライブラリに頼らず自分の手で書けるようにする（numpy のみ使用）。
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 問1（易）: ベクトルの L2 正規化
#   v を L2 ノルム 1 に正規化して返す。ゼロベクトルはそのまま返す（0 除算回避）。
# ============================================================================
def l2_normalize_vec(v):
    # TODO: norm = sqrt(sum(v^2))。norm が 0（に近い）なら v をそのまま、そうでなければ v/norm。
    raise NotImplementedError


# ============================================================================
# 問2（易）: コサイン類似度
#   a, b のコサイン類似度（内積 / (|a||b|)）を返す。
# ============================================================================
def cosine_similarity(a, b) -> float:
    # TODO: それぞれを L2 正規化してから内積をとると楽（問1 を再利用してよい）。
    raise NotImplementedError


# ============================================================================
# 問3（易）: コサイン距離
#   コサイン距離 = 1 - コサイン類似度。
# ============================================================================
def cosine_distance(a, b) -> float:
    # TODO: 問2 を使って 1 - cos を返す。
    raise NotImplementedError


# ============================================================================
# 問4（中）: 照合ペアの genuine/impostor ラベル付け
#   labels（人物 ID 配列）から、全ペア i<j を作り (i, j, is_genuine) のリストを返す。
#   is_genuine は labels[i]==labels[j]（同一人物なら True）。
# ============================================================================
def make_pair_labels(labels):
    # TODO: 二重ループ i<j で (i, j, bool(labels[i]==labels[j])) を集める。
    raise NotImplementedError


# ============================================================================
# 問5（中）: しきい値での FAR / TAR
#   scores（各ペアのコサイン類似度）, is_genuine（同一人物か）, thr を受け取り、
#   score>=thr を「同一人物」と判定したときの (far, tar) を返す。
#     TAR = genuine のうち accept した割合, FAR = impostor のうち accept した割合。
#   genuine / impostor が 0 件のときは、その率を 0.0 とする。
# ============================================================================
def far_tar_at_threshold(scores, is_genuine, thr):
    # TODO: scores>=thr を accept とし、genuine 側・impostor 側で平均をとる。
    raise NotImplementedError


# ============================================================================
# 問6（中）: ROC 曲線から EER
#   far, tar（同じ長さの配列。各しきい値に対応）から EER を返す。
#   FRR = 1 - TAR。|FAR - FRR| が最小の点で eer = (FAR + FRR)/2。
# ============================================================================
def eer_from_curves(far, tar) -> float:
    # TODO: frr を作り、argmin(|far-frr|) の位置で (far+frr)/2 を返す。
    raise NotImplementedError


# ============================================================================
# 問7（中）: TAR@FAR
#   far, tar とターゲット FAR を受け取り、FAR<=target を満たす中で最大の TAR を返す。
#   満たす点が無ければ 0.0。
# ============================================================================
def tar_at_far(far, tar, target) -> float:
    # TODO: mask = far<=target。どれも満たさなければ 0.0、満たせば tar[mask] の最大。
    raise NotImplementedError


# ============================================================================
# 問8（中）: rank-1 識別精度
#   sim（probe×gallery のコサイン類似度行列）, gallery_labels, probe_labels から、
#   各プローブの「最も似たギャラリー」の人物が本人と一致した割合を返す。
# ============================================================================
def rank1_accuracy(sim, gallery_labels, probe_labels) -> float:
    # TODO: 各行で argmax を取り、その gallery_labels を予測ラベルとして probe_labels と比較。
    raise NotImplementedError


# ============================================================================
# 問9（難）: purity（クラスタの純度）
#   labels_true, labels_pred から purity を返す。
#   各予測クラスタを「中の多数派の真ラベル」に割り当てたときの全体正解率。
# ============================================================================
def purity(labels_true, labels_pred) -> float:
    # TODO: 予測クラスタごとに、含まれる真ラベルの最頻値の個数を足し、全体数で割る。
    raise NotImplementedError


# ============================================================================
# 問10（難）: 検出と正解の IoU（[x,y,w,h] 形式）
#   2 つの矩形 [x,y,w,h] の IoU を返す。重ならなければ 0。検出評価の土台。
# ============================================================================
def iou_xywh(a, b) -> float:
    # TODO: 各矩形を [x1,y1,x2,y2] に直し、交差幅/高さを 0 でクリップ。union=areaA+areaB-inter。
    raise NotImplementedError


# ============================================================================
# 自己採点ハーネス（編集不要）。未実装は FAIL だが exit 0 で終了する。
# ============================================================================
def _approx(got, exp, tol=1e-3):
    try:
        return bool(np.all(np.abs(np.asarray(got, float) - np.asarray(exp, float)) <= tol))
    except Exception:
        return False


def _run(name, fn):
    try:
        ok = fn()
    except NotImplementedError:
        print(f"  [TODO] {name}: 未実装")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: 例外 {type(e).__name__}: {e}")
        return False
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return bool(ok)


def _checks():
    return [
        (
            "問1 l2_normalize_vec",
            lambda: (
                _approx(l2_normalize_vec([3.0, 4.0]), [0.6, 0.8])
                and _approx(l2_normalize_vec([0.0, 0.0]), [0.0, 0.0])
            ),
        ),
        (
            "問2 cosine_similarity",
            lambda: (
                _approx(cosine_similarity([1, 0], [1, 1]), 0.70710678)
                and _approx(cosine_similarity([1, 0], [0, 1]), 0.0)
                and _approx(cosine_similarity([1, 0], [-1, 0]), -1.0)
            ),
        ),
        ("問3 cosine_distance", lambda: _approx(cosine_distance([1, 0], [1, 1]), 0.29289322)),
        ("問4 make_pair_labels", lambda: _pairs_ok()),
        ("問5 far_tar_at_threshold", lambda: _far_tar_ok()),
        (
            "問6 eer_from_curves",
            lambda: _approx(eer_from_curves([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]), 0.5),
        ),
        (
            "問7 tar_at_far",
            lambda: (
                _approx(tar_at_far([0.0, 0.01, 0.1, 1.0], [0.5, 0.8, 0.95, 1.0], 0.01), 0.8)
                and _approx(tar_at_far([0.5, 1.0], [0.9, 1.0], 0.01), 0.0)
            ),
        ),
        (
            "問8 rank1_accuracy",
            lambda: (
                _approx(rank1_accuracy([[0.9, 0.1], [0.2, 0.8]], [0, 1], [0, 1]), 1.0)
                and _approx(rank1_accuracy([[0.1, 0.9]], [0, 1], [0]), 0.0)
            ),
        ),
        ("問9 purity", lambda: _approx(purity([0, 0, 1, 1, 2], [0, 0, 0, 1, 1]), 0.6)),
        (
            "問10 iou_xywh",
            lambda: (
                _approx(iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]), 0.142857)
                and _approx(iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
            ),
        ),
    ]


def _pairs_ok():
    out = make_pair_labels([0, 0, 1])
    norm = sorted((int(i), int(j), bool(g)) for i, j, g in out)
    return norm == [(0, 1, True), (0, 2, False), (1, 2, False)]


def _far_tar_ok():
    s = [0.9, 0.8, 0.4, 0.3]
    g = [True, True, False, False]
    return _approx(far_tar_at_threshold(s, g, 0.5), [0.0, 1.0]) and _approx(
        far_tar_at_threshold(s, g, 0.35), [0.5, 1.0]
    )


def main():
    print("=" * 60)
    print("30_face_detection_recognition 演習 自己採点")
    checks = _checks()
    passed = sum(_run(n, f) for n, f in checks)
    print("-" * 60)
    print(f"合計: {passed}/{len(checks)} PASS")
    if passed == len(checks):
        print("全問正解！ 顔の検出・認識・クラスタリング評価の核を自力で書けています。")


if __name__ == "__main__":
    main()
