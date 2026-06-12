"""第16回 演習問題（CLIP/SigLIP ゼロショットと画像テキスト検索）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ＝exit 0）。
  2. 自己採点:  uv run python lectures/16_clip_zeroshot_retrieval/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る（解答は別ファイル exercises_solutions.py に分離）:
        SHOW_SOLUTION=1 uv run python lectures/16_clip_zeroshot_retrieval/exercises.py
     もしくは直接:   uv run python lectures/16_clip_zeroshot_retrieval/exercises_solutions.py

この9問は本モジュール（3スクリプト + ミニプロジェクト）の核を1つずつ抜き出したもの。
モデルDLは不要・純計算なので一瞬で回る（易→難に並べてある）:
  ex1: L2 正規化（get_*_features は未正規化 → これが無いとコサインにならない） … 02/03
  ex2: コサイン類似度行列（正規化してから内積）                              … 03
  ex3: ゼロショット確率（softmax=相互排他 / sigmoid=独立）                    … 02
  ex4: 上位k の取り出し（torch.topk 相当）                                    … 03
  ex5: Recall@k（上位k に正解が入った割合）                                   … 03
  ex6: MRR の素（最初の正解の逆順位）                                         … 03/07
  ex7: Average Precision（クエリ1本の AP。平均すると mAP）                    … 03/07
  ex8: logit_scale（温度）の適用 — forward の logits を埋め込みから再構成     … 04
  ex9: 開集合の棄却（コサイン閾値で「該当なし=-1」を返す）                    … 05/ミニプロジェクト
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """演習1: 行ベクトルを L2 正規化する（各行のノルムを 1 にする）。

    - x: 形 (N, D) のテンソル。各行を ‖row‖_2 で割る。
    - ゼロ割を避けるため、分母に微小値 eps=1e-12 を足すか clamp_min する。
    - 返り値は x と同じ形。torch.nn.functional.normalize(x, p=2, dim=-1) と一致すること。
    """
    # TODO: 各行の L2 ノルムで割って返す（dim=-1, p=2）
    raise NotImplementedError


def ex2_cosine_sim_matrix(text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
    """演習2: テキスト埋め込みと画像埋め込みのコサイン類似度行列を返す。

    - text_emb: (Q, D), image_emb: (N, D)（どちらも未正規化で渡される）。
    - 正しいコサインにするには『両方を L2 正規化してから内積』を取る。
    - 返り値は (Q, N)。要素は -1〜1 に収まる。ex1 を使ってよい。
    """
    # TODO: 両方を正規化し、text_n @ image_n.T を返す
    raise NotImplementedError


def ex3_zeroshot_probs(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """演習3: 類似度ロジットから確率を作る。mode で softmax / sigmoid を切り替える。

    - logits: (n_images, n_labels)。
    - mode == "softmax": ラベル方向(dim=-1)に softmax。各行の合計は 1（相互排他＝CLIP流）。
    - mode == "sigmoid": 要素ごとに sigmoid。各値が独立に 0〜1（SigLIP流。合計は1にならない）。
    - それ以外の mode は ValueError を投げる。
    """
    # TODO: mode に応じて torch.softmax(logits, dim=-1) または torch.sigmoid(logits) を返す
    raise NotImplementedError


def ex4_topk_indices(scores: torch.Tensor, k: int) -> list[int]:
    """演習4: 1次元スコアの上位 k 個のインデックスを『スコア降順』で返す。

    - scores: 形 (N,) の1次元テンソル。k は 1〜N。
    - torch.topk(scores, k) を使うと値とインデックスが降順で得られる。
    - 返り値は Python の int のリスト（長さ k）。
    """
    # TODO: torch.topk で上位 k のインデックスを取り、.tolist() で返す
    raise NotImplementedError


def ex5_recall_at_k(ranking: list[int], relevant: set[int], k: int) -> float:
    """演習5: ランキング上位 k における Recall@k を返す。

    - ranking: 類似度降順に並べた候補インデックスのリスト（先頭が最も似ている）。
    - relevant: 正解インデックスの集合（1個以上）。
    - Recall@k = (上位k に入った正解の数) / (正解の総数)。
    - relevant が空なら 0.0 を返す（ゼロ割回避）。
    """
    # TODO: set(ranking[:k]) と relevant の積集合の大きさ / len(relevant) を返す
    raise NotImplementedError


def ex6_reciprocal_rank(ranking: list[int], relevant: set[int]) -> float:
    """演習6: ランキング中『最初の正解』の逆順位を返す（平均すると MRR）。

    - ranking: 類似度降順のインデックス列。先頭が 1 位（1-based）。
    - 最初に現れる relevant 要素の順位 r に対し 1/r を返す。
    - 1位で当たれば 1.0、3位なら 1/3。正解が1つも現れなければ 0.0。
    """
    # TODO: ranking を先頭から見て、最初に relevant に含まれる位置 i（0-based）で 1/(i+1) を返す
    raise NotImplementedError


def ex7_average_precision(ranking: list[int], relevant: set[int]) -> float:
    """演習7: クエリ1本の Average Precision（AP）を返す（平均すると mAP）。

    - ranking: 類似度降順のインデックス列。
    - AP = (各正解ヒット位置での precision@その位置 の総和) / (正解の総数)。
      例: ranking=[A,B,C,D]、relevant={A,C} なら
          A は1位 precision=1/1、C は3位 precision=2/3 → AP=(1.0+0.667)/2=0.8333。
    - relevant が空なら 0.0。ranking 中に正解が1つも無ければ 0.0。
    """
    # TODO: ranking を走査し hit 数を数えながら precision を積み、len(relevant) で割る
    raise NotImplementedError


def ex8_clip_logits_from_embeds(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale_raw: torch.Tensor
) -> torch.Tensor:
    """演習8: 埋め込みから CLIP forward の logits_per_image を再構成する（§4 の非対称性）。

    - image_emb: (Ni, D), text_emb: (Nt, D)。どちらも『未正規化』で渡される。
    - CLIP forward と一致させるには:
        (1) 画像・テキストとも L2 正規化する。
        (2) 正規化済み内積（コサイン）に温度 logit_scale.exp() を掛ける。
      → logits_per_image = (img_n @ txt_n.T) * exp(logit_scale_raw)。
    - logit_scale_raw は学習対象の生パラメータ（log 空間）。必ず .exp() してから掛ける。
    - 返り値は (Ni, Nt)。ex1/ex2 を使ってよい。
    """
    # TODO: 両者を正規化し、内積に logit_scale_raw.exp() を掛けて返す
    raise NotImplementedError


def ex9_abstain_or_pick(scores: torch.Tensor, threshold: float) -> int:
    """演習9: 開集合の棄却。最大スコアが閾値未満なら『該当なし』を表す -1 を返す。

    - scores: 形 (N,) の1次元コサインスコア（候補ごとの類似度）。
    - 最大スコア >= threshold なら、その最大スコアのインデックス（int）を返す。
    - 最大スコア < threshold なら -1（abstain＝該当なし）を返す。
    - softmax は必ずどれかを選んでしまうが、閾値つきコサインは『選ばない』を表現できる。
    """
    # TODO: scores.max() と argmax を見て、閾値未満なら -1 を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（採点ロジックの単一の置き場所。解答ファイルもこれを再利用する）
# =====================================================================

def _grade() -> bool:
    """全演習を採点して結果を表示し、全 PASS かどうかを返す。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 既知のベクトルを正規化 → 各行ノルム1、F.normalize と一致。
    def _c1():
        import torch.nn.functional as F
        x = torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 1.0]])
        got = ex1_l2_normalize(x)
        ref = F.normalize(x, p=2, dim=-1)
        norms_ok = torch.allclose(got.norm(dim=-1), torch.ones(3), atol=1e-5)
        return torch.allclose(got, ref, atol=1e-5) and norms_ok, "各行ノルム=1"
    check("ex1_l2_normalize", _c1)

    # ex2: コサイン行列の対角が最大（同じ向きのベクトルどうし）。
    def _c2():
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        im = torch.tensor([[2.0, 0.0], [0.0, 5.0]])   # 大きさは違うが向きは t と一致
        got = ex2_cosine_sim_matrix(t, im)
        ok_shape = tuple(got.shape) == (2, 2)
        diag_one = torch.allclose(torch.diag(got), torch.ones(2), atol=1e-5)
        off_zero = abs(float(got[0, 1])) < 1e-5 and abs(float(got[1, 0])) < 1e-5
        return ok_shape and diag_one and off_zero, "対角=1・非対角=0（大きさ非依存）"
    check("ex2_cosine_sim_matrix", _c2)

    # ex3: softmax は行合計1、sigmoid は独立（合計が1にならない例を作る）。
    def _c3():
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        sm = ex3_zeroshot_probs(logits, "softmax")
        sg = ex3_zeroshot_probs(logits, "sigmoid")
        sm_sum1 = torch.allclose(sm.sum(dim=-1), torch.ones(1), atol=1e-5)
        sg_indep = torch.allclose(sg, torch.sigmoid(logits), atol=1e-6)
        sg_not1 = not torch.allclose(sg.sum(dim=-1), torch.ones(1), atol=1e-3)
        try:
            ex3_zeroshot_probs(logits, "bogus")
            raised = False
        except ValueError:
            raised = True
        return sm_sum1 and sg_indep and sg_not1 and raised, "softmax合計1 / sigmoid独立 / 不正modeでValueError"
    check("ex3_zeroshot_probs", _c3)

    # ex4: 既知スコアの上位3。
    def _c4():
        scores = torch.tensor([0.1, 0.9, 0.5, 0.7, 0.2])
        got = ex4_topk_indices(scores, 3)
        return got == [1, 3, 2], f"top3={got}"
    check("ex4_topk_indices", _c4)

    # ex5: ranking と正解集合から Recall@k。
    def _c5():
        ranking = [4, 1, 7, 2, 9]
        relevant = {1, 2, 5}        # 5 は上位5に入らない
        r2 = ex5_recall_at_k(ranking, relevant, 2)   # 上位2: {4,1} ∩ {1,2,5} = {1} → 1/3
        r5 = ex5_recall_at_k(ranking, relevant, 5)   # 上位5: {1,2} → 2/3
        empty = ex5_recall_at_k(ranking, set(), 5)   # 正解なし → 0.0
        return abs(r2 - 1 / 3) < 1e-6 and abs(r5 - 2 / 3) < 1e-6 and empty == 0.0, f"R@2={r2:.3f}, R@5={r5:.3f}"
    check("ex5_recall_at_k", _c5)

    # ex6: 最初の正解の逆順位。
    def _c6():
        rr_first = ex6_reciprocal_rank([4, 1, 7, 2, 9], {4})       # 1位 → 1.0
        rr_third = ex6_reciprocal_rank([4, 1, 7, 2, 9], {7, 9})    # 最初の正解=7 が3位 → 1/3
        rr_none = ex6_reciprocal_rank([4, 1, 7], {99})             # 不在 → 0.0
        return (abs(rr_first - 1.0) < 1e-6 and abs(rr_third - 1 / 3) < 1e-6 and rr_none == 0.0,
                f"RR(1st)={rr_first:.3f}, RR(3rd)={rr_third:.3f}")
    check("ex6_reciprocal_rank", _c6)

    # ex7: Average Precision。ranking=[0,1,2,3], relevant={0,2} → (1/1 + 2/3)/2 = 0.8333。
    def _c7():
        ap = ex7_average_precision([0, 1, 2, 3], {0, 2})
        ap_none = ex7_average_precision([0, 1, 2], {9})            # 不在 → 0.0
        ap_empty = ex7_average_precision([0, 1, 2], set())         # 正解なし → 0.0
        return (abs(ap - (1.0 + 2 / 3) / 2) < 1e-6 and ap_none == 0.0 and ap_empty == 0.0,
                f"AP={ap:.4f}")
    check("ex7_average_precision", _c7)

    # ex8: 埋め込みから forward の logits を再構成（正規化＋温度）。
    def _c8():
        import torch.nn.functional as F
        torch.manual_seed(0)
        img = torch.randn(2, 8) * 3.0      # 未正規化（ノルムはバラバラ）
        txt = torch.randn(3, 8) * 0.5
        logit_scale_raw = torch.tensor(4.6052)   # exp ≈ 100（CLIP の温度に近い）
        got = ex8_clip_logits_from_embeds(img, txt, logit_scale_raw)
        ref = (F.normalize(img, p=2, dim=-1) @ F.normalize(txt, p=2, dim=-1).t()) * logit_scale_raw.exp()
        return tuple(got.shape) == (2, 3) and torch.allclose(got, ref, atol=1e-4), "logits=(img_n@txt_n.T)*exp(scale)"
    check("ex8_clip_logits_from_embeds", _c8)

    # ex9: コサイン閾値で棄却。
    def _c9():
        scores = torch.tensor([0.10, 0.35, 0.20])
        pick = ex9_abstain_or_pick(scores, 0.30)    # max=0.35>=0.30 → argmax=1
        absta = ex9_abstain_or_pick(scores, 0.50)   # max=0.35<0.50 → -1（棄却）
        return pick == 1 and absta == -1, f"pick={pick}, abstain={absta}"
    check("ex9_abstain_or_pick", _c9)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:26s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def _install_solutions() -> None:
    """模範解答を本体へ差し替える（解答は exercises_solutions.py に集約・重複なし）。"""
    import exercises_solutions as S

    S.install(sys.modules[__name__])


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します — 解答は exercises_solutions.py)\n")
        _install_solutions()
    _grade()
