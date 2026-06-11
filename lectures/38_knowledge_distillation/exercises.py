"""第38回 演習問題（知識蒸留 / 温度付きKD・特徴量蒸留・DeiT・RKD）。

使い方:
  1. 各 exN_*() の中の TODO を自分で実装する（最初は NotImplementedError で FAIL）。
  2. 自己採点（未実装でも例外で落とさず PASS/FAIL を表示し、必ず exit 0）:
         uv run python lectures/38_knowledge_distillation/exercises.py
  3. 全問の模範解答（全 PASS）はこちら:
         uv run python lectures/38_knowledge_distillation/exercises_solutions.py

テーマ（易→難）:
  1 温度ソフトターゲット   2 温度付きKL(T^2)     3 合成損失(alpha結合)
  4 圧縮率                 5 ソフトのエントロピー 6 特徴量蒸留(射影+正規化+MSE)
  7 DeiT 2ヘッド平均       8 関係蒸留(RKD 距離行列)

採点は決定的な合成テンソルだけで行うので、モデル DL も学習も不要で速く再現的。
実モデルへの適用は 01〜05・mini_project.py を参照。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =====================================================================
# 演習（ここを実装する）
# =====================================================================

def ex1_soften_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """演習1: 温度 T でロジットを軟化した確率分布 softmax(logits / T) を返す（dim=1）。"""
    # TODO: softmax(logits / temperature) を dim=1 で返す
    raise NotImplementedError


def ex2_kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                   temperature: float) -> torch.Tensor:
    """演習2: 温度付き KL（Hinton 蒸留のソフト損失）をスカラーで返す。

      soft = F.kl_div(log_softmax(student/T, 1), softmax(teacher/T, 1),
                      reduction='batchmean') * T**2
    入力は log_softmax(student)・ターゲットは softmax(teacher)・reduction='batchmean'・
    最後に T**2 を掛けること（順序/log/reduction/T^2 のどれを欠いても不正解）。
    """
    # TODO: 正準形の温度付き KL を実装して返す
    raise NotImplementedError


def ex3_combined_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         targets: torch.Tensor, temperature: float,
                         alpha: float) -> torch.Tensor:
    """演習3: 合成損失 = alpha*ソフト(温度付きKL) + (1-alpha)*ハード(F.cross_entropy) を返す。

    ソフト項は ex2 と同じ式。ハード項は F.cross_entropy(student_logits, targets)。
    """
    # TODO: alpha でソフトとハードを線形結合して返す
    raise NotImplementedError


def ex4_compression_ratio(teacher_params: int, student_params: int) -> float:
    """演習4: 圧縮率 = teacher のパラメータ数 / student のパラメータ数 を返す。

    student_params が 0 のときは 0.0 を返す（ゼロ割回避）。
    """
    # TODO: teacher_params / student_params を返す（student_params==0 は 0.0）
    raise NotImplementedError


def ex5_soft_target_entropy(logits: torch.Tensor, temperature: float) -> float:
    """演習5: softmax(logits/T) の平均エントロピー(nats) を float で返す。

      p = softmax(logits/T);  entropy = -sum(p*log(p), dim=1);  return mean(entropy)
    log(0) を避けるため p に微小値(1e-12)を足してから log を取ること。
    T を上げるほどこの値は大きくなる（分布がなだらかになる）。
    """
    # TODO: 温度ソフトターゲットの平均エントロピーを返す
    raise NotImplementedError


def ex6_feature_distill_mse(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                            proj_weight: torch.Tensor) -> torch.Tensor:
    """演習6: 特徴量蒸留の MSE を返す（射影 → 正規化 → MSE）。

    student_feat:(N,Cs) / teacher_feat:(N,Ct) / proj_weight:(Ct,Cs)。
      proj = student_feat @ proj_weight.T            # (N,Ct) に次元を合わせる
      s = F.normalize(proj, dim=1); t = F.normalize(teacher_feat, dim=1)
      return F.mse_loss(s, t)
    射影層が無いと次元が合わず MSE が取れない／正規化でスケール差を吸収する、が肝。
    """
    # TODO: 射影 → L2正規化 → MSE を実装して返す
    raise NotImplementedError


def ex7_deit_inference_logits(cls_logits: torch.Tensor,
                              dist_logits: torch.Tensor) -> torch.Tensor:
    """演習7: DeiT 推論時の出力 = class ヘッドと distill ヘッドの平均ロジットを返す。"""
    # TODO: (cls_logits + dist_logits) / 2 を返す
    raise NotImplementedError


def ex8_rkd_distance_matrix(emb: torch.Tensor) -> torch.Tensor:
    """演習8: 関係蒸留(RKD) の正規化距離行列を返す。

    emb:(N,D)。
      dist = torch.cdist(emb, emb)            # (N,N) のペア間ユークリッド距離
      mu = dist[dist > 0].mean()              # 0(対角)を除いた平均で正規化
      return dist / mu
    対角は 0、行列は対称になる。RKD は「個々の特徴」ではなく「この距離関係」を student に写す。
    """
    # TODO: cdist → 正の要素平均で正規化した距離行列を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも import して再利用）
# =====================================================================

def grade(funcs: dict) -> bool:
    """funcs（名前→関数）を採点して PASS/FAIL を表示。全PASSなら True。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    g = torch.Generator().manual_seed(0)

    def c1():
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        p = funcs["ex1"](logits, 1.0)
        ref = F.softmax(logits, dim=1)
        soft = funcs["ex1"](logits, 4.0)
        # T=1 は素の softmax、T を上げると最大確率が下がる（なだらか）
        return (torch.allclose(p, ref, atol=1e-6) and float(soft.max()) < float(ref.max()),
                f"max p (T=1)={float(ref.max()):.3f} -> (T=4)={float(soft.max()):.3f}")

    def c2():
        s = torch.randn(32, 6, generator=g)
        t = torch.randn(32, 6, generator=g) * 2.0
        T = 4.0
        got = funcs["ex2"](s, t, T)
        ref = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                       reduction="batchmean") * T ** 2
        # T^2 補正の確認: T=2 のとき素の KL の 4 倍になっている
        raw2 = F.kl_div(F.log_softmax(s / 2, 1), F.softmax(t / 2, 1), reduction="batchmean")
        got2 = funcs["ex2"](s, t, 2.0)
        return (abs(float(got) - float(ref)) < 1e-5 and abs(float(got2) - 4 * float(raw2)) < 1e-5,
                f"kl(T=4)={float(got):.4f} (ref {float(ref):.4f})")

    def c3():
        s = torch.randn(16, 6, generator=g)
        t = torch.randn(16, 6, generator=g) * 2.0
        y = torch.randint(0, 6, (16,), generator=g)
        T = 4.0
        soft = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                        reduction="batchmean") * T ** 2
        hard = F.cross_entropy(s, y)
        for a in (0.0, 0.5, 1.0):
            ref = a * soft + (1 - a) * hard
            got = funcs["ex3"](s, t, y, T, a)
            if abs(float(got) - float(ref)) > 1e-5:
                return (False, f"alpha={a}: got {float(got):.4f} != {float(ref):.4f}")
        return (True, "alpha=0/0.5/1 すべて一致")

    def c4():
        return (abs(funcs["ex4"](23910, 1398) - 23910 / 1398) < 1e-6
                and funcs["ex4"](100, 0) == 0.0,
                f"ratio={funcs['ex4'](23910, 1398):.2f}, zero-div=0.0")

    def c5():
        logits = torch.randn(128, 6, generator=g) * 3.0
        e1 = funcs["ex5"](logits, 1.0)
        e8 = funcs["ex5"](logits, 8.0)
        ref1 = float(-(F.softmax(logits, 1) * (F.softmax(logits, 1) + 1e-12).log()).sum(1).mean())
        # T を上げるとエントロピーは増える（なだらか）
        return (abs(e1 - ref1) < 1e-4 and e8 > e1,
                f"H(T=1)={e1:.3f} < H(T=8)={e8:.3f}")

    def c6():
        sfeat = torch.randn(8, 16, generator=g)
        w = torch.randn(64, 16, generator=g)
        tfeat = sfeat @ w.T                       # 射影後にぴったり一致するよう作る
        zero = funcs["ex6"](sfeat, tfeat, w)      # 正規化後 MSE ≈ 0
        other = torch.randn(8, 64, generator=g)
        pos = funcs["ex6"](sfeat, other, w)       # 一致しない → MSE > 0
        return (float(zero) < 1e-6 and float(pos) > 1e-3,
                f"match MSE={float(zero):.2e}, mismatch MSE={float(pos):.4f}")

    def c7():
        a = torch.randn(4, 6, generator=g)
        b = torch.randn(4, 6, generator=g)
        got = funcs["ex7"](a, b)
        return (torch.allclose(got, (a + b) / 2, atol=1e-6), "average of two heads")

    def c8():
        emb = torch.randn(5, 8, generator=g)
        m = funcs["ex8"](emb)
        diag0 = float(m.diag().abs().max()) < 1e-6
        symm = torch.allclose(m, m.T, atol=1e-5)
        mean1 = abs(float(m[m > 0].mean()) - 1.0) < 1e-5   # 正の要素平均が 1 に正規化
        return (diag0 and symm and mean1,
                f"diag0={diag0} symmetric={symm} mean(pos)={float(m[m > 0].mean()):.3f}")

    check("ex1_soften_logits", c1)
    check("ex2_kd_kl_loss", c2)
    check("ex3_combined_kd_loss", c3)
    check("ex4_compression_ratio", c4)
    check("ex5_soft_target_entropy", c5)
    check("ex6_feature_distill_mse", c6)
    check("ex7_deit_inference_logits", c7)
    check("ex8_rkd_distance_matrix", c8)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


def current_funcs() -> dict:
    """このファイルで定義された ex* を採点ランナーへ渡す形にまとめる。"""
    return {
        "ex1": ex1_soften_logits,
        "ex2": ex2_kd_kl_loss,
        "ex3": ex3_combined_kd_loss,
        "ex4": ex4_compression_ratio,
        "ex5": ex5_soft_target_entropy,
        "ex6": ex6_feature_distill_mse,
        "ex7": ex7_deit_inference_logits,
        "ex8": ex8_rkd_distance_matrix,
    }


if __name__ == "__main__":
    grade(current_funcs())
