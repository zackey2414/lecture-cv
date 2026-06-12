"""第18回 演習問題（物体検出 入門）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ → exit 0）。
  2. 自己採点:  uv run python lectures/18_object_detection_intro/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/18_object_detection_intro/exercises.py
     模範解答だけ走らせたいときは exercises_solutions.py を実行する。

全10問。易→難で、本モジュールの核（後処理）から検出評価（mAP の前段）まで一気通貫:
  ex1 : IoU（2つの box の交差/和集合）             … 検出評価とNMSの土台
  ex2 : xyxy → cxcywh への変換                      … box_convert の中身
  ex3 : score 閾値フィルタ                          … 生検出の間引き
  ex4 : 1クラスNMS（confidence降順の貪欲抑制）      … torchvision.ops.nms の中身
  ex5 : target_sizes=(H,W) で正規化cxcywh→xyxy絶対  … DETR後処理の核・典型バグ回避
  ex6 : クラス別NMS（batched_nms）                  … 隣接別クラスを潰さない後処理
  ex7 : 検出のマッチング（TP/FP 判定）             … ★mAP の心臓部（貪欲対応付け）
  ex8 : PR 曲線（累積 TP/FP → precision/recall）    … AP を出す直前の表
  ex9 : AP（全点補間 / all-point）                  … PR 曲線の下面積（PASCAL VOC2010+）
  ex10: AP（11点補間 / PASCAL VOC2007）             … 補間方式で値が変わることを体感

ex7〜ex10 を通すと、IoUマッチング→PR曲線→AP という mAP の全手順を自力で再現したこと
になる（複数画像・複数クラスへ拡張して平均すれば mAP。numpy 実装の総仕上げは第19回）。
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

def ex1_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """演習1: 2つの box（xyxy）の IoU = 交差面積 / 和集合面積 を返す。

    - box_a, box_b: 形 (4,) の [x1,y1,x2,y2]（x2>x1, y2>y1）。
    - 交差の幅・高さは負になりうるので 0 でクランプする（重なりなしは IoU=0）。
    - 和集合 = areaA + areaB - 交差。ゼロ割は 0.0 を返す。
    - 返り値は float（torchvision.ops.box_iou と一致すること）。
    """
    # TODO: 交差矩形 [max(x1),max(y1),min(x2),min(y2)] の面積を出し、IoU を返す
    raise NotImplementedError


def ex2_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """演習2: xyxy 形式の box を cxcywh（中心x,中心y,幅,高さ）に変換する。

    - boxes: 形 (N,4) の [x1,y1,x2,y2]。
    - cx=(x1+x2)/2, cy=(y1+y2)/2, w=x2-x1, h=y2-y1。
    - 返り値は同じ形 (N,4)。torchvision.ops.box_convert(boxes,'xyxy','cxcywh') と一致。
    """
    # TODO: cx,cy,w,h を計算して torch.stack(..., dim=-1) で返す
    raise NotImplementedError


def ex3_score_filter(
    boxes: torch.Tensor, scores: torch.Tensor, thresh: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """演習3: score >= thresh の検出だけを残す（生検出の間引き）。

    - boxes: (N,4), scores: (N,)。
    - 返り値は (残った boxes, 残った scores)。順序は元のまま。
    - 1件も残らなければ shape (0,4) と (0,) を返す（ブールマスクで自然にそうなる）。
    """
    # TODO: mask = scores >= thresh を作り、boxes[mask], scores[mask] を返す
    raise NotImplementedError


def ex4_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> list[int]:
    """演習4: 1クラス NMS。confidence 降順に貪欲に箱を選び、重なる箱を抑制する。

    手順（torchvision.ops.nms と同じ挙動）:
      1. scores 降順に index を並べる。
      2. 先頭（最高スコア）を keep に追加。
      3. それと IoU > iou_thresh の残り候補を捨てる。
      4. 残りについて 2〜3 を繰り返す。
    - 返り値は keep した index の list（スコア降順）。ex1 の IoU を使ってよい。
    """
    # TODO: argsort(降順)→先頭採用→IoU>閾値を除外、を繰り返して keep を返す
    raise NotImplementedError


def ex5_cxcywh_norm_to_xyxy_abs(
    boxes_norm: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """演習5: 正規化 cxcywh（0〜1）を、画像サイズに合わせた xyxy 絶対座標に変換する。

    DETR の後処理の核。target_sizes=(H,W) の (H,W) 順を取り違えると box が歪む。
    - boxes_norm: (N,4) の [cx,cy,w,h]（いずれも 0〜1 の正規化値）。
    - cx,w は width 倍、cy,h は height 倍してスケール。
    - そのあと x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2 で xyxy 絶対座標へ。
    - 返り値は (N,4) の xyxy。★height と width を取り違えないこと。
    """
    # TODO: cx,cy,w,h を width/height でスケールしてから xyxy にして返す
    raise NotImplementedError


def ex6_batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float
) -> list[int]:
    """演習6: クラス別 NMS（torchvision.ops.batched_nms 相当）。

    全クラスまとめて NMS を掛けると、たまたま重なった「人」と「車」の箱が誤って
    一方に潰される。これを避けるため、クラス（labels）ごとに独立に NMS する。
    - boxes:(N,4) xyxy, scores:(N,), labels:(N,) クラスID（整数）。
    - 各クラスの部分集合に ex4 の 1クラス NMS を掛け、残った index を全クラス分まとめる。
    - 返り値は keep した index の list（順序は問わない。採点は集合で比較）。
    ヒント: labels.unique() で回し、各クラスの index を boolマスクで取り出す。
    """
    # TODO: クラスごとに ex4_nms を掛け、元index へ写し戻して結合する
    raise NotImplementedError


def ex7_match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """演習7: ★mAP の心臓部 — 予測を GT に貪欲対応付けして TP/FP を決める（単一クラス）。

    手順（COCO/PASCAL 共通の評価前処理）:
      1. 予測を score 降順に並べる。
      2. 各予測を順に見て、まだマッチしていない GT のうち IoU が最大のものを選ぶ。
      3. その IoU が iou_thresh 以上なら TP（その GT を「使用済み」にする）、未満なら FP。
      4. 1つの GT は1つの予測にしか対応しない（2つ目以降の重複予測は FP になる）。
    - pred_boxes:(N,4), pred_scores:(N,), gt_boxes:(M,4)（すべて xyxy）。
    - 返り値 (tp, fp): どちらも形 (N,) の 0/1 テンソル。★score 降順に並べた順序で返す。
      未検出 GT（FN）は GT 数 M から TP の総数を引けば分かるのでここでは返さない。
    ヒント: 既に解いた IoU を使ってよい。matched 集合で使用済み GT を管理する。
    """
    # TODO: score 降順に走査し、未マッチGTへ貪欲対応 → tp/fp(0/1) を降順で返す
    raise NotImplementedError


def ex8_pr_curve(
    tp: torch.Tensor, fp: torch.Tensor, n_gt: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """演習8: score 降順の TP/FP 列から、累積 precision / recall を作る。

    - tp, fp: 形 (N,) の 0/1（ex7 の出力。score 降順に並んでいる前提）。
    - TP_cum = cumsum(tp), FP_cum = cumsum(fp)。
    - recall    = TP_cum / n_gt（GT 総数で割る。検出順に単調増加）。
    - precision = TP_cum / (TP_cum + FP_cum)（0割は微小値クランプで回避）。
    - 返り値 (precision, recall): どちらも形 (N,) のテンソル。
    """
    # TODO: cumsum で累積し、recall=TP/n_gt, precision=TP/(TP+FP) を返す
    raise NotImplementedError


def ex9_ap_all_points(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """演習9: 全点補間（all-point, PASCAL VOC2010 以降 / COCO の考え方）で AP を出す。

    PR 曲線の下面積を、precision を右側から単調非増加に補正してから台形和で求める。
    手順:
      1. mrec = [0] ++ recall ++ [1],  mpre = [0] ++ precision ++ [0] と番兵を付ける。
      2. mpre を後ろから前へ累積 max で単調非増加にする（envelope 化）。
      3. recall が変化する位置 i だけ取り、Σ (mrec[i+1]-mrec[i]) * mpre[i+1]。
    - precision, recall: 形 (N,) のテンソル（recall は単調増加）。
    - 返り値は float の AP。numpy を使ってよい。
    """
    # TODO: 番兵→envelope→recall変化点で面積を積む
    raise NotImplementedError


def ex10_ap_11_point(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """演習10: 11点補間（PASCAL VOC2007）で AP を出す。

    recall 閾値 t ∈ {0.0, 0.1, ..., 1.0} の11点で、
      p_interp(t) = max{ precision[k] | recall[k] >= t }（無ければ 0）
    を求め、その平均（11個の平均）を AP とする。
    - 全点補間（ex9）とは値が一致しないのが普通（補間方式で mAP は変わる）。
    - 返り値は float の AP。numpy を使ってよい。
    """
    # TODO: t=0..1 を 0.1 刻みで回し、recall>=t の precision 最大の平均を返す
    raise NotImplementedError


# =====================================================================
# 自己採点ランナー（exercises_solutions.py からも再利用する：採点ロジックは一箇所だけ）
# =====================================================================

def _grade() -> bool:
    """全演習を採点し、結果を表示する。全問 PASS なら True を返す。"""
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
            results.append((name, bool(ok), detail))
        except NotImplementedError:
            results.append((name, False, "未実装（TODOを埋めてください）"))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"例外: {type(e).__name__}: {e}"))

    # ex1: 既知の重なりを torchvision.ops.box_iou と突き合わせ。
    def _c1():
        from torchvision.ops import box_iou
        a = torch.tensor([0.0, 0.0, 10.0, 10.0])
        b = torch.tensor([5.0, 5.0, 15.0, 15.0])  # 交差 5x5=25, 和 100+100-25=175
        got = ex1_iou(a, b)
        ref = float(box_iou(a[None], b[None])[0, 0])
        disjoint = ex1_iou(a, torch.tensor([20.0, 20.0, 30.0, 30.0]))  # 重なりなし=0
        return abs(got - ref) < 1e-6 and abs(got - 25 / 175) < 1e-6 and disjoint == 0.0, \
            f"IoU={got:.4f} (ref {ref:.4f})"
    check("ex1_iou", _c1)

    # ex2: box_convert と一致。
    def _c2():
        from torchvision.ops import box_convert
        boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 4.0, 6.0]])
        got = ex2_xyxy_to_cxcywh(boxes)
        ref = box_convert(boxes, "xyxy", "cxcywh")
        return tuple(got.shape) == (2, 4) and torch.allclose(got, ref, atol=1e-5), \
            f"cxcywh[0]={got[0].tolist()}"
    check("ex2_xyxy_to_cxcywh", _c2)

    # ex3: 閾値で正しく間引く。
    def _c3():
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 2., 2.], [0., 0., 3., 3.]])
        scores = torch.tensor([0.9, 0.2, 0.6])
        b, s = ex3_score_filter(boxes, scores, 0.5)
        empty_b, empty_s = ex3_score_filter(boxes, scores, 0.99)
        return b.shape[0] == 2 and torch.allclose(s, torch.tensor([0.9, 0.6])) \
            and empty_b.shape == (0, 4) and empty_s.shape == (0,), f"kept={s.tolist()}"
    check("ex3_score_filter", _c3)

    # ex4: NMS を torchvision.ops.nms と突き合わせ。
    def _c4():
        from torchvision.ops import nms
        boxes = torch.tensor([
            [0., 0., 10., 10.],     # A: 高スコア
            [1., 1., 11., 11.],     # B: A とほぼ重なる（抑制される）
            [50., 50., 60., 60.],   # C: 離れている（残る）
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        got = ex4_nms(boxes, scores, 0.5)
        ref = nms(boxes, scores, 0.5).tolist()
        return got == ref and set(got) == {0, 2}, f"keep={got} (ref {ref})"
    check("ex4_nms", _c4)

    # ex5: 正規化 cxcywh → xyxy 絶対。(H,W) の取り違えがないか。
    def _c5():
        # 中心(0.5,0.5)・幅0.5・高さ0.25、画像 200x100(W=200,H=100)
        boxes_norm = torch.tensor([[0.5, 0.5, 0.5, 0.25]])
        got = ex5_cxcywh_norm_to_xyxy_abs(boxes_norm, height=100, width=200)
        # cx=100,cy=50,w=100,h=25 → x1=50,y1=37.5,x2=150,y2=62.5
        ref = torch.tensor([[50.0, 37.5, 150.0, 62.5]])
        return torch.allclose(got, ref, atol=1e-4), f"xyxy={got[0].tolist()}"
    check("ex5_cxcywh_norm_to_xyxy_abs", _c5)

    # ex6: クラス別 NMS を torchvision.ops.batched_nms と突き合わせ（集合で比較）。
    def _c6():
        from torchvision.ops import batched_nms
        # box0 と box1 は重なるが別クラス → 両方残るべき（全クラス一括NMSなら誤って消える）。
        # box2 は box0 と同クラスかつ重なる＆低スコア → 抑制される。
        boxes = torch.tensor([
            [0., 0., 10., 10.],     # 0: class 1, score 0.9 -> 残る
            [1., 1., 11., 11.],     # 1: class 2, score 0.8 -> 別クラスなので残る
            [0.5, 0.5, 10.5, 10.5],  # 2: class 1, score 0.7 -> 0 と重なり抑制
            [50., 50., 60., 60.],   # 3: class 1, score 0.6 -> 離れて残る
        ])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        labels = torch.tensor([1, 2, 1, 1])
        got = ex6_batched_nms(boxes, scores, labels, 0.5)
        ref = batched_nms(boxes, scores, labels, 0.5).tolist()
        return set(got) == set(ref) and set(got) == {0, 1, 3}, f"keep={sorted(got)} (ref {sorted(ref)})"
    check("ex6_batched_nms", _c6)

    # ex7: 貪欲マッチングで TP/FP を判定（単一クラス）。
    def _c7():
        # GT2つ。予測4つ（score 降順で 0.95,0.8,0.7,0.6）。
        gt = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
        pred = torch.tensor([
            [0., 0., 10., 10.],     # 0.95: G0 に完全一致 -> TP
            [0., 0., 9., 9.],       # 0.80: G0 は使用済み -> FP（重複検出）
            [20., 20., 30., 30.],   # 0.70: G1 に一致 -> TP
            [100., 100., 110., 110.],  # 0.60: どの GT とも無関係 -> FP
        ])
        scores = torch.tensor([0.95, 0.80, 0.70, 0.60])
        tp, fp = ex7_match_detections(pred, scores, gt, 0.5)
        exp_tp = torch.tensor([1., 0., 1., 0.])
        exp_fp = torch.tensor([0., 1., 0., 1.])
        ok = torch.equal(tp.float(), exp_tp) and torch.equal(fp.float(), exp_fp)
        return ok, f"tp={tp.tolist()} fp={fp.tolist()}"
    check("ex7_match_detections", _c7)

    # ex8: 累積 precision/recall を手計算と突き合わせ。
    def _c8():
        tp = torch.tensor([1., 0., 1., 0.])
        fp = torch.tensor([0., 1., 0., 1.])
        precision, recall = ex8_pr_curve(tp, fp, n_gt=2)
        exp_recall = torch.tensor([0.5, 0.5, 1.0, 1.0])
        exp_prec = torch.tensor([1.0, 0.5, 2 / 3, 0.5])
        ok = torch.allclose(recall, exp_recall, atol=1e-4) and \
            torch.allclose(precision, exp_prec, atol=1e-4)
        return ok, f"P={[round(p,3) for p in precision.tolist()]} R={recall.tolist()}"
    check("ex8_pr_curve", _c8)

    # ex9: 全点補間 AP。手計算 = 0.5*1.0 + 0.5*(2/3) = 0.8333。
    def _c9():
        precision = torch.tensor([1.0, 0.5, 2 / 3, 0.5])
        recall = torch.tensor([0.5, 0.5, 1.0, 1.0])
        ap = ex9_ap_all_points(precision, recall)
        return abs(ap - 0.83333) < 1e-3, f"AP_all={ap:.4f} (expect 0.8333)"
    check("ex9_ap_all_points", _c9)

    # ex10: 11点補間 AP。手計算 = (6*1.0 + 5*(2/3))/11 = 0.8485。
    def _c10():
        precision = torch.tensor([1.0, 0.5, 2 / 3, 0.5])
        recall = torch.tensor([0.5, 0.5, 1.0, 1.0])
        ap = ex10_ap_11_point(precision, recall)
        return abs(ap - 0.84848) < 1e-3, f"AP_11pt={ap:.4f} (expect 0.8485)"
    check("ex10_ap_11_point", _c10)

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")
    return all_ok


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。exercises_solutions.py もこの解答を使う。
# =====================================================================

def _sol_ex1(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = torch.max(box_a[0], box_b[0])
    y1 = torch.max(box_a[1], box_b[1])
    x2 = torch.min(box_a[2], box_b[2])
    y2 = torch.min(box_a[3], box_b[3])
    iw = (x2 - x1).clamp(min=0)
    ih = (y2 - y1).clamp(min=0)
    inter = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if float(union) <= 0:
        return 0.0
    return float(inter / union)


def _sol_ex2(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def _sol_ex3(boxes: torch.Tensor, scores: torch.Tensor, thresh: float):
    mask = scores >= thresh
    return boxes[mask], scores[mask]


def _sol_ex4(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> list[int]:
    order = scores.argsort(descending=True).tolist()
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _sol_ex1(boxes[i], boxes[j]) <= iou_thresh]
    return keep


def _sol_ex5(boxes_norm: torch.Tensor, height: int, width: int) -> torch.Tensor:
    cx = boxes_norm[:, 0] * width
    cy = boxes_norm[:, 1] * height
    w = boxes_norm[:, 2] * width
    h = boxes_norm[:, 3] * height
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _sol_ex6(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float
) -> list[int]:
    keep: list[int] = []
    for c in labels.unique().tolist():
        idx = (labels == c).nonzero(as_tuple=True)[0]  # このクラスの元index
        sub_keep = _sol_ex4(boxes[idx], scores[idx], iou_thresh)  # 部分集合内のローカルindex
        keep.extend(int(idx[k]) for k in sub_keep)  # 元indexへ写し戻す
    return keep


def _sol_ex7(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = pred_scores.argsort(descending=True)
    n = pred_scores.shape[0]
    tp = torch.zeros(n)
    fp = torch.zeros(n)
    matched: set[int] = set()
    for out_i, p in enumerate(order.tolist()):
        best_iou, best_g = 0.0, -1
        for g in range(gt_boxes.shape[0]):
            if g in matched:
                continue
            iou = _sol_ex1(pred_boxes[p], gt_boxes[g])
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_g >= 0 and best_iou >= iou_thresh:
            tp[out_i] = 1.0
            matched.add(best_g)
        else:
            fp[out_i] = 1.0
    return tp, fp


def _sol_ex8(
    tp: torch.Tensor, fp: torch.Tensor, n_gt: int
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_cum = torch.cumsum(tp.float(), dim=0)
    fp_cum = torch.cumsum(fp.float(), dim=0)
    recall = tp_cum / max(int(n_gt), 1)
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-12)
    return precision, recall


def _to_np(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype("float64")
    return np.asarray(x, dtype="float64")


def _sol_ex9(precision: torch.Tensor, recall: torch.Tensor) -> float:
    import numpy as np
    rec = _to_np(recall)
    pre = _to_np(precision)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], pre, [0.0]))
    # 後ろから前へ累積 max（precision を単調非増加の envelope に）
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sol_ex10(precision: torch.Tensor, recall: torch.Tensor) -> float:
    import numpy as np
    rec = _to_np(recall)
    pre = _to_np(precision)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = rec >= t
        p = float(pre[mask].max()) if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def _install_solutions() -> None:
    """グローバルの ex*_ を模範解答に差し替える（SHOW_SOLUTION / solutions ランナー用）。"""
    g = globals()
    g["ex1_iou"] = _sol_ex1
    g["ex2_xyxy_to_cxcywh"] = _sol_ex2
    g["ex3_score_filter"] = _sol_ex3
    g["ex4_nms"] = _sol_ex4
    g["ex5_cxcywh_norm_to_xyxy_abs"] = _sol_ex5
    g["ex6_batched_nms"] = _sol_ex6
    g["ex7_match_detections"] = _sol_ex7
    g["ex8_pr_curve"] = _sol_ex8
    g["ex9_ap_all_points"] = _sol_ex9
    g["ex10_ap_11_point"] = _sol_ex10


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
