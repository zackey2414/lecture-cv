"""第18回 演習問題（物体検出 入門）。

使い方:
  1. 各 exN_*() の TODO を自分で実装する（最初は NotImplementedError だが、
     採点ランナーが拾うのでプロセスは落ちず、FAIL と表示されるだけ → exit 0）。
  2. 自己採点:  uv run python lectures/18_object_detection_intro/exercises.py
     全問 pass すれば "ALL PASS" と表示される。
  3. 詰まったら模範解答の挙動を見る:
        SHOW_SOLUTION=1 uv run python lectures/18_object_detection_intro/exercises.py

この5問は本モジュールの核を1つずつ抜き出したもの（モデルDL不要・純計算）:
  ex1: IoU（2つの box の交差/和集合）             … 検出評価とNMSの土台
  ex2: xyxy → cxcywh への変換                      … box_convert の中身
  ex3: score 閾値フィルタ                          … 生検出の間引き
  ex4: 1クラスNMS（confidence降順の貪欲抑制）      … torchvision.ops.nms の中身
  ex5: target_sizes=(H,W) で正規化cxcywh→xyxy絶対  … DETR後処理の核・典型バグ回避
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

    print("=== 採点結果 ===")
    all_ok = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name:28s} {detail}")
    print("\nALL PASS 🎉" if all_ok else "\nまだ未達の演習があります。TODO を埋めましょう。")


# =====================================================================
# 模範解答（SHOW_SOLUTION=1 のときに本体へ差し替えて実行）
# まずは自力で解いてから見ること。
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


def _install_solutions() -> None:
    g = globals()
    g["ex1_iou"] = _sol_ex1
    g["ex2_xyxy_to_cxcywh"] = _sol_ex2
    g["ex3_score_filter"] = _sol_ex3
    g["ex4_nms"] = _sol_ex4
    g["ex5_cxcywh_norm_to_xyxy_abs"] = _sol_ex5


if __name__ == "__main__":
    if os.environ.get("SHOW_SOLUTION") == "1":
        print("(模範解答モードで実行します)\n")
        _install_solutions()
    _grade()
