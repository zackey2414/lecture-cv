"""04_mot_metrics.py — MOT 評価指標(MOTA/MOTP/IDF1/簡易HOTA)を numpy で自前実装

多物体追跡の良し悪しは「検出が当たっているか」と「ID が一貫しているか」の両面で測る。
正準ライブラリ（motmetrics, TrackEval）は本講座の numpy2 系/巨大依存と衝突しうるため
**使わず**、定義式どおりに numpy で実装して指標の意味を体で覚える。

  CLEAR-MOT:
    MOTA = 1 - (FN + FP + IDSW) / GT総数      … 検出誤りと ID 切替を 1 本に統合（負にもなる）
    MOTP = マッチした組の平均 IoU            … 当たった検出の位置精度（高いほど良い）
  ID 指標:
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)    … ID を通した F1（軌跡全体の一貫性）
  HOTA（簡易・単一IoU閾値）:
    HOTA = √(DetA × AssA)                      … 検出 DetA と 関連付け AssA を分離

実行:
    uv run python lectures/28_tracking/04_mot_metrics.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import iou_matrix, linear_assignment, make_mot_scene  # noqa: E402


# ----------------------------------------------------------------------------
# CLEAR-MOT + IDF1 + 簡易 HOTA をまとめて計算する
# ----------------------------------------------------------------------------


def evaluate_mot(gt_list, pred_list, iou_thr=0.5):
    """フレーム列の GT と予測から MOT 指標一式を計算する。

    引数:
        gt_list[t]   : {gt_id: box(xyxy)}
        pred_list[t] : {pred_id: box(xyxy)}
        iou_thr      : マッチとみなす IoU 閾値
    返り値: dict（MOTA/MOTP/IDF1/HOTA ほか内訳）
    """
    FN = FP = IDSW = GT = 0
    iou_sum = 0.0
    n_match = 0
    prev_match: dict[int, int] = {}              # gt_id -> 直近で対応した pred_id（IDSW 判定用）
    match_count: dict[tuple, int] = defaultdict(int)  # (gt_id,pred_id) が一致したフレーム数
    gt_tp_by_id: dict[int, int] = defaultdict(int)    # gt_id が TP になった総回数
    pred_tp_by_id: dict[int, int] = defaultdict(int)  # pred_id が TP になった総回数
    gt_total = pred_total = 0

    for gt_frame, pred_frame in zip(gt_list, pred_list):
        gids = list(gt_frame.keys())
        pids = list(pred_frame.keys())
        GT += len(gids)
        gt_total += len(gids)
        pred_total += len(pids)

        if gids and pids:
            gt_boxes = np.array([gt_frame[i] for i in gids])
            pred_boxes = np.array([pred_frame[i] for i in pids])
            iou = iou_matrix(gt_boxes, pred_boxes)        # (G,P)
            # IoU 閾値未満は対応不可にするため巨大コストを入れてからハンガリアン
            cost = 1.0 - iou
            cost[iou < iou_thr] = 1e6
            pairs = linear_assignment(cost)

            matched_g, matched_p = set(), set()
            for g, p in pairs:
                if iou[g, p] < iou_thr:
                    continue
                gid, pid = gids[g], pids[p]
                matched_g.add(g)
                matched_p.add(p)
                n_match += 1
                iou_sum += iou[g, p]
                match_count[(gid, pid)] += 1
                gt_tp_by_id[gid] += 1
                pred_tp_by_id[pid] += 1
                # ID スイッチ: 同じ GT が前回と違う pred に乗り換えたら +1
                if gid in prev_match and prev_match[gid] != pid:
                    IDSW += 1
                prev_match[gid] = pid
            FN += len(gids) - len(matched_g)
            FP += len(pids) - len(matched_p)
        else:
            FN += len(gids)
            FP += len(pids)

    # --- CLEAR-MOT ---
    mota = 1.0 - (FN + FP + IDSW) / GT if GT > 0 else 0.0
    motp = iou_sum / n_match if n_match > 0 else 0.0

    # --- IDF1: GT-ID と Pred-ID を全系列で 1対1 対応させ、一致フレーム総数を最大化 ---
    gt_ids = sorted({k[0] for k in match_count} | set(gt_tp_by_id))
    pred_ids = sorted({k[1] for k in match_count} | set(pred_tp_by_id))
    idtp = 0
    if gt_ids and pred_ids:
        gi = {g: i for i, g in enumerate(gt_ids)}
        pi = {p: j for j, p in enumerate(pred_ids)}
        m = np.zeros((len(gt_ids), len(pred_ids)))
        for (gid, pid), c in match_count.items():
            m[gi[gid], pi[pid]] = c
        # 一致数最大化 = -m の最小コスト割当
        for g, p in linear_assignment(-m):
            idtp += int(m[g, p])
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

    # --- 簡易 HOTA（単一 IoU 閾値）---
    # DetA: 検出レベルの正答率
    deta = n_match / (n_match + FN + FP) if (n_match + FN + FP) > 0 else 0.0
    # AssA: 各 TP の「関連付けの正しさ」を平均
    assa_num = 0.0
    for (gid, pid), tpa in match_count.items():
        fna = gt_tp_by_id[gid] - tpa     # その GT が別 pred に取られた回数
        fpa = pred_tp_by_id[pid] - tpa   # その pred が別 GT に取られた回数
        a = tpa / (tpa + fna + fpa) if (tpa + fna + fpa) > 0 else 0.0
        assa_num += a * tpa              # この組の TP 検出数ぶん重み付け
    assa = assa_num / n_match if n_match > 0 else 0.0
    hota = float(np.sqrt(deta * assa))

    return {
        "MOTA": mota, "MOTP": motp, "IDF1": idf1, "HOTA": hota,
        "DetA": deta, "AssA": assa,
        "FN": FN, "FP": FP, "IDSW": IDSW, "GT": GT,
        "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
    }


# ----------------------------------------------------------------------------
# 評価を試すための最小トラッカ（IoU 貪欲）— 04 を単体で完結させるため同梱
# ----------------------------------------------------------------------------


class SimpleIouTracker:
    """前フレームのボックスと IoU で対応付けるだけの素朴なトラッカ（Kalman なし）。"""

    def __init__(self, iou_thr=0.3, max_age=2):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = []  # dict(id, box, age)
        self._next = 1

    def update(self, dets):
        dets = np.asarray(dets, dtype=float).reshape(-1, 4)
        out = {}
        if not self.tracks:
            for d in dets:
                self.tracks.append({"id": self._next, "box": d, "age": 0})
                out[self._next] = d
                self._next += 1
            return out
        tb = np.array([t["box"] for t in self.tracks]).reshape(-1, 4)
        iou = iou_matrix(dets, tb)
        cost = 1 - iou
        md, mt = set(), set()
        for d, t in linear_assignment(cost):
            if iou[d, t] < self.iou_thr:
                continue
            self.tracks[t]["box"] = dets[d]
            self.tracks[t]["age"] = 0
            out[self.tracks[t]["id"]] = dets[d]
            md.add(int(d))
            mt.add(int(t))
        for d in range(len(dets)):
            if d not in md:
                self.tracks.append({"id": self._next, "box": dets[d], "age": 0})
                out[self._next] = dets[d]
                self._next += 1
        for t in range(len(self.tracks)):
            if t not in mt:
                self.tracks[t]["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return out


def main():
    scene = make_mot_scene(num_objects=4, num_frames=40, seed=7)
    gt_list = scene.gt

    print("=" * 64)
    print("MOT 評価指標を自前実装で検証")

    # (1) サニティチェック: 予測 = GT（完璧）なら MOTA=IDF1=HOTA=1, IDSW=0
    perfect = evaluate_mot(gt_list, gt_list, iou_thr=0.5)
    print("\n[完璧な予測（pred = GT）]")
    print(f"  MOTA={perfect['MOTA']:.3f}  MOTP={perfect['MOTP']:.3f}  "
          f"IDF1={perfect['IDF1']:.3f}  HOTA={perfect['HOTA']:.3f}  IDSW={perfect['IDSW']}")
    assert abs(perfect["MOTA"] - 1.0) < 1e-9 and abs(perfect["IDF1"] - 1.0) < 1e-9

    # (2) 素朴トラッカ（ノイズ検出を入力）を評価 → 誤りが指標に反映される
    tracker = SimpleIouTracker(iou_thr=0.3, max_age=2)
    pred_list = []
    for t in range(scene.num_frames):
        dets = scene.det[t][:, :4] if len(scene.det[t]) else np.empty((0, 4))
        pred_list.append(tracker.update(dets))
    res = evaluate_mot(gt_list, pred_list, iou_thr=0.5)
    print("\n[素朴 IoU トラッカ（ノイズ検出入力）]")
    print(f"  MOTA={res['MOTA']:.3f}  MOTP={res['MOTP']:.3f}  "
          f"IDF1={res['IDF1']:.3f}  HOTA={res['HOTA']:.3f}")
    print(f"  内訳: FN={res['FN']} FP={res['FP']} IDSW={res['IDSW']} GT={res['GT']}")
    print(f"  HOTA 分解: DetA={res['DetA']:.3f}  AssA={res['AssA']:.3f}")

    print("\n読み方:")
    print("  - MOTA は FN+FP+IDSW を GT で割って 1 から引く。誤りが多いと簡単に負になる。")
    print("  - MOTP は『当たった組』だけの位置精度。検出を取りこぼしても下がらない点に注意。")
    print("  - IDF1 は ID の通し一貫性。MOTA が高くても ID が切れれば下がる。")
    print("  - HOTA=√(DetA×AssA) は検出と関連付けを分離。厳密な多閾値版は TrackEval（任意）。")


if __name__ == "__main__":
    main()
