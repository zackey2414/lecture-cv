"""第14回 共通ヘルパ — 合成データ生成と出力先管理。

評価指標を学ぶのに、実画像も学習済みモデルも要らない。
評価指標は結局「正解ラベル」と「予測（ラベル or スコア）」の2つだけから計算できる。
そこで本モジュールは、現実的な“分類器の出力”を合成で作って、指標の定義と計算手順だけに集中する。

各スクリプトはこのヘルパを import し、同じ seed のデータで実験するので結果は再現する。
実データで試したい人は、自分の (y_true, proba) を作って同じ関数に流せばそのまま動く。
"""

from __future__ import annotations

import pathlib

import numpy as np

# 多クラスの題材ラベル名（4クラス）。混同行列の軸ラベルに使うだけで、計算には影響しない。
CLASS_NAMES = ["cat", "dog", "bird", "car"]


def output_dir() -> pathlib.Path:
    """結果の保存先 lectures/14_eval_classification/outputs/ を返す（無ければ作る）。

    __file__ は lectures/14_eval_classification/eval_helpers.py。
    .parent がこのモジュールのディレクトリなので、その直下の outputs/ に保存する。
    """
    base = pathlib.Path(__file__).resolve().parent / "outputs"
    base.mkdir(parents=True, exist_ok=True)  # exist_ok=True で二重作成エラーを防ぐ
    return base


def make_multiclass_scores(
    n: int = 600, n_classes: int = 4, signal: float = 2.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """多クラス分類器の出力（予測確率）を合成する。

    返り値:
      y_true : shape (n,)        正解ラベル 0..n_classes-1
      proba  : shape (n, K)      各クラスの予測確率（行ごとに合計1の softmax 出力）

    作り方の発想:
      1. 正解ラベルを少しだけ偏った頻度で決める（不均衡の練習。完全一様にはしない）。
      2. ロジットは標準正規ノイズで初期化し、「正解クラスのロジットだけ」平均 signal 持ち上げる。
      3. softmax で確率に変換。signal を大きくするほど argmax が当たりやすい“賢い分類器”になる。
    こうすると、当たるが時々外す、現実的な混同行列・スコア分布が得られる。
    """
    rng = np.random.default_rng(seed)
    # クラス頻度を 1.0:…:1.8 くらいに傾ける（後で「不均衡で accuracy が誤誘導」を見せる布石）
    weights = np.linspace(1.0, 1.8, n_classes)
    weights /= weights.sum()
    y_true = rng.choice(n_classes, size=n, p=weights)

    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    # 正解クラスのロジットだけ持ち上げる（fancy index で各行の y_true 列を選ぶ）
    logits[np.arange(n), y_true] += rng.normal(signal, 1.0, size=n)

    # 数値安定な softmax（行ごとの最大値を引いてから exp する）
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    proba = e / e.sum(axis=1, keepdims=True)
    return y_true, proba


def make_binary_scores(
    n: int = 2000, pos_ratio: float = 0.1, gap: float = 1.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """二値分類スコアを合成する（ROC/PR の題材）。

    返り値:
      y_true : shape (n,) int 0/1
      score  : shape (n,)     「陽性である確率」っぽいスコア [0,1]

    陰性スコアは N(0,1)、陽性スコアは N(gap,1) から引いてシグモイドで [0,1] に潰す。
    - gap を大きくすると2つの分布が分離し AUC が上がる（=分類しやすい）。
    - pos_ratio で陽性の割合（不均衡）を制御する。ROC と PR の違いはここで効いてくる。
    """
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pos_ratio).astype(int)
    # 連続値スコアにしておくと「同点(タイ)」がほぐ生じず、自作とライブラリの値がきれいに一致する
    raw = np.where(y_true == 1, rng.normal(gap, 1.0, n), rng.normal(0.0, 1.0, n))
    score = 1.0 / (1.0 + np.exp(-raw))
    return y_true, score
