# 12_eval_classification: 評価指標の基礎(A) — 混同行列・precision/recall/F1・ROC/PR・AUC

> トラック: **評価指標** ／ レベル: **初級** ／ 必要な依存グループ: `dl` `metrics`

## 🎯 この章のゴール
全評価の最小単位TP/FP/FN/TNと混同行列を理解し、precision/recall/F1とmacro/micro/weighted平均、top-k accuracy、しきい値非依存のROC-AUC/PR-AUC(=AP)を、scikit-learn/torchmetrics利用と自作の両方で計算でき、クラス不均衡でaccuracyやROC-AUCが楽観的になる罠を説明できる。

## 扱うトピック
- TP/FP/FNと混同行列、accuracyの不均衡での誤誘導
- precision/recall/F1とmacro/micro/weighted平均の使い分け
- top-k accuracyと多クラスのone-vs-rest分解
- ROC曲線/ROC-AUCとPR曲線/PR-AUC(=AP)、不均衡での違い
- sklearnとtorchmetricsのupdate→compute→resetサイクル
- 自作実装とライブラリ値の突き合わせ

## 主要API
`sklearn.metrics.confusion_matrix` / `sklearn.metrics.classification_report` / `sklearn.metrics.roc_auc_score` / `sklearn.metrics.average_precision_score` / `torchmetrics.classification.Accuracy` / `torchmetrics.classification.AUROC` / `torchmetrics.classification.AveragePrecision`

## 評価方法
本モジュール自体が評価指標の実装回。confusion_matrixからprecision/recall/F1/accuracyをnumpyで自作し、sklearn/torchmetricsの結果と一致を検証。スコア閾値を掃引してROC/PR曲線を描き、台形則でAUC/APを自前計算してライブラリ値と突き合わせる。

## 完成物
予測スコアと正解ラベルから混同行列・各種指標・ROC/PR曲線を自作実装で計算し、sklearn/torchmetricsと一致確認するレポート生成スクリプト。

## CPU / GPU メモ
完全CPU。torchmetricsはdevice='cpu'で完結、preds/targetのdeviceを揃えるだけ。

## 予定スクリプト
- `01_confusion_matrix_prf.py`
- `02_roc_pr_auc.py`
- `03_torchmetrics_vs_manual.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `metrics`）
