# 30_anomaly_iqa: 異常検知と画像品質評価 — anomalib(PaDiM/PatchCore)・pyiqa

> トラック: **異常検知・品質** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `anomaly` `metrics`

## 🎯 この章のゴール
正常画像のみで学習しキズ/欠陥を検出するメモリバンク型(PaDiM/PatchCore)をanomalibでCPU推論でき、異常マップとスコア閾値・image/pixelレベルAUROC/AUPRで評価し、pyiqaで参照あり(PSNR/SSIM/LPIPS)/参照なし(BRISQUE/NIQE/MUSIQ)の画像品質指標を計算して復元・生成評価に接続できる。

## 扱うトピック
- anomalib PaDiM/PatchCore(メモリバンク型、正常のみ学習)とEngine(accelerator='cpu')
- 異常マップ(anomaly_map)とスコア閾値、MVTec ADでの評価
- 評価: image-level/pixel-level AUROC・AUPR
- pyiqaの参照あり(PSNR/SSIM/LPIPS)と参照なし(BRISQUE/NIQE/MUSIQ/TOPIQ)
- metric.lower_betterで指標の良し悪し方向を確認
- 生成・超解像(29回)評価との接続

## 主要API
`anomalib.models.Padim` / `anomalib.models.Patchcore` / `anomalib.engine.Engine` / `pyiqa.create_metric` / `metric.lower_better` / `sklearn.metrics.roc_auc_score` / `sklearn.metrics.average_precision_score`

## 評価方法
異常検知をimage-levelとpixel-levelのAUROC(ROC下面積)とAUPR(不均衡で実態を映す)で評価し、両者を併用する。画像品質はpyiqaで参照あり(PSNR/SSIM/LPIPS)/参照なし(BRISQUE/NIQE)指標を計算し、lower_betterを確認して混同しないようにする。

## 完成物
正常画像でPaDiM/PatchCoreを学習し欠陥の異常マップとAUROC/AUPRを出す異常検知スクリプトと、複数IQA指標を一括計算する品質評価コード。

## CPU / GPU メモ
PaDiM/PatchCoreの推論はEngine(accelerator='cpu')でCPU現実的、学習は小データに限定。anomalibはLightning依存で重く専用グループに隔離。pyiqaのIQAも単画像ならCPUで実用的。

## 予定スクリプト
- `01_anomalib_padim_patchcore.py`
- `02_anomaly_auroc_aupr.py`
- `03_pyiqa_metrics.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `anomaly` `metrics`）
