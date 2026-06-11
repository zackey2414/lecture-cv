# 02_cv_libraries_overview: 画像・動画処理ライブラリの地図 — OpenCV/Pillow/scikit-image/albumentations/kornia ほか

> トラック: **画像の基礎** ／ レベル: **入門** ／ 必要な依存グループ: （基礎のみ・追加依存なし）

## 🎯 この章のゴール
画像・動画処理で使う主要ライブラリ（OpenCV/Pillow/NumPy/scikit-image/imageio、データ拡張: torchvision transforms v2/albumentations/kornia、動画I/O: PyAV/imageio-ffmpeg）の役割分担・長所短所・相互運用・選び方を地図として把握し、課題に応じて適切なライブラリを選べるようになる。

## 扱うトピック
- 各ライブラリの位置づけと使い分け（速度/機能/微分可能性/GPU/エコシステム）
- 同一処理（読込・リサイズ・ぼかし・回転）をOpenCV/Pillow/scikit-imageで書き比べ
- データ拡張ライブラリ: torchvision transforms v2 / albumentations(Compose・bbox/mask対応) / kornia(微分可能・GPU)
- 動画I/O: OpenCV VideoCapture / imageio / PyAV の違い
- 相互運用（ndarray⇄PIL⇄tensor）とライブラリ選択の判断基準

## 主要API
`cv2` / `PIL.Image` / `skimage` / `albumentations.Compose` / `torchvision.transforms.v2` / `kornia.augmentation` / `imageio.v3` / `av`

## 評価方法
概念回（数値評価は無し）。ただしデータ拡張の前後を可視化で比較し、拡張が分布に与える影響を目で確認する。

## 完成物
主要ライブラリの早見表＋同一処理を各ライブラリで書き比べた比較スクリプトと、albumentations による拡張パイプラインのデモ。

## CPU / GPU メモ
全てCPUで動く。実行コードは main 依存(cv2/PIL/numpy/matplotlib)で完結させ、albumentations/scikit-image/kornia 等は import を try/except でガードし、未導入なら案内のみ（`uv add --group aug` 等）。

## 予定スクリプト
- `01_library_map.py`
- `02_same_op_across_libs.py`
- `03_augmentation_albumentations.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
