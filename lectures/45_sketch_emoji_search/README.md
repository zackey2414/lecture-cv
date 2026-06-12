# 45_sketch_emoji_search: 実践: 手書きスケッチで絵文字を検索（CLIP＋FAISS のスケッチ画像検索 SBIR）

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`
> 前提モジュール: `16_clip_zeroshot_retrieval`, `17_faiss_image_search`

## 🎯 この章のゴール
手書きスケッチを入力に絵文字を検索する スケッチベース画像検索(SBIR) システムを CLIP埋め込み＋FAISS で一から作れる。絵文字をグレースケール化して埋め込み→FAISSに保存し、Tkinter のマウス描画ウィンドウで手書き入力→埋め込み→類似上位N件を返す。ローカルGUIと headless(合成スケッチ/ファイル) の両対応、スケッチ↔絵文字のドメインギャップとその緩和(グレースケール/エッジ/反転)、Recall@N 評価まで実装する。

## 扱うトピック
- スケッチベース画像検索(SBIR)の全体像と設計
- 絵文字コレクションの用意(フォント描画→合成→data/上書き)とグレースケール化
- CLIP画像埋め込み→L2正規化→FAISS(IndexFlatIP)登録・保存/読込
- Tkinter キャンバスでのマウス手書き入力(headlessは合成スケッチに自動フォールバック)
- スケッチの前処理(白背景/黒線/正方化/CLIP前処理)
- FAISS検索で類似上位N件・スコア可視化
- スケッチ↔絵文字のドメインギャップと緩和(グレースケール/エッジ/反転)
- Recall@N 評価

## 主要API
`CLIPModel.get_image_features` / `faiss.IndexFlatIP` / `faiss.normalize_L2` / `tkinter.Canvas` / `PIL.ImageDraw` / `cv2.cvtColor`

## 評価方法
合成スケッチ(既知の絵文字に対応)での Recall@N / top-N ヒット率

## 完成物
絵文字をFAISSに索引化し、手書き(マウス描画 or 合成)スケッチで類似絵文字 上位N件を返すエンドツーエンドの検索アプリ

## CPU / GPU メモ
全てCPU。描画ウィンドウは Tkinter(標準ライブラリ)＝opencv headless でも動く。display 無し(Docker/CI)では合成スケッチに自動フォールバックして完走。CLIP は小型 clip-vit-base-patch32。

## 予定スクリプト
- `emoji_lab.py`
- `01_build_emoji_index.py`
- `02_sketch_input_tk.py`
- `03_search_topn.py`
- `04_eval_domaingap.py`
- `mini_project.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体は順次作成します。
