# 39_cluster_clip_pipeline: Cluster-CLIPパイプライン統合(総合) — Split→Build→Search→Stream

> トラック: **応用(Cluster-CLIP)** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `vector` `video`

## 🎯 この章のゴール
講座の全要素を統合し、動画分割→dense CLIP+クラスタリングでFAISSインデックス+SQLite構築→テキスト検索→クラスタマスク重畳の可視化までをCPU完結のミニ版で実装し、さらにmultiprocessingのproducer/consumer/writer・キュー満杯時のフレームドロップ・ヒストグラム差分の適応サンプリングというリアルタイムストリーム処理設計を読み解いて自力で再構築できる。

## 扱うトピック
- Split(OpenCVで動画→フレームJPEG)とadaptive sampling(ヒストグラム差分)
- Build(dense CLIP+Agglomerative→クラスタ代表ベクトルをFAISS、メタデータSQLite)
- Search(テキストクエリ→CLIPテキストエンコーダ→FAISSコサイン近傍→該当フレーム+クラスタマスクHTML可視化)
- Stream(camera/YouTubeをproducer/consumer/writerの3プロセス、put_nowait即ドロップ、POISON_PILL)
- cv2.addWeightedによるマスク重畳と可視化
- 参照: stream/capture.py・profiler.py・search/pipeline.py

## 主要API
`cv2.VideoCapture` / `faiss.IndexFlatIP` / `faiss.write_index` / `sqlite3.connect` / `multiprocessing.Process` / `multiprocessing.Queue` / `put_nowait` / `cv2.addWeighted`

## 評価方法
検索品質を、Cluster-CLIP本体と同じくGT BBoxとクラスタマスクのカバレッジ閾値スイープからP@k(上位kの適合率)とNDCG(関連度の割引利得正規化)を算出して評価する。ストリームは処理FPS・レイテンシ・フレームドロップ率でリアルタイム性を併せて計測する。

## 完成物
動画→dense CLIPクラスタFAISS構築→テキスト検索→マスク可視化までをCPUで通すミニCluster-CLIPと、P@k/NDCG評価+ストリーム3プロセス版の実装。

## CPU / GPU メモ
全行程CPU完結(ViT-B-32・7x7・faiss-cpu)。ストリームはCPUで実時間に追いつかない前提でfps上限+put_nowaitドロップを必ず入れる。faiss.get_num_gpusはtry/exceptで守る。yt-dlpはURL解決のみ。

## 予定スクリプト
- `01_split_adaptive_sampling.py`
- `02_build_dense_cluster_index.py`
- `03_search_visualize.py`
- `04_stream_multiprocessing.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `embed` `vector` `video`）
