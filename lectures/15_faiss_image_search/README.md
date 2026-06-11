# 15_faiss_image_search: FAISSベクトルDBと画像検索システム(評価込み)

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`

## 🎯 この章のゴール
NN/ANNの概念とFAISSの基本ループを理解し、IndexFlatL2/IP(正規化でコサイン)、IndexIDMapでのメタデータ紐付け、write_index永続化、IVF/HNSW/PQ(index_factory)の精度速度トレードオフを扱い、CLIP/ResNet埋め込みでend-to-endの画像検索を構築してRecall@k/QPSで定量評価できる。

## 扱うトピック
- IndexFlatL2/IndexFlatIP+normalize_L2(コサイン)、float32/C連続必須
- IndexIDMap/add_with_idsとSQLite等メタデータ別管理、-1ガード
- write_index/read_indexとメタデータDBのセット永続化
- IVFFlat(train/nlist/nprobe)・HNSW(M/efSearch)・IVFPQ/OPQ(index_factory)
- 埋め込み→正規化→add→search のend-to-endパイプライン
- インクリメンタル更新(Cluster-CLIPのstream/writer.pyが実例)

## 主要API
`faiss.IndexFlatIP` / `faiss.normalize_L2` / `faiss.IndexIDMap` / `index.add_with_ids` / `index.search` / `faiss.write_index` / `faiss.IndexIVFFlat` / `faiss.index_factory` / `np.ascontiguousarray`

## 評価方法
ANN品質を評価する: IndexFlat(厳密)の結果をground truthとし、IVF/HNSW/PQのRecall@kを集合一致(np.intersect1d)で算出。nprobe/efSearchをスイープしてQPS(time計測)-recall曲線を描き、ラベル付き検索ではretrieval mAPも測る。ground truthをANN自身で作る誤りを避ける。

## 完成物
画像群を埋め込み・正規化してFAISSへadd・永続化し、クエリ画像/テキストで検索、Recall@kとQPS-recall曲線を出力する画像検索システム一式。

## CPU / GPU メモ
faiss-cpuで全機能(Flat/IVF/HNSW/PQ)が動作。GPU版faiss-gpu-cuvsはLinux+NVIDIA限定でcpuと排他、CPU環境ではindex_cpu_to_gpuをtry/exceptで守る注記のみ。

## 予定スクリプト
- `01_flat_ip_cosine.py`
- `02_idmap_persist_sqlite.py`
- `03_ivf_hnsw_pq.py`
- `04_recall_qps_eval.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `vector` `metrics`）
