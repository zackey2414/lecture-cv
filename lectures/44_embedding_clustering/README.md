# 44_embedding_clustering: 埋め込みのクラスタリング — 画像・テキスト・クロスモーダルを教師なしで束ねる

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `metrics` `vector`
> 前提モジュール: `16_clip_zeroshot_retrieval`, `17_faiss_image_search`

## 🎯 この章のゴール
CLIP/ResNet/SigLIP で得た埋め込み(画像・テキスト・顔)を、ラベル無しで クラスタリング して自動グルーピングできる。k-means/DBSCAN/agglomerative(HDBSCAN任意)の使い分け、クラスタ数kの選び方(エルボー/シルエット)、評価(silhouette/NMI/purity/homogeneity)、次元削減(PCA/t-SNE/UMAP任意)での可視化を実装できる。顔クラスタリング(30)を一般化し、画像コレクションのトピック分け・テキスト(キャプション/ラベル)のクラスタリング・クロスモーダルまで扱える。

## 扱うトピック
- 埋め込みクラスタリングとは(検索との違い・正規化)
- k-means/DBSCAN/agglomerative/HDBSCAN(任意)の使い分け
- クラスタ数kの選び方(エルボー・シルエット)
- 評価: silhouette(教師なし)/ NMI・purity・homogeneity(正解があるとき)
- 次元削減で可視化(PCA/t-SNE/UMAP任意)
- 画像コレクションのクラスタリング(CLIP/ResNet画像埋め込み)
- テキスト(キャプション/ラベル)のクラスタリング
- クロスモーダル・顔(30)との関係(同じ枠組み)

## 主要API
`sklearn.cluster.KMeans` / `sklearn.cluster.DBSCAN` / `sklearn.cluster.AgglomerativeClustering` / `sklearn.metrics.silhouette_score` / `sklearn.metrics.normalized_mutual_info_score` / `sklearn.decomposition.PCA` / `sklearn.manifold.TSNE`

## 評価方法
silhouette(教師なし)・NMI/purity/homogeneity(正解があるとき)・エルボー/シルエットでのk選択

## 完成物
画像コレクションとテキスト集合を埋め込み→クラスタリングして自動グルーピングし、k選択・評価・2D可視化まで行うツール

## CPU / GPU メモ
全てCPU。埋め込みは小型CLIP/ResNet(dl/hf)、クラスタリング/可視化は scikit-learn(metrics)。UMAP は任意・ガード。

## 予定スクリプト
- `01_kmeans_image_embeddings.py`
- `02_choosing_k.py`
- `03_dbscan_agglomerative.py`
- `04_text_and_crossmodal.py`
- `05_visualize_reduce.py`
- `mini_project.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
