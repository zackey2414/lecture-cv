# 38_cluster_clip_dense_cluster: Cluster-CLIP中核 — dense CLIP特徴と空間連結クラスタリング

> トラック: **応用(Cluster-CLIP)** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `hf` `embed` `vector` `metrics`

## 🎯 この章のゴール
CLIPのvisual encoderを手でforward展開し、CLSではなくパッチ単位のdense特徴[C,H,W](ViT:パッチトークン、ResNet:attnpoolのv_proj/c_proj)を取り出し、grid_to_graph付きAgglomerativeClusteringで領域にまとめ各クラスタ平均→L2正規化で代表ベクトルとクラスタマップを得る、という『領域単位の開語彙検索』の核をCPU(H=W=7程度)で再実装できる。

## 扱うトピック
- visual encoderのforward展開(ViT: transformer/パッチトークン、ResNet: attnpool v_proj/c_proj→1x1conv化)
- CLSトークンを落としパッチ特徴[C,H,W]を取得(位置埋め込み/正規化の順序)
- sklearn AgglomerativeClustering + grid_to_graph(空間連結)で領域クラスタ
- クラスタ平均→F.normalizeで代表ベクトル、cmap/vecsの保存
- CPU制約(H=W=7〜14、AgglomerativeはO(n^2)メモリ)
- 参照: build/models.pyのdense_clip_embeddings_vit/_resnet、cluster_agglomerative

## 主要API
`model.visual.transformer` / `model.visual.attnpool.v_proj` / `sklearn.cluster.AgglomerativeClustering` / `sklearn.feature_extraction.image.grid_to_graph` / `torch.nn.functional.normalize` / `np.save`

## 評価方法
クラスタリング品質を、各クラスタ代表ベクトルとテキストクエリのコサイン類似度がGT領域(BBox)とどれだけ重なるか=カバレッジ(クラスタマスク∩BBox/BBox面積)で評価し、クラスタ数やしきい値を変えてカバレッジを比較する。定性的にはクラスタマップ重畳で確認。

## 完成物
ViT-B-32のdense特徴を手で展開して7x7パッチ特徴を取り、空間連結Agglomerativeで領域クラスタと代表ベクトルを生成・可視化するCPU版スクリプト。

## CPU / GPU メモ
CPUでViT-B-32・7x7パッチなら十分動作。dense特徴抽出でCLSを落とし忘れ/正規化順序を誤るとクラスタがノイズだらけになる。高解像度パッチはAgglomerativeのメモリ爆発に注意。

## 予定スクリプト
- `01_dense_clip_vit.py`
- `02_dense_clip_resnet.py`
- `03_agglomerative_cluster.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf` `embed` `vector` `metrics`）
