# 42_multimodal_vector_search: マルチモーダル・ベクトル検索（FAISS）— 画像・テキスト・クロスモーダル

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`
> 前提モジュール: `16_clip_zeroshot_retrieval`, `17_faiss_image_search`

## 🎯 この章のゴール
CLIP/SigLIP で画像とテキストを同じ埋め込み空間へ写し、FAISS で 画像→画像 / テキスト→画像 / テキスト→テキスト / クロスモーダル の検索を1つの統一インターフェースで構築できる。L2正規化と内積（コサイン）、index 選択（Flat/IVF/HNSW）、id↔メタデータ付き永続化、Recall@k 評価まで。音声(CLAP)等の新モダリティ追加の設計も理解する。

## 扱うトピック
- 共有埋め込み空間（CLIP/SigLIP）でのモダリティ横断検索
- 画像→画像 / テキスト→画像 / テキスト→テキスト / クロスモーダル
- L2正規化と IndexFlatIP(コサイン)・IVF・HNSW の使い分け
- id↔元データのメタデータ付き永続化
- Recall@k / mAP 評価
- 新モダリティ（音声 CLAP 等）を足す統一設計

## 主要API
`faiss.IndexFlatIP` / `faiss.IndexIDMap` / `faiss.normalize_L2` / `CLIPModel.get_image_features` / `CLIPModel.get_text_features`

## 評価方法
Recall@k / mAP（クエリ→正解集合）。クロスモーダル検索の定性確認。

## 完成物
画像とテキストを1つのFAISS基盤に載せ、テキストでも画像でも引ける統一マルチモーダル検索エンジン（永続化・評価込み）

## CPU / GPU メモ
全てCPU・faiss-cpu。小型 CLIP/SigLIP を実DL&CPU推論。

## 予定スクリプト
- `01_shared_space_clip.py`
- `02_faiss_multimodal_index.py`
- `03_crossmodal_search.py`
- `04_recall_eval.py`
- `mini_project.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。
