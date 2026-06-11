# 40. Cluster-CLIP — dense CLIP 特徴と空間連結クラスタリング（講座の総仕上げ）

> 版: torch 2.12+cpu / open-clip-torch 3.3 / faiss-cpu 1.14 / scikit-learn 1.9（2026-06）
> 元ネタ: 参照リポ **cluster-clip**（`build/models.py` の dense CLIP + 空間連結クラスタリングが本章の核）

---

## 🎯 ゴール

この章は本講座のクライマックスです。これまで積み上げた「画像 I/O → テンソル → ViT/CLIP 埋め込み → クラスタリング → FAISS 近傍探索 → SQLite → ストリーム処理」を、**1 枚の画像を意味のある領域に分けて検索できるようにする** という 1 つの目的のために統合します。題材は実在の研究システム **Cluster-CLIP**（テキストで動画フレームを開語彙検索する）で、その中核である「dense CLIP 特徴を空間連結クラスタリングで領域に束ね、領域代表ベクトルで検索する」という発想を、CPU だけで動く小型版として手を動かして再構築します。

具体的な到達点は次の通りです。

- **なぜ「画像全体 1 ベクトル」では足りないのか** を、小物体の希釈という観点から説明できる。
- CLIP の visual encoder を **手で forward 展開** して、CLS ではなく **パッチトークン**（dense 特徴 `[C, H, W]`）を取り出せる。
- **空間連結（grid_to_graph）付き AgglomerativeClustering** で、特徴が似ていて かつ 空間的に隣接したパッチだけを束ね、飛び地のない領域に分割できる。
- 各領域の **代表ベクトル（平均 → L2 正規化）** を作り、**FAISS（IndexFlatIP + IDMap）** に登録し、**SQLite** で `faiss_id ↔ (フレーム, 領域, bbox)` を引けるようにできる。
- テキスト / 画像クエリで領域検索し、ヒット領域をマスク重畳で可視化できる。
- これらを **Split → Build → Search → Stream** の 1 本のパイプライン（capstone）に束ね、`multiprocessing` で取得と推論を分離し、キュー満杯でフレームをドロップする実時間設計まで通せる。

このディレクトリのスクリプトは、リポジトリのルートから次のように動かします（出力は `outputs/40_cluster_clip_dense_cluster/`）。

```bash
uv run python lectures/40_cluster_clip_dense_cluster/01_dense_vs_global_clip.py
```

---

## 本編

### 1. なぜ「全体 1 ベクトル」ではなく「領域（dense）」なのか

CLIP の標準的な使い方（16 章・17 章）は、画像 1 枚を 1 本の埋め込みベクトルに潰し、テキストと同じ空間でコサイン類似度を測ることでした。これは「この画像は犬っぽいか」「全体としてビーチの写真か」を測るには強力です。しかし 1 本に潰すという操作は、**画面のどこに何があるかという空間情報を捨てる** ことを意味します。複数の物体が写ったフレームから「黄色いボールが写っている瞬間と、その位置」を引きたいとき、全体ベクトルは「ボールらしさ」を他の大きな物体や背景と平均してしまい、小物体の信号は薄まって消えます。

`01_dense_vs_global_clip.py` はこれを数値で見せます。224×224 の合成シーンを ViT-B/32 に通すと、パッチは 7×7 = 49 個。小さな黄色いボールはそのうち約 4 パッチ、つまり全体の **約 8%** しか占めません。全体を 1 本に pool すると、ボールの寄与は 49 分の数に薄まります。一方 dense 特徴なら、ボールは自分のパッチ群（やがて自分の「領域」）に固有のベクトルを持てます。これが「領域単位で検索したい」「小物体・複数物体を扱いたい」という要求に dense が答える理由です。

ただし正直に言うと、**生の ViT パッチトークンは、pool 後のベクトルほどテキストと綺麗に揃っていません**（CLIP の対照学習が整列させているのは pool 後のトークンだけだからです）。この章で学ぶのは「領域に分けて検索する仕組み（パイプライン）」であり、合成のフラットな画像ではテキスト検索の順位はあてになりません。実写画像では、この dense + クラスタリングの仕組みがきちんと領域を当てます。仕組みを正しく組めることを最優先に進めます。

### 2. dense CLIP 特徴の理論 — ViT を手で展開する

ViT の前半は「画像を 32×32 のパッチに切って、各パッチを 1 本のトークンに埋め込む」処理です（`conv1` がストライド 32 の畳み込みとしてこれを担います）。その後、先頭に学習可能な **CLS トークン** を 1 本足し、位置埋め込みを加え、Transformer の自己注意でトークン同士を混ぜ合わせます。標準の `encode_image` は最後に CLS（または pool）を 1 本取り出して返すので、パッチ単位の情報は外からは見えません。

dense 特徴を得るには、この forward を **自分で順に通して最終トークン列を全部受け取り、CLS を捨ててパッチだけを空間 `(gh, gw)` に並べ直す** 必要があります。手順は `conv1`（パッチ埋め込み）→ CLS 連結 → 位置埋め込み加算 → `ln_pre` → `transformer` → `ln_post` → `proj`（テキストと同じ共通潜在空間へ射影）→ **CLS 除去** → パッチごとに L2 正規化、です。`02_dense_features_extraction.py` は各ステップの shape を実況します（`conv1 -> (1,768,7,7)` → `+CLS -> (1,50,768)` → `proj -> (1,50,512)` → `drop CLS -> (1,49,512)`）。

前処理にも重要な勘所があります。標準の CLIP 前処理は短辺リサイズ + **CenterCrop** で正方形にしますが、これは画面端を切り落とすため、端にある小物体が dense 特徴から消えてしまいます。Cluster-CLIP は CenterCrop を捨て、**アスペクト比を無視して正方形へ強制 Resize** します（`load_encoder` の `Resize((224,224))`）。多少歪んでも端を残すほうが、領域分割では得だという判断です。

### 3. 空間連結クラスタリングの理論 — connectivity の意味

49 個のパッチベクトルを、似たもの同士で k 個の領域に束ねます。素朴に k-means や普通の凝集型クラスタリングをかけると、**特徴空間での近さだけ** で併合するので、画面の右上のパッチと左下のパッチが同じクラスタに入る「飛び地」が起こります。領域として扱いたいのに、空間的にバラバラでは検索結果のマスクが破綻します。

ここで効くのが **connectivity（連結）制約** です。`sklearn.feature_extraction.image.grid_to_graph(h, w)` は h×w グリッドの **4 近傍隣接グラフ** を作ります。これを `AgglomerativeClustering(connectivity=...)` に渡すと、凝集の各ステップで **グラフ上で隣接するクラスタ同士しか併合できなく** なります。結果として、どのクラスタも必ず空間的に 1 つながり（連結成分が 1 つ）になります。`03_spatial_connectivity.py` はこれを連結成分数で定量化します。connectivity ありなら「総連結成分数 == クラスタ数」（全領域が連結）、なしだと成分数がクラスタ数を上回り断片化します（実行例では ON=6 / OFF=8）。

linkage は `ward`（クラスタ内分散の増加が最小になるよう併合）を既定にします。ward は metric が `euclidean` であることを要求します。パッチを L2 正規化してあるので、ユークリッド距離はコサイン距離と単調な関係になり、「向きが似たパッチ」を素直に束ねられます。

### 4. 正準 API — 何を呼べばよいか

dense 特徴の取り出しは `open_clip` の visual encoder を直接触ります。クラスタリングと FAISS は sklearn / faiss の素直な API で済みます。本章の中核 API を一望すると次の通りです（`cc_common.py` が薄くラップしています）。

```python
import open_clip
# 1) モデル。force_quick_gelu=True は openai 系 ViT の活性化を一致させる定石。
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", force_quick_gelu=True)
model.eval()

# 2) dense 特徴: visual encoder を手で forward（cc_common.dense_vit_tokens）。
#    visual.conv1 / class_embedding / positional_embedding / transformer / ln_post / proj を使う。

# 3) 空間連結クラスタリング
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
conn = grid_to_graph(n_x=H, n_y=W)
labels = AgglomerativeClustering(n_clusters=k, linkage="ward",
                                 metric="euclidean", connectivity=conn).fit_predict(Y)

# 4) FAISS: コサイン = L2 正規化 + 内積(IP)。任意 ID を振るため IDMap。
import faiss
index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
index.add_with_ids(np.ascontiguousarray(vecs, np.float32), ids.astype(np.int64))
D, I = index.search(np.ascontiguousarray(query, np.float32), k)
```

`force_quick_gelu=True` は地味ですが重要です。openai 配布の ViT は QuickGELU を使って学習されており、これを合わせないと活性化がわずかにずれて特徴の質が落ちます。`metric="euclidean"` と `linkage="ward"` はセットで、L2 正規化済みベクトルに対して安定して働きます。FAISS は **float32・C連続** 以外を受け付けないので、`np.ascontiguousarray(x, dtype=np.float32)`（= `cc_common.as_faiss`）を必ず通します。

### 5. 実装を 1 つずつ

**dense 特徴（`cc_common.dense_vit_tokens` / 02）**: ViT を手で通し、`x = x[:, 1:, :]` で CLS を落とし、`x / x.norm(...)` でパッチごとに L2 正規化し、`[B, C, gh, gw]` に並べ替えます。`02` を実行すると、全 49 パッチのノルムが 1.000、水平方向の隣接パッチの平均コサインが 0.97 程度（空間的に滑らか）であることが確認できます。

**領域クラスタリングと代表ベクトル（`cc_common.cluster_regions` / 04）**: `[C,H,W]` を `[H*W, C]` に直し、connectivity 付き AgglomerativeClustering で `labels` を得て、クラスタごとに平均 → L2 正規化して **代表ベクトル `reps[k, C]`** を作ります。`04` の出力では、小物体の黄色いボールが自分専用のクラスタを得て、その bbox を **coverage 1.00** で覆えていることが分かります（= dense にした甲斐があった、の数値的裏付け）。

**FAISS + SQLite で検索基盤を作る（mini_project の Build / 17 章の復習）**: 代表ベクトルを `IndexFlatIP + IDMap` に `add_with_ids` し、`faiss_id` を SQLite の `VectorMapping(faiss_id, frame_id, cluster_idx, bbox)` と `Frames(frame_id, image_path, cmap_path)` に対応づけます。参照リポでは `faiss_id` を SQLite の `AUTOINCREMENT` 行 ID として採番しており、本章の mini_project も `cur.lastrowid` を `faiss_id` に使ってこれを再現します。FAISS はベクトルしか持たない（メタは持てない）ので、**インデックス本体（.faiss）とメタ DB（.db）は必ずセットで永続化・整合させる** のが鉄則です。

**検索と可視化（`search/engine.py` 相当 / 05・mini_project の Search）**: クエリ（テキストは `encode_text`、画像領域はその代表ベクトル）を L2 正規化して `index.search`、返った `faiss_id`（**-1 は近傍不足なので必ずスキップ**）で SQLite を join し、`(frame, cluster, bbox, image_path)` を得て、クラスタマップから該当領域のマスクを作って `cv2.addWeighted` で重畳します。

**ストリーム（`stream/pipeline.py` 相当 / mini_project の Stream）**: `multiprocessing` で **capture（取得）/ consumer（dense CLIP 推論 = CPU バウンド）/ writer（記録）** の 3 プロセスに分け、間を `Queue` でつなぎます。取得が推論を追い越したら、`put_nowait` で投入し `queue.Full` を捕まえて **フレームを捨てる**（落とさず待つと遅延が無限に積み上がる）。終端は **POISON_PILL**（番兵オブジェクト）を流して各プロセスを綺麗に終わらせます。

### 6. 落とし穴（このトピック固有）

最大の落とし穴は **dense 特徴で CLS を落とし忘れる / 位置埋め込みや ln の順序を間違える** ことです。CLS が混ざると 1 パッチ分だけ余計なトークンが領域に化け、クラスタリングがノイズだらけになります。手で forward を書くときは、各ステップの shape（`+CLS` で系列長が 50、`drop CLS` で 49）を必ず print して確かめてください。

次に **AgglomerativeClustering を高解像度のパッチに適用する** こと。AgglomerativeClustering は O(n²) のメモリを食うので、14×14 = 196 程度までに抑えるのが現実的です（本章は 7×7 = 49）。dense 特徴を高解像度に取れても、そのまま全パッチをクラスタリングに渡すとメモリと時間が爆発します。

そして **FAISS のお約束**。コサインのつもりで `IndexFlatIP` を使いながら正規化を忘れると単なる内積になり順位が崩れます。`faiss.normalize_L2` は **in-place で元配列を破壊** する点、float32・C連続でないと落ちる点、検索結果に **-1** が混じる点（メタ参照前にガード）も定番の罠です。

### 7. 実務の使い分け

「画像全体で似た画像を引きたい」なら従来の global CLIP + FAISS（17 章）で十分で、dense は不要です。dense + クラスタリングが効くのは、**1 枚に複数物体があり、その中の特定の領域・小物体をテキストで引きたい** ときや、検索結果に「画像のどこがヒットしたか」のマスクを出したいときです。コストは上がります（パッチ forward + クラスタリングがフレームごとに走る）。

クラスタ数 k は「1 フレームあたり何本のベクトルを index に積むか」を直接決めます。k を増やすほど細かい領域を引けますが、index は重く、ノイズ領域も増えます。参照リポは適応的に k を決める実験もしていますが、入門としては固定 k（5〜8）で十分です。ストリームでは、推論が実時間に追いつかない前提で **fps 上限 + キュー満杯ドロップ** を必ず入れ、「全フレームを処理する」のではなく「落としても破綻しない」設計にします。

---

## 🛠 章末ミニプロジェクト — ミニ Cluster-CLIP

`mini_project.py` は学んだ全要素を 1 本に束ねます。

```bash
uv run python lectures/40_cluster_clip_dense_cluster/mini_project.py
```

- **Split**: 黄色いボールが左→右に動く合成フレーム列を JPEG に分解（`outputs/.../mini_project/frames/`）。OpenCV は BGR をそのまま保存します。
- **Build**: 各フレームを dense CLIP → 空間連結クラスタリング → 代表ベクトル化。`cluster_maps/*.npy`・`vectors/*.npy` を保存し、`local_index.faiss`（IndexFlatIP+IDMap）と `local_metadata.db`（`Frames` / `VectorMapping`、`faiss_id` は SQLite の `lastrowid`）を構築。各クラスタの bbox（領域マスクの外接矩形）も登録。
- **Search**: テキストクエリ → `encode_text` → FAISS → SQLite join → ヒット領域をマスク重畳で `mini_project_search.png` に出力。`faiss_id == -1` をスキップし、メタ解決できることを assert で検証。
- **Stream**: `multiprocessing`（spawn）の capture / consumer / writer。`queue_size=2` の満杯キューに 8 フレームを投入し、推論が追いつかない分はドロップ。実行例では **投入 8 / 処理 2 / ドロップ 6**、実効 FPS と合わせて「取得が推論を追い越すと捨てる」挙動が観察できます。

この 1 本で、参照リポの `split / build / search / stream` の対応関係（`build/models.py`・`indexer.py`・`db_writer.py`・`search/engine.py`・`stream/pipeline.py`）が腑に落ちるはずです。

---

## ✅ 到達チェックリスト

- [ ] global ベクトルと dense 特徴の違いと、小物体が希釈される理由を説明できる。
- [ ] ViT を手で forward 展開し、CLS を落として `[C, gh, gw]` の dense 特徴を取り出せる。
- [ ] 前処理で CenterCrop を排し正方形 Resize にする理由を言える。
- [ ] `grid_to_graph` の connectivity を渡すと各クラスタが連結領域になることを、連結成分数で確認できる。
- [ ] クラスタ平均 → L2 正規化で代表ベクトルを作れる。
- [ ] `IndexFlatIP + IDMap` でコサイン検索し、`faiss_id` を SQLite メタと往復できる。
- [ ] 検索結果の `-1` をガードし、ヒット領域をマスク重畳で可視化できる。
- [ ] `multiprocessing` + 満杯キュードロップ + POISON_PILL でストリームを組める。
- [ ] mini_project（Split→Build→Search→Stream）を最後まで通せる。
- [ ] exercises.py を 10/10 PASS にできる。

---

## ❓ 落とし穴・FAQ・デバッグ

**Q. dense 特徴のノルムが 1 にならない / クラスタリングが砂嵐になる。**
A. CLS の落とし忘れ、L2 正規化の忘れ、`proj` 適用の有無が定番原因です。`02` のように各ステップの shape を print し、最後に `np.linalg.norm(fmap.reshape(C, -1), axis=0)` が全部 1.0 になるか確かめます。

**Q. テキスト検索で黄色いボールが当たらない。**
A. 想定内です。合成のフラット画像では生のパッチトークンとテキストの整列が弱く、スコアは 0.22〜0.25 に団子になります。**機構の確認には画像領域クエリ**（ある領域の代表ベクトルで検索 → 自分が rank-1）を使ってください（`05` の image-query）。実写ではテキスト検索が領域を当てます。

**Q. `AgglomerativeClustering` が遅い / メモリを食う。**
A. パッチ数（H×W）を増やしすぎです。7×7〜14×14 に抑えます。O(n²) メモリなので解像度を上げる前にダウンサンプルします。

**Q. FAISS の検索で落ちる / 結果が変。**
A. `np.ascontiguousarray(x, dtype=np.float32)`（`cc_common.as_faiss`）を通したか、IP なのに正規化を忘れていないか、`d`（次元）が add と search で一致しているかを確認します。`normalize_L2` は in-place で元配列を壊すので注意。

**Q. `multiprocessing` で固まる / 子プロセスが torch でこける。**
A. 本章は `mp.get_context("spawn")` を使い、子プロセス内でエンコーダをロードします（fork + torch のスレッド衝突を避けるため）。エントリは必ず `if __name__ == "__main__":` でガードします。終端の POISON_PILL を流し忘れると writer が `get()` で永久待ちになります。

**Q. CLIP の重みがダウンロードできない環境。**
A. `cc_common.load_encoder` は CLIP のロードに失敗すると、決定論的なフォールバック特徴抽出器に自動で切り替えます（機構の確認用。意味的な検索精度は出ません）。本番では常に CLIP が使われます。

---

## 🚀 発展トピック・参考

- **MaskCLIP / dense CLIP の整列改善**: 最後の自己注意の value 射影だけを使うと、パッチトークンがテキストにより整列します。参照リポの ResNet 経路（`attnpool` の `v_proj`/`c_proj` を 1×1 conv 化）はこの発想に近いものです（`build/models.py: dense_clip_embeddings_resnet`）。
- **適応サンプリング（AFS-MI）**: 参照リポはヒストグラム差分や相互情報量でフレームを間引き、似たフレームの無駄な推論を避けます（`build/producer.py`、`stream/capture.py`）。
- **カバレッジ評価**: GT bbox とクラスタマスクの被覆率で P@k / NDCG を測る評価系（`eval/coverage.py`）。本章の `coverage_ratio` / 演習 Q10 がその最小版です。
- **エッジ展開**: faiss-cpu と TorchScript 化した小型 CLIP で Jetson まで運ぶ設計（`docs/EDGE_DEPLOYMENT.md`）。35〜37 章（量子化・ONNX・エッジ最適化）とつながります。
- 参照リポ: `cluster-clip`（`README.md`、`src/adaptive_cluster_clip/build/models.py`、`.../indexer.py`、`.../search/engine.py`、`.../stream/pipeline.py`）。

関連章: 16（CLIP ゼロショット）/ 17（FAISS 画像検索）/ 33・42（マルチモーダル / ベクトル検索）/ 41（Cluster-CLIP パイプライン）。

---

## ▶ 動かし方

```bash
# 本編（dense -> connectivity -> clustering -> retrieval）
uv run python lectures/40_cluster_clip_dense_cluster/01_dense_vs_global_clip.py
uv run python lectures/40_cluster_clip_dense_cluster/02_dense_features_extraction.py
uv run python lectures/40_cluster_clip_dense_cluster/03_spatial_connectivity.py
uv run python lectures/40_cluster_clip_dense_cluster/04_agglomerative_region_cluster.py
uv run python lectures/40_cluster_clip_dense_cluster/05_region_retrieval.py

# 章末ミニプロジェクト（Split -> Build -> Search -> Stream）
uv run python lectures/40_cluster_clip_dense_cluster/mini_project.py

# 演習（10問・自己採点）と模範解答
uv run python lectures/40_cluster_clip_dense_cluster/exercises.py
uv run python lectures/40_cluster_clip_dense_cluster/exercises_solutions.py
```

出力画像は `outputs/40_cluster_clip_dense_cluster/` に保存されます（matplotlib は Agg バックエンド、`imshow` は呼びません。BGR↔RGB の変換に注意）。

---

版: torch 2.12+cpu / open-clip-torch 3.3 / faiss-cpu 1.14 / scikit-learn 1.9 ・ 2026-06 ・ CPU 前提（`model.eval()` + `inference_mode()`）
