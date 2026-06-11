# 第17回 FAISSベクトルDBと画像検索システム（評価込み）

> トラック: 埋め込み・検索 ／ レベル: 中級 ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`
> （`uv sync --group dl --group hf --group vector --group metrics`）

## 🎯 この章のゴール

この章を終えたとき、あなたは「画像検索システム」を構成要素に分解して自分の手で組み立てられるようになります。画像検索とは煎じ詰めれば、(1) 画像をモデルでベクトル（埋め込み）に変換し、(2) そのベクトル群を高速に近傍探索できる索引（インデックス）に格納し、(3) クエリも同じ手順でベクトル化して「似ているもの」を引いてくる、という3段のパイプラインです。本章ではその中核を担う **FAISS**（Facebook AI Similarity Search）を、最も単純な総当たり検索から、大規模化に耐える近似最近傍（ANN）まで、段階的に手を動かして習得します。

特に重視するのは「正しさを自分で測れること」です。ANN は速い代わりに答えが近似なので、どれだけ厳密解からズレているかを **Recall@k** で定量化し、**QPS（毎秒クエリ数）と Recall のトレードオフ曲線**を自分で描けるようにします。評価の鉄則は一つで、**正解（ground truth）は必ず厳密検索 `IndexFlat` で作る**こと。ANN 自身の結果を正解にしてしまうと「自分を自分で採点して満点」になり、評価が無意味になります。

到達点を一言でいえば、**「埋め込み → 正規化 → add → search → 永続化 → 評価」という一連を、AI 補助なしでそらで書け、コサイン類似度のための正規化や `float32`・C連続といった FAISS の作法、IVF/HNSW/PQ の使い分けを自分の言葉で説明できる**こと。最終的には CLIP の埋め込みを使って「テキストで画像を検索する」マルチモーダル検索まで通します。

---

## 1. ベクトル検索とは — 近傍探索（NN）と近似近傍探索（ANN）

ベクトル検索の出発点は「似ているもの＝ベクトル空間で近いもの」という考え方です。画像やテキストを固定長のベクトル（埋め込み）に変換しておけば、「似た画像を探す」という曖昧な問いが「クエリベクトルに距離が近い点を探す」という明確な数学の問題に変わります。これを厳密に解くのが **最近傍探索（NN: Nearest Neighbor）** で、全データとの距離を総当たりで計算します。答えは常に正確ですが、データ数 N に比例して遅くなります（O(N)）。

データが数十万・数百万件に増えると総当たりは現実的でなくなります。そこで登場するのが **近似最近傍探索（ANN: Approximate NN）** です。空間をあらかじめ区切ったり（IVF）、近傍をたどるグラフを作ったり（HNSW）、ベクトルを圧縮したり（PQ）して、「ほぼ正しい答えを、桁違いに速く」返します。ANN は精度をいくらか犠牲にして速度とメモリを稼ぐ技術で、その「いくらか」を測って管理するのがエンジニアの仕事になります。

FAISS はこの NN/ANN の索引を多数そろえたライブラリで、**CPU だけで全機能（Flat/IVF/HNSW/PQ）が動きます**。本章は `faiss-cpu` 1.14.2 を前提に、すべて CPU で完結させます。まずは全体像として「Flat＝厳密だが遅い／ANN＝近似だが速い」「両者は同じ `add`/`search` の API で切り替えられる」という二点を頭に入れておけば、以降の各 index は同じ枠組みのバリエーションとして理解できます。

## 2. FAISS の基本ループと「float32・C連続」の鉄則

FAISS のコードはどの index でも同じ形をしています。`index = faiss.IndexXxx(d)` で次元 `d` の索引を作り、`index.add(xb)` でデータベースのベクトル群を入れ、`index.search(xq, k)` でクエリごとに上位 `k` 件を引く——これだけです。`search` は **距離 `D` と ID `I` の2つの行列**（ともに形 `(クエリ数, k)`）を返します。FAISS の世界では公式ドキュメントに倣って戻り値を `D, I` と書くのが慣例なので、本講座もその名前を使います。

唯一にして最大の落とし穴が入力配列の型です。**FAISS に渡す配列は必ず `float32` かつ C連続（row-major）でなければなりません**。Python で何気なく作る配列は `float64` だったり、スライスや転置でメモリが非連続だったりします。これを `search` に渡すと、静かに誤動作したり例外で落ちたりします。だからこそ、index に渡す直前に必ず `np.ascontiguousarray(x, dtype=np.float32)` を通すのを手癖にします。本章のヘルパ `as_faiss_array` がまさにこれです。

```python
import faiss, numpy as np

xb = np.ascontiguousarray(xb, dtype=np.float32)  # float32・C連続を保証
index = faiss.IndexFlatL2(d)   # L2距離の総当たり索引
index.add(xb)                  # DBを格納（Flatはtrain不要）
D, I = index.search(xq, k)     # D:距離(nq,k), I:ID(nq,k)
```

上のコードで覚えてほしいのは、`IndexFlatL2` は学習（train）が要らず `add` した瞬間から検索できること、そして `I[q]` の先頭にはクエリ自身（DBに含まれていれば距離0）が来ること、です。`01_flat_ip_cosine.py` を実行すると、クエリ0の近傍IDの先頭が `0`、その L2 距離が `0.0` になることを確認できます。「自分が自分の最近傍になる」のは、パイプラインが正しく繋がっている何よりの証拠なので、動作確認の定番テクニックとして使ってください。

## 3. L2距離 と 内積、そして「正規化でコサイン」

FAISS の距離尺度には大きく2系統あります。`IndexFlatL2` は **L2距離（ユークリッド距離）** で「小さいほど近い」、`IndexFlatIP` は **内積（Inner Product）** で「大きいほど近い」です。ソートの向きが逆である点（L2は昇順、IPは降順が「近い順」）を最初に意識しないと、スコアの閾値処理や上位選択で取り違えます。下の表に整理します。

| 索引 | 尺度 | 「近い」の向き | コサインにするには |
| --- | --- | --- | --- |
| `IndexFlatL2(d)` | L2距離 ‖a−b‖² | 小さいほど近い（昇順） | 両方を正規化すれば順位はIPと一致 |
| `IndexFlatIP(d)` | 内積 a·b | 大きいほど近い（降順） | **両方を L2 正規化**すると内積＝コサイン |

この表の右列が本章で最も重要なポイントです。実務で使いたいのはたいてい **コサイン類似度**（ベクトルの「向き」の近さ、長さに依存しない）ですが、FAISS にはコサイン専用の index がありません。代わりに、**ベクトルを L2 正規化（ノルムを1にする）してから `IndexFlatIP`（内積）を使う**と、内積がそのままコサイン類似度になります。鍵は「DB側とクエリ側の両方を正規化する」こと。片方だけだと崩れます。

```python
faiss.normalize_L2(xb)          # ← in-place！ xb 自身が書き換わる（破壊的）
index = faiss.IndexFlatIP(d)
index.add(xb)
faiss.normalize_L2(xq)
D, I = index.search(xq, k)      # D はコサイン類似度（-1〜1, 1が最も似ている）
```

ここで `faiss.normalize_L2` は **引数の配列をその場で（in-place）書き換える**点に注意してください。元の未正規化ベクトルを後で使いたいなら、コピーしてから渡すか、本章ヘルパの非破壊版 `l2_normalize`（新しい配列を返す）を使います。`01_flat_ip_cosine.py` では、未正規化のまま `IndexFlatIP` を使った「素の内積」検索と、正規化後の「コサイン」検索を並べて出力し、正規化するとクエリ自身とのスコアがちょうど `≈1.0` になることを見せています。さらに、正規化後は L2 と IP で近傍の集合が一致する（並び方向だけ逆）ことも確認できます。

## 4. ID の管理 — `IndexIDMap` とメタデータの別管理（SQLite）

ここから実際の画像検索に踏み込みます。素の `IndexFlat` に `add` すると、各ベクトルには `0..N-1` の連番が機械的に振られるだけです。しかし実務では「このベクトルは画像 `cat_001.jpg` のもの」「あのベクトルは商品ID 8675309」のように、**自分で決めた ID を紐づけたい**ことがほとんどです。そこで `IndexIDMap`（または ID からベクトルを復元できる上位版 `IndexIDMap2`）で素の index を包み、`add_with_ids(vectors, ids)` で任意の `int64` ID を割り当てます。

重要なのは、**FAISS が覚えてくれるのは「ベクトルと ID」だけ**で、画像パスやキャプションといったメタデータは持てない、という割り切りです。メタデータは SQLite や辞書など FAISS の外で管理し、検索結果の ID から引く——これが実運用の定石です。下のコードでは ID→ラベルの対応表を SQLite に作っています（`label` 列に実際の画像パスやサムネイル、撮影日時などを入れるイメージです）。

```python
ids = np.arange(1000, 1000 + len(images)).astype(np.int64)  # 連番でない任意ID
index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
index.add_with_ids(db_vecs, ids)            # ベクトルとIDだけFAISSへ

con = sqlite3.connect(db_path)              # メタデータは別DBで管理
con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
con.executemany("INSERT INTO items VALUES (?, ?)", zip(ids.tolist(), labels))
```

`02_idmap_persist_sqlite.py` はこの構成で、合成画像（赤い円・青い四角など色×形の24枚）を CLIP で埋め込み、`add_with_ids` で 1000 番台の ID を振って格納します。検索後は得られた ID で SQLite を引き、「赤い円」というラベルに変換して表示します。FAISS の ID とアプリのメタデータを分離して管理する感覚——これが小さな検索エンジンを自作するときの背骨になります。

## 5. 永続化 — `write_index` / `read_index` とメタDBの「セット保存」

作った index は毎回ゼロから作り直すのではなく、ファイルに保存して再利用します。FAISS では `faiss.write_index(index, "x.faiss")` で保存し、`faiss.read_index("x.faiss")` で読み戻します（バイト列にしたいときは `serialize_index`/`deserialize_index`）。読み戻した index の `ntotal`（登録件数）が保存前と一致すれば、永続化は成功です。アプリの再起動をまたいでも索引を引き継げるようになります。

ただしここに見落としやすい罠があります。**`write_index` が保存するのは index 本体（ベクトルと内部ID）だけで、別管理しているメタDB（ID→画像パス）は保存しません**。したがって `.faiss` ファイルと `.db`（あるいは `.json`）を必ず**セットで、バージョンを揃えて**永続化する必要があります。片方だけ更新して取り違えると、検索は通るのに ID からメタデータが引けない、という不整合が起きます。

```python
faiss.write_index(index, "image_index.faiss")   # ベクトル＋内部IDを保存
index = faiss.read_index("image_index.faiss")    # 別オブジェクトとして復元
assert index.ntotal == 24                         # 件数一致＝永続化OK
# 注意: メタDB(image_meta.db) は別途セットで保存・復元すること！
```

`02_idmap_persist_sqlite.py` では index を保存→まっさらな状態から読み戻し（アプリ再起動を模す）→そのまま検索、という流れを通します。「索引ファイルとメタデータDBは一心同体」という運用上の感覚を、ここで体に入れてください。これは後半の Cluster-CLIP 統合（第41回）で、ストリーム処理しながら `.faiss` と SQLite を整合させる設計にそのまま繋がります。

## 6. テキスト→画像のマルチモーダル検索（CLIP / transformers v5 の注意）

埋め込みを CLIP（`openai/clip-vit-base-patch32`）で作ると、画像とテキストが**同じ潜在空間**に置かれます。これが効くと「`a photo of a red circle` というテキストで、赤い円の画像を検索する」というマルチモーダル検索ができます。やることは画像のときと同じで、テキストを埋め込み→正規化→同じ index に対して `search` するだけ。FAISS から見ればクエリがテキスト由来か画像由来かは関係なく、ただのベクトルです。

ここで **transformers v5（5.11）の破壊的変更**を2つ押さえます。1つ目は画像プロセッサ。`AutoFeatureExtractor` は廃止され、画像は **`AutoImageProcessor`（または `AutoProcessor`）** を使います。プロセッサは torchvision バックエンドの fast 実装のみになったため、画像モデルを使うなら **torchvision が事実上必須**です（入れ忘れると `AutoImageProcessor` がエラーになります）。2つ目は埋め込みの取り出し方です。

```python
from transformers import CLIPModel, AutoProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc  = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.inference_mode():                       # 推論は勾配を切る（CPUでも必須）
    inp  = proc(images=imgs, return_tensors="pt").to(device)
    out  = model.get_image_features(**inp)         # v5: 返り値はオブジェクト！
    feats = out.pooler_output                       # 射影ベクトル(未正規化, (n,512))
emb = torch.nn.functional.normalize(feats, p=2, dim=-1)  # 検索前に必ず正規化
```

v5 では `get_image_features` / `get_text_features` の戻り値が **`BaseModelOutputWithPooling` オブジェクトに変わり**、射影後ベクトルは `.pooler_output` に入っています（古いコードのように戻り値を直接テンソル扱いすると `'...' object has no attribute 'shape'` で落ちます）。しかもこの `.pooler_output` は **L2 正規化されていません**（ノルムは約11.7）。一方 `model(**inputs).logits_per_image` の経路は内部で正規化済み、という非対称があります。検索やコサイン類似度の前には **必ず自分で `F.normalize`** してください。これが CLIP まわりで最頻出の落とし穴です。`02_idmap_persist_sqlite.py` を実行すると、`a photo of a red circle`→赤い円、`a blue square`→青い四角、`a green triangle`→緑い三角、と妥当に当たることが確認できます（CLIP を取得できないオフライン環境では自動で色記述子にフォールバックし、テキスト検索だけスキップして完走します）。

## 7. 近似最近傍 — IVFFlat / HNSW / IVFPQ・OPQ と `index_factory`

データが増えてきたら ANN に切り替えます。代表的な3系統を押さえれば実務はだいたい回ります。**IVFFlat** は空間を `nlist` 個のセルに区切り、クエリに近い `nprobe` 個のセルだけを探す「転置インデックス」方式です。`add` の前に代表データでセル中心を学習する **`train` が必須**で、これを忘れて `add` すると例外になります。検索時の `nprobe` が精度↔速度のダイヤルで、`nprobe=1` のまま使うと Recall が大きく落ちる（既定の罠）ので必ず調整します。

**HNSW** は近傍グラフをたどる方式で、**train 不要**・高速・高精度ですがメモリを多く食います。グラフの次数 `M`、構築品質 `efConstruction`、検索幅 `efSearch` を持ち、検索時の主ダイヤルは `efSearch` です。**PQ（Product Quantization）** はベクトルを部分ベクトルに割って量子化し、メモリを劇的に削減する圧縮技術です（精度と引き換え）。これらは文字列レシピで組み立てる **`index_factory`**（例 `"IVF100,PQ8"`, `"OPQ8_64,IVF100,PQ8"`, `"HNSW32"`）が便利です。

| 索引 | train | 主ダイヤル | 長所 | 注意 |
| --- | --- | --- | --- | --- |
| `IndexFlatIP/L2` | 不要 | — | 厳密・実装単純 | O(N) で遅い／評価の正解作りに使う |
| `IndexIVFFlat` | **必須** | `nprobe` | 速い・メモリ普通 | `nprobe=1` は低Recall。`nlist`目安 4√N〜16√N |
| `IndexHNSWFlat` | 不要 | `efSearch` | 高速・高精度 | メモリ大・削除に弱い |
| `IVFPQ`/`OPQ`(factory) | **必須** | `nprobe` | メモリ激減 | 精度低下。十分な学習データが必要 |

`03_ivf_hnsw_pq.py` は 1万件・64次元のデータで上記を実際に構築・検索し、Recall とファイルサイズ（≒メモリ量）を比べます。例えば `IVF100,PQ8` は 1ベクトルを 8 バイトに圧縮し（`float32` 生の `64×4=256` バイトの1/32）、保存ファイルも `IVF100,Flat` の約1/10になります。代償として Recall@10 は約0.45まで落ちます——これが「メモリと精度のトレードオフ」の生の数字です。スクリプトは `nprobe` を上げると IVF の Recall が `0.79→1.0` と改善する様子も出力するので、ダイヤルの効きを体感してください。

## 8. 評価 — Recall@k の自前計算・QPS-recall曲線・retrieval mAP

ANN の良し悪しは「どれだけ厳密解に近いか」で測ります。その指標が **Recall@k** で、定義は「クエリごとに、厳密検索の上位k件のうち ANN の上位k件に入っていた割合」、これを全クエリで平均したものです。計算は集合の一致で素直に書けます。**正解（ground truth）は必ず厳密な `IndexFlat` で作る**のが鉄則で、ANN 自身の結果を正解にすると評価が無意味になります（高く見えるだけ）。

```python
_, gt = flat.search(xq, k)          # ground truth は厳密Flatで作る（重要）
_, ann = ivf.search(xq, k)          # 評価したいANNの結果
def recall_at_k(gt, ann, k):
    return np.mean([np.intersect1d(g[:k], a[:k]).size / k
                    for g, a in zip(gt, ann)])
# np.intersect1d は -1（近傍不足の埋め草）を自然に無視してくれる
```

上の `recall_at_k` で `np.intersect1d` を使うのがコツで、`search` が返す `-1`（近傍が足りないときの埋め草）が混ざっても誤カウントせず無視できます。`04_recall_qps_eval.py` は `nprobe`（IVF）と `efSearch`（HNSW）をスイープし、各設定の Recall@10 と **QPS（毎秒クエリ数）**を測って、**横軸QPS（対数）・縦軸Recall の曲線**を `04_qps_recall_curve.png` に描きます。右上ほど「速くて正確」で良い設定です。QPS は1回計測だとブレるので、ウォームアップ＋複数回平均で安定させています。

最後に、ラベル付きデータでは **retrieval mAP**（同じラベルを正例とした平均適合率）も測ります。Recall@k が「ANN が厳密検索にどれだけ忠実か」を見るのに対し、mAP は「同じカテゴリの仲間を上位にどれだけ集められるか＝埋め込み自体の良さ」を見る別観点の指標です（観点が違うので両方見ます）。本章の合成データでは mAP@10 ≈ 0.999 と、クラスタ構造がきれいに引けていることが数字で確認できます。これらの結果は `04_eval_report.json` にも保存され、後から再現・比較できます。

## 9. 実運用の作法 — `-1` ガードとインクリメンタル更新

実運用でまず効くのが `-1` ガードです。`k` が登録件数 `ntotal` より大きいときや、IVF で十分な近傍が見つからないとき、`search` の返す ID 行列 `I` には `-1` が混じります。これをそのまま「ID」としてメタDB参照に使うとクラッシュや誤参照になるので、**`-1` を弾く（`(none)` 等に置換する）処理を必ず入れます**。`02_idmap_persist_sqlite.py` の `lookup_labels` は `i < 0` を早期 return でガードしています。

```python
for i in result_ids:
    i = int(i)
    if i < 0:                 # -1 ガード（最初に弾く）
        labels.append("(none)")
        continue
    labels.append(lookup(i))  # 正常なIDだけメタDBを引く
```

もう一つ実務で頻出するのが **インクリメンタル更新**です。`IndexIDMap` に `add_with_ids` で逐次追加すれば、検索可能な状態を保ったままデータを増やせます。定期的に `write_index` で永続化し、メタDBと整合させる——これが画像が流れ込み続けるストリーム処理の基本設計です（参照リポジトリ Cluster-CLIP の `stream/writer.py` がまさにこの実例で、第40・41回で扱います）。ただし IVF/PQ 系は最初に学習したセル中心が前提なので、データ分布が大きく変わったら `train` のやり直し（再構築）が必要になる点は覚えておきましょう。

## 10. CPU / GPU と faiss のインストールの注意

本章は `faiss-cpu` だけで Flat/IVF/HNSW/PQ の全機能・保存・評価が CPU で完結します（MacBook を含む CPU のみ環境でも問題なく動きます）。検索の挙動と API は CPU/GPU で同一なので、学習目的では CPU で十分です。GPU に載せ替えたいときも `index = faiss.index_cpu_to_gpu(res, 0, index)` の一行を足すだけで、検索コードは変わりません。

インストールで紛らわしいのが GPU 版です。**`pip install faiss-gpu` というパッケージ名は（現在）存在しません**。GPU 版は `faiss-gpu-cuvs` という名前で、しかも **Linux x86_64 + NVIDIA（CUDA 12.4 系）限定**、PyPI からは `--extra-index-url https://pypi.nvidia.com` 経由、という制約があります。さらに `faiss-cpu` と `faiss-gpu-cuvs` は**同じ `import faiss` 名前空間を奪い合うため、同一環境に共存させてはいけません**。本講座では `vector` グループに `faiss-cpu` のみを入れ、GPU は「手元に対応GPUがあれば試す」補足に留めています。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で、上から順に読むと「FAISS の基礎 → 実画像検索 → ANN → 評価」と理解が積み上がります。すべて `outputs/17_faiss_image_search/` に結果（PNG/JSON/.faiss）を保存し、画面表示には依存しません。共通処理（出力先・正規化・Recall・合成データ・CLIP埋め込み）は `search_helpers.py` にまとめ、各スクリプトが import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `search_helpers.py` | 出力先/デバイス判定・`as_faiss_array`/`l2_normalize`・`recall_at_k`/AP・合成ベクトル/画像・CLIP埋め込み(`Embedder`、色記述子フォールバック付) |
| `01_flat_ip_cosine.py` | FAISS 基本ループ、`float32`・C連続、L2 vs IP、正規化でコサイン、`D,I` の読み方 |
| `02_idmap_persist_sqlite.py` | CLIP埋め込み→正規化→`IndexIDMap2`+`add_with_ids`→SQLite→`write/read_index`→画像/テキスト検索→`-1`ガード |
| `03_ivf_hnsw_pq.py` | `IVFFlat`(train/nprobe)・`HNSW`(efSearch)・`index_factory`(IVFPQ/OPQ)、Recall とメモリ比較 |
| `04_recall_qps_eval.py` | Recall@k 自前計算、`nprobe`/`efSearch` スイープの QPS-recall 曲線、retrieval mAP、JSON保存 |
| `exercises.py` | TODO 形式の演習（自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |

`search_helpers.py` だけは「読み物」ではなく「再利用する道具」です。トップレベルでは torch を import せず、CLIP が要る `Embedder` の中だけで遅延 import している点（FAISS だけ学ぶ 01/03/04 を軽くするため）に注目すると、依存を絞る設計の意図が伝わります。

## 12. 動かし方

このモジュールは `faiss-cpu`（vector）に加え、`02` で CLIP を使うため `dl`/`hf` グループが要ります。評価で `metrics` も入れておきます。入力画像は合成生成されるので、いきなり実行できます（`data/17_faiss_image_search/` に画像を置けば、そちらが優先して使われます）。プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）
uv sync --group dl --group hf --group vector --group metrics

# 各スクリプトを実行（結果は outputs/17_faiss_image_search/ に保存される）
uv run python lectures/17_faiss_image_search/01_flat_ip_cosine.py
uv run python lectures/17_faiss_image_search/02_idmap_persist_sqlite.py   # 初回はCLIP重みDL
uv run python lectures/17_faiss_image_search/03_ivf_hnsw_pq.py
uv run python lectures/17_faiss_image_search/04_recall_qps_eval.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL、でも exit 0）
uv run python lectures/17_faiss_image_search/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/17_faiss_image_search/exercises.py
```

実行後は `outputs/17_faiss_image_search/` の図を確認してください。`01_flat_scores.png`（IP素/コサイン/L2 のスコアの並び）、`02_query_results.png`（クエリ画像と類似画像）、`03_ann_tradeoff.png`（精度↔速度の散布図）、`04_qps_recall_curve.png`（スイープ曲線）と `04_eval_report.json`（数値）が出ます。`02` で CLIP を取得できない環境では色記述子に自動フォールバックし、テキスト検索だけスキップして完走します（画像→画像検索は動きます）。

## 13. よくあるエラーと対処（チェックリスト）

最後に、この章で詰まりやすい点を「症状 → 原因 → 対処」でまとめます。多くは FAISS の入力作法か、正規化忘れ、評価の作り方に集約されます。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `search` が落ちる/結果が変 | 配列が `float32`・C連続でない | `np.ascontiguousarray(x, dtype=np.float32)`（`as_faiss_array`）を通す |
| コサインのつもりが順位が崩れる | 正規化忘れ（or 片側だけ） | DB・クエリ両方を `normalize_L2`/`l2_normalize` |
| 正規化したのに元配列が壊れた | `faiss.normalize_L2` は in-place | コピーしてから渡す、または非破壊版を使う |
| `add` で例外 | IVF/PQ を `train` せず `add` | `index.train(代表データ)` を先に呼ぶ |
| IVF の Recall が異常に低い | `nprobe=1` のまま | `nprobe` を上げる（5〜数十） |
| メタDB参照でクラッシュ | 結果 `I` の `-1` をIDに使った | `i < 0` を弾く（`-1` ガード） |
| ANN の評価が満点で怪しい | ground truth を ANN 自身で作った | 正解は必ず `IndexFlat`(厳密)で作る |
| `'...Pooling' object has no attribute 'shape'` | v5: `get_image_features` の戻り値はオブジェクト | `.pooler_output` を取り、`F.normalize` する |
| `AutoImageProcessor` がエラー | torchvision 未導入（v5 は fast 実装のみ） | `dl` グループ（torchvision）を入れる |
| `pip install faiss-gpu` が失敗 | そのパッケージ名は無い | CPUは `faiss-cpu`。GPUは `faiss-gpu-cuvs`(Linux+NVIDIA限定・cpuと排他) |

この表の項目を自分で説明でき、回避コードを書けるようになれば、本章のゴールに到達しています。特に「`float32`・C連続」「両側を正規化」「正解は Flat で作る」の3つは、FAISS を使い続ける限り何度も効いてくる原則です。

## 14. まとめ

本章では、画像検索を「埋め込み → 正規化 → add → search → 永続化 → 評価」に分解し、FAISS の基本ループ、`float32`・C連続の鉄則、L2/IP と「正規化でコサイン」、`IndexIDMap` とメタデータ別管理、`write/read_index` のセット永続化、CLIP によるテキスト→画像のマルチモーダル検索、IVF/HNSW/PQ の使い分け、そして Recall@k・QPS-recall曲線・retrieval mAP による定量評価までを、すべて自分の手で動かしました。

次回以降の検出・セグメンテーションでも、また最終応用の Cluster-CLIP（第40・41回）でも、この「埋め込みをベクトルDBに入れて検索・評価する」骨格は繰り返し登場します。`search_helpers.py` のような自分用の道具を一つ持っておくと、以降は定型処理に煩わされず本質に集中できます。まずは演習を全問 PASS させ、正規化と評価の感覚を体に入れてから次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ faiss-cpu 1.14.2 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub（transformers 依存）／ scikit-learn 1.9.0 ／ torchmetrics 1.9.0 ／ matplotlib 3.10 系。
> モデル: `openai/clip-vit-base-patch32`（初回のみ HuggingFace Hub から重みDL→ローカルキャッシュ）。GPU版 FAISS は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA, CUDA 12.4 系限定、`faiss-cpu` と排他）。