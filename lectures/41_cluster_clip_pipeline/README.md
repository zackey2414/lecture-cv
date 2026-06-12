# 41_cluster_clip_pipeline: Cluster-CLIP 総仕上げ — Split → Build → Search → Stream

> トラック: **応用・統合（capstone）** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `embed` `vector` `metrics`
> 前提モジュール: `40_cluster_clip_dense_cluster`（dense CLIP + 空間連結クラスタリング）／ `16_clip_zeroshot_retrieval`（CLIP の正準フロー）／ `17_faiss_image_search`（FAISS と SQLite メタ管理）／ `11_realtime_stream`（multiprocessing ストリーム）
> 題材: **Cluster-CLIP**（dense CLIP + 空間連結クラスタリング + FAISS + ストリームの手法。本章はその CPU 小型再構築）

---

## 🎯 この章のゴール

この章は講座の**総仕上げ**だ。これまで別々に学んだ「画像 I/O → テンソル → ViT/CLIP 埋め込み → クラスタリング → FAISS 近傍探索 → SQLite メタ管理 → multiprocessing ストリーム」を、**1 本のパイプライン**に束ねる。題材は **Cluster-CLIP**（テキストで動画フレームを**開語彙＝オープンボキャブラリ**に検索するシステム）で、その設計に忠実な CPU 完結版を自分の手で再構築する。

具体的には次ができるようになる。

- **Split**: 動画を連番フレームに分割する（`cv2.VideoCapture` のループ、None/False ガード）。
- **Build**: 各フレームから **dense CLIP 特徴（パッチ単位埋め込み）** を取り出し、**空間連結ありの凝集型クラスタリング**で「意味の似た領域」に分割、各領域の**代表ベクトル**を作って **FAISS(IndexFlatIP+IDMap)** に登録、`faiss_id ↔ (frame, cluster)` を **SQLite** に保存する。
- **Search**: テキストクエリ（CLIP text）や**領域クエリ（画像）**で FAISS を引き、ヒットした**領域マスクを重畳して可視化**する。「画像全体 1 ベクトル」検索との違い（どこが効いたかを局在化できる）を説明できる。
- **Stream**: `multiprocessing` で**取得（軽い）と推論（CPU バウンドで重い）を別プロセス**に分け、キュー満杯時はフレームを**ドロップしてリアルタイム性を守る**。ヒストグラム差分の**適応サンプリング**で場面転換だけを残す。
- これら全体を `mini_project.py` で一気通貫に動かし、各部品が参考実装のどのモジュールに対応するかを言える。

本章のスクリプトはすべて **CPU・`model.eval()` + `inference_mode()`** で動き、入力は合成の小さな「動画」（色つきパネルが場面ごとに入れ替わる連番フレーム）だ。学習はしない（推論＋クラスタリング＋検索のみ）。

> **合成データの正直な注意**: 本章のフレームは平坦な色領域の合成画像なので、**テキスト→領域**の意味整合はノイジー（CLIP のパッチ特徴は自己注意で文脈が混ざるため、玩具データでは色テキストと領域が綺麗に対応しない）。一方、**領域→領域**（画像クエリ）検索は埋め込みが素直に効くので安定して当たる。実写動画では両方とも実用的になる。この差を体感するのも狙いの一つだ。

---


## 1. 直感 — なぜ「画像全体 1 ベクトル」ではなく「領域（dense）」なのか

16・17 章でやった CLIP 画像検索は、1 枚の画像を **1 本のベクトル**（`encode_image` が返す CLS 埋め込み）に潰してから FAISS に入れていた。これは「この**画像**は猫っぽいか」には強いが、「この画像の**どこに**小さな標識があるか」「複数物体のうち**どれ**がバスか」には弱い。画像全体の平均的な意味に薄まってしまい、画面の片隅にある小物体は埋もれる。動画フレームのように 1 枚に複数の物体が散らばっている場面では、これが致命的になる。

Cluster-CLIP の発想は単純だ。**画像を 1 ベクトルに潰す前に、領域に分けてから領域ごとにベクトル化する**。CLIP の visual encoder は内部で画像を 7×7（ViT-B-32 の場合）のパッチに分け、各パッチにトークン埋め込みを持っている。普段は最後に CLS トークンだけ使うが、**パッチトークン列を捨てずに空間グリッドに並べ直す**と、画像の「どの場所が何か」を持った密な特徴マップ `[C, H, W]` が得られる。これが **dense CLIP 特徴**だ（40 章で詳しくやった）。

ただしパッチ 49 個をそのまま FAISS に入れると、1 フレームあたり 49 ベクトルでインデックスが膨れるうえ、隣接パッチはほぼ同じ意味で冗長だ。そこで**空間的に隣り合う・意味の似たパッチをまとめて領域にし、その代表ベクトル（平均→正規化）だけを索引化する**。これで「小物体・複数物体を、領域単位で開語彙検索できる」という Cluster-CLIP の核ができる。本章はこの考えを Split→Build→Search→Stream の流れに乗せて完成させる。

## 2. 理論 — dense CLIP 特徴の取り出し（ViT を手で展開する）

`model.encode_image()` は CLS トークンしか返さないので、パッチ特徴を得るには **visual transformer を手で forward 展開**する必要がある。流れはこうだ（`cluster_clip_helpers.dense_clip_embeddings`）。`visual.conv1` で画像をパッチ埋め込み `[B, width, gh, gw]` にし、平坦化して `[B, gh*gw, width]` に。先頭に `class_embedding`（CLS）を連結し、`positional_embedding` を足して `ln_pre` を通す。transformer は `(seq, batch, dim)` 並びを期待するので `permute(1,0,2)` してから通し、戻して `ln_post`。最後に **`visual.proj`（射影行列）を掛けて text と同じ共有埋め込み空間に揃える**。

ここで 2 つの肝がある。第一に、**CLS トークンを最後に捨てる**（`x[:, 1:, :]`）。CLS は画像全体の要約なので、パッチごとの局所特徴が欲しい dense 表現では邪魔になる。捨て忘れるとクラスタリングがノイズだらけになる、というのが 40 章でも出た典型的な落とし穴だ。第二に、**パッチごとに L2 正規化**してから返す。CLIP の類似度はコサインで測るので、ここで正規化しておくと後段の代表ベクトル計算や FAISS 内積がそのままコサインになる。

前処理も重要だ。標準の CLIP 前処理は `Resize(短辺) → CenterCrop(224)` で画像の端を切るが、Cluster-CLIP は **CenterCrop を排して正方形に強制 Resize** する（`load_clip`）。端を切ると画面端の小物体の dense 特徴が丸ごと消えてしまうからだ。アスペクト比は崩れるが、「端を捨てない」ことを優先する。これは `force_quick_gelu=True`（openai 重みは quick-gelu 活性化）とセットで、参考実装 `build/models.py::load_clip_model` と同じ勘所だ。

## 3. 理論 — 空間連結クラスタリングと代表ベクトル

dense 特徴 `[C, H, W]` を `[H*W, C]`（各パッチ＝1 サンプル）に並べ替え、`sklearn.cluster.AgglomerativeClustering` で `n_clusters` 個の領域に凝集する（`cluster_agglomerative`）。普通の凝集型クラスタリングは「意味が似ていれば画面の反対側のパッチでも併合」してしまい、領域がバラバラに散る。それを防ぐのが **`grid_to_graph` による connectivity 制約**だ。これは「隣接するパッチ同士の辺」だけを持つグラフで、これを `connectivity=` に渡すと、**空間的に隣り合うパッチ同士しか併合されない**。結果として、意味も位置も近いパッチがまとまり、画像が**連続した領域**に分かれる。

`linkage="ward"` + `metric="euclidean"` を使う。パッチ特徴は L2 正規化済みなので、ユークリッド距離はコサイン距離とほぼ同義になる（`‖a-b‖² = 2 - 2·cosθ`）。クラスタリングが終わったら、各クラスタに属するパッチ特徴の**平均を取り、もう一度 L2 正規化**して**代表ベクトル**にする。平均は「その領域の代表的な意味」を表し、正規化で FAISS の内積＝コサインに乗せる。空クラスタ（connectivity 制約下では基本起きないが念のため）は全体平均で代用する。

CPU で現実的に回すコツは**パッチ数を小さく保つ**ことだ。AgglomerativeClustering は `O(n²)` メモリなので、高解像度パッチ（14×14=196 や 16×16=256）だと重くなる。ViT-B-32 の 7×7=49 パッチなら一瞬で終わる。参考実装も同じ理由でパッチ数を抑えている。出力は代表ベクトル `reps [k, C]` とラベルマップ `cmap [H, W]`（各パッチがどのクラスタか）の 2 つで、`cmap` は後で領域マスクの可視化に使う。

## 4. 正準 API — Split → Build → Search → Stream の 4 段と参考実装対応

Cluster-CLIP は 4 つのステージからなる。本章はそれを `cluster_clip_helpers.py` の関数として実装し、番号付きスクリプトは**薄いドライバ**としてそれを呼ぶ（これは参考実装の `scripts/run_*.py` が `src/adaptive_cluster_clip/` を呼ぶ構造そのものだ）。対応表は次のとおり。

| ステージ | 本章のヘルパ関数 | 参考実装のモジュール | 役割 |
|---|---|---|---|
| **Split** | `run_split` | `split/processor.py` | 動画 → 連番フレーム JPEG |
| **Build** | `run_build` | `build/consumer.py` + `indexer.py` + `db_writer.py` | dense CLIP → 領域クラスタ → 代表ベクトル → FAISS + SQLite |
| **Search** | `search_index` / `overlay_cluster_mask` | `search/engine.py` + `visualizer.py` | クエリ → FAISS → SQLite → 領域マスク可視化 |
| **Stream** | `run_stream_pipeline` | `stream/pipeline.py` + `capture.py` + `consumer.py` + `writer.py` | 取得/推論/書込の 3 プロセス、ドロップ、適応サンプリング |

正準的な最小コードはこうなる。Build は、代表ベクトルを `faiss.IndexIDMap(faiss.IndexFlatIP(dim))` に `add_with_ids` し、`faiss_id` を SQLite の `VectorMapping(faiss_id, frame_id, cluster_idx)` に対応づける。**FAISS は連番の内部 ID しか持たない**ので、画像パス・クラスタ番号などのメタは別管理（SQLite）にし、検索結果の ID から JOIN で引く、というのが 17 章で学んだ実運用パターンだ。

```python
# Build（要点）: 代表ベクトルを FAISS に、メタを SQLite に
index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))   # コサイン類似 = 正規化 + 内積
index.add_with_ids(vectors.astype("float32"), ids.astype("int64"))
faiss.write_index(index, "index.faiss")

# Search（要点）: クエリ → FAISS → SQLite で -1 をガードしつつメタ引き
qv = encode_text(model, tokenizer, ["a red region"])     # L2 正規化済み [1, C]
D, I = index.search(np.ascontiguousarray(qv, "float32"), top_k)
for fid, score in zip(I[0], D[0]):
    if fid == -1:        # 近傍が足りないと -1 が来る → 使うとクラッシュ
        continue
    frame_id, cluster_idx = lookup_sqlite(fid)            # メタを引く
```

## 5. 実装を 1 つずつ — 01〜04 を読む

**`01_split_frames.py`（Split）**。合成動画を `cv2.VideoWriter`(mp4v) で作り、`cv2.VideoCapture` で開いてフレームを連番 JPEG に保存する。ここで叩き込むのは、`cap.isOpened()` が `False` を返す／`cap.read()` が `(False, None)` で終端を知らせる／`cv2.imread` は失敗時に**例外でなく `None`** を返す、という OpenCV の流儀だ。VideoWriter が使えない環境のためにフレーム直書きへフォールバックも入れてある。出力はフレームのモンタージュ図。

**`02_build_index.py`（Build）**。`load_clip` で ViT-B-32 をロードし、`run_build` で全フレームを dense CLIP → クラスタリング → 代表ベクトル化し、`vectors/*.npy`・`cluster_maps/*.npy`・`index.faiss`・`metadata.db` を生成する。生成後に **FAISS の `ntotal` と SQLite の件数が一致する**ことを確認し（不整合はバグの典型）、先頭 3 フレームについて「原画像 ｜ 色分けクラスタマップ ｜ 最大領域の重畳」を並べて領域分割の見え方を可視化する。`vectors = frames × clusters` という関係（12 フレーム×6 = 72 ベクトル）を必ず腹落ちさせること。

**`03_search_regions.py`（Search）**。3 種類の検索を比較する。(A) **テキスト→領域**: `encode_text` でクエリをベクトル化し `search_index` で領域を引く（Cluster-CLIP の看板機能。玩具データではノイジー）。(B) **領域→領域**: フレーム 0 の「赤い領域」の代表ベクトルをクエリにして、他フレームの赤領域を引く（**自己一致でスコア≈1.0**、上位が赤に偏る＝安定）。(C) **画像全体ベクトル**でのテキスト検索（ベースライン。「どのフレームか」までで「どの領域か」は出せない）。ヒット領域は `overlay_cluster_mask` で半透明赤の重畳にしてギャラリー化する。

**`04_stream_pipeline.py`（Stream）**。`run_stream_pipeline` を 2 シナリオで回す。シナリオ A は **passthrough + 小さいキュー**で、取得が推論を追い越すと `put_nowait` で**ドロップ**し実時間を守る様子（15 取得 / 9 ドロップ等）。シナリオ B は**ヒストグラム差分サンプラー**で、4 つの場面（ショット）の**変わり目だけをキーフレームとして残し**、24 枚中 20 枚（約 83%）の推論を省く。最後に、ストリームで逐次構築した index がそのまま検索可能であることを確認する。torch を載せるので各プロセスは **spawn** で独立起動し、`if __name__ == "__main__":` ガードが必須だ。

## 6. 落とし穴 — このパイプラインで実際にハマる所

最頻出は**色順とレイアウト**だ。OpenCV は BGR、Pillow/PyTorch/CLIP は RGB。フレームを `cv2.imread` で読んだら CLIP に渡す前に RGB へ、`matplotlib` で表示する前にも RGB へ変換する（忘れると色が反転し、推論精度が静かに落ちる）。テンソルは `(H,W,C)`↔`(C,H,W)` を取り違えない。dense 特徴の取り出しでは **CLS トークンの落とし忘れ**と**正規化の順序**が二大事故だ。

**FAISS 周り**も定番。コサインのつもりで `IndexFlatIP` を使いながら**正規化を忘れる**と単なる内積になり結果が崩れる。入力が `float32`・**C 連続**でないと落ちる（`np.ascontiguousarray(x.astype("float32"))`）。`search` の戻り値 `I` に **`-1`**（近傍不足）が混じることがあり、ID として使うと SQLite 参照でクラッシュするので必ずガードする。`IndexIDMap` を使わず `IndexFlat` に `add` すると ID は 0..N-1 固定になり、後からメタと対応づけられない。

**ヒストグラム差分サンプラー**には数学的な罠がある。各チャネルのヒストグラムを**別々に L2 正規化して連結**し `HISTCMP_INTERSECT` を取ると、交差値が 1 を超えて `delta = 1 - intersection` が負になり、`max(0, delta)` で**全部 0 にクランプされて 1 枚も間引けない**。本章では連結ヒストグラム全体を**和=1 の確率分布に L1 正規化**してから交差を取り、`delta ∈ [0,1]` を保証している（`HistogramDeltaSampler._compute_hist` のコメント参照）。**multiprocessing** では、torch を載せる子プロセスは `spawn` で起動し（fork は OpenMP デッドリスクがある）、モデルは**各子プロセスの中でロード**する。`spawn` は `__main__` ガードと「ターゲット関数がモジュールトップレベルで import 可能」であることを要求する。

## 7. 実務の使い分け — 領域 vs 全体、Flat vs ANN、サンプラー、GPU

**領域検索（Cluster-CLIP）と画像全体検索の使い分け**。「この画像は何の写真か」「似た雰囲気の画像」を探すなら**画像全体 1 ベクトル**で十分・高速だ。「画面のどこに小さな標識／特定の人物／特定の物体があるか」「複数物体を別々に引きたい」なら**領域検索**が要る。代償はベクトル数の増加（フレーム×クラスタ）と Build コストで、クラスタ数 `k` がそのトレードオフのダイヤルになる。`k` を増やすと細かい物体を拾えるが索引が膨れ、減らすと領域が粗くなる。

**FAISS のインデックス選択**。本章は厳密・学習不要の `IndexFlatIP` を使う（数千〜数万ベクトルの規模なら CPU で十分）。規模が上がったら `IndexIVFFlat`（`nlist`/`nprobe` で精度↔速度）や `IndexHNSWFlat`（`efSearch` が主ダイヤル）に差し替える——API（`add`/`search`）は同じで、`index_factory("IVF1024,Flat")` のように文字列で組める（17 章）。**サンプリング戦略**は、密に全フレーム処理するか、固定間引き（N 枚に 1 枚）か、適応（ヒストグラム差分で場面転換を検出）かを、検索の取りこぼし（recall）と計算コストのバランスで選ぶ。**GPU** は本講座では使わないが、コードは `cuda` 可用なら使う書き方（`pick_device`）にしてあり、FAISS は `index_cpu_to_gpu` の有無だけで切り替えられる（`faiss.get_num_gpus()` を try/except でガードする）。

---

## 🛠 章末ミニプロジェクト

`mini_project.py` は **Split → Build → Search → Stream を一気通貫**で実行する総合課題だ。

1. **Split**: 合成動画を 10 フレームに分割。
2. **Build**: dense CLIP + 空間連結クラスタリングで 10×6 = 60 本の領域代表ベクトルを作り、FAISS + SQLite を構築。
3. **Search**: テキストクエリ（`"a green region"`）と**領域クエリ**（フレーム 0 の赤領域）の両方で検索し、結果ギャラリー（上段＝テキスト・下段＝領域）を `mini_project_summary.png` に保存。
4. **Stream**: 同じ部品で multiprocessing のストリーム索引を構築（適応サンプリングで 18 → 4 キーフレーム）、それも検索可能であることを確認。

最後に**自己検証**として、(a) `vectors == frames × clusters`、(b) 領域クエリで自分自身がスコア≈1.0 で最上位、(c) 領域検索の上位が赤領域に偏る、(d) ストリーム索引が検索可能、(e) サンプラーが間引いている、をアサートする。実行：

```bash
uv run python lectures/41_cluster_clip_pipeline/mini_project.py
```

これが全 PASS すれば、「学んだ全要素を統合して Cluster-CLIP を自力で再構築できた」ことになる。

---

## ✅ 到達チェックリスト

- [ ] `encode_image`（CLS=画像 1 ベクトル）と `dense_clip_embeddings`（パッチ特徴 `[C,H,W]`）の違いを説明できる。
- [ ] ViT を手で展開して dense 特徴を取る手順（conv1 → CLS 連結 → pos → transformer → ln_post → **proj** → **CLS 除去** → 正規化）を書ける。
- [ ] `grid_to_graph` の connectivity が「なぜ領域を連続させるか」を説明できる。
- [ ] クラスタ代表ベクトル＝「平均 → L2 正規化」を実装できる。
- [ ] `IndexFlatIP + IndexIDMap` を作り `add_with_ids` → `search` → SQLite で `faiss_id` からメタを引ける。
- [ ] `search` の `-1` を必ずガードできる／FAISS 入力を `float32`・C 連続にできる。
- [ ] テキスト→領域・領域→領域・画像全体の 3 検索の違い（局在の有無、安定性）を言える。
- [ ] multiprocessing で取得/推論/書込を分け、キュー満杯ドロップと POISON_PILL 終了を説明できる。
- [ ] ヒストグラム差分サンプラーの数値的な罠（L1 正規化しないと delta が壊れる）を理解している。
- [ ] 本章の各部品が参考実装のどのモジュールに対応するか対応表で言える。

---

## ❓ 落とし穴・FAQ・デバッグ

**Q. テキスト検索（`"a red region"`）の上位に赤が来ない。バグ?**
A. ほぼ仕様だ。本章のフレームは平坦色の合成画像で、CLIP のパッチ特徴は自己注意で文脈が混ざるため、色テキストと領域の意味整合が弱い。**領域→領域**検索（自己一致≈1.0、上位が赤に偏る）が安定して当たることを確認すればパイプラインは正しい。実写画像なら text→region も実用になる。

**Q. `RuntimeError: ... could not open index.faiss` が出る。**
A. Build を先に実行していない／別ディレクトリを見ている。03・mini は `ensure_build` で自動的に Build するが、手動で消した場合は `02_build_index.py` を回す。`run_build` は冪等で、再実行すると DB・index・npy を作り直す。

**Q. ストリームが固まる／終わらない。**
A. POISON_PILL の流れを確認する。Capture は終了時に `task_queue.put(POISON_PILL)` を送り、Consumer はそれを受けて残バッチを flush して return、main が `result_queue.put(POISON_PILL)` を送って Writer を終わらせる。各キューを最後まで drain する者がいないとデッドロックする。また `spawn` のため、ターゲット関数は**モジュールトップレベル**にあり、起動側に `__main__` ガードが要る。

**Q. `faiss` の `search` が落ちる／結果が変。**
A. 入力が `float32`・C 連続か（`np.ascontiguousarray(x, dtype=np.float32)`）、次元が index と一致しているか、正規化済みか（`IndexFlatIP` でコサインにするには DB 側もクエリ側も L2 正規化）を順に疑う。

**Q. 色が反転して見える／推論が変。**
A. BGR↔RGB。`cv2.imread`/`cv2.VideoCapture` は BGR。CLIP へ渡す前・`matplotlib` で描く前に `cv2.cvtColor(..., COLOR_BGR2RGB)`。

**Q. `cv2.VideoWriter` が開けない（`isOpened()` が False）。**
A. mp4v コーデックが無い環境。`01` はその場合フレーム直書きにフォールバックする。実運用では ffmpeg をインストールする。

**Q. クラスタリングが遅い／メモリを食う。**
A. パッチ数が大きすぎる。ViT-B-32 の 7×7=49 に抑える（高解像度パッチは `O(n²)` で爆発）。`n_clusters` も `H*W` 以下にクランプされる（`cluster_agglomerative`）。

---

## 🚀 発展トピック・参考

- **適応サンプリングの高度化**: 参考実装はヒストグラム差分に加え **AFS-MI（相互情報量ベース）** のサンプリングを持つ（`build/producer.py`）。場面の情報量変化でフレームを間引く。
- **評価（Eval）**: 参考実装は GT BBox と**クラスタマスクのカバレッジ**で P@k / NDCG を測る（`eval/coverage.py`）。本章の演習 Q7 `mask_coverage` がその最小版。Cluster マスク ∩ BBox / BBox 面積で「領域が正解物体をどれだけ覆ったか」を測る。
- **Build カプセルと転送**: 参考実装は Build 結果を独立した「カプセル」（`index.faiss` + `metadata.db` + manifest）として保存し、Edge(Jetson)→Server へ tar.gz で転送する（`transfer/sender.py`）。本章の `build/` ディレクトリがその縮小版。
- **エッジ最適化**: 参考実装は MobileCLIP の TorchScript（`mobileclip_blt.ts`, 599MB）で Jetson 推論する。35〜37 章の量子化・ONNX・ランタイム最適化が効いてくる領域。
- **ANN へのスケール**: 規模が上がったら `IndexIVFFlat` / `IndexHNSWFlat` / PQ 圧縮へ（17 章）。検索 API は同じまま差し替えられる。
- **構成を押さえる**: コアは dense CLIP 特徴抽出（`dense_clip_embeddings_vit` / `cluster_agglomerative` 相当）、検索エンジン（search/engine）、ストリーム（stream）。本章の各関数のコメントに対応構成を明記してある。

---

## ▶ 動かし方

```bash
# 依存（このトラックで使うグループ）
uv sync --group dl --group embed --group vector --group metrics

# 4 ステージを順に（01→02→03 は依存。03/mini は単体でも自動で前段を補う）
uv run python lectures/41_cluster_clip_pipeline/01_split_frames.py
uv run python lectures/41_cluster_clip_pipeline/02_build_index.py
uv run python lectures/41_cluster_clip_pipeline/03_search_regions.py
uv run python lectures/41_cluster_clip_pipeline/04_stream_pipeline.py

# 総仕上げ（Split→Build→Search→Stream 一気通貫）
uv run python lectures/41_cluster_clip_pipeline/mini_project.py

# 演習（自己採点）と模範解答（全 PASS）
uv run python lectures/41_cluster_clip_pipeline/exercises.py
uv run python lectures/41_cluster_clip_pipeline/exercises_solutions.py
```

出力（フレーム・カプセル・図）は `outputs/41_cluster_clip_pipeline/` に保存される。初回は CLIP（ViT-B-32, openai）の重みダウンロードが走る（以降キャッシュ）。すべて CPU で数十秒以内に完走する。

---

> 版: torch **2.12+cpu** / open_clip **3.3** / faiss-cpu **1.14** / transformers 5.11 / scikit-learn 1.9 ／ 2026-06
> 題材: **Cluster-CLIP**（本章はその CPU 小型再構築。dense CLIP → 空間連結クラスタリング → 代表ベクトル → FAISS → SQLite → multiprocessing ストリーム を一貫して再現）
