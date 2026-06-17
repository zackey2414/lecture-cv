# 第17回 FAISSベクトルDBと画像検索システム（評価込み）

> トラック: 埋め込み・検索 ／ レベル: 中級 ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`
> （`uv sync --group dl --group hf --group vector --group metrics`）

## 🎯 この章のゴール

この章を終えたとき、あなたは「画像検索システム」を構成要素に分解し、自分の手で組み立てられるようになります。画像検索とは煎じ詰めれば、(1) 画像をモデルでベクトル（埋め込み）に変換し、(2) そのベクトル群を高速に近傍探索できる索引（インデックス）に格納し、(3) クエリも同じ手順でベクトル化して「似ているもの」を引いてくる、という3段のパイプラインです。本章では、その中核を担う **FAISS**（Facebook AI Similarity Search）を、最も単純な総当たり検索から大規模化に耐える近似最近傍（ANN）まで、段階的に手を動かしながら習得します。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="画像検索はDB登録もクエリ検索も埋め込みとL2正規化という同じ前処理を通る3段パイプライン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" fill="#3f3f46">DB登録（add）も クエリ（search）も 同じ前処理を通る</text><rect x="24" y="64" width="120" height="46" rx="5" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="84" y="92" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">DBの画像群</text><rect x="24" y="158" width="120" height="46" rx="5" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><text x="84" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">クエリ画像/文</text><rect x="232" y="92" width="176" height="78" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="320" y="124" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">CLIP 埋め込み</text><text x="320" y="148" text-anchor="middle" font-size="14" font-weight="700" fill="#ea580c">→ L2 正規化</text><rect x="470" y="64" width="166" height="46" rx="5" fill="#f4f4f5" stroke="#16a34a" stroke-width="1.8"/><text x="553" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">FAISS 索引へ add</text><rect x="470" y="158" width="166" height="46" rx="5" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="553" y="186" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">search → 上位k件</text><line x1="144" y1="87" x2="232" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="232,116 220.9,117.6 224.1,108.1" fill="#71717a"/><line x1="144" y1="181" x2="232" y2="146" stroke="#71717a" stroke-width="2"/><polygon points="232,146 224.6,154.3 220.9,145.1" fill="#71717a"/><line x1="408" y1="116" x2="470" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="470,87 463.1,95.8 458.8,86.7" fill="#71717a"/><line x1="408" y1="146" x2="470" y2="181" stroke="#71717a" stroke-width="2"/><polygon points="470,181 458.8,180.4 463.8,171.7" fill="#71717a"/></svg><figcaption>画像検索は3段のパイプラインです。<b>DBの画像群</b>も<b>クエリ</b>（画像でもテキストでも）も、まず <b>CLIP で埋め込み → L2 正規化</b>という同じ前処理を通り、その後 DB 側は索引へ <code>add</code>、クエリ側は <code>search(k)</code> で<b>上位k件</b>を引きます。FAISS から見ればクエリの出自は無関係で、どちらもただのベクトルです。</figcaption></figure>

特に重視するのは「正しさを自分で測れること」です。ANN は速い代わりに答えが近似になるため、厳密解からどれだけズレているかを **Recall@k** で定量化し、**QPS（毎秒クエリ数）と Recall のトレードオフ曲線**を自分で描けるようにします。評価の鉄則はただ一つ、**正解（ground truth）は必ず厳密検索 `IndexFlat` で作る**ことです。なぜなら、ANN 自身の結果を正解にしてしまうと「自分を自分で採点して満点」になり、評価そのものが無意味になるからです。

到達点を一言でいえば、**「埋め込み → 正規化 → add → search → 永続化 → 評価」という一連を、AI 補助なしでそらで書け、コサイン類似度のための正規化や `float32`・C連続といった FAISS の作法、IVF/HNSW/PQ の使い分けを自分の言葉で説明できる**ことです。そして最終的には、CLIP の埋め込みを使って「テキストで画像を検索する」マルチモーダル検索まで通します。

---

## 1. ベクトル検索とは — 近傍探索（NN）と近似近傍探索（ANN）

ベクトル検索の出発点は「似ているもの＝ベクトル空間で近いもの」という考え方です。画像やテキストをあらかじめ固定長のベクトル（埋め込み）に変換しておけば、「似た画像を探す」という曖昧な問いは、「クエリベクトルに距離が近い点を探す」という明確な数学の問題に変わります。これを厳密に解くのが **最近傍探索（NN: Nearest Neighbor）** で、全データとの距離を総当たりで計算します。答えは常に正確ですが、そのぶんデータ数 N に比例して遅くなります（O(N)）。

データが数十万・数百万件に増えると、総当たりはもはや現実的でなくなります。そこで登場するのが **近似最近傍探索（ANN: Approximate NN）** です。空間をあらかじめ区切ったり（IVF）、近傍をたどるグラフを作ったり（HNSW）、ベクトルを圧縮したり（PQ）することで、「ほぼ正しい答えを、桁違いに速く」返します。つまり ANN は、精度をいくらか犠牲にして速度とメモリを稼ぐ技術であり、その「いくらか」を測って管理するのがエンジニアの仕事になります。

FAISS は、この NN/ANN の索引を多数そろえたライブラリで、**CPU だけで全機能（Flat/IVF/HNSW/PQ）が動きます**。本章は `faiss-cpu` 1.14.2 を前提に、すべて CPU で完結させます。まずは全体像として、「Flat＝厳密だが遅い／ANN＝近似だが速い」「両者は同じ `add`/`search` の API で切り替えられる」という二点を頭に入れておきましょう。そうすれば、以降の各 index も同じ枠組みのバリエーションとして理解できます。

## 2. FAISS の基本ループと「float32・C連続」の鉄則

FAISS のコードは、どの index でも同じ形をしています。`index = faiss.IndexXxx(d)` で次元 `d` の索引を作り、`index.add(xb)` でデータベースのベクトル群を入れ、`index.search(xq, k)` でクエリごとに上位 `k` 件を引く——基本はこれだけです。`search` は **距離 `D` と ID `I` の2つの行列**（ともに形 `(クエリ数, k)`）を返します。なお、FAISS の世界では公式ドキュメントに倣って戻り値を `D, I` と書くのが慣例なので、本講座もその名前を踏襲します。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="searchは距離Dと近傍IDのIという2つのnq×k行列を返し、I[q]の先頭はクエリ自身で距離0" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" fill="#3f3f46">search(xq, k) は 距離 D と ID I を返す（各 nq×k）</text><rect x="28" y="78" width="86" height="96" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><line x1="28" y1="110" x2="114" y2="110" stroke="#dbeafe" stroke-width="1.5"/><line x1="28" y1="142" x2="114" y2="142" stroke="#dbeafe" stroke-width="1.5"/><text x="71" y="68" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">クエリ xq</text><text x="71" y="190" text-anchor="middle" font-size="12" fill="#52525b">(nq, d)</text><line x1="114" y1="126" x2="152" y2="126" stroke="#71717a" stroke-width="2"/><polygon points="152,126 142,121 142,131" fill="#71717a"/><rect x="154" y="100" width="120" height="52" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="214" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">index</text><text x="214" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#ea580c">search(k)</text><line x1="274" y1="126" x2="312" y2="126" stroke="#71717a" stroke-width="2"/><polygon points="312,126 302,121 302,131" fill="#71717a"/><text x="386" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">D：距離</text><rect x="326" y="84" width="120" height="90" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><line x1="356" y1="84" x2="356" y2="174" stroke="#e4e4e7"/><line x1="386" y1="84" x2="386" y2="174" stroke="#e4e4e7"/><line x1="416" y1="84" x2="416" y2="174" stroke="#e4e4e7"/><line x1="326" y1="114" x2="446" y2="114" stroke="#e4e4e7"/><line x1="326" y1="144" x2="446" y2="144" stroke="#e4e4e7"/><text x="341" y="104" text-anchor="middle" font-size="11" fill="#18181b">0.0</text><text x="371" y="104" text-anchor="middle" font-size="11" fill="#18181b">.21</text><text x="401" y="104" text-anchor="middle" font-size="11" fill="#18181b">.55</text><text x="431" y="104" text-anchor="middle" font-size="11" fill="#18181b">.73</text><text x="530" y="74" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">I：ID</text><rect x="470" y="84" width="120" height="90" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="470" y="84" width="30" height="30" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><line x1="500" y1="84" x2="500" y2="174" stroke="#e4e4e7"/><line x1="530" y1="84" x2="530" y2="174" stroke="#e4e4e7"/><line x1="560" y1="84" x2="560" y2="174" stroke="#e4e4e7"/><line x1="470" y1="114" x2="590" y2="114" stroke="#e4e4e7"/><line x1="470" y1="144" x2="590" y2="144" stroke="#e4e4e7"/><text x="485" y="104" text-anchor="middle" font-size="11" font-weight="700" fill="#c2410c">0</text><text x="515" y="104" text-anchor="middle" font-size="11" fill="#18181b">5</text><text x="545" y="104" text-anchor="middle" font-size="11" fill="#18181b">2</text><text x="575" y="104" text-anchor="middle" font-size="11" fill="#18181b">9</text><text x="458" y="206" text-anchor="middle" font-size="12" fill="#52525b">各行が1クエリの上位k件。先頭が最も近い（自身は D=0.0）</text></svg><figcaption><code>index.search(xq, k)</code> は、<b>距離 D</b> と <b>ID I</b> の2つの行列を返します（どちらも形 <b>(nq, k)</b> ＝ クエリ数 × 上位k）。<code>I</code> の各行が、そのクエリに近い順の<b>近傍ID</b>です。DB に自分自身が入っていれば <code>I[q]</code> の先頭はクエリ自身（<code>IndexFlatL2</code> なら距離 <b>0.0</b>）になり、配線が正しい証拠になります。</figcaption></figure>

唯一にして最大の落とし穴が、入力配列の型です。**FAISS に渡す配列は、必ず `float32` かつ C連続（row-major）でなければなりません**。ところが Python で何気なく作る配列は、`float64` だったり、スライスや転置でメモリが非連続だったりします。これを `search` に渡すと、静かに誤動作したり例外で落ちたりします。だからこそ、index に渡す直前に必ず `np.ascontiguousarray(x, dtype=np.float32)` を通すのを手癖にします。本章のヘルパ `as_faiss_array` が、まさにこれを担います。

```python
import faiss, numpy as np

xb = np.ascontiguousarray(xb, dtype=np.float32)  # float32・C連続を保証
index = faiss.IndexFlatL2(d)   # L2距離の総当たり索引
index.add(xb)                  # DBを格納（Flatはtrain不要）
D, I = index.search(xq, k)     # D:距離(nq,k), I:ID(nq,k)
```

上のコードで覚えてほしいのは、次の二点です。すなわち、`IndexFlatL2` は学習（train）が要らず `add` した瞬間から検索できること、そして `I[q]` の先頭にはクエリ自身（DBに含まれていれば距離0）が来ること、です。実際に `01_flat_ip_cosine.py` を実行すると、クエリ0の近傍IDの先頭が `0`、その L2 距離が `0.0` になることを確認できます。「自分が自分の最近傍になる」のは、パイプラインが正しく繋がっている何よりの証拠なので、動作確認の定番テクニックとして使ってください。

## 3. L2距離 と 内積、そして「正規化でコサイン」

FAISS の距離尺度には、大きく2系統あります。`IndexFlatL2` は **L2距離（ユークリッド距離）** で「小さいほど近い」、`IndexFlatIP` は **内積（Inner Product）** で「大きいほど近い」です。ここでソートの向きが逆である点（L2は昇順、IPは降順が「近い順」）を最初に意識しておかないと、スコアの閾値処理や上位選択で取り違えてしまいます。両者の違いを下の表に整理します。

| 索引 | 尺度 | 「近い」の向き | コサインにするには |
| --- | --- | --- | --- |
| `IndexFlatL2(d)` | L2距離 ‖a−b‖² | 小さいほど近い（昇順） | 両方を正規化すれば順位はIPと一致 |
| `IndexFlatIP(d)` | 内積 a·b | 大きいほど近い（降順） | **両方を L2 正規化**すると内積＝コサイン |

この表の右列こそ、本章で最も重要なポイントです。実務で使いたいのは、たいてい **コサイン類似度**（ベクトルの「向き」の近さで、長さに依存しない）ですが、FAISS にはコサイン専用の index がありません。そこで代わりに、**ベクトルを L2 正規化（ノルムを1にする）してから `IndexFlatIP`（内積）を使う**と、内積がそのままコサイン類似度になります。鍵は「DB側とクエリ側の両方を正規化する」ことで、片方だけだと値が崩れます。

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="L2距離は先端間の距離、内積は長さに依存。両ベクトルをL2正規化し単位円に乗せると内積がコサイン類似度になる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="120" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">① 正規化前（長さがバラバラ）</text><circle cx="90" cy="170" r="3" fill="#18181b"/><line x1="90" y1="170" x2="188" y2="82" stroke="#2563eb" stroke-width="2.5"/><polygon points="188,82 182,94 176,86" fill="#2563eb"/><line x1="90" y1="170" x2="214" y2="132" stroke="#ea580c" stroke-width="2.5"/><polygon points="214,132 204,140 201,131" fill="#ea580c"/><line x1="188" y1="82" x2="214" y2="132" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5 3"/><text x="196" y="78" font-size="14" font-weight="700" fill="#1d4ed8">a</text><text x="224" y="132" font-size="14" font-weight="700" fill="#c2410c">b</text><text x="120" y="158" font-size="12" font-weight="700" fill="#3f3f46">θ</text><text x="232" y="104" font-size="11.5" font-weight="700" fill="#dc2626">L2</text><line x1="300" y1="150" x2="366" y2="150" stroke="#71717a" stroke-width="2.2"/><polygon points="366,150 356,145 356,155" fill="#71717a"/><text x="333" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">正規化</text><text x="333" y="170" text-anchor="middle" font-size="10.5" fill="#71717a">‖·‖→1</text><text x="470" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">② 正規化後（単位円）</text><circle cx="455" cy="158" r="70" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><circle cx="455" cy="158" r="3" fill="#18181b"/><line x1="455" y1="158" x2="507" y2="111" stroke="#2563eb" stroke-width="2.5"/><polygon points="507,111 501,122 495,115" fill="#2563eb"/><line x1="455" y1="158" x2="522" y2="139" stroke="#ea580c" stroke-width="2.5"/><polygon points="522,139 512,147 509,138" fill="#ea580c"/><text x="515" y="106" font-size="14" font-weight="700" fill="#1d4ed8">a</text><text x="532" y="140" font-size="14" font-weight="700" fill="#c2410c">b</text><text x="482" y="150" font-size="12" font-weight="700" fill="#3f3f46">θ</text><text x="455" y="80" text-anchor="middle" font-size="11.5" fill="#52525b">‖a‖=‖b‖=1</text><text x="455" y="244" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">内積 = cos θ</text></svg><figcaption><b>L2距離</b>はベクトルの先端どうしの距離（小さいほど近い）、<b>内積</b>は <code>‖a‖‖b‖cosθ</code> で長さにも依存します。そこで両方を <b>L2 正規化</b>して長さを 1 に揃える（単位円に乗せる）と、<b>内積 ＝ cos θ ＝ コサイン類似度</b>になります。<b>DB 側・クエリ側の両方</b>を正規化するのが鍵で、片方だけだと値が崩れます。</figcaption></figure>

```python
faiss.normalize_L2(xb)          # ← in-place！ xb 自身が書き換わる（破壊的）
index = faiss.IndexFlatIP(d)
index.add(xb)
faiss.normalize_L2(xq)
D, I = index.search(xq, k)      # D はコサイン類似度（-1〜1, 1が最も似ている）
```

ここで注意したいのは、`faiss.normalize_L2` が **引数の配列をその場で（in-place）書き換える**点です。元の未正規化ベクトルを後で使いたいなら、コピーしてから渡すか、本章ヘルパの非破壊版 `l2_normalize`（新しい配列を返す）を使います。`01_flat_ip_cosine.py` では、未正規化のまま `IndexFlatIP` を使った「素の内積」検索と、正規化後の「コサイン」検索を並べて出力し、正規化するとクエリ自身とのスコアがちょうど `≈1.0` になることを見せています。さらに、正規化後は L2 と IP で近傍の集合が一致する（並び方向だけ逆）ことも確認できます。

## 4. ID の管理 — `IndexIDMap` とメタデータの別管理（SQLite）

ここからは、実際の画像検索に踏み込みます。素の `IndexFlat` に `add` すると、各ベクトルには `0..N-1` の連番が機械的に振られるだけです。しかし実務では、「このベクトルは画像 `cat_001.jpg` のもの」「あのベクトルは商品ID 8675309」のように、**自分で決めた ID を紐づけたい**ことがほとんどです。そこで `IndexIDMap`（または ID からベクトルを復元できる上位版 `IndexIDMap2`）で素の index を包み、`add_with_ids(vectors, ids)` で任意の `int64` ID を割り当てます。

ここで割り切っておきたいのは、**FAISS が覚えてくれるのは「ベクトルと ID」だけ**で、画像パスやキャプションといったメタデータは持てない、という点です。したがってメタデータは、SQLite や辞書など FAISS の外で管理し、検索結果の ID から引く——これが実運用の定石です。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="FAISSはベクトルとID(int64)だけを保持し、画像パスなどのメタデータはSQLiteで別管理してIDで引く" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">FAISS は『ベクトル＋ID』だけ。メタデータは SQLite で別管理</text><rect x="28" y="52" width="268" height="176" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="162" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">FAISS インデックス</text><rect x="48" y="98" width="16" height="16" fill="#2563eb"/><rect x="66" y="98" width="16" height="16" fill="#16a34a"/><rect x="84" y="98" width="16" height="16" fill="#dc2626"/><rect x="102" y="98" width="16" height="16" fill="#f97316"/><text x="128" y="111" font-size="13" font-weight="700" fill="#18181b">→ id = 1000</text><rect x="48" y="138" width="16" height="16" fill="#ea580c"/><rect x="66" y="138" width="16" height="16" fill="#2563eb"/><rect x="84" y="138" width="16" height="16" fill="#16a34a"/><rect x="102" y="138" width="16" height="16" fill="#dc2626"/><text x="128" y="151" font-size="13" font-weight="700" fill="#18181b">→ id = 1001</text><text x="70" y="194" font-size="12.5" fill="#71717a">…（ベクトル, id）の組だけ</text><rect x="364" y="52" width="268" height="176" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="498" y="80" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">SQLite（メタDB）</text><text x="400" y="106" font-size="12" font-weight="700" fill="#3f3f46">id</text><text x="470" y="106" font-size="12" font-weight="700" fill="#3f3f46">label</text><line x1="384" y1="114" x2="612" y2="114" stroke="#d4d4d8" stroke-width="1.5"/><text x="400" y="138" font-size="12" fill="#18181b">1000</text><text x="470" y="138" font-size="12" fill="#18181b">red_circle.jpg</text><text x="400" y="166" font-size="12" fill="#18181b">1001</text><text x="470" y="166" font-size="12" fill="#18181b">blue_square.jpg</text><text x="400" y="196" font-size="12" fill="#71717a">…</text><line x1="296" y1="120" x2="364" y2="120" stroke="#71717a" stroke-width="2.2"/><polygon points="364,120 354,115 354,125" fill="#71717a"/><text x="330" y="112" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">ID で引く</text><text x="330" y="244" text-anchor="middle" font-size="12" fill="#52525b">.faiss（左）と .db（右）は必ずセットで保存・復元</text></svg><figcaption>FAISS が覚えるのは <b>ベクトルとID（int64）</b>だけで、画像パスなどの<b>メタデータは持てません</b>。そこでメタデータは <b>SQLite など FAISS の外</b>で管理し、<code>search</code> が返した <b>ID で引いて</b>ラベルへ変換します。<code>write_index</code> は左の索引しか保存しないため、<code>.faiss</code> と <code>.db</code> は必ず<b>セットで保存・復元</b>します。</figcaption></figure>下のコードでは、ID→ラベルの対応表を SQLite に作っています（`label` 列に実際の画像パスやサムネイル、撮影日時などを入れるイメージです）。

```python
ids = np.arange(1000, 1000 + len(images)).astype(np.int64)  # 連番でない任意ID
index = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
index.add_with_ids(db_vecs, ids)            # ベクトルとIDだけFAISSへ

con = sqlite3.connect(db_path)              # メタデータは別DBで管理
con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)")
con.executemany("INSERT INTO items VALUES (?, ?)", zip(ids.tolist(), labels))
```

`02_idmap_persist_sqlite.py` は、この構成で合成画像（赤い円・青い四角など色×形の24枚）を CLIP で埋め込み、`add_with_ids` で 1000 番台の ID を振って格納します。検索後は、得られた ID で SQLite を引き、「赤い円」というラベルに変換して表示します。こうして FAISS の ID とアプリのメタデータを分離して管理する感覚こそ、小さな検索エンジンを自作するときの背骨になります。

## 5. 永続化 — `write_index` / `read_index` とメタDBの「セット保存」

作った index は、毎回ゼロから作り直すのではなく、ファイルに保存して再利用します。FAISS では `faiss.write_index(index, "x.faiss")` で保存し、`faiss.read_index("x.faiss")` で読み戻します（バイト列にしたいときは `serialize_index`/`deserialize_index`）。読み戻した index の `ntotal`（登録件数）が保存前と一致すれば、永続化は成功です。これで、アプリの再起動をまたいでも索引を引き継げるようになります。

ただし、ここに見落としやすい罠があります。**`write_index` が保存するのは index 本体（ベクトルと内部ID）だけで、別管理しているメタDB（ID→画像パス）は保存しません**。したがって、`.faiss` ファイルと `.db`（あるいは `.json`）は必ず**セットで、バージョンを揃えて**永続化する必要があります。片方だけ更新して取り違えると、検索は通るのに ID からメタデータが引けない、という不整合が起きます。

```python
faiss.write_index(index, "image_index.faiss")   # ベクトル＋内部IDを保存
index = faiss.read_index("image_index.faiss")    # 別オブジェクトとして復元
assert index.ntotal == 24                         # 件数一致＝永続化OK
# 注意: メタDB(image_meta.db) は別途セットで保存・復元すること！
```

`02_idmap_persist_sqlite.py` では、index を保存→まっさらな状態から読み戻し（アプリ再起動を模す）→そのまま検索、という流れを通します。「索引ファイルとメタデータDBは一心同体」という運用上の感覚を、ここで体に入れてください。この感覚は、後半の Cluster-CLIP 統合（第41回）で、ストリーム処理しながら `.faiss` と SQLite を整合させる設計にそのまま繋がります。

## 6. テキスト→画像のマルチモーダル検索（CLIP / transformers v5 の注意）

埋め込みを CLIP（`openai/clip-vit-base-patch32`）で作ると、画像とテキストが**同じ潜在空間**に置かれます。この性質を使うと、「`a photo of a red circle` というテキストで、赤い円の画像を検索する」というマルチモーダル検索ができます。やること自体は画像のときと同じで、テキストを埋め込み→正規化→同じ index に対して `search` するだけです。FAISS から見れば、クエリがテキスト由来か画像由来かは関係なく、ただのベクトルにすぎません。

ここで、**transformers v5（5.11）の破壊的変更**を2つ押さえておきます。1つ目は画像プロセッサです。`AutoFeatureExtractor` は廃止され、画像には **`AutoImageProcessor`（または `AutoProcessor`）** を使います。さらにプロセッサは torchvision バックエンドの fast 実装のみになったため、画像モデルを使うなら **torchvision が事実上必須**です（入れ忘れると `AutoImageProcessor` がエラーになります）。2つ目は、埋め込みの取り出し方です。

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

v5 では `get_image_features` / `get_text_features` の戻り値が **`BaseModelOutputWithPooling` オブジェクトに変わり**、射影後ベクトルは `.pooler_output` に入っています（古いコードのように戻り値を直接テンソル扱いすると `'...' object has no attribute 'shape'` で落ちます）。しかも、この `.pooler_output` は **L2 正規化されていません**（ノルムは約11.7）。一方で `model(**inputs).logits_per_image` の経路は内部で正規化済み、という非対称があります。したがって、検索やコサイン類似度の前には **必ず自分で `F.normalize`** してください。これが、CLIP まわりで最頻出の落とし穴です。実際に `02_idmap_persist_sqlite.py` を実行すると、`a photo of a red circle`→赤い円、`a blue square`→青い四角、`a green triangle`→緑の三角、と妥当に当たることが確認できます（CLIP を取得できないオフライン環境では自動で色記述子にフォールバックし、テキスト検索だけスキップして完走します）。

## 7. 近似最近傍 — IVFFlat / HNSW / IVFPQ・OPQ と `index_factory`

データが増えてきたら、ANN に切り替えます。代表的な3系統を押さえれば、実務はだいたい回ります。まず **IVFFlat** は、空間を `nlist` 個のセルに区切り、クエリに近い `nprobe` 個のセルだけを探す「転置インデックス」方式です。`add` の前に代表データでセル中心を学習する **`train` が必須**で、これを忘れて `add` すると例外になります。検索時の `nprobe` が精度↔速度のダイヤルで、`nprobe=1` のまま使うと Recall が大きく落ちる（既定の罠）ので、必ず調整します。

<figure class="lec-fig"><svg viewBox="0 0 640 250" role="img" aria-label="IVFは空間をnlist個のセルに分割し、クエリに近いnprobe個のセルだけを探索する。灰色のセルは探索しない" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="24" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">IVF：空間を nlist セルに分割し、近い nprobe セルだけ探す</text><rect x="40" y="44" width="192" height="192" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="40" y="108" width="64" height="64" fill="#ffedd5"/><rect x="104" y="108" width="64" height="64" fill="#ffedd5"/><line x1="104" y1="44" x2="104" y2="236" stroke="#e4e4e7"/><line x1="168" y1="44" x2="168" y2="236" stroke="#e4e4e7"/><line x1="40" y1="108" x2="232" y2="108" stroke="#e4e4e7"/><line x1="40" y1="172" x2="232" y2="172" stroke="#e4e4e7"/><circle cx="72" cy="76" r="3" fill="#3f3f46"/><circle cx="136" cy="76" r="3" fill="#3f3f46"/><circle cx="200" cy="76" r="3" fill="#3f3f46"/><circle cx="72" cy="140" r="3" fill="#3f3f46"/><circle cx="136" cy="140" r="3" fill="#3f3f46"/><circle cx="200" cy="140" r="3" fill="#3f3f46"/><circle cx="72" cy="204" r="3" fill="#3f3f46"/><circle cx="136" cy="204" r="3" fill="#3f3f46"/><circle cx="200" cy="204" r="3" fill="#3f3f46"/><circle cx="120" cy="150" r="4" fill="#2563eb"/><circle cx="150" cy="128" r="4" fill="#2563eb"/><circle cx="86" cy="140" r="4" fill="#2563eb"/><circle cx="64" cy="158" r="4" fill="#2563eb"/><circle cx="200" cy="70" r="4" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><circle cx="70" cy="72" r="4" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><circle cx="206" cy="200" r="4" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><circle cx="150" cy="205" r="4" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><circle cx="196" cy="150" r="4" fill="#d4d4d8" stroke="#71717a" stroke-width="1"/><polygon points="134,131 136.6,138.4 144.5,138.6 138.3,143.4 140.5,150.9 134,146.5 127.5,150.9 129.7,143.4 123.5,138.6 131.4,138.4" fill="#ea580c" stroke="#c2410c" stroke-width="1"/><rect x="300" y="64" width="26" height="18" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="334" y="78" font-size="12.5" fill="#3f3f46">探索するセル（nprobe 個）</text><rect x="300" y="98" width="26" height="18" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="334" y="112" font-size="12.5" fill="#3f3f46">探索しないセル（だから速い）</text><circle cx="313" cy="146" r="4" fill="#3f3f46"/><text x="334" y="150" font-size="12.5" fill="#3f3f46">セル中心（nlist 個）</text><polygon points="313,172 314.9,177.3 320.6,177.5 316.1,181 317.7,186.5 313,183.3 308.3,186.5 309.9,181 305.4,177.5 311.1,177.3" fill="#ea580c" stroke="#c2410c" stroke-width="1"/><text x="334" y="184" font-size="12.5" fill="#3f3f46">クエリ</text><text x="300" y="222" font-size="12.5" font-weight="700" fill="#c2410c">nprobe を上げる → Recall↑（速度↓）</text></svg><figcaption><b>IVFFlat</b> は空間を <b>nlist 個のセル</b>に区切り（各セルに中心＝重心を学習）、クエリに近い <b>nprobe 個のセルだけ</b>を探します。灰色のセルは見ないので速い反面、<code>nprobe=1</code> のままだと取りこぼして <b>Recall が落ちます</b>。<b>nprobe を上げる</b>ほど精度は上がり、速度とのトレードオフになります。</figcaption></figure>

次に **HNSW** は、近傍グラフをたどる方式で、**train 不要**・高速・高精度ですが、メモリを多く食います。グラフの次数 `M`、構築品質 `efConstruction`、検索幅 `efSearch` を持ち、検索時の主ダイヤルは `efSearch` です。最後に **PQ（Product Quantization）** は、ベクトルを部分ベクトルに割って量子化し、メモリを劇的に削減する圧縮技術です（精度と引き換え）。これらはいずれも、文字列レシピで組み立てる **`index_factory`**（例 `"IVF100,PQ8"`, `"OPQ8_64,IVF100,PQ8"`, `"HNSW32"`）で手軽に作れます。

| 索引 | train | 主ダイヤル | 長所 | 注意 |
| --- | --- | --- | --- | --- |
| `IndexFlatIP/L2` | 不要 | — | 厳密・実装単純 | O(N) で遅い／評価の正解作りに使う |
| `IndexIVFFlat` | **必須** | `nprobe` | 速い・メモリ普通 | `nprobe=1` は低Recall。`nlist`目安 4√N〜16√N |
| `IndexHNSWFlat` | 不要 | `efSearch` | 高速・高精度 | メモリ大・削除に弱い |
| `IVFPQ`/`OPQ`(factory) | **必須** | `nprobe` | メモリ激減 | 精度低下。十分な学習データが必要 |

`03_ivf_hnsw_pq.py` は、1万件・64次元のデータで上記を実際に構築・検索し、Recall とファイルサイズ（≒メモリ量）を比べます。例えば `IVF100,PQ8` は 1ベクトルを 8 バイトに圧縮し（`float32` 生の `64×4=256` バイトの1/32）、保存ファイルも `IVF100,Flat` の約1/10になります。その代償として Recall@10 は約0.45まで落ちます——これが「メモリと精度のトレードオフ」の生の数字です。さらにスクリプトは、`nprobe` を上げると IVF の Recall が `0.79→1.0` と改善する様子も出力するので、ダイヤルの効きを体感してください。

## 8. 評価 — Recall@k の自前計算・QPS-recall曲線・retrieval mAP

ANN の良し悪しは、「どれだけ厳密解に近いか」で測ります。その指標が **Recall@k** で、定義は「クエリごとに、厳密検索の上位k件のうち ANN の上位k件に入っていた割合」を全クエリで平均したものです。計算は集合の一致で素直に書けます。ここでも鉄則は、**正解（ground truth）を必ず厳密な `IndexFlat` で作る**ことです。ANN 自身の結果を正解にすると、評価が無意味になります（数字が高く見えるだけです）。

```python
_, gt = flat.search(xq, k)          # ground truth は厳密Flatで作る（重要）
_, ann = ivf.search(xq, k)          # 評価したいANNの結果
def recall_at_k(gt, ann, k):
    return np.mean([np.intersect1d(g[:k], a[:k]).size / k
                    for g, a in zip(gt, ann)])
# np.intersect1d は -1（近傍不足の埋め草）を自然に無視してくれる
```

上の `recall_at_k` では、`np.intersect1d` を使うのがコツです。これにより、`search` が返す `-1`（近傍が足りないときの埋め草）が混ざっても、誤カウントせず無視できます。`04_recall_qps_eval.py` は、`nprobe`（IVF）と `efSearch`（HNSW）をスイープし、各設定の Recall@10 と **QPS（毎秒クエリ数）**を測って、**横軸QPS（対数）・縦軸Recall の曲線**を `04_qps_recall_curve.png` に描きます。曲線は、右上ほど「速くて正確」で良い設定です。なお QPS は1回計測だとブレるので、ウォームアップ＋複数回平均で安定させています。

最後に、ラベル付きデータでは **retrieval mAP**（同じラベルを正例とした平均適合率）も測ります。Recall@k が「ANN が厳密検索にどれだけ忠実か」を見るのに対し、mAP は「同じカテゴリの仲間を上位にどれだけ集められるか＝埋め込み自体の良さ」を見ます。観点が違うので、両方を見ます。本章の合成データでは mAP@10 ≈ 0.999 と、クラスタ構造がきれいに引けていることが数字で確認できます。これらの結果は `04_eval_report.json` にも保存されるので、後から再現・比較できます。

## 9. 実運用の作法 — `-1` ガードとインクリメンタル更新

実運用でまず効くのが、`-1` ガードです。`k` が登録件数 `ntotal` より大きいときや、IVF で十分な近傍が見つからないとき、`search` の返す ID 行列 `I` には `-1` が混じります。これをそのまま「ID」としてメタDB参照に使うと、クラッシュや誤参照になります。そこで、**`-1` を弾く（`(none)` 等に置換する）処理を必ず入れます**。`02_idmap_persist_sqlite.py` の `lookup_labels` は、`i < 0` を早期 return でガードしています。

```python
for i in result_ids:
    i = int(i)
    if i < 0:                 # -1 ガード（最初に弾く）
        labels.append("(none)")
        continue
    labels.append(lookup(i))  # 正常なIDだけメタDBを引く
```

もう一つ、実務で頻出するのが **インクリメンタル更新**です。`IndexIDMap` に `add_with_ids` で逐次追加すれば、検索可能な状態を保ったままデータを増やせます。あとは定期的に `write_index` で永続化し、メタDBと整合させる——これが、画像が流れ込み続けるストリーム処理の基本設計です（Cluster-CLIP の `stream/writer.py` がまさにこの実例で、第40・41回で扱います）。ただし IVF/PQ 系は、最初に学習したセル中心が前提なので、データ分布が大きく変わったら `train` のやり直し（再構築）が必要になる点は覚えておきましょう。

## 10. CPU / GPU と faiss のインストールの注意

本章は、`faiss-cpu` だけで Flat/IVF/HNSW/PQ の全機能・保存・評価が CPU で完結します（MacBook を含む CPU のみ環境でも問題なく動きます）。検索の挙動と API は CPU/GPU で同一なので、学習目的では CPU で十分です。GPU に載せ替えたいときも、`index = faiss.index_cpu_to_gpu(res, 0, index)` の一行を足すだけで、検索コードは変わりません。

インストールで紛らわしいのが、GPU 版です。**`pip install faiss-gpu` というパッケージ名は（現在）存在しません**。GPU 版は `faiss-gpu-cuvs` という名前で、しかも **Linux x86_64 + NVIDIA（CUDA 12.4 系）限定**、PyPI からは `--extra-index-url https://pypi.nvidia.com` 経由、という制約があります。さらに `faiss-cpu` と `faiss-gpu-cuvs` は、**同じ `import faiss` 名前空間を奪い合うため、同一環境に共存させてはいけません**。そこで本講座では、`vector` グループに `faiss-cpu` のみを入れ、GPU は「手元に対応GPUがあれば試す」補足に留めています。

## 11. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で書かれており、上から順に読むと「FAISS の基礎 → 実画像検索 → ANN → 評価」と理解が積み上がります。いずれも結果（PNG/JSON/.faiss）を `lectures/17_faiss_image_search/outputs/` に保存し、画面表示には依存しません。共通処理（出力先・正規化・Recall・合成データ・CLIP埋め込み）は `search_helpers.py` にまとめてあり、各スクリプトがこれを import します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `search_helpers.py` | 出力先/デバイス判定・`as_faiss_array`/`l2_normalize`・`recall_at_k`/AP・合成ベクトル/画像・CLIP埋め込み(`Embedder`、色記述子フォールバック付) |
| `01_flat_ip_cosine.py` | FAISS 基本ループ、`float32`・C連続、L2 vs IP、正規化でコサイン、`D,I` の読み方 |
| `02_idmap_persist_sqlite.py` | CLIP埋め込み→正規化→`IndexIDMap2`+`add_with_ids`→SQLite→`write/read_index`→画像/テキスト検索→`-1`ガード |
| `03_ivf_hnsw_pq.py` | `IVFFlat`(train/nprobe)・`HNSW`(efSearch)・`index_factory`(IVFPQ/OPQ)、Recall とメモリ比較 |
| `04_recall_qps_eval.py` | Recall@k 自前計算、`nprobe`/`efSearch` スイープの QPS-recall 曲線、retrieval mAP、JSON保存 |
| `use_case.py` | **実践ユースケース**: 逆画像検索（reverse image search）。クエリ画像1枚でフォルダ内の類似/重複を FAISS で引き、コサイン閾値で near-duplicate を判定する“動く小ツール”（実画像差し替え・永続化対応） |
| `mini_project.py` | **章末ミニプロジェクト**: CLIP 埋め込みで画像検索エンジンを構築→永続化→再読込→画像/テキスト検索→retrieval mAP、さらに大規模合成データで IVF/HNSW を Recall@k・QPS 評価して最速設定を推薦 |
| `exercises.py` | TODO 形式の演習 **全10問**（易→難・自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の模範解答ランナー（全10問 PASS。採点ロジックは `exercises` 側を再利用） |

`search_helpers.py` だけは、「読み物」ではなく「再利用する道具」です。トップレベルでは torch を import せず、CLIP が要る `Embedder` の中だけで遅延 import している点（FAISS だけ学ぶ 01/03/04 を軽くするため）に注目すると、依存を絞る設計の意図が伝わります。

## 12. 動かし方

このモジュールは、`faiss-cpu`（vector）に加え、`02` で CLIP を使うため `dl`/`hf` グループが要ります。評価のために `metrics` も入れておきます。入力画像は合成生成されるので、いきなり実行できます（`data/17_faiss_image_search/` に画像を置けば、そちらが優先して使われます）。準備ができたら、プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）
uv sync --group dl --group hf --group vector --group metrics

# 各スクリプトを実行（結果は lectures/17_faiss_image_search/outputs/ に保存される）
uv run python lectures/17_faiss_image_search/01_flat_ip_cosine.py
uv run python lectures/17_faiss_image_search/02_idmap_persist_sqlite.py   # 初回はCLIP重みDL
uv run python lectures/17_faiss_image_search/03_ivf_hnsw_pq.py
uv run python lectures/17_faiss_image_search/04_recall_qps_eval.py

# 章末ミニプロジェクト: 画像検索エンジンを組み立て・評価する総合課題（初回はCLIP重みDL）
uv run python lectures/17_faiss_image_search/mini_project.py

# 演習: まずは TODO を自分で埋める（全10問。最初は全部 FAIL、でも exit 0）
uv run python lectures/17_faiss_image_search/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る（2通り。どちらも全10問 PASS）
SHOW_SOLUTION=1 uv run python lectures/17_faiss_image_search/exercises.py
uv run python lectures/17_faiss_image_search/exercises_solutions.py
```

実行後は、`lectures/17_faiss_image_search/outputs/` の図を確認してください。`01_flat_scores.png`（IP素/コサイン/L2 のスコアの並び）、`02_query_results.png`（クエリ画像と類似画像）、`03_ann_tradeoff.png`（精度↔速度の散布図）、`04_qps_recall_curve.png`（スイープ曲線）と `04_eval_report.json`（数値）が出ます。なお `02` で CLIP を取得できない環境では、色記述子に自動フォールバックし、テキスト検索だけスキップして完走します（画像→画像検索は動きます）。

## 13. よくあるエラーと対処（チェックリスト）

最後に、この章で詰まりやすい点を「症状 → 原因 → 対処」の形でまとめます。多くは、FAISS の入力作法か、正規化忘れ、評価の作り方のいずれかに集約されます。

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

この表の項目を自分で説明でき、回避コードを書けるようになれば、本章のゴールに到達しています。とりわけ「`float32`・C連続」「両側を正規化」「正解は Flat で作る」の3つは、FAISS を使い続ける限り何度も効いてくる原則です。

## 14. まとめ

本章では、画像検索を「埋め込み → 正規化 → add → search → 永続化 → 評価」に分解し、FAISS の基本ループ、`float32`・C連続の鉄則、L2/IP と「正規化でコサイン」、`IndexIDMap` とメタデータ別管理、`write/read_index` のセット永続化、CLIP によるテキスト→画像のマルチモーダル検索、IVF/HNSW/PQ の使い分け、そして Recall@k・QPS-recall曲線・retrieval mAP による定量評価までを、すべて自分の手で動かしました。

次回以降の検出・セグメンテーションでも、また最終応用の Cluster-CLIP（第40・41回）でも、この「埋め込みをベクトルDBに入れて検索・評価する」骨格は繰り返し登場します。だからこそ、`search_helpers.py` のような自分用の道具を一つ持っておくと、以降は定型処理に煩わされず本質に集中できます。まずは演習を全問 PASS させ、正規化と評価の感覚を体に入れてから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — 小さな画像検索エンジンを作って評価する

ここまでの学びを 1 本に統合する総合課題が、`mini_project.py` です。これは「読み物」ではなく、**実行すると本当に動く画像検索エンジン**ができあがり、その性能まで自分で測ります。中身は、大きく 2 部構成です。

**Part A: CLIP 埋め込みで本物の検索エンジンを組む。** 合成画像（色×形の 48 枚）を CLIP で埋め込み → L2 正規化 → `IndexIDMap2` に `add_with_ids`（1000 番台の任意 ID）→ メタデータ（ID→ラベル）を SQLite に保存 → `write_index` で `.faiss` と `.db` を**セット永続化** → まっさらな状態から `read_index` で読み戻し（=再起動を模す）→ **画像→画像**検索と、CLIP 共有空間を使った**テキスト→画像**検索（`a photo of a red circle` → 赤い円）を実行します。最後に、同ラベルを正例とした **retrieval mAP@5** で埋め込み自体の良さを定量化します（合成データでは mAP ≈ 0.99 と、クラスタ構造がきれいに引けていることが数字で出ます）。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="ミニプロジェクトPart Aの流れ。CLIP埋め込みとL2正規化、IndexIDMap2へadd_with_ids、SQLite保存とwrite_index、read_indexで再起動を模し、画像やテキストで検索、mAP@5で評価する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">Part A：構築 → 永続化 → 再読込 → 検索 → mAP 評価</text><rect x="30" y="78" width="180" height="62" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="120" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">CLIP 埋め込み</text><text x="120" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#ea580c">→ L2 正規化</text><rect x="240" y="78" width="180" height="62" rx="8" fill="#f4f4f5" stroke="#16a34a" stroke-width="2"/><text x="330" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">IndexIDMap2 に</text><text x="330" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">add_with_ids</text><rect x="450" y="78" width="180" height="62" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="540" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">SQLite 保存 +</text><text x="540" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">write_index</text><line x1="210" y1="109" x2="234" y2="109" stroke="#71717a" stroke-width="2"/><polygon points="240,109 230,104 230,114" fill="#71717a"/><line x1="420" y1="109" x2="444" y2="109" stroke="#71717a" stroke-width="2"/><polygon points="450,109 440,104 440,114" fill="#71717a"/><line x1="540" y1="140" x2="540" y2="216" stroke="#71717a" stroke-width="2"/><polygon points="540,222 535,212 545,212" fill="#71717a"/><rect x="450" y="222" width="180" height="62" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="540" y="248" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">read_index</text><text x="540" y="268" text-anchor="middle" font-size="12.5" fill="#52525b">（再起動を模す）</text><rect x="240" y="222" width="180" height="62" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="330" y="248" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">画像/テキスト</text><text x="330" y="268" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">で検索</text><rect x="30" y="222" width="180" height="62" rx="8" fill="#f4f4f5" stroke="#16a34a" stroke-width="2"/><text x="120" y="248" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">mAP@5 で</text><text x="120" y="268" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">埋め込みを評価</text><line x1="450" y1="253" x2="426" y2="253" stroke="#71717a" stroke-width="2"/><polygon points="420,253 430,248 430,258" fill="#71717a"/><line x1="240" y1="253" x2="216" y2="253" stroke="#71717a" stroke-width="2"/><polygon points="210,253 220,248 220,258" fill="#71717a"/><text x="330" y="312" text-anchor="middle" font-size="12.5" fill="#52525b">.faiss と .db は必ずセットで保存・復元する</text></svg><figcaption><b>Part A</b> は一直線のパイプラインです。画像を <b>CLIP で埋め込み → L2 正規化</b>し、<code>IndexIDMap2</code> に <code>add_with_ids</code> で任意IDごと格納、<b>SQLite にメタ</b>を保存して <code>write_index</code>（<code>.faiss</code> ＋ <code>.db</code> をセット）。次に <code>read_index</code> で<b>再起動を模して</b>読み戻し、<b>画像／テキストで検索</b>、最後に <b>retrieval mAP@5</b> で埋め込み自体の良さを測ります。</figcaption></figure>

**Part B: 規模が増えたときの ANN を評価して設定を「推薦」する。** 画像 48 枚では Flat で十分なので、ANN の威力は規模を上げて見せます。具体的には、2 万件・64 次元の合成埋め込みに対し、**ground truth を厳密 `IndexFlat` で作り**、`IVFFlat`（`nprobe` スイープ）と `HNSW`（`efSearch` スイープ）の **Recall@10 と QPS** を測定します。そのうえで「**目標 Recall（既定 0.95）を満たす中で最速（QPS 最大）の設定**」を自動で選んで推薦し、QPS-recall 曲線にその点を星印で重ねます。こうして「速度と精度のどちらを取るか」を“感覚”ではなく数字で決める、という実務の意思決定をコードに落とし込んでいます。

<figure class="lec-fig"><svg viewBox="0 0 660 348" role="img" aria-label="ミニプロジェクトPart Bの流れ。2万件64次元データから厳密IndexFlatで正解を作り、IVFとHNSWをスイープしてRecallとQPSを測定し、目標Recallを満たす最速設定を推薦する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">Part B：正解は Flat で作り、ANN を Recall・QPS で評価</text><rect x="24" y="150" width="120" height="64" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="84" y="178" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">2万件</text><text x="84" y="198" text-anchor="middle" font-size="13" fill="#52525b">64 次元</text><rect x="188" y="72" width="216" height="60" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="296" y="97" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">厳密 IndexFlat で</text><text x="296" y="118" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">正解(GT) を作る</text><rect x="188" y="232" width="216" height="60" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="296" y="257" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">IVF・HNSW を</text><text x="296" y="278" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">nprobe・efSearch スイープ</text><rect x="448" y="152" width="188" height="60" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="542" y="177" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">Recall@10 ・ QPS</text><text x="542" y="197" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">を測定</text><line x1="148" y1="168" x2="181" y2="118" stroke="#71717a" stroke-width="2"/><polygon points="186,110 184.7,121.1 176.3,115.6" fill="#71717a"/><line x1="148" y1="196" x2="181" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="186,254 176.3,248.4 184.7,242.9" fill="#71717a"/><line x1="406" y1="110" x2="440" y2="162" stroke="#71717a" stroke-width="2"/><polygon points="446,170 436.3,164.5 444.6,158.9" fill="#71717a"/><line x1="406" y1="254" x2="440" y2="202" stroke="#71717a" stroke-width="2"/><polygon points="446,194 444.6,205.1 436.3,199.6" fill="#71717a"/><line x1="542" y1="212" x2="542" y2="264" stroke="#71717a" stroke-width="2"/><polygon points="542,270 537,260 547,260" fill="#71717a"/><rect x="440" y="270" width="204" height="62" rx="8" fill="#f4f4f5" stroke="#16a34a" stroke-width="2"/><text x="542" y="294" text-anchor="middle" font-size="13.5" font-weight="700" fill="#15803d">目標 Recall 0.95 を満たす</text><text x="542" y="316" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">最速設定を ★ で推薦</text></svg><figcaption><b>Part B</b> は規模を上げて ANN を評価します。<b>2万件・64次元</b>のデータに対し、<b>正解(ground truth) は必ず厳密 <code>IndexFlat</code></b> で作り（ANN 自身で作ると満点になり無意味）、<code>IVFFlat</code>（<code>nprobe</code>）と <code>HNSW</code>（<code>efSearch</code>）をスイープして <b>Recall@10 と QPS</b> を測定します。<b>目標 Recall（既定 0.95）を満たす中で最速</b>の設定を自動で選び、QPS-recall 曲線に ★ で重ねます。</figcaption></figure>

実行すると、`lectures/17_faiss_image_search/outputs/` に `mini_project_search.png`（検索結果サムネ）、`mini_project_curve.png`（推薦点つき QPS-recall 曲線）、`mini_project_report.json`（エンジン統計＋mAP＋ANN 推薦）、`mini_project_engine.faiss` ＋ `mini_project_meta.db`（永続化したエンジン）が出ます。

```bash
uv run python lectures/17_faiss_image_search/mini_project.py
```

**腕試し（発展課題）。** 余力があれば、次を足してみてください。(1) `data/17_faiss_image_search/` に実画像を置いて差し替える（自動で優先読込されます）。(2) エンジンを `IndexIDMap2(IndexIVFFlat(...))` に替え、Part B の推薦設定を Part A の本番エンジンに反映する。(3) 検索結果のしきい値（コサイン < 0.2 は「該当なし」とする）を入れて“見つからない”を表現する。(4) 画像を逐次 `add_with_ids` しながら一定間隔で `write_index` する**インクリメンタル更新**版に拡張する（第40・41回の `stream/writer.py` の予行演習）。

## ✅ 到達チェックリスト

この章を「マスターした」と言えるのは、次を**AI 補助なしで**できるときです。手を動かして、1 つずつ潰していってください。

- [ ] FAISS の基本ループ（`index = faiss.IndexXxx(d)` → `add` → `search(xq, k)` が `D, I` を返す）をそらで書ける。
- [ ] index に渡す配列を必ず `np.ascontiguousarray(x, dtype=np.float32)` で **float32・C連続**にする理由を説明できる。
- [ ] `IndexFlatL2`（小さいほど近い）と `IndexFlatIP`（大きいほど近い）の**ソート向きの違い**を言える。
- [ ] **L2 正規化してから `IndexFlatIP`** で内積＝コサイン類似度になること、DB 側・クエリ側の**両方**を正規化する必要があることを説明できる。
- [ ] `faiss.normalize_L2` が **in-place（破壊的）**であることを知っていて、非破壊版と使い分けられる。
- [ ] `IndexIDMap`/`IndexIDMap2` ＋ `add_with_ids` で任意の `int64` ID を付け、メタデータは**別管理（SQLite）**して ID から引く設計を書ける。
- [ ] `write_index`/`read_index` で永続化でき、**メタDB と `.faiss` をセットで**保存・整合させる必要性を説明できる。
- [ ] `IVFFlat` は **train 必須**で `nprobe` が、`HNSW` は **train 不要**で `efSearch` が精度↔速度のダイヤル、と区別できる。
- [ ] `index_factory`（`"IVF100,PQ8"` 等）で組み、**PQ がメモリを激減させる代わりに精度が落ちる**トレードオフを数字で示せる。
- [ ] **Recall@k を集合一致（`np.intersect1d`）で自前計算**でき、**ground truth は必ず厳密 `IndexFlat` で作る**鉄則を守れる。
- [ ] `nprobe`/`efSearch` をスイープして **QPS-recall 曲線**を描き、用途に応じた設定を定量的に選べる。
- [ ] CLIP の `get_image_features`/`get_text_features` が **v5 ではオブジェクトを返し `.pooler_output` が未正規化**である落とし穴を回避できる。
- [ ] `search` 結果 `I` に紛れる **`-1` をガード**してからメタDB を引ける。
- [ ] 演習 `exercises.py` を**全10問 PASS**できる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/17_faiss_image_search/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 任意の配列を FAISS 仕様（`float32`・C連続）に変換して返す（`ex1_to_faiss_array` の TODO）。
2. ベクトルを L2 正規化してから `IndexFlatIP` に `add` し、コサイン類似度で検索できる Flat インデックスを作って返す（`ex2_cosine_index` の TODO）。
3. `IndexIDMap` で素の index を包み、任意の `int64` ID を付けて検索し、近傍の ID 行列を返す（`ex3_idmap_search` の TODO）。
4. index を `write_index` で保存し、`read_index` で読み戻した index の `ntotal` を返す（`ex4_save_load` の TODO）。
5. 各クエリで厳密解と ANN の上位 k 件の集合一致から Recall@k を求め、全クエリで平均する（`ex5_recall_at_k` の TODO）。
6. `IVFFlat`（転置インデックス）を `train` してから `nprobe` を設定して検索し、ID 行列を返す（`ex6_ivf_search` の TODO）。
7. `HNSW`（近傍グラフ・train 不要）を構築し `efSearch` を設定して検索し、ID 行列を返す（`ex7_hnsw_search` の TODO）。
8. 検索結果の ID 列をラベルに変換し、`-1` は `(none)`・未登録は `(missing)` にガードする（`ex8_guard_lookup` の TODO）。
9. `index_factory` で PQ 圧縮インデックス（例 `"IVF16,PQ8"`）を組み、`train`→`add` して返す（`ex9_factory_index` の TODO）。
10. 類似度降順のラベル列から Average Precision@k（retrieval mAP の素）を計算する（`ex10_average_precision` の TODO）。

## ❓ よくある落とし穴・FAQ・デバッグ

本文 §13 の「症状→原因→対処」表と合わせて、つまずきやすい点を Q&A 形式で補足します。

**Q. `search` が `RuntimeError` で落ちる／結果がデタラメ。** まずは配列の `dtype` と `flags['C_CONTIGUOUS']` を `print` してください。`float64` や非連続（スライス・転置の結果）が、最頻原因です。`as_faiss_array`（= `np.ascontiguousarray(x, dtype=np.float32)`）を、index に渡す直前で必ず通します。次に、`index.d`（次元）と `xq.shape[1]` が一致しているかを確認します（モデルを変えて埋め込み次元が変わると、静かにバグります）。

**Q. コサインのつもりが順位が変。** `IndexFlatIP` を使っているのに正規化を忘れている、あるいは**片側しか正規化していない**のが定番です。DB もクエリも `normalize_L2`（or 非破壊版 `l2_normalize`）を通します。検算は、「クエリに DB の点そのものを入れたら、その点が top-1 でスコア ≈ 1.0」になるかどうかです。ならなければ、正規化漏れを疑ってください。

**Q. 正規化したら元の配列が壊れた。** `faiss.normalize_L2(x)` は **in-place** で `x` 自体を書き換えます。未正規化ベクトルを後で使うなら、コピーしてから渡すか、新しい配列を返す `l2_normalize` を使います。

**Q. `add` で例外（`Error: assertion ... is_trained`）。** `IVF`/`PQ`/`OPQ` 系は、`add` の前に `train(代表データ)` が必須です。一方 `HNSW` と `Flat` は train 不要です（`index_factory("HNSW32")` の `train` は no-op）。

**Q. PQ を train したら大量の WARNING が出る。** `WARNING clustering N points to 256 centroids: please provide at least ...` は、PQ の各サブ量子化器が 256 個（8bit）の代表点を学習するのに学習データが少ない、という**警告**で、エラーではありません（faiss は自動でサブサンプルして続行します）。本物の大規模データでは、十分な学習点を用意します。学習用の合成データでは、この警告が出ても `exit 0` で完走します。

**Q. IVF の Recall がやけに低い。** `nprobe=1`（既定）のままが、原因の筆頭です。`nprobe` を 5〜数十に上げると、Recall が大きく改善します（`mini_project.py` の Part B で `nprobe=1→0.79`、`nprobe=5→1.0` と動くのが見られます）。

**Q. ANN の評価が満点で“怪しい”。** ground truth を ANN 自身の結果から作ってしまうと、「自分で採点して満点」になります。**正解は必ず厳密 `IndexFlat` で**作ってください。なお Recall は「ANN が厳密検索にどれだけ忠実か」、retrieval mAP は「同カテゴリを上位に集められるか（埋め込みの良さ）」を見るもので、**観点が違う**ので両方を見ます。

**Q. メタDB 参照でクラッシュする。** `k > ntotal` や IVF で近傍が足りないとき、`I` に `-1` が混じります。`i < 0` を**早期 return で弾いて** `(none)` 等に置換してから、DB を引きます。

**Q. `'...Pooling' object has no attribute 'shape'`（CLIP）。** transformers v5 で、`get_image_features`/`get_text_features` の戻り値が `BaseModelOutputWithPooling` オブジェクトに変わりました。射影ベクトルは `.pooler_output` に入っており、しかも**未正規化**です。そこで `out.pooler_output` を取り出し、`F.normalize`（or `l2_normalize`）してから検索します。

**Q. `AutoImageProcessor` がエラーになる。** transformers v5 では、画像プロセッサが torchvision バックエンドの fast 実装のみになりました。そのため `torchvision` 未導入だと落ちます（`dl` グループを入れてください）。なお、旧 `AutoFeatureExtractor` は廃止です。

**Q. `pip install faiss-gpu` が見つからない。** そのパッケージ名は存在しません。CPU は `faiss-cpu`、GPU は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA, CUDA 12.4 系限定）で、両者は `import faiss` 名前空間を奪い合うため、**同一環境に共存させない**でください。

**デバッグの定石。** ① 次元 `index.d == xq.shape[1]` を最初に `assert`。② 「DB の点をクエリにして自分が top-1・スコア≈1.0」で配線確認。③ `index.ntotal` を `add`/`read_index` 前後で突き合わせる。④ Recall が変なら、ground truth を Flat で作り直す。⑤ スコアが想定外なら、正規化の有無と L2/IP の向きを疑う。

## 🚀 発展トピック・参考

本章で骨格は身についたので、ここから先は「規模・精度・運用」を深掘りする入り口を示します。

- **スカラー量子化 / より強い圧縮**: `IndexScalarQuantizer`（`SQ8` 等）は PQ より手軽にメモリを削減できる中間策。`index_factory("IVF1024,SQ8")` のように混ぜて使う。PQ/OPQ/SQ・HNSW の組み合わせ最適化は FAISS Wiki の "Guidelines to choose an index" が決定版です。
- **OPQ と再順位付け（re-ranking）**: `OPQ` で回転してから PQ すると量子化誤差が下がる。さらに ANN で粗く絞ってから厳密スコアで上位を並べ替える 2 段検索（`IndexRefineFlat`）は、メモリと精度を両取りする実務常套手段。
- **大規模・GPU**: 数千万件規模では IVF のセル数を増やし、`faiss.index_cpu_to_gpu(res, 0, index)` の一行で GPU に載せ替える（検索 API は不変）。GPU 版は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA 限定）。複数 GPU/シャーディングは `IndexShards`。
- **高レベル API**: 検索を素早く試すなら `sentence-transformers`（`SentenceTransformer('clip-ViT-B-32').encode(..., normalize_embeddings=True)` ＋ `util.cos_sim`）や `open-clip-torch`（LAION 学習・MobileCLIP/SigLIP 系）も対比教材として有用。本章で“中身”を理解したうえで使うと納得感が違います。
- **評価をさらに厳密に**: `torchmetrics.retrieval`（`RetrievalRecall`/`RetrievalMAP`/`RetrievalMRR`/`RetrievalNormalizedDCG`）で Recall@k・mAP・MRR・nDCG を正準実装と突き合わせる。検索評価の指標体系は第14回・第33回とも接続します。
- **実運用設計**: インクリメンタル更新（`add_with_ids` しながら検索可能を維持し、定期 `write_index`）、メタDB（SQLite）との整合・バージョン管理、分布が変わったときの `train` 再構築。これらは最終応用 **Cluster-CLIP（第40・41回）** の `stream/writer.py` がまさに実例で、本章の `mini_project.py` はその予行演習になっています。
- **公式リファレンス**: FAISS Wiki（<https://github.com/facebookresearch/faiss/wiki>）、CLIP/transformers（<https://huggingface.co/docs/transformers>）。索引選択に迷ったら Wiki の "The index factory" と "Guidelines to choose an index" を最初に読むのが近道です。

## 💡 実践ユースケース集

ここまでで身につけた「埋め込み → 正規化 → add → search」は、現実の小ツールに落とすと、一気に実用品になります。`mini_project.py` が“評価まで含む総合課題”なのに対し、ここでは **1 本で完結し、すぐ自分のデータで動かせる現実の応用** を 3 つ紹介します。このうち 1 つ目は、このモジュール同梱の `use_case.py` として実際に動きます。

### 1.（同梱・動く）逆画像検索 — フォルダの「似てる／そっくり」を引く `use_case.py`

- **何に使うか**: 「この写真、もう持ってたっけ？」を判定する逆画像検索（reverse image search）。写真整理での重複削除、素材ライブラリの再利用チェック、スクショやロゴの重複検出など。クエリ画像 1 枚を投げると、フォルダ内の類似画像を上位から並べ、**コサイン類似度がしきい値以上のものを「near-duplicate（ほぼ重複）」とフラグ付け**して報告します。
- **作り方の要点**: フォルダの画像を CLIP で埋め込み → `l2_normalize` → `IndexIDMap2(IndexFlatIP)` に `add_with_ids` でギャラリー索引を構築 → `write_index` で保存し `read_index` で読み戻す（実ツールは「一度作って何度も引く」）→ クエリ画像を同じ手順でベクトル化して `search` → コサイン `>= DUP_THRESHOLD` を重複と判定。ID は `0..N-1` を振り、`images[id]` でそのままサムネイルを引けるようにしています。
- **注意**: コサイン類似度なので **DB 側・クエリ側の両方を正規化**します（片側だけだと崩れる）。`k > ntotal` や近傍不足で混じる **`-1` は弾いて**から扱います。重複の閾値（既定 `0.90`）は埋め込みモデルやデータで最適値が変わるので、自分のデータで何件かを目視して調整してください（厳しすぎると取りこぼし、緩すぎると誤検出）。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="逆画像検索use_case.pyの流れ。ギャラリー索引を作り、クエリ画像をベクトル化してsearchし、コサイン類似度が0.90以上なら重複と判定する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="28" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">逆画像検索（use_case.py）：閾値で near-duplicate を判定</text><rect x="20" y="120" width="150" height="70" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="95" y="144" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">ギャラリー索引</text><text x="95" y="163" text-anchor="middle" font-size="11.5" fill="#52525b">CLIP→正規化→add</text><text x="95" y="180" text-anchor="middle" font-size="11.5" fill="#52525b">→ 永続化</text><rect x="210" y="120" width="160" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="290" y="144" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">クエリ画像を</text><text x="290" y="163" text-anchor="middle" font-size="12" fill="#3f3f46">ベクトル化 →</text><text x="290" y="182" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">search(k)</text><rect x="410" y="120" width="160" height="70" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="490" y="148" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">コサイン類似度</text><text x="490" y="172" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">≥ 0.90 ?</text><line x1="170" y1="155" x2="204" y2="155" stroke="#71717a" stroke-width="2"/><polygon points="210,155 200,150 200,160" fill="#71717a"/><line x1="370" y1="155" x2="404" y2="155" stroke="#71717a" stroke-width="2"/><polygon points="410,155 400,150 400,160" fill="#71717a"/><line x1="490" y1="120" x2="490" y2="98" stroke="#71717a" stroke-width="2"/><polygon points="490,92 485,102 495,102" fill="#71717a"/><line x1="490" y1="190" x2="490" y2="212" stroke="#71717a" stroke-width="2"/><polygon points="490,218 485,208 495,208" fill="#71717a"/><rect x="410" y="40" width="160" height="52" rx="8" fill="#fff7ed" stroke="#dc2626" stroke-width="2"/><text x="490" y="62" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">はい → [DUP]</text><text x="490" y="82" text-anchor="middle" font-size="12" fill="#dc2626">near-duplicate</text><rect x="410" y="218" width="160" height="52" rx="8" fill="#f4f4f5" stroke="#16a34a" stroke-width="2"/><text x="490" y="240" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">いいえ → 通常の</text><text x="490" y="260" text-anchor="middle" font-size="12.5" fill="#15803d">類似ヒット</text></svg><figcaption><b>逆画像検索</b>（<code>use_case.py</code>）の流れです。フォルダ画像を <b>CLIP→正規化→<code>IndexIDMap2(IndexFlatIP)</code></b> に <code>add_with_ids</code> してギャラリー索引を作り永続化、クエリ画像を同じ手順でベクトル化して <code>search(k)</code>。返ったコサイン類似度が<b>しきい値（既定 0.90）以上なら near-duplicate（<code>[DUP]</code>）</b>、未満なら通常の類似ヒットと判定します。<b>DB・クエリ両方を正規化</b>し、<code>-1</code> は弾きます。</figcaption></figure>

```bash
# 合成画像（色×形）で即実行（exit 0）。結果は lectures/17_faiss_image_search/outputs/ に保存
uv run python lectures/17_faiss_image_search/use_case.py

# 外部の画像をクエリにする
QUERY_IMAGE=/path/to/photo.jpg uv run python lectures/17_faiss_image_search/use_case.py
```

**自分のデータで実運用にする**: `data/17_faiss_image_search/` に自分の画像（`*.png` / `*.jpg` など）を置くだけで、合成画像の代わりにそれらが**自動で優先**して読み込まれます（ファイル名がラベルになります）。あとは上のコマンドを実行すれば、あなたのフォルダに対する逆画像検索になります。出力は `use_case_reverse_search.png`（クエリ＋上位サムネ、重複は `[DUP]` 印）、`use_case_report.json`（判定の詳細）、`use_case_index.faiss` ＋ `use_case_meta.json`（永続化したギャラリー索引）です。

### 2. 重複ファイル・クラスタの一括検出（フォルダ丸ごと棚卸し）

- **何に使うか**: 1 枚クエリではなく、**フォルダ内の全ペアから重複グループを洗い出す**棚卸しツール。バックアップやダウンロードフォルダの重複整理に効きます。
- **作り方の要点**: 全画像を索引に入れ、各画像を順にクエリとして `search(k=数件)` し、コサイン `>= 閾値` の相手を「重複候補」として無向グラフの辺にする → 連結成分（Union-Find）でグループ化。`IndexIDMap2` の任意 ID と SQLite メタデータ（`02_idmap_persist_sqlite.py`）を併用すると、各グループに元ファイルパスを添えて「どれを残しどれを消すか」まで提案できます。
- **注意**: 自分自身（クエリ＝DB の同一画像）が必ず top-1 で来るので除外します。総当たりは O(N²) なので、件数が増えたら IVF/HNSW（`03_ivf_hnsw_pq.py`）で候補を絞ってから厳密スコアで確認する 2 段構えにします。

### 3. テキストで自分の画像フォルダを引く（マルチモーダル検索）

- **何に使うか**: 「`a red car`」「`sunset over the sea`」のような**自然文で手元の写真を検索**する。タグ付けしていない大量の写真から目的の 1 枚を探すのに便利です。
- **作り方の要点**: CLIP は画像とテキストを**同じ空間**に置くので、`use_case.py` の索引はそのまま流用できます。クエリだけ `embedder.encode_texts([...])` に差し替え → 正規化 → 同じ `index.search` を呼ぶだけ（FAISS から見ればテキスト由来か画像由来かは無関係、ただのベクトル）。第 16 章「CLIP/SigLIP ゼロショット」と本章を橋渡しする応用です。
- **注意**: テキスト検索は CLIP バックエンド時のみ（色記述子フォールバックでは不可）。`get_text_features` の戻り値は **v5 では未正規化の `.pooler_output`** なので、検索前に必ず `F.normalize`（or `l2_normalize`）します。スコアは絶対値より**相対順位**で見て、しきい値は用途ごとに調整します。

**練習（拡張）アイデア**: (1) `data/17_faiss_image_search/` に自分の写真を置いて `use_case.py` を実運用にする。(2) `DUP_THRESHOLD` を振って重複判定の厳しさを体感する。(3) クエリをテキストに差し替えてマルチモーダル検索にする（応用 3）。(4) 索引を `IndexIDMap2(IndexIVFFlat(...))` に替えて大量画像でも高速化する。(5) メタデータを SQLite に移してファイルパス・撮影日時・サムネを持たせ、重複の「削除候補」を提案する CLI に育てる。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ faiss-cpu 1.14.2 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub（transformers 依存）／ scikit-learn 1.9.0 ／ torchmetrics 1.9.0 ／ matplotlib 3.10 系。
> モデル: `openai/clip-vit-base-patch32`（初回のみ HuggingFace Hub から重みDL→ローカルキャッシュ）。GPU版 FAISS は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA, CUDA 12.4 系限定、`faiss-cpu` と排他）。
