# 第44回 埋め込みのクラスタリング — 画像・テキスト・クロスモーダルを教師なしで束ねる

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers ほか）・`metrics`（scikit-learn）・`vector`（faiss-cpu, 任意）。CPU だけで完走します（初回のみ CLIP の重みを HuggingFace からダウンロード）。
> 前提: 第16回（CLIP ゼロショット & 検索）・第17回（FAISS 検索）。関連: 第15回（埋め込み）・第30回（顔クラスタリング）。

## 🎯 この章のゴール

第16・17回では、「クエリに近いものを並べる」**検索（retrieval）** を学びました。本章が扱うのは、その裏返しにあたる**クラスタリング（grouping）** です。すなわち、クエリも正解ラベルも無い大量の埋め込みを、似たもの同士で自動的に束へと分けていきます。スマホの写真アプリが写真を「人物」ごとにまとめたり、未整理の画像コレクションを「だいたいこういう種類が N 個ある」と俯瞰したりするのが、まさにこれにあたります。この章を終えるころには、CLIP/ResNet で得た埋め込み（画像・テキスト・顔）を、AI の補助なしに自分の手でクラスタリングし、その品質を数字で評価し、さらに 2D に可視化して検証できるようになっているはずです。

具体的な到達点は5つあります。第一に、埋め込みを **L2 正規化**してから **k-means** で束ね、その結果を **silhouette / NMI / purity / homogeneity** で評価できること。第二に、k-means が必要とする**クラスタ数 k** を、**エルボー法**と**シルエット法**でラベル無しに選べること。第三に、k を決めずに済む **DBSCAN** と **AgglomerativeClustering** を使い分け、それぞれの限界（DBSCAN の単一 eps 問題など）を体感すること。第四に、**テキスト**のクラスタリングと**クロスモーダル**（画像とテキストを同じ空間で扱う）に踏み込み、その過程で CLIP の **モダリティギャップ**という落とし穴を知ること。そして第五に、**PCA / t-SNE**（UMAP は任意）でクラスタを可視化し、結果を目で検証できることです。

本章のスクリプトはすべて、ネット接続もデータセットのダウンロードも無しで完走できるよう、入力を**その場で合成**します。画像は「赤い丸」「青い三角」のように**色×形で 6 グループ**を作り、テキストは「動物/乗り物/食べ物/天気」の **4 トピック**を用意します。ここで重要なのは、これらの**正解ラベルをクラスタリングには一切渡さず、評価（答え合わせ）にだけ使う**点です。これは本章の精神そのものです——そもそもクラスタリングはラベルが無いからこそ行う操作であり、ラベルはあくまで「うまく束ねられたか」を後から確認するためだけのものだからです。なお、ダウンロードが走るのは初回の CLIP 重み取得のときだけで、以降はローカルキャッシュから即座に起動します。

---

## 1. 検索とクラスタリングはどう違うのか（直感）

検索とクラスタリングは、どちらも「埋め込み空間での近さ」を使う双子のようなタスクですが、問いの形がまったく違います。まず**検索**は、「**この1点（クエリ）**に近いものを、近い順に並べて」という問いであり、答えはランキングになります。つまり基準点があり、相対的な順位だけが問題になります。これに対して**クラスタリング**は、「**全体**を、似たもの同士のいくつかの束に分けて」という問いであり、答えは各点へのグループ ID の割り当てになります。こちらには基準点が無く、データ全体の構造を一度に捉えます。この違いがあるからこそ、検索は**教師あり的に評価**でき（クエリの正解が分かる）、クラスタリングは本質的に**教師なし**になるのです。

<figure class="lec-fig"><svg viewBox="0 0 640 280" role="img" aria-label="検索はクエリ1点に近い順のランキング、クラスタリングは全体を束に分けて各点にグループIDを割り当てる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="20" y="44" width="288" height="216" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="332" y="44" width="288" height="216" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="164" y="70" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">検索（retrieval）</text><text x="476" y="70" text-anchor="middle" font-size="16" font-weight="700" fill="#2563eb">クラスタリング（clustering）</text><line x1="72" y1="170" x2="140" y2="110" stroke="#ea580c" stroke-width="1.6"/><line x1="72" y1="170" x2="165" y2="165" stroke="#ea580c" stroke-width="1.6"/><line x1="72" y1="170" x2="120" y2="215" stroke="#ea580c" stroke-width="1.6"/><circle cx="215" cy="105" r="6" fill="#d4d4d8"/><circle cx="225" cy="200" r="6" fill="#d4d4d8"/><circle cx="205" cy="150" r="6" fill="#d4d4d8"/><circle cx="140" cy="110" r="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><circle cx="165" cy="165" r="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><circle cx="120" cy="215" r="6" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><circle cx="72" cy="170" r="10" fill="#ea580c" stroke="#ffffff" stroke-width="2"/><text x="72" y="200" text-anchor="middle" font-size="12" fill="#c2410c">クエリ</text><text x="152" y="104" font-size="13" fill="#c2410c">①</text><text x="178" y="162" font-size="13" fill="#c2410c">②</text><text x="106" y="222" font-size="13" fill="#c2410c">③</text><text x="164" y="248" text-anchor="middle" font-size="12.5" fill="#52525b">1点に近い順 → ランキング</text><ellipse cx="408" cy="115" rx="42" ry="30" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><ellipse cx="535" cy="150" rx="42" ry="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><ellipse cx="430" cy="210" rx="42" ry="28" fill="#ffffff" stroke="#16a34a" stroke-width="1.5"/><circle cx="392" cy="106" r="5" fill="#ea580c"/><circle cx="418" cy="120" r="5" fill="#ea580c"/><circle cx="402" cy="128" r="5" fill="#ea580c"/><circle cx="520" cy="140" r="5" fill="#2563eb"/><circle cx="548" cy="156" r="5" fill="#2563eb"/><circle cx="532" cy="162" r="5" fill="#2563eb"/><circle cx="415" cy="202" r="5" fill="#16a34a"/><circle cx="442" cy="216" r="5" fill="#16a34a"/><circle cx="425" cy="222" r="5" fill="#16a34a"/><text x="360" y="119" font-size="15" font-weight="700" fill="#c2410c">0</text><text x="487" y="153" font-size="15" font-weight="700" fill="#2563eb">1</text><text x="382" y="212" font-size="15" font-weight="700" fill="#16a34a">2</text><text x="476" y="248" text-anchor="middle" font-size="12.5" fill="#52525b">全体 → 各点にグループIDを割り当て</text></svg><figcaption>検索とクラスタリングの「問いの形」の違いです。<b>検索</b>は1点（クエリ）に近いものを近い順に並べる<b>ランキング</b>で、基準点があり相対順位だけを見ます。<b>クラスタリング</b>は<b>全体</b>を似たもの同士の束に分け、各点に<b>グループID</b>（0/1/2…）を割り当てる<b>教師なし</b>の操作です。前者は教師あり的に評価でき、後者は本質的に教師なしになります。</figcaption></figure>


この違いは、実装の仕方にもそのまま効いてきます。検索は「正規化した埋め込みの内積を取って topk」で済み、第17回ではこれを FAISS で大規模化しました。一方クラスタリングは「全点の距離関係から束を見つける」処理なので、k-means なら反復、DBSCAN なら密度の連結、Agglomerative なら階層的な併合と、専用のアルゴリズムが必要になります。評価も別物で、検索が Recall@k / mAP（正解ランキングとの一致）を使うのに対し、クラスタリングでは silhouette（束の締まり具合）や NMI（既知グループとの一致）を使います。本章では、この**クラスタリング側の道具一式**を揃えていきます。

とはいえ、両者は地続きでもあります。クラスタの「代表ベクトル（中心）」を作れば、新しい点がどの束に近いかを**検索**で判定できますし、逆に検索結果を連結していけば粗いクラスタが得られます。実際、第40・41回の Cluster-CLIP は、「密な CLIP 特徴をクラスタリングして領域を作り、それを検索インデックスにする」という、まさに両者を組み合わせたパイプラインでした。本章の到達点は、その土台となる「埋め込みを束ねる」基本操作を、原理から自分の手で書けるようにすることです。

## 2. すべての出発点 — 埋め込みと L2 正規化

クラスタリングの入力は、「1サンプル＝1本のベクトル」の集まりです。本章では、画像も文もすべて CLIP で **512 次元のベクトル**に変換します（第15・16回で学んだ `get_image_features` / `get_text_features` の `.pooler_output`）。この変換を担うのが `cluster_lab.embed_images` / `embed_texts` で、(N, 512) の float32 行列を返します。ここで注意したいのは、こうして得た生の埋め込みは**ノルムがバラバラ（CLIP 画像ベクトルは約 11）**だという点です。ベクトルの「長さ」は明るさやコントラスト、語数などで揺れやすく、距離計算がそれに引きずられると、束ね方が乱れてしまいます。

そこで本章でも、**最初に必ず L2 正規化**を行います。各ベクトルを長さ 1 にそろえると、ユークリッド距離・内積・コサイン類似度が一対一に対応し、距離は純粋に「**向きの違い**」だけを測るようになります。具体的には、正規化後のベクトル `u, v`（`|u|=|v|=1`）について `コサイン類似度 = u·v`、`コサイン距離 = 1 - u·v`、そして `ユークリッド距離² = 2(1 - u·v)` が成り立ちます。したがって正規化さえしておけば、k-means の既定（ユークリッド）も DBSCAN/Agglomerative の `metric="cosine"` も、本質的に同じ「向きの近さ」で動きます。`cluster_lab.embed_*` はこの正規化まで込みでベクトルを返すので、各スクリプトは安心して距離ベースの手法を当てられます。

<figure class="lec-fig"><svg viewBox="0 0 600 290" role="img" aria-label="L2正規化でベクトルを長さ1の単位円に乗せると距離は向きの違いだけを測り、コサイン距離とユークリッド距離が対応する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="16" y="44" width="320" height="226" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="348" y="44" width="236" height="226" rx="8" fill="#fff7ed" stroke="#f97316" stroke-width="1.5"/><polyline points="220,235 216,201 203,170 182,143 155,122 124,109 90,105" fill="none" stroke="#71717a" stroke-width="1.8"/><line x1="90" y1="235" x2="155" y2="122" stroke="#ea580c" stroke-width="2"/><line x1="155" y1="122" x2="178" y2="83" stroke="#ea580c" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="178,83 170,87 176,95" fill="#ea580c"/><line x1="90" y1="235" x2="203" y2="170" stroke="#2563eb" stroke-width="2"/><line x1="203" y1="170" x2="220" y2="160" stroke="#2563eb" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="220,160 211,159 217,168" fill="#2563eb"/><circle cx="155" cy="122" r="5" fill="#c2410c"/><circle cx="203" cy="170" r="5" fill="#1d4ed8"/><circle cx="90" cy="235" r="4" fill="#18181b"/><polyline points="125,215 118,207 110,200" fill="none" stroke="#3f3f46" stroke-width="1.5"/><text x="138" y="201" font-size="14" fill="#3f3f46">θ</text><text x="148" y="112" font-size="14" font-weight="700" fill="#c2410c">u</text><text x="214" y="182" font-size="14" font-weight="700" fill="#1d4ed8">v</text><text x="200" y="72" text-anchor="middle" font-size="11.5" fill="#71717a">未正規化（長さ様々）</text><text x="252" y="226" font-size="11.5" fill="#52525b">単位円 |x|=1</text><text x="364" y="86" font-size="14" font-weight="700" fill="#c2410c">正規化後（|u|=|v|=1）</text><text x="364" y="120" font-size="12.5" fill="#18181b">コサイン類似度 = u·v</text><text x="364" y="152" font-size="12.5" fill="#18181b">コサイン距離 = 1 − u·v</text><text x="364" y="184" font-size="12.5" fill="#18181b">ユークリッド距離² = 2(1−u·v)</text><text x="364" y="220" font-size="12.5" fill="#15803d">→ 距離は「向きの違い」だけを測る</text></svg><figcaption><b>L2 正規化</b>は各ベクトルを長さ 1 にそろえ、すべてを<b>単位円（高次元では単位球）</b>の上に乗せます。長さ（ノルム）の揺れが消え、距離は純粋に<b>向きの違い</b>だけを測るようになります。正規化後（<code>|u|=|v|=1</code>）は <code>コサイン類似度 = u·v</code>、<code>コサイン距離 = 1 − u·v</code>、<code>ユークリッド距離² = 2(1 − u·v)</code> が一対一に対応し、k-means（ユークリッド）も cosine 指定の手法も同じ「向きの近さ」で動きます。</figcaption></figure>


この「まず正規化」は、第16回の検索でも口酸っぱく述べた鉄則です。これを忘れると、「ノルムが大きいだけの1点」がクラスタを乗っ取ってしまいます。そうならないよう、`01` で `各行ノルム≈1.000` と表示されることを毎回確認する習慣をつけてください。なお、埋め込みの作り方そのもの（CLIP の `.pooler_output` が未正規化であるといった非対称など）は第16回で深掘り済みなので、本章では「正規化済みの良いベクトルが手元にある」という前提から始め、**束ね方と評価**に集中します。

## 3. k-means — 最も基本的な束ね方

k-means は、データを **k 個の球状の束**に分ける、最も基本的なアルゴリズムです。やることは2ステップの反復だけで、まず**(E) 割り当て**として各点を最も近い「中心（centroid）」のクラスタに入れ、次に**(M) 更新**として各クラスタの所属点の平均を新しい中心にします。あとはこれを、中心が動かなくなるまで繰り返すだけです。ただし初期中心はランダムに選ぶため結果が初期値に依存しがちで、そこで `n_init=10` のように**複数回試して inertia（後述）が最小の解を採用**するのが定石です（sklearn の既定は、賢く初期中心を散らす `k-means++` 初期化）。なお、この E/M ステップは演習の問3・問4で numpy を使って自分の手で書きます。

<figure class="lec-fig"><svg viewBox="0 0 660 290" role="img" aria-label="k-meansの反復。割り当てEで各点を最も近い中心へ、更新Mで各クラスタの平均を新しい中心にし、中心が動くまで繰り返す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="24" y="64" width="150" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="258" y="64" width="150" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="492" y="64" width="150" height="150" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="99" y="52" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">① 初期化</text><text x="333" y="52" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">② 割り当て (E)</text><text x="567" y="52" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3f3f46">③ 更新 (M)</text><path d="M 567 64 C 567 26 333 26 333 58" fill="none" stroke="#c2410c" stroke-width="1.8"/><polygon points="333,66 328,56 338,56" fill="#c2410c"/><text x="450" y="20" text-anchor="middle" font-size="12.5" fill="#c2410c">中心が動くまで反復</text><circle cx="55" cy="95" r="5" fill="#d4d4d8"/><circle cx="85" cy="105" r="5" fill="#d4d4d8"/><circle cx="70" cy="130" r="5" fill="#d4d4d8"/><circle cx="130" cy="140" r="5" fill="#d4d4d8"/><circle cx="155" cy="155" r="5" fill="#d4d4d8"/><circle cx="140" cy="170" r="5" fill="#d4d4d8"/><line x1="94" y1="84" x2="106" y2="96" stroke="#c2410c" stroke-width="3"/><line x1="106" y1="84" x2="94" y2="96" stroke="#c2410c" stroke-width="3"/><line x1="104" y1="169" x2="116" y2="181" stroke="#1d4ed8" stroke-width="3"/><line x1="116" y1="169" x2="104" y2="181" stroke="#1d4ed8" stroke-width="3"/><line x1="180" y1="139" x2="250" y2="139" stroke="#71717a" stroke-width="1.8"/><polygon points="256,139 246,134 246,144" fill="#71717a"/><text x="215" y="128" text-anchor="middle" font-size="11.5" fill="#3f3f46">近い中心へ</text><circle cx="289" cy="95" r="5" fill="#ea580c"/><circle cx="319" cy="105" r="5" fill="#ea580c"/><circle cx="304" cy="130" r="5" fill="#ea580c"/><circle cx="364" cy="140" r="5" fill="#2563eb"/><circle cx="389" cy="155" r="5" fill="#2563eb"/><circle cx="374" cy="170" r="5" fill="#2563eb"/><line x1="328" y1="84" x2="340" y2="96" stroke="#c2410c" stroke-width="3"/><line x1="340" y1="84" x2="328" y2="96" stroke="#c2410c" stroke-width="3"/><line x1="338" y1="169" x2="350" y2="181" stroke="#1d4ed8" stroke-width="3"/><line x1="350" y1="169" x2="338" y2="181" stroke="#1d4ed8" stroke-width="3"/><line x1="414" y1="139" x2="484" y2="139" stroke="#71717a" stroke-width="1.8"/><polygon points="490,139 480,134 480,144" fill="#71717a"/><text x="449" y="128" text-anchor="middle" font-size="11.5" fill="#3f3f46">平均で更新</text><circle cx="523" cy="95" r="5" fill="#ea580c"/><circle cx="553" cy="105" r="5" fill="#ea580c"/><circle cx="538" cy="130" r="5" fill="#ea580c"/><circle cx="598" cy="140" r="5" fill="#2563eb"/><circle cx="623" cy="155" r="5" fill="#2563eb"/><circle cx="608" cy="170" r="5" fill="#2563eb"/><line x1="568" y1="90" x2="538" y2="110" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="578" y1="175" x2="610" y2="155" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="532" y1="104" x2="544" y2="116" stroke="#c2410c" stroke-width="3"/><line x1="544" y1="104" x2="532" y2="116" stroke="#c2410c" stroke-width="3"/><line x1="604" y1="149" x2="616" y2="161" stroke="#1d4ed8" stroke-width="3"/><line x1="616" y1="149" x2="604" y2="161" stroke="#1d4ed8" stroke-width="3"/><text x="333" y="248" text-anchor="middle" font-size="12.5" fill="#52525b">× ＝ クラスタ中心 ・ 色 ＝ 所属クラスタ</text></svg><figcaption>k-means の 2 ステップ反復です。<b>(E) 割り当て</b>で各点を最も近い<b>中心（centroid・×）</b>のクラスタに入れ、<b>(M) 更新</b>で各クラスタの所属点の<b>平均</b>を新しい中心にします（点線が中心の移動）。これを<b>中心が動かなくなるまで</b>繰り返します。初期中心に依存するので <code>n_init=10</code> 回試し、inertia 最小の解を採用します。</figcaption></figure>


正準 API は `sklearn.cluster.KMeans` で、`01_kmeans_image_embeddings.py` の核は次の3行に集約されます。L2 正規化済み埋め込みに対し、クラスタ数 `k` を指定して `fit_predict` を呼ぶだけで、各点のクラスタ ID 配列が返ります。

```python
from sklearn.cluster import KMeans
emb = cl.embed_images(images, device)            # (N, 512) L2 正規化済み
pred = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(emb)  # 各点のクラスタID
```

合成6グループ（n_per_group=6, 計36枚）でこれを回すと、**silhouette=0.759、NMI=1.000、purity=1.000、homogeneity=1.000** となり、色×形のグループを完全に当てます。実際に、`01_album_kmeans.png`（クラスタ別に画像を並べたアルバム）の各行が綺麗に1グループで揃うこと、そして `01_pca_scatter.png` で同じ色（予測クラスタ）の点がまとまることを、目で確認してください。ただし、ここでは「真の数は6」とこっそり k に教えてしまっています。実務では k は未知なのが普通であり、その選び方は次節で学びます。あわせて、k-means の弱点も押さえておきましょう。①k を事前に決める必要がある、②球状で同程度の大きさの束を仮定する（細長い束や密度の違う束は苦手）、③外れ値に弱い（中心が引っ張られる）、の3点です。

## 4. クラスタ数 k の選び方 — エルボー法とシルエット法

k-means の最大の悩みは、「k をいくつにするか」です。ラベルが無いなかで、これをどう決めればよいのでしょうか。古典的な手法が**エルボー法**です。k を増やすと **inertia**（各点と所属中心の二乗距離の総和で、クラスタの締まり具合を表す）は必ず単調に下がります。なぜなら、k=N（1点1クラスタ）で 0 になるからです。そこで、「下がり方が急に緩む“肘（elbow）”」を探します。`02` の実測では inertia が `k=2:3.09 → 4:1.57 → 6:0.69 → 8:0.52` と推移し、k=5〜6 あたりで折れ曲がります。直感的で分かりやすい一方、肘がはっきり出ないデータも多く、判断が主観的になりがちなのが弱点です。

これに対し、より自動化しやすいのが**シルエット法**です。各点について「自分のクラスタ内の平均距離 a」と「最も近い別クラスタへの平均距離 b」を測り、シルエット係数 `s = (b - a) / max(a, b)` を計算します。`s` は −1〜1 の値をとり、**1 に近いほど「自分の束に近く、隣の束から遠い」＝良い分離**を意味します。この全点の平均が `silhouette_score` であり、**これが最大になる k を選びます**。ラベルが不要なので、実運用でもそのまま使えるのが強みです。正準 API は `sklearn.metrics.silhouette_score(emb, labels, metric="cosine")` です。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="シルエット係数の幾何。点iの束内平均距離aと最も近い別の束への平均距離bからs=(b-a)/max(a,b)を計算する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><ellipse cx="155" cy="150" rx="60" ry="48" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><ellipse cx="350" cy="150" rx="60" ry="48" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><line x1="155" y1="150" x2="120" y2="130" stroke="#c2410c" stroke-width="1.4"/><line x1="155" y1="150" x2="135" y2="180" stroke="#c2410c" stroke-width="1.4"/><line x1="155" y1="150" x2="190" y2="165" stroke="#c2410c" stroke-width="1.4"/><line x1="155" y1="150" x2="175" y2="125" stroke="#c2410c" stroke-width="1.4"/><line x1="155" y1="150" x2="320" y2="130" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="155" y1="150" x2="335" y2="180" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="155" y1="150" x2="360" y2="135" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="5 3"/><circle cx="120" cy="130" r="5" fill="#ea580c"/><circle cx="135" cy="180" r="5" fill="#ea580c"/><circle cx="190" cy="165" r="5" fill="#ea580c"/><circle cx="175" cy="125" r="5" fill="#ea580c"/><circle cx="320" cy="130" r="5" fill="#2563eb"/><circle cx="360" cy="135" r="5" fill="#2563eb"/><circle cx="335" cy="180" r="5" fill="#2563eb"/><circle cx="378" cy="160" r="5" fill="#2563eb"/><circle cx="155" cy="150" r="8" fill="#ea580c" stroke="#ffffff" stroke-width="2"/><text x="155" y="154" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">i</text><text x="155" y="92" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">束 A（自分の束）</text><text x="350" y="92" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">束 B（最も近い別の束）</text><text x="120" y="238" font-size="12.5" fill="#c2410c">a ＝ 束内の平均距離（短い）</text><text x="298" y="264" font-size="12.5" fill="#2563eb">b ＝ 別の束への平均距離（長い）</text><rect x="430" y="78" width="172" height="120" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="446" y="108" font-size="14" font-weight="700" fill="#18181b">シルエット係数 s</text><text x="446" y="140" font-size="13.5" fill="#18181b">s = (b − a) / max(a,b)</text><text x="446" y="168" font-size="12.5" fill="#52525b">−1 ≤ s ≤ 1</text><text x="446" y="190" font-size="12.5" fill="#15803d">1 に近い＝良い分離</text></svg><figcaption><b>シルエット係数</b>の幾何です。点 <code>i</code> について、<b>a</b>＝自分の束内の点への平均距離（小さいほど締まっている）、<b>b</b>＝最も近い別の束への平均距離（大きいほど離れている）を測ります。<code>s = (b − a) / max(a, b)</code> は <b>−1〜1</b> の値をとり、<b>1 に近いほど「自分の束に近く隣の束から遠い」＝良い分離</b>を意味します。全点の平均が <code>silhouette_score</code> で、これを最大化する k を選びます。</figcaption></figure>


```python
from sklearn.metrics import silhouette_score
best = max(range(2, 11),
           key=lambda k: silhouette_score(emb, KMeans(k, n_init=10).fit_predict(emb), metric="cosine"))
```

実際、`02_choosing_k.py` のシルエットは `k=2:0.37 → 5:0.69 → 6:0.759 → 7:0.73 → 10:0.57` と推移し、**k=6 で綺麗にピーク**を打って、真のグループ数とぴたりと一致します。図 `02_choosing_k.png` の右パネルを見ると、ラベルを使った NMI（`k=6` で 1.0）のピークとシルエットのピークが重なっており、「ラベルが無くてもシルエットだけで正しい k を選べた」ことが見て取れます。もっとも、実務ではこの NMI 曲線は**見えません**（ラベルが無いため）。したがって現実には、シルエットや、文脈上の制約（「だいたい数十カテゴリのはず」）を頼りに k を絞り込むことになります。

## 5. k を決めない仲間 — DBSCAN と Agglomerative

「k を先に決める」のが嫌なら、クラスタ数を**データに決めさせる**手法があります。まず **DBSCAN** は密度ベースの手法で、「`eps` 半径内に `min_samples` 個以上の点があれば芯（core）」とみなして束を広げ、どの束にも届かない点は**ノイズ（ラベル -1）**として切り捨てます。クラスタ数の指定が要らず、任意形状の束を捉えられ、外れ値に強いのが長所です。正準形は `sklearn.cluster.DBSCAN(eps=..., min_samples=..., metric="cosine")` です。ただし**弱点は `eps`**にあります。これは**全体で1つだけの大域的なしきい値**なので、グループによって密度（広がり）が違うと、一つの `eps` では全グループを同時に綺麗には切り分けられません。

この限界は、本章の合成データで綺麗に再現されます。`03` でコサイン距離の構造を覗くと、**グループ内距離の最大が 0.079、グループ間距離の最小が 0.048** と、両者の範囲が**重なって**います。そのため単一 `eps` では、`0.03→6クラスタ(ただし1点ノイズ)、0.05→5、0.07→4、0.08→2` というように、しきい値をわずかに動かすだけでクラスタ数が乱高下し、真の6に安定して届きません。これに対し **AgglomerativeClustering（凝集型）** は、近いペアから順に併合していく階層木を作り、それを `distance_threshold` で切ります。密度の偏りに DBSCAN より強く、`distance_threshold=0.06`（cosine, average linkage）を当てれば **6クラスタ・NMI=1.000・purity=1.000** と一発で決まります。

```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering
pred_db = DBSCAN(eps=0.06, min_samples=3, metric="cosine").fit_predict(emb)        # -1 はノイズ
pred_ag = AgglomerativeClustering(n_clusters=None, distance_threshold=0.06,
                                  metric="cosine", linkage="average").fit_predict(emb)
```

使い分けの指針は、次のとおりです。まず、**k が事前に分かる、あるいは束が素直**なら k-means（速くて安定し、大規模には `MiniBatchKMeans`）。次に、**外れ値やノイズを弾きたい、あるいは任意形状を捉えたい**なら DBSCAN（ただし `eps` 調整がシビアで、`min_samples` も効く）。そして、**k を決めず、密度の偏りに強く、しきい値で切りたい**なら Agglomerative（埋め込みのクラスタリングで定番。`linkage` は `average`/`ward` を比較）です。顔クラスタリング（第30回）が DBSCAN と Agglomerative を使ったのは、まさに「人数が未知で、外れ値（知らない人）も混じる」からでした。本章の合成データでは Agglomerative が最も素直に決まるので、その結果をぜひ自分の目で確認してください。

## 6. 評価 — silhouette（教師なし）と NMI/purity/homogeneity（正解あり）

クラスタリングの評価には、大きく2系統あります。ひとつは**ラベル不要の内部指標**で、その代表が前述の **silhouette_score**（束の締まりと分離を測り、1 に近いほど良い）です。これは実運用でそのまま使える唯一の系統であり、k 選びにも使えます。ただし「球状の束」を前提とするため、DBSCAN のような非球状クラスタでは過小評価しがちな点に注意してください（なお、DBSCAN のノイズ -1 は評価から除くのが普通で、`cluster_lab.silhouette_safe` がその処理を担います）。

もうひとつが、**正解ラベルがあるときの外部指標**で、本章では4つを使います。まず **purity（純度）** は「各クラスタを、中の多数派の真ラベルに割り当てたときの全体正解率」で直感的ですが、**クラスタを増やすほど上がってしまう**（1点1クラスタで purity=1）ため、単独では使えません。次に **NMI（normalized mutual information）** は「予測クラスタと真ラベルの相互情報量」を 0〜1 に正規化したもので、クラスタ数を増やしても得をしにくく、本章の主指標とします。さらに **homogeneity（各クラスタが単一の真クラスで占められているか＝混ざっていないか）** と **completeness（各真クラスが単一クラスタにまとまっているか＝割れていないか）** の対を使うと、誤りの種類を切り分けられます。

```python
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
nmi  = normalized_mutual_info_score(labels_true, pred)   # 主指標（0〜1, 高いほど一致）
homo = homogeneity_score(labels_true, pred)              # クラスタが混ざっていないか
comp = completeness_score(labels_true, pred)             # 真クラスが割れていないか
```

数式を一つだけ押さえるなら、NMI です。真ラベル分布 `U` と予測分布 `V` の相互情報量 `I(U;V)` を、両者のエントロピーを使って `NMI = I(U;V) / mean(H(U), H(V))` のように正規化します（sklearn の既定は算術平均）。`cluster_lab.clustering_report` は、silhouette を常に、そしてラベルがあれば purity/NMI/homogeneity/completeness を、まとめて dict で返します。これらを使って、**purity だけ高くて NMI が低い**ときは「クラスタを刻みすぎ（homogeneity 高・completeness 低）」、逆に **completeness だけ高い**ときは「束ねすぎ」と読み解けるようにしてください。なお、これらの土台は演習の問6（purity）・問8（分割表）で numpy を使って自作します。

## 7. テキストのクラスタリング — 同じ枠組みが別モダリティでも動く

クラスタリングは、決して画像専用の技術ではありません。CLIP の `get_text_features` を使えば文も同じ 512 次元空間に乗るので、キャプションやラベルの集合を**まったく同じ手順**（埋め込み→L2正規化→k-means→評価）で束ねられます。実例として、`04_text_and_crossmodal.py` は「動物/乗り物/食べ物/天気」の4トピック・計20文を埋め込み、k-means でトピック分けします。なお `embed_texts` は内部で `padding=True` を指定しており（複数文の長さを揃えないとエラーになるため）、戻り値は画像と同じく正規化済みです。

ただし、結果は画像ほど綺麗には出ません。`04` の実測は `k=4` で **silhouette=0.124、NMI=0.623**（04 は silhouette と NMI のみを出力）と、画像の NMI=1.0 に比べてかなり控えめです。とはいえ、これは**ごく自然**な結果であり、むしろ教材として重要なポイントです。というのも、色×形の画像は概念の境界がくっきりしているのに対し、自然文は「a red sports car on the highway」のように複数の概念（乗り物・色・場所）が混ざり、トピックの境界がどうしても曖昧になるからです。シルエットも `k=3:0.113 → 5:0.134` と低く平坦なままで、「実データのクラスタリングは、トイデータほど綺麗には割れない」という現実を体感できます。

この「画像は綺麗、テキストは曖昧」という差そのものが、大きな学びになります。つまり、クラスタリングの良し悪しは、**アルゴリズムよりも埋め込みの質と概念の分離度**で決まるのです。だからテキストをもっと綺麗に割りたいなら、より強いテキストエンコーダ（`sentence-transformers` の文埋め込み専用モデル）を使う、トピックを粗くする、前処理で表現を揃える、といった**入力側の改善**こそが効きます。「クラスタが汚いときは、アルゴリズムを変える前に、まず埋め込みを疑う」——この順序をぜひ身につけてください。

## 8. クロスモーダルの落とし穴 — モダリティギャップ

画像とテキストが「同じ CLIP 空間」に乗っているのなら、両方を混ぜて一緒にクラスタリングし、「概念」ごとに画像と文を同じ束に入れられそうに思えます。ところが、ここに有名な落とし穴である **モダリティギャップ（modality gap）** が潜んでいます。CLIP の画像ベクトル群とテキストベクトル群は、同じ空間の中で**別々の“円錐（cone）”に偏って分布**しており、両者の平均ベクトルのコサインは `04` の実測で **0.267** と低い値にとどまります——つまり、両者は**離れた2つの島**を成しているのです。この状態で素朴に混ぜて `k=2` で割ると、クラスタは**概念ではなくモダリティ（画像か文か）で割れて**しまいます（`NMI vs モダリティ=1.000、vs 概念=0.000`）。

ここで面白いのは、**クロスモーダル検索のほうは無事**だという点です。`04` で「各画像に最も近いキャプション」を引くと **top1 正解率=1.00** となり、赤い丸の画像には、ちゃんと「a photo of a red circle」が一番近いキャプションとして返ります。なぜでしょうか。検索は**相対的な順位**だけを見るので、画像island全体が文islandから一様にずれていても、文の中での「どれが一番近いか」という順序は保たれるからです。これに対しクラスタリングは**絶対的な位置**で束ねるので、この一様なズレ（ギャップ）に正面から引っかかってしまいます。「検索は相対、クラスタリングは絶対」というこの対比こそが、本節の核心です。

その対策が、**モダリティごとに中心（平均ベクトル）を引いて島の位置を揃える（centering）**ことです。`04` では、画像群・文群それぞれから自モダリティの平均を引いて再正規化したうえで `k=6` でクラスタリングすると、今度は **NMI vs 概念=1.000、vs モダリティ=0.000** となり、見事に**概念で束ねられます**（赤い丸の画像と「red circle」の文が同じクラスタに入る）。`04_modality_gap.png` の2島構造とあわせて、「同じ空間に乗っている＝そのまま混ぜてよい、というわけではない」ことを体に刻んでください。これは ImageBind や SigLIP など、すべてのマルチモーダル埋め込みに共通する注意点です。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="モダリティギャップ。画像とテキストのCLIPベクトルは別々の島に偏り素朴な混合はモダリティで割れるがcenteringで概念ごとに重なる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="18" y="58" width="300" height="224" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="342" y="58" width="300" height="224" rx="8" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="168" y="80" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">そのまま混ぜる</text><text x="492" y="80" text-anchor="middle" font-size="14.5" font-weight="700" fill="#18181b">centering 後</text><ellipse cx="168" cy="130" rx="120" ry="40" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><ellipse cx="168" cy="230" rx="120" ry="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="91" y="120" width="8" height="8" fill="#ea580c"/><rect x="131" y="112" width="8" height="8" fill="#ea580c"/><rect x="166" y="130" width="8" height="8" fill="#ea580c"/><rect x="201" y="114" width="8" height="8" fill="#ea580c"/><rect x="231" y="124" width="8" height="8" fill="#ea580c"/><circle cx="100" cy="228" r="5" fill="#2563eb"/><circle cx="138" cy="236" r="5" fill="#2563eb"/><circle cx="172" cy="226" r="5" fill="#2563eb"/><circle cx="205" cy="234" r="5" fill="#2563eb"/><circle cx="236" cy="224" r="5" fill="#2563eb"/><line x1="28" y1="180" x2="308" y2="180" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="6 4"/><text x="168" y="106" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">画像ベクトルの島</text><text x="168" y="262" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">テキストベクトルの島</text><text x="300" y="174" text-anchor="end" font-size="11" fill="#dc2626">k=2 の境界</text><ellipse cx="492" cy="170" rx="128" ry="80" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="444" y1="132" x2="456" y2="138" stroke="#d4d4d8" stroke-width="1.2"/><line x1="532" y1="124" x2="544" y2="130" stroke="#d4d4d8" stroke-width="1.2"/><line x1="474" y1="214" x2="486" y2="218" stroke="#d4d4d8" stroke-width="1.2"/><rect x="440" y="128" width="8" height="8" fill="#ea580c"/><circle cx="456" cy="138" r="5" fill="#ea580c"/><rect x="528" y="120" width="8" height="8" fill="#2563eb"/><circle cx="544" cy="130" r="5" fill="#2563eb"/><rect x="470" y="210" width="8" height="8" fill="#16a34a"/><circle cx="486" cy="218" r="5" fill="#16a34a"/><text x="492" y="106" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">概念ごとに重なる</text><text x="492" y="266" text-anchor="middle" font-size="11.5" fill="#52525b">□＝画像 ○＝テキスト ・ 色＝概念</text></svg><figcaption><b>モダリティギャップ</b>です。CLIP の画像ベクトル（□）とテキストベクトル（○）は同じ空間でも<b>別々の島</b>に偏って分布し（平均同士の <code>cos ≈ 0.27</code>）、左のように素朴に混ぜて <code>k=2</code> で割ると<b>概念ではなくモダリティで割れて</b>しまいます。右の <b>centering</b>（モダリティごとに平均を引いて再正規化）で島を重ねると、<b>概念</b>ごとに画像と文が同じ束に入ります。</figcaption></figure>


## 9. 高次元を2Dで見る — PCA / t-SNE（UMAP・HDBSCAN は任意）

512 次元のクラスタが「本当に分離しているか」は、数字だけでなく**目で**も確かめたいものです。そこで `05_visualize_reduce.py` は、次元削減で 2D に落として散布図にします。まず **PCA** は分散最大の軸へ線形射影する古典的手法で、**速くて決定的、しかも全体構造（大局）を保つ**のが長所です。`explained_variance_ratio_` を見れば「2軸で全分散の何%を説明できたか」が読め、`05` では `0.261 + 0.227 = 0.489`——つまり**約半分しか見えていない**ことが分かります。残りの次元に隠れた構造があるかもしれない、と自覚しておくのが、PCA を正しく使う作法です。

一方 **t-SNE** は非線形で、**近傍関係（局所構造）を保つ**ように確率的に点を配置します。クラスタの“島”を浮かび上がらせるのは得意ですが、その代わり**島どうしの距離や島の大きさには意味が無く**、`perplexity` と乱数にも敏感です。したがって t-SNE は**可視化専用**と割り切るべきで、「t-SNE の2D座標の上でクラスタリングをやり直す」のは厳禁です（局所構造を歪めた座標の上で距離を測ることになるため）。`05` では `init="pca"`・`learning_rate="auto"`・点数に応じた `perplexity` を指定して安定させ、`05_pca_vs_tsne.png` に両者を並べます。色＝k-means クラスタ（マーカーは単一）となっているので、両表現でグループが分離して見えることを確認してください（真グループ別マーカーは `05_pca.png` / `05_tsne.png` を参照）。

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
xy_pca  = PCA(n_components=2).fit_transform(emb)
xy_tsne = TSNE(n_components=2, perplexity=10, init="pca", learning_rate="auto").fit_transform(emb)
```

より新しい **UMAP** は、t-SNE より大局構造も保ちやすく高速ですが、本リポジトリの既定環境には未導入です。そのため `05` は import をガードし、「`uv add umap-learn` で入れれば `05_umap.png` も出る」という導線だけを示します。同様に **HDBSCAN**（DBSCAN の階層・自動 eps 版）は密度ベースの強化版で、`05` では `sklearn.cluster.HDBSCAN`（sklearn 1.3+ に同梱）が使えれば実行し、本章データで `clusters=6・noise=0・NMI=1.000` と、DBSCAN の `eps` 問題を回避できることを示します。なお `05` が使う HDBSCAN は sklearn 同梱の `sklearn.cluster.HDBSCAN` で、外部 `hdbscan` パッケージは参照しません（`uv add` は不要）。**任意ライブラリは import をガードし、無くても本体は完走する**——これこそがマスター水準の堅牢な教材設計です。

## 10. 顔クラスタリング（第30回）との接続とスクリプト一覧

第30回の「写真アルバムの人物自動仕分け」は、実は本章の**特殊例**にあたります。顔検出で切り出した顔を顔認識モデルで埋め込み、L2 正規化して DBSCAN / Agglomerative でクラスタリングし、purity/NMI/homogeneity で評価する——その構造は本章とまったく同じで、違いは**埋め込みが「CLIP の概念ベクトル」から「顔の同一性ベクトル」に替わっただけ**です。つまり、本章で身につけた「埋め込み→正規化→クラスタリング→k選択→評価→可視化」という骨格は、顔・商品画像・文書・音声など、**あらゆる埋め込みにそのまま再利用できる一般形**なのです。この一般性こそが、本章の最大の収穫です。

各スクリプトは単一責務で、上から順に読むと「束ねる → k を選ぶ → k なしで束ねる → 別モダリティ → 可視化」と、理解が自然に積み上がるよう並べてあります。いずれも結果を `outputs/44_embedding_clustering/` に図と json で保存し、画面表示には依存しません（matplotlib は Agg、cv2.imshow は使いません）。また、device 判定・合成データ生成・CLIP 埋め込み・評価・可視化といった共通処理は `cluster_lab.py` にまとめてあり、各スクリプトはそれを `import cluster_lab as cl` で呼び出します。

| ファイル | 役割（単一責務） |
| --- | --- |
| `cluster_lab.py` | device 判定・合成データ生成（画像6グループ/テキスト4トピック）・CLIP 埋め込み（正規化込み）・評価（purity/NMI/silhouette ほか）・可視化（散布図/アルバム）。道具箱 |
| `01_kmeans_image_embeddings.py` | 画像コレクションを CLIP 埋め込み → k-means → silhouette/NMI/purity 評価・アルバム・PCA 散布図 |
| `02_choosing_k.py` | k=2..10 を掃引し inertia（エルボー）と silhouette で k を自動選択。NMI で答え合わせ |
| `03_dbscan_agglomerative.py` | k を決めない DBSCAN / Agglomerative。距離構造から DBSCAN の単一 eps 限界を実演し比較 |
| `04_text_and_crossmodal.py` | テキストのトピック分け。クロスモーダルのモダリティギャップと centering 対策、顔(30)への接続 |
| `05_visualize_reduce.py` | PCA（説明分散）/ t-SNE（局所構造）で 2D 可視化。UMAP・HDBSCAN は任意ガード |
| `mini_project.py` | 章末の統合課題。画像＋テキストを埋め込み→k自動選択→クラスタリング→評価→2D可視化を一気通貫 |
| `use_case.py` | 実践ユースケース。実フォルダの写真を埋め込み→クラスタリングし、内容ごとのサブフォルダ分けを提案（`--apply` で実コピー） |
| `exercises.py` | TODO 形式の演習8問（易→難、自己採点ランナー付き）。numpy だけでクラスタリングの核を実装 |
| `exercises_solutions.py` | 演習の模範解答（全問 PASS）。採点ロジックは `exercises.py` を再利用し、解答実装だけを保持 |

なお `cluster_lab.py` だけは、「読み物」ではなく「再利用する道具」という位置づけです。まず helper を一読し、`embed_images`（正規化済み 512 次元を返す）と `clustering_report`（評価を一括）が全スクリプトの土台になっていることを掴んでから 01 へ進むと、各スクリプトが何を import しているのかが腑に落ちます。

## 🛠 章末ミニプロジェクト — 埋め込みクラスタリング一気通貫ツール

ここまでの学び（埋め込み→正規化→k選択→クラスタリング→評価→可視化）を**1本に統合**したのが `mini_project.py` です。画像コレクション（6グループ）とテキスト集合（4トピック）に対し、実運用さながらの「ラベル無しデータを自動で束ねて中身を把握する」流れを通しで実行し、`outputs/44_embedding_clustering/mini_project_summary.png`（6パネル要約）と `mini_project_report.json`（全数値）を出力します。所要は CPU で数十秒、ネット接続が必要なのも初回の CLIP 重みダウンロードのときだけです。

- **Stage A: 画像 — k 自動選択 → k-means**（§3/§4）。`silhouette` を `k=2..10` で掃引し、最大の **k=6** を自動選択して k-means。**silhouette=0.759・NMI=1.000・purity=1.000・homogeneity=1.000**。
- **Stage B: 画像 — Agglomerative（k 不要）**（§5）。同じデータに `distance_threshold=0.06` を当て、**clusters=6・NMI=1.000・purity=1.000**。k を決めずとも k-means と同等以上に決まることを確認。
- **Stage C: テキスト — 同じ枠組みを別モダリティに**（§7）。4トピックを k-means し **silhouette=0.124・NMI=0.623・purity=0.70**。境界が曖昧な分だけ画像より低い、という現実を体感。
- **Stage D: 可視化**（§9）。画像の PCA / t-SNE と、テキストの PCA を6パネルに並べ、「色＝予測クラスタ・マーカー＝真ラベル」で分離を目視検証。

```bash
uv run python lectures/44_embedding_clustering/mini_project.py
# → outputs/44_embedding_clustering/mini_project_summary.png, mini_project_report.json
```

このミニプロジェクトを自分の手で読み解き、4つの数字（自動選択した k・画像 NMI・k 不要手法の NMI・テキスト NMI）が**それぞれ何を測っており、なぜ画像とテキストで差が出るのか**を説明できれば、本章のゴールに到達したと言えます。

## ✅ 到達チェックリスト

次の項目をすべて「コードで再現でき、理由を一言で説明できる」状態を目標にしてください。

- [ ] **検索とクラスタリングの違い**: 検索＝相対順位（教師あり評価）、クラスタリング＝全体を束に（教師なし）を説明できる（§1）。
- [ ] **L2 正規化**: クラスタリング前に必ず正規化し、正規化後はユークリッド距離＝コサイン距離（向きの近さ）になる理由を言える（§2）。
- [ ] **k-means**: `KMeans(n_clusters=k, n_init=10)` で束ね、E/M ステップを numpy でも書ける（§3・演習3/4）。
- [ ] **k 選択**: エルボー（inertia）とシルエットで k を選び、シルエット最大の k が真の k と一致することを示せる（§4）。
- [ ] **DBSCAN/Agglomerative**: k を決めずに束ね、DBSCAN の単一 eps 限界と Agglomerative の `distance_threshold` を使い分けられる（§5）。
- [ ] **評価指標**: silhouette（教師なし）と NMI/purity/homogeneity/completeness（正解あり）の意味と使い分けを言える（§6）。
- [ ] **purity の罠**: purity はクラスタを増やすほど上がるので NMI と併読する、を説明できる（§6）。
- [ ] **テキスト/クロスモーダル**: `get_text_features` でテキストも同枠組みで束ね、モダリティギャップで素朴な混合が破綻する理由と centering 対策を言える（§7/§8）。
- [ ] **可視化**: PCA（説明分散を確認）と t-SNE（局所構造・島の距離は無意味）でクラスタを検証でき、t-SNE 座標上で再クラスタリングしてはいけない理由を言える（§9）。
- [ ] **一般性**: 顔クラスタリング(30)が本章の特殊例＝埋め込みが替わるだけ、と説明できる（§10）。
- [ ] **演習**: `exercises.py` を8問すべて自力で PASS させた（`exercises_solutions.py` で答え合わせ）。

## ❓ よくある落とし穴・FAQ・デバッグ

**Q1. クラスタリング結果がデタラメ／1つの巨大クラスタになる。** まず、埋め込みを **L2 正規化**したか（`emb.norm(axis=1)≈1.0` か）を確認します。次に、DBSCAN/Agglomerative なら `eps`/`distance_threshold` が大きすぎて全点がつながっていないかを疑い、`03` の距離構造（グループ内 max / グループ間 min）を見て妥当な範囲かを確かめます。k-means なら、k が小さすぎないかを `02` のシルエットで点検します。

**Q2. silhouette が NaN／エラーになる。** silhouette は、クラスタが **2個以上**ないと定義できません。そのため、DBSCAN が全点を1クラスタにまとめたり、逆に全部ノイズにしたりした場合に起こります。`cluster_lab.silhouette_safe` はノイズ(-1)を除外し、クラスタ数が2未満のときは NaN を返してクラッシュを防ぎます。

**Q3. DBSCAN がうまく k 個に割れない。** そもそも DBSCAN は k を指定しない手法であり、`eps` 一つで全グループを切るため、密度の異なる束には弱いのです（§5）。クラスタ数を制御したいなら、k-means（k 指定）か Agglomerative（`distance_threshold`）、あるいは HDBSCAN を検討してください。なお、`min_samples` を上げるとノイズが増え、下げると細かい束ができます。

**Q4. purity は高いのに NMI が低い。** これは、クラスタを**刻みすぎ**ているサインです（極端には、1点1クラスタで purity=1 になります）。homogeneity（高い）と completeness（低い）の対を見れば、「混ざってはいないが、同じグループが複数クラスタに割れている」と分かります。対処としては、k を下げるか `distance_threshold` を上げて、より大きく束ねます（§6）。

**Q5. 画像とテキストを混ぜたら、概念でなくモダリティで割れた。** これがモダリティギャップです（§8）。CLIP の画像/文ベクトルは別々の島に偏っているため、素朴に混ぜてクラスタリングすると、概念ではなくモダリティを拾ってしまいます。対策は、モダリティごとに平均を引いて再正規化（centering）するか、あるいはクロスモーダルを「検索（相対比較）」として扱うことです。

**Q6. t-SNE の図で島が遠い＝そのグループは無関係、と読んでよい?** いいえ、読んではいけません。t-SNE は**島どうしの距離や島の大きさには意味を持たない**からです（保存されるのは局所構造のみ）。大局を見たいなら PCA や UMAP を併用し、t-SNE はあくまで「分離しているか」の確認だけに留めてください。当然、t-SNE 座標の上でクラスタリングをやり直すのも厳禁です（§9）。

**デバッグの切り分け順**: ①`emb.shape` と `emb.norm(axis=1)`（次元・正規化）→ ②距離構造（`03` のグループ内/間距離）→ ③`02` のシルエット曲線で k を点検 → ④散布図（PCA/t-SNE）で**目視** → ⑤評価は silhouette と NMI/homogeneity/completeness を**併読**、という順で見ます。この順序をたどれば、たいていの「クラスタが変」は、正規化漏れ・しきい値・k のいずれかに行き着きます。

## 🚀 発展トピック・参考

本章の骨格（埋め込み→正規化→クラスタリング→k選択→評価→可視化）は、そのまま実務へ、そして次の応用へと伸ばしていけます。

- **大規模化**: 数万件を超えたら `KMeans` は `MiniBatchKMeans` に、距離計算は第17回の **FAISS**（`faiss.Kmeans` や、近傍グラフ＋HDBSCAN）に載せ替えます。本章の `vector` グループ（faiss-cpu）はその伏線です。
- **HDBSCAN / UMAP**: 密度が不均一・クラスタ数が本当に未知な実データでは、`HDBSCAN`（自動 eps・階層）＋ `UMAP`（大局も保つ次元削減）の組み合わせが定番です。`uv add umap-learn` を入れれば `05` のガードが UMAP を自動で拾います（HDBSCAN は `05` では sklearn 同梱版を使うため、外部 `hdbscan` を `uv add` しても `05` では自動利用されません）。
- **他のクラスタリング**: `GaussianMixture`（軟クラスタ・楕円形に対応）、`SpectralClustering`（グラフベース・非凸形状）、`MeanShift`（モード探索・k 不要）など。データの形に応じて使い分けます。
- **モダリティギャップの研究**: "The Modality Gap"（Liang et al., 2022）が、対照学習が画像/文を別円錐に置く理由を分析しています。クロスモーダルなクラスタリング/検索の設計に直結します。
- **クラスタ ID の安定化**: k-means のラベル番号は実行ごとに入れ替わります。クラスタ間で比較・追跡したいときは、`scipy.optimize.linear_sum_assignment`（ハンガリアン法）で真ラベルや前回結果と対応付けます。
- **公式ドキュメント**: [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html) ／ [clustering の評価](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) ／ [manifold（t-SNE）](https://scikit-learn.org/stable/modules/manifold.html) ／ [UMAP](https://umap-learn.readthedocs.io/) ／ [HDBSCAN](https://hdbscan.readthedocs.io/)。

## 💡 実践ユースケース集

本章の骨格（埋め込み→正規化→クラスタリング→評価→可視化）は、そのまま現実の小ツールになります。検索（17章）が「クエリに近いものを並べる」のに対し、クラスタリングは「**ラベルが無い山を、似たもの同士に自動で束ねる**」ので、「とにかく溜まった大量の何かを、人手をかけずにざっくり整理・把握したい」という場面で効きます。以下では3つの現実応用を挙げ、そのうち1つ目を**動く出発点 `use_case.py`** として同梱しています。

### 1. 写真の自動フォルダ分け（auto-organizer）← `use_case.py` で実装

- **何に使うか**: 撮りっぱなしで散らかった画像フォルダを、中身（人物・料理・風景・乗り物…）ごとのサブフォルダに**自動仕分け**する。「だいたい何種類の写真が何枚ずつあるか」を一望し、整理の下書きを作る用途。
- **作り方の要点**: フォルダ内の画像を CLIP で埋め込み→L2 正規化→**silhouette で k を自動選択**して k-means（§3/§4）。各クラスタの**中心ベクトルに最も近い語彙ラベル**を CLIP のゼロショットで引いて、人間が読める**フォルダ名を自動命名**します（例: `people` / `food` / `nature`）。結果は「どのファイルをどのフォルダへ」という**計画 JSON** と**クラスタ別アルバム PNG** にし、`--apply` で初めて実コピーする（既定はドライランで安全）。
- **注意**: 命名はあくまで CLIP の当て推量なので、`use_case.py` の `PHOTO_VOCAB`（候補ラベル語彙）を**自分の用途に合わせて書き換える**と精度が上がります。k はデータ依存——`--k` で固定して比べるとよいです。元ファイルは**消さずコピー**する設計（破壊的操作をしない）を踏襲してください。

```bash
# 提案だけ（ドライラン・何も移動しない）。data/ が空なら合成データで動作デモ
uv run python lectures/44_embedding_clustering/use_case.py
# 自分の写真で実運用するには: data/44_embedding_clustering/ に画像を置くだけ（再帰探索）
#   （jpg/png/webp ... 何枚でも。サブフォルダに分かれていてもOK）
uv run python lectures/44_embedding_clustering/use_case.py --k 8     # サブフォルダ数を固定
uv run python lectures/44_embedding_clustering/use_case.py --apply   # 提案どおり実コピー
# → outputs/44_embedding_clustering/use_case_organizer.png / use_case_organizer_plan.json
#   （--apply 時のみ outputs/.../organized/<folder>/ に実ファイルをコピー）
```

`data/44_embedding_clustering/` に画像が**無ければ合成データ（色×形6グループ）で必ず完走**し、画像を置けばそのまま実運用に切り替わります。**練習（拡張）アイデア**: ①`PHOTO_VOCAB` を旅行/料理/家族など自分のカテゴリへ書き換える、②k-means を `AgglomerativeClustering(distance_threshold=...)` に差し替えて「k を決めずに」整理する、③同一クラスタ内でコサイン類似が極端に高いペアを**重複（ニアデュープ）候補**として別出力する、④DBSCAN にして「どの束にも入らない外れ写真」を `noise/` に隔離する。

### 2. 未整理データセットの“素性”プロファイリング（ラベル付け前の俯瞰）

- **何に使うか**: アノテーション前の生画像/テキストの山に対し、「どんなカテゴリが、どれくらいの偏りで含まれるか」を**ラベルを作る前に**把握する。重複・外れ値・想定外カテゴリの早期発見にも使え、ラベリング計画やサンプリング設計の土台になります。
- **作り方の要点**: 全件を埋め込み→Agglomerative か HDBSCAN（§5/§9）で束ね、**クラスタごとの件数分布**と**代表サンプル（中心に最も近い数枚）**を出力。silhouette と、もし一部に既知ラベルがあれば NMI/purity（§6）で「束ね方が妥当か」を点検します。`use_case.py` の計画 JSON をそのまま「データ統計レポート」に転用できます。
- **注意**: クラスタ数や粒度は埋め込みの質に強く依存します（§7 の「まずアルゴリズムより埋め込みを疑う」）。**t-SNE の島の距離・大きさに意味を読み込まない**（§9）。テキストが混じるなら**モダリティギャップ**で破綻するので、画像と文は別々に束ねるか centering する（§8）。

### 3. クラスタ代表による「多様性サンプリング／間引き」

- **何に使うか**: 大量画像から**重複を減らし、満遍なく多様な少数を選び出す**（学習データの間引き、レビュー用サムネ選定、データセット軽量化）。動画の連続フレームや、同じ被写体の連写を**1枚に代表させる**用途に有効です。
- **作り方の要点**: 埋め込み→k-means（k＝欲しい枚数の目安）→**各クラスタ中心に最も近い1枚**を代表として採用すれば、空間を覆う代表集合が得られます。逆に「各クラスタから1枚だけ残す」だけでニアデュープ除去になります。中心に近い順＝**典型度ランキング**としても使えます。
- **注意**: k を大きくしすぎると「ほぼ全部採用」になり間引きになりません（silhouette/件数で適正 k を点検）。**正規化必須**（§2）——忘れるとノルムの大きい1枚が中心を乗っ取ります。決定論性が要るなら `random_state` を固定（k-means のラベル番号は実行ごとに入れ替わる点も §発展 参照）。

> 共通の作法: いずれも **CPU・headless（matplotlib=Agg、`cv2.imshow` 不使用）** で動き、結果は `outputs/44_embedding_clustering/` に PNG/JSON で保存します。実データは `data/44_embedding_clustering/` に置けば自動で使われ、無ければ合成データでデモが完走します。**破壊的なファイル操作はオプトイン（`--apply`）**にし、既定はドライラン（提案のみ）にするのが安全な小ツールの設計です。

## ▶ 動かし方

このモジュールは、`dl`（torch/torchvision）・`hf`（transformers ほか）・`metrics`（scikit-learn）グループに依存します（`vector` は発展用なので必須ではありません）。CPU だけで完走し、初回のみ CLIP（`openai/clip-vit-base-patch32`）の重みを HuggingFace からダウンロードします（以降はキャッシュから即起動します）。準備ができたら、プロジェクトルートで以下を順に実行してください。

```bash
# 依存グループをインストール（初回のみ）
uv sync --group dl --group hf --group metrics

# 道具箱のスモークテスト（合成データ生成→埋め込み→k-means が一通り動く）
uv run python lectures/44_embedding_clustering/cluster_lab.py

# 各スクリプトを順に実行（結果は outputs/44_embedding_clustering/ に保存される）
uv run python lectures/44_embedding_clustering/01_kmeans_image_embeddings.py
uv run python lectures/44_embedding_clustering/02_choosing_k.py
uv run python lectures/44_embedding_clustering/03_dbscan_agglomerative.py
uv run python lectures/44_embedding_clustering/04_text_and_crossmodal.py
uv run python lectures/44_embedding_clustering/05_visualize_reduce.py

# 章末ミニプロジェクト（一気通貫。6パネル図 + JSON を出力）
uv run python lectures/44_embedding_clustering/mini_project.py

# 実践ユースケース: 写真の自動フォルダ分け（data/ に画像を置けば実運用。既定はドライラン）
uv run python lectures/44_embedding_clustering/use_case.py

# 演習: まずは TODO を自分で埋める（最初は全部 TODO だが exit 0）
uv run python lectures/44_embedding_clustering/exercises.py
uv run python lectures/44_embedding_clustering/exercises_solutions.py   # 全問 PASS の確認
```

実行後は、`outputs/44_embedding_clustering/` の図を本文の解説と照らし合わせてください。とくに `02_choosing_k.png`（シルエットが k=6 でピーク）、`03_album_agglomerative.png`（クラスタ別アルバム）、`04_modality_gap.png`（画像と文が2島に分かれる）、`mini_project_summary.png`（6パネル統合）を見れば、本章のテーマ（k選択・k不要手法・モダリティギャップ・可視化）が視覚的に腑に落ちるはずです。なお、図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。また、色が反転して見える場合は、合成画像を RGB のまま扱っているか（cv2 経由で BGR が混ざっていないか）を確認してください。

---

> 本教材で参照・検証したライブラリとバージョン（torch 2.12+cpu / transformers 5.11 / scikit-learn 1.9 / faiss-cpu、2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ scikit-learn 1.9.0 ／ faiss-cpu 1.14.2（発展・任意）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成画像の描画）
> 使用モデル: `openai/clip-vit-base-patch32`（CLIP）。初回のみ HuggingFace から重みを取得しキャッシュします。任意ライブラリ（umap-learn / hdbscan）は未導入でもガードにより本体は完走します。
