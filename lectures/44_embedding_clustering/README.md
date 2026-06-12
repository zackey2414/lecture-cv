# 第44回 埋め込みのクラスタリング — 画像・テキスト・クロスモーダルを教師なしで束ねる

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 依存グループ: `dl`（torch/torchvision）・`hf`（transformers ほか）・`metrics`（scikit-learn）・`vector`（faiss-cpu, 任意）。CPU だけで完走します（初回のみ CLIP の重みを HuggingFace からダウンロード）。
> 前提: 第16回（CLIP ゼロショット & 検索）・第17回（FAISS 検索）。関連: 第15回（埋め込み）・第30回（顔クラスタリング）。

## 🎯 この章のゴール

第16・17回では「クエリに近いものを並べる」**検索（retrieval）** を学びました。本章のテーマはその裏返し、**クラスタリング（grouping）** です。クエリも正解ラベルも無い大量の埋め込みを、似たもの同士で自動的に束に分けます。スマホの写真アプリが「人物」ごとに写真をまとめたり、未整理の画像コレクションを「だいたいこういう種類がN個ある」と俯瞰したりするのが、まさにこれです。この章を終えると、CLIP/ResNet で得た埋め込み（画像・テキスト・顔）を、AI 補助なしで自分の手でクラスタリングし、品質を数字で評価し、2D に可視化して検証できるようになります。

具体的な到達点は5つです。第一に、埋め込みを **L2 正規化**してから **k-means** で束ね、結果を **silhouette / NMI / purity / homogeneity** で評価できること。第二に、k-means が必要とする**クラスタ数 k** を、**エルボー法**と**シルエット法**でラベル無しに選べること。第三に、k を決めずに済む **DBSCAN** と **AgglomerativeClustering** を使い分け、それぞれの限界（DBSCAN の単一 eps 問題など）を体感すること。第四に、**テキスト**のクラスタリングと**クロスモーダル**（画像とテキストを同じ空間で）を扱い、CLIP の **モダリティギャップ**という落とし穴を知ること。第五に、**PCA / t-SNE**（UMAP は任意）でクラスタを可視化し、結果を目で検証できることです。

本章のスクリプトはすべて、ネット接続もデータセット DL も無しで完走するよう、入力を**その場で合成**します。画像は「赤い丸」「青い三角」など**色×形で 6 グループ**を作り、テキストは「動物/乗り物/食べ物/天気」の **4 トピック**を用意します。重要なのは、これらの**正解ラベルはクラスタリングには一切渡さず、評価（答え合わせ）にだけ使う**点です。これが本章の精神です——本来クラスタリングはラベルが無いから行うものであり、ラベルは「うまく束ねられたか」を後から確認するためだけのものです。ダウンロードが走るのは初回の CLIP 重み取得だけで、以降はローカルキャッシュから即起動します。

---

## 1. 検索とクラスタリングはどう違うのか（直感）

検索とクラスタリングは、どちらも「埋め込み空間での近さ」を使う双子のようなタスクですが、問いの形がまったく違います。**検索**は「**この1点（クエリ）**に近いものを、近い順に並べて」という問いで、答えはランキングです。基準点があり、相対的な順位だけが問題になります。一方**クラスタリング**は「**全体**を、似たもの同士のいくつかの束に分けて」という問いで、答えは各点へのグループ ID の割り当てです。基準点は無く、データ全体の構造を一度に捉えます。だから検索は**教師あり的に評価**でき（クエリの正解が分かる）、クラスタリングは本質的に**教師なし**です。

この違いは実装にも効きます。検索は「正規化した埋め込みの内積を取って topk」で済み、第17回では FAISS でこれを大規模化しました。クラスタリングは「全点の距離関係から束を見つける」ので、k-means なら反復、DBSCAN なら密度の連結、Agglomerative なら階層的な併合、と専用のアルゴリズムが要ります。評価も別物で、検索は Recall@k / mAP（正解ランキングとの一致）、クラスタリングは silhouette（束の締まり具合）や NMI（既知グループとの一致）を使います。本章はこの**クラスタリング側の道具一式**を揃えます。

なお、両者は地続きでもあります。クラスタの「代表ベクトル（中心）」を作れば、新しい点がどの束に近いかを**検索**で判定できますし、検索結果を連結していけば粗いクラスタになります。第40・41回の Cluster-CLIP は、まさに「密な CLIP 特徴をクラスタリングして領域を作り、それを検索インデックスにする」という、両者を組み合わせたパイプラインでした。本章の到達点は、その土台となる「埋め込みを束ねる」基本操作を、原理から自分の手で書けるようにすることです。

## 2. すべての出発点 — 埋め込みと L2 正規化

クラスタリングの入力は「1サンプル＝1本のベクトル」の集まりです。本章では画像も文も CLIP で **512 次元のベクトル**に変換します（第15・16回で学んだ `get_image_features` / `get_text_features` の `.pooler_output`）。`cluster_lab.embed_images` / `embed_texts` がこれを担い、(N, 512) の float32 行列を返します。重要なのは、これらの生の埋め込みは**ノルムがバラバラ（CLIP 画像ベクトルは約 11）**だという点です。ベクトルの「長さ」は明るさやコントラスト、語数などで揺れやすく、それに距離計算が引きずられると束ねが乱れます。

そこで本章でも**最初に必ず L2 正規化**します。各ベクトルを長さ 1 にそろえると、ユークリッド距離・内積・コサイン類似度が一対一に対応し、距離は「**向きの違い**」だけを測るようになります。具体的には、正規化後のベクトル `u, v`（`|u|=|v|=1`）について `コサイン類似度 = u·v`、`コサイン距離 = 1 - u·v`、そして `ユークリッド距離² = 2(1 - u·v)` です。つまり正規化さえすれば、k-means の既定（ユークリッド）も DBSCAN/Agglomerative の `metric="cosine"` も、本質的に同じ「向きの近さ」で動きます。`cluster_lab.embed_*` は正規化まで込みで返すので、各スクリプトは安心して距離ベースの手法を当てられます。

この「正規化が先」は第16回の検索でも口酸っぱく述べた鉄則で、忘れると「ノルムが大きいだけの1点」がクラスタを乗っ取ります。実際、`01` で `各行ノルム≈1.000` と表示されることを毎回確認する習慣をつけてください。埋め込みの作り方そのもの（CLIP の `.pooler_output` が未正規化である非対称など）は第16回で深掘りしたので、本章では「正規化済みの良いベクトルが手元にある」前提から始め、**束ね方と評価**に集中します。

## 3. k-means — 最も基本的な束ね方

k-means は、データを **k 個の球状の束**に分ける最も基本的なアルゴリズムです。やることは2ステップの反復だけ。**(E) 割り当て**: 各点を最も近い「中心（centroid）」のクラスタに入れる。**(M) 更新**: 各クラスタの所属点の平均を新しい中心にする。これを中心が動かなくなるまで繰り返します。初期中心はランダムなので結果が初期値依存になりがちで、`n_init=10` のように**複数回試して inertia（後述）最小の解を採用**するのが定石です（sklearn の既定は賢い `k-means++` 初期化）。演習の問3・問4で、この E/M ステップを numpy で自分で書きます。

正準 API は `sklearn.cluster.KMeans` です。`01_kmeans_image_embeddings.py` の核は次の3行に集約されます。L2 正規化済み埋め込みに対し、クラスタ数 `k` を指定して `fit_predict` するだけで、各点のクラスタ ID 配列が返ります。

```python
from sklearn.cluster import KMeans
emb = cl.embed_images(images, device)            # (N, 512) L2 正規化済み
pred = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(emb)  # 各点のクラスタID
```

合成6グループ（n_per_group=6, 計36枚）でこれを回すと、**silhouette=0.759、NMI=1.000、purity=1.000、homogeneity=1.000** と、色×形のグループを完全に当てます。`01_album_kmeans.png`（クラスタ別に画像を並べたアルバム）の各行が綺麗に1グループで揃い、`01_pca_scatter.png` で同じ色（予測クラスタ）の点がまとまることを目で確認してください。ただしここでは「真の数は6」とこっそり k に教えています。実務では k は未知です——次節でその選び方を学びます。なお k-means の弱点も押さえておきましょう: ①k を事前に決める必要がある、②球状で同程度の大きさの束を仮定する（細長い/密度の違う束は苦手）、③外れ値に弱い（中心が引っ張られる）。

## 4. クラスタ数 k の選び方 — エルボー法とシルエット法

k-means 最大の悩みは「k をいくつにするか」です。ラベルが無いのにどう決めるのか。古典は**エルボー法**です。k を増やすと **inertia**（各点と所属中心の二乗距離の総和。クラスタの締まり具合）は必ず単調に下がります。k=N（1点1クラスタ）で 0 になるからです。そこで「下がり方が急に緩む“肘（elbow）”」を探します。`02` の実測では inertia が `k=2:3.09 → 4:1.57 → 6:0.69 → 8:0.52` と、k=5〜6 あたりで折れ曲がります。直感的ですが、肘がはっきり出ないデータも多く、判断が主観的なのが弱点です。

より自動化しやすいのが**シルエット法**です。各点について「自分のクラスタ内の平均距離 a」と「最も近い別クラスタへの平均距離 b」を測り、シルエット係数 `s = (b - a) / max(a, b)` を計算します。`s` は −1〜1 で、**1 に近いほど「自分の束に近く、隣の束から遠い」＝良い分離**です。全点の平均が `silhouette_score` で、**これが最大になる k を選びます**。ラベル不要なので実運用でそのまま使えます。正準 API は `sklearn.metrics.silhouette_score(emb, labels, metric="cosine")` です。

```python
from sklearn.metrics import silhouette_score
best = max(range(2, 11),
           key=lambda k: silhouette_score(emb, KMeans(k, n_init=10).fit_predict(emb), metric="cosine"))
```

`02_choosing_k.py` のシルエットは `k=2:0.37 → 5:0.69 → 6:0.759 → 7:0.73 → 10:0.57` と **k=6 で綺麗にピーク**を打ち、真のグループ数と一致します。図 `02_choosing_k.png` の右パネルでは、ラベルを使った NMI（`k=6` で 1.0）のピークとシルエットのピークが重なり、「ラベルが無くてもシルエットだけで正しい k を選べた」ことが見て取れます。実務ではこの NMI 曲線は**見えません**（ラベルが無い）——シルエットや、文脈上の制約（「だいたい数十カテゴリのはず」）で k を絞るのが現実です。

## 5. k を決めない仲間 — DBSCAN と Agglomerative

「k を先に決める」のが嫌なら、クラスタ数を**データに決めさせる**手法があります。**DBSCAN** は密度ベースで、「`eps` 半径内に `min_samples` 個以上の点があれば芯（core）」とみなして束を広げ、どの束にも届かない点を**ノイズ（ラベル -1）**にします。クラスタ数の指定が要らず、任意形状の束を捉え、外れ値に強いのが長所です。`sklearn.cluster.DBSCAN(eps=..., min_samples=..., metric="cosine")` が正準形です。ただし**弱点が `eps`**で、これは**全体で1つの大域的なしきい値**です。グループによって密度（広がり）が違うと、一つの `eps` では全グループを同時に綺麗に切れません。

本章の合成データでこの限界が綺麗に出ます。`03` でコサイン距離の構造を覗くと、**グループ内距離の最大が 0.079、グループ間距離の最小が 0.048** と**重なって**います。だから単一 `eps` では、`0.03→6クラスタ(ただし1点ノイズ)、0.05→5、0.07→4、0.08→2` と、しきい値をわずかに動かすだけでクラスタ数が乱高下し、真の6に安定して届きません。一方 **AgglomerativeClustering（凝集型）** は、近いペアから順に併合する階層木を作り、`distance_threshold` で切ります。密度の偏りに DBSCAN より強く、`distance_threshold=0.06`（cosine, average linkage）で **6クラスタ・NMI=1.000・purity=1.000** と一発で決まります。

```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering
pred_db = DBSCAN(eps=0.06, min_samples=3, metric="cosine").fit_predict(emb)        # -1 はノイズ
pred_ag = AgglomerativeClustering(n_clusters=None, distance_threshold=0.06,
                                  metric="cosine", linkage="average").fit_predict(emb)
```

使い分けの指針はこうです。**k が事前に分かる/束が素直**なら k-means（速くて安定、大規模にも `MiniBatchKMeans`）。**外れ値やノイズを弾きたい/任意形状**なら DBSCAN（ただし `eps` 調整がシビア、`min_samples` も効く）。**k を決めず、密度の偏りに強く、しきい値で切りたい**なら Agglomerative（埋め込みのクラスタリングで定番。`linkage` は `average`/`ward` を比較）。顔クラスタリング（第30回）が DBSCAN と Agglomerative を使ったのは、まさに「人数が未知で外れ値（知らない人）がいる」からでした。本章の合成データでは Agglomerative が最も素直に決まる、という結果を自分の目で確認してください。

## 6. 評価 — silhouette（教師なし）と NMI/purity/homogeneity（正解あり）

クラスタリングの評価は2系統あります。ひとつは**ラベル不要の内部指標**で、代表が前述の **silhouette_score**（束の締まり・分離。1 に近いほど良い）です。これは実運用でそのまま使える唯一の系統で、k 選びにも使えます。ただし「球状の束」を前提にするので、DBSCAN のような非球状クラスタでは過小評価しがちな点に注意します（DBSCAN のノイズ -1 は評価から除くのが普通で、`cluster_lab.silhouette_safe` はそれを行います）。

もうひとつが、**正解ラベルがあるときの外部指標**です。本章では4つ使います。**purity（純度）**は「各クラスタを中の多数派の真ラベルに割り当てたときの全体正解率」で直感的ですが、**クラスタを増やすほど上がる**（1点1クラスタで purity=1）ので単独では使えません。**NMI（normalized mutual information）**は「予測クラスタと真ラベルの相互情報量」を 0〜1 に正規化したもので、クラスタ数を増やしても得しにくく、本章の主指標です。さらに **homogeneity（各クラスタが単一の真クラスで占められているか＝混ざっていないか）** と **completeness（各真クラスが単一クラスタにまとまっているか＝割れていないか）** の対が、誤りの種類を切り分けます。

```python
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
nmi  = normalized_mutual_info_score(labels_true, pred)   # 主指標（0〜1, 高いほど一致）
homo = homogeneity_score(labels_true, pred)              # クラスタが混ざっていないか
comp = completeness_score(labels_true, pred)             # 真クラスが割れていないか
```

数式を一つだけ押さえるなら NMI です。真ラベル分布 `U` と予測分布 `V` の相互情報量 `I(U;V)` を、両者のエントロピーで `NMI = I(U;V) / mean(H(U), H(V))` のように正規化します（sklearn の既定は算術平均）。`cluster_lab.clustering_report` は、silhouette を常に、ラベルがあれば purity/NMI/homogeneity/completeness を、まとめて dict で返します。**purity だけ高くて NMI が低い**ときは「クラスタを刻みすぎ（homogeneity 高・completeness 低）」、逆に **completeness だけ高い**ときは「束ねすぎ」と読み解けるようにしてください。演習の問6（purity）・問8（分割表）で、これらの土台を numpy で自作します。

## 7. テキストのクラスタリング — 同じ枠組みが別モダリティでも動く

クラスタリングは画像専用ではありません。CLIP の `get_text_features` で文も同じ 512 次元空間に乗るので、キャプションやラベルの集合を**まったく同じ手順**（埋め込み→L2正規化→k-means→評価）で束ねられます。`04_text_and_crossmodal.py` は「動物/乗り物/食べ物/天気」の4トピック・計20文を埋め込み、k-means でトピック分けします。`embed_texts` は内部で `padding=True` を指定しており（複数文の長さを揃えないとエラー）、戻り値は画像と同じく正規化済みです。

ただし結果は画像ほど綺麗には出ません。`04` の実測は `k=4` で **silhouette=0.124、NMI=0.623、purity=0.70** と、画像の NMI=1.0 に比べて控えめです。これは**ごく自然**で、教材として重要なポイントです。色×形の画像は概念の境界がくっきりしていますが、自然文は「a red sports car on the highway」のように複数の概念（乗り物・色・場所）が混ざり、トピックの境界が曖昧だからです。シルエットも `k=3:0.113 → 5:0.134` と低く平坦で、「実データのクラスタリングはトイデータほど綺麗に割れない」という現実を体感できます。

この「画像は綺麗・テキストは曖昧」という差そのものが学びです。クラスタリングの良し悪しは**アルゴリズムよりも埋め込みの質と概念の分離度**で決まります。テキストをもっと綺麗に割りたいなら、より強いテキストエンコーダ（`sentence-transformers` の文埋め込み専用モデル）を使う、トピックを粗くする、前処理で表現を揃える、といった**入力側の改善**が効きます。「クラスタが汚いときに、まずアルゴリズムを変える前に埋め込みを疑う」——この順序を身につけてください。

## 8. クロスモーダルの落とし穴 — モダリティギャップ

画像とテキストが「同じ CLIP 空間」に乗っているなら、両方を混ぜて一緒にクラスタリングし、「概念」ごとに画像と文を同じ束に入れられそうです。ところがここに有名な落とし穴 **モダリティギャップ（modality gap）** があります。CLIP の画像ベクトル群とテキストベクトル群は、同じ空間の中で**別々の“円錐（cone）”に偏って分布**しており、両者の平均ベクトルのコサインは `04` の実測で **0.267** と低い——つまり**離れた2つの島**になっています。この状態で素朴に混ぜて `k=2` で割ると、クラスタは**概念ではなくモダリティ（画像か文か）で割れて**しまいます（`NMI vs モダリティ=1.000、vs 概念=0.000`）。

面白いのは、**クロスモーダル検索は無事**なことです。`04` で「各画像に最も近いキャプション」を引くと **top1 正解率=1.00**——赤い丸の画像は、ちゃんと「a photo of a red circle」が一番近いキャプションになります。なぜか。検索は**相対的な順位**だけを見るので、画像island全体が文islandから一様にずれていても、文の中での「どれが一番近いか」の順序は保たれるからです。一方クラスタリングは**絶対的な位置**で束ねるので、一様なズレ（ギャップ）に正面から引っかかります。「検索は相対・クラスタリングは絶対」というこの対比が、本節の核心です。

対策は、**モダリティごとに中心（平均ベクトル）を引いて島の位置を揃える（centering）**ことです。`04` では画像群・文群それぞれから自モダリティの平均を引き、再正規化してから `k=6` でクラスタリングすると、今度は **NMI vs 概念=1.000、vs モダリティ=0.000** と、見事に**概念で束ねられます**（赤い丸の画像と「red circle」の文が同じクラスタに入る）。`04_modality_gap.png` の2島構造と合わせて、「同じ空間に乗っている＝そのまま混ぜて良い、ではない」ことを体に刻んでください。これは ImageBind や SigLIP など全てのマルチモーダル埋め込みに共通する注意点です。

## 9. 高次元を2Dで見る — PCA / t-SNE（UMAP・HDBSCAN は任意）

512 次元のクラスタが「本当に分離しているか」は、数字だけでなく**目で**確かめたい。`05_visualize_reduce.py` は次元削減で 2D に落として散布図にします。**PCA** は分散最大の軸へ線形射影する古典で、**速くて決定的・全体構造（大局）を保つ**のが長所です。`explained_variance_ratio_` で「2軸で全分散の何%を説明できたか」が読め、`05` では `0.261 + 0.227 = 0.489`——**約半分しか見えていない**ことが分かります。残りの次元に隠れた構造があるかもしれない、と自覚するのが PCA を正しく使う作法です。

**t-SNE** は非線形で、**近傍関係（局所構造）を保つ**よう確率的に点を配置します。クラスタの“島”を見せるのが得意な一方、**島どうしの距離や島の大きさには意味が無く**、`perplexity` と乱数に敏感です。だから t-SNE は**可視化専用**であり、「t-SNE の2D座標の上でクラスタリングをやり直す」のは厳禁です（局所構造を歪めた座標で距離を測ることになる）。`05` では `init="pca"`・`learning_rate="auto"`・点数に応じた `perplexity` で安定させ、`05_pca_vs_tsne.png` に両者を並べます。色＝k-means クラスタ、マーカー＝真グループで、両表現でグループが分離して見えることを確認してください。

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
xy_pca  = PCA(n_components=2).fit_transform(emb)
xy_tsne = TSNE(n_components=2, perplexity=10, init="pca", learning_rate="auto").fit_transform(emb)
```

より新しい **UMAP** は、t-SNE より大局構造も保ちやすく高速ですが、本リポジトリ既定環境には未導入なので `05` は import をガードし、「`uv add umap-learn` で入れれば `05_umap.png` も出る」導線だけ示します。同様に **HDBSCAN**（DBSCAN の階層・自動 eps 版）は密度ベースの強化版で、`05` では `sklearn.cluster.HDBSCAN`（sklearn 1.3+ に同梱）が使えれば実行し、本章データで `clusters=6・noise=0・NMI=1.000` と DBSCAN の `eps` 問題を回避できることを示します。外部 `hdbscan` パッケージを使う場合は同じくガードで「あれば使う」設計にします。**任意ライブラリは import をガードし、無くても本体は完走する**——これがマスター水準の堅牢な教材設計です。

## 10. 顔クラスタリング（第30回）との接続とスクリプト一覧

第30回の「写真アルバムの人物自動仕分け」は、本章の**特殊例**です。顔検出で切り出した顔を顔認識モデルで埋め込み、L2 正規化して DBSCAN / Agglomerative でクラスタリングし、purity/NMI/homogeneity で評価する——構造は本章とまったく同じで、**埋め込みが「CLIP の概念ベクトル」から「顔の同一性ベクトル」に替わっただけ**です。つまり本章で身につけた「埋め込み→正規化→クラスタリング→k選択→評価→可視化」の骨格は、顔・商品画像・文書・音声など、**あらゆる埋め込みにそのまま再利用できる一般形**です。この一般性こそが本章の最大の収穫です。

各スクリプトは単一責務で、上から読むと「束ねる → k を選ぶ → k なしで束ねる → 別モダリティ → 可視化」と理解が積み上がります。すべて `outputs/44_embedding_clustering/` に図と json を保存し、画面表示には依存しません（matplotlib は Agg、cv2.imshow は使いません）。device 判定・合成データ生成・CLIP 埋め込み・評価・可視化といった共通処理は `cluster_lab.py` にまとめ、各スクリプトはそれを `import cluster_lab as cl` で使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `cluster_lab.py` | device 判定・合成データ生成（画像6グループ/テキスト4トピック）・CLIP 埋め込み（正規化込み）・評価（purity/NMI/silhouette ほか）・可視化（散布図/アルバム）。道具箱 |
| `01_kmeans_image_embeddings.py` | 画像コレクションを CLIP 埋め込み → k-means → silhouette/NMI/purity 評価・アルバム・PCA 散布図 |
| `02_choosing_k.py` | k=2..10 を掃引し inertia（エルボー）と silhouette で k を自動選択。NMI で答え合わせ |
| `03_dbscan_agglomerative.py` | k を決めない DBSCAN / Agglomerative。距離構造から DBSCAN の単一 eps 限界を実演し比較 |
| `04_text_and_crossmodal.py` | テキストのトピック分け。クロスモーダルのモダリティギャップと centering 対策、顔(30)への接続 |
| `05_visualize_reduce.py` | PCA（説明分散）/ t-SNE（局所構造）で 2D 可視化。UMAP・HDBSCAN は任意ガード |
| `mini_project.py` | 章末の統合課題。画像＋テキストを埋め込み→k自動選択→クラスタリング→評価→2D可視化を一気通貫 |
| `exercises.py` | TODO 形式の演習8問（易→難、自己採点ランナー付き）。numpy だけでクラスタリングの核を実装 |
| `exercises_solutions.py` | 演習の模範解答（全問 PASS）。採点ロジックは `exercises.py` を再利用し、解答実装だけを保持 |

`cluster_lab.py` だけは「読み物」ではなく「再利用する道具」です。まず helper を一読し、`embed_images`（正規化済み 512 次元を返す）と `clustering_report`（評価を一括）が全スクリプトの土台であることを掴んでから 01 へ進むと、各スクリプトが何を import しているかが腑に落ちます。

## 🛠 章末ミニプロジェクト — 埋め込みクラスタリング一気通貫ツール

ここまでの学び（埋め込み→正規化→k選択→クラスタリング→評価→可視化）を**1本に統合**するのが `mini_project.py` です。画像コレクション（6グループ）とテキスト集合（4トピック）に対し、実運用の「ラベル無しデータを自動で束ねて中身を把握する」流れを通しで実行し、`outputs/44_embedding_clustering/mini_project_summary.png`（6パネル要約）と `mini_project_report.json`（全数値）を出力します。CPU で数十秒、ネットは初回の CLIP 重み DL のみです。

- **Stage A: 画像 — k 自動選択 → k-means**（§3/§4）。`silhouette` を `k=2..10` で掃引し、最大の **k=6** を自動選択して k-means。**silhouette=0.759・NMI=1.000・purity=1.000・homogeneity=1.000**。
- **Stage B: 画像 — Agglomerative（k 不要）**（§5）。同じデータに `distance_threshold=0.06` を当て、**clusters=6・NMI=1.000・purity=1.000**。k を決めずとも k-means と同等以上に決まることを確認。
- **Stage C: テキスト — 同じ枠組みを別モダリティに**（§7）。4トピックを k-means し **silhouette=0.124・NMI=0.623・purity=0.70**。境界が曖昧な分だけ画像より低い、という現実を体感。
- **Stage D: 可視化**（§9）。画像の PCA / t-SNE と、テキストの PCA を6パネルに並べ、「色＝予測クラスタ・マーカー＝真ラベル」で分離を目視検証。

```bash
uv run python lectures/44_embedding_clustering/mini_project.py
# → outputs/44_embedding_clustering/mini_project_summary.png, mini_project_report.json
```

このミニプロジェクトを自分の手で読み解き、4つの数字（自動選択した k・画像 NMI・k 不要手法の NMI・テキスト NMI）が**何を測り、なぜ画像とテキストで差が出るか**を説明できれば、本章のゴールに到達しています。

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

**Q1. クラスタリング結果がデタラメ／1つの巨大クラスタになる。** まず埋め込みを **L2 正規化**したか（`emb.norm(axis=1)≈1.0` か）を確認します。次に DBSCAN/Agglomerative なら `eps`/`distance_threshold` が大きすぎて全部つながっていないか、`03` の距離構造（グループ内 max / グループ間 min）を見て妥当な範囲か確かめます。k-means なら k が小さすぎないかを `02` のシルエットで点検します。

**Q2. silhouette が NaN／エラーになる。** silhouette はクラスタが **2個以上**ないと定義できません。DBSCAN が全点を1クラスタ or 全部ノイズにした場合に起きます。`cluster_lab.silhouette_safe` はノイズ(-1)を除外し、クラスタ数<2 のとき NaN を返してクラッシュを防ぎます。

**Q3. DBSCAN がうまく k 個に割れない。** DBSCAN は k を指定しない手法で、`eps` 一つで全グループを切るため、密度の異なる束には弱いです（§5）。クラスタ数を制御したいなら k-means（k 指定）か Agglomerative（`distance_threshold`）、あるいは HDBSCAN を検討します。`min_samples` を上げるとノイズが増え、下げると細かい束ができます。

**Q4. purity は高いのに NMI が低い。** クラスタを**刻みすぎ**ています（極端には1点1クラスタで purity=1）。homogeneity（高い）と completeness（低い）の対を見ると「混ざってはいないが、同じグループが複数クラスタに割れている」と分かります。k を下げるか `distance_threshold` を上げて束ねます（§6）。

**Q5. 画像とテキストを混ぜたら、概念でなくモダリティで割れた。** モダリティギャップです（§8）。CLIP の画像/文ベクトルは別々の島に偏っているため、素朴な混合クラスタリングはモダリティを拾います。モダリティごとに平均を引いて再正規化（centering）するか、クロスモーダルは「検索（相対比較）」として扱います。

**Q6. t-SNE の図で島が遠い＝そのグループは無関係、と読んでよい?** いいえ。t-SNE は**島どうしの距離・島の大きさに意味がありません**（局所構造のみ保存）。大局を見たいなら PCA や UMAP を併用し、t-SNE はあくまで「分離しているか」の確認に留めます。t-SNE 座標の上でクラスタリングをやり直すのも厳禁です（§9）。

**デバッグの切り分け順**: ①`emb.shape` と `emb.norm(axis=1)`（次元・正規化）→ ②距離構造（`03` のグループ内/間距離）→ ③`02` のシルエット曲線で k を点検 → ④散布図（PCA/t-SNE）で**目視** → ⑤評価は silhouette と NMI/homogeneity/completeness を**併読**。この順で見ると、たいていの「クラスタが変」は正規化漏れ・しきい値・k のどれかに行き着きます。

## 🚀 発展トピック・参考

本章の骨格（埋め込み→正規化→クラスタリング→k選択→評価→可視化）は、そのまま実務と次の応用に伸びます。

- **大規模化**: 数万件を超えたら `KMeans` は `MiniBatchKMeans` に、距離計算は第17回の **FAISS**（`faiss.Kmeans` や、近傍グラフ＋HDBSCAN）に載せ替えます。本章の `vector` グループ（faiss-cpu）はその伏線です。
- **HDBSCAN / UMAP**: 密度が不均一・クラスタ数が本当に未知な実データでは、`HDBSCAN`（自動 eps・階層）＋ `UMAP`（大局も保つ次元削減）の組み合わせが定番です。`uv add hdbscan umap-learn` で導入し、`05` のガードが自動で拾います。
- **他のクラスタリング**: `GaussianMixture`（軟クラスタ・楕円形に対応）、`SpectralClustering`（グラフベース・非凸形状）、`MeanShift`（モード探索・k 不要）など。データの形に応じて使い分けます。
- **モダリティギャップの研究**: "The Modality Gap"（Liang et al., 2022）が、対照学習が画像/文を別円錐に置く理由を分析しています。クロスモーダルなクラスタリング/検索の設計に直結します。
- **クラスタ ID の安定化**: k-means のラベル番号は実行ごとに入れ替わります。クラスタ間で比較・追跡したいときは、`scipy.optimize.linear_sum_assignment`（ハンガリアン法）で真ラベルや前回結果と対応付けます。
- **公式ドキュメント**: [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html) ／ [clustering の評価](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) ／ [manifold（t-SNE）](https://scikit-learn.org/stable/modules/manifold.html) ／ [UMAP](https://umap-learn.readthedocs.io/) ／ [HDBSCAN](https://hdbscan.readthedocs.io/)。

## ▶ 動かし方

このモジュールは `dl`（torch/torchvision）・`hf`（transformers ほか）・`metrics`（scikit-learn）グループに依存します（`vector` は発展用で必須ではありません）。CPU だけで完走し、初回のみ CLIP（`openai/clip-vit-base-patch32`）の重みを HuggingFace からダウンロードします（以降はキャッシュから即起動）。プロジェクトルートで以下を順に実行してください。

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

# 演習: まずは TODO を自分で埋める（最初は全部 TODO だが exit 0）
uv run python lectures/44_embedding_clustering/exercises.py
uv run python lectures/44_embedding_clustering/exercises_solutions.py   # 全問 PASS の確認
```

実行後は `outputs/44_embedding_clustering/` の図を解説と照らし合わせてください。とくに `02_choosing_k.png`（シルエットが k=6 でピーク）、`03_album_agglomerative.png`（クラスタ別アルバム）、`04_modality_gap.png`（画像と文が2島に分かれる）、`mini_project_summary.png`（6パネル統合）を見ると、本章のテーマ（k選択・k不要手法・モダリティギャップ・可視化）が視覚的に腑に落ちます。図中の文字は CJK フォントの豆腐（□）を避けるため ASCII にしてあります。色が反転して見える場合は、合成画像を RGB のまま扱っているか（cv2 経由で BGR が混ざっていないか）を確認してください。

---

> 本教材で参照・検証したライブラリとバージョン（torch 2.12+cpu / transformers 5.11 / scikit-learn 1.9 / faiss-cpu、2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ torch 2.12.0+cpu ／ torchvision 0.27.0+cpu ／ transformers 5.11.0 ／ huggingface-hub 1.18.0 ／ scikit-learn 1.9.0 ／ faiss-cpu 1.14.2（発展・任意）／ numpy 2.4.6 ／ Pillow 12.2.0 ／ matplotlib 3.10.9 ／ opencv-python-headless 4.13.0（合成画像の描画）
> 使用モデル: `openai/clip-vit-base-patch32`（CLIP）。初回のみ HuggingFace から重みを取得しキャッシュします。任意ライブラリ（umap-learn / hdbscan）は未導入でもガードにより本体は完走します。
