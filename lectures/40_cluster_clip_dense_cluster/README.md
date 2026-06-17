# 40. Cluster-CLIP — dense CLIP 特徴と空間連結クラスタリング（講座の総仕上げ）

> 版: torch 2.12+cpu / open-clip-torch 3.3 / faiss-cpu 1.14 / scikit-learn 1.9（2026-06）
> 元ネタ: **Cluster-CLIP** という手法（dense CLIP + 空間連結クラスタリングが本章の核）

---

## 🎯 ゴール

この章は本講座のクライマックスです。これまで積み上げてきた「画像 I/O → テンソル → ViT/CLIP 埋め込み → クラスタリング → FAISS 近傍探索 → SQLite → ストリーム処理」を、**1 枚の画像を意味のある領域に分けて検索できるようにする** という 1 つの目的のもとに統合します。題材に選ぶのは、テキストで動画フレームを開語彙検索する実在の研究システム **Cluster-CLIP** です。その中核にある「dense CLIP 特徴を空間連結クラスタリングで領域に束ね、領域代表ベクトルで検索する」という発想を、CPU だけで動く小型版として手を動かしながら再構築していきます。

具体的な到達点は次の通りです。

- **なぜ「画像全体 1 ベクトル」では足りないのか** を、小物体の希釈という観点から説明できる。
- CLIP の visual encoder を **手で forward 展開** して、CLS ではなく **パッチトークン**（dense 特徴 `[C, H, W]`）を取り出せる。
- **空間連結（grid_to_graph）付き AgglomerativeClustering** で、特徴が似ていて かつ 空間的に隣接したパッチだけを束ね、飛び地のない領域に分割できる。
- 各領域の **代表ベクトル（平均 → L2 正規化）** を作り、**FAISS（IndexFlatIP + IDMap）** に登録し、**SQLite** で `faiss_id ↔ (フレーム, 領域, bbox)` を引けるようにできる。
- テキスト / 画像クエリで領域検索し、ヒット領域をマスク重畳で可視化できる。
- これらを **Split → Build → Search → Stream** の 1 本のパイプライン（capstone）に束ね、`multiprocessing` で取得と推論を分離し、キュー満杯でフレームをドロップする実時間設計まで通せる。

このディレクトリのスクリプトは、リポジトリのルートから次のように動かします（出力は `lectures/40_cluster_clip_dense_cluster/outputs/`）。

```bash
uv run python lectures/40_cluster_clip_dense_cluster/01_dense_vs_global_clip.py
```

---


## 1. なぜ「全体 1 ベクトル」ではなく「領域（dense）」なのか

CLIP の標準的な使い方（16 章・17 章）は、画像 1 枚を 1 本の埋め込みベクトルに潰し、テキストと同じ空間でコサイン類似度を測ることでした。これは「この画像は犬っぽいか」「全体としてビーチの写真か」を測るには強力です。しかし 1 本に潰すという操作は、裏を返せば **画面のどこに何があるかという空間情報を捨てる** ことを意味します。そのため、複数の物体が写ったフレームから「黄色いボールが写っている瞬間と、その位置」を引きたいとき、全体ベクトルは「ボールらしさ」を他の大きな物体や背景と平均してしまい、小物体の信号は薄まって消えてしまいます。

`01_dense_vs_global_clip.py` は、この希釈を数値で見せてくれます。224×224 の合成シーンを ViT-B/32 に通すと、パッチは 7×7 = 49 個できます。このうち小さな黄色いボールが占めるのは約 4 パッチ、つまり全体の **約 8%** にすぎません。したがって全体を 1 本に pool すると、ボールの寄与は 49 分の数にまで薄まってしまいます。一方 dense 特徴なら、ボールは自分のパッチ群（やがて自分の「領域」）に固有のベクトルを持てます。これこそが、「領域単位で検索したい」「小物体・複数物体を扱いたい」という要求に dense が答えられる理由です。

<figure class="lec-fig"><svg viewBox="0 0 660 250" role="img" aria-label="画像全体を1ベクトルにpoolすると小物体は約8%に希釈されて消えるが、dense特徴なら領域ごとに固有ベクトルを持てる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">全体 1 ベクトル vs dense（領域）特徴</text><text x="129" y="50" text-anchor="middle" font-size="12.5" fill="#52525b">224×224 → 7×7 = 49 パッチ</text><rect x="56" y="56" width="147" height="147" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="77" y1="56" x2="77" y2="203" stroke="#e4e4e7"/><line x1="98" y1="56" x2="98" y2="203" stroke="#e4e4e7"/><line x1="119" y1="56" x2="119" y2="203" stroke="#e4e4e7"/><line x1="140" y1="56" x2="140" y2="203" stroke="#e4e4e7"/><line x1="161" y1="56" x2="161" y2="203" stroke="#e4e4e7"/><line x1="182" y1="56" x2="182" y2="203" stroke="#e4e4e7"/><line x1="56" y1="77" x2="203" y2="77" stroke="#e4e4e7"/><line x1="56" y1="98" x2="203" y2="98" stroke="#e4e4e7"/><line x1="56" y1="119" x2="203" y2="119" stroke="#e4e4e7"/><line x1="56" y1="140" x2="203" y2="140" stroke="#e4e4e7"/><line x1="56" y1="161" x2="203" y2="161" stroke="#e4e4e7"/><line x1="56" y1="182" x2="203" y2="182" stroke="#e4e4e7"/><rect x="119" y="119" width="42" height="42" fill="#f97316" stroke="#c2410c" stroke-width="2"/><text x="129" y="222" text-anchor="middle" font-size="12.5" fill="#52525b">黄色いボール ≈ 4 / 49 パッチ</text><line x1="205" y1="118" x2="256" y2="96" stroke="#71717a" stroke-width="1.8"/><polygon points="262,94 250,92 254,103" fill="#71717a"/><line x1="205" y1="142" x2="256" y2="172" stroke="#71717a" stroke-width="1.8"/><polygon points="262,174 254,165 250,176" fill="#71717a"/><text x="268" y="88" font-size="13" font-weight="700" fill="#3f3f46">全体 pool（平均）</text><rect x="268" y="96" width="150" height="26" fill="#f4f4f5" stroke="#d4d4d8" stroke-width="1.5"/><rect x="406" y="96" width="12" height="26" fill="#f97316"/><text x="343" y="142" text-anchor="middle" font-size="12" fill="#dc2626">1 本に潰れ、ボールは 8% に希釈</text><text x="268" y="160" font-size="13" font-weight="700" fill="#3f3f46">dense（49 パッチを保持）</text><rect x="268" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="286" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="304" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="322" y="170" width="14" height="26" fill="#f97316"/><rect x="340" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="358" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="376" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="394" y="170" width="14" height="26" fill="#d4d4d8"/><rect x="412" y="170" width="14" height="26" fill="#d4d4d8"/><text x="347" y="214" text-anchor="middle" font-size="12" fill="#15803d">ボールは固有ベクトルを保持</text></svg><figcaption>CLIP の標準手法は画像 1 枚を <b>1 本のベクトル</b>に潰します。すると画面の約 <b>8%（4/49 パッチ）</b>しかない小さなボールの信号は、大きな物体や背景と平均されて<b>希釈</b>され消えます。<b>dense 特徴</b>は各パッチ（やがて各領域）が固有のベクトルを保つので、小物体・複数物体を<b>領域単位</b>で検索できます。</figcaption></figure>

ただし正直に言うと、**生の ViT パッチトークンは、pool 後のベクトルほどテキストと綺麗には揃っていません**。CLIP の対照学習が整列させているのは pool 後のトークンだけだからです。したがって、この章で学ぶのはあくまで「領域に分けて検索する仕組み（パイプライン）」であり、合成のフラットな画像ではテキスト検索の順位はあてになりません。とはいえ実写画像であれば、この dense + クラスタリングの仕組みはきちんと領域を当ててくれます。まずは仕組みを正しく組めることを最優先に進めましょう。

## 2. dense CLIP 特徴の理論 — ViT を手で展開する

ViT の前半は「画像を 32×32 のパッチに切って、各パッチを 1 本のトークンに埋め込む」処理です（この役割は、ストライド 32 の畳み込みである `conv1` が担います）。その後、先頭に学習可能な **CLS トークン** を 1 本足し、位置埋め込みを加え、Transformer の自己注意でトークン同士を混ぜ合わせます。ところが標準の `encode_image` は、最後に CLS（または pool）を 1 本だけ取り出して返すため、パッチ単位の情報は外からは見えません。

そこで dense 特徴を得るには、この forward を **自分で順に通して最終トークン列を全部受け取り、CLS を捨ててパッチだけを空間 `(gh, gw)` に並べ直す** 必要があります。手順は `conv1`（パッチ埋め込み）→ CLS 連結 → 位置埋め込み加算 → `ln_pre` → `transformer` → `ln_post` → `proj`（テキストと同じ共通潜在空間へ射影）→ **CLS 除去** → パッチごとに L2 正規化、という流れです。`02_dense_features_extraction.py` は、この各ステップの shape を実況してくれます（`conv1 -> (1,768,7,7)` → `+CLS -> (1,50,768)` → `proj -> (1,50,512)` → `drop CLS -> (1,49,512)`）。

<figure class="lec-fig"><svg viewBox="0 0 680 244" role="img" aria-label="CLIPのViTを手でforward展開し、conv1で49パッチに切りCLSを連結して50、projを経てCLSを捨て49パッチのdense特徴にする流れ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ViT を手で forward 展開し dense 特徴を取り出す（shape の変化）</text><text x="147" y="84" text-anchor="middle" font-size="11" fill="#c2410c">conv1 ↓32</text><text x="281" y="84" text-anchor="middle" font-size="11" fill="#c2410c">＋CLS ＋pos</text><text x="415" y="84" text-anchor="middle" font-size="11" fill="#c2410c">proj</text><text x="549" y="84" text-anchor="middle" font-size="11" fill="#c2410c">drop CLS</text><rect x="22" y="96" width="116" height="58" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="80" y="131" text-anchor="middle" font-size="13" fill="#1d4ed8" font-family="'JetBrains Mono', monospace">(1,3,224,224)</text><rect x="156" y="96" width="116" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="214" y="131" text-anchor="middle" font-size="13" fill="#c2410c" font-family="'JetBrains Mono', monospace">(1,768,7,7)</text><rect x="290" y="96" width="116" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="348" y="131" text-anchor="middle" font-size="13" fill="#c2410c" font-family="'JetBrains Mono', monospace">(1,50,768)</text><rect x="424" y="96" width="116" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="482" y="131" text-anchor="middle" font-size="13" fill="#c2410c" font-family="'JetBrains Mono', monospace">(1,50,512)</text><rect x="558" y="96" width="116" height="58" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2.2"/><text x="616" y="131" text-anchor="middle" font-size="13" fill="#15803d" font-family="'JetBrains Mono', monospace">(1,49,512)</text><line x1="139" y1="125" x2="149" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="156,125 148,121 148,129" fill="#71717a"/><line x1="273" y1="125" x2="283" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="290,125 282,121 282,129" fill="#71717a"/><line x1="407" y1="125" x2="417" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="424,125 416,121 416,129" fill="#71717a"/><line x1="541" y1="125" x2="551" y2="125" stroke="#71717a" stroke-width="1.8"/><polygon points="558,125 550,121 550,129" fill="#71717a"/><text x="80" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">入力画像</text><text x="348" y="176" text-anchor="middle" font-size="12" fill="#2563eb">50 ＝ CLS 1 ＋ パッチ 49</text><text x="616" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">dense 特徴</text><text x="340" y="206" text-anchor="middle" font-size="12" fill="#52525b">CLS を捨てた 49 パッチを [512, 7, 7] に並べ替え、各パッチを L2 正規化</text></svg><figcaption>標準の <code>encode_image</code> は最後に CLS（pool）を 1 本返すだけで、パッチ情報が見えません。そこで forward を手で通します: <code>conv1</code>（ストライド 32）で <b>49 パッチ</b>に切り、<b>CLS を 1 本連結</b>して系列長 <b>50</b>、位置埋め込み・Transformer・<code>proj</code>（テキストと同じ潜在空間へ）を経て、最後に <b>CLS を捨て 49 パッチ</b>だけ残し <b>[512, 7, 7]</b> の dense マップへ並べ替えます。各 shape を必ず print して確かめます。</figcaption></figure>

前処理にも、もう一つ重要な勘所があります。標準の CLIP 前処理は短辺リサイズ + **CenterCrop** で正方形にしますが、これは画面端を切り落とすため、端にある小物体が dense 特徴から消えてしまいます。そこで Cluster-CLIP は CenterCrop を捨て、**アスペクト比を無視して正方形へ強制 Resize** します（`load_encoder` の `Resize((224,224))`）。これは、多少歪んでも端を残すほうが領域分割では得だ、という判断によるものです。

## 3. 空間連結クラスタリングの理論 — connectivity の意味

次に、この 49 個のパッチベクトルを、似たもの同士で k 個の領域に束ねます。ところが、素朴に k-means や普通の凝集型クラスタリングをかけると、**特徴空間での近さだけ** で併合してしまうので、画面の右上のパッチと左下のパッチが同じクラスタに入る「飛び地」が起こります。領域として扱いたいのに空間的にバラバラでは、検索結果のマスクが破綻してしまいます。

ここで効いてくるのが **connectivity（連結）制約** です。`sklearn.feature_extraction.image.grid_to_graph(h, w)` は、h×w グリッドの **4 近傍隣接グラフ** を作ります。これを `AgglomerativeClustering(connectivity=...)` に渡すと、凝集の各ステップで **グラフ上で隣接するクラスタ同士しか併合できなく** なります。その結果、どのクラスタも必ず空間的に 1 つながり（連結成分が 1 つ）になります。`03_spatial_connectivity.py` は、これを連結成分数で定量化してくれます。connectivity ありなら「総連結成分数 == クラスタ数」（全領域が連結）となり、なしだと成分数がクラスタ数を上回って断片化します（実行例では ON=6 / OFF=8）。

<figure class="lec-fig"><svg viewBox="0 0 680 260" role="img" aria-label="connectivityありなら各クラスタが空間的に連結し、なしだと同じ色が離れた場所に飛び地として分裂する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">空間連結（connectivity）の有無でクラスタが連結 / 断片化する</text><text x="133" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">connectivity ON</text><rect x="70" y="78" width="54" height="126" fill="#f97316"/><rect x="124" y="78" width="36" height="126" fill="#2563eb"/><rect x="160" y="78" width="36" height="126" fill="#16a34a"/><line x1="88" y1="78" x2="88" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="106" y1="78" x2="106" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="124" y1="78" x2="124" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="142" y1="78" x2="142" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="160" y1="78" x2="160" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="178" y1="78" x2="178" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="96" x2="196" y2="96" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="114" x2="196" y2="114" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="132" x2="196" y2="132" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="150" x2="196" y2="150" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="168" x2="196" y2="168" stroke="#ffffff" stroke-opacity="0.5"/><line x1="70" y1="186" x2="196" y2="186" stroke="#ffffff" stroke-opacity="0.5"/><rect x="70" y="78" width="126" height="126" fill="none" stroke="#3f3f46" stroke-width="1.8"/><text x="133" y="226" text-anchor="middle" font-size="12" fill="#15803d">連結成分 = クラスタ数</text><line x1="308" y1="118" x2="308" y2="94" stroke="#71717a" stroke-width="1.8"/><line x1="308" y1="118" x2="308" y2="142" stroke="#71717a" stroke-width="1.8"/><line x1="308" y1="118" x2="284" y2="118" stroke="#71717a" stroke-width="1.8"/><line x1="308" y1="118" x2="332" y2="118" stroke="#71717a" stroke-width="1.8"/><circle cx="308" cy="94" r="5" fill="#fafafa" stroke="#71717a" stroke-width="1.5"/><circle cx="308" cy="142" r="5" fill="#fafafa" stroke="#71717a" stroke-width="1.5"/><circle cx="284" cy="118" r="5" fill="#fafafa" stroke="#71717a" stroke-width="1.5"/><circle cx="332" cy="118" r="5" fill="#fafafa" stroke="#71717a" stroke-width="1.5"/><circle cx="308" cy="118" r="6" fill="#f97316"/><text x="308" y="172" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">grid_to_graph</text><text x="308" y="188" text-anchor="middle" font-size="11.5" fill="#52525b">＝ 4 近傍のみ連結</text><text x="483" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">connectivity OFF</text><rect x="420" y="78" width="54" height="54" fill="#f97316"/><rect x="474" y="78" width="72" height="54" fill="#2563eb"/><rect x="420" y="132" width="72" height="72" fill="#16a34a"/><rect x="492" y="132" width="54" height="72" fill="#f97316"/><line x1="438" y1="78" x2="438" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="456" y1="78" x2="456" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="474" y1="78" x2="474" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="492" y1="78" x2="492" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="510" y1="78" x2="510" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="528" y1="78" x2="528" y2="204" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="96" x2="546" y2="96" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="114" x2="546" y2="114" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="132" x2="546" y2="132" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="150" x2="546" y2="150" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="168" x2="546" y2="168" stroke="#ffffff" stroke-opacity="0.5"/><line x1="420" y1="186" x2="546" y2="186" stroke="#ffffff" stroke-opacity="0.5"/><rect x="420" y="78" width="126" height="126" fill="none" stroke="#3f3f46" stroke-width="1.8"/><line x1="447" y1="105" x2="519" y2="168" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.7"/><ellipse cx="447" cy="105" rx="33" ry="32" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3"/><ellipse cx="519" cy="168" rx="33" ry="40" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3"/><text x="483" y="226" text-anchor="middle" font-size="12" fill="#dc2626">連結成分 ＞ クラスタ数（断片化）</text></svg><figcaption>49 個のパッチを似たもの同士で束ねるとき、素朴なクラスタリングは<b>特徴の近さだけ</b>で併合するため、画面の離れた場所が同じクラスタに入る<b>飛び地</b>が生じます（右）。<code>grid_to_graph</code> が作る <b>4 近傍グラフ</b>を <code>connectivity</code> に渡すと、<b>隣接パッチ同士しか併合できず</b>、各クラスタが必ず空間的に 1 つながり（<b>連結成分 = クラスタ数</b>）になります（左）。</figcaption></figure>

linkage は `ward`（クラスタ内分散の増加が最小になるよう併合）を既定にします。この ward は、metric が `euclidean` であることを要求します。今回はパッチを L2 正規化してあるため、ユークリッド距離はコサイン距離と単調な関係になり、「向きが似たパッチ」を素直に束ねられます。

## 4. 正準 API — 何を呼べばよいか

dense 特徴の取り出しでは `open_clip` の visual encoder を直接触りますが、クラスタリングと FAISS のほうは sklearn / faiss の素直な API だけで済みます。本章の中核 API を一望すると、次の通りです（これらは `cc_common.py` が薄くラップしています）。

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

これらのうち `force_quick_gelu=True` は、地味ですが重要です。openai 配布の ViT は QuickGELU を使って学習されており、これを合わせないと活性化がわずかにずれて特徴の質が落ちてしまうからです。また `metric="euclidean"` と `linkage="ward"` はセットで、L2 正規化済みベクトルに対して安定して働きます。なお FAISS は **float32・C連続** 以外を受け付けないので、`np.ascontiguousarray(x, dtype=np.float32)`（= `cc_common.as_faiss`）を必ず通すようにします。

## 5. 実装を 1 つずつ

<figure class="lec-fig"><svg viewBox="0 0 680 240" role="img" aria-label="1フレームを索引化するまでの処理パイプライン。フレームをdense CLIP特徴にし空間連結クラスタで領域に束ね領域平均で代表ベクトルを作りFAISSとSQLiteに登録する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">1 フレームを索引に積むまでの処理パイプライン</text><text x="137" y="80" text-anchor="middle" font-size="10" fill="#c2410c">forward</text><text x="272" y="80" text-anchor="middle" font-size="10" fill="#c2410c">cluster</text><text x="407" y="80" text-anchor="middle" font-size="10" fill="#c2410c">平均 + L2</text><text x="542" y="80" text-anchor="middle" font-size="10" fill="#c2410c">add_with_ids</text><rect x="16" y="92" width="108" height="70" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><rect x="151" y="92" width="108" height="70" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><rect x="286" y="92" width="108" height="70" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><rect x="421" y="92" width="108" height="70" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><rect x="556" y="92" width="108" height="70" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="70" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">フレーム</text><text x="205" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">dense CLIP</text><text x="340" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">連結クラスタ</text><text x="475" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">代表ベクトル</text><text x="610" y="122" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">索引化</text><text x="70" y="146" text-anchor="middle" font-size="10.5" fill="#52525b" font-family="'JetBrains Mono', monospace">(H,W,3)</text><text x="205" y="146" text-anchor="middle" font-size="10.5" fill="#52525b" font-family="'JetBrains Mono', monospace">[C,gh,gw]</text><text x="340" y="146" text-anchor="middle" font-size="10.5" fill="#52525b" font-family="'JetBrains Mono', monospace">labels</text><text x="475" y="146" text-anchor="middle" font-size="10.5" fill="#52525b" font-family="'JetBrains Mono', monospace">reps[k,C]</text><text x="610" y="146" text-anchor="middle" font-size="10.5" fill="#52525b" font-family="'JetBrains Mono', monospace">.faiss + .db</text><line x1="126" y1="127" x2="145" y2="127" stroke="#71717a" stroke-width="1.8"/><polygon points="151,127 143,123 143,131" fill="#71717a"/><line x1="261" y1="127" x2="280" y2="127" stroke="#71717a" stroke-width="1.8"/><polygon points="286,127 278,123 278,131" fill="#71717a"/><line x1="396" y1="127" x2="415" y2="127" stroke="#71717a" stroke-width="1.8"/><polygon points="421,127 413,123 413,131" fill="#71717a"/><line x1="531" y1="127" x2="550" y2="127" stroke="#71717a" stroke-width="1.8"/><polygon points="556,127 548,123 548,131" fill="#71717a"/><text x="340" y="204" text-anchor="middle" font-size="11.5" fill="#52525b">dense → 連結クラスタ → 領域平均で代表ベクトル → FAISS／SQLite に登録</text></svg><figcaption><b>Build</b> は各フレームをこの順で索引に積みます。まず ViT を手で forward して <b>dense CLIP 特徴 <code>[C, gh, gw]</code></b> を取り出し、<code>grid_to_graph</code> 付き<b>空間連結クラスタ</b>で領域ラベルに束ねます。続いて領域ごとに<b>平均 → L2 正規化</b>して <b>代表ベクトル <code>reps[k, C]</code></b> を作り、<code>add_with_ids</code> で <b>FAISS（索引）</b>へ、対応するメタ（frame・bbox）を <b>SQLite</b> へ登録します。<code>.faiss</code>（索引）と <code>.db</code>（メタ）は必ずセットで持ちます。</figcaption></figure>

**dense 特徴（`cc_common.dense_vit_tokens` / 02）**: ViT を手で通し、`x = x[:, 1:, :]` で CLS を落とし、`x / x.norm(...)` でパッチごとに L2 正規化したうえで、`[B, C, gh, gw]` に並べ替えます。`02` を実行すると、全 49 パッチのノルムが 1.000、かつ水平方向の隣接パッチの平均コサインが 0.97 程度（空間的に滑らか）であることが確認できます。

**領域クラスタリングと代表ベクトル（`cc_common.cluster_regions` / 04）**: `[C,H,W]` を `[H*W, C]` に直し、connectivity 付き AgglomerativeClustering で `labels` を得たうえで、クラスタごとに平均 → L2 正規化して **代表ベクトル `reps[k, C]`** を作ります。`04` の出力では、小物体の黄色いボールが自分専用のクラスタを得て、その bbox を **coverage 1.00** で覆えていることが分かります（dense にした甲斐があった、という数値的な裏付けです）。

**FAISS + SQLite で検索基盤を作る（mini_project の Build / 17 章の復習）**: 代表ベクトルを `IndexFlatIP + IDMap` に `add_with_ids` し、`faiss_id` を SQLite の `VectorMapping(faiss_id, frame_id, cluster_idx, bbox)` と `Frames(frame_id, image_path, cmap_path)` に対応づけます。参考実装では `faiss_id` を SQLite の `AUTOINCREMENT` 行 ID として採番しており、本章の mini_project も `cur.lastrowid` を `faiss_id` に使ってこれを再現しています。FAISS はベクトルしか持たず、メタは持てません。だからこそ、**インデックス本体（.faiss）とメタ DB（.db）は必ずセットで永続化・整合させる** のが鉄則になります。

<figure class="lec-fig"><svg viewBox="0 0 680 300" role="img" aria-label="代表ベクトルはFAISSに、frameやbboxなどのメタはSQLiteに保存し、faiss_idで両者をjoinする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">代表ベクトルは FAISS、メタは SQLite、faiss_id で結ぶ</text><text x="166" y="72" text-anchor="middle" font-size="11.5" fill="#52525b">領域 → 平均 → L2 正規化 → 代表ベクトル</text><line x1="166" y1="76" x2="166" y2="86" stroke="#71717a" stroke-width="1.5"/><polygon points="166,90 162,82 170,82" fill="#71717a"/><rect x="50" y="88" width="232" height="148" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="166" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">FAISS: IndexFlatIP ＋ IDMap</text><text x="166" y="136" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c" font-family="'JetBrains Mono', monospace">faiss_id = 102</text><rect x="66" y="172" width="16" height="18" fill="#f97316"/><rect x="90" y="162" width="16" height="28" fill="#2563eb"/><rect x="114" y="176" width="16" height="14" fill="#71717a"/><rect x="138" y="166" width="16" height="24" fill="#f97316"/><rect x="162" y="174" width="16" height="16" fill="#2563eb"/><rect x="186" y="164" width="16" height="26" fill="#71717a"/><rect x="210" y="170" width="16" height="20" fill="#f97316"/><text x="166" y="214" text-anchor="middle" font-size="11.5" fill="#52525b">512 次元・L2 ノルム = 1</text><rect x="398" y="88" width="232" height="148" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="514" y="112" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">SQLite: VectorMapping</text><text x="420" y="140" font-size="12.5" fill="#c2410c" font-family="'JetBrains Mono', monospace">faiss_id = 102</text><text x="420" y="164" font-size="12.5" fill="#18181b" font-family="'JetBrains Mono', monospace">frame_id = f0</text><text x="420" y="188" font-size="12.5" fill="#18181b" font-family="'JetBrains Mono', monospace">cluster_idx = 3</text><text x="420" y="212" font-size="12.5" fill="#18181b" font-family="'JetBrains Mono', monospace">bbox = (12,40,30,30)</text><text x="340" y="120" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46">faiss_id</text><text x="340" y="135" text-anchor="middle" font-size="11" fill="#52525b">で join</text><line x1="282" y1="150" x2="398" y2="150" stroke="#c2410c" stroke-width="2" stroke-dasharray="6 4"/><polygon points="398,150 390,146 390,154" fill="#c2410c"/><polygon points="282,150 290,146 290,154" fill="#c2410c"/><text x="340" y="264" text-anchor="middle" font-size="12" fill="#52525b">FAISS はベクトルのみ。検索で返る faiss_id（−1 は近傍なし → スキップ）で SQLite を join。</text><text x="340" y="286" text-anchor="middle" font-size="11.5" fill="#71717a">.faiss（索引）と .db（メタ）は必ずセットで永続化・整合させる。</text></svg><figcaption><b>FAISS</b> は ID 付きの<b>ベクトルだけ</b>を持ち、フレーム番号や bbox などの<b>メタ情報は持てません</b>。そこで領域の代表ベクトル（平均 → L2 正規化）を <code>IndexFlatIP + IDMap</code> に登録し、同じ <b>faiss_id</b> をキーに <b>SQLite</b> の <code>VectorMapping</code> と結びます。検索では FAISS が返す <code>faiss_id</code>（<b>−1 は近傍なし。必ずスキップ</b>）で SQLite を join し、フレームと bbox を取り出します。索引(.faiss) とメタ(.db) は必ずセットで永続化します。</figcaption></figure>

**検索と可視化（`search/engine.py` 相当 / 05・mini_project の Search）**: まずクエリ（テキストは `encode_text`、画像領域はその代表ベクトル）を L2 正規化して `index.search` にかけます。次に、返ってきた `faiss_id`（**-1 は近傍不足なので必ずスキップ**）で SQLite を join し、`(frame, cluster, bbox, image_path)` を得ます。最後に、クラスタマップから該当領域のマスクを作り、`cv2.addWeighted` で重畳します。

**ストリーム（`stream/pipeline.py` 相当 / mini_project の Stream）**: `multiprocessing` で **capture（取得）/ consumer（dense CLIP 推論 = CPU バウンド）/ writer（記録）** の 3 プロセスに分け、間を `Queue` でつなぎます。取得が推論を追い越したときは、`put_nowait` で投入して `queue.Full` を捕まえ、**フレームを捨てます**（落とさず待つと遅延が無限に積み上がるためです）。そして終端では **POISON_PILL**（番兵オブジェクト）を流し、各プロセスを綺麗に終わらせます。

<figure class="lec-fig"><svg viewBox="0 0 680 300" role="img" aria-label="capture consumer writerを別プロセスにしQueueでつなぐ。満杯ならput_nowaitがFullを投げフレームをドロップし、終端でPOISON_PILLを流す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ストリーム: 取得・推論・記録を別プロセス化し、満杯キューはドロップ</text><rect x="40" y="78" width="120" height="60" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="100" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">capture</text><text x="100" y="124" text-anchor="middle" font-size="12" fill="#1d4ed8">取得</text><rect x="186" y="93" width="30" height="30" rx="4" fill="#fff7ed" stroke="#71717a" stroke-width="1.5"/><rect x="220" y="93" width="30" height="30" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="218" y="150" text-anchor="middle" font-size="11" fill="#52525b">Queue (max=2)</text><rect x="262" y="78" width="160" height="60" rx="7" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="342" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">consumer</text><text x="342" y="124" text-anchor="middle" font-size="11.5" fill="#c2410c">dense CLIP 推論 (CPU)</text><rect x="448" y="93" width="30" height="30" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><rect x="482" y="93" width="30" height="30" rx="4" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="480" y="150" text-anchor="middle" font-size="11" fill="#52525b">Queue</text><rect x="524" y="78" width="120" height="60" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="584" y="104" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">writer</text><text x="584" y="124" text-anchor="middle" font-size="12" fill="#15803d">記録</text><line x1="160" y1="108" x2="180" y2="108" stroke="#71717a" stroke-width="1.8"/><polygon points="186,108 178,104 178,112" fill="#71717a"/><line x1="250" y1="108" x2="256" y2="108" stroke="#71717a" stroke-width="1.8"/><polygon points="262,108 254,104 254,112" fill="#71717a"/><line x1="422" y1="108" x2="442" y2="108" stroke="#71717a" stroke-width="1.8"/><polygon points="448,108 440,104 440,112" fill="#71717a"/><line x1="512" y1="108" x2="518" y2="108" stroke="#71717a" stroke-width="1.8"/><polygon points="524,108 516,104 516,112" fill="#71717a"/><line x1="218" y1="124" x2="218" y2="180" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3"/><polygon points="218,186 214,178 222,178" fill="#dc2626"/><text x="218" y="206" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">× ドロップ</text><rect x="150" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><rect x="170" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><rect x="190" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><rect x="210" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><rect x="230" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><rect x="250" y="214" width="14" height="12" fill="#e4e4e7" stroke="#d4d4d8"/><text x="218" y="244" text-anchor="middle" font-size="11" fill="#52525b">推論が追いつかない分は捨てる</text><text x="500" y="172" text-anchor="middle" font-size="11" fill="#52525b">終端: POISON_PILL で全プロセス停止</text><text x="160" y="276" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">投入 8</text><text x="360" y="276" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">処理 2</text><text x="540" y="276" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">ドロップ 6</text></svg><figcaption>取得（<code>capture</code>）・dense CLIP 推論（<code>consumer</code>, CPU バウンド）・記録（<code>writer</code>）を <b>別プロセス</b>に分け、<code>Queue</code> でつなぎます。取得が推論を追い越すと、<code>put_nowait</code> が <code>queue.Full</code> を投げるので<b>フレームを捨てます</b>（待つと遅延が無限に積み上がるため）。終端では <b>POISON_PILL</b>（番兵）を流して各プロセスを綺麗に終わらせます。実行例は <b>投入 8 / 処理 2 / ドロップ 6</b>。</figcaption></figure>

## 6. 落とし穴（このトピック固有）

最大の落とし穴は、**dense 特徴で CLS を落とし忘れる / 位置埋め込みや ln の順序を間違える** ことです。CLS が混ざると、1 パッチ分だけ余計なトークンが領域に化け、クラスタリングがノイズだらけになってしまいます。したがって手で forward を書くときは、各ステップの shape（`+CLS` で系列長が 50、`drop CLS` で 49）を必ず print して確かめてください。

次に注意したいのが、**AgglomerativeClustering を高解像度のパッチに適用する** ことです。AgglomerativeClustering は O(n²) のメモリを食うため、14×14 = 196 程度までに抑えるのが現実的です（本章は 7×7 = 49）。たとえ dense 特徴を高解像度に取れたとしても、そのまま全パッチをクラスタリングに渡すと、メモリと時間が爆発してしまいます。

そして忘れてはならないのが、**FAISS のお約束** です。コサインのつもりで `IndexFlatIP` を使いながら正規化を忘れると、単なる内積になって順位が崩れます。このほか、`faiss.normalize_L2` が **in-place で元配列を破壊** する点、float32・C連続でないと落ちる点、検索結果に **-1** が混じる点（メタ参照前にガードする）も、定番の罠です。

## 7. 実務の使い分け

「画像全体で似た画像を引きたい」だけなら、従来の global CLIP + FAISS（17 章）で十分であり、dense は不要です。dense + クラスタリングが効いてくるのは、**1 枚に複数物体があり、その中の特定の領域・小物体をテキストで引きたい** ときや、検索結果に「画像のどこがヒットしたか」のマスクを出したいときに限られます。そのぶんコストは上がります（パッチ forward + クラスタリングがフレームごとに走るためです）。

クラスタ数 k は、「1 フレームあたり何本のベクトルを index に積むか」を直接決めるパラメータです。k を増やすほど細かい領域を引けますが、その代わりに index は重くなり、ノイズ領域も増えます。参考実装では適応的に k を決める実験もしていますが、入門としては固定 k（5〜8）で十分です。なおストリームでは、推論が実時間に追いつかない前提で **fps 上限 + キュー満杯ドロップ** を必ず入れ、「全フレームを処理する」のではなく「落としても破綻しない」設計にします。

---

## 🛠 章末ミニプロジェクト — ミニ Cluster-CLIP

`mini_project.py` は学んだ全要素を 1 本に束ねます。

```bash
uv run python lectures/40_cluster_clip_dense_cluster/mini_project.py
```

- **Split**: 黄色いボールが左→右に動く合成フレーム列を JPEG に分解（`lectures/40_cluster_clip_dense_cluster/outputs/mini_project/frames/`）。OpenCV は BGR をそのまま保存します。
- **Build**: 各フレームを dense CLIP → 空間連結クラスタリング → 代表ベクトル化。`cluster_maps/*.npy`・`vectors/*.npy` を保存し、`local_index.faiss`（IndexFlatIP+IDMap）と `local_metadata.db`（`Frames` / `VectorMapping`、`faiss_id` は SQLite の `lastrowid`）を構築。各クラスタの bbox（領域マスクの外接矩形）も登録。
- **Search**: テキストクエリ → `encode_text` → FAISS → SQLite join → ヒット領域をマスク重畳で `mini_project_search.png` に出力。`faiss_id == -1` をスキップし、メタ解決できることを assert で検証。
- **Stream**: `multiprocessing`（spawn）の capture / consumer / writer。`queue_size=2` の満杯キューに 8 フレームを投入し、推論が追いつかない分はドロップ。実行例では **投入 8 / 処理 2 / ドロップ 6**、実効 FPS と合わせて「取得が推論を追い越すと捨てる」挙動が観察できます。

<figure class="lec-fig"><svg viewBox="0 0 680 240" role="img" aria-label="ミニCluster-CLIPはSplitでフレーム分解、Buildで索引構築、Searchで領域検索、Streamで実時間化する4段パイプライン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニ Cluster-CLIP の 4 段パイプライン</text><rect x="20" y="86" width="130" height="66" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="190" y="86" width="130" height="66" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="360" y="86" width="130" height="66" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="530" y="86" width="130" height="66" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="85" y="124" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">① Split</text><text x="255" y="124" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">② Build</text><text x="425" y="124" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">③ Search</text><text x="595" y="124" text-anchor="middle" font-size="16" font-weight="700" fill="#15803d">④ Stream</text><line x1="152" y1="119" x2="184" y2="119" stroke="#71717a" stroke-width="2"/><polygon points="190,119 182,115 182,123" fill="#71717a"/><line x1="322" y1="119" x2="354" y2="119" stroke="#71717a" stroke-width="2"/><polygon points="360,119 352,115 352,123" fill="#71717a"/><line x1="492" y1="119" x2="524" y2="119" stroke="#71717a" stroke-width="2"/><polygon points="530,119 522,115 522,123" fill="#71717a"/><text x="171" y="108" text-anchor="middle" font-size="10.5" fill="#3f3f46">frames</text><text x="341" y="108" text-anchor="middle" font-size="10.5" fill="#3f3f46">.faiss/.db</text><text x="511" y="108" text-anchor="middle" font-size="10.5" fill="#3f3f46">実時間化</text><text x="85" y="178" text-anchor="middle" font-size="11" fill="#52525b">フレームを JPEG 分解</text><text x="255" y="178" text-anchor="middle" font-size="11" fill="#52525b">dense → クラスタ → 代表</text><text x="425" y="178" text-anchor="middle" font-size="11" fill="#52525b">クエリ → マスク重畳</text><text x="595" y="178" text-anchor="middle" font-size="11" fill="#52525b">取得・推論・記録を分離</text><text x="340" y="210" text-anchor="middle" font-size="11.5" fill="#52525b">① → ③ は順に流れ、④ Stream は ② と同じ推論を実時間で回す（満杯はドロップ）</text></svg><figcaption><b>章末ミニプロジェクト</b>は <b>Split → Build → Search → Stream</b> の 4 段で進みます。<b>① Split</b> が合成フレームを JPEG に分解し、<b>② Build</b> が各フレームを dense CLIP → 空間連結クラスタ → 代表ベクトルにして <code>.faiss</code>（索引）と <code>.db</code>（メタ）を作ります。<b>③ Search</b> はテキストクエリで領域を引き<b>マスク重畳</b>で可視化し、<b>④ Stream</b> は ② と同じ推論を <code>multiprocessing</code> で実時間化して、満杯キューのフレームを<b>ドロップ</b>します。</figcaption></figure>

この 1 本を通せば、参考実装の `split / build / search / stream` の対応関係（`build/models.py`・`indexer.py`・`db_writer.py`・`search/engine.py`・`stream/pipeline.py`）が腑に落ちるはずです。

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

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/40_cluster_clip_dense_cluster/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. **各行を L2 正規化する**（`ex1_l2_normalize_rows` の TODO）。ノルム 0 の行でも 0 除算で落ちないよう、eps を足してから割る。
2. **ViT トークン列 `[1+gh*gw, C]` から CLS を捨て、`[C, gh, gw]` の特徴マップに並べ替える**（`ex2_tokens_to_feature_map` の TODO）。
3. **ラベルマップに従って各クラスタの平均 → L2 正規化で代表ベクトル `reps[k, C]` を作る**（`ex3_mean_pool_clusters` の TODO）。
4. **空間連結クラスタリングで `feat_map[C,H,W]` を k 領域に分け、`(reps, label_map)` を返す**（`ex4_cluster_regions_connected` の TODO）。connectivity を有効にして飛び地を防ぐ。
5. **connectivity あり/なしでクラスタリングし、総連結成分数を `(with, without)` で返す**（`ex5_connectivity_components` の TODO）。
6. **代表ベクトルを `IndexFlatIP + IDMap` に登録し、各ベクトルで検索した top-1 の id 列を返す**（`ex6_self_search_top1` の TODO）。正規化済みなら自分自身が rank-1 になる。
7. **`(faiss_id, frame_id, cluster_idx)` を SQLite に格納し、指定した id で `{faiss_id: (frame_id, cluster_idx)}` を引き戻す**（`ex7_sqlite_roundtrip` の TODO）。
8. **FAISS 検索 → SQLite join で `(frame_id, cluster_idx)` の top-k を返す**（`ex8_search_regions` の TODO）。`faiss_id == -1`（近傍不足）は必ずスキップする。
9. **消費者のいない満杯キューへ `put_nowait` し、溢れてドロップした要素数を返す**（`ex9_bounded_put` の TODO）。`queue.Full` を捕まえて数える。
10. **クラスタマスクが bbox を覆う割合（マスク ∩ bbox / bbox 面積）を返す**（`ex10_coverage_ratio` の TODO）。

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

- **MaskCLIP / dense CLIP の整列改善**: 最後の自己注意の value 射影だけを使うと、パッチトークンがテキストにより整列します。参考実装の ResNet 経路（`attnpool` の `v_proj`/`c_proj` を 1×1 conv 化）はこの発想に近いものです（`build/models.py: dense_clip_embeddings_resnet`）。
- **適応サンプリング（AFS-MI）**: 参考実装はヒストグラム差分や相互情報量でフレームを間引き、似たフレームの無駄な推論を避けます（`build/producer.py`、`stream/capture.py`）。
- **カバレッジ評価**: GT bbox とクラスタマスクの被覆率で P@k / NDCG を測る評価系（`eval/coverage.py`）。本章の `coverage_ratio` / 演習 Q10 がその最小版です。
- **エッジ展開**: faiss-cpu と TorchScript 化した小型 CLIP で Jetson まで運ぶ設計（`docs/EDGE_DEPLOYMENT.md`）。35〜37 章（量子化・ONNX・エッジ最適化）とつながります。
- 本章のもとにした構成: dense CLIP 特徴抽出（build/models 相当）→ 空間連結クラスタリング → FAISS 索引（indexer）→ 検索エンジン（search/engine）→ ストリームパイプライン（stream/pipeline）。

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

出力画像は `lectures/40_cluster_clip_dense_cluster/outputs/` に保存されます（matplotlib は Agg バックエンド、`imshow` は呼びません。BGR↔RGB の変換に注意）。

---

版: torch 2.12+cpu / open-clip-torch 3.3 / faiss-cpu 1.14 / scikit-learn 1.9 ・ 2026-06 ・ CPU 前提（`model.eval()` + `inference_mode()`）
