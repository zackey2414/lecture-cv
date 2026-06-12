# 41_cluster_clip_pipeline: Cluster-CLIP 総仕上げ — Split → Build → Search → Stream

> トラック: **応用・統合（capstone）** ／ レベル: **上級** ／ 必要な依存グループ: `dl` `embed` `vector` `metrics`
> 前提モジュール: `40_cluster_clip_dense_cluster`（dense CLIP + 空間連結クラスタリング）／ `16_clip_zeroshot_retrieval`（CLIP の正準フロー）／ `17_faiss_image_search`（FAISS と SQLite メタ管理）／ `11_realtime_stream`（multiprocessing ストリーム）
> 題材: **Cluster-CLIP**（dense CLIP + 空間連結クラスタリング + FAISS + ストリームの手法。本章はその CPU 小型再構築）

---

## 🎯 この章のゴール

この章は講座の**総仕上げ**だ。これまで別々に学んできた「画像 I/O → テンソル → ViT/CLIP 埋め込み → クラスタリング → FAISS 近傍探索 → SQLite メタ管理 → multiprocessing ストリーム」を、ここで**1 本のパイプライン**に束ねる。題材は **Cluster-CLIP**（テキストで動画フレームを**開語彙＝オープンボキャブラリ**に検索するシステム）であり、その設計に忠実な CPU 完結版を自分の手で再構築する。

具体的には、次ができるようになる。

- **Split**: 動画を連番フレームに分割する（`cv2.VideoCapture` のループ、None/False ガード）。
- **Build**: 各フレームから **dense CLIP 特徴（パッチ単位埋め込み）** を取り出し、**空間連結ありの凝集型クラスタリング**で「意味の似た領域」に分割、各領域の**代表ベクトル**を作って **FAISS(IndexFlatIP+IDMap)** に登録、`faiss_id ↔ (frame, cluster)` を **SQLite** に保存する。
- **Search**: テキストクエリ（CLIP text）や**領域クエリ（画像）**で FAISS を引き、ヒットした**領域マスクを重畳して可視化**する。「画像全体 1 ベクトル」検索との違い（どこが効いたかを局在化できる）を説明できる。
- **Stream**: `multiprocessing` で**取得（軽い）と推論（CPU バウンドで重い）を別プロセス**に分け、キュー満杯時はフレームを**ドロップしてリアルタイム性を守る**。ヒストグラム差分の**適応サンプリング**で場面転換だけを残す。
- これら全体を `mini_project.py` で一気通貫に動かし、各部品が参考実装のどのモジュールに対応するかを言える。

本章のスクリプトはすべて **CPU・`model.eval()` + `inference_mode()`** で動き、入力は合成の小さな「動画」（色つきパネルが場面ごとに入れ替わる連番フレーム）だ。学習はせず、推論＋クラスタリング＋検索だけを行う。

> **合成データの正直な注意**: 本章のフレームは平坦な色領域の合成画像なので、**テキスト→領域**の意味整合はノイジーだ（CLIP のパッチ特徴は自己注意で文脈が混ざるため、玩具データでは色テキストと領域が綺麗に対応しない）。一方、**領域→領域**（画像クエリ）検索は埋め込みが素直に効くので安定して当たる。実写動画では両方とも実用的になるので、この差を体感するのも狙いの一つだ。

---


## 1. 直感 — なぜ「画像全体 1 ベクトル」ではなく「領域（dense）」なのか

16・17 章で扱った CLIP 画像検索は、1 枚の画像を **1 本のベクトル**（`encode_image` が返す CLS 埋め込み）に潰してから FAISS に入れていた。この方式は「この**画像**は猫っぽいか」には強い一方、「この画像の**どこに**小さな標識があるか」「複数物体のうち**どれ**がバスか」には弱い。なぜなら、画像全体の平均的な意味に薄まってしまい、画面の片隅にある小物体は埋もれてしまうからだ。とりわけ動画フレームのように 1 枚へ複数の物体が散らばる場面では、この弱点が致命的になる。

これに対する Cluster-CLIP の発想は単純だ。すなわち、**画像を 1 ベクトルに潰す前に、領域に分けてから領域ごとにベクトル化する**。CLIP の visual encoder は内部で画像を 7×7（ViT-B-32 の場合）のパッチに分け、各パッチにトークン埋め込みを持っている。普段は最後に CLS トークンだけを使うが、**パッチトークン列を捨てずに空間グリッドへ並べ直す**と、画像の「どの場所が何か」を保った密な特徴マップ `[C, H, W]` が得られる。これが **dense CLIP 特徴**だ（40 章で詳しく扱った）。

ただし、パッチ 49 個をそのまま FAISS に入れると、1 フレームあたり 49 ベクトルでインデックスが膨れるうえ、隣接パッチはほぼ同じ意味で冗長になる。そこで**空間的に隣り合う・意味の似たパッチをまとめて領域にし、その代表ベクトル（平均→正規化）だけを索引化する**。これで「小物体・複数物体を、領域単位で開語彙検索できる」という Cluster-CLIP の核ができる。本章は、この考えを Split→Build→Search→Stream の流れに乗せて完成させる。

<figure class="lec-fig"><svg viewBox="0 0 560 280" role="img" aria-label="画像全体を1本のベクトルに潰す方式と、領域に分けてk本にするdense方式の比較" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="90" y="50" width="100" height="100" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="100" y="62" width="22" height="22" fill="#dc2626"/><rect x="150" y="106" width="26" height="26" fill="#16a34a"/><rect x="120" y="120" width="15" height="15" fill="#2563eb"/><line x1="140" y1="150" x2="140" y2="174" stroke="#71717a" stroke-width="2"/><polygon points="140,180 134,168 146,168" fill="#71717a"/><rect x="103" y="184" width="74" height="26" rx="4" fill="#2563eb"/><text x="140" y="202" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">1 本</text><text x="140" y="240" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像全体 = 1 ベクトル</text><text x="140" y="262" text-anchor="middle" font-size="12.5" fill="#dc2626">小物体は平均に埋もれる</text><line x1="280" y1="46" x2="280" y2="214" stroke="#e4e4e7" stroke-width="1.5"/><text x="280" y="123" text-anchor="middle" font-size="15" font-weight="700" fill="#71717a">vs</text><rect x="370" y="50" width="100" height="100" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><rect x="370" y="50" width="50" height="50" fill="#dbeafe"/><rect x="420" y="50" width="50" height="50" fill="#ffedd5"/><rect x="370" y="100" width="50" height="50" fill="#f4f4f5"/><rect x="420" y="100" width="50" height="50" fill="#fafafa"/><rect x="370" y="50" width="50" height="50" fill="none" stroke="#2563eb" stroke-width="2"/><rect x="420" y="50" width="50" height="50" fill="none" stroke="#c2410c" stroke-width="2"/><rect x="370" y="100" width="50" height="50" fill="none" stroke="#16a34a" stroke-width="2"/><rect x="420" y="100" width="50" height="50" fill="none" stroke="#dc2626" stroke-width="2"/><line x1="420" y1="150" x2="420" y2="174" stroke="#71717a" stroke-width="2"/><polygon points="420,180 414,168 426,168" fill="#71717a"/><rect x="372" y="184" width="22" height="26" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="398" y="184" width="22" height="26" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><rect x="424" y="184" width="22" height="26" fill="#f4f4f5" stroke="#16a34a" stroke-width="1.5"/><rect x="450" y="184" width="22" height="26" fill="#fafafa" stroke="#dc2626" stroke-width="1.5"/><text x="420" y="240" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">領域 = k 本のベクトル</text><text x="420" y="262" text-anchor="middle" font-size="12.5" fill="#15803d">どこが何かを局在化</text></svg><figcaption>従来の <code>encode_image</code> は 1 枚を <b>1 本の CLS ベクトル</b>に潰すため、画面の片隅にある<b>小物体は平均に埋もれる</b>。Cluster-CLIP は画像を<b>領域(dense)に分けてから</b>領域ごとにベクトル化し、<b>k 本</b>を索引化するので、<b>どこに何があるか</b>を局在化して検索できる。</figcaption></figure>

## 2. 理論 — dense CLIP 特徴の取り出し（ViT を手で展開する）

`model.encode_image()` は CLS トークンしか返さないので、パッチ特徴を得るには **visual transformer を手で forward 展開**する必要がある。流れはこうだ（`cluster_clip_helpers.dense_clip_embeddings`）。まず `visual.conv1` で画像をパッチ埋め込み `[B, width, gh, gw]` にし、平坦化して `[B, gh*gw, width]` にする。次に先頭へ `class_embedding`（CLS）を連結し、`positional_embedding` を足して `ln_pre` を通す。transformer は `(seq, batch, dim)` 並びを期待するので、`permute(1,0,2)` してから通し、戻して `ln_post` にかける。最後に **`visual.proj`（射影行列）を掛けて、text と同じ共有埋め込み空間に揃える**。

ここで肝が 2 つある。第一に、**CLS トークンを最後に捨てる**（`x[:, 1:, :]`）。CLS は画像全体の要約なので、パッチごとの局所特徴が欲しい dense 表現ではかえって邪魔になる。捨て忘れるとクラスタリングがノイズだらけになる、というのが 40 章でも出た典型的な落とし穴だ。第二に、**パッチごとに L2 正規化**してから返す。CLIP の類似度はコサインで測るので、ここで正規化しておけば、後段の代表ベクトル計算や FAISS 内積がそのままコサインになる。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="ViTを手で展開してdense CLIP特徴[C,7,7]を取り出すパイプライン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="24" y="48" width="176" height="50" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="112" y="78" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">画像 224×224</text><rect x="222" y="48" width="176" height="50" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="310" y="67" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">conv1 → パッチ埋め込み<tspan x="310" dy="18">[49, w]</tspan></text><rect x="420" y="48" width="176" height="50" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="508" y="67" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">+CLS, +pos, ln_pre<tspan x="508" dy="18">→ Transformer</tspan></text><line x1="200" y1="73" x2="215" y2="73" stroke="#71717a" stroke-width="2"/><polygon points="221,73 212,68 212,78" fill="#71717a"/><line x1="398" y1="73" x2="413" y2="73" stroke="#71717a" stroke-width="2"/><polygon points="419,73 410,68 410,78" fill="#71717a"/><line x1="508" y1="98" x2="508" y2="151" stroke="#71717a" stroke-width="2"/><polygon points="508,157 502,147 514,147" fill="#71717a"/><rect x="420" y="158" width="176" height="50" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="508" y="177" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">ln_post · proj<tspan x="508" dy="18">共有埋め込み空間へ</tspan></text><rect x="222" y="158" width="176" height="50" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><text x="310" y="177" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">CLS 除去 x[:, 1:]<tspan x="310" dy="18" fill="#52525b">（局所特徴だけ残す）</tspan></text><rect x="24" y="158" width="176" height="50" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><text x="112" y="177" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">L2 正規化<tspan x="112" dy="18" fill="#52525b">（パッチ毎・コサイン用）</tspan></text><line x1="420" y1="183" x2="406" y2="183" stroke="#71717a" stroke-width="2"/><polygon points="398,183 407,178 407,188" fill="#71717a"/><line x1="222" y1="183" x2="208" y2="183" stroke="#71717a" stroke-width="2"/><polygon points="200,183 209,178 209,188" fill="#71717a"/><line x1="112" y1="208" x2="112" y2="235" stroke="#71717a" stroke-width="2"/><polygon points="112,241 106,231 118,231" fill="#71717a"/><rect x="24" y="242" width="252" height="46" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="150" y="270" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">dense 特徴 [C, 7, 7]</text></svg><figcaption><code>encode_image</code> は CLS しか返さないので、<b>visual transformer を手で forward 展開</b>する。<code>conv1</code> でパッチ埋め込みにし、CLS と位置埋め込みを足して Transformer→<code>ln_post · proj</code> で <b>text と同じ共有空間へ整列</b>。最後に <b>CLS を除去</b>(<code>x[:, 1:]</code>) して<b>パッチ毎に L2 正規化</b>し、<b>dense 特徴 [C, 7, 7]</b> を得る。CLS の落とし忘れと正規化漏れが二大事故。</figcaption></figure>

前処理も重要だ。標準の CLIP 前処理は `Resize(短辺) → CenterCrop(224)` で画像の端を切るが、Cluster-CLIP は **CenterCrop を排して正方形に強制 Resize** する（`load_clip`）。端を切ると、画面端の小物体の dense 特徴が丸ごと消えてしまうからだ。アスペクト比は崩れるものの、「端を捨てない」ことを優先するわけである。これは `force_quick_gelu=True`（openai 重みは quick-gelu 活性化）とセットで、参考実装 `build/models.py::load_clip_model` と同じ勘所だ。

## 3. 理論 — 空間連結クラスタリングと代表ベクトル

dense 特徴 `[C, H, W]` を `[H*W, C]`（各パッチ＝1 サンプル）に並べ替え、`sklearn.cluster.AgglomerativeClustering` で `n_clusters` 個の領域に凝集する（`cluster_agglomerative`）。ところが普通の凝集型クラスタリングは、「意味が似ていれば画面の反対側のパッチでも併合」してしまい、領域がバラバラに散ってしまう。それを防ぐのが **`grid_to_graph` による connectivity 制約**だ。これは「隣接するパッチ同士の辺」だけを持つグラフで、`connectivity=` に渡すと、**空間的に隣り合うパッチ同士しか併合されなくなる**。結果として、意味も位置も近いパッチがまとまり、画像が**連続した領域**に分かれる。

`linkage="ward"` + `metric="euclidean"` を使う。パッチ特徴は L2 正規化済みなので、ユークリッド距離はコサイン距離とほぼ同義になる（`‖a-b‖² = 2 - 2·cosθ`）。クラスタリングが終わったら、各クラスタに属するパッチ特徴の**平均を取り、もう一度 L2 正規化**して**代表ベクトル**にする。平均は「その領域の代表的な意味」を表し、正規化で FAISS の内積＝コサインに乗せられる。なお空クラスタ（connectivity 制約下では基本起きないが念のため）は、全体平均で代用する。

<figure class="lec-fig"><svg viewBox="0 0 600 285" role="img" aria-label="grid_to_graphの連結制約で隣接パッチを連続領域に併合し、平均とL2正規化で代表ベクトルを作る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="114" y="44" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">領域マップ cmap（連続領域）</text><rect x="30" y="55" width="72" height="72" fill="#dbeafe"/><rect x="102" y="55" width="96" height="48" fill="#ffedd5"/><rect x="30" y="127" width="72" height="96" fill="#f4f4f5"/><rect x="102" y="103" width="96" height="120" fill="#fafafa"/><line x1="54" y1="55" x2="54" y2="223" stroke="#e4e4e7"/><line x1="78" y1="55" x2="78" y2="223" stroke="#e4e4e7"/><line x1="126" y1="55" x2="126" y2="223" stroke="#e4e4e7"/><line x1="150" y1="55" x2="150" y2="223" stroke="#e4e4e7"/><line x1="174" y1="55" x2="174" y2="223" stroke="#e4e4e7"/><line x1="30" y1="79" x2="198" y2="79" stroke="#e4e4e7"/><line x1="30" y1="103" x2="198" y2="103" stroke="#e4e4e7"/><line x1="30" y1="127" x2="198" y2="127" stroke="#e4e4e7"/><line x1="30" y1="151" x2="198" y2="151" stroke="#e4e4e7"/><line x1="30" y1="175" x2="198" y2="175" stroke="#e4e4e7"/><line x1="30" y1="199" x2="198" y2="199" stroke="#e4e4e7"/><rect x="30" y="55" width="72" height="72" fill="none" stroke="#2563eb" stroke-width="2"/><rect x="102" y="55" width="96" height="48" fill="none" stroke="#c2410c" stroke-width="2"/><rect x="30" y="127" width="72" height="96" fill="none" stroke="#16a34a" stroke-width="2"/><rect x="102" y="103" width="96" height="120" fill="none" stroke="#dc2626" stroke-width="2"/><text x="114" y="245" text-anchor="middle" font-size="12.5" fill="#52525b">grid_to_graph：隣接のみ併合</text><line x1="198" y1="139" x2="219" y2="139" stroke="#71717a" stroke-width="2"/><polygon points="225,139 216,134 216,144" fill="#71717a"/><rect x="226" y="92" width="110" height="58" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="281" y="116" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">各領域の<tspan x="281" dy="18">パッチ平均</tspan></text><line x1="336" y1="121" x2="356" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="362,121 353,116 353,126" fill="#71717a"/><rect x="362" y="92" width="100" height="58" rx="6" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><text x="412" y="126" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">L2 正規化</text><line x1="462" y1="121" x2="489" y2="121" stroke="#71717a" stroke-width="2"/><polygon points="495,121 486,116 486,126" fill="#71717a"/><rect x="500" y="87" width="82" height="14" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="500" y="105" width="82" height="14" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><rect x="500" y="123" width="82" height="14" fill="#f4f4f5" stroke="#16a34a" stroke-width="1.5"/><rect x="500" y="141" width="82" height="14" fill="#fafafa" stroke="#dc2626" stroke-width="1.5"/><text x="541" y="173" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">reps[k, C]</text></svg><figcaption>dense 特徴を <code>[H·W, C]</code>（各パッチ=1 サンプル）に並べ、<code>AgglomerativeClustering</code> で併合する。<b><code>grid_to_graph</code> の connectivity 制約</b>により<b>空間的に隣り合うパッチ同士しか併合されず</b>、画像が<b>連続した領域</b>に分かれる（反対側の飛び地ができない）。各領域は<b>パッチ平均 → L2 正規化</b>で<b>代表ベクトル <code>reps[k, C]</code></b>になり、そのまま FAISS の内積＝コサインに乗る。</figcaption></figure>

CPU で現実的に回すコツは、**パッチ数を小さく保つ**ことだ。AgglomerativeClustering は `O(n²)` メモリなので、高解像度パッチ（14×14=196 や 16×16=256）だと重くなる。一方、ViT-B-32 の 7×7=49 パッチなら一瞬で終わる。参考実装も同じ理由でパッチ数を抑えている。出力は代表ベクトル `reps [k, C]` とラベルマップ `cmap [H, W]`（各パッチがどのクラスタか）の 2 つで、`cmap` は後で領域マスクの可視化に使う。

## 4. 正準 API — Split → Build → Search → Stream の 4 段と参考実装対応

Cluster-CLIP は 4 つのステージからなる。本章はそれを `cluster_clip_helpers.py` の関数として実装し、番号付きスクリプトは**薄いドライバ**としてそれを呼ぶ（これは参考実装の `scripts/run_*.py` が `src/adaptive_cluster_clip/` を呼ぶ構造そのものだ）。対応表は次のとおり。

<figure class="lec-fig"><svg viewBox="0 0 660 290" role="img" aria-label="Cluster-CLIPの全体フロー。動画をSplitでフレーム化しBuildで代表ベクトルをFAISSとSQLiteに索引化、Searchが索引を引く。Streamは同形式の索引を逐次構築する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">全体フロー — Split・Build・Search と、同形式の索引を作る Stream</text><rect x="14" y="58" width="70" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="1.8"/><text x="49" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">動画</text><rect x="112" y="58" width="92" height="58" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="158" y="84" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">① Split</text><text x="158" y="103" text-anchor="middle" font-size="10.5" fill="#52525b">フレーム分割</text><rect x="232" y="58" width="100" height="58" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="282" y="84" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">② Build</text><text x="282" y="103" text-anchor="middle" font-size="10.5" fill="#52525b">領域を索引化</text><rect x="360" y="58" width="152" height="58" rx="8" fill="#fafafa" stroke="#71717a" stroke-width="1.8"/><text x="436" y="84" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">index.faiss</text><text x="436" y="103" text-anchor="middle" font-size="12.5" fill="#3f3f46">metadata.db</text><rect x="540" y="58" width="100" height="58" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="590" y="84" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">③ Search</text><text x="590" y="103" text-anchor="middle" font-size="10.5" fill="#52525b">領域を検索</text><line x1="84" y1="87" x2="106" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="112,87 104,82 104,92" fill="#71717a"/><line x1="204" y1="87" x2="226" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="232,87 224,82 224,92" fill="#71717a"/><text x="218" y="50" text-anchor="middle" font-size="11" fill="#3f3f46">frames</text><line x1="332" y1="87" x2="354" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="360,87 352,82 352,92" fill="#71717a"/><text x="346" y="50" text-anchor="middle" font-size="10.5" fill="#3f3f46">add_with_ids</text><line x1="512" y1="87" x2="534" y2="87" stroke="#71717a" stroke-width="2"/><polygon points="540,87 532,82 532,92" fill="#71717a"/><text x="526" y="50" text-anchor="middle" font-size="10.5" fill="#3f3f46">search(q)</text><rect x="360" y="200" width="152" height="58" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="436" y="224" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">④ Stream</text><text x="436" y="243" text-anchor="middle" font-size="10.5" fill="#52525b">取得 → 推論 → 書込</text><line x1="436" y1="200" x2="436" y2="124" stroke="#71717a" stroke-width="2"/><polygon points="436,116 430,128 442,128" fill="#71717a"/><text x="448" y="166" text-anchor="start" font-size="10.5" fill="#3f3f46">逐次に索引化</text></svg><figcaption>Cluster-CLIP は <b>Split → Build → Search</b> を基本線に、<code>動画</code> → <code>frame_*.jpg</code> → 代表ベクトルを <code>add_with_ids</code> → <b><code>index.faiss</code> + <code>metadata.db</code></b> → <code>search(q)</code> とデータが流れる。<b>Stream</b> は同じ部品で<b>同形式の索引を逐次構築</b>するオンライン経路で、こちらも検索可能になる。各段は参考実装の <code>split/build/search/stream</code> に対応する。</figcaption></figure>

| ステージ | 本章のヘルパ関数 | 参考実装のモジュール | 役割 |
|---|---|---|---|
| **Split** | `run_split` | `split/processor.py` | 動画 → 連番フレーム JPEG |
| **Build** | `run_build` | `build/consumer.py` + `indexer.py` + `db_writer.py` | dense CLIP → 領域クラスタ → 代表ベクトル → FAISS + SQLite |
| **Search** | `search_index` / `overlay_cluster_mask` | `search/engine.py` + `visualizer.py` | クエリ → FAISS → SQLite → 領域マスク可視化 |
| **Stream** | `run_stream_pipeline` | `stream/pipeline.py` + `capture.py` + `consumer.py` + `writer.py` | 取得/推論/書込の 3 プロセス、ドロップ、適応サンプリング |

正準的な最小コードはこうなる。Build では、代表ベクトルを `faiss.IndexIDMap(faiss.IndexFlatIP(dim))` に `add_with_ids` し、`faiss_id` を SQLite の `VectorMapping(faiss_id, frame_id, cluster_idx)` に対応づける。**FAISS は連番の内部 ID しか持たない**ので、画像パス・クラスタ番号などのメタは別管理（SQLite）にし、検索結果の ID から JOIN で引く、というのが 17 章で学んだ実運用パターンだ。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="FAISSは連番のfaiss_idと代表ベクトルを持ち、SQLiteがfaiss_idからframeとclusterを引く対応関係" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="145" y="46" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">FAISS（IndexFlatIP + IndexIDMap）</text><rect x="20" y="56" width="250" height="150" rx="8" fill="#fafafa" stroke="#2563eb" stroke-width="1.8"/><rect x="36" y="84" width="46" height="30" rx="4" fill="#ffedd5" stroke="#c2410c" stroke-width="1.6"/><text x="59" y="104" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">id 0</text><rect x="94" y="86" width="158" height="26" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><line x1="126" y1="86" x2="126" y2="112" stroke="#dbeafe"/><line x1="158" y1="86" x2="158" y2="112" stroke="#dbeafe"/><line x1="190" y1="86" x2="190" y2="112" stroke="#dbeafe"/><line x1="220" y1="86" x2="220" y2="112" stroke="#dbeafe"/><rect x="36" y="146" width="46" height="30" rx="4" fill="#ffedd5" stroke="#c2410c" stroke-width="1.6"/><text x="59" y="166" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">id 1</text><rect x="94" y="148" width="158" height="26" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><line x1="126" y1="148" x2="126" y2="174" stroke="#dbeafe"/><line x1="158" y1="148" x2="158" y2="174" stroke="#dbeafe"/><line x1="190" y1="148" x2="190" y2="174" stroke="#dbeafe"/><line x1="220" y1="148" x2="220" y2="174" stroke="#dbeafe"/><text x="310" y="104" text-anchor="middle" font-size="11.5" fill="#3f3f46">同じ faiss_id</text><line x1="274" y1="118" x2="344" y2="118" stroke="#71717a" stroke-width="1.6" stroke-dasharray="5 3"/><polygon points="350,118 341,113 341,123" fill="#71717a"/><text x="475" y="46" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">SQLite VectorMapping</text><rect x="350" y="56" width="250" height="150" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="1.8"/><text x="397" y="92" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">faiss_id<tspan x="478">frame</tspan><tspan x="548">cluster</tspan></text><line x1="362" y1="102" x2="588" y2="102" stroke="#d4d4d8"/><line x1="438" y1="84" x2="438" y2="196" stroke="#e4e4e7"/><line x1="512" y1="84" x2="512" y2="196" stroke="#e4e4e7"/><text x="397" y="132" text-anchor="middle" font-size="12.5" fill="#3f3f46">0<tspan x="478">f00</tspan><tspan x="548">c2</tspan></text><text x="397" y="166" text-anchor="middle" font-size="12.5" fill="#3f3f46">1<tspan x="478">f00</tspan><tspan x="548">c5</tspan></text><rect x="20" y="230" width="580" height="46" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><text x="310" y="258" text-anchor="middle" font-size="12.5" fill="#c2410c">search(q) → I=[faiss_id…] を SQLite で JOIN（id = -1 はスキップ）</text></svg><figcaption><b>FAISS は連番の <code>faiss_id</code> しか持たない</b>ので、<code>add_with_ids(vec, ids)</code> で代表ベクトルを入れつつ、画像パス・クラスタ番号などのメタは <b>SQLite の <code>VectorMapping</code></b> に分けて持つ。検索は <code>D, I = index.search(q)</code> が返す <b><code>faiss_id</code></b> を鍵に SQLite を JOIN して <code>(frame, cluster)</code> を引く。近傍不足の <b><code>id = -1</code> は必ずスキップ</b>する。</figcaption></figure>

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

**`01_split_frames.py`（Split）**。合成動画を `cv2.VideoWriter`(mp4v) で作り、`cv2.VideoCapture` で開いてフレームを連番 JPEG に保存する。ここで叩き込むのは、`cap.isOpened()` が `False` を返す／`cap.read()` が `(False, None)` で終端を知らせる／`cv2.imread` は失敗時に**例外でなく `None`** を返す、という OpenCV の流儀だ。なお VideoWriter が使えない環境のために、フレーム直書きへのフォールバックも入れてある。出力はフレームのモンタージュ図。

**`02_build_index.py`（Build）**。`load_clip` で ViT-B-32 をロードし、`run_build` で全フレームを dense CLIP → クラスタリング → 代表ベクトル化し、`vectors/*.npy`・`cluster_maps/*.npy`・`index.faiss`・`metadata.db` を生成する。生成後には、**FAISS の `ntotal` と SQLite の件数が一致する**ことを確認し（不整合はバグの典型）、先頭 3 フレームについて「原画像 ｜ 色分けクラスタマップ ｜ 最大領域の重畳」を並べて領域分割の見え方を可視化する。`vectors = frames × clusters` という関係（12 フレーム×6 = 72 ベクトル）を必ず腹落ちさせること。

**`03_search_regions.py`（Search）**。3 種類の検索を比較する。(A) **テキスト→領域**: `encode_text` でクエリをベクトル化し `search_index` で領域を引く（Cluster-CLIP の看板機能。玩具データではノイジー）。(B) **領域→領域**: フレーム 0 の「赤い領域」の代表ベクトルをクエリにして、他フレームの赤領域を引く（**自己一致でスコア≈1.0**、上位が赤に偏る＝安定）。(C) **画像全体ベクトル**でのテキスト検索（ベースライン。「どのフレームか」までで「どの領域か」は出せない）。ヒット領域は `overlay_cluster_mask` で半透明赤の重畳にしてギャラリー化する。

**`04_stream_pipeline.py`（Stream）**。`run_stream_pipeline` を 2 シナリオで回す。シナリオ A は **passthrough + 小さいキュー**で、取得が推論を追い越すと `put_nowait` で**ドロップ**し実時間を守る様子（15 取得 / 9 ドロップ等）を見る。シナリオ B は**ヒストグラム差分サンプラー**で、4 つの場面（ショット）の**変わり目だけをキーフレームとして残し**、24 枚中 20 枚（約 83%）の推論を省く。最後に、ストリームで逐次構築した index がそのまま検索可能であることを確認する。torch を載せるので各プロセスは **spawn** で独立起動し、`if __name__ == "__main__":` ガードが必須だ。

<figure class="lec-fig"><svg viewBox="0 0 560 300" role="img" aria-label="取得・推論・書込を別プロセスに分け、キュー満杯時はドロップしPOISON_PILLで終了するストリーム構成" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="14" y="64" width="120" height="66" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="74" y="92" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">Capture<tspan x="74" dy="18" font-size="12" font-weight="400" fill="#52525b">取得（軽い）</tspan></text><rect x="146" y="80" width="46" height="34" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><rect x="149" y="85" width="12" height="24" fill="#d4d4d8"/><rect x="163" y="85" width="12" height="24" fill="#d4d4d8"/><rect x="177" y="85" width="12" height="24" fill="#d4d4d8"/><text x="169" y="128" text-anchor="middle" font-size="11" fill="#52525b">task_queue（小）</text><rect x="204" y="64" width="120" height="66" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/><text x="264" y="92" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">Consumer<tspan x="264" dy="18" font-size="12" font-weight="400" fill="#52525b">推論（重い・CPU）</tspan></text><rect x="336" y="80" width="46" height="34" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><rect x="339" y="85" width="12" height="24" fill="#d4d4d8"/><rect x="353" y="85" width="12" height="24" fill="#d4d4d8"/><rect x="367" y="85" width="12" height="24" fill="#d4d4d8"/><text x="359" y="128" text-anchor="middle" font-size="11" fill="#52525b">result_queue</text><rect x="394" y="64" width="120" height="66" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="454" y="92" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">Writer<tspan x="454" dy="18" font-size="12" font-weight="400" fill="#52525b">FAISS + SQLite</tspan></text><line x1="134" y1="97" x2="143" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="146,97 139,92 139,102" fill="#71717a"/><line x1="192" y1="97" x2="201" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="204,97 197,92 197,102" fill="#71717a"/><line x1="324" y1="97" x2="333" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="336,97 329,92 329,102" fill="#71717a"/><line x1="382" y1="97" x2="391" y2="97" stroke="#71717a" stroke-width="2"/><polygon points="394,97 387,92 387,102" fill="#71717a"/><line x1="169" y1="114" x2="169" y2="157" stroke="#dc2626" stroke-width="2"/><polygon points="169,163 163,153 175,153" fill="#dc2626"/><rect x="148" y="166" width="42" height="30" fill="#fafafa" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4 3"/><line x1="153" y1="170" x2="185" y2="192" stroke="#dc2626" stroke-width="1.6"/><line x1="185" y1="170" x2="153" y2="192" stroke="#dc2626" stroke-width="1.6"/><text x="169" y="216" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">満杯 → ドロップ</text><text x="380" y="240" text-anchor="middle" font-size="12" fill="#3f3f46">各キューに POISON_PILL を流して全プロセス終了</text></svg><figcaption>ストリームは <b>取得（軽い）</b>と <b>推論（重い・CPU バウンド）</b>を<b>別プロセス</b>に分ける。<code>task_queue</code> が満杯なら <code>put_nowait</code> で<b>フレームをドロップ</b>して実時間性を守る（15 取得 / 9 ドロップ 等）。終了は各キューに <b>POISON_PILL</b> を流して drain する。torch を載せる子は <b>spawn</b> 起動＋<code>__main__</code> ガードが必須。</figcaption></figure>

## 6. 落とし穴 — このパイプラインで実際にハマる所

最頻出は**色順とレイアウト**だ。OpenCV は BGR、Pillow/PyTorch/CLIP は RGB なので、フレームを `cv2.imread` で読んだら CLIP に渡す前に RGB へ、`matplotlib` で表示する前にも RGB へ変換する（忘れると色が反転し、推論精度が静かに落ちる）。テンソルは `(H,W,C)`↔`(C,H,W)` を取り違えない。また dense 特徴の取り出しでは、**CLS トークンの落とし忘れ**と**正規化の順序**が二大事故になる。

**FAISS 周り**も定番だ。コサインのつもりで `IndexFlatIP` を使いながら**正規化を忘れる**と、単なる内積になり結果が崩れる。入力が `float32`・**C 連続**でないと落ちる（`np.ascontiguousarray(x.astype("float32"))`）。さらに `search` の戻り値 `I` には **`-1`**（近傍不足）が混じることがあり、ID として使うと SQLite 参照でクラッシュするので必ずガードする。`IndexIDMap` を使わず `IndexFlat` に `add` すると、ID は 0..N-1 固定になり、後からメタと対応づけられなくなる。

**ヒストグラム差分サンプラー**には数学的な罠がある。各チャネルのヒストグラムを**別々に L2 正規化して連結**し `HISTCMP_INTERSECT` を取ると、交差値が 1 を超えて `delta = 1 - intersection` が負になり、`max(0, delta)` で**全部 0 にクランプされて 1 枚も間引けない**。そこで本章では、連結ヒストグラム全体を**和=1 の確率分布に L1 正規化**してから交差を取り、`delta ∈ [0,1]` を保証している（`HistogramDeltaSampler._compute_hist` のコメント参照）。**multiprocessing** では、torch を載せる子プロセスは `spawn` で起動し（fork は OpenMP デッドリスクがある）、モデルは**各子プロセスの中でロード**する。`spawn` は `__main__` ガードと「ターゲット関数がモジュールトップレベルで import 可能」であることを要求する。

## 7. 実務の使い分け — 領域 vs 全体、Flat vs ANN、サンプラー、GPU

**領域検索（Cluster-CLIP）と画像全体検索の使い分け**。「この画像は何の写真か」「似た雰囲気の画像」を探すなら、**画像全体 1 ベクトル**で十分かつ高速だ。一方、「画面のどこに小さな標識／特定の人物／特定の物体があるか」「複数物体を別々に引きたい」なら**領域検索**が要る。その代償はベクトル数の増加（フレーム×クラスタ）と Build コストで、クラスタ数 `k` がそのトレードオフのダイヤルになる。`k` を増やすと細かい物体を拾えるが索引が膨れ、減らすと領域が粗くなる。

**FAISS のインデックス選択**。本章は厳密・学習不要の `IndexFlatIP` を使う（数千〜数万ベクトルの規模なら CPU で十分）。規模が上がったら `IndexIVFFlat`（`nlist`/`nprobe` で精度↔速度）や `IndexHNSWFlat`（`efSearch` が主ダイヤル）に差し替える——API（`add`/`search`）は同じで、`index_factory("IVF1024,Flat")` のように文字列で組める（17 章）。**サンプリング戦略**は、密に全フレーム処理するか、固定間引き（N 枚に 1 枚）か、適応（ヒストグラム差分で場面転換を検出）かを、検索の取りこぼし（recall）と計算コストのバランスで選ぶ。**GPU** は本講座では使わないが、コードは `cuda` 可用なら使う書き方（`pick_device`）にしてあり、FAISS は `index_cpu_to_gpu` の有無だけで切り替えられる（`faiss.get_num_gpus()` を try/except でガードする）。

---

## 🛠 章末ミニプロジェクト

`mini_project.py` は **Split → Build → Search → Stream を一気通貫**で実行する総合課題だ。

1. **Split**: 合成動画を 10 フレームに分割。
2. **Build**: dense CLIP + 空間連結クラスタリングで 10×6 = 60 本の領域代表ベクトルを作り、FAISS + SQLite を構築。
3. **Search**: テキストクエリ（`"a green region"`）と**領域クエリ**（フレーム 0 の赤領域）の両方で検索し、結果ギャラリー（上段＝テキスト・下段＝領域）を `mini_project_summary.png` に保存。
4. **Stream**: 同じ部品で multiprocessing のストリーム索引を構築（適応サンプリングで 18 → 4 キーフレーム）、それも検索可能であることを確認。

<figure class="lec-fig"><svg viewBox="0 0 660 280" role="img" aria-label="ミニプロジェクトの4段。SplitでフレームBuildで60本を索引化Searchでテキストと領域クエリStreamで適応サンプリング、最後にa〜eを自己検証する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">ミニプロジェクト — 4 段を一気通貫し、(a)〜(e) を自己検証</text><rect x="18" y="56" width="144" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="90" y="86" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① Split</text><text x="90" y="106" text-anchor="middle" font-size="11" fill="#52525b">動画 → 10 フレーム</text><rect x="178" y="56" width="144" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/><text x="250" y="86" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② Build</text><text x="250" y="106" text-anchor="middle" font-size="11" fill="#52525b">10×6 = 60 本を索引化</text><rect x="338" y="56" width="144" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="410" y="86" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">③ Search</text><text x="410" y="106" text-anchor="middle" font-size="11" fill="#52525b">テキスト + 領域クエリ</text><rect x="498" y="56" width="144" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="570" y="86" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">④ Stream</text><text x="570" y="106" text-anchor="middle" font-size="11" fill="#52525b">18 → 4 キーフレーム</text><line x1="162" y1="88" x2="172" y2="88" stroke="#71717a" stroke-width="2"/><polygon points="178,88 170,83 170,93" fill="#71717a"/><line x1="322" y1="88" x2="332" y2="88" stroke="#71717a" stroke-width="2"/><polygon points="338,88 330,83 330,93" fill="#71717a"/><line x1="482" y1="88" x2="492" y2="88" stroke="#71717a" stroke-width="2"/><polygon points="498,88 490,83 490,93" fill="#71717a"/><line x1="90" y1="120" x2="90" y2="186" stroke="#71717a" stroke-width="2"/><polygon points="90,192 84,180 96,180" fill="#71717a"/><line x1="250" y1="120" x2="250" y2="186" stroke="#71717a" stroke-width="2"/><polygon points="250,192 244,180 256,180" fill="#71717a"/><line x1="410" y1="120" x2="410" y2="186" stroke="#71717a" stroke-width="2"/><polygon points="410,192 404,180 416,180" fill="#71717a"/><line x1="570" y1="120" x2="570" y2="186" stroke="#71717a" stroke-width="2"/><polygon points="570,192 564,180 576,180" fill="#71717a"/><rect x="60" y="192" width="540" height="72" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="330" y="214" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">自己検証 — assert (a)〜(e) がすべて True</text><text x="330" y="236" text-anchor="middle" font-size="10.5" fill="#3f3f46">(a) ベクトル数 = フレーム×クラスタ    (b) 領域クエリ自己一致 ≈ 1.0</text><text x="330" y="256" text-anchor="middle" font-size="10.5" fill="#3f3f46">(c) 領域検索の上位が赤に偏る    (d) ストリーム索引も検索可    (e) 間引きが効く</text></svg><figcaption><b>章末ミニプロジェクト</b>は <b>① Split → ② Build → ③ Search → ④ Stream</b> を一気通貫で走らせる。<code>10</code> フレームから <b>10×6 = 60 本</b>の領域代表ベクトルを索引化し、テキストと領域の両クエリで検索、ストリームは <b>18 → 4 キーフレーム</b>に間引く。最後に <b>(a) ベクトル数=フレーム×クラスタ</b>、<b>(b) 自己一致 ≈ 1.0</b>、(c) 上位が赤、(d) ストリーム索引も検索可、(e) 間引き、を <code>assert</code> で自己検証する。</figcaption></figure>

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
A. ほぼ仕様だ。本章のフレームは平坦色の合成画像で、CLIP のパッチ特徴は自己注意で文脈が混ざるため、色テキストと領域の意味整合が弱い。**領域→領域**検索（自己一致≈1.0、上位が赤に偏る）が安定して当たることを確認すれば、パイプラインは正しい。実写画像なら text→region も実用になる。

**Q. `RuntimeError: ... could not open index.faiss` が出る。**
A. Build を先に実行していないか、別ディレクトリを見ている。03 は `ensure_build`（03 固有）で索引が無い時だけ自動 Build し、mini は実行のたびに `run_build`（冪等）で必ず作り直す。手動で消した場合は `02_build_index.py` を回す。`run_build` は冪等なので、再実行すると DB・index・npy を作り直す。

**Q. ストリームが固まる／終わらない。**
A. POISON_PILL の流れを確認する。Capture は終了時に `task_queue.put(POISON_PILL)` を送り、Consumer はそれを受けて残バッチを flush して return、main が `result_queue.put(POISON_PILL)` を送って Writer を終わらせる。各キューを最後まで drain する者がいないとデッドロックする。また `spawn` のため、ターゲット関数は**モジュールトップレベル**にあり、起動側に `__main__` ガードが要る。

**Q. `faiss` の `search` が落ちる／結果が変。**
A. 入力が `float32`・C 連続か（`np.ascontiguousarray(x, dtype=np.float32)`）、次元が index と一致しているか、正規化済みか（`IndexFlatIP` でコサインにするには DB 側もクエリ側も L2 正規化）を順に疑う。

**Q. 色が反転して見える／推論が変。**
A. BGR↔RGB の取り違えだ。`cv2.imread`/`cv2.VideoCapture` は BGR を返すので、CLIP へ渡す前・`matplotlib` で描く前に `cv2.cvtColor(..., COLOR_BGR2RGB)` をかける。

**Q. `cv2.VideoWriter` が開けない（`isOpened()` が False）。**
A. mp4v コーデックが無い環境だ。`01` はその場合フレーム直書きにフォールバックする。実運用では ffmpeg をインストールする。

**Q. クラスタリングが遅い／メモリを食う。**
A. パッチ数が大きすぎる。ViT-B-32 の 7×7=49 に抑える（高解像度パッチは `O(n²)` で爆発する）。`n_clusters` も `H*W` 以下にクランプされる（`cluster_agglomerative`）。

---

## 🚀 発展トピック・参考

- **適応サンプリングの高度化**: 参考実装はヒストグラム差分に加え **AFS-MI（相互情報量ベース）** のサンプリングを持つ（`build/producer.py`）。場面の情報量変化でフレームを間引く。
- **評価（Eval）**: 参考実装は GT BBox と**クラスタマスクのカバレッジ**で P@k / NDCG を測る（`eval/coverage.py`）。本章の演習 Q7 `mask_coverage` がその最小版で、Cluster マスク ∩ BBox / BBox 面積によって「領域が正解物体をどれだけ覆ったか」を測る。
- **Build カプセルと転送**: 参考実装は Build 結果を独立した「カプセル」（`index.faiss` + `metadata.db` + manifest）として保存し、Edge(Jetson)→Server へ tar.gz で転送する（`transfer/sender.py`）。本章の `build/` ディレクトリがその縮小版だ。
- **エッジ最適化**: 参考実装は MobileCLIP の TorchScript（`mobileclip_blt.ts`, 599MB）で Jetson 推論する。35〜37 章の量子化・ONNX・ランタイム最適化が効いてくる領域だ。
- **ANN へのスケール**: 規模が上がったら `IndexIVFFlat` / `IndexHNSWFlat` / PQ 圧縮へ（17 章）。検索 API は同じまま差し替えられる。
- **構成を押さえる**: コアは dense CLIP 特徴抽出（`dense_clip_embeddings_vit` / `cluster_agglomerative` 相当）、検索エンジン（search/engine）、ストリーム（stream）だ。本章の各関数のコメントに対応構成を明記してある。

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

出力（フレーム・カプセル・図）は `outputs/41_cluster_clip_pipeline/` に保存される。初回は CLIP（ViT-B-32, openai）の重みダウンロードが走る（以降はキャッシュされる）。すべて CPU で数十秒以内に完走する。

---

> 版: torch **2.12+cpu** / open_clip **3.3** / faiss-cpu **1.14** / transformers 5.11 / scikit-learn 1.9 ／ 2026-06
> 題材: **Cluster-CLIP**（本章はその CPU 小型再構築。dense CLIP → 空間連結クラスタリング → 代表ベクトル → FAISS → SQLite → multiprocessing ストリーム を一貫して再現）
