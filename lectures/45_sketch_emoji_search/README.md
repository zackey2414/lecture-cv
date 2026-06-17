# 第45回 手書きスケッチで絵文字を検索する（CLIP＋FAISS のスケッチ画像検索 SBIR）

> トラック: 埋め込み・検索 ／ レベル: 中級 ／ 必要な依存グループ: `dl` `hf` `vector`
> （`uv sync --group dl --group hf --group vector`／`metrics` は発展課題向けの任意グループ）
> 前提回: 第16回（CLIP ゼロショット検索）・第17回（FAISS 画像検索）

## 🎯 この章のゴール

この章を終えたとき、あなたは**手書きのスケッチを入力に絵文字を検索する「スケッチベース画像検索（SBIR: Sketch-Based Image Retrieval）」システムを、CLIP 埋め込みと FAISS で一から組み立てられる**ようになります。やることは煎じ詰めれば第16・17回の応用で、(1) 絵文字を CLIP でベクトル化して FAISS 索引に貯め、(2) マウスで描いた線画を同じ手順でベクトル化し、(3) コサイン類似度で「似ている絵文字」を上位 N 件引いてくる、という3段のパイプラインです。これまでと違う新しい山場は2つあります。1つは「クエリが写真ではなく**手書き線画**である」こと、もう1つは入力 UI を **headless な OpenCV でも動く Tkinter キャンバス**で作ることです。

ただし、SBIR には固有のむずかしさがあります。検索対象（絵文字＝色つきの塗り）とクエリ（スケッチ＝白地に黒線）は**見た目の様式がまるで違う**からです——この隔たりを**ドメインギャップ**と呼びます。古典的な SBIR はこのギャップを埋めるために、画像側をエッジ抽出して線画に寄せる、といった前処理を重ねてきました。本章でもまず、絵文字を**グレースケール化**してクエリ様式へ寄せる素直な方針を既定にします。そのうえで `04_eval_domaingap.py` を使い、「グレースケール／エッジ／反転」が Recall@N をどう動かすかを**正解ラベルつきで実測**します。すると見えてくるのは、「**強い共有埋め込み（CLIP）はドメインギャップをかなり自前で橋渡しする**」という、現代的で少し意外な結論です。

到達点を一言でいえば、**「絵文字を索引化 → 手書き（マウス描画 or 合成）スケッチで検索 → 上位 N 件を返す」エンドツーエンドのアプリを、ローカル GUI と headless（Docker/CI）の両方で完走させられる**ことです。加えて、なぜグレースケール化するのか、なぜクエリと DB を同じエンコーダ・同じ正規化に通すのか、ドメインギャップとは何でどう測るのか、を自分の言葉で説明できることも目標になります。

---

## 1. SBIR の直感と全体設計

ベクトル検索の発想は第17回と同じで、「似ているもの＝埋め込み空間で近いもの」です。SBIR が普通の画像検索と違うのは、**クエリと検索対象がモダリティ（様式）の異なる画像**だという点だけです。検索対象は完成された絵文字、クエリは人が描いたラフな線画——この2つを**同じ埋め込み空間の同じ座標系**に写せれば、「この落書きに一番近い絵文字はどれ？」という問いはコサイン類似度の計算に落とせます。その鍵は、**絵文字もスケッチも同一の CLIP 画像エンコーダに通す**ことです。CLIP は画像とテキストを共有空間に置くモデルですが、本章では「画像エンコーダ side」だけを2種類の画像（絵文字／スケッチ）に使い、画像どうしの類似検索を行います。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="絵文字とスケッチを同じCLIP画像エンコーダに通すと共有埋め込み空間の同じ座標に写り、コサイン類似度で最近傍の絵文字が引ける" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="15" fill="#3f3f46">同じ CLIP に通せば、様式が違っても同じ空間へ</text><circle cx="66" cy="80" r="28" fill="#f97316" stroke="#c2410c" stroke-width="2"/><circle cx="56" cy="74" r="3.5" fill="#18181b"/><circle cx="76" cy="74" r="3.5" fill="#18181b"/><path d="M54 86 Q66 98 78 86" fill="none" stroke="#18181b" stroke-width="2.5"/><text x="66" y="128" text-anchor="middle" font-size="12.5" fill="#52525b">絵文字（塗り）</text><rect x="40" y="180" width="54" height="54" fill="#ffffff" stroke="#71717a" stroke-width="1.8"/><circle cx="57" cy="200" r="3" fill="#18181b"/><circle cx="77" cy="200" r="3" fill="#18181b"/><path d="M55 212 Q67 222 79 212" fill="none" stroke="#18181b" stroke-width="2.2"/><text x="67" y="250" text-anchor="middle" font-size="12.5" fill="#52525b">スケッチ（線画）</text><line x1="100" y1="86" x2="244" y2="128" stroke="#71717a" stroke-width="1.8"/><polygon points="250,131 239,123 240,134" fill="#71717a"/><line x1="100" y1="206" x2="244" y2="162" stroke="#71717a" stroke-width="1.8"/><polygon points="250,159 240,156 239,167" fill="#71717a"/><rect x="250" y="108" width="148" height="72" rx="8" fill="#ea580c"/><text x="324" y="140" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">CLIP 画像</text><text x="324" y="164" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">エンコーダ</text><line x1="398" y1="144" x2="462" y2="144" stroke="#71717a" stroke-width="2"/><polygon points="470,144 460,139 460,149" fill="#71717a"/><rect x="470" y="66" width="174" height="168" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/><text x="557" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">共有埋め込み空間</text><circle cx="560" cy="120" r="7" fill="#2563eb"/><circle cx="582" cy="132" r="7" fill="#2563eb"/><circle cx="546" cy="136" r="7" fill="#2563eb"/><circle cx="612" cy="194" r="7" fill="#2563eb"/><circle cx="504" cy="168" r="7" fill="#2563eb"/><line x1="566" y1="146" x2="560" y2="126" stroke="#c2410c" stroke-width="1.5" stroke-dasharray="4 3"/><polygon points="566,138 576,150 566,162 556,150" fill="#f97316" stroke="#c2410c" stroke-width="1.2"/><text x="557" y="226" text-anchor="middle" font-size="11.5" fill="#52525b">近い＝似た絵文字</text></svg><figcaption>検索対象の<b>絵文字（塗り）</b>とクエリの<b>スケッチ（線画）</b>は見た目の様式が違っても、<b>同じ CLIP 画像エンコーダ</b>に通せば<b>共有埋め込み空間</b>の同じ座標系に写ります。あとはオレンジの<b>クエリ</b>に最も近い点を<b>コサイン類似度</b>で引くだけ——SBIR は「近いもの＝似た絵文字」に帰着します。</figcaption></figure>

この発想を踏まえると、本章のパイプラインは「準備フェーズ」と「検索フェーズ」の2つに分かれます。準備フェーズ（`01`）は、絵文字コレクションを用意 → グレースケール化 → CLIP 埋め込み → L2 正規化 → `IndexFlatIP`（コサイン）を `IndexIDMap2` で包んで `add` → `.faiss` と `.json` をセット永続化、という流れです。続く検索フェーズ（`02`→`03`）は、Tkinter キャンバスで手書き入力 → 前処理（白背景・黒線・正方化）→ 同じ CLIP で埋め込み → 正規化 → `index.search` → 上位 N 件をスコア付きで可視化、という流れになります。なお `04` は評価専用、`mini_project.py` はこれらを1本に統合した完成アプリです。

<figure class="lec-fig"><svg viewBox="0 0 660 320" role="img" aria-label="準備フェーズと検索フェーズの2系統が、中央のCLIP埋め込みとL2正規化を共通の核として共有するパイプライン" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="250" y="120" width="160" height="80" rx="10" fill="#ea580c"/><text x="330" y="108" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">両フェーズで共通</text><text x="330" y="153" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">CLIP 埋め込み</text><text x="330" y="177" text-anchor="middle" font-size="15" font-weight="700" fill="#ffffff">→ L2 正規化</text><rect x="40" y="70" width="150" height="50" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><text x="115" y="100" text-anchor="middle" font-size="13" fill="#c2410c">絵文字（グレー化）</text><rect x="40" y="200" width="150" height="50" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="115" y="230" text-anchor="middle" font-size="13" fill="#1d4ed8">スケッチ（前処理）</text><line x1="190" y1="96" x2="245" y2="140" stroke="#c2410c" stroke-width="2"/><polygon points="251,144 240,138 242,149" fill="#c2410c"/><line x1="190" y1="224" x2="245" y2="180" stroke="#2563eb" stroke-width="2"/><polygon points="251,176 242,171 240,182" fill="#2563eb"/><line x1="410" y1="140" x2="465" y2="96" stroke="#c2410c" stroke-width="2"/><polygon points="470,92 459,96 466,105" fill="#c2410c"/><line x1="410" y1="180" x2="465" y2="224" stroke="#2563eb" stroke-width="2"/><polygon points="470,228 466,215 459,224" fill="#2563eb"/><rect x="470" y="70" width="160" height="52" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/><text x="550" y="92" text-anchor="middle" font-size="12.5" fill="#c2410c">FAISS 索引へ add</text><text x="550" y="110" text-anchor="middle" font-size="11.5" fill="#c2410c">.faiss / .json 保存</text><rect x="470" y="198" width="160" height="52" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="550" y="220" text-anchor="middle" font-size="12.5" fill="#1d4ed8">index.search</text><text x="550" y="238" text-anchor="middle" font-size="11.5" fill="#1d4ed8">→ 上位 N 件</text></svg><figcaption>本章のパイプラインは<b>準備フェーズ</b>（絵文字をグレー化して索引化）と<b>検索フェーズ</b>（スケッチで検索）の2系統ですが、中央の <b>CLIP 埋め込み → L2 正規化</b> は<b>両フェーズで完全に同じ</b>でなければなりません。クエリと DB を同じエンコーダ・同じ正規化に通すことが、コサイン検索が成立する大前提です。</figcaption></figure>

このパイプラインを組むうえで、設計上の制約が2つあります。1つは、**本講座の OpenCV が headless ビルド**であり、`cv2.imshow` / `namedWindow` / `setMouseCallback` を**持たない**ことです。そのため手書き UI は OpenCV ではなく**標準ライブラリの `tkinter.Canvas`** で作ります。もう1つは、**display が無い環境（Docker/CI/SSH）では Tkinter のウィンドウ生成自体が失敗する**ことです。これを `try/except` で捕まえ、**合成スケッチ（線画）に自動フォールバック**して必ず `exit 0` で完走させます。この「ローカルでは本物の手書き、CI では合成」という二段構えこそが、教材を“どこでも動く”ものにしています。なお共通の道具はすべて `emoji_lab.py` に集約し、`01`〜`04` と `mini_project` はそれを組み合わせるだけ、という構成です。

## 2. 絵文字コレクションの用意（data → フォント → 合成）

検索対象の絵文字は、環境に左右されず**必ず手に入る**必要があります。そこで `emoji_lab.build_emoji_collection()` は3段の優先順位で集合を用意します。**(1)** `data/45_sketch_emoji_search/emoji/*.png` があればそれを最優先で採用します（自分の絵文字セットに差し替えたい人向けの導線です）。**(2)** 無ければ**絵文字フォント**（Noto Color Emoji / Symbola / Noto Emoji 等を既知パスと `fc-list` で探索）を使い、😀😃😄😁😅😂🙂😉😊😍😜🤔😐😑🙄😏😪😴😡😢😱😳 など多様な表情の顔絵文字を PIL でラスタライズします。**(3)** フォントも使えなければ、**合成絵文字**（黄色い円＋表情ちがいの目・眉・口を PIL で描く happy/sad/neutral/surprised/angry/wink/laugh など16種）にフォールバックします。要は、どの経路でも必ず完走するのが肝です。

<figure class="lec-fig"><svg viewBox="0 0 660 290" role="img" aria-label="絵文字コレクションは3段フォールバック。dataのPNGがあれば採用、無ければ絵文字フォントでラスタライズ、それも無ければ合成絵文字を生成し、どの経路でも必ず集合を返す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">絵文字コレクション — 3 段フォールバック</text><rect x="30" y="64" width="180" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="240" y="64" width="180" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><rect x="450" y="64" width="180" height="58" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="120" y="91" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">① data の PNG?</text><text x="120" y="110" text-anchor="middle" font-size="11" fill="#71717a">あれば最優先</text><text x="330" y="91" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">② 絵文字フォント?</text><text x="330" y="110" text-anchor="middle" font-size="11" fill="#71717a">fc-list で探索</text><text x="540" y="91" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">③ 合成絵文字</text><text x="540" y="110" text-anchor="middle" font-size="11" fill="#71717a">常に成功</text><line x1="210" y1="93" x2="234" y2="93" stroke="#71717a" stroke-width="2"/><polygon points="240,93 230,88 230,98" fill="#71717a"/><text x="225" y="83" text-anchor="middle" font-size="11" fill="#3f3f46">なし</text><line x1="420" y1="93" x2="444" y2="93" stroke="#71717a" stroke-width="2"/><polygon points="450,93 440,88 440,98" fill="#71717a"/><text x="435" y="83" text-anchor="middle" font-size="11" fill="#3f3f46">なし</text><rect x="30" y="188" width="180" height="56" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><rect x="240" y="188" width="180" height="56" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><rect x="450" y="188" width="180" height="56" rx="8" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="120" y="221" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">data 絵文字を採用</text><text x="330" y="221" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">フォントを描画</text><text x="540" y="221" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">合成絵文字を生成</text><line x1="120" y1="124" x2="120" y2="182" stroke="#71717a" stroke-width="2"/><polygon points="120,188 115,178 125,178" fill="#71717a"/><text x="140" y="158" text-anchor="start" font-size="11" fill="#3f3f46">あり</text><line x1="330" y1="124" x2="330" y2="182" stroke="#71717a" stroke-width="2"/><polygon points="330,188 325,178 335,178" fill="#71717a"/><text x="350" y="158" text-anchor="start" font-size="11" fill="#3f3f46">あり</text><line x1="540" y1="124" x2="540" y2="182" stroke="#71717a" stroke-width="2"/><polygon points="540,188 535,178 545,178" fill="#71717a"/><text x="330" y="272" text-anchor="middle" font-size="12" fill="#52525b">どの経路でも必ず RGB 絵文字集合を返す（exit 0）</text></svg><figcaption><code>build_emoji_collection()</code> は <b>3 段の優先フォールバック</b>で検索対象を必ず用意します。<b>① <code>data/.../emoji/*.png</code> があれば最優先で採用</b>、無ければ <b>② 絵文字フォント</b>（<code>fc-list</code> で探索）でラスタライズ、それも無ければ <b>③ 合成絵文字</b>（PIL 描画）を生成します。<b>どの経路でも必ず RGB の絵文字集合を返す</b>ので、フォントの有無に関わらず <code>exit 0</code> で完走します。グレースケール化は索引側（<code>01</code>）の役割です。</figcaption></figure>

ところが、絵文字フォントの描画には**地味だが重要な罠**があります。Noto Color Emoji のような**カラー絵文字フォント（埋め込みビットマップ）**は、`ImageFont.truetype(path, size)` の `size` に**特定の値（例: 109px）しか受け付けません**。任意サイズで開こうとすると `OSError: invalid pixel size` で落ちてしまいます。そこで `_render_emoji_glyph()` は複数サイズ（109, 128, 96, 64, …）を順に試し、最初に開けたサイズで描いてから目的サイズへリサイズします。さらに、フォントにそのグリフが無いと真っ白な画像になるため、「非白画素が極端に少なければ描けていない」とみなしてスキップする健全性チェックも入れています。なおカラー絵文字を描くときは `draw.text(..., embedded_color=True)` が必要で、旧 Pillow ではこの引数が無いため `try/except` で単色描画にフォールバックします。

最後に、**どの経路でも最終的にグレースケール化して索引に入れる**のが本章の方針です（理由は次節で述べます）。`build_emoji_collection()` は色つき RGB のまま返し、グレースケール化は索引構築側（`01`）が `to_grayscale_rgb()` で行います。役割をこう分けておくと、`04` のドメインギャップ実験で「色のまま／グレースケール／エッジ／反転」を**同じコレクションから作り分けられる**ため、前処理を1か所（`style_transform()`）に集約できます。実行環境にフォントがあるかどうかは、`01` の出力ログ（`絵文字フォント描画（NotoColorEmoji.ttf）26 種` など）で確認できます。

## 3. CLIP 埋め込みと「なぜグレースケール？」

埋め込みは第16回と同じく `openai/clip-vit-base-patch32` の画像エンコーダを使います。CLIP は CPU でも現実的な速度で動く定番で、`model.get_image_features(pixel_values=...)` から射影後ベクトルを得ます。ここで **transformers v5 の落とし穴**を改めて強調しておきます。`get_image_features` の戻り値は**テンソルではなく `BaseModelOutputWithPooling` オブジェクト**で、射影後ベクトル `(N, 512)` は `.pooler_output` に入っており、**しかも L2 正規化されていません**。したがって、コサイン類似度で検索する前に自分で正規化する必要があります。`emoji_lab.EmojiEmbedder` はこの作法を内側に隠したうえで、CLIP を取得できない環境では**画素記述子（24×24 グレースケールの平坦化）へ自動フォールバック**して `exit 0` を守ります。

では**なぜ絵文字をグレースケール化**するのでしょうか。素直な狙いは「**スケッチ（白地に黒線）と絵文字のドメインを寄せる**」ことです。スケッチには色がありません。にもかかわらず絵文字側に「黄色い」「赤い」といった色の手がかりが残っていると、それは検索の役に立たないどころか、**無関係な軸でベクトルを散らす雑音**になりかねません（例えば実画像コレクションで「色」が表情よりも強く効いてしまう、など）。色を落として「形・線・濃淡」へ寄せるのは、古典 SBIR がエッジ抽出で線画化してきたのと同じ発想に立つ、最も軽量な一手です。

ただし——ここが本章の誠実なところですが——**CLIP のような強い意味ベースのエンコーダでは、グレースケール化の効果は小さい**ことが実測で分かります（`04`）。CLIP は「笑った顔」「驚いた顔」という**高レベルの意味**で照合するため、塗りであれ線画であれ「同じ表情の顔」をそこそこ近くに置いてくれるからです。つまりグレースケール化は「**当たりを劇的に上げる魔法**」ではなく、「**色という無関係な手がかりを落とす安全策**」と理解するのが正確です。劇的な差が出るのは、むしろ**低レベルの記述子を使うとき**です（次節と§8）。そこで本章は、「既定はグレースケール、その理由は安全策、効果の大小はエンコーダ次第」という立場で統一します。

## 4. FAISS 索引 — 正規化＋IndexFlatIP＋IndexIDMap、そして永続化

索引づくりは第17回そのままです。CLIP の埋め込みは未正規化なので、まず `emoji_lab.l2_normalize()` で**各行をノルム1**にしてから `faiss.IndexFlatIP`（内積）に入れます。**L2 正規化したベクトルどうしの内積＝コサイン類似度**なので、これで「向きの近さ」による検索が成立します。注意点も第17回と同じです。すなわち、DB 側もクエリ側も**両方**正規化すること、`faiss.normalize_L2` は in-place（破壊的）なので非破壊版 `l2_normalize`（新しい配列を返す）と使い分けること、そして FAISS に渡す配列は必ず `float32`・C連続にすること（`as_faiss_array`）の3点です。

加えて、絵文字には**自分で決めた ID** を振りたいので、`IndexFlatIP` を `IndexIDMap2` で包み、`add_with_ids(vecs, ids)` で任意の `int64` ID を割り当てます（本章は連番と区別できるよう 100 番台にします）。FAISS が覚えるのは「ベクトルと ID」だけなので、**ID → ラベル（絵文字名）の対応表は別管理**にします。第17回では SQLite を使いましたが、絵文字は数十件と小さいので本章は **JSON**（`emoji_meta.json`）で十分です。`.faiss`（ベクトル＋ID）と `.json`（メタ）は「**セットで1つの検索DB**」であり、どちらか片方だけ更新すると ID から中身が引けなくなる——この整合の鉄則は規模に依らず同じです。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="faiss側はIndexIDMap2が正規化ベクトルとID(100番台)を保持し、JSON側はID対ラベルを持ち、IDで結合してひとつの検索DBになる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="160" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">.faiss ＝ IndexIDMap2(IndexFlatIP)</text><rect x="30" y="72" width="260" height="156" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="58" y="117" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" font-family="'JetBrains Mono', monospace">101</text><rect x="90" y="99" width="185" height="26" rx="4" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><line x1="125" y1="101" x2="125" y2="123" stroke="#f97316" stroke-width="0.8"/><line x1="160" y1="101" x2="160" y2="123" stroke="#f97316" stroke-width="0.8"/><line x1="195" y1="101" x2="195" y2="123" stroke="#f97316" stroke-width="0.8"/><line x1="230" y1="101" x2="230" y2="123" stroke="#f97316" stroke-width="0.8"/><text x="58" y="161" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" font-family="'JetBrains Mono', monospace">102</text><rect x="90" y="143" width="185" height="26" rx="4" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><line x1="125" y1="145" x2="125" y2="167" stroke="#f97316" stroke-width="0.8"/><line x1="160" y1="145" x2="160" y2="167" stroke="#f97316" stroke-width="0.8"/><line x1="195" y1="145" x2="195" y2="167" stroke="#f97316" stroke-width="0.8"/><line x1="230" y1="145" x2="230" y2="167" stroke="#f97316" stroke-width="0.8"/><text x="58" y="205" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c" font-family="'JetBrains Mono', monospace">103</text><rect x="90" y="187" width="185" height="26" rx="4" fill="#ffedd5" stroke="#f97316" stroke-width="1.3"/><line x1="125" y1="189" x2="125" y2="211" stroke="#f97316" stroke-width="0.8"/><line x1="160" y1="189" x2="160" y2="211" stroke="#f97316" stroke-width="0.8"/><line x1="195" y1="189" x2="195" y2="211" stroke="#f97316" stroke-width="0.8"/><line x1="230" y1="189" x2="230" y2="211" stroke="#f97316" stroke-width="0.8"/><line x1="290" y1="112" x2="350" y2="112" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="290" y1="156" x2="350" y2="156" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="5 3"/><line x1="290" y1="200" x2="350" y2="200" stroke="#16a34a" stroke-width="1.4" stroke-dasharray="5 3"/><text x="470" y="58" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">emoji_meta.json</text><rect x="350" y="72" width="240" height="156" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="372" y="117" text-anchor="start" font-size="13" fill="#18181b" font-family="'JetBrains Mono', monospace">101 → grinning</text><text x="372" y="161" text-anchor="start" font-size="13" fill="#18181b" font-family="'JetBrains Mono', monospace">102 → smiling</text><text x="372" y="205" text-anchor="start" font-size="13" fill="#18181b" font-family="'JetBrains Mono', monospace">103 → wink</text><text x="310" y="252" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">ID で結合 → セットで1つの検索DB</text></svg><figcaption><code>.faiss</code> は <b>IndexIDMap2(IndexFlatIP)</b> で「<b>正規化ベクトル＋自分で振った ID（100 番台）</b>」だけを覚えます。<b>ID→ラベル</b>の対応表は <code>emoji_meta.json</code> に分けて持ち、検索で返った ID をこの表で絵文字名に引き直します。<b>.faiss と .json はセットで1つの検索DB</b>——片方だけ更新すると ID から中身が引けなくなります。</figcaption></figure>

```python
# emoji_lab の中核（抜粋）
vn = l2_normalize(feats)                       # 各行ノルム1（コサイン用）
index = faiss.IndexIDMap2(faiss.IndexFlatIP(vn.shape[1]))
index.add_with_ids(vn, ids.astype(np.int64))   # 100番台の任意IDを付与
faiss.write_index(index, "emoji_index.faiss")  # ベクトル+IDを保存
meta_path.write_text(json.dumps(meta, ...))    # ID->ラベルは別ファイル
```

`01_build_emoji_index.py` を実行すると、`emoji_index.faiss` と `emoji_meta.json`、それに「色 vs グレースケール」を見比べられるギャラリー画像が `lectures/45_sketch_emoji_search/outputs/` に出力されます。`read_index` 後の `ntotal` が保存前と一致すれば永続化は成功で、アプリ再起動をまたいでも索引を引き継げます。

## 5. Tkinter での手書き入力（headless フォールバック）

手書き UI の実装が本章の新要素です。前述のとおり OpenCV は headless なのでマウスコールバックが使えません。そこで代わりに**標準ライブラリ `tkinter`** の `Canvas` を使い、`<B1-Motion>`（左ドラッグ）で直前の点から現在の点へ黒い線分を引きます。ここでの定石は、**Canvas（画面表示用）と PIL.Image（保存用）に「同時に」線を引く**ことです。Tk の Canvas からピクセルを直接吸い出す移植性の高い方法が無いので、保存の真実は別に持っている PIL 画像側に置くわけです。操作は最小限にとどめ、`c` キーで消去、`s` / Enter / Save ボタンで保存して終了、としています。

そして最大の山場が **display が無い環境での挙動**です。Docker・CI・X11 転送なしの SSH などでは、`tkinter.Tk()` の呼び出しが `TclError`（`no display name and no $DISPLAY environment variable`）で失敗します。さらに環境によっては `import tkinter` 自体が失敗することもあります。そこで `02_sketch_input_tk.py` は **`tkinter` の import からウィンドウ生成まで丸ごと `try/except` で囲み**、失敗したら**合成スケッチ（線画）に切り替えて保存**し、案内を出して `exit 0` で終わります。こうすることで、「ローカルでは本物の手書き、CI では合成」が透過的に切り替わります。

```python
try:
    raw = run_tk_canvas()                 # display 無しなら TclError
    source = "Tkinter キャンバスでの手書き入力"
except Exception as e:                    # import 不可 / TclError などを全捕捉
    print(f"[fallback] 手書きウィンドウを開けません（{type(e).__name__}）")
    raw = make_synthetic_sketch()         # 合成スケッチ（線画）で代替
    source = "合成スケッチ（display 無しのフォールバック）"
```

保存先は2か所あります。1つは `lectures/45_sketch_emoji_search/outputs/02_sketch.png`（成果物）、もう1つは `data/45_sketch_emoji_search/sketch.png`（`03` が既定で読みに行く受け渡し場所）です。ローカルで実際に絵を描いて試す手順は §「▶ 動かし方」に書きました。手元に GUI があるなら、ぜひ自分の落書きで検索してみてください。

## 6. スケッチの前処理（白背景・黒線・正方化）

手書き入力は素のままでは CLIP に入れられません。そこで `emoji_lab.preprocess_sketch()` が**単一責務の小さな手順の積み重ね**で標準形へ整えます。具体的には、**(1)** グレースケール化、**(2)** 背景が暗ければ反転して「白地に黒線」へ統一（暗背景に白チョークで描いた画像も救済）、**(3)** 長辺基準で**正方化**（白パディングで縦横比を保つ）、**(4)** `CANVAS`（224px）へリサイズして 3ch グレースケール RGB で返す、という流れです。これにより、絵文字側（グレースケール）とクエリ側（線画）の前処理が**同じ終点**に揃います。

このうち正方化を入れる理由は、CLIP の前処理が内部で正方リサイズ（短辺合わせ＋センタークロップ等）を行うため、**縦長・横長のまま渡すと中心部が切れて線が欠ける**からです。先に白背景で正方パディングしておけば、描いた線を切らずに収められます。また背景の明暗判定（平均輝度が閾値より暗ければ反転）は、入力経路によって極性が揺れても**常に白背景・黒線に正規化する**ための保険です。クエリと DB で前処理がズレると検索が崩れるので、「同じ関数に通す」ことをここでも徹底します。

<figure class="lec-fig"><svg viewBox="0 0 600 290" role="img" aria-label="縦長の線画スケッチを白い余白で正方1対1に詰めてからCLIPへ渡すことで、中央クロップで線が切れるのを防ぐ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="28" text-anchor="middle" font-size="14" fill="#3f3f46">正方化：白で詰めて 1:1 にしてから CLIP へ</text><rect x="70" y="66" width="80" height="160" fill="#ffffff" stroke="#71717a" stroke-width="1.8"/><circle cx="95" cy="120" r="4.5" fill="#18181b"/><circle cx="125" cy="120" r="4.5" fill="#18181b"/><path d="M92 150 Q110 168 128 150" fill="none" stroke="#18181b" stroke-width="2.5"/><text x="110" y="246" text-anchor="middle" font-size="12.5" fill="#52525b">縦長の線画</text><line x1="160" y1="146" x2="242" y2="146" stroke="#c2410c" stroke-width="2.5"/><polygon points="250,146 240,141 240,151" fill="#c2410c"/><text x="201" y="136" text-anchor="middle" font-size="12" fill="#c2410c">白パディング</text><rect x="290" y="66" width="160" height="160" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><rect x="292" y="68" width="38" height="156" fill="#f4f4f5"/><rect x="410" y="68" width="38" height="156" fill="#f4f4f5"/><circle cx="355" cy="120" r="4.5" fill="#18181b"/><circle cx="385" cy="120" r="4.5" fill="#18181b"/><path d="M352 150 Q370 168 388 150" fill="none" stroke="#18181b" stroke-width="2.5"/><text x="311" y="146" text-anchor="middle" font-size="11" fill="#71717a" style="writing-mode:vertical-rl;text-orientation:upright">余白</text><text x="370" y="246" text-anchor="middle" font-size="12.5" fill="#15803d">正方 1:1（線を保つ）</text></svg><figcaption>手書き入力は <b>グレースケール化 → 暗背景なら反転（白地・黒線へ統一）→ 正方化 → 224px リサイズ</b>の順で整えます。山場は<b>正方化</b>です。CLIP は内部で<b>正方リサイズ＋センタークロップ</b>をするため、縦長のまま渡すと中心以外が切れて線が欠けます。先に<b>白い余白で 1:1 に詰める</b>と、描いた線を切らずに収められます。</figcaption></figure>

## 7. 検索と上位 N 件の可視化

検索フェーズ（`03_search_topn.py`）は短く済みます。`01` が保存した索引とメタを `load_index` で読み戻し、クエリのスケッチ（`data/.../sketch.png` があればそれ、無ければ合成）を `preprocess_sketch` → `EmojiEmbedder.encode_images` → `l2_normalize` → `index.search(xq, N)` と流すだけです。あとは返ってきた ID をメタの辞書で**ラベルに変換**し、ID から位置を引いて**絵文字サムネ**を取り出し、左にクエリ・右に上位 N 件をスコア（コサイン類似度）付きで並べたパネル画像を保存します。

実装で気を配るのは2点です。1つは**次元の整合**で、クエリと索引の埋め込み次元が食い違うと `search` が落ちます。そこで `03` は `index.d == embedder.dim` を確認し、もし不一致（例: 索引は CLIP の 512 次元だが、いまはオフラインで画素記述子 576 次元になっている）なら、**その場で現在のエンコーダで索引を組み直して** `exit 0` を守ります。もう1つは **`-1` ガード**です。`N > ntotal` などで近傍が足りないと `I` に `-1` が混じり、それをそのまま辞書参照するとクラッシュするので、`i < 0` を早期に弾いてから引きます。

実際に「笑顔の落書き」を入れると、グレースケール絵文字の中から `slight_smile / smiling / grinning / smiling_eyes …` のような**笑い系の顔がコサイン 0.88 前後で並ぶ**のが見られます（フォント絵文字 26 種・CLIP の場合）。スケッチが多少ラフでも CLIP が「笑った顔」という意味で寄せてくれるのを体感できるはずです。検算のコツは第17回と同じで、**DB の絵文字そのものをクエリにしたら、それが top-1・スコア≈1.0** になるかを見ます。ならなければ、正規化漏れや前処理の不一致を疑ってください。

## 8. ドメインギャップとその緩和（実測と、少し意外な結論）

`04_eval_domaingap.py` は本章のハイライトです。ここでは**正解がはっきりする合成ペア**——表情ごとに「合成絵文字（塗り）」と「合成スケッチ（線画）」を独立に描いたもの——を使い、絵文字に `style ∈ {color, gray, edge, invert}` を施して索引を作り、スケッチをクエリに**Recall@N（= top-N ヒット率）**を測ります。各表情の正解絵文字は1個なので、「正しい表情が上位 N 件に入った割合」がそのまま Recall@N になります。さらに**2つのエンコーダ**——CLIP（意味ベース）と画素記述子（低レベル）——を並べて比べます。

実測の要点（16表情・小サンプルなので大局を読む）:

| style | CLIP R@1 | CLIP R@5 | 画素記述子 R@1 | 画素記述子 R@5 |
| --- | --- | --- | --- | --- |
| color（塗り） | 0.81 | 1.00 | 1.00 | 1.00 |
| gray（グレースケール） | 0.69 | 1.00 | 1.00 | 1.00 |
| edge（エッジ＝線画化） | 0.31 | 0.88 | 0.62 | 0.69 |
| invert（明暗反転） | 0.56 | 1.00 | **0.00** | **0.12** |

ここから読み取れることは、当初の素朴な期待——「線画に寄せる（edge/gray）ほど当たる」——とは**違います**。まず**画素記述子（低レベル）は様式に脆く**、`invert`（白黒の極性反転）で Recall がほぼ 0 まで崩壊します。低レベル照合では「黒線か白線か」という極性が決定的だからです。エッジ化も、平面的な絵文字では**二重輪郭などのノイズ**を生んで必ずしも得になりません。一方 `gray` は `color` と同点で、**グレースケール化は「色という無関係な手がかりを落とすだけ」で安全**、という本章の主張がここで裏づけられます。

そして最も重要なのが **CLIP の頑健さ**です。CLIP は様式を変えても Recall の振れ幅が小さく、極性を反転した `invert` でも R@5＝1.00 を保ちます（画素記述子は 0.12）。これは「**強い共有埋め込みは、ドメインギャップを意味の力で大きく橋渡しする**」ことを意味します。古典 SBIR が前処理でドメインを必死に寄せていたのに対し、CLIP では「寄せる」より「**良い埋め込みを使う**」方がよほど効く——これが現代的な結論です。`04` は `04_domaingap_recall.png`（エンコーダ×style の棒グラフ）と `04_styles_example.png`（同じ絵文字の4様式）を保存するので、数字と見た目の両方で確かめてください。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="style別Recall@5の棒グラフ。CLIPはinvertでも1.00を保つが、画素記述子はinvertで0.12まで崩壊する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">style 別 Recall@5（高いほど良い）</text><rect x="120" y="40" width="14" height="14" fill="#ea580c"/><text x="140" y="52" text-anchor="start" font-size="12" fill="#3f3f46">CLIP</text><rect x="200" y="40" width="14" height="14" fill="#2563eb"/><text x="220" y="52" text-anchor="start" font-size="12" fill="#3f3f46">画素記述子</text><line x1="68" y1="58" x2="68" y2="252" stroke="#71717a" stroke-width="1.5"/><line x1="68" y1="252" x2="608" y2="252" stroke="#71717a" stroke-width="1.5"/><line x1="68" y1="60" x2="608" y2="60" stroke="#e4e4e7" stroke-width="1"/><line x1="68" y1="156" x2="608" y2="156" stroke="#f4f4f5" stroke-width="1"/><text x="60" y="64" text-anchor="end" font-size="11" fill="#52525b">1.0</text><text x="60" y="256" text-anchor="end" font-size="11" fill="#52525b">0</text><rect x="118" y="60" width="30" height="192" fill="#ea580c"/><rect x="152" y="60" width="30" height="192" fill="#2563eb"/><rect x="238" y="60" width="30" height="192" fill="#ea580c"/><rect x="272" y="60" width="30" height="192" fill="#2563eb"/><rect x="358" y="83" width="30" height="169" fill="#ea580c"/><rect x="392" y="119" width="30" height="133" fill="#2563eb"/><rect x="478" y="60" width="30" height="192" fill="#ea580c"/><rect x="512" y="229" width="30" height="23" fill="#2563eb"/><text x="150" y="270" text-anchor="middle" font-size="11.5" fill="#3f3f46">color</text><text x="270" y="270" text-anchor="middle" font-size="11.5" fill="#3f3f46">gray</text><text x="390" y="270" text-anchor="middle" font-size="11.5" fill="#3f3f46">edge</text><text x="510" y="270" text-anchor="middle" font-size="11.5" fill="#3f3f46">invert</text><text x="527" y="223" text-anchor="middle" font-size="11" font-weight="700" fill="#dc2626">0.12</text></svg><figcaption>同じ実験を <b>CLIP</b>（意味ベース）と<b>画素記述子</b>（低レベル）で比べた <b>Recall@5</b> です。<b>color と gray はほぼ互角</b>——グレースケール化は色という無関係な手がかりを落とす<b>安全策</b>にすぎません。決定的なのは <code>invert</code>（明暗反転）で、<b>画素記述子は 0.12 まで崩壊</b>するのに <b>CLIP は 1.00 を保ちます</b>。強い共有埋め込みほど様式のギャップに頑健、という本章の結論が一目で読み取れます。</figcaption></figure>

---

## 🛠 章末ミニプロジェクト — 手書き絵文字検索アプリ

`mini_project.py` は、ここまでの要素を**1本に統合した完成形**です。実行すると次が一気通貫で走ります。**(1)** 絵文字コレクション（data→フォント→合成）を用意 → グレースケール化 → CLIP 埋め込み → `IndexIDMap2` に `add` → `.faiss`/`.json` をセット永続化。**(2)** スケッチ入力（**display があれば Tkinter キャンバスで手書き、無ければ** `data/.../sketch.png` か合成スケッチに自動フォールバック）。**(3)** スケッチを同じ CLIP で埋め込み → `index.search` → 上位 N 件。**(4)** 結果を `mini_project_result.png`（左:クエリ／右:上位 N 件の絵文字＋スコア）と `mini_project_report.json`（バックエンド・件数・各ヒットの id/label/cosine）に出力。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="ミニプロジェクトの実行順序。絵文字を索引化、スケッチ入力(手書きか合成)、同じCLIPで検索、結果をPNGとJSONに出力の4ステップが左から右へ流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — ① から ④ を一気通貫で実行</text><rect x="16" y="92" width="140" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="178" y="92" width="140" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="340" y="92" width="140" height="70" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="502" y="92" width="140" height="70" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="86" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① 索引化</text><text x="86" y="144" text-anchor="middle" font-size="11" fill="#71717a">絵文字 → .faiss/.json</text><text x="248" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">② スケッチ入力</text><text x="248" y="144" text-anchor="middle" font-size="11" fill="#71717a">手書き or 合成</text><text x="410" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">③ CLIP で検索</text><text x="410" y="144" text-anchor="middle" font-size="11" fill="#71717a">index.search</text><text x="572" y="122" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">④ 結果を出力</text><text x="572" y="144" text-anchor="middle" font-size="11" fill="#71717a">PNG + JSON</text><line x1="156" y1="127" x2="172" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="178,127 168,122 168,132" fill="#71717a"/><line x1="318" y1="127" x2="334" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="340,127 330,122 330,132" fill="#71717a"/><line x1="480" y1="127" x2="496" y2="127" stroke="#71717a" stroke-width="2"/><polygon points="502,127 492,122 492,132" fill="#71717a"/><line x1="248" y1="162" x2="248" y2="206" stroke="#2563eb" stroke-width="1.8" stroke-dasharray="5 3"/><polygon points="248,212 243,202 253,202" fill="#2563eb"/><rect x="178" y="212" width="140" height="46" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/><text x="248" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">合成スケッチ</text><text x="248" y="249" text-anchor="middle" font-size="10.5" fill="#52525b">display 無し時に自動切替</text><text x="330" y="282" text-anchor="middle" font-size="12" fill="#52525b">二重フォールバック（画素記述子・合成）で必ず exit 0</text></svg><figcaption><code>mini_project.py</code> は <b>① 絵文字を索引化（.faiss/.json）→ ② スケッチ入力 → ③ 同じ CLIP で検索 → ④ 結果を出力（PNG＋JSON）</b>の順に一気通貫で走ります。山場は <b>②</b> で、<b>display があれば Tkinter キャンバスで手書き</b>、無ければ <b>合成スケッチへ自動フォールバック</b>します。さらに CLIP が取れなければ画素記述子へ切り替わり、<b>二重のフォールバック</b>で必ず <code>exit 0</code> で完走します。</figcaption></figure>

すべて CPU で数十秒以内に完走し、CLIP が取れない環境では画素記述子へ、display が無い環境では合成スケッチへ、と**二重のフォールバック**で必ず `exit 0` になります。出力先は `lectures/45_sketch_emoji_search/outputs/` で、`mini_project_result.png`・`mini_project_report.json`・`mini_emoji_index.faiss`＋`mini_emoji_meta.json`・`mini_sketch.png` が並びます。

```bash
uv run python lectures/45_sketch_emoji_search/mini_project.py
```

**腕試し（発展課題）。** 余力があれば: (1) `data/45_sketch_emoji_search/emoji/*.png` に好きな絵文字を置いて差し替える（自動で最優先採用）。(2) 索引を `IndexIDMap2(IndexHNSWFlat(...))` に替え、絵文字数を数千に増やしても速いことを確かめる（第17回 ANN の応用）。(3) スコアにしきい値（コサイン < 0.2 は「該当なし」）を入れて“見つからない”を表現する。(4) `04` の style 比較を自分の手書きスケッチで回し、`gray` と `color` で結果がどう変わるか観察する。(5) スケッチ側にもエッジ強調やランダムな線の揺らぎ（データ拡張）を足し、頑健性が上がるか試す。

## ✅ 到達チェックリスト

この章を「マスターした」と言えるのは、次を**AI 補助なしで**できるときです。手を動かして1つずつ潰してください。

- [ ] SBIR を「クエリと検索対象が様式の異なる画像である画像検索」と説明でき、その全体像（準備→検索）を描ける。
- [ ] 絵文字もスケッチも**同じ CLIP 画像エンコーダ・同じ正規化**に通す必要があることを説明できる。
- [ ] `get_image_features` が v5 では**オブジェクトを返し `.pooler_output` が未正規化**である落とし穴を回避できる。
- [ ] **L2 正規化 → `IndexFlatIP`** でコサイン検索になること、DB 側・クエリ側の**両方**を正規化することを説明できる。
- [ ] `IndexIDMap2` ＋ `add_with_ids` で任意 ID を付け、**ID→ラベルは別管理（JSON）**して引く設計を書ける。
- [ ] `.faiss` と メタ（JSON）を**セットで**永続化・整合させる必要性を説明できる。
- [ ] **なぜ絵文字をグレースケール化するか**（色という無関係な手がかりを落とす安全策）を自分の言葉で言える。
- [ ] 手書き UI を **Tkinter キャンバス**で書け、`cv2.imshow` が headless で使えない理由を説明できる。
- [ ] **display 無しで Tkinter が `TclError` で落ちる**ことを `try/except` で捕まえ、合成スケッチへフォールバックして `exit 0` にできる。
- [ ] スケッチ前処理（白背景・黒線・**正方化**・グレースケール）を、なぜ正方化するかまで含めて説明できる。
- [ ] `search` 結果 `I` の **`-1` をガード**し、**次元不一致**を `assert`/組み直しで防げる。
- [ ] **Recall@N（top-N ヒット率）**を自前計算でき、合成ペアで正解ラベルを作る意味を説明できる。
- [ ] ドメインギャップ実験から「**強い埋め込みほど様式に頑健**」「グレースケールは安全策」「**invert は低レベル照合を壊す**」を読み取れる。
- [ ] 演習 `exercises.py` を**全8問 PASS**できる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/45_sketch_emoji_search/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 任意の配列を FAISS 仕様（`float32`・C連続）へ変換して返す（`ex1_to_faiss_array` の TODO）。
2. 2次元配列の各行を L2 ノルム1に正規化して返す。0除算を避けるためノルムを下限クリップする（`ex2_l2_normalize_rows` の TODO）。
3. PIL 画像をグレースケール化し、3ch の RGB（各画素 R==G==B）に戻して返す（`ex3_grayscale_rgb` の TODO）。
4. 2次元配列を長辺基準で正方形に中央パディングして返す。元の領域を中央に置き、余白は `fill` で埋める（`ex4_square_pad` の TODO）。
5. 正規化 + `IndexFlatIP` + `IndexIDMap` で任意IDのコサイン検索を行い、近傍の ID 行列を返す。`xb` と `xq` は同じ正規化に通す（`ex5_cosine_search_ids` の TODO）。
6. 検索結果の ID 列をラベルへ変換する。`-1` は `(none)`、未登録は `(missing)` にガードする（`ex6_guard_lookup` の TODO）。
7. 類似度降順に並んだラベル列で、正しいラベルが上位 N 件に入っているか（top-N ヒット）を判定する（`ex7_topn_hit` の TODO）。
8. 複数クエリの Recall@N（= top-N ヒット率）を平均して返す。`pairs` が空なら `0.0`（`ex8_recall_at_n` の TODO）。

## ❓ よくある落とし穴・FAQ・デバッグ

**Q. `cv2.imshow` でスケッチを描かせたい。** 本講座の OpenCV は **headless ビルド**で `imshow` / `namedWindow` / `setMouseCallback` を**持ちません**（呼ぶと `cv2.error` か `AttributeError`）。手書き UI は **`tkinter.Canvas`** で作ります。OpenCV は色変換（`cvtColor`）やエッジ抽出（`Canny`）など**変換用途だけ**に使ってください。

**Q. Docker/CI で `_tkinter.TclError: no display name and no $DISPLAY` が出る。** display が無い環境では Tk のウィンドウを作れません。これは**異常ではなく想定内**で、`02`/`mini_project` は `try/except` で捕まえて**合成スケッチへフォールバック**し `exit 0` で完走します。ローカルで本物の手書きを試したいときだけ display のある環境で実行してください（§▶動かし方）。

**Q. `import tkinter` 自体が `ModuleNotFoundError`。** 一部の最小構成 Python には Tk が同梱されていません（Linux なら `apt-get install python3-tk`、pyenv 等でビルドした Python は Tk ヘッダ入りで入れ直し）。本章は import からガードしているので、入っていなくても合成スケッチで完走します。手書きを使いたい場合だけ導入してください。

**Q. 絵文字フォントで `OSError: invalid pixel size`。** Noto Color Emoji など**カラー絵文字フォントは特定サイズ（例 109px）でしか開けません**。`emoji_lab._render_emoji_glyph` は複数サイズを順に試し、開けたサイズで描いてからリサイズします。任意サイズ決め打ちは禁物です。

**Q. 絵文字が真っ白／豆腐（□）になる。** そのフォントに当該グリフが無いか、`embedded_color=True` が効いていません。本章は「非白画素が少なければ描けていない」とみなしてスキップし、最終的に**合成絵文字**へフォールバックします。フォント自体が見つからない環境でも `build_emoji_collection()` は合成絵文字で必ず集合を返します。

**Q. グレースケールにしたら逆に当たりが悪くなった。** 起こり得ます。本章の実測でも CLIP では `color` が `gray` をわずかに上回りました（§8）。グレースケール化は「当たりを上げる魔法」ではなく「**色という無関係な手がかりを落とす安全策**」。効果はエンコーダ次第で、**強い CLIP では差は小さい**と理解してください。劇的に効く/壊れるのは低レベル記述子のときです（`invert` で Recall≈0）。

**Q. `edge`（エッジ化）にすれば線画どうしで当たるはずでは？** 写真のように**テクスチャ豊かな対象**ならエッジ化はドメインを大きく寄せて効きます。しかし**平面的な絵文字**では Canny が二重輪郭などのノイズを生み、むしろ Recall が下がることがあります（§8 の表）。前処理は対象の性質に依存する、というのが教訓です。

**Q. `search` が落ちる／結果がデタラメ。** まず `index.d` と `xq.shape[1]`（埋め込み次元）が一致しているか。CLIP（512）で作った索引を、オフラインで画素記述子（576）にクエリすると落ちます。`03` は次元不一致を検出して索引を組み直します。次に配列が `float32`・C連続か（`as_faiss_array`）、DB とクエリの**両方**を正規化したか、を確認します。

**Q. 検索結果のメタ参照でクラッシュ。** `N > ntotal` などで `I` に `-1` が混じります。`i < 0` を早期に弾いて `(none)` 等に置換してから JSON/辞書を引いてください（`-1` をそのまま参照しない）。

**Q. `'...Pooling' object has no attribute ...`（CLIP）。** transformers v5 で `get_image_features` の戻り値が `BaseModelOutputWithPooling` になりました。射影ベクトルは `.pooler_output` にあり**未正規化**。`out.pooler_output` を取り出して正規化してから検索します。

**Q. 図のタイトルが文字化け／警告が出る。** matplotlib 既定フォントに日本語が無いためです。**図タイトル・凡例は ASCII** にし、日本語はコンソール出力に回しています（本章のヘルパもそうしています）。`matplotlib.use("Agg")` で headless 描画。

**デバッグの定石。** ① `index.d == embedder.dim` を最初に確認。② 「絵文字そのものをクエリにして自分が top-1・スコア≈1.0」で配線確認。③ `index.ntotal` を `add`/`read_index` 前後で突き合わせ。④ スケッチが白背景・黒線・正方になっているか保存画像で目視。⑤ Recall が変なら、合成ペアの正解ラベルと検索結果ラベルを並べて print。

## 🚀 発展トピック・参考

本章で SBIR の骨格は身についたので、ここから先は精度と適用範囲を広げるための入り口を示します。

- **専用 SBIR モデル**: 本章は汎用 CLIP の画像エンコーダを流用しましたが、スケッチと写真を**別ブランチ**で埋め込み、対照学習で揃える専用アーキテクチャ（Sketch-a-Net、Deep SBIR の triplet/共有空間、**ZSE-SBIR** のゼロショット拡張）は、ラフな手描きに対する頑健性が桁違いです。CLIP をスケッチ・写真ペアで fine-tune するだけでも当たりが大きく改善します。
- **エッジ／線画化の前処理**: 検索対象が**写真**のときは、HED や `cv2.Canny`、近年の学習ベースのエッジ検出で写真を線画化してからスケッチと照合すると効きます（古典 SBIR の王道）。本章で見たとおり、平面的なイラストでは効果が薄い点に注意。
- **スケッチ拡張（data augmentation）**: 手描きは人によってブレます。線の太さ・かすれ・ランダムな揺らぎ・部分欠落・回転を加えてクエリ/学習を水増しすると、実利用での頑健性が上がります。RDP 等で**ストロークを単純化**して様式を揃えるのも有効。
- **規模を上げる（ANN）**: 絵文字が数万種を超えたら、`IndexFlatIP` から `IndexHNSWFlat` / `IndexIVFFlat` へ。第17回の `efSearch`/`nprobe` スイープと Recall@k・QPS 評価がそのまま使えます。GPU なら `faiss.index_cpu_to_gpu` の一行で載せ替え。
- **マルチモーダル化**: CLIP は画像とテキストを共有空間に置くので、**スケッチ＋テキスト**（「笑っている」＋落書き）でクエリベクトルを合成して絞り込む、テキストだけで絵文字検索する、といった拡張が自然にできます（第16・42回）。
- **評価をきちんと**: 本章の Recall@N は小サンプルの目安です。`torchmetrics.retrieval`（`RetrievalRecall`/`RetrievalMAP`/`RetrievalMRR`）で mAP・MRR・nDCG まで測り、ユーザの主観評価（上位 N に納得の絵文字が来るか）と突き合わせます。
- **公式リファレンス**: FAISS Wiki（<https://github.com/facebookresearch/faiss/wiki>）、transformers/CLIP（<https://huggingface.co/docs/transformers>）、SBIR サーベイ（"Deep Learning for Sketch-Based Image Retrieval"）。索引選択は Wiki の "Guidelines to choose an index" が決定版です。

## ▶ 動かし方

プロジェクトルートから順に実行します（初回は CLIP の重みダウンロードが走り、以降キャッシュ）。

```bash
# 依存の用意（初回のみ）
uv sync --group dl --group hf --group vector

# 1) 絵文字コレクションを索引化（.faiss + .json + ギャラリー画像）
uv run python lectures/45_sketch_emoji_search/01_build_emoji_index.py

# 2) 手書きスケッチ入力（ローカルGUIなら手書き、headlessは合成へ自動切替）
uv run python lectures/45_sketch_emoji_search/02_sketch_input_tk.py

# 3) スケッチから類似 絵文字 上位N件を検索（パネル画像を保存）
uv run python lectures/45_sketch_emoji_search/03_search_topn.py

# 4) ドメインギャップ評価（style別 Recall@N の棒グラフ）
uv run python lectures/45_sketch_emoji_search/04_eval_domaingap.py

# 5) 章末ミニプロジェクト（エンドツーエンドの手書き絵文字検索アプリ）
uv run python lectures/45_sketch_emoji_search/mini_project.py

# 演習（自己採点）と模範解答
uv run python lectures/45_sketch_emoji_search/exercises.py
uv run python lectures/45_sketch_emoji_search/exercises_solutions.py
```

**ローカルで本物の手書きを試す手順。** ① **display のある環境**で実行すること（Linux デスクトップ／macOS／Windows のローカル端末。SSH なら X11 転送 `ssh -X`、WSL2 なら WSLg）。② Python に **Tk が入っている**こと（`python -c "import tkinter"` が通る。Linux は `sudo apt-get install python3-tk`）。③ `02_sketch_input_tk.py` を実行するとキャンバスが開くので、**マウス左ドラッグで絵文字の顔を描く**（笑顔・困り顔など）。**`c` で消去**、**`s` または Enter で保存して終了**。④ 保存されたスケッチ（`data/45_sketch_emoji_search/sketch.png`）を使って **`03_search_topn.py`** を実行すると、あなたの落書きに似た絵文字が上位 N 件で返ります。`mini_project.py` なら ②〜④ が1コマンドで通ります。display が無い環境（Docker/CI）では、これらは自動で合成スケッチに切り替わり、同じく `exit 0` で完走します。

なお、自分の絵文字セットで試したいときは、`data/45_sketch_emoji_search/emoji/` に PNG を置いてから `01`（または `mini_project.py`）を実行すると、その画像が最優先で索引化されます。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ **faiss-cpu 1.14.2** ／ **torch 2.12.0+cpu** ／ torchvision 0.27.0+cpu ／ **transformers 5.11.0** ／ opencv-python-headless 4.13 ／ Pillow 12 系 ／ scikit-learn 1.9.0（本章コードでは未使用）／ matplotlib 3.10 系 ／ tkinter（標準ライブラリ, Tk 8.6）。
> モデル: `openai/clip-vit-base-patch32`（初回のみ HuggingFace Hub から重みDL→ローカルキャッシュ）。描画ウィンドウは **Tkinter**（OpenCV は headless のため `imshow` 不可）。display 無し環境では合成スケッチへ自動フォールバックして `exit 0`。GPU 版 FAISS は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA, CUDA 12.4 系限定、`faiss-cpu` と排他）。
