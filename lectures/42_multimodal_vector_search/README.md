# 42_multimodal_vector_search: マルチモーダル・ベクトル検索（FAISS）— 画像・テキスト・クロスモーダル

> トラック: **埋め込み・検索** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `vector` `metrics`
> 前提モジュール: `16_clip_zeroshot_retrieval`, `17_faiss_image_search`

---

## 🎯 この章のゴール

CLIP/SigLIP で **画像とテキストを同じ埋め込み空間へ写し**、FAISS で
**画像→画像 / テキスト→画像 / テキスト→テキスト / クロスモーダル** の4方向検索を
**1つの統一インターフェース**で構築できるようになる。具体的には次を自力で書ける状態を目指す。

- CLIP の `get_image_features` / `get_text_features` で 512 次元の共有ベクトルを得る（transformers v5 では `.pooler_output`）。
- `faiss.normalize_L2` + `IndexFlatIP` で **コサイン類似度検索**にする理屈と手順。
- `IndexIDMap` + `add_with_ids` で **任意ID** を割り当て、`id ↔ メタデータ`（modality / 画像パス / 説明文）を別管理する実運用パターン。
- `write_index` / `read_index` + メタJSON で **永続化** し、別プロセスで読み戻す。
- `Flat / IVF / HNSW` を **同じ search API** のまま使い分け、**Recall@k / mAP** で定量評価する。
- 音声（CLAP）など**新モダリティを後から足す設計**を理解する。

本章は CPU のみで完結する。ネットへ出るのは **CLIP の重み DL の初回だけ**で、入力画像・テキストはすべて合成生成する。

---

## 1. 直感 — なぜ「同じ空間」なら何でも引けるのか

ふつうの画像検索（17回）は「画像 → ベクトル → 近いベクトルを探す」だった。ここで効いていたのは、
**似たものが近くに来るようにベクトルが配置されている**という性質ひとつだけだ。では、もし
**画像のベクトルとテキストのベクトルが同じ部屋（空間）に置かれていて、内容が対応するものほど近い**
としたらどうだろう。「赤い円の画像」と「a photo of a red circle という文」がほぼ同じ場所に立つなら、
**文をベクトルにして画像の山に投げ込めば、対応する画像が最近傍として返ってくる**。これがクロスモーダル検索の正体だ。

CLIP（Contrastive Language–Image Pre-training）はまさにこれを実現するモデルである。画像用とテキスト用の
2つのエンコーダを持ち、**対応する画像と文のペアを近く、無関係なペアを遠く**なるように同時学習してある。
結果として、画像もテキストも **同じ 512 次元空間** に写り、`cos(画像, 文)` が「その文が画像をどれだけ説明しているか」になる。
本章の `01_shared_space_clip.py` では、合成した「色×形」の画像と説明文を CLIP に通し、`text → image` の
最近傍が 12/12 で正しく一致することを確認する。ここが全ての出発点だ。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="画像エンコーダとテキストエンコーダが対応する画像と文を同じ512次元の共有空間の近い位置へ写す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><circle cx="52" cy="72" r="22" fill="#dc2626"/><text x="52" y="112" text-anchor="middle" font-size="12" fill="#52525b">画像</text><rect x="14" y="190" width="104" height="38" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><text x="66" y="214" text-anchor="middle" font-size="13" fill="#c2410c">a red circle</text><line x1="78" y1="72" x2="144" y2="72" stroke="#71717a" stroke-width="1.8"/><polygon points="150,72 140,67 140,77" fill="#71717a"/><line x1="120" y1="209" x2="144" y2="209" stroke="#71717a" stroke-width="1.8"/><polygon points="150,209 140,204 140,214" fill="#71717a"/><rect x="150" y="50" width="116" height="44" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/><text x="208" y="77" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">画像エンコーダ</text><rect x="150" y="186" width="116" height="44" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.8"/><text x="208" y="213" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">テキストエンコーダ</text><rect x="330" y="40" width="312" height="232" rx="12" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="486" y="62" text-anchor="middle" font-size="13" fill="#3f3f46">共有埋め込み空間（512次元）</text><line x1="266" y1="74" x2="456" y2="143" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="5 3"/><polygon points="462,146 451,141 452,152" fill="#2563eb"/><line x1="266" y1="206" x2="486" y2="164" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 3"/><polygon points="492,162 481,159 482,170" fill="#ea580c"/><rect x="372" y="96" width="14" height="14" fill="#2563eb"/><rect x="556" y="208" width="14" height="14" fill="#2563eb"/><circle cx="582" cy="110" r="7" fill="#ea580c"/><circle cx="388" cy="232" r="7" fill="#ea580c"/><ellipse cx="482" cy="150" rx="40" ry="30" fill="none" stroke="#c2410c" stroke-width="2" stroke-dasharray="4 3"/><rect x="460" y="136" width="18" height="18" fill="#2563eb"/><circle cx="498" cy="158" r="9" fill="#ea580c"/><text x="482" y="214" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">対応ペアは近い</text></svg><figcaption><b>CLIP</b> は画像用とテキスト用の2つのエンコーダを持ち、対応する画像と文を<b>同じ512次元空間の近い位置</b>へ写します。図の<b>青い ■ が画像ベクトル</b>、<b>橙の ● がテキストベクトル</b>で、内容が対応するペアほどコサインが大きく（＝距離が近く）なります。だから文をベクトル化して画像の山へ投げれば、対応する画像が<b>最近傍</b>として返ってきます。</figcaption></figure>

そして検索エンジン側（FAISS）から見れば、ベクトルが画像由来かテキスト由来かは**どうでもいい**。次元さえ揃っていれば、
両方を同じ index に放り込み、どんなモダリティのクエリでも同じ `search()` で引ける。だから本章の設計は
「モダリティごとに別エンジンを作る」のではなく、**1つの index に全部載せ、必要に応じて結果側のモダリティで絞る**という
シンプルな統一形になる。これが `03_crossmodal_search.py` と `mini_project.py` の核である。

---

## 2. 理論 — 対照学習・コサイン類似度・なぜ L2 正規化するのか

**対照学習（contrastive learning）** の気持ちはこうだ。N 枚の画像と、それぞれに対応する N 本の文があるとき、
画像 i とその正しい文 i のコサイン類似度を上げ、画像 i と他の文 j(≠i) のコサイン類似度を下げる。
これを温度付き softmax の交差エントロピー（InfoNCE）で最適化すると、対応ペアが近づき、無関係ペアが押し合って
**意味的に整列した共有空間**ができる。CLIP は softmax 損失、SigLIP は各ペア独立の sigmoid 損失という違いこそあるが、
「同じ空間にマップして内積で測る」という点は共通であり、検索での使い方も変わらない。

検索で使う距離は **コサイン類似度** `cos(a,b) = (a·b)/(‖a‖‖b‖)`。ベクトルの「向き」だけを見て「長さ」を無視するのがミソだ。
というのも、CLIP の生の出力ベクトルは長さ（ノルム）がバラバラ（本章の実測で約 11 前後）なので、長さに引っ張られると検索そのものが壊れてしまうからだ。
そこで **各ベクトルを L2 正規化して ‖x‖=1 にしてしまえば、内積 a·b がそのままコサイン類似度になる**。
FAISS には「コサイン index」が存在しないが、この性質を使い **「L2 正規化 → 内積 index（IndexFlatIP）」でコサイン検索を再現する**のが定石だ。

<figure class="lec-fig"><svg viewBox="0 0 640 285" role="img" aria-label="L2正規化でノルムがばらばらのベクトルを単位円上に揃えると内積がコサイン類似度になる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="20" y="44" width="244" height="222" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="142" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">L2 正規化 前</text><line x1="72" y1="236" x2="248" y2="236" stroke="#e4e4e7" stroke-width="1.5"/><line x1="72" y1="236" x2="72" y2="84" stroke="#e4e4e7" stroke-width="1.5"/><line x1="72" y1="236" x2="222" y2="106" stroke="#2563eb" stroke-width="2.5"/><polygon points="222,106 216,118 210,110" fill="#2563eb"/><line x1="72" y1="236" x2="178" y2="170" stroke="#ea580c" stroke-width="2.5"/><polygon points="178,170 171,180 165,172" fill="#ea580c"/><text x="142" y="256" text-anchor="middle" font-size="11.5" fill="#52525b">‖x‖ ≈ 11（長さバラバラ）</text><line x1="266" y1="155" x2="290" y2="155" stroke="#c2410c" stroke-width="2.5"/><polygon points="296,155 286,150 286,160" fill="#c2410c"/><rect x="300" y="44" width="320" height="222" rx="10" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><text x="460" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">L2 正規化 後（‖x‖=1）</text><line x1="360" y1="238" x2="522" y2="238" stroke="#e4e4e7" stroke-width="1.5"/><line x1="360" y1="238" x2="360" y2="82" stroke="#e4e4e7" stroke-width="1.5"/><path d="M 510 238 A 150 150 0 0 0 360 88" fill="none" stroke="#d4d4d8" stroke-width="1.5"/><line x1="360" y1="238" x2="430" y2="106" stroke="#2563eb" stroke-width="2.5"/><polygon points="430,106 428,119 420,115" fill="#2563eb"/><line x1="360" y1="238" x2="492" y2="168" stroke="#ea580c" stroke-width="2.5"/><polygon points="492,168 483,178 479,170" fill="#ea580c"/><text x="438" y="104" font-size="14" font-weight="700" fill="#1d4ed8">a</text><text x="502" y="170" font-size="14" font-weight="700" fill="#c2410c">b</text><path d="M 397 218 A 42 42 0 0 0 380 201" fill="none" stroke="#16a34a" stroke-width="2"/><text x="404" y="200" font-size="15" font-weight="700" fill="#15803d">θ</text><text x="556" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#16a34a">a·b = cos θ</text></svg><figcaption>CLIP の生ベクトルは<b>長さ（ノルム）がバラバラ</b>（左、実測で約 11）。各ベクトルを <b>L2 正規化</b>して長さを <b>‖x‖=1</b>（単位円上）に揃えると（右）、<b>内積 a·b がそのままコサイン類似度 cos θ</b> になります。だから「L2 正規化 → <code>IndexFlatIP</code>（内積）」でコサイン検索を再現できます。<b>DB 側もクエリ側も同じ前処理で正規化</b>するのが鉄則です。</figcaption></figure>

正規化で最も多い事故は **片側だけ正規化する**ことと、**クエリの正規化を忘れる**ことだ。DB 側もクエリ側も同じ前処理で
正規化しないと内積はコサインにならない。本章では学習用に非破壊の `l2_normalize()`（コピーを返す）を helper に用意しつつ、
実運用で定番の **in-place な `faiss.normalize_L2(x)`（元配列を破壊する）** も `02` で体験する。`normalize_L2` は
「その場で書き換える」ので、生ベクトルを後で使いたいなら **コピーしてから**渡すこと。最後に、距離の向きにも注意がいる。
`IndexFlatIP`（内積）は **大きいほど近い**、`IndexFlatL2`（L2距離）は **小さいほど近い** で、両者ではソート方向が逆になるからだ。

---

## 3. 正準 API — CLIP 埋め込みと FAISS の最小セット

**CLIP（transformers v5）**。`CLIPModel` / `CLIPProcessor`（`AutoModel` / `AutoProcessor` でも可）を使う。
v5 の最重要ポイントは、`get_image_features` / `get_text_features` が **テンソルではなく
`BaseModelOutputWithPooling` を返す**ことだ。射影後の埋め込みは **`.pooler_output`** に入っている（旧版はテンソル直返しだった）。
本章の helper はどちらでも壊れないよう `features.pooler_output if hasattr(...) else features` で吸収している。

```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
with torch.inference_mode():                      # 勾配を完全に切る（軽い・速い）
    pi = processor(images=imgs, return_tensors="pt")
    img_vec = model.get_image_features(pixel_values=pi["pixel_values"]).pooler_output  # (N, 512)
    ti = processor(text=txts, return_tensors="pt", padding=True)
    txt_vec = model.get_text_features(input_ids=ti["input_ids"],
                                      attention_mask=ti["attention_mask"]).pooler_output  # (M, 512)
```

**FAISS のコサイン検索＋ID管理**。入力は必ず **float32・C連続**（`np.ascontiguousarray(x, dtype=np.float32)`）。

```python
import faiss, numpy as np
d = 512
xb = np.ascontiguousarray(embeddings, dtype=np.float32)
faiss.normalize_L2(xb)                              # in-place 正規化（内積=コサインへ）
index = faiss.IndexIDMap(faiss.IndexFlatIP(d))      # 任意 int64 ID を割り当て可能に
index.add_with_ids(xb, ids.astype(np.int64))        # ids[i] ↔ メタデータ[i] は別管理
scores, found = index.search(xq, k)                 # scores=コサイン(大きいほど近い), found=ID
faiss.write_index(index, "engine.faiss")            # 永続化（メタJSONも必ずセットで）
index2 = faiss.read_index("engine.faiss")
```

`search` の戻り値 `found` には、近傍が足りないと **`-1`** が混じる。`-1` を ID としてメタ参照に使うとクラッシュするので、
**必ずガード**する。また `IndexIDMap` を使わず `IndexFlat` に `add` すると ID は `0..N-1` 固定になり、後からメタと
対応づけにくい。**最初から `IndexIDMap` + `add_with_ids`** にしておくのが実務の鉄則だ。規模が出たら `IndexFlatIP` を
`IndexIVFFlat`（要 `train`、`nprobe` で精度調整）や `IndexHNSWFlat`（`train` 不要、`efSearch` で精度調整）に差し替えるが、
**`add` / `search` の呼び出しは同じ**まま使える。

<figure class="lec-fig"><svg viewBox="0 0 660 280" role="img" aria-label="IndexIDMapで画像に1000番台テキストに2000番台のIDを付与し、idとメタデータを別テーブルで管理する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="34" y="54" width="300" height="150" rx="10" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="184" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">1つの index（IndexIDMap）</text><text x="70" y="120" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">画像</text><rect x="104" y="100" width="64" height="26" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="136" y="118" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">1001</text><rect x="176" y="100" width="64" height="26" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="208" y="118" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">1002</text><rect x="248" y="100" width="64" height="26" rx="6" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="280" y="118" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">1003</text><text x="70" y="172" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">テキスト</text><rect x="104" y="152" width="64" height="26" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="136" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">2001</text><rect x="176" y="152" width="64" height="26" rx="6" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="208" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">2002</text><text x="184" y="196" text-anchor="middle" font-size="11" fill="#52525b">1000番台=画像 ／ 2000番台=テキスト</text><line x1="334" y1="128" x2="388" y2="128" stroke="#71717a" stroke-width="1.8"/><polygon points="394,128 384,123 384,133" fill="#71717a"/><rect x="396" y="54" width="240" height="150" rx="10" fill="#fff7ed" stroke="#f97316" stroke-width="1.5"/><text x="516" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">別テーブル：id → メタ</text><text x="412" y="112" font-size="11" fill="#18181b">1001 → 画像 / red_circle.png</text><text x="412" y="140" font-size="11" fill="#18181b">1002 → 画像 / blue_square.png</text><text x="412" y="168" font-size="11" fill="#18181b">2001 → テキスト / a red circle</text><text x="330" y="240" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">search が返す −1（近傍不足）はメタ参照前にガード</text></svg><figcaption><code>IndexIDMap</code> + <code>add_with_ids</code> を使うと、各ベクトルに<b>任意の整数ID</b>を割り当てられます。<b>画像を 1000 番台・テキストを 2000 番台</b>のように番号帯で分けておくと、後で <code>IDSelectorRange</code> で絞りやすくなります。<code>id ↔ メタデータ</code>（モダリティ・画像パス・説明文）は<b>別テーブル（JSON 等）で管理</b>し、<code>index</code> と必ずセットで永続化します。<code>search</code> が返す <b>−1</b>（近傍不足）を ID として使うとクラッシュするので、メタ参照の前に必ずガードします。</figcaption></figure>

---

## 4. 実装を1つずつ — 4本のスクリプトで段階的に組む

**`01_shared_space_clip.py` — 共有空間の確認。** 合成「色×形」画像 48 枚と説明文 12 本を CLIP で埋め込み、
両者が同じ 512 次元になること、正規化前後のノルム、`text × image` のコサイン行列を確認する。`text → image` の
top-1 が 12/12 一致し、`text → text`（"red circle" vs "red square" ≈ 0.68）も自然に測れることを見て、
「同じ空間なら全方向に比較できる」感覚を掴む。出力は `01_text_image_similarity.png`（類似度ヒートマップ）。

**`02_faiss_multimodal_index.py` — エンジンの中身を生 API で。** 画像とテキストの埋め込みを縦に積んで1つの行列にし、
`faiss.normalize_L2`（in-place）→ `IndexIDMap(IndexFlatIP)` → `add_with_ids` で **画像に 1000 番台・テキストに 2000 番台**の
ID を割り当てる。`id ↔ メタ` 辞書を作り、テキストクエリで検索 → ID からメタを引く流れを手で書く。さらに index を
`.faiss`＋メタを `.json` で**セット保存**し、読み戻して件数一致を確認。最後に **同じデータで IVF / HNSW を作り、Flat（厳密）と
上位 k が一致するか**を見て近似 index の入口に立つ。ここで学ぶ「正規化・ID管理・永続化」が、次の helper クラスの中身そのものだ。

**`03_crossmodal_search.py` — 統一インターフェース。** `mm_helpers.MultimodalSearchEngine`（=02 を packaging したもの）に
画像とテキストを両方 `add` し、**4方向すべてを同じ `engine.search()` で**実行する。`text → image` は
`search(text_vec, modality="image")`、`image → image` は `search(image_vec, modality="image")`、
`text → text` は `modality="text"`、混在クロスモーダルは `modality=None`。ここで重要な観察がある。index には画像とテキストの
両方が入っているので、**テキストクエリを素で投げるとテキスト同士（コサイン 0.85 前後）が画像（0.33 前後）より上位に来る**。
だから「テキストで画像を引きたい」なら結果側を `modality="image"` で絞るのが実務の定石になる。出力は `03_text_to_image.png` /
`03_image_to_image.png`。

<figure class="lec-fig"><svg viewBox="0 0 640 285" role="img" aria-label="クエリと結果のモダリティの2×2で4方向検索。1つのsearchにmodality絞りを付けて使い分ける" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="80" y="56" text-anchor="middle" font-size="11" fill="#52525b">クエリ ↓</text><text x="80" y="112" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">テキスト</text><text x="80" y="198" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">画像</text><text x="262" y="56" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">→ 画像</text><text x="494" y="56" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">→ テキスト</text><rect x="156" y="70" width="212" height="74" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="262" y="100" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">text → image</text><text x="262" y="123" text-anchor="middle" font-size="11.5" fill="#52525b">modality=&quot;image&quot;</text><rect x="388" y="70" width="212" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="494" y="100" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">text → text</text><text x="494" y="123" text-anchor="middle" font-size="11.5" fill="#52525b">modality=&quot;text&quot;</text><rect x="156" y="156" width="212" height="74" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="262" y="186" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">image → image</text><text x="262" y="209" text-anchor="middle" font-size="11.5" fill="#52525b">modality=&quot;image&quot;</text><rect x="388" y="156" width="212" height="74" rx="8" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="494" y="186" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">image → text</text><text x="494" y="209" text-anchor="middle" font-size="11.5" fill="#52525b">modality=&quot;text&quot;</text><text x="320" y="262" text-anchor="middle" font-size="11.5" fill="#52525b">modality=None なら全モダリティ混在（クロスモーダル）</text></svg><figcaption>index に画像とテキストの両方を入れておけば、<b>4方向すべてを1つの <code>engine.search()</code></b> で実行できます（クエリのベクトルを変え、結果側を <code>modality</code> で絞るだけ）。注意点として、<b>テキストクエリを素で投げるとテキスト同士（コサイン 0.85 前後）が画像（0.33 前後）より上位</b>に来ます。だから「テキストで画像を引きたい」ときは結果を <b><code>modality="image"</code></b> で絞るのが定石です。<code>modality=None</code> なら全モダリティ混在のクロスモーダル検索になります。</figcaption></figure>

**`04_recall_eval.py` — 定量評価。** 画像だけを DB にし、各テキスト（色×形）の **正解集合を「色と形が一致する画像」**として
人手定義する（**正解は検索器の出力で作らない**のが鉄則）。`Recall@k` と `mAP@k` を測ると、本章の合成データでは
おおよそ **Recall@1≈0.25 / Recall@4≈0.88 / Recall@8≈1.00 / mAP@8≈0.93** になる。`Recall@1` が 0.25 で頭打ちなのは
バグではなく、正解が 4 枚あるのに 1 件しか取れない k=1 では最大でも 1/4=0.25 だからだ（この「指標の上限」を理解するのも評価の勘所）。
出力は `04_recall_curve.png` と `04_recall_report.json`。

<figure class="lec-fig"><svg viewBox="0 0 560 290" role="img" aria-label="Recall@kの曲線。正解が4枚あるためk=1の上限は1/4=0.25でkを増やすとRecallが1.0へ伸びる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="74" y1="52" x2="74" y2="236" stroke="#3f3f46" stroke-width="1.8"/><line x1="74" y1="236" x2="515" y2="236" stroke="#3f3f46" stroke-width="1.8"/><line x1="74" y1="193" x2="510" y2="193" stroke="#f4f4f5" stroke-width="1"/><line x1="74" y1="150" x2="510" y2="150" stroke="#f4f4f5" stroke-width="1"/><line x1="74" y1="107" x2="510" y2="107" stroke="#f4f4f5" stroke-width="1"/><line x1="74" y1="64" x2="510" y2="64" stroke="#f4f4f5" stroke-width="1"/><text x="64" y="240" text-anchor="end" font-size="11" fill="#71717a">0</text><text x="64" y="197" text-anchor="end" font-size="11" fill="#71717a">0.25</text><text x="64" y="154" text-anchor="end" font-size="11" fill="#71717a">0.5</text><text x="64" y="111" text-anchor="end" font-size="11" fill="#71717a">0.75</text><text x="64" y="68" text-anchor="end" font-size="11" fill="#71717a">1.0</text><line x1="74" y1="193" x2="490" y2="193" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4"/><polyline points="126,193 282,85 490,64" fill="none" stroke="#ea580c" stroke-width="2.8"/><circle cx="126" cy="193" r="5" fill="#c2410c"/><circle cx="282" cy="85" r="5" fill="#c2410c"/><circle cx="490" cy="64" r="5" fill="#c2410c"/><text x="282" y="72" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">0.88</text><text x="490" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#16a34a">1.0</text><text x="126" y="252" text-anchor="middle" font-size="11.5" fill="#3f3f46">1</text><text x="282" y="252" text-anchor="middle" font-size="11.5" fill="#3f3f46">4</text><text x="490" y="252" text-anchor="middle" font-size="11.5" fill="#3f3f46">8</text><text x="294" y="276" text-anchor="middle" font-size="12" fill="#52525b">k（取得した件数）</text><text x="30" y="150" text-anchor="middle" font-size="12" font-weight="700" fill="#3f3f46" transform="rotate(-90 30 150)">Recall@k</text><text x="360" y="212" text-anchor="middle" font-size="11.5" font-weight="700" fill="#dc2626">k=1 の上限 0.25 ＝ 1/正解4枚</text></svg><figcaption>正解画像が <b>4 枚</b>あるとき、<code>Recall@k</code> は「上位 k 件に入った正解数 ÷ 4」です。<b>k=1 では最大でも 1/4 = 0.25</b> が上限で、これはバグではなく<b>指標の定義上の天井</b>です。k を増やすほど取りこぼしが減り、<code>Recall@4 ≈ 0.88</code>、<code>Recall@8 ≈ 1.0</code> へ伸びます。正解は検索器の出力ではなく<b>人手で定義</b>するのが鉄則です。</figcaption></figure>

---

## 🛠 章末ミニプロジェクト — 統一マルチモーダル検索エンジンの完成形

`mini_project.py` は本章の学びを **end-to-end の一周**に統合した完成形で、そのまま動く。

1. **構築 (STEP1)**: 画像＋テキストを CLIP で埋め込み、`MultimodalSearchEngine` に載せる。
   `data/42_multimodal_vector_search/` に実画像があればそれを優先し、無ければ合成（色×形）を使う。
2. **永続化 (STEP2)**: `engine.save()` で `.faiss` + `.meta.json` を書き出し、`MultimodalSearchEngine.load()` で
   **別オブジェクトとして読み戻して**件数一致をアサート（保存と復元の整合）。
3. **統一検索 (STEP3)**: 読み戻したエンジンで **4方向検索**（画像→画像 / テキスト→画像 / テキスト→テキスト / クロスモーダル）を実行。
4. **評価 (STEP4)**: `text → image` の `Recall@k` / `mAP@8` を測り、`mini_report.json` と `mini_text_to_image.png` に保存。

```bash
uv run python lectures/42_multimodal_vector_search/mini_project.py
```

**腕試し（発展課題）**: ①`build_collection(variants=8)` で画像を増やし Recall がどう動くか観察する。②`MultimodalSearchEngine` の
`IndexFlatIP` を `index_factory(d, "HNSW32", faiss.METRIC_INNER_PRODUCT)` に差し替え、検索結果が変わらないことを確かめる。
③`data/42_multimodal_vector_search/` に手元の実写画像を置き、テキストクエリ（英語）で引けるか試す。

---

## ✅ 到達チェックリスト

- [ ] CLIP の `get_image_features` / `get_text_features` から **`.pooler_output`** で 512 次元ベクトルを取り出せる（v5）。
- [ ] 画像とテキストが**同じ次元・同じ空間**にある理由（対照学習）を自分の言葉で説明できる。
- [ ] 「L2 正規化 → `IndexFlatIP`」が**コサイン検索**になる理屈を説明でき、DB・クエリ両方を正規化できる。
- [ ] 入力を **float32・C連続**に揃え、`search` の **`-1`** をガードできる。
- [ ] `IndexIDMap` + `add_with_ids` で**任意ID**を付け、`id ↔ メタデータ`を別管理できる。
- [ ] `write_index`/`read_index` と **メタJSON をセットで**永続化し、別プロセスで復元できる。
- [ ] **4方向検索**を1つの `search()`（＋`modality` 絞り）で実装できる。
- [ ] `Flat / IVF / HNSW` を**同じ API**で差し替えられ、`Recall@k` / `mAP` で評価できる。
- [ ] 新モダリティ（音声 CLAP / ImageBind）を足すときの**空間共有の条件**を説明できる。

---

## ❓ 落とし穴・FAQ・デバッグ

- **`get_image_features(...)` がテンソルじゃない / `.shape` で落ちる** → v5 は `BaseModelOutputWithPooling` を返す。
  **`.pooler_output`** を取る。古いチュートリアルのコードはここで壊れる。
- **コサインのつもりが結果が崩れる** → `IndexFlatIP` なのに正規化を忘れている（=素の内積になる）。
  **DB 側もクエリ側も** `normalize_L2` する。`normalize_L2` は **in-place で元配列を破壊**する点にも注意（生ベクトルが要るならコピー）。
- **`search` が落ちる / 結果がおかしい** → 入力が float32 でない／C連続でない。`np.ascontiguousarray(x, dtype=np.float32)` を徹底。
- **メタ参照で `KeyError` / クラッシュ** → `found` に混じる **`-1`**（近傍不足）を ID として使っている。ループで `if i == -1: continue`。
- **テキストで画像を引いたのにテキストばかり返る** → index に両モダリティが入っており、テキスト同士の方が近い。
  **`modality="image"` で結果を絞る**（本 helper の後フィルタ）。厳密に絞るなら FAISS の `IDSelector`（後述）。
- **次元不一致で静かにバグる** → モデルを変えて埋め込み次元が変わった等。index 生成時の `d` と `add/search` の次元を**最初にアサート**。
- **IVF/PQ で `add` 時に例外** → `train` を忘れている。代表データで `index.train(xt)` してから `add`。
- **IVF の Recall が低い** → `nprobe` が 1 のまま（デフォルトの罠）。`nprobe` を上げて精度↑（速度↓）。
- **読み戻したらメタが消えた** → `write_index` は **index 本体のみ**保存する。`id → メタ` の対応表（JSON/SQLite）を**必ず一緒に**保存・バージョン整合。
- **Recall がやけに高い／評価が信用できない** → ground truth を **ANN 自身**から作っている。正解は**人手ラベル**か **Flat（厳密）**で作る。
- **`Recall@1` が 1.0 にならない** → 正解が複数枚あると k=1 では取り切れない（上限 = 1/正解数）。指標の定義通りで、バグではない。
- **合成画像で CLIP の精度がいまいち** → 実写の方が CLIP の得意分野。`data/` に実画像を置けば本章のコードはそのまま発火する。

---

## 🚀 発展トピック・参考

**SigLIP に差し替える。** `google/siglip2-base-patch16-224` を `AutoModel`/`AutoProcessor` で読み、同じく
`get_image_features` / `get_text_features` で埋め込みを得る（要 `sentencepiece`）。多言語・高精度で、検索の質を上げやすい。
損失が sigmoid なので確率解釈は CLIP と違うが、**検索（コサインで並べる）用途では使い方は同じ**だ。

**近似 index と速度–精度のチューニング。** 数十万〜数百万件になったら `Flat` は遅い。`index_factory(d, "IVF4096,PQ64", METRIC_INNER_PRODUCT)` で
転置インデックス＋PQ 圧縮、または `"HNSW32"` でグラフ ANN にする。`nprobe`（IVF）/`efSearch`（HNSW）を振って
**QPS–Recall 曲線**を描き、用途に合う点を選ぶ（17回の `04_recall_qps_eval.py` が実践例）。

**modality を厳密に絞る（IDSelector）。** 本章は「多めに取って Python で後フィルタ」で簡明に modality を絞ったが、
実運用で大規模なら FAISS の `IDSelectorBatch` / `IDSelectorRange` を `SearchParameters` に渡して **index 内で絞る**方が速い。
画像 ID とテキスト ID を**番号帯で分けて採番**（02 でやった 1000/2000 番台）しておくと `IDSelectorRange` と相性が良い。

**新モダリティ（音声）を足す設計 — CLAP / ImageBind。** ここが「統一設計」の真価。**鍵は“同じ空間か”だ。**
CLAP（`laion/clap-htsat-unfused`、transformers の `ClapModel`、`get_audio_features` / `get_text_features`）は
**音声とテキスト**を1つの空間に揃えるが、その空間は **CLIP の空間とは別物**。よって「テキストで音声も画像も同時に引く」には、
(a) **空間ごとに index を分け**、テキストクエリを各モデルで埋め込んで各 index を引き、結果を後段で統合する（実務的で堅い）か、
(b) **画像・テキスト・音声を1つの空間に束ねる ImageBind** を使う（`ModalityType.VISION/TEXT/AUDIO` を同次元へ）かの二択になる。
本 `MultimodalSearchEngine` は **(b) のように同一空間に揃ったベクトルなら、modality タグを付けて `add` するだけで拡張できる**設計に
なっている。下のように **重い依存は import を try/except でガード**し、未導入なら案内のみ出すのが安全だ（実行経路では使わない）。

```python
# 音声モダリティ拡張のスケッチ（任意・本章の実行経路では使わない）
try:
    from transformers import ClapModel, ClapProcessor   # uv add --group audio transformers でロード可
    # clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    # audio_vec = clap.get_audio_features(**proc(audios=wavs, sampling_rate=48000, return_tensors="pt"))
    # ※CLAP 空間は CLIP 空間と別。engine を分けるか ImageBind で同一空間に束ねる。
    HAS_CLAP = True
except Exception:
    HAS_CLAP = False
    print("CLAP 未導入。概念のみ（`uv add --group audio transformers torchaudio` で試せる）")
```

**インストールの注意。** 本章は `faiss-cpu` で全機能が動く（検索ロジックは CPU/GPU で同一）。
**`faiss-gpu` という pip 名は無い** — GPU 版の PyPI 名は `faiss-gpu-cuvs`（Linux x86_64 + NVIDIA 限定、`faiss-cpu` と排他）であり、
同一環境に両方入れると `import faiss` が衝突する。`insightface` / `mediapipe` / `ImageBind` / `CLAP` 系の重い依存は
**実行経路では使わず**、必要なら別グループ（例: `uv add --group audio ...`）で隔離し、コード側は try/except でガードする。

**参考**: FAISS wiki（<https://github.com/facebookresearch/faiss/wiki>）/ CLIP・SigLIP（HuggingFace docs）/
ImageBind（<https://github.com/facebookresearch/ImageBind>）/ 17回（FAISS 基礎）・16回（CLIP ゼロショット）。

---

## 💡 実践ユースケース集

ここまでの「共有空間 × FAISS × id↔メタ × 永続化」は、そのまま現実の小ツールになる。ここでは代表的な応用を3つ挙げよう。
なかでも1つ目はこのフォルダの **`use_case.py` として実際に動く**ので、まず動かしてみてから、自分のデータ・用途に合わせて育てていくとよい。

### ① パーソナル画像検索（セマンティック写真検索）← `use_case.py`

- **何に使うか**: 手元の写真フォルダを「a red car」「a dog on the beach」のような**自然文**で引く、Google Photos の
  「犬」「海」検索のローカル版。**見本画像1枚**で似た写真を引くこともできる。家族写真・素材集・スクショ整理の実用ツール。
- **作り方の要点**: 写真を CLIP で1度だけ埋め込み、`MultimodalSearchEngine`（L2正規化 → `IndexFlatIP` でコサイン）に `add` し、
  `.faiss` + `.meta.json` で**ディスクにキャッシュ**する。2回目以降は **埋め込みをやり直さず読み込むだけ**で即検索 ——
  これがベクトルDBを永続化する最大の利点。クエリは「テキストを埋め込む / 見本画像を埋め込む」だけで、`search()` は同じ。
- **注意**: CLIP は**英語**前提（クエリは英語が無難）。`get_*_features` は未正規化なので **DB・クエリ両方を正規化**。
  写真を足し引きしたらキャッシュは作り直す（本ツールは枚数一致で簡易判定。実運用は mtime ハッシュ等で厳密化）。
  合成の「色×形」では CLIP のコサインは 0.3 前後と低めに出るが、**実写の方が CLIP の得意分野**で素直に効く。

```bash
uv run python lectures/42_multimodal_vector_search/use_case.py
```

実行すると `outputs/42_multimodal_vector_search/` に `use_case_text_search.png`（自然文→写真）/
`use_case_image_search.png`（見本→似た写真）/ `use_case_report.json` / `use_case_library.faiss`（+`.meta.json`、次回起動を速くするキャッシュ）が出る。
`data/42_multimodal_vector_search/` に自分の写真（`.png/.jpg`）を置いて再実行すれば、**合成より優先**して実写ライブラリ検索になる
（1枚も無ければ合成「色×形」で必ず完走）。
**練習（拡張）アイデア**: ①クエリを `argparse`（`--text "a dog"` / `--image path`）で受け取る対話CLIにする。
②写真が増えたら `IndexFlatIP` を `index_factory(d, "HNSW32", faiss.METRIC_INNER_PRODUCT)` に差し替える（`add/search` は同じ）。
③EXIF の日時・GPS をメタに足して日付・場所で絞る。④image→image で「そっくり写真」を炙り出す near-dup ファインダーに発展させる。

### ② クロスモーダル商品検索（EC・カタログ）

- **何に使うか**: EC の商品画像カタログを、ユーザの**自然文**（「白い夏物ワンピース」）でも、**お気に入り画像**でも引ける検索。
  「文でも画像でも同じ1つの index を引く」本章の統一設計がそのまま刺さる代表例。
- **作り方の要点**: 商品画像を埋め込んで FAISS に載せ、`id ↔ 商品メタ`（SKU・価格・在庫・カテゴリ）を**別管理**（JSON/SQLite）。
  テキストクエリ・画像クエリのどちらも同じ `search()` で引き、結果 ID からメタを引いてカードを描画。在庫切れ等は
  `IDSelector` で**index内で除外**するか後フィルタする。
- **注意**: 商品説明は多言語になりがち —— 多言語なら **SigLIP2** に差し替えると素直（使い方は同じ、要 `sentencepiece`）。
  価格・在庫のような**頻繁に変わる属性は埋め込みに混ぜず**メタ側で持つ（再埋め込み不要にする）。`-1`（近傍不足）は必ずガード。

### ③ 重複・そっくり画像の棚卸し（near-duplicate / DAM）

- **何に使うか**: 素材ライブラリやスクショの山から、**重複・ほぼ同一の画像**を見つけて整理・容量削減する。
  アップロード時の「既出チェック」やコンテンツの取り違え防止（DAM = デジタル資産管理）にも使える。
- **作り方の要点**: 全画像を埋め込み・正規化して `IndexFlatIP` に載せ、各画像を**自分自身をクエリ**にして近傍を引く。
  コサインが**しきい値（例 0.95 以上）**の組を「重複候補」として束ねる。自分自身（cos≈1.0）は必ず除外する。
- **注意**: しきい値は**実データで調整**（高すぎると取りこぼし／低すぎると別物まで束ねる）。トリミングや色調補正に強くしたいなら
  CLIP 埋め込みが向くが、完全一致のバイト重複検出ならハッシュ（pHash 等）の方が速い —— **用途で道具を選ぶ**。

---

## ▶ 動かし方

```bash
# 依存（未導入なら）: 深層学習・HuggingFace・FAISS・評価指標
uv sync --group dl --group hf --group vector --group metrics

# 本編（番号順）。初回のみ CLIP の重みを DL（以後キャッシュ）
uv run python lectures/42_multimodal_vector_search/01_shared_space_clip.py
uv run python lectures/42_multimodal_vector_search/02_faiss_multimodal_index.py
uv run python lectures/42_multimodal_vector_search/03_crossmodal_search.py
uv run python lectures/42_multimodal_vector_search/04_recall_eval.py

# 章末ミニプロジェクト（統一エンジンの完成形）
uv run python lectures/42_multimodal_vector_search/mini_project.py

# 実践ユースケース（パーソナル画像検索ツール。data/<id>/ に写真を置くと実運用に切替）
uv run python lectures/42_multimodal_vector_search/use_case.py

# 演習（自己採点。未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/42_multimodal_vector_search/exercises.py
uv run python lectures/42_multimodal_vector_search/exercises_solutions.py
```

成果物（図・JSON・index）は `outputs/42_multimodal_vector_search/` に保存される。
`data/42_multimodal_vector_search/` に実画像（`.png/.jpg`）を置くと、合成より優先して使われる。

---

> 参照ライブラリ: **torch 2.12+cpu** / **torchvision 0.27+cpu** / **transformers 5.11** / **faiss-cpu 1.14**
> （CLIP: `openai/clip-vit-base-patch32`、headless OpenCV、matplotlib=Agg・BGR→RGB 注意、CPU・`model.eval()`+`torch.inference_mode()`） — 2026-06