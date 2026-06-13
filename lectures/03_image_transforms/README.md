# 第3回 色空間・描画・幾何変換 — 前処理パイプラインの土台

> トラック: 画像の基礎 ／ レベル: 初級 ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加グループ不要）

## 🎯 この章のゴール

第1回では「画像は `(H, W, 3)` の `uint8` numpy 配列にすぎない」という地盤を作りました。この章では、その配列を**目的に合わせて作り替える基本操作**——色空間の変換、図形やテキストの描き込み、リサイズ・反転・クロップといった幾何変換——を、すべて「正準的な（現行版で正しい）書き方」で手に馴染ませていきます。どれも一見地味な操作ですが、検出・セグメンテーション・分類など後段のあらゆるタスクにおいて、**前処理**と**結果の可視化**の両面で必ず使う、避けて通れない基礎技能です。

この章の隠れた主題は「**軸順と数値スケールの食い違いを、毎回意識的に変換する**」ことです。たとえば `cv2.resize` の `dsize` は `(幅W, 高さH)` 順なのに numpy の `shape` は `(高さH, 幅W)` 順であり、PIL の `size` も `(W, H)` 順、さらに OpenCV の HSV は色相 `H` が `0-179`（一般的な `0-360` ではない）——こうした「順序が逆」「スケールが半分」といった罠は、知らなければ一日溶かしかねませんが、知ってさえいれば一行で直せます。そこで本章では、これらを単なる知識としてではなく「自分で再現し、回避コードを書ける」レベルにまで落とし込みます。

到達点を一言でいえば、**HSV 色域で特定の物体を抜き出すマスク生成器**と、**アスペクト比を保ったリサイズ・正方形への整形・EXIF 向き正規化までを含む再利用可能な前処理関数群**を、AI 補助なしでそらで書けることです。これがこの章の合格ラインであり、その成果物そのものは `preprocess.py` にまとまっています。

---

## 1. 色空間変換 — Gray は次元が減り、HSV は独自スケール

色空間とは「色をどんな数値の組で表すか」という取り決めです。OpenCV が読み込む画像は既定で **BGR**（青・緑・赤）ですが、用途に応じてこれを別の表現へ変換します。なかでも最頻出は2つです。ひとつは**グレースケール**（輝度だけの白黒）で、エッジ検出や閾値処理の前段としてほぼ必ず通ります。もうひとつが **HSV**（色相・彩度・明度）で、こちらは「色で物体を選り分けたい」ときの定番です。いずれの変換も `cv2.cvtColor(img, コード)` の一行で済みます。

ここで第1回の知識が効いてきます。BGR（3ch）をグレースケールに変換すると、戻り値の `shape` は `(H, W, 1)` ではなく **`(H, W)` の2次元**になり、チャンネル軸そのものが消えます。これは「色を1つに落とすと次元が1つ減る」という対応です。逆に、グレースケールを他のカラー画像と並べたり重ねたりしたいときは、`cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` によって見た目は白黒のまま3チャンネルへ戻します。次元数が揃っていないと連結や重ね合わせで形が合わずに落ちるため、`shape` を一目見て次元を数える癖をつけておきましょう。

一方、HSV で最も重要なのは**スケールの癖**です。一般的なカラーピッカーが色相 `H` を 0〜360 度で表すのに対し、**OpenCV の HSV は `H` を `0-179`（半分の値）**で持ち、彩度 `S`・明度 `V` は `0-255` を取ります。これは `uint8`（最大255）に色相を収めるための仕様であり、そのため Web の色相環の角度をそのまま入れると色域抽出に失敗します。`01_colorspace_hsv_mask.py` は純色の画素を実際に変換し、赤 `H=0`・黄 `H=30`・緑 `H=60`・青 `H=120` という値を数値で示してくれます。

<figure class="lec-fig"><svg viewBox="0 0 600 230" role="img" aria-label="OpenCVのHSVは色相Hが0から179で一般的な0から360度の半分。角度を2で割った値を使う" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="42" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">一般的な H（色相環の角度）</text><line x1="75" y1="80" x2="525" y2="80" stroke="#71717a" stroke-width="2.5"/><polygon points="540,80 526,74 526,86" fill="#71717a"/><text x="550" y="85" font-size="14" font-weight="700" fill="#18181b">360°</text><line x1="75" y1="152" x2="525" y2="152" stroke="#2563eb" stroke-width="2.5"/><polygon points="540,152 526,146 526,158" fill="#2563eb"/><text x="548" y="157" font-size="14" font-weight="700" fill="#1d4ed8">179</text><text x="55" y="121" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">0</text><line x1="75" y1="80" x2="75" y2="152" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="525" y1="80" x2="525" y2="152" stroke="#c2410c" stroke-width="1.6" stroke-dasharray="4 3"/><text x="300" y="201" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">OpenCV の H ＝ 角度 ÷ 2（0–179）</text><circle cx="75" cy="152" r="7" fill="#dc2626"/><circle cx="226" cy="152" r="7" fill="#16a34a"/><circle cx="377" cy="152" r="7" fill="#2563eb"/></svg><figcaption>OpenCV の <b>HSV</b> は、色相 <b>H</b> を一般的な <b>0–360°</b> の<b>半分の 0–179</b> で表します（<code>uint8</code> に収めるため）。Web の色相環の角度をそのまま入れると色域指定がずれるので、<b>角度を ÷2 した値</b>を使います。図の点は OpenCV 値で 赤 <code>H=0</code>・緑 <code>H=60</code>・青 <code>H=120</code>。彩度 S・明度 V は <code>0–255</code> です。</figcaption></figure>

```python
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)   # (H, W) ← 次元が1つ減る
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)     # H=0-179, S/V=0-255
```

上のコードを実行したら、必ず `gray.shape` が2次元・`hsv` が3次元であること、そして各純色の `H` 値を確認してください。さしあたり「`H` は 0-360 の半分」とだけ覚えておけば、後述する色マスクの数値設計でつまずくことはありません。

## 2. `inRange` による色マスク — 赤は 0/179 をまたぐ

特定の色だけを抜き出したいとき、BGR のまま「赤が強い画素」を条件で書くのは至難です（明るさによって R の値が大きく動いてしまうため）。そこで画像を HSV に変換し、**色相 `H` の範囲**で選別します。`cv2.inRange(hsv, lower, upper)` は、各画素が `[lower, upper]` の範囲内なら 255、外なら 0 とする2値マスク（`(H, W)` の `uint8`）を返す関数です。このとき色相に加えて彩度 `S`・明度 `V` にも下限を設けておくと、背景の灰色（彩度が低い）や暗部までまとめて確実に除外できます。

ここで初学者が必ず引っかかるのが**赤**です。赤の色相は `H=0` 付近にありますが、色相環は一周してつながっているため、赤は反対側の `179` 付近にもはみ出します。したがって `[0, 10]` の一区間だけでは赤の半分を取りこぼしてしまいます。正攻法は、**`[0, 10]` と `[170, 179]` の2つのマスクを作り、`cv2.bitwise_or` で合成する**ことです。これは「色相環の端をまたぐ色は2区間に分ける」という、HSV 色抽出の最重要パターンといえます。

<figure class="lec-fig"><svg viewBox="0 0 560 245" role="img" aria-label="赤は色相環の0と179の両端にまたがるため、inRangeを0から10と170から179の2区間に分けbitwise_orで合成する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><circle cx="135" cy="120" r="80" fill="none" stroke="#d4d4d8" stroke-width="14"/><path d="M 135 40 A 80 80 0 0 1 192 63" fill="none" stroke="#dc2626" stroke-width="14"/><path d="M 135 40 A 80 80 0 0 0 78 63" fill="none" stroke="#dc2626" stroke-width="14"/><circle cx="204" cy="160" r="7" fill="#16a34a"/><circle cx="66" cy="160" r="7" fill="#2563eb"/><text x="58" y="52" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">179</text><text x="212" y="52" text-anchor="middle" font-size="14" font-weight="700" fill="#dc2626">0</text><text x="135" y="125" text-anchor="middle" font-size="12" fill="#71717a">色相環</text><line x1="228" y1="108" x2="332" y2="108" stroke="#71717a" stroke-width="2"/><polygon points="346,108 332,102 332,114" fill="#71717a"/><rect x="352" y="46" width="150" height="40" fill="#ffedd5" stroke="#dc2626" stroke-width="2"/><text x="427" y="72" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">[0, 10]</text><text x="427" y="109" text-anchor="middle" font-size="15" font-weight="700" fill="#3f3f46">OR</text><rect x="352" y="120" width="150" height="40" fill="#ffedd5" stroke="#dc2626" stroke-width="2"/><text x="427" y="146" text-anchor="middle" font-size="16" font-weight="700" fill="#c2410c">[170, 179]</text><line x1="427" y1="162" x2="427" y2="180" stroke="#71717a" stroke-width="2"/><polygon points="427,188 421,178 433,178" fill="#71717a"/><rect x="352" y="188" width="150" height="42" fill="#dc2626"/><text x="427" y="215" text-anchor="middle" font-size="16" font-weight="700" fill="#ffffff">mask_red</text></svg><figcaption><b>赤</b>は色相環の <b>0</b> 付近と <b>179</b> 付近の<b>両端にまたがる</b>ため、<code>[0,10]</code> の1区間だけでは半分を取りこぼします。<code>inRange</code> を <b>[0,10]</b> と <b>[170,179]</b> の2区間で作り、<code>cv2.bitwise_or</code> で合成して <code>mask_red</code> を得ます（緑・青は単一区間で済みます）。</figcaption></figure>

下のコードは、青（単一区間でよい）と赤（2区間を OR）の対比になっています。あわせて `S`・`V` の下限を `80, 60` 程度に上げている点にも注目してください。ここが、背景のグレーを弾く効きどころです。

```python
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))         # 青は単一区間
red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
red2 = cv2.inRange(hsv, (170, 80, 60), (179, 255, 255))
mask_red = cv2.bitwise_or(red1, red2)                                # 赤は2区間を合成
```

抜き出したマスクを使って元画像の色だけを残すには、`cv2.bitwise_and(bgr, bgr, mask=mask)` を用います。これら一連の処理を色名で呼べるようまとめたのが成果物 `preprocess.extract_color(bgr, "red")` で、赤の両端またぎや軽いノイズ除去まで内側で面倒を見てくれます。`01_colorspace_hsv_mask.py` は4色すべてを抽出して保存するので、`01_mask_red.png` などを開き、「狙った物体だけが白くなっている」ことを確認してください。

## 3. 図形・テキストの描画 — 検出結果の可視化に直結する

`ndarray` には、専用 API を使って線・矩形・円・多角形・テキストを直接描き込めます。具体的には `cv2.line` / `cv2.rectangle` / `cv2.circle` / `cv2.polylines` / `cv2.putText` がそれにあたります。ここでの約束事は3つです。すなわち、**座標は `(x, y)`**（numpy の `img[y, x]` とは引数順が逆！）、**色は BGR タプル**、そして **`thickness=-1` で塗りつぶし**になること——この3点です。さらに `lineType=cv2.LINE_AA` を付けると縁がアンチエイリアスされ、斜線や円のギザギザが滑らかになります。

ただし、`cv2.putText` だけは少し癖があります。座標が指す基準点は文字の**左下（ベースライン）**であって、左上ではありません。また、文字の背景に帯を敷きたい（検出ラベルを読みやすくしたい）場合、`putText` 自体には背景機能がないため、まず `cv2.getTextSize` で文字の幅・高さを測り、その寸法に合わせて塗りつぶし矩形を描いてから、その上に文字を載せます。この「測ってから背景→文字」という手順こそ、見やすいラベル表示の定石です。

<figure class="lec-fig"><svg viewBox="0 0 560 240" role="img" aria-label="検出ラベルは枠と塗りラベル帯と白文字の重ね描き。getTextSizeで測り背景矩形を描きputTextで文字を載せる。座標はxとyの順" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="95" y="80" width="180" height="120" fill="none" stroke="#16a34a" stroke-width="3"/><rect x="95" y="54" width="120" height="26" fill="#16a34a"/><text x="155" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">person 0.97</text><circle cx="95" cy="80" r="5" fill="#ea580c"/><text x="88" y="70" text-anchor="end" font-size="13" fill="#18181b">(x1, y1)</text><text x="185" y="228" text-anchor="middle" font-size="13" fill="#52525b">座標は (x, y) 順</text><rect x="330" y="50" width="190" height="38" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><text x="425" y="74" text-anchor="middle" font-size="13" fill="#c2410c">① getTextSize で測る</text><line x1="425" y1="88" x2="425" y2="99" stroke="#71717a" stroke-width="2"/><polygon points="425,105 419,95 431,95" fill="#71717a"/><rect x="330" y="105" width="190" height="38" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/><text x="425" y="129" text-anchor="middle" font-size="13" fill="#c2410c">② 矩形で背景帯</text><line x1="425" y1="143" x2="425" y2="154" stroke="#71717a" stroke-width="2"/><polygon points="425,160 419,150 431,150" fill="#71717a"/><rect x="330" y="160" width="190" height="38" fill="#16a34a" stroke="#15803d" stroke-width="1.5"/><text x="425" y="184" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">③ putText で白文字</text></svg><figcaption>物体検出の可視化は <b>枠（rectangle）＋ ラベル帯（塗りつぶし <code>thickness=-1</code>）＋ 白文字（putText）</b>の重ね描きです。手順は <b>① <code>getTextSize</code> で文字寸法を測る → ② 帯の矩形を塗る → ③ 文字を載せる</b>。座標は <b><code>(x, y)</code> 順</b>（numpy の <code>img[y, x]</code> とは逆）、色は BGR、<code>putText</code> の基準点は文字の<b>左下</b>です。</figcaption></figure>

```python
# 座標は (x, y)、色は BGR。thickness=-1 で塗りつぶし、LINE_AA で滑らかに。
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2, lineType=cv2.LINE_AA)
(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 200, 0), -1)
cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
```

この一連の処理を関数化したものが `02_drawing.py` の `draw_label_box` で、物体検出の出力でおなじみの「バウンディングボックス＋ラベル帯＋白文字」を、1つの関数（単一責務）で描き上げます。後の検出・追跡の章で結果を可視化するときには、まさにこの形をそのまま使うことになります。生成される `02_detections.png` を開いて、ラベルが枠にきれいに収まっていることを確認してください。

## 4. リサイズ — `dsize=(W, H)` の罠と補間の使い分け

リサイズは前処理の心臓部ですが、`cv2.resize` には2つの落とし穴があります。1つ目は**サイズの順序**です。`cv2.resize(img, dsize)` の `dsize` は **`(幅W, 高さH)` 順**であり、numpy の `shape=(高さH, 幅W)` とは逆になっています。たとえば `(200, 100)` を渡すと「幅200・高さ100」、つまり `shape` は `(100, 200, 3)` になります。ここを取り違えると、縦横が入れ替わるか、あるいはエラーになります。対策として「`shape` は H が先、`dsize` は W が先」と声に出して覚えてしまいましょう。なお、倍率で指定したいときは `dsize=None` にして `fx`・`fy` を使います。

2つ目は**補間方法**です。拡大・縮小の際には、存在しない画素を周囲から推定（補間）します。経験則は明快で、**縮小は `cv2.INTER_AREA`**（複数画素を平均するのでモアレやジャギーが出にくい）、**拡大は `cv2.INTER_CUBIC`（高品質）か `cv2.INTER_LINEAR`（高速）**を選ぶのが定石です。一方 `cv2.INTER_NEAREST`（最近傍）は最速ですが、拡大するとカクカクのドット絵になり、縮小では情報を取りこぼします。補間を指定しないと画質が無駄に落ちてしまうので、用途に応じて明示しましょう。

| 補間フラグ | 向く場面 | 特徴 |
| --- | --- | --- |
| `cv2.INTER_AREA` | **縮小** | 領域平均。モアレ・ジャギーが出にくい。縮小の第一候補 |
| `cv2.INTER_CUBIC` | **拡大**（高品質） | 4×4 の三次補間。滑らかだがやや重い |
| `cv2.INTER_LINEAR` | 拡大（既定・高速） | 2×2 の線形補間。速度と品質の標準的なバランス |
| `cv2.INTER_NEAREST` | マスク/ラベル画像 | 最近傍。値を作らない＝ラベルIDを壊さない用途に限る |

ただし表の最終行が示すように、セグメンテーションのラベル画像のような「値そのものに意味がある（中間値を作ってはいけない）」画像では、あえて `INTER_NEAREST` を選びます。`03_resize_crop_flip.py` は高周波の同心円パターンを縮小→拡大し、`03_interpolation_compare.png` に `NEAREST` と `AREA`/`CUBIC` の差を並べて出力します。拡大結果のギザギザと滑らかさの違いを、ぜひ目で確かめてください。

## 5. 反転とクロップ — `flip` と numpy スライス

反転は `cv2.flip(img, flipCode)` の一行で済みます。`flipCode` の意味は **`1`=左右反転、`0`=上下反転、`-1`=上下左右**であり、なかでも左右反転はデータ拡張（学習データの水増し）で頻出します。ただし「左右だけ」を `0`、「上下だけ」を `1` と取り違えやすいので、`1` を水平（horizontal）と結びつけて覚えておくと間違えません。

一方、クロップ（切り出し）に専用 API は不要で、**numpy のスライス**をそのまま使います。具体的には `img[y0:y1, x0:x1]` で矩形領域を取り出せます。ここでもやはり軸順が肝心で、スライスは**先に行（何行目 y）、次に列（何列目 x）の順**です。「`x`（列）が先」と思い込むと、切り出す場所がずれてしまいます。また、スライスは多くの場合コピーではなくビュー（元配列への参照）を返すため、切り出した領域を書き換えると元画像にも反映される——この点も第1回で見たとおりです。

```python
flip_h = cv2.flip(bgr, 1)      # 左右反転（データ拡張で多用）
crop = bgr[50:200, 100:300]    # y: 50→200, x: 100→300 を切り出す（[y, x] の順！）
```

このように、反転とクロップは覚えることこそ少ないものの、軸順を間違えると静かにずれた結果が返ってきます。`03_resize_crop_flip.py` を実行し、`03_flip_h.png` で左右が反転していること、そして `03_crop.png` の `shape` が `(150, 200, 3)`（高さ150・幅200）になっていることを確認してください。

## 6. 成果物 — アスペクト比保持と正方形への整形

ここまでの部品を組み合わせると、実務でそのまま使える前処理関数ができあがります。本章の成果物 `preprocess.py` には、3つの整形関数を用意しました。まず `resize_keep_aspect(img, long_side)` は、**長辺を指定値に合わせつつ縦横比を保つ**リサイズで、画像を歪ませません。内部では拡大か縮小かを判定して補間を自動で選び、計算した `(new_w, new_h)` を必ず `dsize` の `(W, H)` 順で渡します——つまり、前節の罠を関数の中に閉じ込めているわけです。

分類・検出モデルの多くは**正方形の入力**を要求しますが、単純に正方形へリサイズすると縦横比が崩れて物体が歪んでしまいます。そこで、正攻法が2つあります。ひとつ目の `resize_to_square(img, size, pad_color)` は**レターボックス**方式で、比率を保って収めたうえで、余白を `pad_color` で埋めて正方形にします（画像全体が残るが余白ができる）。もうひとつの `center_crop_square(img)` は**中央正方形クロップ**で、内接する最大の正方形を切り出します（余白は出ないが端の情報を捨てる）。どちらを選ぶかは、「全体を残したいか／歪みも余白も避けたいか」という観点で決めます。

<figure class="lec-fig"><svg viewBox="0 0 600 250" role="img" aria-label="横長画像を正方形にする2方式。レターボックスは余白を足し全体を残す。中央クロップは端を捨て内接正方形を切る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="28" y="92" width="150" height="86" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="103" cy="135" r="14" fill="#f97316"/><text x="103" y="84" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">元画像（横長）</text><line x1="178" y1="135" x2="250" y2="135" stroke="#71717a" stroke-width="2"/><line x1="250" y1="68" x2="250" y2="200" stroke="#71717a" stroke-width="2"/><line x1="250" y1="68" x2="306" y2="68" stroke="#71717a" stroke-width="2"/><polygon points="318,68 304,62 304,74" fill="#71717a"/><line x1="250" y1="200" x2="282" y2="200" stroke="#71717a" stroke-width="2"/><polygon points="294,200 280,194 280,206" fill="#71717a"/><rect x="320" y="20" width="96" height="20" fill="#71717a"/><rect x="320" y="40" width="96" height="56" fill="#dbeafe"/><rect x="320" y="96" width="96" height="20" fill="#71717a"/><circle cx="368" cy="68" r="13" fill="#f97316"/><rect x="320" y="20" width="96" height="96" fill="none" stroke="#18181b" stroke-width="1.5"/><text x="430" y="58" font-size="13" font-weight="700" fill="#18181b">レターボックス</text><text x="430" y="78" font-size="12" fill="#52525b">（全体が残る・余白）</text><rect x="296" y="150" width="24" height="96" fill="#f4f4f5" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3"/><rect x="416" y="150" width="24" height="96" fill="#f4f4f5" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3"/><rect x="320" y="150" width="96" height="96" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><circle cx="368" cy="198" r="13" fill="#f97316"/><text x="308" y="198" text-anchor="middle" font-size="11" fill="#dc2626" style="writing-mode:vertical-rl;text-orientation:upright">捨てる</text><text x="452" y="190" font-size="13" font-weight="700" fill="#18181b">中央クロップ</text><text x="452" y="210" font-size="12" fill="#52525b">（端を捨てる）</text></svg><figcaption>横長の画像を<b>正方形</b>に整える2つの正攻法です。<b>レターボックス</b>（<code>resize_to_square</code>）は比率を保って収め、上下を <code>pad_color</code> で<b>埋めて全体を残す</b>（余白が出る）。<b>中央クロップ</b>（<code>center_crop_square</code>）は<b>内接する最大の正方形を切り出す</b>（余白は出ないが端を捨てる）。どちらも歪ませない点は共通で、用途で選び分けます。</figcaption></figure>

```python
keep = resize_keep_aspect(bgr, long_side=256)            # 歪まない・比率保持
square = resize_to_square(bgr, 256, pad_color=(0, 0, 0))  # 余白で正方形・全体が残る
crop = center_crop_square(bgr)                           # 端を捨てて正方形
```

これら3つの関数は、検出・分類の章で「モデルに入れる前の整形」として再利用していきます。`03_resize_crop_flip.py` の出力 `03_keep_aspect.png`・`03_square_letterbox.png`・`03_square_centercrop.png` を見比べると、同じ画像が「歪まないが余白あり」「正方形だが端が切れる」というように、方式ごとにどう変わるかが一目で分かります。前処理は一度書いて使い回すのが鉄則なので、自分の手に馴染むこの関数群を、ぜひ一つ手元に持っておきましょう。

## 7. Pillow の幾何変換と `size=(W, H)` vs `shape=(H, W)`

OpenCV が「配列をゴリゴリ計算する」のに向くのに対し、Pillow（`PIL`）は「画像を直感的に編集する」のに向いています。実際、`Image.resize` / `crop` / `rotate` / `thumbnail` といった操作を、読みやすい API で書けます。ただし、ここで再び**軸順**が立ちはだかります。`Image.size` は **`(幅W, 高さH)`** を返し、`resize((200, 150))` も `(幅, 高さ)` の順で指定します。これは numpy の `shape=(高さH, 幅W, ch)` とは逆順なので、OpenCV と Pillow を行き来するたびに「いま幅と高さのどちらが先か」を意識しなければなりません。

さらに、Pillow には固有の注意点もいくつかあります。まず高品質な縮小には **`Image.Resampling.LANCZOS`** を使います（OpenCV の `INTER_AREA` に相当する高品質縮小です）。なお Pillow 10 以降では旧来の `Image.ANTIALIAS` 定数が削除済みのため、古い記事の `ANTIALIAS` をそのまま書くとエラーになる点に注意してください。次に `Image.rotate` は**反時計回りが正**で、`expand=True` を付けると回転後の画像が切れないよう外接箱を広げます。最後に `Image.thumbnail` は「与えた箱に収まる最大」へ比率を保って縮小しますが、**元のオブジェクトをその場で書き換える**（戻り値を返さない）点が他と異なるので、元画像を残したいなら `copy()` してから呼びましょう。

```python
pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # cv2(BGR) → PIL(RGB)
print(pil.size)                              # (W, H)  ← 幅が先！
resized = pil.resize((200, 150), Image.Resampling.LANCZOS)   # 高品質縮小
rotated = pil.rotate(30, expand=True, fillcolor=(0, 0, 0))   # 反時計回り・外接箱を拡張
thumb = pil.copy(); thumb.thumbnail((120, 120))              # その場で縮小（戻り値なし）
```

`04_exif_transpose.py` は、これらを一通り実行し、各操作後の `size` を表示します。`resize` と `crop` は指定どおりのサイズに、`rotate(expand=True)` は外接箱の分だけ大きく、そして `thumbnail` は長辺が箱に収まるサイズになる——この対応を数値で確認してください。

## 8. EXIF Orientation — スマホ写真が横倒しで読まれる罠

スマホやデジカメで撮った写真を読み込むと、**画像が90度横倒し**になることがあります。これはバグではなく、EXIF の **Orientation（向き）** という仕組みによるものです。多くのカメラは、撮影時に**画素を回転させず、向きの情報だけをメタデータ（EXIF）に記録**します。賢い画像ビューアはこの向きタグを見て自動で回して表示しますが、`cv2.imread` や素朴な `Image.open` は**画素をそのまま**読み込むため、タグを無視すれば横倒しのまま処理してしまいます。そのまま検出や認識の前処理にかけると、モデルが倒れた画像を見せられることになり、精度が落ちてしまいます。

この正規化の正攻法が、Pillow の **`ImageOps.exif_transpose`** です。これは EXIF の向き情報に従って**実際に画素を回転・反転し、向きタグを消した**画像を返してくれます。そのため、以降は誰が読んでも正しい向きになります。`cv2` には同等の標準関数が無いので、向きの正規化は Pillow に任せ、その後 numpy 経由で BGR へ変換するのが定石です。本章の成果物 `load_image_oriented(path)` は、まさにこの「Pillow で向きを直して BGR で返す」関数であり、写真を扱う前処理の入口に必ず一枚かませる価値があります。

<figure class="lec-fig"><svg viewBox="0 0 600 250" role="img" aria-label="EXIFのOrientationは画素を回さず向きタグだけ記録する。素朴にopenすると横倒し。exif_transposeは画素を回しタグを消して正立にする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="40" y="70" width="86" height="104" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><line x1="62" y1="122" x2="98" y2="122" stroke="#1d4ed8" stroke-width="7"/><polygon points="112,122 96,111 96,133" fill="#1d4ed8"/><rect x="44" y="148" width="78" height="22" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="83" y="163" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">Orient=6</text><text x="83" y="60" text-anchor="middle" font-size="12" font-weight="700" fill="#18181b">保存画素 ＋ タグ</text><line x1="126" y1="122" x2="200" y2="122" stroke="#71717a" stroke-width="2"/><line x1="200" y1="60" x2="200" y2="190" stroke="#71717a" stroke-width="2"/><line x1="200" y1="60" x2="256" y2="60" stroke="#71717a" stroke-width="2"/><polygon points="268,60 254,54 254,66" fill="#71717a"/><line x1="200" y1="190" x2="276" y2="190" stroke="#71717a" stroke-width="2"/><polygon points="288,190 274,184 274,196" fill="#71717a"/><rect x="272" y="28" width="110" height="72" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><line x1="300" y1="64" x2="342" y2="64" stroke="#1d4ed8" stroke-width="7"/><polygon points="356,64 340,53 340,75" fill="#1d4ed8"/><line x1="392" y1="44" x2="414" y2="66" stroke="#dc2626" stroke-width="3"/><line x1="414" y1="44" x2="392" y2="66" stroke="#dc2626" stroke-width="3"/><text x="327" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">素朴に open → 横倒し</text><rect x="292" y="150" width="82" height="94" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><line x1="333" y1="232" x2="333" y2="192" stroke="#1d4ed8" stroke-width="7"/><polygon points="333,178 322,194 344,194" fill="#1d4ed8"/><polyline points="390,202 401,214 423,186" fill="none" stroke="#16a34a" stroke-width="3"/><text x="333" y="142" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">exif_transpose → 正立</text></svg><figcaption>多くのカメラは撮影時に<b>画素を回さず、向き情報（EXIF Orientation）だけ</b>を記録します。<code>cv2.imread</code> や素朴な <code>Image.open</code> はタグを無視して<b>横倒し</b>のまま読み（✕）、検出・認識の精度が落ちます。<code>ImageOps.exif_transpose</code> は<b>画素を実際に回転し、向きタグを消した</b>正しい向きの画像を返します（✓）。写真の前処理は、まずこの正規化から。</figcaption></figure>

```python
from PIL import Image, ImageOps
naive = Image.open(path)                            # 画素は回らない（横倒しのまま）
fixed = ImageOps.exif_transpose(Image.open(path))  # 画素を実際に回し、向きタグを消す
# 以降 numpy/cv2 へ: np.asarray(fixed.convert("RGB")) → cv2.cvtColor(..., RGB2BGR)
```

`04_exif_transpose.py` は、わざと Orientation=6（時計回り90度の指示）を付けた JPEG を作り、素朴に開いた場合と `exif_transpose` した場合の両方を保存します。`04_exif_naive.png`（横倒し）と `04_exif_fixed.png`（正しい向き）を見比べ、`size` が `(400, 300)` から `(300, 400)` へ入れ替わり、あわせて向きタグが消えることを確認してください。そして「写真の前処理は、まず EXIF 正規化から」を手癖にしてしまいましょう。

## 9. PIL ↔ numpy ↔ cv2 の相互変換

実務では、OpenCV と Pillow を頻繁に行き来します。その橋渡しの基本は2本で、**PIL → numpy が `np.asarray(pil)`、numpy → PIL が `Image.fromarray(arr)`** です。ただし第1回で強調したとおり、**Pillow と matplotlib は RGB、OpenCV は BGR**を採用しています。そのため境界をまたぐたびに `cv2.cvtColor(..., COLOR_BGR2RGB / COLOR_RGB2BGR)` で色順を変換しないと、赤と青が入れ替わった画像になってしまいます。ここは「OpenCV だけが BGR の仲間外れ」と覚えておけば混乱しません。

変換が正しく閉じているかを確かめる最良の方法が、**ラウンドトリップ**です。`cv2(BGR) → RGB → PIL → numpy → cv2(BGR)` と一周して元の配列に戻れれば、3者の橋渡しを完全に理解できた証拠になります。下のコードを実行し、`np.array_equal` が `True` を返すことを確認してください。なお `np.asarray` は多くの場合コピーを作らない読み取り専用ビューを返すので、書き換えたい場合は `np.array(pil)`（コピー）を使う——この点も合わせて押さえておきましょう。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="cv2とPILとnumpyのラウンドトリップ。BGRをRGBに変換しPIL経由でnumpyにし、RGB2BGRで戻すと元のBGRと一致する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">cv2 ↔ PIL ↔ numpy のラウンドトリップ</text><text x="161" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">BGR2RGB</text><text x="331" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">fromarray</text><text x="501" y="104" text-anchor="middle" font-size="10.5" fill="#3f3f46">asarray</text><rect x="14" y="80" width="124" height="66" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="76" y="110" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">cv2 (BGR)</text><text x="76" y="130" text-anchor="middle" font-size="11.5" fill="#52525b">元画像</text><rect x="184" y="80" width="124" height="66" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="246" y="110" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">RGB 配列</text><text x="246" y="130" text-anchor="middle" font-size="11.5" fill="#52525b">RGB 順</text><rect x="354" y="80" width="124" height="66" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="416" y="110" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">PIL</text><text x="416" y="130" text-anchor="middle" font-size="11.5" fill="#52525b">Image</text><rect x="524" y="80" width="124" height="66" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="2"/><text x="586" y="110" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">numpy 配列</text><text x="586" y="130" text-anchor="middle" font-size="11.5" fill="#52525b">(H, W, 3)</text><line x1="138" y1="113" x2="178" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="184,113 174,108 174,118" fill="#71717a"/><line x1="308" y1="113" x2="348" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="354,113 344,108 344,118" fill="#71717a"/><line x1="478" y1="113" x2="518" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="524,113 514,108 514,118" fill="#71717a"/><line x1="586" y1="146" x2="586" y2="210" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/><line x1="586" y1="210" x2="76" y2="210" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/><line x1="76" y1="210" x2="76" y2="152" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/><polygon points="76,146 71,156 81,156" fill="#16a34a"/><text x="331" y="196" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">RGB2BGR で戻すと元の cv2(BGR) と一致（array_equal）</text></svg><figcaption><b>cv2 ↔ PIL ↔ numpy</b> の<b>ラウンドトリップ</b>です。<code>cv2(BGR)</code> を <code>cvtColor(BGR2RGB)</code> で RGB にし、<code>Image.fromarray</code> で PIL へ、<code>np.asarray</code> で numpy へ、最後に <code>cvtColor(RGB2BGR)</code> で <code>cv2(BGR)</code> に戻します。<b>境界をまたぐたびに色順を変換</b>すれば、一周して <code>np.array_equal</code> が <b>True</b>（元と一致）になります。これが3者の橋渡しを正しく理解できた証拠です。</figcaption></figure>

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # cv2(BGR) → RGB
arr = np.asarray(Image.fromarray(rgb))        # RGB → PIL → numpy（読み取り専用ビュー）
back = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)   # numpy → cv2(BGR) へ戻す
assert np.array_equal(bgr, back)              # 1周して一致 = 変換が正しく閉じた証拠
```

この一致さえ取れれば、どんな組み合わせでも色は崩れません。鍵は常に同じで、**境界をまたぐたびに色順を変換する**——結局はこの一点に尽きます。`04_exif_transpose.py` の末尾でこのラウンドトリップを検証しているので、出力ログに `True` が出ることを確認してください。

## 10. このモジュールの構成（スクリプト一覧）

各スクリプトは単一責務で書かれており、上から順に読めば理解が積み上がるよう並んでいます。いずれも結果を `outputs/03_image_transforms/` に保存し、画面表示には依存しません（headless 安全）。共通処理（合成画像生成・日本語パス対応 I/O・出力先管理）は `cv_helpers.py` に、本章の成果物である再利用前処理関数は `preprocess.py` にまとめてあり、各スクリプトはそれらを import して使います。

| ファイル | 役割（単一責務） |
| --- | --- |
| `cv_helpers.py` | 合成画像生成（色シーン／高周波パターン）・`imread/imwrite_unicode`・出力先。各スクリプトの道具箱 |
| `preprocess.py` | **成果物**。HSV色抽出・アスペクト比保持/正方形整形・EXIF正規化の再利用関数群 |
| `01_colorspace_hsv_mask.py` | Gray/HSV 変換、HSV の H=0-179 スケール、`inRange` 色マスク（赤の2区間合成）、`split`/`merge` |
| `02_drawing.py` | `line`/`rectangle`/`circle`/`polylines`/`putText`、LINE_AA、検出可視化（枠＋ラベル） |
| `03_resize_crop_flip.py` | `resize` の `dsize=(W,H)` 罠と補間の使い分け、`flip`、スライスクロップ、整形関数の実演 |
| `04_exif_transpose.py` | PIL の `resize/crop/rotate/thumbnail`、`size` vs `shape`、EXIF `exif_transpose`、相互変換 |
| `mini_project.py` | **章末ミニプロジェクト**。色検出 → bbox 可視化 → 正方形整形 → EXIF/相互変換検証を1本に統合し、図と JSON を出力 |
| `exercises.py` | TODO 形式の演習10問（易→難・自己採点ランナー付き。`SHOW_SOLUTION=1` で模範解答） |
| `exercises_solutions.py` | 演習の模範解答（実行すると全10問 PASS。答え合わせ・教材検証用） |

なお `cv_helpers.py` と `preprocess.py` は、「読み物」ではなく「再利用する道具」です。とくに `preprocess.py` は本章のゴールそのものなので、最初に一読してから 01 へ進むと、各スクリプトが何を import しているのかが腑に落ちます。

## 11. 動かし方

このモジュールは `numpy` / `opencv-python-headless` / `pillow` / `matplotlib` だけに依存しており、GPU もネット接続も不要です。サンプル画像が無くても合成画像が自動生成されるため、いきなり実行できます（`data/sample.jpg` を置けば、そちらが優先して使われます）。それでは、プロジェクトルートで以下を順に実行してください。

```bash
# 依存をインストール（初回のみ）
uv sync

# 各スクリプトを実行（結果は outputs/03_image_transforms/ に保存される）
uv run python lectures/03_image_transforms/01_colorspace_hsv_mask.py
uv run python lectures/03_image_transforms/02_drawing.py
uv run python lectures/03_image_transforms/03_resize_crop_flip.py
uv run python lectures/03_image_transforms/04_exif_transpose.py

# 章末ミニプロジェクト: この回の要素を統合した総合課題（図＋JSON を出力）
uv run python lectures/03_image_transforms/mini_project.py

# 演習: まずは TODO を自分で埋める（最初は全部 FAIL）
uv run python lectures/03_image_transforms/exercises.py
# どうしても分からない時だけ、模範解答の挙動を見る
SHOW_SOLUTION=1 uv run python lectures/03_image_transforms/exercises.py
# 模範解答そのもの（実行すると全10問 PASS）
uv run python lectures/03_image_transforms/exercises_solutions.py
```

実行を終えたら `outputs/03_image_transforms/` の画像を開き、解説と照らし合わせてください。とりわけ `01_mask_red.png`（赤の2区間抽出）、`03_interpolation_compare.png`（補間の差）、`03_square_letterbox.png` と `03_square_centercrop.png`（正方形整形の2方式）、`04_exif_naive.png` と `04_exif_fixed.png`（EXIF 向きの違い）を見比べると、本章の要点が一目で腑に落ちます。

## 12. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」の形で一覧にまとめます。実装中に詰まったら、まずここを見てください。多くの不具合は、結局この章で扱った数個の罠に集約されます。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| リサイズしたら縦横が入れ替わった | `dsize` を `(H, W)` 順で渡した | `cv2.resize` の `dsize` は `(W, H)` 順。`shape` と逆と覚える |
| 色マスクで赤の半分しか取れない | 赤が色相環の `0/179` をまたぐ | `[0,10]` と `[170,179]` の2マスクを `bitwise_or` で合成 |
| `inRange` で背景まで白くなる | `S`・`V` の下限が低すぎる | 彩度・明度の下限を上げて低彩度のグレーを除外 |
| HSV の色域指定が効かない | `H` に 0-360 の角度を入れた | OpenCV の `H` は `0-179`。角度を半分にする |
| クロップ位置がずれる | スライスを `[x, y]` 順で書いた | スライスは `[y0:y1, x0:x1]`（行 y が先、列 x が後） |
| 描画した図形の位置がずれる | 座標を `(y, x)` で渡した | `cv2` の描画系は座標が `(x, y)` 順 |
| スマホ写真が横倒しで処理される | EXIF Orientation を無視した | `ImageOps.exif_transpose` で向きを正規化してから使う |
| PIL に渡したら赤青が入れ替わった | BGR を RGB として渡した | 境界で `cv2.cvtColor(BGR2RGB)` してから `Image.fromarray` |

この8項目こそが、第3回でつまずく原因のほぼ全てです。逆に言えば、この8つを自分の言葉で説明でき、かつ回避コードを書けるようになれば、この章のゴールに到達したといえます。

## 13. まとめ

この章では、画像という配列を目的に合わせて作り替える基本——色空間変換（Gray の次元減・HSV の `H=0-179`）、`inRange` による色マスク（赤の2区間合成）、図形とテキストの描画と検出可視化、`dsize=(W,H)` の罠と補間の使い分け、反転とスライスクロップ、そして成果物としてのアスペクト比保持・正方形整形・EXIF 正規化——を、すべて「自分で再現し回避できる」レベルで扱ってきました。そして、これらに共通する教訓は「**軸順とスケールの食い違いを、境界ごとに意識的に変換する**」ことに尽きます。

ここで作った `preprocess.py` は、以降の検出・分類・セグメンテーションの章で、前処理としてそのまま使い回せます。まずは演習を自力で全問 PASS させ、`dsize` の順序・HSV のスケール・EXIF 正規化を手に馴染ませてから次へ進んでください。次回は、この章で作ったマスクや前処理を土台に、フィルタ・エッジ・閾値・モルフォロジー・輪郭抽出・ワーピングへと進んでいきます。

---

## 🛠 章末ミニプロジェクト — 色で物体を検出して「可視化 → モデル入力前処理」まで一気通貫

ここまで、各部品はバラバラに学んできました。最後に、それらを**1 本のパイプライン**へ束ね、この章の技能が「単独で使える」だけでなく「つながって動く」ことを体感します。題材は **HSV 色域による簡易物体検出**です。これは、後段の検出・分類の章でそのまま雛形になる「前処理 ＋ 結果の可視化」の最小完成形にあたります。実装は `mini_project.py` にあり、実行すると図と総合レポート（JSON）が `outputs/03_image_transforms/` に出力されます。

パイプラインは次の6段からなり、この章で扱った要素を順に踏んでいきます。**(1) 色で抜く**——`preprocess.extract_color` で赤・黄・緑・青それぞれの HSV マスクを作る（赤は `0/179` の2区間合成）。**(2) bbox を取る**——マスクの白画素に `np.where` をかけ、`(ys, xs)` の**軸順 `[y, x]`** から外接矩形 `(x0, y0, x1, y1)` を求める。**(3) 可視化する**——`draw_label_box` で「枠＋ラベル帯＋白文字」を重ねる（座標は `(x, y)`、色は BGR）。**(4) 整形する**——各検出物体を切り出し（スライス `[y0:y1, x0:x1]`）、`resize_to_square` で歪ませずに正方形のモデル入力へ整える。**(5) EXIF 検証**——`Orientation=6` を付けた JPEG を作り、`exif_transpose` で `(W,H)` が入れ替わり向きタグが消えることを確認する。**(6) 相互変換**——`cv2(BGR) → PIL(RGB) → numpy → cv2(BGR)` のラウンドトリップ一致を確認する。

<figure class="lec-fig"><svg viewBox="0 0 600 240" role="img" aria-label="色検出ミニプロジェクトの6段パイプライン。色で抜く→bbox→可視化→整形→EXIF検証→相互変換の順に処理が流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">色検出ミニプロジェクトの 6 段パイプライン</text><rect x="24" y="58" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="106" y="84" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 色で抜く</text><text x="106" y="104" text-anchor="middle" font-size="11.5" fill="#52525b">extract_color (HSV)</text><rect x="218" y="58" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="300" y="84" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② bbox を取る</text><text x="300" y="104" text-anchor="middle" font-size="11.5" fill="#52525b">np.where → 外接矩形</text><rect x="412" y="58" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="494" y="84" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 可視化</text><text x="494" y="104" text-anchor="middle" font-size="11.5" fill="#52525b">枠＋ラベル帯＋文字</text><line x1="188" y1="87" x2="212" y2="87" stroke="#ea580c" stroke-width="2.2"/><polygon points="218,87 208,82 208,92" fill="#ea580c"/><line x1="382" y1="87" x2="406" y2="87" stroke="#ea580c" stroke-width="2.2"/><polygon points="412,87 402,82 402,92" fill="#ea580c"/><line x1="494" y1="116" x2="494" y2="150" stroke="#ea580c" stroke-width="2.2"/><polygon points="494,156 489,146 499,146" fill="#ea580c"/><rect x="412" y="158" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="494" y="184" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">④ 整形</text><text x="494" y="204" text-anchor="middle" font-size="11.5" fill="#52525b">resize_to_square</text><rect x="218" y="158" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="300" y="184" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">⑤ EXIF 検証</text><text x="300" y="204" text-anchor="middle" font-size="11.5" fill="#52525b">exif_transpose</text><rect x="24" y="158" width="164" height="58" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="106" y="184" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">⑥ 相互変換</text><text x="106" y="204" text-anchor="middle" font-size="11.5" fill="#52525b">round-trip 一致</text><line x1="412" y1="187" x2="388" y2="187" stroke="#ea580c" stroke-width="2.2"/><polygon points="382,187 392,182 392,192" fill="#ea580c"/><line x1="218" y1="187" x2="194" y2="187" stroke="#ea580c" stroke-width="2.2"/><polygon points="188,187 198,182 198,192" fill="#ea580c"/></svg><figcaption><b>HSV 色検出</b>のミニプロジェクトは <b>6 段のパイプライン</b>です。<b>① 色で抜く</b>（<code>extract_color</code>、赤は <code>0/179</code> の2区間合成）→ <b>② bbox を取る</b>（<code>np.where</code> の <code>[y, x]</code> から外接矩形）→ <b>③ 可視化</b>（枠＋ラベル帯＋白文字、座標は <code>(x, y)</code>）→ <b>④ 整形</b>（<code>resize_to_square</code> で歪ませず正方形へ）→ <b>⑤ EXIF 検証</b>（<code>exif_transpose</code>）→ <b>⑥ 相互変換</b>（<code>cv2 ↔ PIL ↔ numpy</code> のラウンドトリップ一致）。部品が<b>つながって動く</b>、前処理＋可視化の最小完成形です。</figcaption></figure>

```bash
uv run python lectures/03_image_transforms/mini_project.py
# → mini_detections.png（色検出＋ラベル）、mini_summary.png（入力→検出→前処理の総合パネル）、
#    mini_exif_naive.png / mini_exif_fixed.png（EXIF 向きの違い）、mini_report.json（機械可読の総合レポート）
```

`mini_report.json` には、検出した各色の `bbox` と画素数、3方式の前処理 shape、EXIF 正規化の成否、そして相互変換の一致が、機械可読の形でまとまります。さらに**発展課題**として、(a) `extract_color` の `s_min`/`v_min` を変えると背景グレーの拾い方がどう変わるか、(b) `resize_to_square` を `center_crop_square` に差し替えると検出パッチがどう変わるか、(c) `data/sample.jpg` に自分の写真を置いて実画像でも検出が動くか、をぜひ試してみてください。

## ✅ 到達チェックリスト

この章を「できた」と言えるための基準です。実際に手を動かして、できる／説明できるの両方を確認してください。

- [ ] **できる**: `cv2.cvtColor` で BGR を Gray／HSV に変換でき、Gray の `shape` が `(H, W)` の2次元になる（次元が1つ減る）ことを自分で確かめられる。
- [ ] **できる**: `inRange` で色マスクを作れる。とくに**赤**は `[0,10]` と `[170,179]` の2区間を `bitwise_or` で合成できる。
- [ ] **できる**: `rectangle`／`putText` と `getTextSize` を組み合わせ、「枠＋ラベル帯＋白文字」を自分で描ける。
- [ ] **できる**: `cv2.resize` を `dsize=(W, H)` の正しい順で呼び、縮小に `INTER_AREA`・拡大に `INTER_CUBIC` を選べる。
- [ ] **できる**: numpy スライス `[y0:y1, x0:x1]` でクロップでき、`flip` の `1/0/-1` を取り違えない。
- [ ] **できる**: アスペクト比保持リサイズ・正方形レターボックス・中央正方形クロップを書ける（`preprocess.py` をそらで再現できる）。
- [ ] **できる**: `ImageOps.exif_transpose` で写真の向きを正規化し、`cv2 ↔ PIL ↔ numpy` をラウンドトリップで色を崩さず往復できる。
- [ ] **説明できる**: なぜ OpenCV の HSV は `H=0-179` なのか（`uint8` に色相を収めるため）。
- [ ] **説明できる**: `dsize=(W,H)` と `shape=(H,W)`、PIL の `size=(W,H)` という**軸順の食い違い**がどこで効くか。
- [ ] **説明できる**: EXIF Orientation とは何で、放置すると検出・認識精度がなぜ落ちるか。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/03_image_transforms/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. BGR を HSV に変換し、`[lower_hsv, upper_hsv]` の色域を抜き出す2値マスク（`(H, W)` の uint8）を返す（`ex1_color_mask` の TODO）。
2. 画像を「幅 width × 高さ height」にリサイズして返す。`cv2.resize` の `dsize` は `(W, H)` 順である点が要（`ex2_resize_to` の TODO）。
3. 矩形 `(x0, y0)-(x1, y1)` の領域を numpy スライス `[y0:y1, x0:x1]` で切り出して返す（`ex3_crop` の TODO）。
4. `cv2.flip` で画像を反転して返す。mode は 1=左右・0=上下・-1=両方（`ex4_flip` の TODO）。
5. BGR をグレースケール化して返す。返り値の shape は `(H, W)` の2次元になる（`ex5_to_gray` の TODO）。
6. 色相環の `0/179` をまたぐ「赤」の2値マスクを作る。`[0,10]` と `[170,179]` の2区間を `cv2.bitwise_or` で合成して返す（`ex6_red_mask_wraparound` の TODO）。
7. アスペクト比を保ったまま `size×size` の正方形に収める（黒余白・中央配置のレターボックス）。返り値の shape は `(size, size, 3)`（`ex7_letterbox_square` の TODO）。
8. 長辺が `long_side` になるよう、アスペクト比を保ってリサイズして返す（縮小は `INTER_AREA`・拡大は `INTER_CUBIC`）（`ex8_keep_aspect` の TODO）。
9. 画像中央から「内接する最大の正方形」を切り出して返す。`side = min(H, W)` で中央クロップする（`ex9_center_crop_square` の TODO）。
10. 2値マスク（0/255）から白画素の外接矩形 `(x0, y0, x1, y1)` を返す。白画素が無ければ `None` を返す（`ex10_bbox_from_mask` の TODO）。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずはここを見てください。この章のバグは、ほぼ「軸順」「数値スケール」「色順」のどれかに集約されます。

- **Q. 色マスクで赤が半分しか取れない。** → 赤は色相環の `0/179` をまたぎます。`[0,10]` の1区間だけでは取りこぼすので、`[170,179]` のマスクも作って `cv2.bitwise_or` で合成してください（`mini_project.py` の検出が赤も拾えるのはこのためです）。
- **Q. `inRange` で背景まで白くなる。** → `S`・`V` の下限が低すぎます。背景の灰色は**彩度が低い**ので、`s_min`/`v_min` を `80/60` 程度まで上げると低彩度のグレーを弾けます。
- **Q. HSV の色域指定がまったく効かない。** → 一般的な色相環の角度（0–360）をそのまま入れていませんか。OpenCV の `H` は `0-179` なので、**角度を半分**にした値を使います。
- **Q. リサイズしたら縦横が入れ替わった／エラーになる。** → `cv2.resize` の `dsize` は `(W, H)` 順です。`shape=(H, W)` と逆と覚え、計算した `(new_w, new_h)` を必ずこの順で渡します。
- **Q. クロップ位置や描画位置がずれる。** → numpy スライスは `[y0:y1, x0:x1]`（行 y が先、列 x が後）、`cv2` の描画系は座標 `(x, y)`。**スライスと描画で x/y の順が逆**なのを混同していないか確認してください。
- **Q. グレースケール画像を他の画像と並べると形が合わずに落ちる。** → Gray は `(H, W)` の2次元でチャンネル軸がありません。連結・重ね合わせの前に `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` で3チャンネルへ戻します。
- **Q. PIL に渡したら赤と青が入れ替わった。** → OpenCV だけが BGR です。境界で `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` してから `Image.fromarray` に渡してください。
- **Q. スマホ写真が横倒しで処理される。** → EXIF Orientation を無視しています。`ImageOps.exif_transpose` で画素を実際に回してから使います（`preprocess.load_image_oriented` がこれを内側で行います）。
- **Q. `Image.ANTIALIAS` でエラーになる。** → Pillow 10 以降で削除済みです。高品質縮小は `Image.Resampling.LANCZOS` を使います。
- **デバッグの定石**: 形が合わない・色が変なときは、まず `print(img.shape, img.dtype)` を挟む。`shape` の長さで Gray/カラーを、先頭2要素の並びで軸順を、一目で確認できます。

## 🚀 発展トピック・参考

この章の先に広がるテーマです。興味のある方向へ掘り進めてください。

- **`filter2D` と任意カーネル畳み込み**: 平滑化・シャープ化・エッジ抽出はすべて「カーネルとの畳み込み」で統一的に書けます（次章 `04_filtering_edges_morphology` で詳説）。
- **輪郭抽出による物体検出**: 本章のミニプロジェクトは bbox を `np.where` で素朴に求めましたが、`cv2.findContours` を使えば面積・周囲長・凸包・近似多角形まで形状解析できます（次章）。OpenCV 4 系では返り値が `(contours, hierarchy)` の**2つ**である点に注意。
- **`Lab`／`YCrCb` 色空間**: HSV 以外にも、知覚的な色差に強い `Lab`、輝度と色差を分ける `YCrCb` があり、肌色検出やコントラスト補正で使い分けます。
- **アフィン・透視変換（ワーピング）**: 回転・せん断・台形補正は `getRotationMatrix2D`／`getPerspectiveTransform` ＋ `warpAffine`／`warpPerspective` で行列として扱います（次章で書類のまっすぐ化を実装）。
- **データ拡張ライブラリ**: 本章の幾何変換は `albumentations` や `torchvision.transforms.v2` が高速・宣言的に提供します（第2回・第12回）。まず素の OpenCV で「中で何が起きているか」を理解してからライブラリに移ると、パラメータの意味が腑に落ちます。
- 公式ドキュメント: [OpenCV Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html) ／ [Pillow Handbook](https://pillow.readthedocs.io/en/stable/handbook/index.html) ／ [Pillow EXIF/ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html)

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.x ／ opencv-python-headless 4.13（`cv2` 4.13.0）／ Pillow 12.2 ／ matplotlib 3.10 ／（深層トラックで使う torch は 2.12+cpu。本章では未使用）
