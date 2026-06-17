# 第43回 色空間と画像の調整 — 明るさ・彩度・色相・コントラスト・ガンマ・ホワイトバランス

> トラック: 画像の基礎 ／ レベル: 初級〜中級 ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加グループ不要）
> 前提モジュール: `03_image_transforms`（`cvtColor`／`inRange`／BGR の軸順は前提知識として使います）

## 🎯 この章のゴール

第3回では「`cv2.cvtColor` で BGR から HSV やグレースケールへ変換でき、`inRange` で色マスクが作れる」ところまで到達しました。本章はそこを足場として、**画像が持つ「色」と「明るさ」を別々の軸として捉え、片方だけを意図して動かす**技術を体系化します。というのも、RGB は「赤・緑・青の強さ」をひとまとめに持つため、「明るさだけ」「色味だけ」を変えるのが苦手だからです。そこで HSV・Lab・YCrCb といった「**輝度と色を分離した色空間**」へ移り、明るさ・彩度・色相・コントラストという軸を一つずつ操作していきます。

この章の隠れた主題は、第3回と同じく「**スケールと前提の食い違いを毎回意識的に変換する**」ことです。たとえば OpenCV の HSV は色相 `H` が `0-179`（一般的な `0-360` ではない）であり、8bit の Lab では本来 `0-100` の明度 `L` が `0-255` にスケールされ、`a*`/`b*` は `±127` を `128` 中心へずらして格納されています。さらに厄介なことに、同じ `cvtColor` でも入力が `uint8` か `float32` かでスケールが変わる（float なら `H=0-360`、Lab は本来値）という罠もあります。そして知覚的色差 ΔE は、**float の L\*a\*b\*** で測らねば意味を成しません。本章ではこうした癖を、単なる知識ではなく「自分で再現し、回避コードを書ける」レベルへ落とし込みます。

到達点を一言でいえば、**色かぶり・露出不足・低コントラストで撮れた一枚を、ホワイトバランス→露出(ガンマ)→局所コントラスト(CLAHE)→彩度の順で補正し、その効果を 輝度ヒストグラム・平均彩度・平均ΔE で定量評価できる**こと。成果物は `mini_project.py`（補正パイプライン）と、再利用関数を束ねた `color_helpers.py` です。

---

## 1. 色空間の地図 — RGB/HSV/HLS/Lab/YCrCb のスケール

色空間とは「色をどんな数値の組で表すか」の取り決めです。RGB（OpenCV では BGR）は素直ですが、`(120, 120, 200)` という値を見て「やや明るい赤」と即答するのは難しいものです。そこで、人が色を語るときの「色味・鮮やかさ・明るさ」に近い軸へ移したのが **HSV**（色相・彩度・明度）や **HLS** であり、印刷・知覚の分野で重宝するのが **Lab**（明度 `L*` と色度 `a*`/`b*`）、放送・JPEG 圧縮で使われるのが **YCrCb**（輝度 `Y` と色差 `Cr`/`Cb`）です。これらに共通する発想は「**明るさを表す1軸と、色味を表す残りの軸を分ける**」ことであり、これによって「明るさだけ」「色だけ」を独立に触れるようになります。

<figure class="lec-fig"><svg viewBox="0 0 660 240" role="img" aria-label="RGBは色と明るさが混在するが、HSV・Lab・YCrCbは輝度の1軸と色の2軸に分離される" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">明るさ(輝度)の1軸と、色の2軸に分ける</text><text x="103" y="60" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">RGB / BGR</text><rect x="53" y="70" width="100" height="32" fill="#e4e4e7" stroke="#71717a" stroke-width="1.5"/><rect x="53" y="104" width="100" height="32" fill="#d4d4d8" stroke="#71717a" stroke-width="1.5"/><rect x="53" y="138" width="100" height="32" fill="#71717a" stroke="#52525b" stroke-width="1.5"/><text x="103" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">B</text><text x="103" y="125" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">G</text><text x="103" y="159" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">R</text><text x="103" y="196" text-anchor="middle" font-size="12.5" fill="#52525b">色と明るさが混在</text><line x1="166" y1="120" x2="226" y2="120" stroke="#c2410c" stroke-width="2.5"/><polygon points="234,120 222,114 222,126" fill="#c2410c"/><text x="198" y="108" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">分離</text><text x="296" y="70" text-anchor="end" font-size="13" font-weight="700" fill="#3f3f46">HSV</text><rect x="304" y="50" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="360" y="50" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="416" y="50" width="52" height="30" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="330" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">H</text><text x="386" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">S</text><text x="442" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">V</text><text x="296" y="124" text-anchor="end" font-size="13" font-weight="700" fill="#3f3f46">Lab</text><rect x="304" y="104" width="52" height="30" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="360" y="104" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="416" y="104" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="330" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">L</text><text x="386" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">a</text><text x="442" y="124" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">b</text><text x="296" y="178" text-anchor="end" font-size="13" font-weight="700" fill="#3f3f46">YCrCb</text><rect x="304" y="158" width="52" height="30" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><rect x="360" y="158" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><rect x="416" y="158" width="52" height="30" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="330" y="178" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">Y</text><text x="386" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">Cr</text><text x="442" y="178" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">Cb</text></svg><figcaption>RGB(BGR)は <b>B・G・R のどれにも明るさと色が混ざる</b>ため、片方だけを動かせません。<b>HSV・Lab・YCrCb</b> はいずれも<b>輝度を表す1軸</b>(オレンジ: <code>V</code>/<code>L</code>/<code>Y</code>)と<b>色を表す2軸</b>(青: <code>H,S</code>/<code>a,b</code>/<code>Cr,Cb</code>)に分かれており、「明るさだけ」「色だけ」を独立に操作できます。これが本章を貫く原則です。</figcaption></figure>

ここで必ず押さえておきたいのが **8bit でのスケールの癖**です。各軸を `uint8`（0-255）に収めるため、OpenCV は独自のスケーリングを施します。下表のとおり、`H` は `0-179`（角度 0-360 の半分）に圧縮され、Lab は `L` も含めた全チャンネルが `0-255` に押し込まれます。さらに `a*`/`b*` は符号付き（緑↔赤、青↔黄）なので、`128` を中性として格納されます。01 の `print_scale_table()` は純色を各色空間へ変換し、これらの数値を一覧で見せてくれます。

| 色空間 | チャンネルと値域(8bit, uint8) | 役割 |
|---|---|---|
| BGR/RGB | B,G,R: 0-255 | 各原色の強さ |
| HSV | H: **0-179**, S: 0-255, V: 0-255 | 色相で色を選ぶ |
| HLS | H: **0-179**, L: 0-255, S: 0-255 | HSV と H 共通・明度定義が違う |
| Lab | **L: 0-255**, a: 0-255, b: 0-255 | 本来 L:0-100, a/b:±127→128中心。知覚的 |
| YCrCb | Y: 0-255, Cr: 0-255, Cb: 0-255 | 輝度Yと色差を分離。WB/平坦化向き |

さらに重要なのが **dtype 依存のスケール変化**と**往復の非可逆性**です。`float32`（0-1）を入力すると、`cvtColor` は「本来の」スケールを返します（`H=0-360`、Lab は `L:0-100, a/b:±127`）。したがって `uint8` と `float` のスケールを混ぜると、領域抽出も ΔE も静かに壊れてしまいます。また `BGR→HSV→BGR` のような往復変換は完全可逆ではなく、とくに 8bit Lab には量子化誤差が乗ります（01 で実測すると Lab の `max|diff|` が最大になります）。最後にもう一点、matplotlib・PIL は **RGB 前提**なので、BGR をそのまま `imshow` に渡すと赤と青が入れ替わってしまいます。

```python
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)     # uint8: H=0-179, S/V=0-255
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)     # uint8: L/a/b すべて 0-255
ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)   # uint8: Y/Cr/Cb 0-255
hsv_f = cv2.cvtColor((bgr/255).astype(np.float32), cv2.COLOR_BGR2HSV)  # float: H=0-360
plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # matplotlib は RGB 前提
```

## 2. 5つの「軸」 — 明るさ・彩度・色相・明度・コントラスト

調整を語る前に、まず軸の言葉を揃えておきましょう。**明るさ（輝度 / luminance）**は「どれだけ光っているか」で、YCrCb の `Y` や Lab の `L` が代表。**色相（hue）**は「赤・緑・青…という色味」で HSV の `H`。**彩度（saturation）**は「鮮やかさ／灰色からの遠さ」で HSV の `S`。**明度（value）**は HSV の `V` で「その色のピクセルの最大原色値」。そして **コントラスト（contrast）**は単一画素の属性ではなく「画像全体の明暗の開き（輝度分布の広がり）」を指します。

ただし、これらの軸は互いに独立ではありません。たとえば RGB を一様に持ち上げると、明るさと一緒に彩度や見かけの色まで動いてしまいます。そこで HSV/Lab/YCrCb に移る最大の利点は、「**`S` だけ」「`H` だけ」「`Y` だけ**」と軸を1つだけ動かせる点にあります。彩度を上げたいのに RGB をいじると明るさまで変わってしまいますが、HSV の `S` を `1.5` 倍すれば、明度 `V` は不変のまま鮮やかさだけが増します（03 で `mean_V` が一定であることを数値で確認します）。

コントラストの代理量として、本章では**輝度の標準偏差** `std(Y)` を使います。`std` が大きいほど明暗がはっきりした高コントラスト、小さいほど眠い低コントラストです。同様に、明るさは `mean(Y)`、鮮やかさは `mean(S)` で測れます。このように「動かしたい軸の指標」をあらかじめ決めておくと、調整が効いているかを目視ではなく数値で語れます。これは7節の定量評価につながる構えです。

## 3. 明るさ・コントラスト・ガンマ・露出 — トーンを動かす

最も基本的な明るさ／コントラスト調整は**線形変換** `out = alpha * in + beta` です。ここで `beta` が明るさ（全体のオフセット）、`alpha` が傾き＝コントラスト（`1.0` で等倍、`>1` で明暗を開く）を担います。OpenCV ではこれを `cv2.convertScaleAbs(img, alpha, beta)` 一発で、しかも **飽和演算**（255 を超えたら頭打ち）で行えます。このとき `uint8` の罠を必ず体験しておいてください。素の numpy `+` は **オーバーフローでラップアラウンド**してしまい、`200 + 100` が `44` へ巻き戻ります（02 で実演）。したがって明るさ調整では、`cv2.add` / `convertScaleAbs` の飽和演算を使うのが正解です。

線形変換は暗部も明部も同じ傾きで動かすため、暗い写真を持ち上げると明部が白飛びしがちです。そこで使うのが、**非線形なガンマ補正** `out = 255 * (in/255)^(1/γ)` です。`γ>1` は暗部を大きく持ち上げつつ明部の伸びを抑え（暗所写真の救済）、逆に `γ<1` は全体を締めて暗くします。画素ごとに `pow` を計算すると重いので、`0-255` の対応表（**ルックアップテーブル, LUT**）を一度だけ作り、`cv2.LUT(img, lut)` で一括適用するのが定石です。これが「露出を直す」ときの第一手になります。

<figure class="lec-fig"><svg viewBox="0 0 560 290" role="img" aria-label="ガンマ補正の入出力カーブ。γが1より大きいと対角線より上で暗部を持ち上げ明るく、1より小さいと暗くなる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="280" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ガンマ補正 out = 255·(in/255)^(1/γ)</text><line x1="80" y1="250" x2="500" y2="250" stroke="#3f3f46" stroke-width="2"/><polygon points="508,250 496,244 496,256" fill="#3f3f46"/><line x1="80" y1="250" x2="80" y2="50" stroke="#3f3f46" stroke-width="2"/><polygon points="80,42 74,54 86,54" fill="#3f3f46"/><text x="300" y="278" text-anchor="middle" font-size="13" fill="#52525b">入力 in（暗 0 → 255 明）</text><text x="66" y="46" text-anchor="middle" font-size="13" fill="#52525b">出力</text><line x1="80" y1="250" x2="480" y2="50" stroke="#71717a" stroke-width="1.6" stroke-dasharray="5 4"/><polyline points="80,250 120,180 160,150 280,104 380,74 480,50" fill="none" stroke="#ea580c" stroke-width="3"/><polyline points="80,250 160,238 280,200 380,138 480,50" fill="none" stroke="#2563eb" stroke-width="3"/><text x="120" y="92" font-size="13" font-weight="700" fill="#c2410c">γ＞1：暗部を持ち上げ明るく</text><text x="406" y="84" font-size="12.5" font-weight="700" fill="#52525b">γ＝1（線形）</text><text x="300" y="240" font-size="13" font-weight="700" fill="#1d4ed8">γ＜1：締めて暗く</text></svg><figcaption>ガンマ補正 <code>out = 255·(in/255)^(1/γ)</code> の入出力カーブです。対角線(<b>γ＝1</b>)は素通し。<b>γ＞1</b>(オレンジ)は対角線より上にふくらみ、<b>暗部を大きく持ち上げて</b>明るくします(暗所写真の救済)。<b>γ＜1</b>(青)は下に垂れ、全体を締めて暗くします。実装では <code>0-255</code> の <b>LUT</b> を一度だけ作り <code>cv2.LUT</code> で一括適用します。</figcaption></figure>

```python
# 線形（飽和演算）: alpha=コントラスト, beta=明るさ
out = cv2.convertScaleAbs(bgr, alpha=1.4, beta=30)

# ガンマ（非線形・LUT で高速）: gamma>1 で暗部を持ち上げる
inv = 1.0 / gamma
lut = np.clip((np.arange(256) / 255.0) ** inv * 255.0, 0, 255).astype(np.uint8)
out = cv2.LUT(bgr, lut)
```

ここで注意したいのは、これらを BGR の3チャンネルに同じ係数で掛けている限り、**色味（色相）は概ね保たれる**という点です。`convertScaleAbs` も `LUT` も各チャンネルを独立に同じ写像で動かすため、明るさ・コントラストは変わっても色相のバランスは大きく崩れません（飽和でクリップされた極端な明部を除きます）。色が崩れるのはむしろ、次節で扱う「チャンネルごとに別々の平坦化」をやってしまったときです。

## 4. ヒストグラム平坦化・CLAHE — 輝度だけに掛けて色を崩さない

コントラストを「分布を見て自動で」広げたいときに使うのが**ヒストグラム平坦化**です。`cv2.equalizeHist` は輝度の累積分布を使ってトーンを均一に引き伸ばし、眠い画像の細部を浮かび上がらせます。ただし、その対象は**1チャンネルのグレースケール**に限られます。ここで初学者が必ずやる失敗が、「**BGR の各チャンネルを別々に `equalizeHist` する**」ことです。各色が独立に引き伸ばされるとチャンネル間のバランス（＝色相）が崩れ、色が破綻してしまいます。02 では、入力に対する平均 ΔE が `40` 超まで跳ね上がることで「色が壊れた」ことを定量化します。

正しい作法は、「**輝度チャンネルだけを平坦化し、色差はそのまま戻す**」ことです。具体的には、`BGR→YCrCb` に変換して `Y` だけ `equalizeHist` し、`Cr`/`Cb` には触れずに `YCrCb→BGR` へ戻します。あるいは `BGR→Lab` に変換し、`L` だけを動かしても構いません。こうすれば、明るさのダイナミックレンジは広がるのに色味は保たれます（ΔE が小さく収まります）。この「明るさを動かしたいなら輝度チャンネル、色を動かしたいなら色チャンネル」という分離こそが、本章を貫く原則です。

<figure class="lec-fig"><svg viewBox="0 0 660 245" role="img" aria-label="平坦化はBGR各chに別々に掛けると色が壊れ、YCrCbのYだけに掛けると色が保たれる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">平坦化は「輝度chだけ」に掛ける</text><text x="40" y="56" font-size="13" font-weight="700" fill="#dc2626">× 各chを別々に平坦化</text><rect x="44" y="66" width="58" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="73" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">BGR</text><line x1="102" y1="86" x2="148" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="156,86 144,80 144,92" fill="#71717a"/><rect x="158" y="66" width="150" height="40" fill="#fff7ed" stroke="#dc2626" stroke-width="1.8"/><text x="233" y="82" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3f3f46">B,G,R を個別に</text><text x="233" y="98" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3f3f46">equalizeHist</text><line x1="308" y1="86" x2="354" y2="86" stroke="#71717a" stroke-width="2"/><polygon points="362,86 350,80 350,92" fill="#71717a"/><rect x="364" y="66" width="150" height="40" fill="#ffffff" stroke="#dc2626" stroke-width="2"/><text x="439" y="82" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">色が破綻</text><text x="439" y="98" text-anchor="middle" font-size="12" font-weight="700" fill="#dc2626">平均ΔE ＞ 40</text><text x="40" y="166" font-size="13" font-weight="700" fill="#15803d">○ 輝度chだけ平坦化</text><rect x="40" y="176" width="50" height="40" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="65" y="201" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">BGR</text><line x1="90" y1="196" x2="126" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="134,196 122,190 122,202" fill="#71717a"/><rect x="136" y="176" width="56" height="40" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="164" y="201" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">YCrCb</text><line x1="192" y1="196" x2="226" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="234,196 222,190 222,202" fill="#71717a"/><rect x="236" y="176" width="150" height="40" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/><text x="311" y="193" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">Y だけ equalize</text><text x="311" y="209" text-anchor="middle" font-size="11.5" font-weight="700" fill="#52525b">Cr, Cb はそのまま</text><line x1="386" y1="196" x2="420" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="428,196 416,190 416,202" fill="#71717a"/><rect x="430" y="176" width="56" height="40" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="458" y="201" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1d4ed8">BGR</text><line x1="486" y1="196" x2="520" y2="196" stroke="#71717a" stroke-width="2"/><polygon points="528,196 516,190 516,202" fill="#71717a"/><rect x="530" y="176" width="118" height="40" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="589" y="193" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">色を保持</text><text x="589" y="209" text-anchor="middle" font-size="11.5" font-weight="700" fill="#15803d">ΔE 小</text></svg><figcaption>ヒストグラム平坦化(<code>equalizeHist</code>)を <b>BGR の各チャンネルに別々に掛ける</b>と、チャンネル間のバランス＝色相が崩れ、<b>平均 ΔE が 40 超</b>へ跳ね上がります(上段)。正しくは <code>BGR→YCrCb</code> として <b>輝度 Y だけを平坦化</b>し、<code>Cr/Cb</code> はそのまま戻します(下段)。明暗のレンジは広がるのに<b>色味は保たれ</b>、ΔE は小さく収まります(<code>Lab</code> の <code>L</code> でも同様)。</figcaption></figure>

`equalizeHist` は画像全体で1本のヒストグラムを使うため、局所的に暗い／明るい領域の細部は潰れがちです。これを補うのが **CLAHE（Contrast Limited Adaptive Histogram Equalization）**です。画像をタイルに分けてタイルごとに平坦化し、増幅しすぎを `clipLimit` で抑えます。CLAHE でも、やはり `Lab` の `L`（または `YCrCb` の `Y`）にだけ掛けるのが鉄則です。

```python
# 正解1: 輝度(Y)だけ全体平坦化（色を保つ）
ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

# 正解2: 輝度(L)だけ局所平坦化 CLAHE
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
```

## 5. 彩度・色相の調整 — HSV で「色だけ」を動かす

「明るさはそのままに、鮮やかさや色味だけを変えたい」とき、HSV が真価を発揮します。**彩度**は `S` チャンネルを `factor` 倍するだけ（`0` でグレースケール相当、`>1` で鮮やか）で調整でき、**色相**は `H` に値を足して回します（色相環は一周するので `mod 180`）。いずれも明度 `V` には触れないため、03 では彩度を動かしても色相を回しても `mean_V` が一定であることを数値で確かめます。「軸が分かれている」恩恵が、ここでようやく腑に落ちるはずです。

<figure class="lec-fig"><svg viewBox="0 0 600 270" role="img" aria-label="HSV円柱。色相Hは円周の角度0-179、彩度Sは中心から外周への半径、明度Vは下から上への高さ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">HSV：色相H・彩度S・明度V の3軸</text><line x1="80" y1="92" x2="80" y2="215" stroke="#52525b" stroke-width="2"/><line x1="300" y1="92" x2="300" y2="215" stroke="#52525b" stroke-width="2"/><ellipse cx="190" cy="215" rx="110" ry="36" fill="#52525b" stroke="#3f3f46" stroke-width="2"/><ellipse cx="190" cy="92" rx="110" ry="36" fill="#f4f4f5" stroke="#52525b" stroke-width="2"/><circle cx="300" cy="92" r="8" fill="#dc2626"/><circle cx="190" cy="56" r="8" fill="#f97316"/><circle cx="80" cy="92" r="8" fill="#16a34a"/><circle cx="190" cy="128" r="8" fill="#2563eb"/><circle cx="190" cy="92" r="6" fill="#71717a"/><path d="M 250 64 Q 190 36 130 64" fill="none" stroke="#1d4ed8" stroke-width="2"/><polygon points="130,64 141,58 141,69" fill="#1d4ed8"/><text x="110" y="56" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">H</text><line x1="190" y1="92" x2="248" y2="74" stroke="#2563eb" stroke-width="2"/><polygon points="256,71 248,79 245,69" fill="#2563eb"/><text x="232" y="88" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">S</text><line x1="330" y1="208" x2="330" y2="80" stroke="#ea580c" stroke-width="2"/><polygon points="330,72 324,84 336,84" fill="#ea580c"/><text x="344" y="148" font-size="15" font-weight="700" fill="#c2410c">V</text><rect x="376" y="86" width="16" height="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="400" y="99" font-size="12" fill="#3f3f46">H：角度(0-179)で色を選ぶ</text><rect x="376" y="120" width="16" height="16" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><text x="400" y="133" font-size="12" fill="#3f3f46">S：中心(灰)→外周(鮮やか)</text><rect x="376" y="154" width="16" height="16" fill="#ffedd5" stroke="#ea580c" stroke-width="1.5"/><text x="400" y="167" font-size="12" fill="#3f3f46">V：下(暗)→上(明)</text><text x="376" y="206" font-size="12.5" font-weight="700" fill="#c2410c">S・H を動かしても V は不変</text></svg><figcaption>HSV は色を<b>円柱</b>で捉えると掴みやすい色空間です。<b>色相 H</b> は円周まわりの角度(OpenCV は <b>0-179</b>)で赤・緑・青…という色味、<b>彩度 S</b> は中心(無彩色の灰)から外周(鮮やか)への半径、<b>明度 V</b> は下(暗)から上(明)への高さを表します。<code>S</code> を倍にしたり <code>H</code> を回しても <b>V には触れない</b>ので、明るさを保ったまま色だけを動かせます。</figcaption></figure>

実装上の注意点は、やはり **dtype** です。`S` を `1.5` 倍すると、`uint8`（最大255）を超える画素が出てきます。`uint8` のまま掛けるとオーバーフローでラップして色が壊れるので、**`float32` に上げてから掛け、`np.clip(…, 0, 255)` してから `uint8` に戻す**のが定石です。一方、色相シフトでは `H` が `0-179` の循環値なので、`int` で足してから `% 180` で巻き戻し、`uint8` に戻します。

```python
# 彩度を factor 倍（V は不変）。float で計算 → clip → uint8。
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 色相を回す（mod 180）。色名は変わるが S/V は不変。
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

さらに、「**特定の色相だけ**」を選んで彩度を上げる選択的な調整もできます（03 の `demo_selective_saturation`）。色相マスクを `inRange` で作り、`np.where(mask, boosted, original)` を使えば、主役だけを鮮やかにし背景はそのまま、という編集が実現します。このようにマスクで「どこを」、HSV で「どの軸を」動かすかを分けて考えると、複雑な色編集も単純な部品の組み合わせに還元できます。

## 6. ホワイトバランス・色恒常性 — 照明の色を打ち消す

写真が青っぽい／赤っぽいのは、被写体ではなく**照明の色**のせいであることが多いものです。これを補正するのが**ホワイトバランス（WB）**であり、その背後にある「人間は照明が変わっても物の色を一定に感じる」という性質を**色恒常性（color constancy）**と呼びます。最も簡単で実用的なのが、**gray-world 仮説**——「シーン全体を平均すれば無彩色（灰）になるはず」という考え方です。これに従えば、各チャンネルの平均 `(mB,mG,mR)` を共通の灰へ揃えるゲインを掛けるだけで、色かぶりが打ち消せます。

もう一つの定番が、**white-patch（perfect reflector）仮説**——「画像で一番明るい点は本来 白のはず」という考え方です。こちらは各チャンネルの（ノイズに強い 99 パーセンタイル）最大値を `255` に合わせて引き伸ばします。どちらの手法も数行で書け、`float` で計算して `clip→uint8` に戻すだけです。04 では寒色かぶりをわざと作り、補正後に各チャンネル平均が揃って目標画像との平均 ΔE が下がることで、効果を定量化します。

```python
# gray-world: 各ch平均を全体平均(灰)へ揃える
f = bgr.astype(np.float32)
means = f.reshape(-1, 3).mean(axis=0)          # (mB, mG, mR)
gains = means.mean() / np.clip(means, 1e-6, None)
wb = np.clip(f * gains, 0, 255).astype(np.uint8)
```

ただし gray-world には、明確な**限界**もあります。それは「色が満遍なく散っている」という前提が崩れる場合です。たとえば一面の緑の草原のように**1色が支配的なシーン**では、平均を無理に灰へ寄せるせいで、本来緑のシーンがマゼンタ寄りにくすんでしまいます（04 の `demo_gray_world_failure` で再現）。これは、仮説の前提を理解せずに当てはめると逆効果になるという、良い教訓です。そのため実務では、無彩色のグレーカードを写し込んでそこを基準にしたり、より頑健な手法（Shades-of-Gray など）へ進んだりします。

## 7. 色で「選ぶ」と、色差で「測る」 — inRange と ΔE

色空間は「調整」だけでなく、「**領域抽出**」の道具でもあります。そこで第3回の HSV `inRange` を、ここでは Lab・YCrCb まで広げましょう。HSV は色相で細かく選べる一方、照明変化には弱いという弱点があります。これに対し **Lab の `a*`/`b*` は知覚に近い**ため、「`a* < 120`（緑側）」のように軸の意味そのものでしきい値を引けます（`a*` は `128` が中性で、背景の無彩色=128 を除くため demo では少し厳しめの `a* < 120` を使います。05 の `demo_hsv_vs_lab`）。また**肌色検出では YCrCb の色差 `Cr`/`Cb` が定番**で、`Cr∈[133,173], Cb∈[77,127]` というレンジは、純粋な赤（`Cr` が `246` まで跳ねて上限を超える）を巻き込みにくいのが強みです。HSV で素朴に「暖色なら肌」とやると純赤を誤検出してしまうことを、05 では誤検出画素数で対比します。

「2つの色がどれくらい違うか」を**人の見た目に沿って測る**のが、**知覚的色差 ΔE** です。最も基本的な **ΔE\*76** は、CIE L\*a\*b\* 空間でのユークリッド距離にあたります。目安は「ΔE<1 はほぼ識別不能、2-3 で訓練された目が気づく、>5 で明確に別の色」です。ここで最重要の注意が、「**必ず float の L\*a\*b\* で測る**」こと。なぜなら 8bit Lab（全チャンネル 0-255 にスケール）で距離を取ると、`L` と `a*/b*` のスケール比が歪み、ΔE の意味が崩れてしまうからです。したがって `bgr.astype(float32)/255 → COLOR_BGR2Lab` で本来の `L:0-100, a/b:±127` を得てから、距離を計算します。

<figure class="lec-fig"><svg viewBox="0 0 520 280" role="img" aria-label="Lab色空間のa*b*平面。a*は緑から赤、b*は青から黄、中心が無彩色。2色間の距離が知覚的色差ΔE" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="250" y="24" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">Lab の a*b* 平面と色差 ΔE</text><text x="110" y="42" font-size="11.5" fill="#71717a">※ L*(明度)は紙面に垂直な軸</text><line x1="100" y1="165" x2="404" y2="165" stroke="#3f3f46" stroke-width="2"/><polygon points="412,165 400,159 400,171" fill="#3f3f46"/><line x1="250" y1="255" x2="250" y2="66" stroke="#3f3f46" stroke-width="2"/><polygon points="250,58 244,70 256,70" fill="#3f3f46"/><circle cx="404" cy="165" r="7" fill="#dc2626"/><text x="396" y="152" text-anchor="end" font-size="12.5" font-weight="700" fill="#dc2626">赤 +a*</text><circle cx="100" cy="165" r="7" fill="#16a34a"/><text x="108" y="152" font-size="12.5" font-weight="700" fill="#15803d">緑 −a*</text><circle cx="250" cy="66" r="7" fill="#f97316"/><text x="262" y="72" font-size="12.5" font-weight="700" fill="#c2410c">黄 +b*</text><circle cx="250" cy="255" r="7" fill="#2563eb"/><text x="262" y="256" font-size="12.5" font-weight="700" fill="#1d4ed8">青 −b*</text><circle cx="250" cy="165" r="5" fill="#71717a"/><text x="240" y="190" text-anchor="end" font-size="11.5" fill="#52525b">中性=灰(0 / 8bitは128)</text><line x1="340" y1="112" x2="170" y2="212" stroke="#18181b" stroke-width="2"/><circle cx="340" cy="112" r="8" fill="#ea580c"/><text x="351" y="108" font-size="12.5" font-weight="700" fill="#c2410c">色1</text><circle cx="170" cy="212" r="8" fill="#2563eb"/><text x="159" y="226" text-anchor="end" font-size="12.5" font-weight="700" fill="#1d4ed8">色2</text><text x="296" y="124" font-size="13" font-weight="700" fill="#18181b">ΔE = 2点間の距離</text></svg><figcaption>Lab は明度 <b>L*</b> と色度 <b>a*</b>(緑↔赤)・<b>b*</b>(青↔黄)で色を表し、中心が無彩色(灰)です(8bit では <code>128</code> 中心、float では <code>0</code> 中心)。2 つの色の<b>距離が知覚的色差 ΔE</b> で、<b>ΔE が 1 未満</b>はほぼ識別不能、<b>2-3</b> で気づき、<b>5 超</b>で明確に別の色です。<b>必ず float の L*a*b*</b>(<code>bgr/255 → BGR2Lab</code>)で測ります(8bit Lab はスケールが歪み ΔE が壊れる)。</figcaption></figure>

```python
# float Lab で画素ごとの ΔE*76（知覚的色差）
lab_a = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
lab_b = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
de = np.sqrt(((lab_a - lab_b) ** 2).sum(axis=2))   # (H, W) の ΔE マップ
```

この ΔE は、「目標色に近い画素を抜く」**知覚的セグメンテーション**にも使えます（05 の `demo_delta_e_segmentation`）。目標色との ΔE がしきい値以下の画素をマスクにすれば、「どれだけ似た色まで含めるか」を知覚スケールで直接指定できるからです。こうして見ると、`inRange`（軸ごとの箱）と ΔE（点からの距離）は、色で選ぶときの相補的な2つの道具だといえます。

## 8. 評価で締める — ヒストグラム・平均彩度・ΔE と、色以外の per-pixel 情報

調整は「やった気」になりやすい操作です。だからこそ本章では、**定量評価**を必ずセットにします。明るさ／コントラスト／ガンマ／平坦化の効果は、**輝度ヒストグラム**（分布の位置と広がり）に最も素直に表れます。一方、彩度調整は**平均彩度 `mean(S)`**で、色や WB の変化は**平均 ΔE** で測ります。「動かしたい軸の指標が動き、動かしたくない軸の指標は動いていない」ことを数値で確認できて初めて、調整が制御下にあると言えます。そこでミニプロジェクトでは、各補正段で `mean_Y / mean_S / ΔE` を表にまとめ、ΔE が単調に下がる（目標に近づく）ことを見せます。

最後に、少し視野を広げておきましょう。本章は「per-pixel（画素ごと）の**色**情報」を扱いましたが、画像の各画素には色以外の情報も載り得ます——たとえば透過度を表す **alpha** チャンネル（RGBA）、各画素までの距離を表す **depth（深度マップ）**、明暗の変化の強さを表す **勾配（gradient）**などです。これらもすべて、「`(H, W, C)` の配列を目的の軸で操作する」という本章と同じ枠組みで扱えます。色チャンネルを輝度と色差に分けたのと同じ発想で、深度やマスクを別チャンネルとして合成・調整していく——それが、検出・セグメンテーション・3D へと続く道です。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

「色かぶり＋露出不足＋低コントラスト」で撮れてしまった一枚を、段階的に補正して**元の見え方にどれだけ近づいたか**を定量評価する一気通貫パイプラインです。

1. **劣化版を合成**: 目標画像に色かぶり（`apply_color_cast`）と露出不足・低コントラスト（`convertScaleAbs`）を掛ける。
2. **補正パイプライン**: ①ホワイトバランス（gray-world）→ ②露出（ガンマ）→ ③局所コントラスト（CLAHE on L）→ ④彩度。**順序にも意味がある**（WB を先にしないと後段がかぶり色を増幅する）。
3. **各段を定量評価**: `mean_Y`（明るさ）・`mean_S`（彩度）・**目標との平均 ΔE**。ΔE が単調に下がるのを確認する。
4. **領域抽出**: 補正後の画像から青領域（HSV `inRange`）と肌色領域（YCrCb `inRange`）を抜いて保存。
5. **可視化**: パイプライン各段のモンタージュと輝度ヒストグラムを `lectures/43_color_spaces_and_adjustments/outputs/` に保存。

<figure class="lec-fig"><svg viewBox="0 0 660 310" role="img" aria-label="ミニプロジェクトのフロー。劣化版を合成しWB→ガンマ→CLAHE→彩度の順に補正、各段を定量評価して領域抽出と可視化を出力する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト：劣化した1枚を WB→ガンマ→CLAHE→彩度 で補正</text><rect x="24" y="72" width="92" height="60" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="70" y="100" text-anchor="middle" font-size="13.5" font-weight="700" fill="#18181b">劣化版</text><text x="70" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">色かぶり+露出↓</text><rect x="146" y="72" width="92" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="192" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① WB</text><text x="192" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">gray-world</text><rect x="268" y="72" width="92" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="314" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② ガンマ</text><text x="314" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">露出を直す</text><rect x="390" y="72" width="92" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="436" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">③ CLAHE</text><text x="436" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">L だけ局所</text><rect x="512" y="72" width="92" height="60" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="558" y="100" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">④ 彩度</text><text x="558" y="118" text-anchor="middle" font-size="10.5" fill="#52525b">S を強調</text><line x1="116" y1="102" x2="140" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="146,102 136,97 136,107" fill="#71717a"/><line x1="238" y1="102" x2="262" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="268,102 258,97 258,107" fill="#71717a"/><line x1="360" y1="102" x2="384" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="390,102 380,97 380,107" fill="#71717a"/><line x1="482" y1="102" x2="506" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="512,102 502,97 502,107" fill="#71717a"/><line x1="558" y1="132" x2="558" y2="204" stroke="#71717a" stroke-width="2"/><polygon points="558,210 553,200 563,200" fill="#71717a"/><rect x="472" y="210" width="172" height="64" rx="6" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="558" y="236" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">定量評価</text><text x="558" y="256" text-anchor="middle" font-size="11" fill="#52525b">ΔE↓・mean_Y/S</text><rect x="252" y="210" width="176" height="64" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="340" y="236" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">領域抽出</text><text x="340" y="256" text-anchor="middle" font-size="11" fill="#52525b">青HSV・肌YCrCb</text><rect x="32" y="210" width="176" height="64" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="120" y="236" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">可視化</text><text x="120" y="256" text-anchor="middle" font-size="11" fill="#52525b">montage・hist</text><line x1="472" y1="242" x2="434" y2="242" stroke="#71717a" stroke-width="2"/><polygon points="428,242 438,237 438,247" fill="#71717a"/><line x1="252" y1="242" x2="214" y2="242" stroke="#71717a" stroke-width="2"/><polygon points="208,242 218,237 218,247" fill="#71717a"/><text x="330" y="298" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">目標との平均 ΔE は段を追うごとに単調減少（劣化 37.96 → 補正後 18.44）</text></svg><figcaption><b>章末ミニプロジェクト</b>のパイプラインです。色かぶり+露出不足で<b>劣化させた1枚</b>を、<b>① ホワイトバランス → ② ガンマ → ③ CLAHE（<code>L</code> だけ） → ④ 彩度</b> の順で補正します。<b>順序に意味があり</b>、WB を先に置かないと後段がかぶり色を増幅します。各段で <code>mean_Y</code>・<code>mean_S</code>・<b>目標との平均 ΔE</b> を測ると、ΔE が <b>37.96 → 18.44</b> へ単調に下がります。最後に青（HSV）・肌（YCrCb）の領域抽出と、モンタージュ／輝度ヒストグラムを出力します。</figcaption></figure>

実行すると `目標との平均ΔE: 劣化=37.96 → 補正後=18.44` のように、補正で色が目標へ近づくのが数値で出ます。ここまでを AI 補助なしでそらで書けるのが本章の合格ラインです。

```bash
uv run python lectures/43_color_spaces_and_adjustments/mini_project.py
```

## ✅ 到達チェックリスト

- [ ] **説明できる**: OpenCV 8bit の HSV は `H=0-179`・S/V=0-255、Lab は `L` も含め全ch `0-255`、YCrCb は全ch `0-255`。なぜ `H` が半分なのか（`uint8` に収めるため）。
- [ ] **説明できる**: 同じ `cvtColor` でも `uint8` と `float32` で出力スケールが変わる（float なら `H=0-360`, Lab は `L:0-100, a/b:±127`）。
- [ ] **できる**: `convertScaleAbs(alpha, beta)` で線形に明るさ／コントラストを変えられ、素の numpy `+` のラップと飽和演算の違いを説明できる。
- [ ] **できる**: ガンマ補正の LUT を `np.arange(256)` から自分で組み、`cv2.LUT` で適用できる（`γ>1` で明るくなる向きを言える）。
- [ ] **できる**: 彩度を `S` だけ・色相を `H` だけ動かせる（`float32`→`clip`→`uint8`、`H` は `% 180`）。明度 `V` が不変であることを数値で確認できる。
- [ ] **できる**: ヒストグラム平坦化／CLAHE を **YCrCb の Y / Lab の L** にだけ掛けて色を崩さない。BGR 全chに掛ける失敗との違いを ΔE で言える。
- [ ] **できる**: gray-world ホワイトバランスを実装でき、その前提（色が満遍なく散っている）と破綻するシーンを説明できる。
- [ ] **できる**: HSV／Lab／YCrCb の `inRange` で特定色・肌色を抽出でき、YCrCb が純赤を巻き込みにくい理由を言える。
- [ ] **できる**: **float Lab** で平均 ΔE を計算でき、8bit Lab で測ってはいけない理由を説明できる。
- [ ] **できる**: 調整の効果を 輝度ヒストグラム・平均彩度・ΔE で定量化できる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/43_color_spaces_and_adjustments/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. BGR 画像を HSV に変換して返す（`ex1_to_hsv`）。OpenCV の `H` が `0-179` に収まることを確かめる基礎課題です。
2. 線形調整 `out = alpha*in + beta` を飽和演算で行って返す（`ex2_brightness_contrast`）。`alpha` がコントラスト、`beta` が明るさで、255 を超えたら頭打ちにします。
3. ガンマ補正用の 256 要素 LUT（`uint8`, `shape=(256,)`）を作って返す（`ex3_gamma_lut`）。`LUT[i] = 255*(i/255)^(1/gamma)` を 0-255 にクリップします。
4. 彩度（HSV の `S`）だけを `factor` 倍して返す（`ex4_scale_saturation`）。色相・明度を保つため `float32` で計算し `clip` してから `uint8` に戻します。
5. 色相 `H` を `delta`（0-179 スケール）だけ回して返す（`ex5_shift_hue`）。色相環は一周するので `% 180` で巻き戻します。
6. 輝度チャンネル（YCrCb の `Y`）だけを `equalizeHist` で平坦化して返す（`ex6_equalize_luminance`）。`Cr`/`Cb` には触れず色を崩さないようにします。
7. gray-world ホワイトバランスを実装して返す（`ex7_gray_world_wb`）。各チャンネル平均を全体平均（灰）に揃えるゲインを掛けて色かぶりを打ち消します。
8. YCrCb の定番レンジで肌色マスク（0/255, `(H,W)`）を返す（`ex8_skin_mask_ycrcb`）。`Cr∈[133,173]`・`Cb∈[77,127]`・`Y` 全域で `inRange` します。
9. 2 画像の平均 ΔE*76（float `L*a*b*` のユークリッド距離）を返す（`ex9_mean_delta_e`）。必ず `float32/255 → BGR2Lab` で本来のスケールにしてから距離を取ります。
10. Lab の `L` チャンネルにだけ CLAHE を掛けて返す（`ex10_clahe_luminance`）。`createCLAHE` の局所平坦化を輝度だけに適用して色を保ちます。

## ❓ よくある落とし穴・FAQ・デバッグ

この章のバグはほぼ「**スケール**」「**dtype（オーバーフロー）**」「**どのチャンネルに掛けたか**」のどれかに集約されます。

- **Q. 彩度を上げたら変な色のノイズが出た。** → `uint8` のまま `S` を掛けてオーバーフローしています。`float32` に上げて掛け、`np.clip(…, 0, 255)` してから `uint8` に戻してください。
- **Q. ヒストグラム平坦化したら色が毒々しくなった。** → BGR の各チャンネルを別々に `equalizeHist` していませんか。`YCrCb` の `Y`（または `Lab` の `L`）だけに掛け、色差は触らずに戻します。
- **Q. ΔE の値が妙に大きい／意味が合わない。** → 8bit の Lab で距離を取っていませんか。ΔE は **float の L\*a\*b\***（`bgr/255` を `BGR2Lab`）で計算します。`L` と `a/b` のスケールが揃っていないと壊れます。
- **Q. 明るさを足したら一部の画素が逆に暗くなった。** → 素の numpy `+` が `uint8` のオーバーフローでラップしています（`200+100=44`）。`cv2.add` か `cv2.convertScaleAbs` の飽和演算を使います。
- **Q. HSV の色域指定が効かない。** → OpenCV の `H` は `0-179` です。0-360 の角度や、float 用のスケールをそのまま入れていないか確認してください。
- **Q. ホワイトバランスしたら全体がくすんだ。** → gray-world は「色が満遍なく散っている」前提です。1色が支配的なシーンでは破綻します。グレーカード基準や white-patch、より頑健な手法を検討してください。
- **Q. ガンマを上げたのに暗いまま／白飛びした。** → 向きの取り違えです。本章の定義 `out=(in)^(1/γ)` では `γ>1` で明るく、`γ<1` で暗く。極端な値は明部を飽和させます。
- **Q. matplotlib で表示したら色がおかしい。** → BGR をそのまま渡しています。`cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` してから `imshow` してください（`color_helpers.to_rgb` が担当）。
- **Q. Lab→BGR で往復したら色がわずかにズレた。** → 8bit Lab は量子化誤差が乗ります（往復は完全可逆ではない）。厳密さが要るなら float のまま処理を通します。
- **デバッグの定石**: 「動かしたい軸の指標」と「動かしたくない軸の指標」を両方 print する。彩度を上げたら `mean_S` が上がり `mean_V` が変わっていないか、WB なら各ch平均が揃ったか、を毎回数値で見ます。

## 🚀 発展トピック・参考

- **ΔE の高精度版**: `ΔE*76`（本章）は単純なユークリッド距離で、青や彩度方向で人の知覚とズレます。より知覚的に正確な **ΔE\*94 / ΔE\*2000（CIEDE2000）** があります（`scikit-image` の `color.deltaE_ciede2000` 等）。
- **頑健なホワイトバランス**: gray-world / white-patch を一般化した **Shades-of-Gray（Minkowski ノルム）** や、学習ベースの色恒常性。グレーカード／カラーチェッカーを使ったキャリブレーションも実務の定番です。
- **トーンマッピング・HDR**: 露出の異なる複数枚を合成する HDR（`cv2.createMergeMertens` 等）と、ダイナミックレンジ圧縮のトーンマッピング。ガンマ補正の発展形です。
- **色を使った前段処理**: 本章の色マスクは、第4回のモルフォロジー・輪郭抽出（`findContours`）と組み合わせると物体カウントや領域分割に発展します。
- **色以外の per-pixel 情報**: alpha（RGBA 合成）、depth（深度マップ／第27回）、勾配（エッジ／第4回）も「`(H,W,C)` を軸で操作する」同じ枠組みで扱えます。
- 公式ドキュメント: [OpenCV Color Conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html) ／ [Changing Colorspaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) ／ [Histogram Equalization / CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

## 💡 実践ユースケース集

ここまでに身につけた道具（WB・ガンマ・CLAHE・彩度・inRange・ΔE）は、そのまま現実の小ツールになります。ここでは代表例を3つ挙げます。なかでも最初の1つは、実際に実行できる出発点（`use_case.py`）です。

### 1. ワンクリック写真自動補正ツール（`use_case.py`・実行可能）

- **何に使うか**: スマホ・ドラレコ・古いアルバムなど「色かぶり・暗い・眠い・くすんだ」写真を、人手調整なしでフォルダごと一括で見られる状態に直す。
- **作り方の要点**: 補正の流れはミニプロジェクトと同じ **WB→ガンマ→CLAHE→彩度** ですが、決定的に違うのは **正解画像が無い前提で、補正量を画像自身の統計から自動推定する**点です。たとえば自動ガンマは平均輝度を目標(0.5)へ寄せる `gamma = log(mean)/log(target)`、自動彩度は `target_S / 現在のmean_S` を倍率とし、いずれも暴走しないよう範囲をクリップします。さらに1枚ずつではなく **フォルダ単位のバッチ**で回し、補正後画像と before/after モンタージュを保存します。
- **注意**: 正解が無いので、評価は ΔE ではなく **自己統計**（`mean_Y` が中庸へ寄ったか、`std_Y`／`mean_S` が回復したか）で見ます。色かぶり写真では WB が水増しされた彩度を正すため `mean_S` が下がることもありますが、これは正常な挙動です。なお gray-world が1色の支配的なシーンで破綻する点は、`mini_project.py` と同じく要注意です。

<figure class="lec-fig"><svg viewBox="0 0 660 210" role="img" aria-label="use_case.pyの自動補正フロー。フォルダ内の画像を走査し統計から補正量を自動推定、WB→ガンマ→CLAHE→彩度で補正してbefore/afterを保存、次の画像へバッチで繰り返す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">use_case.py：正解画像なしで、画像の統計から補正量を自動推定</text><rect x="20" y="68" width="102" height="66" rx="6" fill="#f4f4f5" stroke="#71717a" stroke-width="1.5"/><text x="71" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">フォルダ走査</text><text x="71" y="116" text-anchor="middle" font-size="10.5" fill="#52525b">画像を1枚ずつ</text><rect x="150" y="68" width="102" height="66" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="201" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">統計を算出</text><text x="201" y="116" text-anchor="middle" font-size="10" fill="#52525b">mean_Y・mean_S</text><rect x="280" y="68" width="102" height="66" rx="6" fill="#ffedd5" stroke="#c2410c" stroke-width="2.5"/><text x="331" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">補正量を推定</text><text x="331" y="112" text-anchor="middle" font-size="10" fill="#52525b">γ・彩度倍率を</text><text x="331" y="126" text-anchor="middle" font-size="10" fill="#52525b">自動算出+クリップ</text><rect x="410" y="68" width="102" height="66" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="461" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">補正 4 段</text><text x="461" y="112" text-anchor="middle" font-size="10" fill="#52525b">WB→γ→</text><text x="461" y="126" text-anchor="middle" font-size="10" fill="#52525b">CLAHE→彩度</text><rect x="540" y="68" width="102" height="66" rx="6" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="591" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">保存</text><text x="591" y="112" text-anchor="middle" font-size="10" fill="#52525b">before/after</text><text x="591" y="126" text-anchor="middle" font-size="10" fill="#52525b">montage</text><line x1="122" y1="101" x2="144" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="150,101 140,96 140,106" fill="#71717a"/><line x1="252" y1="101" x2="274" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="280,101 270,96 270,106" fill="#71717a"/><line x1="382" y1="101" x2="404" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="410,101 400,96 400,106" fill="#71717a"/><line x1="512" y1="101" x2="534" y2="101" stroke="#71717a" stroke-width="2"/><polygon points="540,101 530,96 530,106" fill="#71717a"/><polyline points="591,134 591,176 71,176 71,146" fill="none" stroke="#16a34a" stroke-width="2"/><polygon points="71,134 66,146 76,146" fill="#16a34a"/><text x="331" y="170" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">次の画像へ ・ フォルダ単位でバッチ処理</text></svg><figcaption><b><code>use_case.py</code> のワンクリック自動補正</b>の全体フローです。<b>正解画像が無い</b>前提なので、フォルダ内の画像を1枚ずつ走査し、各画像の<b>統計（<code>mean_Y</code>・<code>mean_S</code>）から補正量を自動推定</b>します（自動ガンマ <code>γ=log(mean)/log(0.5)</code>、自動彩度 <code>target_S/mean_S</code> をクリップ）。あとは<b>ミニプロジェクトと同じ WB→ガンマ→CLAHE→彩度</b>で補正し、before/after モンタージュを保存。これを<b>フォルダ単位のバッチ</b>で全画像に繰り返します。</figcaption></figure>

```bash
# 実写真を data/43_color_spaces_and_adjustments/ に置くとそれを自動補正（無ければ合成3枚でデモ）
uv run python lectures/43_color_spaces_and_adjustments/use_case.py
```

- **`data/` の置き方**: プロジェクト直下に `data/43_color_spaces_and_adjustments/` を作り、`.jpg`/`.png` などを入れるだけ（日本語・空白パスも自前デコードで読めます）。
- **拡張アイデア**: WB を white-patch／Shades-of-Gray に差し替え比較、自動ガンマを中央値や暗部パーセンタイル基準に、彩度を肌色領域だけ控えめにする選択的彩度、`argparse` で入力フォルダ・目標輝度・clipLimit を CLI 化、補正レシピ（ガンマ・彩度倍率）を JSON ログ出力。

### 2. EC・フリマ出品写真の色／明るさ自動そろえ

- **何に使うか**: 別々の照明で撮った商品写真の「白背景が灰色・黄ばみ・露出バラつき」を整え、一覧で見たときの統一感を出す。返品理由になりがちな「実物と色が違う」を減らす前処理。
- **作り方の要点**: gray-world か white-patch でホワイトバランスを取り、白背景が白に近づくよう露出（ガンマ）を合わせ、最後に輝度(L)だけ CLAHE で軽く立てます。色再現が命なので **彩度は盛らず控えめ**にし、グレーカードを写し込めるなら、そこを基準にすると更に安定します。
- **注意**: 商品色の正確さが目的なので、「映え」狙いの過補正は禁物です。彩度を上げすぎると別色に見えてしまい、`equalizeHist` を BGR 全chに掛けると色が破綻します（必ず輝度chのみに掛けます）。

### 3. 暗所・色かぶり映像の視認性改善（防犯／ドラレコ／内視鏡のフレーム補正）

- **何に使うか**: 夜間の防犯カメラや水中・内視鏡など「暗くてコントラストが低い」映像を、各フレームに同じ補正を掛けて見やすくする。動体検出・OCR・人物確認の前段としても効く。
- **作り方の要点**: 1フレーム＝1枚の画像とみなして `use_case.py` の補正関数を流用し、`cv2.VideoCapture` で読んだ各フレームに適用してから `VideoWriter` で書き戻します。リアルタイム性が要るなら、**CLAHE と WB は数フレームに1回**だけ係数を更新し、間のフレームでは同じ係数を使い回すと軽くなります。
- **注意**: フレームごとに係数を作り直すと明るさがチラつく（フリッカ）ので、係数は時間方向に平滑化（EMA）します。また CLAHE の `clipLimit` を上げ過ぎると、ノイズまで増幅されてしまいます。headless 運用では `imshow` を使わず、ファイル／動画に保存して確認します。

## ▶ 動かし方

```bash
# 1) 色空間の地図（スケール表・float vs uint8・往復誤差・チャンネル分解）
uv run python lectures/43_color_spaces_and_adjustments/01_color_spaces_map.py

# 2) 明るさ・コントラスト・ガンマ・輝度平坦化/CLAHE
uv run python lectures/43_color_spaces_and_adjustments/02_brightness_contrast_gamma.py

# 3) 彩度・色相（HSV で色だけ動かす・選択的彩度）
uv run python lectures/43_color_spaces_and_adjustments/03_saturation_hue.py

# 4) ホワイトバランス（gray-world / white-patch と破綻例）
uv run python lectures/43_color_spaces_and_adjustments/04_white_balance.py

# 5) HSV/Lab/YCrCb 領域抽出（特定色・肌色）と ΔE セグメンテーション
uv run python lectures/43_color_spaces_and_adjustments/05_hsv_lab_segmentation.py

# 6) 章末ミニプロジェクト（補正パイプライン + 評価 + 抽出）
uv run python lectures/43_color_spaces_and_adjustments/mini_project.py

# 7) 実践ユースケース（ワンクリック写真自動補正・バッチ。data/ に実写真があれば優先）
uv run python lectures/43_color_spaces_and_adjustments/use_case.py

# 8) 演習（自己採点。SHOW_SOLUTION=1 で模範解答の挙動を確認）
uv run python lectures/43_color_spaces_and_adjustments/exercises.py
uv run python lectures/43_color_spaces_and_adjustments/exercises_solutions.py
```

結果はすべて `lectures/43_color_spaces_and_adjustments/outputs/` に PNG で保存されます（headless 前提・画面表示はしません）。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.x ／ opencv-python-headless 4.13（`cv2` 4.13.0）／ Pillow 12.2 ／ matplotlib 3.10 ／（深層トラックで使う torch は 2.12+cpu。本章では未使用）
