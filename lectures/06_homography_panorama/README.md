# 第6回 ホモグラフィ推定とパノラマ合成 — findHomography(RANSAC)・warpPerspective・手作りパノラマ・cv2.Stitcher 比較

> トラック: **古典CV** ／ レベル: **初級** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加依存グループなし）

## 🎯 この章のゴール

この章を終えたとき、あなたは「2 枚（あるいは複数枚）の画像が同じ平面・同じ場面を別アングルで写しているとき、片方をもう片方の座標系へ正確に重ね合わせる」という操作を、最初から最後まで自分の手で書けるようになります。その鍵になるのが**ホモグラフィ（homography, 射影変換）**——平面同士を結ぶ 3×3 の変換行列です。前回（第5回）で学んだ特徴点マッチングは、「どの点とどの点が対応するか」までを与えてくれました。そこで本章は、その対応点を入力として `cv2.findHomography` で幾何関係そのものを推定し、`cv2.warpPerspective` で画素を実際に貼り合わせる、という**特徴点マッチングの応用の集大成**を扱います。

同時に、推定を実務で使い物にするための 2 つの必須スキルを体に入れます。1 つは**外れ値への耐性**です。対応点には必ず誤対応が混じるため、Lowe の比率テスト（第5回）に続けて **RANSAC** で幾何的に整合しない対応を捨て、返ってくる**インライア mask** の数と比を必ず検証します。もう 1 つは**品質の数値化**です。「なんとなく合っている」で済ませず、対応点を変換した後の**再投影誤差（平均ユークリッド距離）**で精度を測り、合成結果の良し悪しを**重なり領域の SSIM** で定量比較します。さらに「最低 4 対応点が必要」「3 点では解けない」「でたらめな対応では破綻する」といった原理的な制約も、実験を通して痛感していきます。

到達点を一言でいえば、**サンプル画像も GPU も無い環境で、複数枚を順次つなぐパノラマ合成パイプラインを一人で書き、その精度を再投影誤差とインライア比で説明でき、さらに高レベル API の `cv2.Stitcher` が中で何を自動化しているかを理解した上で使い分けられる**ことです。まず手作りで仕組みを完全に見通し、そのうえで自動化と比べる——この順序をたどるからこそ、ブラックボックスに頼らずに済む地力がつきます。

---

## 1. ホモグラフィとは何か — アフィン変換との違い

ホモグラフィは「**平面**を別の平面へ写す射影変換」であり、3×3 の行列 `H` で表されます。画素 `(x, y)` を同次座標 `(x, y, 1)` にして `H` を掛け、最後に 3 番目の要素で割って正規化する——この手続きで新しい座標へ移します。第3回で扱ったアフィン変換（2×3 行列）が「平行線を平行に保つ」変換（回転・平行移動・拡大縮小・せん断）だったのに対し、ホモグラフィは**遠近（パース）まで表現できる**点が本質的な違いです。実際、手前は大きく奥は小さく写る台形補正のような変形は、アフィンでは表せず、ホモグラフィでしか表せません。

<figure class="lec-fig"><svg viewBox="0 0 660 280" role="img" aria-label="アフィンは平行線を平行に保ち、ホモグラフィは遠近まで表現する射影変換" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="40" y="85" width="120" height="120" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><line x1="80" y1="85" x2="80" y2="205" stroke="#2563eb" stroke-width="1" opacity="0.3"/><line x1="120" y1="85" x2="120" y2="205" stroke="#2563eb" stroke-width="1" opacity="0.3"/><line x1="40" y1="125" x2="160" y2="125" stroke="#2563eb" stroke-width="1" opacity="0.3"/><line x1="40" y1="165" x2="160" y2="165" stroke="#2563eb" stroke-width="1" opacity="0.3"/><text x="100" y="232" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1d4ed8">元の正方形</text><line x1="170" y1="145" x2="222" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="230,145 220,140 220,150" fill="#71717a"/><polygon points="240,95 350,83 360,191 250,203" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><line x1="295" y1="89" x2="305" y2="197" stroke="#ea580c" stroke-width="1" opacity="0.35"/><line x1="245" y1="149" x2="355" y2="137" stroke="#ea580c" stroke-width="1" opacity="0.35"/><text x="303" y="232" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">アフィン (2×3)</text><text x="303" y="251" text-anchor="middle" font-size="12" fill="#52525b">平行線は平行のまま</text><line x1="380" y1="145" x2="432" y2="145" stroke="#71717a" stroke-width="2"/><polygon points="440,145 430,140 430,150" fill="#71717a"/><polygon points="450,90 605,98 577,192 470,182" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><line x1="528" y1="94" x2="524" y2="187" stroke="#c2410c" stroke-width="1" opacity="0.35"/><line x1="460" y1="136" x2="591" y2="145" stroke="#c2410c" stroke-width="1" opacity="0.35"/><text x="524" y="232" text-anchor="middle" font-size="13.5" font-weight="700" fill="#c2410c">ホモグラフィ (3×3)</text><text x="524" y="251" text-anchor="middle" font-size="12" fill="#52525b">遠近(パース)も表現</text></svg><figcaption>第3回の<b>アフィン変換</b>(2×3・自由度6)は平行線を平行に保つ変形（回転・拡縮・せん断）までで、<b>ホモグラフィ</b>(3×3・自由度8)は手前は大きく奥は小さい<b>遠近(パース)</b>まで表せます。同じ正方形でも、アフィンは<b>平行四辺形</b>に、ホモグラフィは<b>台形</b>にできるのが本質的な違いです。</figcaption></figure>

では、なぜパノラマでホモグラフィなのでしょうか。ここが肝心な点です。カメラを三脚で固定し、その場で首だけ振って（パンして）撮った複数枚は、たとえ立体的な風景でも「無限遠の平面を別アングルから見たもの」とみなせ、隣り合う画像どうしは**厳密にホモグラフィで結ばれます**。同じように、ポスターや書類・看板のような**平面の物体**を別角度から撮った 2 枚も、やはりホモグラフィで重なります。だからこそ「パノラマ合成」と「平面物体の位置合わせ（平面物体検出）」は、同じ数学（ホモグラフィ）の表と裏なのです。本章はこの両方を扱います。

ここで下の表に、アフィンとホモグラフィの守備範囲を整理しておきます。どちらを使うべきかは、「平行線が平行のまま保たれる変形か、それとも遠近が入るか」で判断します。

| 変換 | 行列 | 自由度 | 最低必要な対応点 | 保たれる性質 | 主な用途 |
| --- | --- | --- | --- | --- | --- |
| アフィン変換 | 2×3 | 6 | 3 | 平行線は平行のまま | 回転・拡縮・せん断・スキャン補正 |
| ホモグラフィ（射影変換） | 3×3 | 8 | **4** | 直線は直線のまま（平行は崩れる） | パノラマ・台形補正・平面物体の位置合わせ |

この表の「最低必要な対応点」は、後で何度も効いてきます。ホモグラフィの自由度は 8（3×3 の 9 要素からスケールの 1 自由度を除く）で、1 つの対応点が 2 本の方程式（x と y）を与えます。したがって、**最低 4 点**あれば解けます。逆に 3 点ではアフィンしか決まらず、ホモグラフィは決まりません。この「4 点」という数字は、本章の `01_homography_ransac.py` で、実際に 3 点を渡して `None` が返ることにより確認します。

## 2. 対応点からホモグラフィを解く — `findHomography` と「最低4点」

対応点さえ手元にあれば、ホモグラフィの推定は `cv2.findHomography(src_pts, dst_pts, method, ransacReprojThreshold)` の 1 行で済みます。この関数は `src_pts` を `dst_pts` へ写す `H` を返し、`dst ≈ H · src` を満たします。点の形は `(N, 1, 2)` の `float32` 配列にそろえるのが OpenCV の流儀で、本講座のヘルパ `points_from_matches` がマッチ列をこの形に変換してくれます。引数のうち `method` には外れ値除去のための `cv2.RANSAC`（後述）を指定し、`ransacReprojThreshold` は「何ピクセルまでのズレをインライアとみなすか」のしきい値です。

ここで初学者が最初に出会う設計上の選択が、「**どちら向きの `H` を推定するか**」です。`findHomography(src, dst)` は `src→dst` の変換を返します。したがって「画像 B を画像 A の座標系へ重ねたい」なら、B 上の点を `src`、対応する A 上の点を `dst` にして呼びます。もし向きを逆にすると、後で `warpPerspective` するときに画像が反対方向へ飛んでいってしまいます。下のコードは本章の核であり、`01_homography_ransac.py` の中心部分です。

```python
# pts_b（画像Bの点）→ pts_a（画像Aの点）へ写す H を推定する
H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0)
if H is None:                       # 4点未満や退化配置だと None が返る
    raise RuntimeError("ホモグラフィを推定できませんでした")
print("インライア数 =", int(mask.sum()))
```

このコードで必ず守るべき点は 2 つあります。第一に、`H is None` のチェックです。`findHomography` は対応点が 4 未満だったり、点が一直線上に並ぶ（退化する）と、例外ではなく `None` を返します——これは `cv2.imread` の `None` 戻りと同じ静かな罠で、握りつぶすと後段で意味不明なエラーになります。第二に、戻り値の `mask` を捨てないことです。次節で見るとおり、この `mask` こそが推定の信頼性を判断する材料になります。`01_homography_ransac.py` の `[5] 失敗モードの体験` セクションでは、3 点だけを渡すと `H is None: True` となること、でたらめな 60 対応ではインライアがほとんど出ないことを、実際に表示して確かめます。

## 3. RANSAC とインライア mask — 外れ値に強い推定

比率テスト（第5回）を通した「良いマッチ」であっても、なお幾何的にあり得ない対応（外れ値）が必ず残ります。このとき最小二乗法でそのまま `H` を解くと、わずかな外れ値が解全体を引っ張り、推定を壊してしまいます。これを防ぐのが **RANSAC（RANdom SAmple Consensus）** です。その発想は「全部を真面目に使う」のをやめ、**ランダムに 4 点だけ選んで仮の `H` を作り、その `H` に合う対応（インライア）が最も多くなる仮説を採用する**という多数決にあります。外れ値はどの仮説にも合わないため自然に弾かれ、結果として外れ値に極めて強い推定が得られます。

<figure class="lec-fig"><svg viewBox="0 0 600 290" role="img" aria-label="RANSACはしきい値の帯に最も多く入る仮説を選び、外れ値を弾く" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">RANSAC：帯(しきい値)に最も多く入る仮説を採用</text><polygon points="70,179 545,69 545,121 70,231" fill="#dbeafe" opacity="0.6"/><line x1="70" y1="179" x2="545" y2="69" stroke="#2563eb" stroke-width="1" stroke-dasharray="5 4" opacity="0.7"/><line x1="70" y1="231" x2="545" y2="121" stroke="#2563eb" stroke-width="1" stroke-dasharray="5 4" opacity="0.7"/><line x1="70" y1="205" x2="545" y2="95" stroke="#2563eb" stroke-width="2.5"/><circle cx="110" cy="190" r="4" fill="#16a34a"/><circle cx="160" cy="176" r="4" fill="#16a34a"/><circle cx="215" cy="180" r="4" fill="#16a34a"/><circle cx="270" cy="150" r="4" fill="#16a34a"/><circle cx="325" cy="154" r="4" fill="#16a34a"/><circle cx="385" cy="124" r="4" fill="#16a34a"/><circle cx="440" cy="128" r="4" fill="#16a34a"/><circle cx="500" cy="98" r="4" fill="#16a34a"/><circle cx="180" cy="95" r="4" fill="#dc2626"/><circle cx="240" cy="80" r="4" fill="#dc2626"/><circle cx="300" cy="235" r="4" fill="#dc2626"/><circle cx="420" cy="215" r="4" fill="#dc2626"/><text x="505" y="86" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">仮の H</text><text x="150" y="276" text-anchor="middle" font-size="12" fill="#2563eb">帯の幅 = 許容ズレ (px)</text><circle cx="372" cy="272" r="4" fill="#16a34a"/><text x="382" y="276" font-size="12" fill="#15803d">インライア</text><circle cx="478" cy="272" r="4" fill="#dc2626"/><text x="488" y="276" font-size="12" fill="#dc2626">外れ値</text></svg><figcaption>比率テスト後にも残る<b>外れ値</b>(誤対応)を、最小二乗ではなく多数決で弾くのが<b>RANSAC</b>です。ランダムに最小サンプル（ホモグラフィなら<b>4点</b>）から仮の <code>H</code> を作り、<b>しきい値の帯</b>（<code>ransacReprojThreshold</code>）に入る<b>インライア</b>が最多になる仮説を採用します。返る <code>mask</code> の数と比が、推定の信頼度の一次指標です。</figcaption></figure>

`findHomography` に `cv2.RANSAC` を渡すと、第 2 戻り値 `mask` が `(N, 1)` の 0/1 配列で返り、`1` がインライア（採用された対応）、`0` が外れ値を表します。ここから `int(mask.sum())` がインライア数、`mask.sum() / len(good)` がインライア比になります。**この 2 つの数値が推定の信頼度の一次指標**であり、インライアが 4〜十数個しかない、あるいは比が極端に低い場合は、その `H` を信用してはいけません。なお本章の合成データでは、比率テスト後 605 対応のうち 545 がインライア（約 90%）といった健全な値が出ます。

```python
inliers = int(mask.sum())
print(f"インライア {inliers}/{len(good)} (比 {inliers/len(good):.0%})")
# mask をそのまま drawMatches に渡すと、インライアだけを描ける
vis = cv2.drawMatches(img_a, kp1, img_b, kp2, good, None,
                      matchesMask=mask.ravel().tolist(),
                      matchColor=(0, 255, 0))
```

上のように `mask.ravel().tolist()` を `drawMatches` の `matchesMask` に渡すと、RANSAC が採用したインライアだけを緑線で可視化できます。`01_homography_ransac.py` は、この図を `01_matches_inliers.png` に保存します。線がどれだけ間引かれたかを `01_matches_all.png` と見比べれば、「2 段構えの外れ値除去（比率テスト → RANSAC）」が効いている様子が目で分かります。また、RANSAC は内部で乱数を使うため、スクリプト冒頭で `cv2.setRNGSeed(0)` を呼んで結果を再現可能にしている点にも注目してください。

## 4. 推定品質を数値で測る — 再投影誤差とインライア比

「`H` が良いかどうか」を見た目だけで判断するのは危険です。そこで本章は、**再投影誤差（reprojection error）**を一次の品質指標に据えます。定義は単純で、`src` の各点を推定した `H` で写し、対応する `dst` の点との**ユークリッド距離**を測り、その平均を取るだけです。値が小さいほど「対応点がぴったり重なる」良い `H` であり、合成画像に直結する直感的な物差しになります。ただし注意点として、**誤差はインライアだけで測る**ことが重要です。外れ値を含めると、もともと幾何に合わない点の距離が混ざり、値が無意味に大きくなります（本章の例では、インライアのみで 0.94px なのに対し、全点込みだと 27px 程度にまで膨らみます）。

```python
projected = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
dist = np.linalg.norm(projected - pts_dst.reshape(-1, 2), axis=1)
err = dist[mask.ravel().astype(bool)].mean()   # インライアのみで平均
print(f"再投影誤差 = {err:.3f} px")
```

このコードの心臓は `cv2.perspectiveTransform` で、これは「画像」ではなく「点群」にホモグラフィを適用する関数です（画像を変形する `warpPerspective` と混同しないこと——前者は座標を、後者は画素を動かします）。ここで、再投影誤差とインライア比は役割が違う点に注意してください。**インライア比は「対応の純度＝どれだけ外れ値を弾けたか」**を表し、**再投影誤差は「採用した対応がどれだけ正確に重なるか」**を表します。したがって、両方を併せて見ることで推定を多面的に評価できます。

本章の合成データには、大きな利点があります。2 視点は同じ平面シーンを既知のホモグラフィで切り出して作っているため、**真のホモグラフィ `H_true` が分かっている**のです。そこで `01_homography_ransac.py` は、推定した `H` を `H_true` と並べて表示し（`[4]` セクション）、行列差のフロベニウスノルムまで出します。教科書では「答えが分からないから誤差で代用する」と説明されがちですが、本章は答えそのものを持っています。だからこそ、再投影誤差という代理指標が実際に真値との一致と連動していることを、自分の目で確認できます。`01_homography_ransac.py` を実行して、`[2]〜[4]` の数値（インライア数・再投影誤差・真値との一致）が揃って良好になること——これが、この節の確認ポイントです。

## 5. `perspectiveTransform` で点・四隅を写す — キャンバス計算と平面物体検出

`perspectiveTransform` は地味な関数ですが、本章では 2 つの重要な仕事をこなします。1 つ目は**パノラマのキャンバスサイズ計算**です。画像 B を `H` で変形すると、その四隅が A の座標系のどこへ飛ぶかは、あらかじめ分かりません（負の座標へはみ出すこともあります）。そこで先に**四隅だけを `perspectiveTransform` で写し**、A の四隅と合わせた外接矩形を求めます。こうすれば、両方を収めるのに必要なキャンバスの大きさと、負座標を原点側へ押し込む平行移動量が決まります。つまり、画像全体を変形する前に「枠」だけ計算しておく、という発想です。

```python
h, w = img_b.shape[:2]
corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
warped_corners = cv2.perspectiveTransform(corners, H)   # A 座標系での B の四隅
```

2 つ目の仕事が**平面物体の位置合わせ（平面物体検出）**です。検出したい平面物体（看板・本の表紙・書類など）をテンプレートとして、雑然としたシーンに対し特徴点マッチング → `findHomography` を行います。そして得られた `H` でテンプレートの四隅を `perspectiveTransform` すれば、シーンの中で物体が占める**四辺形（傾いた枠）**が求まります。矩形ではなく四辺形になるのがポイントで、物体が斜めから写っていても、パースのついた正しい枠を描けます。これもまた、パノラマと全く同じ数学の応用です。

`01_homography_ransac.py` は、この応用を `[7]` セクションで実演します。合成した「TARGET」カードを射影変換でシーンに貼り込み、テンプレート→シーンのマッチングで `H` を推定し、`cv2.polylines` でカードの四辺形を緑枠で描いて `01_object_detected.png` に保存します。実行後、散らかった背景の中で傾いたカードがぴたりと枠に収まっていれば成功です。この「枠が物体に張り付く」感覚は、後の物体検出・AR・書類のまっすぐ化など、多くの応用の出発点になります。

## 6. `warpPerspective` で2枚を貼り合わせる — キャンバスと平行移動 T

点ではなく画素を実際に動かすのが、`cv2.warpPerspective(img, H, (W, H))` です。ここで初学者が必ず一度はまるのが、「**出力は常に原点 (0,0) から始まる**」という仕様です。`H` が画像を左や上の負の領域へ動かす場合、原点より外の部分は問答無用で切り取られてしまいます。これを防ぐには、5 節で四隅から求めた最小座標 `(x_min, y_min)` を 0 に押し込む**平行移動行列 `T`** を作り、`warpPerspective` には `H` ではなく **`T @ H`** を渡します。基準画像 A 側も、`T` だけ平行移動して同じキャンバスに置きます。

<figure class="lec-fig"><svg viewBox="0 0 660 300" role="img" aria-label="warpPerspectiveは原点始まりで負座標が切れる。四隅から平行移動Tを作りT@Hで全体をキャンバスに収める" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="30" y="48" width="275" height="47" fill="#f4f4f5"/><rect x="30" y="48" width="110" height="212" fill="#f4f4f5"/><line x1="140" y1="95" x2="300" y2="95" stroke="#71717a" stroke-width="1.5"/><polygon points="308,95 298,90 298,100" fill="#71717a"/><line x1="140" y1="95" x2="140" y2="258" stroke="#71717a" stroke-width="1.5"/><polygon points="140,266 135,256 145,256" fill="#71717a"/><text x="148" y="110" font-size="11" fill="#3f3f46">(0,0)</text><polygon points="72,60 178,92 188,172 80,160" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5" opacity="0.85"/><rect x="140" y="95" width="85" height="68" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5" opacity="0.85"/><text x="182" y="134" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">A</text><text x="116" y="142" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">warp(B)</text><text x="82" y="212" text-anchor="middle" font-size="11" fill="#71717a">切れる領域</text><line x1="312" y1="150" x2="352" y2="150" stroke="#c2410c" stroke-width="2.5"/><polygon points="360,150 350,145 350,155" fill="#c2410c"/><text x="336" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">T で平行移動</text><rect x="360" y="72" width="270" height="180" fill="#fafafa" stroke="#52525b" stroke-width="1.5" stroke-dasharray="6 4"/><text x="368" y="88" font-size="11" fill="#3f3f46">(0,0)</text><polygon points="378,96 484,128 494,208 386,196" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5" opacity="0.85"/><rect x="430" y="120" width="85" height="68" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5" opacity="0.85"/><text x="505" y="242" text-anchor="middle" font-size="11" fill="#15803d">全部キャンバス内</text></svg><figcaption><code>warpPerspective</code> の出力は<b>常に原点 (0,0) から始まる</b>ため、<code>H</code> が画像を負の座標へ動かすと<b>はみ出した部分は無言で切られます</b>。四隅を <code>perspectiveTransform</code> で写して外接矩形から最小座標を求め、それを 0 に押し込む<b>平行移動 T</b> を作り、<code>warpPerspective</code> には <code>H</code> ではなく <b>T @ H</b> を渡すと、全体がキャンバスに収まります。</figcaption></figure>

```python
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
warp_a = cv2.warpPerspective(view_a, T,       size)   # 基準は平行移動だけ
warp_b = cv2.warpPerspective(view_b, T @ H,   size)   # B は位置合わせ＋平行移動
```

この `T @ H` という合成こそ、本章で繰り返し出てくる「行列を掛けて変換をつなぐ」考え方の最小形です。`T`（平行移動）と `H`（射影）を別々に画像へ適用するのではなく、行列として先に掛けてから一度だけ `warpPerspective` する——こうすると補間が 1 回で済むため、画質劣化も計算量も最小に抑えられます。`02_panorama_manual.py` はこの過程を段階ごとに保存しており、`02_warp_a.png` / `02_warp_b.png` で「2 枚が同じキャンバス上の正しい位置へ並んだ」中間状態を確認できます。

なお `size`（キャンバスの幅・高さ）は、外接矩形から `(x_max - x_min, y_max - y_min)` で決めます。本講座のヘルパ `canvas_for` が、この「四隅を集めて外接矩形 → `T` とサイズ」の定型処理を 1 関数に閉じ込めています。`02_panorama_manual.py` の `[2]` でキャンバスサイズが表示されるので、入力 2 枚（各 640×480）より横に広い（例: 974×480）出力になることを確認してください。これこそが「2 枚分の視野をつないだ」証拠です。

## 7. シームとブレンディング — 上書き vs フェザー

2 枚を同じキャンバスに置いただけでは、重なり領域で問題が起きます。単純に「B で A を上書き」すると、2 枚の明るさや色がわずかに違うだけで、重なりの境界に**くっきりした継ぎ目（シーム）**が出てしまうのです。たとえ位置合わせ（`H` 推定）が完璧でも、露出差や微小なズレによってシームは必ず目立ちます。そのため、これを消す**ブレンディング**が合成の仕上げになります。本章ではまず素朴な上書きでシームを体験し、次に**フェザー（feather, 加重平均）ブレンド**で滑らかに溶かしていきます。

フェザーの考え方は、「**各画像の縁に近い画素ほど軽く、中心に近いほど重く**扱い、重なり領域では両者の重み付き平均を取る」というものです。重みは、`cv2.distanceTransform` で「画像の縁からの距離」を計算して作ります。こうすると、B の寄与は重なりの A 側で滑らかに 0 へ近づき、境界がグラデーションで溶け合って段差が消えます。下は `02_panorama_manual.py` のブレンド部の核心です。

<figure class="lec-fig"><svg viewBox="0 0 620 300" role="img" aria-label="フェザー合成は重なり領域で縁ほど軽い加重平均を取り、継ぎ目シームを消す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="50" y="55" width="250" height="78" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5" opacity="0.65"/><rect x="250" y="55" width="300" height="78" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5" opacity="0.65"/><line x1="250" y1="50" x2="250" y2="250" stroke="#71717a" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/><line x1="300" y1="50" x2="300" y2="250" stroke="#71717a" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/><text x="150" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">画像 A</text><text x="425" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">画像 B</text><text x="275" y="46" text-anchor="middle" font-size="11" fill="#52525b">重なり</text><line x1="50" y1="248" x2="555" y2="248" stroke="#d4d4d8" stroke-width="1"/><polyline points="50,165 250,165 300,248" fill="none" stroke="#2563eb" stroke-width="2.5"/><polyline points="250,248 300,165 550,165" fill="none" stroke="#c2410c" stroke-width="2.5"/><text x="150" y="157" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">重み w_A</text><text x="440" y="157" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">重み w_B</text><text x="300" y="277" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">重なりで加重平均 → 段差が消える</text></svg><figcaption>2 枚を単純に上書きすると、露出差で重なりの境界に<b>シーム(継ぎ目)</b>が出ます。<b>フェザー</b>は <code>distanceTransform</code> による「縁からの距離」を重みにし、<b>縁ほど軽く中心ほど重く</b>して重なり領域で加重平均します。<code>w_A</code> は右端へ、<code>w_B</code> は左端へ滑らかに 0 へ近づくので段差が溶けて消えます。重みマップも画素と同じ <code>H</code> で warp するのが要点です。</figcaption></figure>

```python
w_a = cv2.warpPerspective(feather_weight(view_a.shape[:2]), T,     size)  # 重みも同じく warp
w_b = cv2.warpPerspective(feather_weight(view_b.shape[:2]), T @ H, size)
wsum = w_a + w_b
wsum[wsum == 0] = 1.0                                   # 空白部のゼロ割り回避
blended = (warp_a * w_a[..., None] + warp_b * w_b[..., None]) / wsum[..., None]
```

ここでのポイントは、画素だけでなく**重みマップも全く同じ `T`／`T @ H` で warp する**ことです。画素と重みが同じ幾何で動くからこそ、重なり領域で正しい比率の平均が取れます。`02_panorama_manual.py` は、上書き版 `02_pano_naive.png` とフェザー版 `02_pano_feather.png` の両方を保存します。実行後にこの 2 枚を見比べ、naive 版の重なり境界に縦の段差が見えること、そして feather 版ではそれが消えて滑らかにつながっていることを確認する——これが、この節の確認ポイントです。なお OpenCV の `cv2.Stitcher`（9 節）は、これをさらに高度化した**マルチバンド・ブレンディング**（周波数帯ごとに別々に混ぜる手法）を内部で行っています。

## 8. 複数枚を順次つなぐ — ホモグラフィの合成

2 枚を貼れれば、3 枚以上も原理は同じです——ただし「どの座標系を基準にするか」を決め、**ホモグラフィを掛け合わせて**全画像を基準フレームへそろえる必要があります。例として、左→中→右に重なる 3 枚 `I0, I1, I2` を、先頭 `I0` を基準にする場合を考えましょう。まず隣り合うペアで「`I1 → I0`」「`I2 → I1`」のホモグラフィをそれぞれ推定します。次に、`I2` を基準まで運ぶ行列は `M2 = M_{1→0} · M_{2→1}` と**合成**します。一般化すると `M_i =（ひとつ前までの合成）· (i → i-1 の H)` であり、これを順に積み上げれば、全画像を `I0` の座標系へ写せます。

<figure class="lec-fig"><svg viewBox="0 0 640 260" role="img" aria-label="複数枚はペアごとのHを掛け合わせ、全画像を基準I0の座標系へそろえる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="40" y="55" width="120" height="80" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/><text x="100" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">I0 (基準)</text><rect x="260" y="55" width="120" height="80" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="320" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">I1</text><rect x="480" y="55" width="120" height="80" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="540" y="101" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">I2</text><line x1="256" y1="80" x2="172" y2="80" stroke="#c2410c" stroke-width="2"/><polygon points="160,80 172,74 172,86" fill="#c2410c"/><text x="212" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">H: 1→0</text><line x1="476" y1="80" x2="396" y2="80" stroke="#2563eb" stroke-width="2"/><polygon points="384,80 396,74 396,86" fill="#2563eb"/><text x="434" y="68" text-anchor="middle" font-size="12" font-weight="700" fill="#1d4ed8">H: 2→1</text><text x="400" y="170" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">M2 = H(1→0) · H(2→1)</text><line x1="600" y1="220" x2="114" y2="220" stroke="#16a34a" stroke-width="2"/><polygon points="102,220 114,214 114,226" fill="#16a34a"/><text x="356" y="208" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">M_i = M_(i-1) · H_i で全画像を I0 座標系へ</text></svg><figcaption>3 枚以上でも原理は同じです。隣り合うペアで <code>H</code> を推定し（「1→0」「2→1」）、<b>行列を掛け合わせて</b>全画像を基準 <b>I0</b> の座標系へそろえます。一般に <b>M_i = M_(i-1) · H_i</b> で、これは 6 節の <code>T @ H</code> の自然な一般化です。ただし各ペアの誤差が積で伝播するため、<b>端の画像ほど累積誤差</b>が出やすい点に注意します。</figcaption></figure>

```python
M_to_ref = [np.eye(3)]                       # M0 = 単位行列（基準はそのまま）
for i in range(1, n):
    H_i = estimate(images[i], images[i-1])   # 「i → i-1」の H を推定
    M_to_ref.append(M_to_ref[-1] @ H_i)      # 掛け合わせて基準フレームへ
```

この「変換を行列の積でつなぐ」操作は、6 節の `T @ H` の自然な一般化にほかなりません。全画像の `M_i` が揃えば、あとは 5〜7 節と同じです——全部の四隅から共通キャンバスを決め、各画像を `T @ M_i` で warp し、フェザー重みで一括ブレンドするだけです。本講座のヘルパ `build_panorama` は、この一連（ペアごとのマッチ → 合成 → キャンバス → 重み付き合成）をまとめており、`02_panorama_manual.py` の `[4]` と `03_stitcher_compare.py` の両方から呼ばれます。

ただし注意点として、**順次合成は誤差が累積する**ことを知っておいてください。各ペアの `H` にわずかな誤差があると、それが掛け算で伝播し、端の画像ほどズレが大きくなりがちです（これを抑えるのが、後述の Stitcher が行う「バンドル調整」です）。本章の合成データは特徴が豊富で各ペアのインライアが数百あるため、累積は軽微です。とはいえ念のため、`02_panorama_manual.py` の `[4]` が出力する「ペアごとのインライア数」が両ペアとも十分大きい（数百）ことを確認してください。最終的に `02_pano_three.png` が 3 枚分の横長パノラマになっていれば成功です。

## 9. `cv2.Stitcher` との比較 — 自動化の中身と使い分け

ここまで手作りしてきた一連の処理（特徴抽出 → マッチ → `findHomography` → warp → ブレンド）を、OpenCV は `cv2.Stitcher` という高レベル API に丸ごとパッケージしています。使い方は驚くほど簡単で、画像のリストを `stitch` に渡すだけです。返り値の `status` が `cv2.Stitcher_OK`（=0）なら成功、それ以外は失敗コードを表します。したがって**`status` を必ず確認する**のが鉄則です（失敗を握りつぶして `pano` を使うと落ちます）。

```python
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch([img0, img1, img2])
if status == cv2.Stitcher_OK:
    cv2.imwrite("pano.png", pano)
```

`cv2.Stitcher` がブラックボックスで終わらないよう、中で何をしているかを手作りパイプラインと対応づけておきましょう。下の表のとおり、Stitcher は手作りの各段に加えて、**カメラパラメータのバンドル調整**（全画像の整合性をまとめて最適化し累積誤差を抑える）と、**露出補正・マルチバンド合成**（明るさ差の補正と高品質ブレンド）まで自動で行います。これこそが、手作り版より継ぎ目が目立たず、端のズレも小さい理由です。

| 段階 | 手作りパイプライン（本章 2〜8 節） | cv2.Stitcher（自動） |
| --- | --- | --- |
| 特徴抽出・マッチ | ORB + 比率テスト（自分で実装） | 内部で自動 |
| 幾何推定 | `findHomography` + RANSAC | 内部で自動 + **バンドル調整** |
| 投影モデル | 平面（そのまま貼る） | 球面/円筒など**カメラ回転モデル** |
| 露出差の補正 | なし | **自動露出補正** |
| 合成 | フェザー（加重平均） | **マルチバンド・ブレンド** |

`cv2.Stitcher` には 2 つのモードがあります。`cv2.Stitcher_PANORAMA` は「その場で首を振って撮った回転カメラ」を想定し、球面/円筒投影でつなぎます（湾曲したパノラマらしい仕上がりになります）。一方 `cv2.Stitcher_SCANS` は「平面をスキャンした」想定で、アフィン中心につなぎます。`03_stitcher_compare.py` は PANORAMA を試し、失敗したら SCANS にフォールバックする実装で、**どちらのモードが何を仮定しているか**を学べます。実務での使い分けは明快です。**まず `cv2.Stitcher` を試し、うまくいけばそれでよい**（露出補正やブレンドまで自動で高品質）。そして Stitcher が失敗する（特徴が少ない・重なりが足りない・特殊な投影が要る）ときや、各段を細かく制御・デバッグしたいときに、手作りパイプラインへ降りていく——これが正しい順序です。

## 10. 評価 — 重なり領域の SSIM で結果を比べる

最後に、手作り版と Stitcher 版という 2 つの合成結果を**客観的に比較**します。再投影誤差（4 節）が「対応点の合い具合」を測るのに対し、合成画像そのものの一致度を測るには **SSIM（Structural SIMilarity, 構造的類似度）** を使います。SSIM は局所的な明るさ・コントラスト・構造の一致を 0〜1 で表し、1 に近いほど「2 枚が構造的に似ている」ことを意味します。なお本講座では scikit-image に依存せず、ガウシアン窓での局所平均・分散・共分散から、定義どおり `cv2` だけで実装しています（`cv_helpers.ssim`）。

2 つのパノラマはキャンバスの座標系が違うため、単純には重ねられません。そこで `03_stitcher_compare.py` は、**まず 2 つの結果どうしを ORB + `findHomography` で位置合わせ**し、両方が中身を持つ**重なり領域だけ**で SSIM を測ります。これが、評価方針「重なり領域の SSIM で比較する」の実装です。あえて重なり領域に限定するのは、片方にしか写っていない端まで含めると不一致が増え、指標がぼやけてしまうためです。

```python
# 2つの結果を位置合わせ（other → ref）してから、重なり領域だけで SSIM
H, mask = cv2.findHomography(pts_other, pts_ref, cv2.RANSAC, 3.0)
aligned = cv2.warpPerspective(pano_other, H, (ref_w, ref_h))
score = ssim(gray_ref_masked, gray_aligned_masked)   # 1 に近いほど一致
```

`03_stitcher_compare.py` は、手作り版の生成時間・サイズ・ペアごとのインライア数、Stitcher の採用モード・サイズ・生成時間、そして両者の重なり領域 SSIM を一通り出力し、`03_compare.png` に 2 つのパノラマを縦に並べて保存します（matplotlib は Agg バックエンドで、`cv2.cvtColor(..., COLOR_BGR2RGB)` で色順を直してから渡しています）。実行後、SSIM の値が表示されること、そして `03_compare.png` で「手作り版は長方形のまっすぐな仕上がり」「Stitcher(PANORAMA) 版は円筒投影で軽く湾曲し露出も均された仕上がり」という**設計思想の違い**が一目で分かること——これが、この節の確認ポイントです。なお Stitcher が失敗した場合でも、スクリプトは握りつぶさずメッセージを出して正常終了するため、環境差に対しても頑健です。

## 11. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」の形で一覧にします。実装中に詰まったら、まずここを見てください。多くの不具合は、結局この数個の原因に集約されます。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `findHomography` が `None` を返す | 対応点が 4 未満、または点が一直線（退化） | 良マッチ数を確認。比率テストを緩める/特徴を増やす。`None` を必ずチェック |
| 合成すると片方の画像が消える/切れる | `warpPerspective` の出力が原点始まりで負座標が切れた | 四隅から `T`（平行移動）を作り `T @ H` を渡す。キャンバスサイズも外接矩形で計算 |
| パノラマがぐちゃぐちゃに歪む | `H` の向きが逆、または外れ値で `H` が破綻 | `src/dst` の順を確認。インライア数/比と再投影誤差を必ず検証 |
| 重なり境界に段差（シーム）が出る | 単純な上書き合成。露出差がそのまま境界に | フェザー（加重平均）でブレンド。重みマップも同じ `H` で warp |
| `m, n = pair` で `ValueError` | `knnMatch` が長さ 1 のペアを返した | `if len(pair) < 2: continue` を入れてから展開（第5回の復習） |
| `cv2.Stitcher` が黒画像/エラー | `status` を確認せず使った、重なり/特徴不足 | `status == cv2.Stitcher_OK` を必ず確認。PANORAMA↔SCANS を切替 |
| matplotlib で色が変（赤青反転） | BGR のまま渡した | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` してから `imshow` |
| 毎回インライア数が微妙に変わる | RANSAC は内部で乱数を使う | 再現したいときは `cv2.setRNGSeed(0)` を冒頭で呼ぶ |

この表の 8 項目が、本章で時間を取られる原因のほぼ全てです。逆にいえば、これらを自分で説明でき、回避コードも書けるようになれば、この章のゴールに到達したといえます。

## 🛠 章末ミニプロジェクト — 4枚パノラマ + 平面物体のまっすぐ化 + 品質レポート

ここまでの部品（特徴マッチ → `findHomography(RANSAC)` → 再投影誤差 → キャンバス計算 → `warpPerspective` → フェザー合成 → ホモグラフィの合成 → `cv2.Stitcher` 比較 → SSIM 評価 → 平面物体の位置合わせ）を、**1 本のパイプラインに統合する総合課題**が `mini_project.py` です。これにより、この章の学びが「個別の関数」の寄せ集めではなく「つながった一連の処理」として手に入っているかを、実際に動く完成形で確認します。

`mini_project.py` が行うことは次の 5 段です（すべて CPUのみ・合成データ・ネット/カメラ不要・headless 安全）。

1. **[A] 4 視点を合成**: 同じ平面シーンを左→右に少しずつ首を振って撮った 4 枚 `mp_view_0..3.png` を生成。
2. **[B] 手作りパノラマ**: 4 枚を順次つないで `mp_panorama_manual.png` を作り、**ペアごとのインライア数・先頭ペアの再投影誤差・インライア比**を数値化。
3. **[C] `cv2.Stitcher` と比較**: 自動合成 `mp_panorama_stitcher.png` を作り、手作り版との**重なり領域 SSIM** を測る（採用モード・サイズ・所要時間も記録）。
4. **[D] 平面物体のまっすぐ化**: 「書類カード」を散らかった背景に射影で貼り込み、テンプレート→シーンのマッチングで検出して四辺形を描き（`mp_document_detected.png`）、**逆ホモグラフィ `inv(H)` で正面へ復元**（`mp_document_rectified.png`）。元カードとの SSIM で「どれだけ元通りに戻せたか」を測る。
5. **[E] レポート化**: 全指標を `mp_report.json` に、図を `mp_summary.png`（2×2 のまとめ）に保存。

この完成形を**読んで理解し、自分で拡張できる**ようになることが到達目標です。腕試しに、次の発展課題へ挑戦してみてください。

- 視点数を 4→6 に増やし、端の画像ほど累積誤差で SSIM が下がる様子を `mp_report.json` で観察する。
- フェザー合成を**単純上書き**に差し替え、重なり境界のシームが SSIM に与える影響を比べる。
- `dst_quad` を変えて書類をより浅い角度（強いパース）で貼り、再投影誤差と rectify SSIM の悪化を確かめる。
- 書類復元（rectify）した正面画像に対し、第4回のしきい値処理を掛けて「スキャン風の白黒文書」に仕上げる。

```bash
uv run python lectures/06_homography_panorama/mini_project.py
# → outputs/06_homography_panorama/ に mp_*.png と mp_report.json が出る
```

## ✅ 到達チェックリスト

この章を「できた」と言うために、次の項目を自分の言葉で説明でき、かつコードで再現できるかを確認してください。

- [ ] アフィン変換（2×3, 自由度 6, 最低 3 点）とホモグラフィ（3×3, 自由度 8, **最低 4 点**）の違いと、それぞれの守備範囲を説明できる。
- [ ] `cv2.findHomography(src, dst, cv2.RANSAC, thr)` の**向き**（`src→dst`）を意識して呼び、`H is None` を必ずチェックできる。
- [ ] RANSAC が返す**インライア mask** の数・比を品質の一次指標として読み、`drawMatches` の `matchesMask` でインライアだけ可視化できる。
- [ ] **再投影誤差**を `perspectiveTransform` で計算し、**インライアだけ**で測るべき理由を説明できる。
- [ ] `perspectiveTransform` で四隅を写し、外接矩形から**キャンバスサイズと平行移動 `T`** を求められる。
- [ ] `warpPerspective` が**原点始まり**である仕様を理解し、`T @ H` を渡して負座標の切れを防げる。
- [ ] **フェザー（加重平均）ブレンド**で重なりのシームを消せる。重みマップも画素と同じ `H` で warp する理由を言える。
- [ ] 3 枚以上を**ホモグラフィの合成 `M_i = M_{i-1} @ H_i`** で基準フレームへそろえ、順次合成の**累積誤差**に気づける。
- [ ] `cv2.Stitcher` の `status` を確認して使い、手作りパイプラインとの**役割分担（バンドル調整・露出補正・マルチバンド合成）**を説明できる。
- [ ] 逆ホモグラフィ `inv(H)` で平面物体を**正面へまっすぐ化**でき、`mini_project.py` を読み解いて拡張できる。

## ❓ よくある落とし穴・FAQ・デバッグ

**Q. パノラマが「ぐちゃぐちゃ」に歪む。まず何を疑う？**
A. 9 割は (1) `H` の**向きが逆**、(2) **外れ値で `H` が破綻**、のどちらかです。デバッグの順序は「① インライア数/比を表示（極端に少なければ対応が悪い）→ ② 再投影誤差をインライアのみで確認（大きければ `H` が悪い）→ ③ `src/dst` の順を確認」とたどります。本章の合成データでは、健全時にインライア比 9 割・再投影誤差 1px 前後が出るので、それと比べれば一目で異常が分かります。

**Q. `warpPerspective` したら画像の左や上が切れる。**
A. `warpPerspective` の出力は、常に**原点 (0,0) 始まり**です。そのため `H` が画像を負の座標へ動かすと、その部分は無言で捨てられてしまいます。対策として、四隅を `perspectiveTransform` で写して最小座標 `(x_min, y_min)` を求め、`T = [[1,0,-x_min],[0,1,-y_min],[0,0,1]]` を作って **`T @ H`** を渡し、キャンバスサイズも外接矩形から決めます（5・6 節）。

**Q. 重なり境界に縦の段差（シーム）が出る。位置合わせは合っているのに。**
A. それは推定の問題ではなく、**合成（ブレンド）の問題**です。単純な上書きは、露出差をそのまま境界に出してしまいます。そこでフェザー（縁ほど軽い重みの加重平均）で溶かしてください。重要なのは、**重みマップも画素と同じ `H` で warp する**ことです（7 節）。`mp_summary.png` で naive と feather を見比べると、その効果がよく分かります。

**Q. `findHomography` が `None` を返す。**
A. 原因は、対応点が **4 未満**か、点が一直線に並ぶ（退化配置）かのどちらかです。これは `cv2.imread` の `None` 戻りと同じ静かな罠なので、必ず `if H is None:` で受けます。良マッチ数を増やすには、比率テストのしきい値を少し緩める（0.75→0.8）か、`ORB_create(nfeatures=...)` を増やします。

**Q. 平面物体検出で枠が物体に張り付かず、1 点に潰れた四辺形になる。**
A. 誤対応が多く、RANSAC が**少数の偽の合意**に乗ってしまった兆候です（インライアが数個だけ、なのに再投影誤差は小さい、という状態）。これは背景に物体と似た特徴（同じ文字など）が多いと起きます。対策は「物体側の特徴を強く・大きくする」「背景の競合特徴を減らす」「インライア数の**下限**（例: 15 以上）を設けて、満たさなければ検出失敗として扱う」の 3 つです。実際 `mini_project.py` では、背景を低周波（角の少ない）画像にして、書類カードを主役に据えています。

**Q. RANSAC の結果が実行ごとに微妙に変わる。**
A. RANSAC は内部で乱数を使います。再現したいときはスクリプト冒頭で `cv2.setRNGSeed(0)` を呼びます（本章の全スクリプトがそうしています）。

**Q. matplotlib に渡すと色が変（赤と青が入れ替わる）。**
A. OpenCV は BGR、matplotlib は RGB です。`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` を挟んでから `imshow` してください。保存系も同様です。

## 🚀 発展トピック・参考

- **バンドル調整（bundle adjustment）**: 順次合成の累積誤差を、全画像のカメラパラメータを**まとめて**最適化して抑える手法。`cv2.Stitcher` が内部で行っているのはこれ。手作り版で端の画像がずれる原因の本質です。
- **投影モデル（円筒・球面）**: 視野角が広い回転パノラマでは、平面に貼ると端が極端に引き伸ばされます。`Stitcher_PANORAMA` は円筒/球面に投影してこれを回避します。湾曲した仕上がりはこの投影の証拠。
- **シーム探索（graph-cut seam finding）+ マルチバンド合成**: フェザーより高度な合成。継ぎ目を「目立たない経路」に通し（graph cut）、周波数帯ごとに別々に混ぜる（multi-band）ことで、動く物体や露出差にも強くなります。
- **ロバスト推定の改良**: 素の RANSAC の上位版に **USAC / MAGSAC++**（`cv2.USAC_MAGSAC` を `findHomography` の `method` に指定可能）。外れ値が多い・しきい値設定が難しい場面で安定します。
- **平面物体検出 → AR**: 本章の「テンプレートの四隅を `perspectiveTransform` で投影」は、平面マーカへ CG を重ねる AR の基礎そのもの。次に進むなら姿勢推定（`solvePnP`）へ。
- **次章への接続**: ホモグラフィは「平面」を結ぶ変換でした。第7回のカメラキャリブレーション／ステレオでは、平面に限らない**一般の 3 次元**幾何（基本行列・基礎行列・エピポーラ拘束）へ進みます。本章の「対応点 → ロバスト推定 → 行列」の型がそのまま土台になります。
- 参考: OpenCV 公式チュートリアル [Feature Matching + Homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) ／ [High level stitching API (Stitcher class)](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html) ／ [`findHomography` リファレンス](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

## 💡 実践ユースケース集

この章の「対応点 → ホモグラフィ → warp → 合成」という流れは、教材を離れた現実の小ツールにそのまま化けます。ここでは代表的な 3 つの応用を、**何に使うか / 作り方の要点 / 注意**の 3 点セットで紹介します。なお 1 つ目は、実際に動く出発点 `use_case.py` として同梱しているので、手を動かしながら自分の道具へと育ててください。

### A. パノラマ作成ツール（同梱・`use_case.py`）

- **何に使うか**: スマホやカメラでその場で首を振って撮った「左→右に少しずつ重なる複数枚」を、1 枚の横長パノラマに自動で貼り合わせる。旅行写真・部屋の全景・ホワイトボードの横長撮影などをつなぐ実用ツール。
- **作り方の要点**: 隣り合うペアごとに ORB マッチ → `findHomography(RANSAC)` で「i → i-1」のホモグラフィを推定し、行列を掛け合わせて全画像を先頭フレームへそろえ、四隅からキャンバスを決めてフェザー合成する（＝本章 2〜8 節そのもの）。本ツールは手作りパイプライン（`cv_helpers.build_panorama`）を本筋にしつつ、難しい写真では `cv2.Stitcher`（自動・露出補正/マルチバンド合成つき）へ自動でフォールバックし、「できるだけ絵を出し切る」設計にしています。
- **注意**: 重なりが足りない（隣どうし 30% 未満）と、対応点が取れず破綻します。並び順は「ファイル名の昇順 = 左→右」と解釈するので、`01_`, `02_`, … と番号を付けてください。また、スマホ写真は EXIF の向き情報で横倒しに読まれることがあるので、`ImageOps.exif_transpose` で正してから渡すと安定します。
- **実行とデータ配置**:

  ```bash
  # 既定（data/06_homography_panorama/ を読む。無ければ合成 3 視点で必ず完走＝exit 0）
  uv run python lectures/06_homography_panorama/use_case.py

  # 自分の写真で: data/06_homography_panorama/ に 01_left.jpg 02_mid.jpg 03_right.jpg … と
  #              「左→右へ重なる順」で置く（フォルダは初回実行時に自動作成される）
  uv run python lectures/06_homography_panorama/use_case.py --ratio 0.8 --max-width 1280

  # 難しめの写真は最初から自動合成で / ローカル GUI ならプレビュー（headless では自動スキップ）
  uv run python lectures/06_homography_panorama/use_case.py --stitcher
  uv run python lectures/06_homography_panorama/use_case.py --show
  ```

  出力は `outputs/06_homography_panorama/use_case_panorama.png`（完成パノラマ）と `use_case_overview.png`（入力サムネ＋完成図）。

- **`mini_project.py` との違い**: ミニプロジェクトは学びを採点・検証する**合成データ専用の総合課題**（4枚パノラマ＋平面物体のまっすぐ化＋SSIM＋JSON レポート）。`use_case.py` は**自分の実データを 1 つの成果物に変える現実の小ツール**で、入力フォルダ・比率テストしきい値・縮小幅などを引数で運用できます。
- **拡張アイデア**: 総当たりで隣接ペアのインライア数を測って「フォルダに入れる順番を気にしない自動順序推定」へ / 重なり領域の平均輝度を合わせる露出補正 / 合成後の黒余白を `boundingRect` で自動トリミング / `Stitcher_PANORAMA`（円筒・球面投影）に寄せて広角・360 度に対応。

### B. スマホ書類スキャナ（台形補正・まっすぐ化）

- **何に使うか**: 斜めから撮った書類・レシート・名刺・ホワイトボードを、正面から撮ったように「まっすぐ」に補正してスキャン風の画像にする。市販のスキャナアプリの中核がこれです。
- **作り方の要点**: まず書類の 4 隅を見つけ（`Canny` → `findContours` → `approxPolyDP` で 4 点の四辺形を探す、または平面テンプレートとのマッチング）、その四辺形を「正面の長方形」へ写す `getPerspectiveTransform` を作り、`warpPerspective` で引き戻す。本章 5 節の「四隅を `perspectiveTransform`」と、ミニプロジェクト [D] の「逆ホモグラフィ `inv(H)` で rectify」が、そのまま土台になります。仕上げに第4回の適応的しきい値（`adaptiveThreshold`）を掛ければ、白黒文書になります。
- **注意**: 4 隅の検出が肝で、背景に書類と似た直線（机の縁など）が多いと誤検出します。そこで「四角形の面積が画面の一定割合以上」「凸四角形であること」といったフィルタを入れて頑健にします。また、射影が浅い（ほぼ真横）と、復元画像が大きく引き伸ばされて画質が落ちる点にも注意してください。

### C. 平面物体検出 → 簡易 AR / 看板差し替え

- **何に使うか**: 雑然としたシーンの中からポスター・本の表紙・ロゴ看板などの**平面物体**を見つけて枠を描く、あるいはその枠に別の画像（広告・CG・翻訳テキスト）を貼り替える（AR・自動ローカライズの基礎）。
- **作り方の要点**: 探したい平面をテンプレートとして、シーンに対し特徴点マッチング → `findHomography` で「テンプレート → シーン」の `H` を推定。テンプレートの四隅を `perspectiveTransform` で写せば、シーン内で物体が占める**傾いた四辺形**が得られます（本章 5 節・`01_homography_ransac.py` の `[7]`）。差し替えは、貼りたい画像をその四辺形へ `warpPerspective` してマスク合成するだけです。
- **注意**: 背景に物体と同じ文字・模様が多いと、RANSAC が**少数の偽の合意**に乗り、四辺形が 1 点に潰れます。これを避けるには、インライア数に**下限**（例: 15 以上）を設け、満たさなければ「検出失敗」として扱うのが定石です。なおテンプレート側は、特徴の豊富な平面に限ります（のっぺりした単色は不可）。

## 動かし方

本章のスクリプトは、すべて CPUのみ・ネット非依存・追加依存なしで動きます（サンプル画像は各スクリプトが `numpy`/`cv2` で合成生成します）。結果はすべて `outputs/06_homography_panorama/` に画像・JSON として保存され、画面表示はしません（headless 安全）。

### 📂 スクリプト一覧

| ファイル | 役割 | 主な出力 |
| --- | --- | --- |
| `cv_helpers.py` | 共通ヘルパ（合成シーン生成・ORB マッチ・推定/評価部品・SSIM）。単体実行でスモークテスト | `helper_view_*.png` |
| `01_homography_ransac.py` | ホモグラフィ推定の核心（RANSAC・インライア・再投影誤差・失敗モード・平面物体検出） | `01_*.png` |
| `02_panorama_manual.py` | 手作りパノラマ（warp・キャンバス・シーム/フェザー・3枚つなぎ） | `02_*.png` |
| `03_stitcher_compare.py` | `cv2.Stitcher` との比較（自動合成・重なり領域 SSIM） | `03_*.png` |
| `mini_project.py` | **章末ミニプロジェクト**（4枚パノラマ + 平面物体のまっすぐ化 + 品質レポート） | `mp_*.png` / `mp_report.json` |
| `use_case.py` | **実践ユースケース**：手持ち写真をつなぐパノラマ作成ツール（実データ優先・合成フォールバック） | `use_case_panorama.png` / `use_case_overview.png` |
| `exercises.py` | 演習 8 問（TODO を実装 → 自己採点。未実装でも FAIL 表示で正常終了） | （標準出力） |
| `exercises_solutions.py` | 演習の模範解答（採点ロジックを共有して全問 PASS する完成形） | （標準出力） |

```bash
# 1) ホモグラフィ推定の核心（RANSAC・インライア・再投影誤差・物体検出）
uv run python lectures/06_homography_panorama/01_homography_ransac.py

# 2) 手作りパノラマ（warpPerspective・キャンバス・シーム/フェザー・3枚つなぎ）
uv run python lectures/06_homography_panorama/02_panorama_manual.py

# 3) cv2.Stitcher との比較（自動合成・重なり領域 SSIM）
uv run python lectures/06_homography_panorama/03_stitcher_compare.py

# 4) 章末ミニプロジェクト（統合課題：4枚パノラマ + 書類のまっすぐ化 + JSON レポート）
uv run python lectures/06_homography_panorama/mini_project.py

# 5) 実践ユースケース：手持ち写真をつなぐパノラマ作成ツール（実データ優先・合成フォールバック）
uv run python lectures/06_homography_panorama/use_case.py
# 自分の写真で試す: data/06_homography_panorama/ に 01_.., 02_.. と左→右の重なり写真を置く

# 演習（TODO を実装 → 自己採点。未実装でも FAIL 表示で正常終了する）
uv run python lectures/06_homography_panorama/exercises.py
# 行き詰まったら模範解答で全 PASS の挙動を確認（まずは自力で！）
uv run python lectures/06_homography_panorama/exercises_solutions.py
```

実行後は、`outputs/06_homography_panorama/` の画像を順に開いて、本文の確認ポイントと照らし合わせてください。特に `01_matches_all.png`→`01_matches_inliers.png`（RANSAC で間引かれる様子）、`01_object_detected.png`（傾いた物体に張り付く緑枠）、`02_pano_naive.png`↔`02_pano_feather.png`（シームの有無）、`03_compare.png`（手作り版と Stitcher 版の投影モデルの違い）、`mp_summary.png`（統合課題のまとめ）を見比べると、各節の内容が一気に腑に落ちます。また `cv_helpers.py` 単体を実行すると、合成 2 視点と真のホモグラフィを生成するスモークテストになります。

## まとめ

この章では、特徴点マッチングの応用として、対応点から `cv2.findHomography(RANSAC)` でホモグラフィを推定し、インライア mask の数/比と再投影誤差でその品質を数値検証し、`cv2.warpPerspective` で 2 枚（さらにホモグラフィの合成で複数枚）を貼り合わせる手作りパノラマパイプラインを、最初から最後まで自分の手で組み立てました。加えて、シームを消すフェザーブレンド、四隅の `perspectiveTransform` によるキャンバス計算と平面物体検出、そして高レベル `cv2.Stitcher`（バンドル調整・露出補正・マルチバンド合成）との比較と使い分けまでを、すべて「自分で再現し説明できる」レベルで扱ってきました。

ここで身につけた「対応点 → ロバスト推定 → 行列の合成 → warp」という流れは、次回以降のカメラキャリブレーション、ステレオ・エピポーラ幾何、さらには AR や物体姿勢推定にまで、そのまま地盤として効いてきます。まずは演習を自力で全問 PASS させ、`findHomography` の `None` チェック・インライア検証・`T @ H` の合成という定石を手に馴染ませてから、次へ進んでください。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4 ／ opencv-python-headless 4.13（`cv2` 4.13.0、SIFT/ORB/Stitcher は本体同梱・contrib 不要）／ Pillow 12.2 ／ matplotlib 3.10
