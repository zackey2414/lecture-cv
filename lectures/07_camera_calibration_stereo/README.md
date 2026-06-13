# 第7回 カメラキャリブレーション・ステレオ・エピポーラ幾何 — calibrateCamera・undistort・StereoSGBM・reprojectImageTo3D

> トラック: **古典CV** ／ レベル: **中級** ／ 依存: numpy・opencv-python-headless・pillow・matplotlib のみ（torch/faiss は使いません・追加依存グループなし）

## 🎯 この章のゴール

この章を終えたとき、あなたは「カメラというハードウェアを、数式（内部行列 `K` と歪み係数 `dist`）で表し、その数式を実画像から逆算（=キャリブレーション）し、得たパラメータで画像の歪みを取り除き、さらに 2 台のカメラから 3 次元の奥行きを復元する」という、古典 CV の幾何パイプラインを最初から最後まで自分の手で書けるようになります。前回（第6回）のホモグラフィは「平面どうしを結ぶ 2D の変換」でしたが、本章は一歩進んで「カメラと 3 次元世界の関係そのもの」を扱います。鍵になるのは、3D 点 → 画像画素への写像を司る `K`（焦点距離 `fx,fy` と主点 `cx,cy`）と、レンズが直線を曲げてしまう `dist`（歪み係数）です。

具体的な到達点は 3 つです。1 つ目は**校正**で、チェスボードを複数視点から撮り、`findChessboardCorners → cornerSubPix` で角を検出し、`calibrateCamera` で `K` と `dist` を推定します。品質は戻り値の **RMS 再投影誤差（画素単位）**で測り、同じ値を `projectPoints` で**自前計算して一致を確認**します。2 つ目は**歪み補正**で、`undistort` と `getOptimalNewCameraMatrix`（有効領域 ROI）で曲がった直線をまっすぐに戻します。3 つ目は**ステレオ**で、エピポーラ幾何（`findFundamentalMat`）と平行化（`stereoRectify`）を理解した上で、`StereoSGBM` の視差マップから `reprojectImageTo3D` で 3 次元点群と深度を復元し、既知の奥行きと突き合わせて検証します。

本章のもう一つの軸は「**実データ無し・GPU 無し・ネット非依存でも完走できる**」ことです。チェスボード画像・歪み画像・ステレオ対は、すべて**既知の真値カメラから `numpy`/`cv2` で合成生成**します。答え（真値 `K, dist` や正解の奥行き）が分かっているからこそ、「校正で推定した値が真値にどれだけ近いか」「視差から復元した深度が正しいか」を自分の目で定量確認できます。最後に、ここで学ぶ**古典のステレオ深度**（2 眼・絶対距離が出る）と、第27回で扱う**深層単眼深度（Depth Anything）**（1 枚・相対深度）の違いと使い分けにも触れ、古典と深層を地続きで理解します。

---

## 1. カメラモデル — 内部行列 `K` と歪み係数 `dist`

カメラがやっていることは、本質的には「3 次元世界の点を、2 次元の画像平面（センサ）へ落とす」写像です。これを**ピンホールカメラモデル**で表すと、3D 点 `(X, Y, Z)`（カメラ座標）は `x = X/Z, y = Y/Z` と正規化され、内部行列 `K` を掛けて画素 `(u, v)` になります。`K` の中身は焦点距離 `fx, fy`（画素単位の「拡大率」）と主点 `cx, cy`（光軸が画像に刺さる点、ふつう画像中心付近）です。`fx` が大きいほど望遠（同じ物体が大きく写る）、小さいほど広角になります。そして `K` はカメラ＋レンズ固有の値なので、一度測れば（ズームしない限り）使い回せます。

<figure class="lec-fig"><svg viewBox="0 0 600 280" role="img" aria-label="ピンホールカメラモデル。3D点がカメラ中心へ向かう光線が画像平面と交わる点が画素(u,v)。焦点距離fと主点cx,cyが内部行列Kを成す" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="455" y1="150" x2="74" y2="150" stroke="#71717a" stroke-width="2"/><polygon points="66,150 78,145 78,155" fill="#71717a"/><text x="150" y="168" font-size="12" fill="#71717a">光軸 (Z軸)</text><rect x="332" y="70" width="16" height="160" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/><text x="340" y="252" text-anchor="middle" font-size="12" fill="#2563eb">画像平面</text><line x1="95" y1="72" x2="455" y2="150" stroke="#c2410c" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.7"/><circle cx="455" cy="150" r="6" fill="#18181b"/><text x="455" y="178" text-anchor="middle" font-size="12" fill="#3f3f46">カメラ中心</text><circle cx="95" cy="72" r="6" fill="#ea580c"/><text x="92" y="60" font-size="14" font-weight="700" fill="#c2410c">P = (X, Y, Z)</text><circle cx="340" cy="150" r="4" fill="#2563eb"/><text x="356" y="153" font-size="12" fill="#2563eb">主点 (cx, cy)</text><circle cx="340" cy="125" r="5" fill="#ea580c"/><text x="356" y="121" font-size="13" font-weight="700" fill="#c2410c">画素 (u, v)</text><line x1="340" y1="205" x2="455" y2="205" stroke="#16a34a" stroke-width="1.5"/><polygon points="340,205 348,201 348,209" fill="#16a34a"/><polygon points="455,205 447,201 447,209" fill="#16a34a"/><text x="397" y="223" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">焦点距離 f</text></svg><figcaption>ピンホールカメラモデルでは、3D点 <code>P=(X,Y,Z)</code> から<b>カメラ中心</b>へ向かう光線が<b>画像平面</b>と交わる点が画素 <code>(u,v)</code> です。式で書くと <code>(u,v) ∝ K·(X/Z, Y/Z, 1)</code> で、<b>内部行列 K</b> は<b>焦点距離 fx,fy</b>（拡大率）と<b>主点 cx,cy</b>（光軸が画像に刺さる点）でできています。fx が大きいほど望遠、小さいほど広角です。</figcaption></figure>

しかし現実のレンズは理想的なピンホールではなく、特に広角ほど**直線を曲げて**写します。これを表すのが歪み係数 `dist = (k1, k2, p1, p2, k3)` です。`k1, k2, k3` は**径方向歪み**（中心から外周へ向かう放射状の歪み。`k1<0` なら樽（バレル）型＝外周が縮む、`k1>0` なら糸巻き（ピンクッション）型）、`p1, p2` は**接線方向歪み**（レンズとセンサの微妙な傾きによるズレ）です。本章の真値カメラ（`cv_helpers.TRUE_K / TRUE_DIST`）は `fx=fy=600, cx=320, cy=240, k1=-0.28, k2=0.10` で、スマホ広角程度の中くらいの樽型歪みを持たせています。

<figure class="lec-fig"><svg viewBox="0 0 620 250" role="img" aria-label="レンズの径方向歪み。歪みなしの格子に対し、k1が負だと樽型で外周が縮み、正だと糸巻き型。中心ほど歪みは小さい" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="310" y="30" text-anchor="middle" font-size="13" fill="#3f3f46">径方向歪み：直線がどう曲がるか</text><rect x="65" y="70" width="110" height="110" fill="none" stroke="#16a34a" stroke-width="2"/><line x1="120" y1="70" x2="120" y2="180" stroke="#16a34a" stroke-width="1.5"/><line x1="65" y1="125" x2="175" y2="125" stroke="#16a34a" stroke-width="1.5"/><circle cx="120" cy="125" r="3" fill="#71717a"/><text x="120" y="208" text-anchor="middle" font-size="14" font-weight="700" fill="#15803d">歪みなし</text><path d="M 262 82 Q 310 66 358 82 Q 374 125 358 168 Q 310 184 262 168 Q 246 125 262 82 Z" fill="none" stroke="#ea580c" stroke-width="2"/><line x1="310" y1="74" x2="310" y2="176" stroke="#ea580c" stroke-width="1.5"/><line x1="254" y1="125" x2="366" y2="125" stroke="#ea580c" stroke-width="1.5"/><circle cx="310" cy="125" r="3" fill="#71717a"/><text x="310" y="208" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">樽型 (k1 ＜ 0)</text><path d="M 445 72 Q 500 88 555 72 Q 539 125 555 178 Q 500 162 445 178 Q 461 125 445 72 Z" fill="none" stroke="#2563eb" stroke-width="2"/><line x1="500" y1="80" x2="500" y2="170" stroke="#2563eb" stroke-width="1.5"/><line x1="453" y1="125" x2="547" y2="125" stroke="#2563eb" stroke-width="1.5"/><circle cx="500" cy="125" r="3" fill="#71717a"/><text x="500" y="208" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">糸巻き (k1 ＞ 0)</text></svg><figcaption>現実のレンズは直線を曲げて写します（<b>径方向歪み</b>）。中心に近いほど歪みは小さく、外周ほど大きくなります。<code>k1＜0</code> は<b>樽（バレル）型</b>で外周が縮み（本章の真値は <code>k1=-0.28</code>）、<code>k1＞0</code> は<b>糸巻き（ピンクッション）型</b>です。歪み係数 <code>dist=(k1,k2,p1,p2,k3)</code> のうち <code>p1,p2</code> は<b>接線方向歪み</b>を表します。</figcaption></figure>

```python
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0,   0,   1]], dtype=np.float64)   # fx,fy,cx,cy
dist = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])     # (k1, k2, p1, p2, k3)
```

この `K` と `dist` が、本章を貫く「カメラの正体」です。`cv2.projectPoints(objp, rvec, tvec, K, dist)` はこの数式そのもので、3D 点群 `objp` を、姿勢 `(rvec, tvec)` のカメラから見た画素へ（歪みも含めて）落とします。本章の合成チェスボード画像は、まさにこの `projectPoints` で各マスの頂点を投影して描いています。だからこそ「本物の写真と同じ幾何」を持ち、校正で `K, dist` をきれいに逆算できるのです。校正とは要するに、`projectPoints` の逆問題（多数の 3D-2D 対応から `K, dist` を解く）に他なりません。

## 2. チェスボード校正の全体像 — なぜチェスボードか

校正のゴールは `K` と `dist` を求めることですが、そのためには「3 次元のどの点が、画像のどの画素に写ったか」という**対応**が大量に必要になります。そこで効くのがチェスボードです。チェスボードが校正に向くのは、(a) 角（コーナー）の 3D 座標が設計上**完全に既知**（平らな盤の上で `(0,0,0),(1,0,0),...` と並ぶ）であり、(b) 白黒の市松模様なのでコーナーを**サブピクセル精度で安定検出**できる、という 2 つの性質を併せ持つからです。したがって、1 枚の盤を色々な角度・距離から 10〜20 枚撮るだけで、十分な対応が集まります。

では、なぜ「複数視点」が必要なのでしょうか。その理由は校正の数学（Zhang の手法）にあります。1 枚の平面だけでは `K` と盤の姿勢が分離できず、解が定まりません。一方、盤を**傾けて**撮ると遠近（パース）の情報が入り、さらに視点を**変える**と `K` が全視点共通という拘束が効くため、ここで初めて `K` と `dist` が一意に絞れます。だからこそ本章の `make_camera_poses` は、盤を ±0.28rad ほど色々に傾け、距離も変えた 15 視点を生成します。逆に、傾きの無い正対画像ばかりでは、たとえ枚数を増やしても校正は不安定なままです。

下の表が、校正パイプラインの全体像です。各段は次節以降で 1 つずつ掘り下げますが、まずこの流れ（3D 用意 → 検出 → 推定 → 評価 → 補正）を頭に入れてください。`01_calibrate_camera.py` がこの表の上 4 段を、`02_undistort.py` が最下段を担当します。

| 段階 | API | 役割 | 本章のスクリプト |
| --- | --- | --- | --- |
| 3D 座標を用意 | （`np.mgrid`） | 盤の内側角の 3D 点 `objp`（Z=0） | `01`（`chessboard_object_points`） |
| コーナー検出 | `findChessboardCorners` + `cornerSubPix` | 各画像の 2D 角を高精度に取る | `01` |
| パラメータ推定 | `calibrateCamera` | 多数の 3D-2D 対応から `K, dist` を解く | `01` |
| 品質評価 | `projectPoints`（RMS 再投影誤差） | 推定の良さを画素単位で測る | `01` |
| 歪み補正 | `undistort` / `getOptimalNewCameraMatrix` | 求めた `K, dist` で歪みを除去 | `02` |

<figure class="lec-fig"><svg viewBox="0 0 648 232" role="img" aria-label="カメラ校正の処理パイプライン。3D点を用意→コーナー検出→K,dist推定→品質評価→歪み補正の5段を左から右へ進む。前4段は01スクリプト、最後の補正は02スクリプトが担当" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="324" y="30" text-anchor="middle" font-size="14.5" fill="#3f3f46">校正パイプライン：3D用意 → 検出 → 推定 → 評価 → 補正</text><rect x="8" y="56" width="114" height="120" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><circle cx="65" cy="86" r="13" fill="#ea580c"/><text x="65" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">1</text><text x="65" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">3D点を用意</text><text x="65" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">objp (Z=0)</text><rect x="137" y="56" width="114" height="120" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><circle cx="194" cy="86" r="13" fill="#ea580c"/><text x="194" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">2</text><text x="194" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">コーナー検出</text><text x="194" y="147" text-anchor="middle" font-size="10.5" fill="#52525b">findChessboard</text><text x="194" y="163" text-anchor="middle" font-size="10.5" fill="#52525b">→ cornerSubPix</text><rect x="266" y="56" width="114" height="120" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><circle cx="323" cy="86" r="13" fill="#ea580c"/><text x="323" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">3</text><text x="323" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">K, dist 推定</text><text x="323" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">calibrateCamera</text><rect x="395" y="56" width="114" height="120" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><circle cx="452" cy="86" r="13" fill="#ea580c"/><text x="452" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">4</text><text x="452" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">品質評価</text><text x="452" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">RMS再投影誤差</text><rect x="524" y="56" width="114" height="120" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><circle cx="581" cy="86" r="13" fill="#2563eb"/><text x="581" y="91" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">5</text><text x="581" y="126" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">歪み補正</text><text x="581" y="150" text-anchor="middle" font-size="10.5" fill="#52525b">undistort</text><line x1="122" y1="116" x2="129" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="137,116 129,112 129,120" fill="#71717a"/><line x1="251" y1="116" x2="258" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="266,116 258,112 258,120" fill="#71717a"/><line x1="380" y1="116" x2="387" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="395,116 387,112 387,120" fill="#71717a"/><line x1="509" y1="116" x2="516" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="524,116 516,112 516,120" fill="#71717a"/><line x1="8" y1="192" x2="509" y2="192" stroke="#f97316" stroke-width="2"/><text x="258" y="212" text-anchor="middle" font-size="12" font-weight="700" fill="#c2410c">01_calibrate_camera.py が担当</text><line x1="524" y1="192" x2="638" y2="192" stroke="#2563eb" stroke-width="2"/><text x="581" y="212" text-anchor="middle" font-size="11" font-weight="700" fill="#2563eb">02_undistort.py</text></svg><figcaption>カメラ校正の<b>処理パイプライン全体</b>です。<b>3D点を用意</b>（盤の内側角 <code>objp</code>・Z=0）→<b>コーナー検出</b>（<code>findChessboardCorners→cornerSubPix</code>）→<b>K, dist 推定</b>（<code>calibrateCamera</code>）→<b>品質評価</b>（RMS 再投影誤差）→<b>歪み補正</b>（<code>undistort</code>）と左から右へ進みます。前 4 段は <code>01_calibrate_camera.py</code>、最後の歪み補正は <code>02_undistort.py</code> が担当します。</figcaption></figure>

## 3. `findChessboardCorners` → `cornerSubPix`（コーナー検出）

コーナー検出は 2 段構えです。まず `cv2.findChessboardCorners(gray, pattern_size, flags)` が市松模様の格子をおおまかに見つけ、内側角を**整数画素**精度で返します。`pattern_size=(9, 6)` は「内側の角が横 9・縦 6」という意味で、盤のマス数ではなく**内部頂点**の数である点に注意してください（10×7 マスの盤なら内側角は 9×6）。この数を間違えると検出は丸ごと失敗します。`flags` には `CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE` を付けるのが定石で、照明ムラに強くなります。

次に `cv2.cornerSubPix` で**サブピクセル精度**へ精緻化します。`findChessboardCorners` の整数画素のままでは校正精度が頭打ちになるため、各角の周囲の輝度勾配を手がかりに「白黒の交点（サドル点）」を 0.01px 単位で追い込むわけです。この一手が効くと、RMS 再投影誤差は目に見えて下がります。なお、本講座のヘルパ `detect_corners` はこの 2 段をまとめて実行し、検出に失敗した視点では `(False, None)` を返します。

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
found, corners = cv2.findChessboardCorners(gray, (9, 6), flags)
if found:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
```

ここで実務上いちばん大事なのは「**検出は失敗しうる**」という前提です。ブレ・低コントラスト・盤が画面からはみ出る・斜めすぎる、といった要因で簡単に `found=False` になります。そのため `01_calibrate_camera.py` は、失敗した視点を握りつぶさず「スキップしてログ」し、残った視点だけで校正を続けます（合成データの本章では 15/15 成功しますが、実データを前提としたこの設計こそが頑健さを生みます）。検出した角は `cv2.drawChessboardCorners` で描画し `01_detected_corners.png` にモンタージュ保存するので、実行後は「全視点でコーナーが格子状に乗っているか」を必ず目視してください。新しめの OpenCV には、より頑健な `cv2.findChessboardCornersSB`（SB=symmetric-based）もあり、難しい実画像ではこちらが有効です（本体同梱・contrib 不要）。

## 4. `calibrateCamera` と RMS 再投影誤差（評価）

対応が集まれば、校正そのものは `cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)` の 1 行で済みます。ここで `objpoints` は各視点の 3D 点（毎回同じ `objp`）のリスト、`imgpoints` は各視点で検出した 2D 角のリストです。戻り値は `(ret, K, dist, rvecs, tvecs)` で、`rvecs/tvecs` は各視点のカメラ姿勢（外部パラメータ）を表し、そして先頭の `ret` こそが品質指標——**RMS 再投影誤差（画素単位）**です。これは「推定した `K, dist, 姿勢` で 3D 角を画像へ投影し直し、実際の検出角とどれだけズレるか」を全点 RMS で測った値で、**1px 未満なら良好**、数 px もあれば検出ミスや盤精度を疑います。実際、本章の合成データでは **RMS ≈ 0.39px** と良好な値が得られます。

この RMS が「定義どおりか」を自分の手で確かめるのが、本章の評価方針です。具体的には、`projectPoints` で全点を再投影し、検出点との二乗距離を**全点合算して点数で割り、平方根**を取ると、`calibrateCamera` の `ret` と一致します（`mean_reprojection_error`）。一方で、よくある「視点ごとに `cv2.norm(...)/N` を取って平均する」書き方は厳密には RMS と一致しません。そこで本章は総二乗誤差ベースで計算し、ライブラリ値と**完全一致**することを表示します。

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (W, H),
                                                  None, None, flags=cv2.CALIB_FIX_K3)
# 自前 RMS（= 総二乗誤差/総点数 の平方根）が ret と一致する
total_sq, total_n = 0.0, 0
for objp, imgp, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
    proj, _ = cv2.projectPoints(objp, r, t, K, dist)
    total_sq += float(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2))
    total_n  += len(imgp)
my_rms = (total_sq / total_n) ** 0.5     # ret と一致する
```

推定結果を真値と比べると、`fx≈599.5（真値600）, cx≈317（320）, cy≈241（240）, k1≈-0.287（-0.28）` と、主要パラメータはよく一致します。一方で `k2, k3` は**やや不定**になりがちです。理由は単純で、「盤が画像の隅まで届かないと、外周で効く高次の径方向歪みが拘束されない」からです。実際、`k3` を自由にしても RMS はほぼ変わらないのに、`k3` だけが大きく暴れます。したがって非魚眼レンズでは `flags=cv2.CALIB_FIX_K3` で `k3=0` に固定するのが実務の定石であり、本章の `calibrate` も既定でこれを使います。`01_calibrate_camera.py` の `[5]` はこの「k3 固定 vs 自由」を並べて表示するので、高次係数の不定性をぜひ自分の目で確認してください。

## 5. 歪み補正 — `undistort` と `getOptimalNewCameraMatrix`（ROI）

`K, dist` さえ手に入れば、歪みは `cv2.undistort(img, K, dist)` で取り除けます。`02_undistort.py` は、まず歪みが**目で見える**直線格子のクリーン画像を用意し、それを既知の `K, dist` で歪ませた画像（`02_distorted.png`、外周ほど格子線が湾曲）を合成したうえで、`undistort` で補正します。ここで `newCameraMatrix` を省略すると `K` がそのまま使われ、補正後は「歪みだけ消えた元の見え方」に戻ります。だからこそ元のクリーン画像と画素単位で比較でき、本章では中央領域の平均絶対差が小さく（細い線の再標本化ぼけ以外はほぼ一致）、直線が直線へ戻ったことを数値で確認できます。

ただし、補正には避けて通れない副作用があります。樽型歪みを戻すと外周が外へ引き伸ばされ、四隅に**黒い無効領域**ができるのです。これを制御するのが `cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)` で、戻り値の `alpha` と `ROI` がポイントになります。まず `alpha=1` は「元画素を 1 つも捨てない」ため黒縁が出ますが、有効範囲を表す `ROI`（矩形）を見ればその外側が無効だと分かります。一方 `alpha=0` は「黒縁が出ないよう内側だけ残す」ため、端が切り取られます。したがって用途次第で、視野を優先するなら `alpha=1`＋後処理、見栄えを優先するなら `alpha=0`＋ROI クロップ、と使い分けます。

```python
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=1.0, (W, H))
undist = cv2.undistort(distorted, K, dist, None, newK)   # 黒縁あり・ROI が有効範囲
x, y, w, h = roi
undist_cropped = undist[y:y+h, x:x+w]                     # 有効範囲だけ切り出す
```

動画で毎フレーム補正するなら、`undistort` を毎回呼ぶより、`cv2.initUndistortRectifyMap` で**補正マップを一度だけ作り**、各フレームに `cv2.remap` を適用するのが定石です。`02_undistort.py` の `[5]` は、この remap 版が `undistort` と（固定小数点マップの丸め差を除いて）一致することを示します。処理の中身は同じでも、マップを使い回せるぶん動画では速くなる——そしてこの「重い計算は前計算してマップ化する」という発想は、次のステレオ平行化でもそのまま効いてきます。実行後は `02_undistort_compare.png` の 4 枚（クリーン／歪み／補正／ROI 付き）を見比べ、湾曲した格子線が補正でまっすぐに戻る様子を確認してください。

## 6. エピポーラ幾何 — 基礎/基本行列と `findFundamentalMat`

ここからはステレオ（2 眼）に入ります。2 台のカメラで同じ点を見たとき、その対応は無秩序ではなく**エピポーラ拘束**に従います。直感的に言えば「左画像のある点に対応する右画像の点は、必ずある 1 本の直線（**エピポーラ線**）の上にある」という拘束で、これは 2D 全面を探さず 1 次元の線上だけ探せばよい、というステレオ探索の根拠になります。この拘束を数式で表すのが **基礎行列 F（fundamental matrix）** と **基本行列 E（essential matrix）** で、対応点 `x_L, x_R`（同次座標）は `x_R^T · F · x_L = 0` を満たします。なお、`F` は画素座標での拘束を、`E = K^T · F · K` はカメラ座標（正規化）での同じ拘束を表します。

<figure class="lec-fig"><svg viewBox="0 0 620 290" role="img" aria-label="エピポーラ幾何。左画像の点x_Lに対応する右画像の点x_Rは1本のエピポーラ線上にあり、対応点探索が2Dから1Dに減る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="45" y="55" width="180" height="140" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="135" y="44" text-anchor="middle" font-size="13" fill="#3f3f46">左画像</text><circle cx="140" cy="115" r="6" fill="#ea580c"/><text x="152" y="110" font-size="14" font-weight="700" fill="#c2410c">x_L</text><rect x="395" y="55" width="180" height="140" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><text x="485" y="44" text-anchor="middle" font-size="13" fill="#3f3f46">右画像</text><line x1="395" y1="165" x2="575" y2="120" stroke="#2563eb" stroke-width="2.5"/><text x="470" y="80" font-size="13" font-weight="700" fill="#2563eb">エピポーラ線</text><circle cx="430" cy="156" r="3.5" fill="#f97316"/><circle cx="545" cy="127" r="3.5" fill="#f97316"/><circle cx="490" cy="141" r="5" fill="#ea580c" stroke="#c2410c" stroke-width="1.5"/><text x="500" y="138" font-size="14" font-weight="700" fill="#c2410c">x_R</text><line x1="235" y1="120" x2="384" y2="120" stroke="#71717a" stroke-width="2"/><polygon points="392,120 382,115 382,125" fill="#71717a"/><text x="310" y="110" text-anchor="middle" font-size="12.5" font-weight="700" fill="#3f3f46">対応点は この線上だけ</text><text x="310" y="252" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">→ 探索が 2D 全面 から 1D 線上 に減る</text><text x="310" y="278" text-anchor="middle" font-size="12" fill="#52525b">エピポーラ拘束: x_R · F · x_L = 0</text></svg><figcaption>2 台のカメラで同じ点を見ると、<b>左画像</b>のある点 <code>x_L</code> に対応する<b>右画像</b>の点 <code>x_R</code> は、必ず 1 本の<b>エピポーラ線</b>の上にあります（点の奥行きが未知なので線上のどこか）。これが<b>エピポーラ拘束</b> <code>x_R^T · F · x_L = 0</code> で、対応点探索が 2D 全面から 1D 線上へと激減します。<b>平行化</b>済みのペアでは、この線は<b>水平</b>になります。</figcaption></figure>

この `F` は、対応点だけから `cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)` で推定できます（最低 8 点、外れ値除去に RANSAC）。ここで重要なのは、`F` が「カメラがどう並んでいるか」を内包している点です。とくに、左右カメラが**平行に並び光軸も平行**（=平行化済み）であれば、エピポーラ線はすべて**水平**になり、`F` は `[[0,0,0],[0,0,-1],[0,1,0]]` という特別な形に近づきます。これを確かめるのが `03_stereo_sgbm_depth.py` の `[5]` で、合成ステレオ対から ORB で対応を取って `F` を推定し、インライアの `|y_L - y_R|`（左右の縦ズレ）が**ほぼ 0**であること＝エピポーラ線が水平であることを、数値と図（`03_epipolar_lines.png`）で示します。

```python
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
lines = cv2.computeCorrespondEpilines(pts_left, 1, F).reshape(-1, 3)  # 右画像の (a,b,c)
# 平行化済みなら a≈0 となり、ax+by+c=0 はほぼ水平線になる
```

では、なぜ「線が水平」だと嬉しいのか——その答えが、次の平行化（rectify）とステレオマッチングの動機になります。エピポーラ線が画像の行（横方向）に一致すれば、対応点探索は「**同じ行を左右にずらして一致を探す**」だけの 1 次元問題になり、`StereoBM/StereoSGBM` のような高速なブロックマッチングが使えるからです。逆に線が斜めだと、探索は一気に複雑になります。だからこそ実務では、まず**平行化で線を水平に揃えてから**視差を計算するのです。

## 7. ステレオ平行化 — `stereoRectify` と再投影行列 `Q`

**平行化（rectification）** とは、2 枚のステレオ画像を変形して「エピポーラ線を完全に水平・同じ行へ揃える」前処理を指します。現実の 2 台のカメラは完璧には平行に置けないので、まずステレオ校正（`stereoCalibrate`）で左右の相対姿勢 `(R, T)` を求め、次に `cv2.stereoRectify(K1, D1, K2, D2, size, R, T)` で左右それぞれの補正回転 `R1, R2` と新しい射影行列 `P1, P2`、そして**再投影行列 `Q`** を得ます。あとは `R1, R2` から `initUndistortRectifyMap`＋`remap` で実際に画像を平行化するだけで、これは 5 節の歪み補正と全く同じ remap の枠組みです。

本章の合成ステレオ対は最初から平行化済み（歪み 0・左右が水平視差のみ）として作っているので、`stereoRectify` には「回転 `R=単位行列`、並進 `T=(-baseline, 0, 0)`、歪み 0」を渡します。すると返ってくる `Q` こそが、視差から 3D を復元するための核心の行列です。実際 `03_stereo_sgbm_depth.py` の `[4]` は、この `stereoRectify` の `Q` が、手作りした `Q` と**完全一致**することを確認します。

```python
R, T = np.eye(3), np.array([-baseline, 0, 0])      # 平行化済みステレオの相対姿勢
_, _, _, _, Q, _, _ = cv2.stereoRectify(K, D0, K, D0, (W, H), R, T,
                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
# 手作り Q（cx=cx' のとき）も同じ:
Q_manual = np.array([[1, 0, 0, -cx],
                     [0, 1, 0, -cy],
                     [0, 0, 0,  f ],
                     [0, 0, 1/baseline, 0]])
```

ここで `Q` の意味を押さえておきましょう。画素 `(u, v)` と視差 `d` から `[X, Y, Z, W]^T = Q · [u, v, d, 1]^T` を計算すると、実 3D 点は `(X/W, Y/W, Z/W)` で得られます。この `Q` では `W = d/baseline`、`Z = f` なので、実 `Z = f·baseline / d` となり、後述の「視差→深度」の式とぴたり一致します。つまり `Q` とは「視差マップを 3D 点群へ一括変換する係数行列」に他ならず、次節の `reprojectImageTo3D` も内部でこの掛け算を行っています。

## 8. `StereoSGBM` / `StereoBM` の視差マップ

平行化された左右画像があれば、いよいよ各画素の**視差 `d`**（左画像の点が右画像で何画素左にあるか）を求めます。基本となるのは「左画像の小ブロックを、右画像の同じ行に沿ってずらしながら、最も一致する位置を探す」ブロックマッチングです。OpenCV にはこれが 2 系統あり、`cv2.StereoBM`（Block Matching）は局所的で高速な反面、穴（無効画素）が多く、`cv2.StereoSGBM`（Semi-Global Block Matching）は近傍との滑らかさも考慮する**準大域**手法で、高品質だが少し重い、というトレードオフの関係にあります。実際、本章の合成シーンでは SGBM の有効画素率（約 80%）が BM（約 76%）を上回り、境界も滑らかになります。

両者に共通する注意が 2 つあります。第一に、`numDisparities`（視差の探索幅）は **16 の倍数**で、想定最大視差を十分カバーする値にすること（本章は最大視差 64 を見込んで 96）。狭すぎると、手前の物体の視差が振り切れて欠けてしまいます。第二に、戻り値は **×16 の固定小数点**（`int16`）なので、`/16.0` して実視差に直すこと。これを忘れると深度が 16 倍ずれます。なお `StereoBM` はグレースケール専用なので、カラー画像は `cvtColor` で変換してから渡します。

```python
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5,
                             P1=8*3*5**2, P2=32*3*5**2,   # 滑らかさペナルティ（P1<P2）
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
disp = sgbm.compute(left, right).astype(np.float32) / 16.0   # ×16 を戻すのが必須
```

ブロックマッチングが成立する大前提は「**テクスチャがあること**」です。のっぺりした無地の壁は、どこを見ても同じに見えるため、マッチングが一意に決まりません（これも開口問題の一種です）。だからこそ本章の合成シーンは、各物体に高周波ノイズのテクスチャを与えてあります。実画像でも、無地の領域や鏡面・繰り返し模様では視差が荒れる、と覚えておいてください。視差マップは `applyColorMap` で色づけし、`03_disparity_sgbm.png` / `03_disparity_bm.png` に保存します（近い＝視差大＝暖色）。出力後は SGBM と BM を見比べ、穴の量と境界の滑らかさの違いを確認しましょう。

## 9. 視差→深度→3D点群（`reprojectImageTo3D`）と Depth Anything の対比

視差から奥行きへの変換は、平行化済みステレオの基本式 **`Z = f · baseline / d`** で一発です。ここで `f` は焦点距離（px）、`baseline` は左右カメラ間距離、`d` は視差（px）を表します。視差が大きい（手前）ほど `Z` は小さく、逆に視差が小さい（遠く）ほど `Z` は大きくなります。なお視差 `d ≤ 0` の無効画素は、ゼロ割りを避けて深度 0 とします。`03_stereo_sgbm_depth.py` の `[3]` は、各物体の中心で SGBM の視差を測り、この式で深度を出して**正解の奥行きと突き合わせ**ます。その結果、本章では推定視差が正解（12/24/40/64）と完全に一致し、深度も `5.0 / 2.5 / 1.5 / 0.94 m` とぴたり合います。

<figure class="lec-fig"><svg viewBox="0 0 600 290" role="img" aria-label="ステレオ三角測量。奥行きZの物体点が左右カメラに視差dでずれて写り、Z=f×baseline÷dで深度が求まる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><line x1="300" y1="230" x2="300" y2="62" stroke="#15803d" stroke-width="1.4" stroke-dasharray="5 3"/><text x="312" y="150" font-size="13" font-weight="700" fill="#15803d">Z（深度）</text><line x1="170" y1="230" x2="300" y2="48" stroke="#c2410c" stroke-width="1.5"/><line x1="430" y1="230" x2="300" y2="48" stroke="#c2410c" stroke-width="1.5"/><line x1="148" y1="170" x2="252" y2="170" stroke="#2563eb" stroke-width="2"/><line x1="348" y1="170" x2="452" y2="170" stroke="#2563eb" stroke-width="2"/><circle cx="213" cy="170" r="4" fill="#ea580c"/><circle cx="387" cy="170" r="4" fill="#ea580c"/><line x1="170" y1="230" x2="430" y2="230" stroke="#71717a" stroke-width="2"/><text x="248" y="222" text-anchor="middle" font-size="13" font-weight="700" fill="#71717a">基線 B</text><line x1="148" y1="170" x2="148" y2="230" stroke="#16a34a" stroke-width="1.5"/><polygon points="148,170 144,178 152,178" fill="#16a34a"/><polygon points="148,230 144,222 152,222" fill="#16a34a"/><text x="136" y="204" font-size="13" font-weight="700" fill="#15803d">f</text><rect x="158" y="222" width="24" height="16" fill="#3f3f46"/><rect x="418" y="222" width="24" height="16" fill="#3f3f46"/><circle cx="300" cy="48" r="6" fill="#ea580c"/><text x="300" y="36" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">物体点 P</text><text x="170" y="256" text-anchor="middle" font-size="12" fill="#3f3f46">左カメラ</text><text x="430" y="256" text-anchor="middle" font-size="12" fill="#3f3f46">右カメラ</text><rect x="372" y="58" width="200" height="50" rx="6" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><text x="472" y="84" text-anchor="middle" font-size="18" font-weight="700" fill="#18181b">Z = f · B / d</text><text x="472" y="102" text-anchor="middle" font-size="12" fill="#52525b">視差 d = x_L − x_R</text></svg><figcaption>平行化された 2 眼ステレオでは、奥行き <code>Z</code> の物体点 P が左右の画像で少しずれて写ります。このずれが<b>視差 d</b>（= x_L − x_R）で、相似三角形から <b>Z = f · B / d</b> が導けます（<code>f</code>=焦点距離 px、<code>B</code>=基線長）。<b>手前ほど視差 d は大きく、遠いほど小さい</b>ので、視差マップがそのまま深度マップになります。<code>reprojectImageTo3D</code> はこれを全画素へ適用します。</figcaption></figure>

画素ごとの深度だけでなく、**3 次元点群**まで欲しいときは `cv2.reprojectImageTo3D(disp, Q)` を使います。これは 7 節の `Q` を全画素に適用し、各画素を `(X, Y, Z)`（カメラ座標）へ変換するものです。ここから `Z` チャンネルを取り出せば深度マップになり、`(X, Y, Z)` を集めれば点群（後段の 3D 認識や計測の入力）になります。本章ではこれを `03_depth.png`（近いほど明るい）として保存します。

```python
depth = np.zeros_like(disp);  valid = disp > 0
depth[valid] = f * baseline / disp[valid]          # 視差→深度（無効は 0）
points_3d = cv2.reprojectImageTo3D(disp, Q)         # 全画素を (X,Y,Z) へ
Z = points_3d[:, :, 2]                              # 深度チャンネル
```

最後に、古典ステレオと深層単眼深度の**使い分け**を押さえておきましょう。本章の**ステレオ深度**は、2 眼の幾何から `Z = f·baseline/d` で**絶対距離（メートル）**が出るのが強みで、ロボット・自動運転・3D 計測の土台になります。一方、第27回で扱う **Depth Anything V2 のような深層単眼深度**は、**1 枚の画像**から奥行きを推定できる手軽さが魅力ですが、その出力は**相対（逆）深度**でスケール（絶対距離）が不定です。つまり「2 台のカメラと校正が要るが絶対距離が出る古典ステレオ」と「1 枚で済むが相対深度の深層単眼」——この二者を、必要な精度・絶対距離の要否・ハードウェア制約で選び分けるのが、実務の勘どころです。とはいえ両者は競合関係にあるわけではなく、深層の相対深度をステレオや既知サイズ物体で**スケール補正**して併用することもあります。

## 10. よくあるエラーと対処（チェックリスト）

最後に、この章でつまずきやすい点を「症状 → 原因 → 対処」の形で一覧にまとめます。実装中に詰まったら、まずはここを見てください。多くの不具合は、この数個に集約されるはずです。

| 症状 | ほぼ確実な原因 | 対処 |
| --- | --- | --- |
| `findChessboardCorners` が常に `False` | `pattern_size` が「マス数」になっている（正しくは内側角の数） | 10×7 マスなら `(9, 6)`。盤が画面からはみ出てないかも確認 |
| 校正の RMS が数 px と大きい | コーナー誤検出・盤が正対ばかり・視点が少ない | 盤を傾けて 10〜20 枚。`cornerSubPix` を入れる。外れ視点を除く |
| `k2, k3` が真値とかけ離れる | 盤が画像隅まで届かず高次歪みが不定 | `flags=cv2.CALIB_FIX_K3`（非魚眼の定石）。隅まで盤を写す |
| `undistort` 後に四隅が黒い | 樽型歪みを戻すと外周が広がる（仕様） | `getOptimalNewCameraMatrix` の `alpha`/`ROI` で制御・クロップ |
| 深度が 16 倍ずれる | 視差の `×16` 固定小数点を割り忘れ | `disp = sgbm.compute(...) / 16.0` を必ず入れる |
| 視差マップが穴だらけ | テクスチャ不足・`numDisparities` が狭い・未平行化 | テクスチャのある領域で評価。探索幅を 16 の倍数で広げる。先に平行化 |
| 手前の物体だけ視差が欠ける | `numDisparities` が最大視差より小さい | 想定最大視差を見積もり `numDisparities` を増やす |
| `StereoBM` がエラー | カラー画像を渡している | `cvtColor` でグレースケールにしてから `compute` |
| 深度が負・無限大になる | 視差 0/負でゼロ割り | `valid = disp > 0` でマスクしてから計算 |
| matplotlib で色が変（赤青反転） | BGR のまま渡した | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` してから `imshow` |

この表の項目こそが、本章で時間を取られる原因のほぼ全てです。逆に言えば、これらを自分で説明でき、回避コードを書けるようになれば、この章のゴールに到達したと言えます。

## スクリプトと演習の構成

このモジュールは、次のファイル群で構成されます。このうち `cv_helpers.py` だけは「読み物」ではなく「合成データを作る道具」で、3 つのスクリプト・ミニプロジェクト・演習がすべて共有します。最初に一読しておくと、各スクリプトがどんな画像を相手にしているかが腑に落ちるはずです。

| ファイル | 扱う内容 |
| --- | --- |
| `cv_helpers.py` | 合成データ生成（チェスボード／歪み格子／平行化ステレオ対）、校正・再投影の共通部品、出力先管理 |
| `01_calibrate_camera.py` | `findChessboardCorners`→`cornerSubPix`→`calibrateCamera`、RMS 再投影誤差の自前検算、k3 固定の理由 |
| `02_undistort.py` | `undistort`／`getOptimalNewCameraMatrix`（alpha・ROI）、`initUndistortRectifyMap`＋`remap`（動画向き） |
| `03_stereo_sgbm_depth.py` | `StereoSGBM`/`StereoBM` 視差、`stereoRectify` の `Q`、`reprojectImageTo3D`、`findFundamentalMat`（エピポーラ線） |
| `mini_project.py` | 章末ミニプロジェクト（校正→歪み補正→ステレオ深度→3D 実寸計測を一気通貫で統合・PASS 判定と JSON 出力） |
| `use_case.py` | 実践ユースケース: **1 枚写真の平面メジャー**（基準チェスボードのホモグラフィで実寸 mm 計測＋`solvePnP` で距離。`data/` に実写があれば優先・無ければ合成で完走） |
| `exercises.py` | TODO 形式の演習 9 問（自己採点付き。`SHOW_SOLUTION=1` で模範解答に差し替え） |
| `exercises_solutions.py` | 演習 9 問の完全な模範解答（実行すると全 PASS を assert で保証） |

演習は易→難の 9 問で、いずれも本章の核心スキルに対応します。演習 1 は盤の 3D 点 `objp` の生成、演習 2 は平均再投影誤差、演習 3 は視差→深度、演習 4 は再投影行列 `Q` の構築、演習 5 は 1 画素の `Q` による 3D 復元、演習 6 は内部行列 `K` の組み立て、演習 7 は深度→視差（視差と深度が反比例である確認）、演習 8 は「総二乗誤差／総点数の平方根」による厳密な RMS 再投影誤差、演習 9 は基本行列 `E = K^T F K` の計算です。まず TODO を自力で埋め、`exercises.py` を実行して全問 PASS を目指してください。

## 動かし方

すべて CPUのみ・ネット非依存・追加依存なしで動きます（チェスボード画像・歪み画像・ステレオ対は各スクリプトが `numpy`/`cv2` で合成生成します）。リポジトリのルートで以下を順に実行してください。結果はすべて `outputs/07_camera_calibration_stereo/` に保存され、画面表示はしません（headless 安全）。

```bash
# 1) カメラ校正（コーナー検出 → calibrateCamera → RMS 評価 → calib.npz 保存）
uv run python lectures/07_camera_calibration_stereo/01_calibrate_camera.py

# 2) 歪み補正（undistort・getOptimalNewCameraMatrix の ROI・remap 版）
uv run python lectures/07_camera_calibration_stereo/02_undistort.py

# 3) ステレオ（SGBM/BM 視差・stereoRectify の Q・reprojectImageTo3D・エピポーラ）
uv run python lectures/07_camera_calibration_stereo/03_stereo_sgbm_depth.py

# 章末ミニプロジェクト（校正→歪み補正→ステレオ深度→3D 計測の統合。図と指標 JSON を出力）
uv run python lectures/07_camera_calibration_stereo/mini_project.py

# 実践ユースケース（1 枚写真の平面メジャー。基準チェスボードで mm 計測・距離推定）
uv run python lectures/07_camera_calibration_stereo/use_case.py

# 演習（TODO を実装 → 自己採点。未実装でも FAIL 表示で正常終了する）
uv run python lectures/07_camera_calibration_stereo/exercises.py
# 行き詰まったら模範解答で挙動を確認（まずは自力で！）
SHOW_SOLUTION=1 uv run python lectures/07_camera_calibration_stereo/exercises.py
# 完全な模範解答（全 9 問 PASS を確認）
uv run python lectures/07_camera_calibration_stereo/exercises_solutions.py
```

`01` は `calib.npz`（`K, dist`）を保存し、`02` はそれがあれば読み込んで使います（無ければ真値カメラにフォールバックするので、`02` 単体でも動きます）。実行後は `outputs/07_camera_calibration_stereo/` の画像を順に開いて、本文の確認ポイントと照らし合わせてください。特に `01_detected_corners.png`（全視点でコーナーが格子に乗る）、`02_undistort_compare.png`（湾曲した格子が補正で直線に戻る）、`03_disparity_sgbm.png`／`03_depth.png`（奥行きで色が変わる）、`03_epipolar_lines.png`（エピポーラ線が水平）を見比べると、各節の内容が一気に腑に落ちます。`cv_helpers.py` 単体を実行すると、合成と検出のスモークテストになります。

## まとめ

この章では、カメラを `K`（内部行列）と `dist`（歪み係数）で表すピンホール＋歪みモデルから出発し、チェスボードを多視点から撮って `findChessboardCorners → cornerSubPix → calibrateCamera` で `K, dist` を逆算し、RMS 再投影誤差を `projectPoints` で自前検算しながら品質を評価し、`undistort` と `getOptimalNewCameraMatrix`（ROI）で歪みを除去する——という単眼校正の一連を、最初から最後まで自分の手で組み立てました。さらにステレオへ進み、`findFundamentalMat` でエピポーラ拘束を、`stereoRectify` で平行化と再投影行列 `Q` を理解し、`StereoSGBM` の視差マップから `Z = f·baseline/d` と `reprojectImageTo3D` で深度・3D 点群を復元し、既知の奥行きと突き合わせて検証しました。

ここで身につけた「3D-2D 対応 → パラメータ推定 → 再投影で評価 → 視差から 3D」という流れは、AR・物体姿勢推定（PnP）・SfM/SLAM・3D 計測といった応用の共通土台になります。古典ステレオが**絶対距離**を出せること、深層単眼（Depth Anything・第27回）が**1 枚で相対深度**を出せること、その使い分けまで含めて、まずは演習を自力で全問 PASS させ、`pattern_size` の意味・RMS の定義・視差の `×16` 割り戻し・`Z=f·baseline/d` という定石を手に馴染ませてから次へ進んでください。

---

## 🛠 章末ミニプロジェクト — 校正 → 歪み補正 → ステレオ深度の一気通貫

ここまでの部品（カメラモデル・チェスボード校正・歪み補正・視差→深度）を **1 本の幾何パイプライン**に統合する総合課題です。`mini_project.py` を実行すると、次の 4 ステージが一気に走り、各ステージが PASS/FAIL を自己判定します。

1. **Stage A 校正**: 合成チェスボードを 15 視点撮り、`findChessboardCorners → cornerSubPix → calibrateCamera` で `K, dist` を推定。RMS 再投影誤差（自前検算込み）と真値ずれで品質を判定する。
2. **Stage B 歪み補正**: 真値カメラ（`TRUE_K`/`TRUE_DIST`）で歪ませた「実写相当」の格子画像を、**Stage A で校正した `K, dist`** で `undistort` し、元のまっすぐな格子に戻るか（中央領域の平均絶対差 MAD）で **end-to-end の校正品質**を測る。校正で当てたパラメータが実レンズの歪みを実際に補正できるか、という最も実用的なテストです。
3. **Stage C ステレオ深度**: 平行化済みステレオ対から `StereoSGBM` で視差を計算し、`Z = f·baseline/d` で深度に変換。各物体の正解深度との**平均絶対誤差（MAE）**で検証する。
4. **Stage D 3D 計測**: `reprojectImageTo3D` の 3D 点群を使い、最前面物体の**メートル幅**を「画素幅 × Z / f」で実寸推定する。これで「画素 → 視差 → 深度 → 3D 実寸」という本章の鎖を最後まで通しきります。

<figure class="lec-fig"><svg viewBox="0 0 648 232" role="img" aria-label="ミニプロジェクトの4ステージ統合フロー。Stage A校正→Stage B歪み補正→Stage Cステレオ深度→Stage D 3D実寸計測と進み、全段PASSで総合ALL PASS" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="324" y="28" text-anchor="middle" font-size="14" fill="#3f3f46">ミニプロジェクトの 4 ステージ：校正 → 補正 → 深度 → 計測</text><rect x="8" y="54" width="140" height="116" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="42" y="66" width="72" height="22" rx="11" fill="#ea580c"/><text x="78" y="81" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">Stage A</text><text x="78" y="114" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">校正</text><text x="78" y="138" text-anchor="middle" font-size="10" fill="#52525b">calibrateCamera</text><text x="78" y="154" text-anchor="middle" font-size="10" fill="#52525b">RMS＜1px・K±5px</text><rect x="172" y="54" width="140" height="116" rx="8" fill="#fff7ed" stroke="#f97316" stroke-width="2"/><rect x="206" y="66" width="72" height="22" rx="11" fill="#f97316"/><text x="242" y="81" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">Stage B</text><text x="242" y="114" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">歪み補正</text><text x="242" y="138" text-anchor="middle" font-size="10" fill="#52525b">校正Kで undistort</text><text x="242" y="154" text-anchor="middle" font-size="10" fill="#52525b">MAD＜8</text><rect x="336" y="54" width="140" height="116" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="370" y="66" width="72" height="22" rx="11" fill="#2563eb"/><text x="406" y="81" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">Stage C</text><text x="406" y="114" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ステレオ深度</text><text x="406" y="138" text-anchor="middle" font-size="10" fill="#52525b">SGBM・Z=f·B/d</text><text x="406" y="154" text-anchor="middle" font-size="10" fill="#52525b">深度MAE＜0.05m</text><rect x="500" y="54" width="140" height="116" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><rect x="534" y="66" width="72" height="22" rx="11" fill="#16a34a"/><text x="570" y="81" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">Stage D</text><text x="570" y="114" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">3D実寸計測</text><text x="570" y="138" text-anchor="middle" font-size="10" fill="#52525b">reprojectImageTo3D</text><text x="570" y="154" text-anchor="middle" font-size="10" fill="#52525b">実寸幅 差＜0.02m</text><line x1="148" y1="112" x2="164" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="172,112 164,108 164,116" fill="#71717a"/><line x1="312" y1="112" x2="328" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="336,112 328,108 328,116" fill="#71717a"/><line x1="476" y1="112" x2="492" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="500,112 492,108 492,116" fill="#71717a"/><line x1="570" y1="170" x2="570" y2="184" stroke="#16a34a" stroke-width="2"/><polygon points="570,192 566,184 574,184" fill="#16a34a"/><rect x="8" y="192" width="630" height="30" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="323" y="212" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">4 ステージ全 PASS → 総合 ALL PASS（絶対距離 m とサイズを測定）</text></svg><figcaption><code>mini_project.py</code> が走らせる<b>4 ステージの統合フロー</b>です。<b>Stage A 校正</b>（<code>calibrateCamera</code>）→<b>Stage B 歪み補正</b>（校正した <code>K,dist</code> で <code>undistort</code>）→<b>Stage C ステレオ深度</b>（<code>StereoSGBM</code> と <code>Z=f·B/d</code>）→<b>Stage D 3D 実寸計測</b>（<code>reprojectImageTo3D</code>）の順に進み、各段が PASS/FAIL を自己判定します。4 段すべて PASS なら総合 <b>ALL PASS</b> となり、シーン中の物体まで<b>絶対距離（m）とサイズ</b>を測れたことになります。</figcaption></figure>

この課題は「2 台のカメラと校正だけから、シーン中の物体までの**絶対距離（メートル）**とサイズを測る」という、ロボット・自動運転・3D 計測の最小核です。合成チェスボードを実写の盤に、合成ステレオを実 2 眼カメラの画像に差し替えれば、そのまま実運用のひな形になります。

**到達の目安**: 4 ステージすべてが PASS（校正 RMS<1px・`K` が真値±5px、補正 MAD<8、深度 MAE<0.05m、実寸幅の参照値との差<0.02m）で、総合判定が `ALL PASS` になること。出力は `outputs/07_camera_calibration_stereo/` に以下が保存されます。

| 生成物 | 内容 |
| --- | --- |
| `mini_project_distorted.png` / `mini_project_undistorted.png` | 実写相当の歪み画像と、校正した `K, dist` での補正結果 |
| `mini_project_left.png` / `mini_project_right.png` / `mini_project_disparity.png` | ステレオ対と、SGBM 視差マップ（近い＝暖色） |
| `mini_project_summary.png` | 校正・歪み・補正・ステレオ・視差・深度を 1 枚に並べたまとめ図 |
| `mini_project_metrics.json` | 各ステージの数値ログ（RMS・`K`・MAD・深度 MAE・実寸幅・総合 PASS） |

```bash
uv run python lectures/07_camera_calibration_stereo/mini_project.py
cat outputs/07_camera_calibration_stereo/mini_project_metrics.json
```

## ✅ 到達チェックリスト

この章を終えたら、次が**できる／説明できる**ことを確認してください。

- [ ] 内部行列 `K`（`fx,fy,cx,cy`）と歪み係数 `dist=(k1,k2,p1,p2,k3)` が「カメラの何を表すか」を説明でき、`K` を自力で組める。
- [ ] チェスボードの `pattern_size` が**マス数ではなく内側角の数**であること、複数視点を**傾けて**撮る理由（Zhang の手法で `K` と姿勢を分離するため）を説明できる。
- [ ] `findChessboardCorners → cornerSubPix` の 2 段検出を書け、検出が**失敗しうる**前提でスキップ＋ログする頑健な実装ができる。
- [ ] `calibrateCamera` の戻り値 RMS 再投影誤差の定義（**総二乗誤差／総点数の平方根**）を理解し、`projectPoints` で**自前計算してライブラリ値と一致**させられる。
- [ ] 高次歪み係数 `k3` が画像隅の拘束不足で**不定**になりやすく、非魚眼では `CALIB_FIX_K3` で固定するのが定石だと説明できる。
- [ ] `undistort` で歪みを除去でき、`getOptimalNewCameraMatrix` の `alpha` と `ROI`（黒縁の扱い）を用途で使い分けられる。動画では `initUndistortRectifyMap`＋`remap` が定石だと説明できる。
- [ ] エピポーラ拘束 `x_R^T F x_L = 0` と、平行化済みペアでは**エピポーラ線が水平**になることを説明でき、`E = K^T F K` を計算できる。
- [ ] `stereoRectify` の再投影行列 `Q` の意味（`[X,Y,Z,W]^T = Q·[u,v,d,1]^T`）を理解し、平行化済みの手作り `Q` を書ける。
- [ ] `StereoSGBM`/`StereoBM` で視差を計算でき、戻り値が **×16 の固定小数点**であること、`numDisparities` は **16 の倍数**であることを押さえている。
- [ ] `Z = f·baseline/d`（無効視差は 0）で深度に変換でき、`reprojectImageTo3D` で 3D 点群を得て**絶対距離（メートル）**を測れる。
- [ ] 古典ステレオ（2 眼・絶対距離）と深層単眼深度（1 枚・相対深度、第27回）の**使い分け**を説明できる。
- [ ] ミニプロジェクトを実行し、校正→補正→深度→実寸計測の総合判定を `ALL PASS` にできる。

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/07_camera_calibration_stereo/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. チェスボード内側角の 3D 座標 `(cols*rows, 3)` を `float32` で作る。盤は平面なので Z=0、x が先に進む行優先で並べ `square` 倍する（`ex1_object_points`）。
2. 3D 点 `objp` を `(rvec, tvec, K, dist)` で再投影し、検出点 `imgp` との平均ユークリッド距離(px)を返す（校正品質の指標）（`ex2_reprojection_error`）。
3. 視差マップを深度マップへ変換する（`Z = f·baseline/disparity`）。視差が 0 以下の無効画素は深度 0 とし、入力と同じ形・`float32` で返す（`ex3_disparity_to_depth`）。
4. 平行化済みステレオ（cx=cx'）の再投影行列 `Q`（4x4 float64）を組み立てる。`[X,Y,Z,W]^T = Q·[u,v,d,1]^T` で実 3D 点が得られる（`ex4_reproject_matrix`）。
5. 1 画素 `(u, v)` と視差を `Q` で 3D 点 `(X, Y, Z)` に変換する。同次ベクトルに `Q` を掛け、最後の要素 `W` で割る（`ex5_pixel_to_3d`）。
6. 焦点距離 `(fx, fy)` と主点 `(cx, cy)` から内部行列 `K`（3x3 float64）を組む（せん断 skew は 0 とみなす）（`ex6_build_intrinsics`）。
7. 深度マップを視差マップへ変換する（`disparity = f·baseline/Z`、ex3 の逆）。深度 0 以下は視差 0 とし、視差↔深度の反比例を確かめる（`ex7_depth_to_disparity`）。
8. 複数視点の RMS 再投影誤差を「総二乗誤差／総点数の平方根」で厳密計算する（`calibrateCamera` の RMS の定義そのもの）（`ex8_rms_reprojection_error`）。
9. 基礎行列 `F` と内部行列 `K` から基本行列 `E = K^T·F·K` を計算する（3x3 float64）（`ex9_essential_from_fundamental`）。

## ❓ よくある落とし穴・FAQ・デバッグ

実装中に詰まったら、まずここを見てください。多くの不具合はこの数個の原因に集約されます（第10節の症状別チェックリストと併せて参照）。

- **Q. `findChessboardCorners` がいつも `False` を返す。** A. 筆頭原因は `pattern_size` を**マス数**にしていることです。正しくは**内側角の数**で、10×7 マスの盤なら `(9, 6)`。盤が画面からはみ出ていないか、コントラストが十分かも確認します。難しい実画像では `cv2.findChessboardCornersSB`（本体同梱）がより頑健です。
- **Q. 校正の RMS が数 px と大きい。** A. コーナー誤検出・盤が正対ばかり・視点が少ない、のいずれかです。盤を**傾けて** 10〜20 枚撮り、`cornerSubPix` を必ず入れ、明らかに外れた視点は除きます。
- **Q. 自前の RMS が `calibrateCamera` の戻り値と一致しない。** A. 「視点ごとに平均してから視点平均」を取ると厳密には RMS と一致しません。**全点の二乗誤差を合算→総点数で割る→平方根**（演習 8 の定義）で計算します。各点の距離は `dx^2+dy^2` の二乗和で集計します。
- **Q. `k2, k3` だけ真値とかけ離れる。** A. 盤が画像隅まで届かず**高次の径方向歪みが拘束されない**のが原因です。非魚眼では `flags=cv2.CALIB_FIX_K3` で固定し、できれば盤を隅まで写します。RMS がほぼ変わらないのに `k3` だけ暴れるなら不定の証拠です。
- **Q. `undistort` 後に四隅が黒くなる。** A. 樽型歪みを戻すと外周が外へ広がる**仕様**です。`getOptimalNewCameraMatrix` の `alpha=1`＋`ROI` で有効範囲を把握するか、`alpha=0` で端を切って黒縁を消します。
- **Q. 深度が 16 倍ずれる。** A. `StereoSGBM/BM.compute()` の戻り値は **×16 の固定小数点（int16）**です。`disp = sgbm.compute(...).astype(np.float32) / 16.0` の割り戻しを必ず入れます。
- **Q. 視差マップが穴だらけ。** A. テクスチャ不足（無地・鏡面・繰り返し模様）・`numDisparities` が狭い・未平行化のいずれかです。テクスチャのある領域で評価し、探索幅を **16 の倍数**で広げ、先に平行化します。
- **Q. 手前の物体だけ視差が欠ける。** A. `numDisparities` が想定最大視差より小さいためです。最大視差を見積もって `numDisparities` を増やします（本章は最大 64 を見込んで 96）。
- **Q. `StereoBM` がエラーになる。** A. カラー画像を渡しています。`cv2.cvtColor(..., COLOR_BGR2GRAY)` で**グレースケール**にしてから `compute` します（BM はグレー専用）。
- **Q. 深度が負・無限大になる。** A. 視差 0／負でゼロ割りしています。`valid = disp > 0` でマスクしてから `Z = f·baseline/d` を計算します。
- **Q. `reprojectImageTo3D` の `Z` が想定とスケールが違う。** A. `Q` の `baseline` 単位（m）と `f` の単位（px）の取り違え、あるいは視差の割り戻し忘れが原因です。本章の `Q` は `Z = f·baseline/d` と一致するよう作っています。
- **Q. matplotlib で色が反転する（赤青が入れ替わる）。** A. cv2 は **BGR**、matplotlib は RGB です。`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` を挟んでから `imshow` し、headless では `matplotlib.use("Agg")` を `pyplot` import 前に呼んで `savefig` で保存します。

## 🚀 発展トピック・参考

- **`findChessboardCornersSB` と ChArUco ボード**: 難しい実画像では SB（symmetric-based）版がより頑健です。さらに `cv2.aruco` の ChArUco ボードは一部が隠れても校正でき、実務の定番になりつつあります（`opencv-contrib-python` 同梱）。
- **魚眼モデル `cv2.fisheye`**: 本章の Brown-Conrady モデル（`k1..k3, p1, p2`）では広角・魚眼を十分に表せません。`cv2.fisheye.calibrate`/`undistortImage` は等距離射影系の専用モデルで、広視野カメラではこちらを使います。
- **`solvePnP`（姿勢推定）**: 校正済み `K, dist` と既知の 3D-2D 対応があれば `cv2.solvePnP` で物体／カメラの姿勢 `(rvec, tvec)` が一発で出ます。AR マーカや物体姿勢推定の核で、本章の `projectPoints` の逆問題です。
- **`stereoCalibrate`（実 2 眼の校正）**: 本章は平行化済みステレオを合成しましたが、現実の 2 台は `cv2.stereoCalibrate` で相対姿勢 `(R, T)` を求め、`stereoRectify`→`initUndistortRectifyMap`→`remap` で平行化してから視差を計算します。
- **視差の後処理（WLS フィルタ）**: `cv2.ximgproc.createDisparityWLSFilter`（contrib）で左右視差を融合・平滑化すると、穴や境界のノイズが大きく改善します。実務のステレオ深度ではほぼ必須の後段です。
- **SGBM のパラメータ調整**: `blockSize`・`P1/P2`（滑らかさ）・`uniquenessRatio`・`speckleWindowSize` は被写体とノイズで最適値が変わります。`disp12MaxDiff` の左右一貫性チェックと併せて、テクスチャの少ないシーンでの破綻を観察すると理解が進みます。
- **深層単眼深度との併用**: Depth Anything V2（第27回）の相対深度を、本章のステレオや既知サイズ物体で**スケール補正**して絶対距離化する、というハイブリッドも実務で有効です。
- 参考ドキュメント: OpenCV 公式チュートリアル「Camera Calibration and 3D Reconstruction」 https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html ／ Zhang, Z. (2000) "A Flexible New Technique for Camera Calibration"（チェスボード校正の原典）／ Hirschmüller, H. (2008) "Stereo Processing by Semiglobal Matching and Mutual Information"（SGBM の原典）。

## 💡 実践ユースケース集

この章で身につけた「カメラを `K, dist` で表す → 校正する → 平面射影（ホモグラフィ）/ `solvePnP` で幾何を解く」というスキルは、そのまま**現実の計測・採寸・位置合わせ**の道具になります。ここでは実アプリに直結する応用を 3 つ挙げ、そのうちの 1 つは動く出発点 `use_case.py` として同梱しています。

### 1. 写真からの実寸採寸（基準物スケール）— `use_case.py`

**何に使うか**: 定規を当てずに、1 枚の写真から机上の部品・傷・葉・料理などの**実寸(mm)**を測る。EC の商品採寸、現場の簡易計測、検査の寸法チェックなど。

**作り方の要点**: 被写体と**同一平面**に「マス幅が既知のチェスボード（または A4 用紙）」を一緒に写す → 盤の内側角は『画像画素』と『盤平面の mm 座標』の対応点になる → 平面どうしなので `cv2.findHomography` で画像→mm 平面の射影 `H` が一発で求まる → 測りたい 2 点を `cv2.perspectiveTransform` で mm 平面へ写し、その距離が実寸。校正済み `K, dist` があれば前段で `undistort` して端の精度を上げ、`cv2.solvePnP` でカメラ〜面の**おおよその距離**も出せます。**ホモグラフィによる mm 計測は K が無くても成立する**のが強み（K は歪み補正と距離推定にだけ効く）。

<figure class="lec-fig"><svg viewBox="0 0 648 232" role="img" aria-label="1枚写真からの実寸メジャーの処理フロー。同一平面で撮影→対応点を取る→射影H推定→2点を変換→距離が実寸。ホモグラフィはK無しで成立" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="324" y="30" text-anchor="middle" font-size="14.5" fill="#3f3f46">1枚写真の実寸メジャー：撮影 → 対応点 → H → 変換 → 実寸</text><rect x="8" y="54" width="114" height="118" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><circle cx="65" cy="82" r="12" fill="#2563eb"/><text x="65" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">1</text><text x="65" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">同一平面で撮影</text><text x="65" y="144" text-anchor="middle" font-size="10.5" fill="#52525b">被写体＋基準盤</text><rect x="137" y="54" width="114" height="118" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><circle cx="194" cy="82" r="12" fill="#2563eb"/><text x="194" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">2</text><text x="194" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">対応点を取る</text><text x="194" y="141" text-anchor="middle" font-size="10.5" fill="#52525b">盤の内側角</text><text x="194" y="157" text-anchor="middle" font-size="10.5" fill="#52525b">画素 ↔ mm</text><rect x="266" y="54" width="114" height="118" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><circle cx="323" cy="82" r="12" fill="#2563eb"/><text x="323" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">3</text><text x="323" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">射影 H 推定</text><text x="323" y="144" text-anchor="middle" font-size="10.5" fill="#52525b">findHomography</text><rect x="395" y="54" width="114" height="118" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><circle cx="452" cy="82" r="12" fill="#2563eb"/><text x="452" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">4</text><text x="452" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">2点を変換</text><text x="452" y="141" text-anchor="middle" font-size="10.5" fill="#52525b">perspective</text><text x="452" y="157" text-anchor="middle" font-size="10.5" fill="#52525b">Transform</text><rect x="524" y="54" width="114" height="118" rx="8" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><circle cx="581" cy="82" r="12" fill="#16a34a"/><text x="581" y="87" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">5</text><text x="581" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#18181b">距離＝実寸</text><text x="581" y="144" text-anchor="middle" font-size="10.5" fill="#52525b">mm で出力</text><line x1="122" y1="113" x2="129" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="137,113 129,109 129,117" fill="#71717a"/><line x1="251" y1="113" x2="258" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="266,113 258,109 258,117" fill="#71717a"/><line x1="380" y1="113" x2="387" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="395,113 387,109 387,117" fill="#71717a"/><line x1="509" y1="113" x2="516" y2="113" stroke="#71717a" stroke-width="2"/><polygon points="524,113 516,109 516,117" fill="#71717a"/><text x="324" y="204" text-anchor="middle" font-size="12" fill="#52525b">ホモグラフィ H は内部行列 K 無しで成立（K は歪み補正と距離推定にだけ効く）</text></svg><figcaption><code>use_case.py</code> の<b>1 枚写真メジャー</b>の処理フローです。被写体と<b>基準チェスボード</b>を<b>同一平面</b>で撮影 →盤の内側角で<b>画素 ↔ mm の対応点</b>を取る →平面どうしなので <code>cv2.findHomography</code> で<b>射影 H</b>（画像→mm 平面）を求める →測る 2 点を <code>cv2.perspectiveTransform</code> で mm 平面へ写す →その距離が<b>実寸(mm)</b>です。<b>ホモグラフィ H は内部行列 K 無しで成立</b>し（K は歪み補正と距離推定にだけ効く）、被写体と基準盤が<b>同一平面</b>であることが前提です。</figcaption></figure>

**注意**: 被写体と基準物が**同一平面に乗っている**ことが前提（厚みや傾きがあると誤差になる）。盤が画面に収まり・マス数は**内側角**で指定し・十分なコントラストを確保すること。広角ほど周辺歪みが効くので、できれば自前カメラを校正して `calib.npz` を用意し `undistort` してから測る。

実行と `data/` 配置:

```bash
# そのまま（合成シーン: 既知サイズの対象物つきなので誤差まで検証できる）
uv run python lectures/07_camera_calibration_stereo/use_case.py

# 実写で測る: data/07_camera_calibration_stereo/ に「基準チェスボードを写し込んだ写真」を置く
#   例 data/07_camera_calibration_stereo/measure.jpg
# 盤のマス幅(mm)や内側角数、自前校正結果は環境変数で渡せる
CHESSBOARD_SQUARE_MM=25 CHESSBOARD_COLS=9 CHESSBOARD_ROWS=6 \
  CALIB_NPZ=outputs/07_camera_calibration_stereo/calib.npz \
  uv run python lectures/07_camera_calibration_stereo/use_case.py

# 2 点クリックで任意の寸法を測る GUI（DISPLAY 有りのデスクトップのみ。headless では自動スキップ）
MEASURE_GUI=1 uv run python lectures/07_camera_calibration_stereo/use_case.py
```

結果は `outputs/07_camera_calibration_stereo/` に `use_case_measured.png`（検出＋mm ラベル付きの計測線）・`use_case_summary.png`・`use_case_metrics.json`（計測値と誤差）で保存されます。`mini_project.py`（2 眼ステレオの 3D 計測）に対し、本ツールは**単眼＋平面拘束の実寸計測**という別アプローチで、2 台もステレオも要りません。

**拡張アイデア**: 基準を `cv2.aruco` の ArUco/ChArUco マーカに変えて一部が隠れても基準を取れるようにする／A4 用紙(210×297mm)の四隅を基準点にする／`warpPerspective` で**鳥瞰図**を作り画素↔mm を一定スケールにする／物体検出と組んで自動採寸する。

### 2. AR マーカ・物体姿勢推定（`solvePnP` / ArUco）

**何に使うか**: 校正済みカメラと既知の 3D-2D 対応から物体（またはカメラ）の姿勢 `(rvec, tvec)` を求め、CG を実写に重ねる AR、ロボットアームのピッキング、マーカ位置決めに使う。

**作り方の要点**: 既知サイズのマーカ（ArUco）や 3D 点を持つ対象を検出し `cv2.solvePnP`（本章 `use_case.py` の距離推定と同じ API）で姿勢を解く → `cv2.projectPoints` で 3D モデルを画像へ重畳。本章の `projectPoints`（3D→2D の順問題）の**逆問題**にあたり、`K, dist` の校正品質がそのまま重畳のズレに直結します。

**注意**: 対応点が同一平面だけだと姿勢が不安定になりやすいので、可能なら非平面の対応や複数マーカを使う。`distCoeffs` に校正で得た歪みを必ず渡す（省くと姿勢がずれる）。

### 3. ステレオ深度カメラによる近接物の距離計測

**何に使うか**: 2 眼カメラ（USB ステレオや自作の左右 2 台）で、目の前の人・障害物・部品までの**絶対距離(m)**を測る。ロボットの衝突回避、簡易 3D スキャン、入退場の距離トリガなど。

**作り方の要点**: 実 2 台は `cv2.stereoCalibrate` で相対姿勢 `(R, T)` を求め、`stereoRectify`→`initUndistortRectifyMap`→`remap` で**平行化**してから `StereoSGBM` で視差を計算し、`Z = f·baseline/d` と `reprojectImageTo3D` で 3D 化（`mini_project.py` の Stage C/D がこの最小核）。基準サイズの物体で視差→距離の妥当性を検証します。

**注意**: 視差はテクスチャの無い面・鏡面・繰り返し模様で破綻するので `cv2.ximgproc` の WLS フィルタで後処理する。`numDisparities` は**最大視差を見込んで 16 の倍数**で設定し、必ず平行化を済ませてから視差を取ること。

---

> 本教材で参照・検証したライブラリとバージョン（2026-06 時点の安定版で動作確認）:
> Python 3.12 ／ numpy 2.4.6 ／ opencv-python-headless 4.13.0.92（`cv2` 4.13.0、calibrateCamera/StereoSGBM/stereoRectify は本体同梱・contrib 不要）／ Pillow 12.2.0 ／ matplotlib 3.10.9（torch は本章では未使用）