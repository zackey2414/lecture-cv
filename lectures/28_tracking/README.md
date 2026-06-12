# 28_tracking: 物体追跡 — 単一物体トラッカ・SORT/DeepSORT・MOT 評価

> トラック: **動画・追跡** ／ レベル: **中級** ／ 主要依存: `numpy` `opencv` `scipy`（任意で `dl`）
>
> この回は **巨大依存・衝突依存を実行経路で使いません**。`supervision(ByteTrack)` /
> `deep_sort_realtime` / `motmetrics` / `TrackEval` は numpy2 系や大きな依存と衝突しうるため、
> **概念紹介＋任意導入**にとどめ、本体は numpy + OpenCV で **自前実装** して完走させます。

---

## 🎯 この章のゴール

- **検出 (detection)** と **追跡 (tracking)** の違いを説明でき、検出器なしの単一物体トラッカ
  （OpenCV）と、検出器の出力に ID を付ける多物体追跡（MOT）を **自分で実装** できる。
- **SORT** の 2 部品（等速カルマンフィルタ + ハンガリアン法による IoU 対応付け）を numpy で
  ゼロから書ける。**DeepSORT** が外見特徴で ID スイッチをどう減らすかを再現できる。
- 追跡品質を **MOTA / MOTP / IDF1 / HOTA(簡易)** で評価でき、それぞれが「検出の良さ」と
  「ID の一貫性」のどちらを測るのかを区別できる。
- 完成形として「**合成動画 → cv2 検出 → 自前 SORT → ID 軌跡描画 → MOT 評価**」の
  パイプラインを外部依存ゼロで動かせる。

---

## 1. 直感 — 「検出」と「追跡」は何が違うのか

物体検出は、各フレームを **独立に** 見て「どこに何があるか」を答えます。しかし検出器は
フレーム間のつながりを知らないため、フレーム 1 で見つけた人とフレーム 2 で見つけた人が
「同じ人」かどうかは、検出だけでは決して分かりません。この **フレームをまたいで同じ物体に
同じ番号（ID）を振り続ける** 仕事こそが追跡 (tracking) です。監視カメラの人数カウント、
スポーツの選手分析、自動運転の周辺車両の挙動予測 — どれも「同一性の維持」が要であり、
そこが追跡の担当範囲になります。

<figure class="lec-fig"><svg viewBox="0 0 600 300" role="img" aria-label="検出は各フレーム独立でIDなし、追跡はフレームをまたいで同じ物体に同じIDを維持する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">① 検出 — 各フレーム独立・ID なし</text><rect x="30" y="42" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="215" y="42" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="400" y="42" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="46" y="58" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="265" y="58" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="484" y="58" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="130" y="96" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="295" y="96" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><rect x="460" y="96" width="30" height="24" fill="#e4e4e7" stroke="#71717a" stroke-width="1.8" stroke-dasharray="4 3"/><text x="300" y="166" text-anchor="middle" font-size="14" font-weight="700" fill="#3f3f46">② 追跡 — 同じ色＝同じ ID を維持</text><rect x="30" y="182" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="215" y="182" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="400" y="182" width="160" height="98" fill="#fafafa" stroke="#d4d4d8" stroke-width="1.5"/><rect x="46" y="196" width="30" height="24" fill="#f97316" stroke="#c2410c" stroke-width="2"/><rect x="265" y="196" width="30" height="24" fill="#f97316" stroke="#c2410c" stroke-width="2"/><rect x="484" y="196" width="30" height="24" fill="#f97316" stroke="#c2410c" stroke-width="2"/><rect x="130" y="234" width="30" height="24" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/><rect x="295" y="234" width="30" height="24" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/><rect x="460" y="234" width="30" height="24" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/><text x="61" y="213" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">1</text><text x="145" y="251" text-anchor="middle" font-size="13" font-weight="700" fill="#ffffff">2</text><text x="110" y="294" text-anchor="middle" font-size="12.5" fill="#52525b">t=1</text><text x="295" y="294" text-anchor="middle" font-size="12.5" fill="#52525b">t=2</text><text x="480" y="294" text-anchor="middle" font-size="12.5" fill="#52525b">t=3</text></svg><figcaption><b>検出</b>は各フレームを独立に見るので、箱は得られても前フレームの物体とは結びつきません（上段の灰色の点線箱＝ID なし）。<b>追跡</b>はその箱に<b>フレームをまたいで同じ ID</b>を振り続けます（下段は<b>同じ色＝同じ物体</b>）。つまり追跡とは「検出結果に一貫した同一性を与える」仕事です。</figcaption></figure>

追跡の手法は、大きく 2 系統に分けられます。一つは **単一物体トラッカ (single object tracker)**
です。最初の 1 フレームで対象の矩形（ROI）を 1 回だけ与えると、以降は前フレームの「見え方」を
手がかりに、その 1 個を追い続けます。検出器が要らないのが利点ですが、その代わり対象が隠れたり
大きく変形したりすると見失います。もう一つは **tracking-by-detection**（検出に基づく追跡）です。
毎フレーム検出器を回し、得られた多数のボックスを **フレーム間で対応付け** て ID を割り振ります。
こちらは多物体で出入りの多いシーンに強く、ByteTrack や DeepSORT、そして本章で自作する SORT が
この系統に属します。

本章ではまず単一物体トラッカで「追跡とは前フレームとの照合だ」という感覚を掴み、次に
tracking-by-detection の核（カルマン予測と対応付け）を自分の手で組み立てていきます。入力は
すべて **合成データ**（壁で反射する複数の長方形）なので、正解の軌跡 (ground truth) を完全に
把握でき、評価指標を厳密に検算できます。

---

## 2. 単一物体トラッカ（OpenCV）— `init` / `update` と「ビルド差」の罠

OpenCV の単一物体トラッカは、どれも同じ 2 つのメソッドで動きます。まず最初のフレームと ROI を
`tracker.init(frame, (x, y, w, h))` で渡し、以降は `ok, box = tracker.update(frame)` を毎フレーム
呼ぶだけです。`ok=False` は「見失った」の合図であり、実務ではここで検出器による **再初期化
(re-detection)** にフォールバックします。代表的な選択肢は **CSRT**（高精度・低速）、
**KCF**（高速・そこそこ）、**MIL**（頑健・依存が軽い）です。速度と精度はトレードオフの関係に
あるため、用途に応じて選びます。

ここで初学者が必ずハマるのが、**ビルドによって使えるトラッカが違う** という問題です。`CSRT` は
新しめのビルドでは `cv2.TrackerCSRT_create`（main 名前空間）にありますが、`KCF` や `MOSSE`
は contrib パッケージの `cv2.legacy.TrackerKCF_create` に置かれていることが多く、さらに本講座の
ような `opencv-python-headless` ビルドでは **そもそも contrib が入っていない** ため、`cv2.legacy`
自体が存在しません。そのため記事のコードをそのまま写すと `AttributeError` になります。対策は
単純で、**`hasattr` で実在を確認してから使う** ことに尽きます。`01_single_object_tracker.py` の
`list_available_trackers()` は CSRT→KCF→MIL の優先順で、実在するものだけを集めます（この
環境では MIL のみが該当します）。

なお `GOTURN` / `Nano` / `DaSiamRPN` は、`hasattr` 上は存在しても **別途モデル重み（.onnx 等）が
必要** であり、重みが無いと `init` で失敗します。したがって、「重み不要で動く」CSRT/KCF/MIL を
優先するのが安全な設計です。`01` は合成クリップ（ジグザグに動く長方形）で MIL を初期化し、
各フレームの予測と GT の IoU を測って `平均IoU` と `成功率(IoU>0.5)` を算出します。

```bash
uv run python lectures/28_tracking/01_single_object_tracker.py
# → このビルドで使えるトラッカ: ['MIL'] / 平均IoU≈0.82 / 成功率1.00
```

---

## 3. SORT を自作する — カルマンフィルタ + ハンガリアン法

多物体追跡の王道 **SORT (Simple Online and Realtime Tracking)** は、驚くほど少ない部品で
できています。必要なのは、(1) 各トラックの次フレーム位置を予測する **等速カルマンフィルタ** と、
(2) 予測ボックスと新しい検出を **IoU を手がかりに 1対1 対応付ける** ハンガリアン法だけです。
外見特徴も深層ネットも使わないにもかかわらず、検出器がそこそこ良ければ実用十分の速度・精度が
出ます。だからこそ、「追跡の最小公倍数」として最初に手で書く価値があるのです。

**カルマンフィルタ** は「予測 (predict) → 更新 (update)」の 2 段で動きます。SORT の状態は
7 次元 `[u, v, s, r, u', v', s']`（中心 u,v・面積 s・アスペクト比 r とそれらの速度）で表します。
まず `predict` が等速モデルにより中心と面積を 1 フレーム進め、検出が当たったら `update` が観測を
取り込んで補正します。この仕組みのおかげで、検出が一時的に欠けても予測だけで生き延びられる
（`max_age` フレームまで）ため、短いオクルージョンに耐えられるのが効きどころです。
`02_sort_from_scratch.py` の `KalmanBoxTracker` は、状態遷移 `F`・観測 `H`・各共分散 `P,Q,R` を
素直に numpy で実装しています。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="カルマンフィルタはpredictで箱を進め不確かさを広げ、updateで検出により補正し不確かさを縮める" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="86" y="76" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">推定 t−1</text><rect x="60" y="92" width="52" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><ellipse cx="86" cy="112" rx="16" ry="12" fill="none" stroke="#2563eb" stroke-width="1.5"/><circle cx="86" cy="112" r="4" fill="#1d4ed8"/><line x1="118" y1="112" x2="244" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="252,112 242,107 242,117" fill="#71717a"/><text x="186" y="92" text-anchor="middle" font-size="12" fill="#3f3f46">predict（等速＋不確かさ拡大）</text><rect x="266" y="90" width="52" height="40" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="5 3"/><ellipse cx="292" cy="110" rx="34" ry="24" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="5 3"/><circle cx="292" cy="110" r="4" fill="#2563eb"/><rect x="316" y="116" width="50" height="38" fill="#ffedd5" stroke="#ea580c" stroke-width="2.5"/><text x="341" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">検出</text><line x1="378" y1="112" x2="498" y2="112" stroke="#71717a" stroke-width="2"/><polygon points="506,112 496,107 496,117" fill="#71717a"/><text x="442" y="92" text-anchor="middle" font-size="12" fill="#3f3f46">update（観測で補正）</text><text x="556" y="76" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">推定 t</text><rect x="524" y="92" width="52" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><ellipse cx="550" cy="112" rx="14" ry="10" fill="none" stroke="#2563eb" stroke-width="1.5"/><circle cx="550" cy="112" r="4" fill="#1d4ed8"/><rect x="140" y="250" width="360" height="34" rx="7" fill="#fff7ed" stroke="#f97316" stroke-width="1.5"/><text x="320" y="272" text-anchor="middle" font-size="13" fill="#c2410c">検出が来ない間も predict だけで生存（max_age まで）</text></svg><figcaption>SORT のカルマンフィルタは <b>predict → update</b> の 2 段を繰り返します。<b>predict</b> は等速モデルで箱を 1 歩進め、中心の<b>不確かさ（破線の楕円）を広げます</b>。検出が当たれば <b>update</b> が観測で<b>補正し、不確かさを縮めます</b>。検出が一時的に欠けても <b>predict だけで生き延びられる</b>ので、<code>max_age</code> フレームまでは短いオクルージョンに耐えられます。</figcaption></figure>

一方 **対応付け** は、「コスト行列を作って総コスト最小の組を選ぶ」問題として定式化できます。
コストを `1 - IoU` とすれば「IoU が高い組ほど結びたい」という意図を表現でき、その最適解は
**ハンガリアン法**（`scipy.optimize.linear_sum_assignment`）で得られます。scipy が無い環境でも
動くよう、`_common.linear_assignment` は **貪欲法（最小コストから順に確定）へ自動フォールバック**
します。ここで重要なのは、割り当てた後に **IoU 閾値で足切り** することです。ハンガリアン法は
「全部を無理やり結ぶ」ので、重なりが薄い組（IoU < 0.3 など）は採用しない、というゲートを必ず
入れます。以上をまとめると、SORT 本体の 1 フレームは「① 全トラックを predict → ② 検出と
対応付け → ③ マッチは update、余った検出は新規トラック生成、長く当たらないトラックは破棄」と
いう流れになります。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="IoUは交差を合併で割った重なり率、コスト1-IoUの行列をハンガリアン法で最小化しIoU閾値で足切りする" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="150" y="34" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">IoU = 交差 ÷ 合併</text><rect x="120" y="108" width="70" height="59" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><rect x="55" y="72" width="135" height="95" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6 4"/><rect x="120" y="108" width="135" height="95" fill="none" stroke="#ea580c" stroke-width="2.5"/><line x1="310" y1="50" x2="310" y2="250" stroke="#e4e4e7" stroke-width="1.5"/><text x="506" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">コスト = 1 − IoU</text><text x="468" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">D1</text><text x="544" y="92" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">D2</text><text x="412" y="150" text-anchor="end" font-size="13" font-weight="700" fill="#1d4ed8">T1</text><text x="412" y="212" text-anchor="end" font-size="13" font-weight="700" fill="#1d4ed8">T2</text><rect x="430" y="110" width="76" height="62" fill="#f4f4f5" stroke="#16a34a" stroke-width="3"/><rect x="506" y="110" width="76" height="62" fill="#52525b" stroke="#dc2626" stroke-width="3"/><rect x="430" y="172" width="76" height="62" fill="#71717a" stroke="#d4d4d8" stroke-width="1.5"/><rect x="506" y="172" width="76" height="62" fill="#f4f4f5" stroke="#16a34a" stroke-width="3"/><circle cx="468" cy="141" r="8" fill="#16a34a"/><circle cx="544" cy="203" r="8" fill="#16a34a"/><text x="544" y="150" text-anchor="middle" font-size="24" font-weight="700" fill="#ffffff">×</text></svg><figcaption>対応付けは「<b>コスト行列</b>から総コスト最小の組を選ぶ」問題です。<b>左</b>: 予測ボックス（青の破線）と検出（オレンジ）の重なり <b>IoU ＝ 交差 ÷ 合併</b> がコストの素。<b>右</b>: コスト <code>1 − IoU</code> を全トラック×全検出で並べ、<b>ハンガリアン法</b>で最小コストの組（緑）を選びます。仕上げに <b>IoU 閾値ゲート</b>で重なりの薄い組（赤の <code>×</code>、例: IoU&lt;0.3）を足切りします。</figcaption></figure>

<figure class="lec-fig"><svg viewBox="0 0 660 256" role="img" aria-label="SORTの1フレームはpredictで全トラックを進め、IoUコストの対応付けの後、マッチはupdate、余った検出は新規トラック、当たらないトラックは破棄の3つに分岐する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">SORT の 1 フレーム — predict → 対応付け → 3 分岐</text><rect x="30" y="120" width="130" height="56" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="95" y="144" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">① predict</text><text x="95" y="162" text-anchor="middle" font-size="11" fill="#71717a">全トラックを進める</text><line x1="160" y1="148" x2="194" y2="148" stroke="#71717a" stroke-width="2"/><polygon points="200,148 190,143 190,153" fill="#71717a"/><rect x="200" y="120" width="140" height="56" rx="8" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><text x="270" y="144" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② 対応付け</text><text x="270" y="162" text-anchor="middle" font-size="11" fill="#71717a">IoU + ハンガリアン</text><line x1="340" y1="148" x2="410" y2="148" stroke="#71717a" stroke-width="2"/><line x1="410" y1="80" x2="410" y2="216" stroke="#71717a" stroke-width="2"/><line x1="410" y1="80" x2="464" y2="80" stroke="#71717a" stroke-width="2"/><polygon points="470,80 460,75 460,85" fill="#71717a"/><line x1="410" y1="148" x2="464" y2="148" stroke="#71717a" stroke-width="2"/><polygon points="470,148 460,143 460,153" fill="#71717a"/><line x1="410" y1="216" x2="464" y2="216" stroke="#71717a" stroke-width="2"/><polygon points="470,216 460,211 460,221" fill="#71717a"/><rect x="470" y="56" width="170" height="48" rx="7" fill="#ffffff" stroke="#16a34a" stroke-width="2"/><text x="555" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">マッチ → update</text><text x="555" y="96" text-anchor="middle" font-size="11" fill="#52525b">観測で補正</text><rect x="470" y="124" width="170" height="48" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="555" y="146" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">新規トラック生成</text><text x="555" y="164" text-anchor="middle" font-size="11" fill="#52525b">余った検出から</text><rect x="470" y="192" width="170" height="48" rx="7" fill="#fafafa" stroke="#71717a" stroke-width="2"/><text x="555" y="214" text-anchor="middle" font-size="13" font-weight="700" fill="#3f3f46">トラック破棄</text><text x="555" y="232" text-anchor="middle" font-size="11" fill="#52525b">max_age 超で消す</text></svg><figcaption><b>SORT の 1 フレーム</b>は <b>① predict</b>（全トラックを等速で 1 歩進める）→ <b>② 対応付け</b>（<code>1 − IoU</code> コストをハンガリアン法で最小化）→ <b>3 つに分岐</b>します。<b>マッチした組</b>は <code>update</code> で観測補正、<b>余った検出</b>は新規トラックを生成、<b>長く当たらないトラック</b>は <code>max_age</code> 超で破棄します。この分岐がトラックの<b>生成・維持・消滅</b>を司ります。</figcaption></figure>

```bash
uv run python lectures/28_tracking/02_sort_from_scratch.py
# → 4 物体 / 40 フレームを追跡し、ID 別の軌跡を outputs/ に保存
```

---

## 4. DeepSORT の核 — 外見特徴で ID スイッチを抑える

SORT は動き（IoU）だけで対応付けるため、**物体同士が交差・重なる** 場面に弱点があります。
重なった瞬間は IoU だけでは「どちらがどちら」を区別できず、ID が入れ替わる **ID スイッチ
(IDSW)** が起きてしまうのです。これに対する **DeepSORT** の発想は明快で、ここに「**外見の指紋**」
を足します。すなわち、各検出から見た目を表す特徴ベクトル（本来は ReID 用の小さな CNN 埋め込み）
を取り出し、**動きのコストと外見のコストを合成** して対応付けるのです。こうすれば、「位置は
近いが見た目が違う」組を弾けるようになります。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="DeepSORTは動きコスト1-IoUと外見コスト1-cosをλで重み付けして合成し、交差時のID取り違えを防ぐ" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="50" y="72" width="150" height="92" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="98" y="100" width="26" height="18" fill="#dbeafe" stroke="none"/><rect x="72" y="84" width="52" height="34" fill="none" stroke="#2563eb" stroke-width="2"/><rect x="98" y="100" width="52" height="34" fill="none" stroke="#2563eb" stroke-width="2"/><text x="125" y="152" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">動き：1 − IoU</text><text x="222" y="125" text-anchor="middle" font-size="24" font-weight="700" fill="#3f3f46">+</text><rect x="244" y="72" width="150" height="92" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="266" y="92" width="12" height="30" fill="#f97316"/><rect x="282" y="100" width="12" height="22" fill="#ea580c"/><rect x="298" y="108" width="12" height="14" fill="#2563eb"/><rect x="314" y="102" width="12" height="20" fill="#16a34a"/><text x="319" y="152" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">外見：1 − cos</text><text x="416" y="125" text-anchor="middle" font-size="24" font-weight="700" fill="#3f3f46">=</text><rect x="438" y="72" width="162" height="92" fill="#f4f4f5" stroke="#52525b" stroke-width="2"/><text x="519" y="123" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">対応付けコスト</text><text x="320" y="210" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">cost = (1 − λ)(1 − IoU) + λ(1 − cos 類似度)</text><text x="320" y="248" text-anchor="middle" font-size="12.5" fill="#52525b">IoU が曖昧でも外見が違えば弾ける → ID スイッチ減少</text></svg><figcaption><b>DeepSORT</b> は SORT の<b>動きコスト <code>1 − IoU</code></b> に<b>外見コスト <code>1 − cos</code> 類似度</b>を足し、<code>cost = (1−λ)(1−IoU) + λ(1−cos)</code> で対応付けます（外見は色ヒストグラムや ReID 埋め込み）。物体が<b>交差して IoU だけでは曖昧</b>でも、見た目（色の指紋）が違えば取り違えを弾けるため <b>ID スイッチが減ります</b>。<code>λ</code> が外見を重視する度合いです。</figcaption></figure>

`03_kalman_appearance.py` は、巨大依存 `deep_sort_realtime` を使わず、外見特徴を
**HSV 色ヒストグラム** で代用することで DeepSORT のエッセンスを再現します。合成シーンを「物体
ごとに固有色の塗りつぶし長方形」として描けば、検出ボックスを切り出した色ヒストグラムが物体
ごとにはっきり区別できるからです。コストは `(1-λ)·(1-IoU) + λ·(1-cos類似度)` の形をとり、λ が
外見の重みを表します。さらに特徴は指数移動平均で滑らかに更新し、見えの一時変化に頑健にして
います。そのうえで同じシーンを「動きのみ」と「動き+外見」で追跡し、**GT 視点の ID スイッチ数**
を数えて比較すると、外見を足した方がスイッチが激減する（このサンプルでは 6 → 0）ことが確認
できます。

```bash
uv run python lectures/28_tracking/03_kalman_appearance.py
# → 動きのみ IDSW=6 / 動き+外見 IDSW=0
```

実運用で色ヒストグラムを ReID 埋め込みに替えれば、これがそのまま DeepSORT になります。なお
**ByteTrack** はこれとは別のアプローチをとり、外見ではなく「**低スコア検出も捨てずに 2 段階で
対応付ける**」ことで隠れ・低信頼の物体を拾い、IDSW と取りこぼしを同時に減らします
（Kalman+Hungarian のみで純 CPU・軽量）。本章の SORT 実装は、この ByteTrack の土台でもあります。

---

## 5. MOT 評価指標 — MOTA / MOTP / IDF1 / HOTA を自作で理解する

追跡の良し悪しは **「検出が当たっているか」** と **「ID が一貫しているか」** の両面から測ります。
なかでも最も有名な **MOTA (Multiple Object Tracking Accuracy)** は、誤りを 1 本にまとめた指標で、

```
MOTA = 1 - (FN + FP + IDSW) / GT総数
```

この式は、`FN`（取りこぼし）・`FP`（誤検出）・`IDSW`（ID 入れ替え）を全フレーム足し、GT 総数で
割って 1 から引いたものです。そのため誤りが多いと **平気で負の値** になります。ここで注意したい
のは、MOTA が「検出寄り」の指標であり、ID 一貫性の比重が小さいことです。一方 **MOTP** は
「**当たった組だけ**」の平均 IoU であり、位置精度を表します。取りこぼしが多くても MOTP は
下がらない（当たった分しか見ない）ので、MOTA とセットで読む必要があります。

**IDF1** は、ID を通した一貫性を測る F1 です。まず GT の各 ID と予測の各 ID を **系列全体で
1対1 対応させ**（一致フレーム数が最大になるように、これもハンガリアン法を使います）、`IDTP`
（正しく同じ ID で当て続けたフレーム数）から `IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)` を計算
します。MOTA が高くても途中で ID が切れれば IDF1 は下がるので、「人物を最後まで同一視できたか」
を見たいときの主指標になります。最後の **HOTA** は近年の標準で、検出の良さ `DetA` と関連付けの
良さ `AssA` を **明示的に分離** したうえで、`HOTA = √(DetA × AssA)` として統合します。本章の
`04_mot_metrics.py` が実装するのは **単一 IoU 閾値の簡易版** HOTA です（公式は 0.05〜0.95 の閾値
で平均する多段版で、そちらは TrackEval が担当します）。

<figure class="lec-fig"><svg viewBox="0 0 600 260" role="img" aria-label="MOTPは位置精度、MOTAは検出寄り、IDF1はID一貫性、HOTAは検出と関連付けの両方を測る" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="28" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">指標が測るもの：検出 と ID 一貫性</text><text x="295" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">検出 (DetA)</text><text x="440" y="66" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">関連付け (AssA)</text><text x="210" y="102" text-anchor="end" font-size="14" font-weight="700" fill="#18181b">MOTP</text><text x="210" y="138" text-anchor="end" font-size="14" font-weight="700" fill="#18181b">MOTA</text><text x="210" y="174" text-anchor="end" font-size="14" font-weight="700" fill="#18181b">IDF1</text><text x="210" y="210" text-anchor="end" font-size="14" font-weight="700" fill="#18181b">HOTA</text><rect x="230" y="84" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="375" y="84" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="230" y="120" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="375" y="120" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="230" y="156" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="375" y="156" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="230" y="192" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="375" y="192" width="130" height="28" fill="none" stroke="#e4e4e7" stroke-width="1.5"/><rect x="233" y="87" width="124" height="22" fill="#2563eb"/><rect x="233" y="123" width="124" height="22" fill="#2563eb"/><rect x="378" y="123" width="38" height="22" fill="#ea580c"/><rect x="378" y="159" width="124" height="22" fill="#ea580c"/><rect x="233" y="195" width="124" height="22" fill="#2563eb"/><rect x="378" y="195" width="124" height="22" fill="#ea580c"/></svg><figcaption>4 つの MOT 指標は「<b>検出の良さ（DetA 系）</b>」と「<b>ID 一貫性（AssA 系）</b>」のどちらを測るかが違います。<b>MOTP</b> は当たった組の <b>IoU ＝ 位置精度</b>、<b>MOTA</b> は誤り総量で<b>検出寄り</b>（ID の比重は小）、<b>IDF1</b> は <b>ID 一貫性</b>、<b>HOTA</b> は <code>√(DetA × AssA)</code> で<b>検出と関連付けを分離</b>して統合します。だから MOTA だけで判断せず、ID は IDF1/HOTA を併読します。</figcaption></figure>

`04` は、実装の正しさを **サニティチェック** で保証します。具体的には、予測＝GT を入れれば
MOTA=IDF1=HOTA=1.0・IDSW=0 になることを `assert` で確認し、続いて素朴な IoU トラッカ（ノイズ
検出入力）を評価することで、誤りが各指標に反映される様子を表示します。

```bash
uv run python lectures/28_tracking/04_mot_metrics.py
# → 完璧予測: MOTA=1.000 ... / 素朴トラッカ: MOTA≈0.85, IDF1≈0.92, HOTA≈0.93
```

---

## 6. 実務の使い分け — どれを選ぶか

- **対象が 1 個・短時間・検出器が無い** → 単一物体トラッカ（CSRT が高精度、KCF が高速）。
  ただし長時間や激しい変形では見失うので、定期的な再検出を併用する。
- **多物体・検出器あり・CPU で軽快に** → **ByteTrack**（外見不要・低スコア 2 段対応で頑健）。
  本章の自前 SORT はその思想の最小実装で、まずこれを理解してから移行する。
- **多物体・見た目で取り違えを防ぎたい（似た動きの群衆・スポーツ）** → **DeepSORT** 系
  （外見 ReID 埋め込み）。本章の `03` がその核を体現している。
- **評価** → 研究比較なら HOTA（TrackEval）、運用監視なら MOTA/IDF1 の併読。MOTA だけで判断
  しない（ID 一貫性は IDF1/AssA を見る）。

> 衝突依存メモ: `supervision`(ByteTrack) / `deep_sort_realtime` / `motmetrics` / `TrackEval` は
> 本講座の numpy2 系・巨大依存と競合しうるため、**実行経路では使いません**。試したい場合のみ
> 隔離グループで任意導入してください（`uv add --group track supervision deep-sort-realtime`、
> `uv add --group metrics motmetrics`、`uv add --group metrics "trackeval @ git+https://github.com/JonathonLuiten/TrackEval"`）。

---

## 🛠 章末ミニプロジェクト — 検出 → 追跡 → 評価の統合

`mini_project.py` は、ここまでの部品を 1 本のパイプラインに統合した **完成形** です。

1. **合成カラー動画**（複数の動く物体）を生成。
2. **① 検出器**: `cv2.connectedComponentsWithStats` で前景 blob のボックスを得る（毎フレーム
   独立・ID 無し。物体が重なると 1 blob に融合する＝検出器の限界も体験できる）。
3. **② 追跡器**: 02 の自前 **SORT** を `importlib` で再利用し、フレーム間に同一 ID を付与。
4. **③ 可視化**: ID 別の色で枠と軌跡を描き、`mp4` 動画とモンタージュ PNG を保存。
5. **④ 評価**: 04 の自前 MOT 指標で **MOTA / MOTP / IDF1 / HOTA** を GT に対して算出。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="ミニプロジェクトのパイプライン。合成動画を生成し検出、自前SORTで追跡、ID別色で可視化、MOT指標で評価する順に流れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ミニプロジェクト — 検出 → 追跡 → 評価のパイプライン</text><rect x="24" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="240" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="64" width="180" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="456" y="214" width="180" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="240" y="214" width="180" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="114" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 合成動画</text><text x="114" y="112" text-anchor="middle" font-size="11" fill="#71717a">動く物体を生成</text><text x="330" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">② 検出（blob）</text><text x="330" y="112" text-anchor="middle" font-size="11" fill="#71717a">connectedComponents</text><text x="546" y="92" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">③ 追跡 SORT</text><text x="546" y="112" text-anchor="middle" font-size="11" fill="#71717a">自前 SORT で ID 付与</text><text x="546" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">④ 可視化</text><text x="546" y="262" text-anchor="middle" font-size="11" fill="#71717a">ID 別色・枠と軌跡</text><text x="330" y="242" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">⑤ MOT 評価</text><text x="330" y="262" text-anchor="middle" font-size="11" fill="#71717a">MOTA / IDF1 / HOTA</text><line x1="206" y1="96" x2="234" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="240,96 230,91 230,101" fill="#71717a"/><line x1="422" y1="96" x2="450" y2="96" stroke="#71717a" stroke-width="2"/><polygon points="456,96 446,91 446,101" fill="#71717a"/><line x1="546" y1="130" x2="546" y2="208" stroke="#71717a" stroke-width="2"/><polygon points="546,214 541,204 551,204" fill="#71717a"/><line x1="454" y1="246" x2="426" y2="246" stroke="#71717a" stroke-width="2"/><polygon points="420,246 430,241 430,251" fill="#71717a"/></svg><figcaption><b>ミニプロジェクト</b>の統合パイプラインです。<b>① 合成動画</b>を作り、<b>② 検出</b>（<code>connectedComponentsWithStats</code> で前景 blob）→ <b>③ 追跡</b>（02 の自前 <b>SORT</b> で同一 ID 付与）→ <b>④ 可視化</b>（ID 別色で枠と軌跡）→ <b>⑤ MOT 評価</b>（<code>MOTA / MOTP / IDF1 / HOTA</code>）の順に流れます。橙が生成・検出・追跡、青が可視化・評価の出力です。</figcaption></figure>

```bash
uv run python lectures/28_tracking/mini_project.py
# → MOTA≈0.85 / IDF1≈0.78 / HOTA≈0.82、追跡動画とモンタージュを outputs/28_tracking/ に保存
```

**発展課題**: `detect_blobs` を torchvision の検出器（`fasterrcnn_resnet50_fpn` 等）に差し替える、
SORT を ByteTrack 流の 2 段対応に拡張する、`03` の外見特徴を ResNet 埋め込みにする、など。
部品の境界（検出 / 対応付け / 評価）が分かれているので差し替えが容易です。

---

## ✅ 到達チェックリスト

- [ ] 検出と追跡の違い、単一物体トラッカと tracking-by-detection の違いを説明できる。
- [ ] `cv2` トラッカを `hasattr` でガードして安全に使い、CSRT(main) と KCF(legacy/contrib) の
      名前空間差を説明できる。
- [ ] カルマンフィルタの predict/update と、IoU コスト + ハンガリアン法の対応付けを自分で書ける。
- [ ] IoU 閾値ゲート・`max_age`・`min_hits` の役割を説明できる。
- [ ] ID スイッチが起きる原因と、外見特徴で減らせる理屈を説明・再現できる。
- [ ] MOTA / MOTP / IDF1 / HOTA の定義式を書け、何を測るかを区別できる。
- [ ] 検出→追跡→評価のパイプラインを最後まで動かし、結果を読める。

---

## ❓ 落とし穴・FAQ・デバッグ

- **`AttributeError: module 'cv2' has no attribute 'TrackerCSRT_create'`**: headless / 非 contrib
  ビルドでは無いことがある。`hasattr` で確認し、`cv2.legacy` も存在チェックしてから使う。
  contrib が要るなら `uv add opencv-contrib-python`（headless と排他）。
- **BGR / RGB の取り違え**: `cv2` は BGR、HF/matplotlib は RGB。外見特徴や可視化で色が壊れたら
  `cv2.cvtColor` の向きを疑う。
- **ハンガリアン法が変な組を結ぶ**: 割り当て後の **IoU 閾値ゲート** を忘れている。最適割当は
  「全部結ぶ」ので、薄い重なりは必ず足切りする。
- **ID がチラつく/すぐ消える**: `min_hits` を上げて確定までの猶予を、`max_age` を上げて
  欠測時の生存期間を調整する。検出の取りこぼしが多いなら検出側を先に直す。
- **ID がすぐ入れ替わる**: 物体が交差している。`03` のように外見特徴を足すか、ByteTrack の
  2 段対応を検討する。IoU 閾値が高すぎても対応が切れてスイッチが増える。
- **MOTA が負になる**: 仕様どおりの挙動。FN+FP+IDSW が GT を超えると負になる。MOTA だけで
  良し悪しを判断せず、IDF1/HOTA も併読する。
- **MOTP が高いのに MOTA が低い**: MOTP は「当たった組」だけを見るので、取りこぼし(FN)が
  多いと MOTA だけ落ちる。両者の役割の違いを思い出す。
- **scipy が無い**: `linear_assignment` は貪欲法に自動フォールバックするので動くが、最適性は
  落ちる。可能なら `scikit-learn`/`scipy` を入れてハンガリアン法を使う。
- **動画 mp4 が書けない**: 環境によっては `VideoWriter` が開けない。`mini_project.py` は
  `isOpened()` を確認し、ダメならモンタージュ PNG にフォールバックして exit 0 を保つ。

---

## 🚀 発展トピック・参考

- **ByteTrack**（Zhang+ 2022）: 高/低スコア検出を 2 段で対応付け、純 Kalman+Hungarian・CPU 軽量。
  正準実装は `supervision.ByteTrack`（任意）。本章の SORT を 2 段化すると近づける。
- **DeepSORT**（Wojke+ 2017）: 動き(Kalman) + 外見(ReID 埋め込み) + マッチングカスケード。
  `deep_sort_realtime`（任意）。本章 `03` がその核。
- **OC-SORT / BoT-SORT / StrongSORT**: SORT 系の改良版（観測中心の補正、カメラ運動補償、強い
  外見特徴など）。SORT を理解していれば差分として読める。
- **HOTA / TrackEval**: 公式 MOT 評価。HOTA は IoU 閾値 0.05〜0.95 で `DetA×AssA` を平均する。
  PyPI 安定版が無いため Git pin で導入（任意）。本章は単一閾値の簡易版で原理を掴む。
- **データセット**: MOT17/MOT20、DanceTrack(交差が多く外見が効く)、KITTI/BDD100K(運転)。
- 参考: SORT (Bewley+ 2016) `arXiv:1602.00763`、DeepSORT `arXiv:1703.07402`、
  ByteTrack `arXiv:2110.06864`、HOTA (Luiten+ 2021) IJCV。

---

## 💡 実践ユースケース集

ここまで学んできた「追跡」は、現実では**「数える・測る・気づく」アプリの土台**になります。
そこで、検出と評価を追う `mini_project.py`（ベンチ寄りの統合課題）とは別に、**そのまま製品に
なりうる小ツール**をいくつか挙げます。これらに共通する作り方は、「**検出 → SORT で ID 付与 →
ID ごとの軌跡にアプリ固有のルールを当てる**」というものです。

<figure class="lec-fig"><svg viewBox="0 0 600 230" role="img" aria-label="ユースケース共通レシピ。検出してSORTでID付与し、ID別の軌跡にアプリ固有のルールを当てる3段でライン通過や在室人数など別アプリになる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="300" y="36" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">共通レシピ — 検出 → ID 付与 → 軌跡にルールを当てる</text><rect x="20" y="80" width="170" height="72" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="105" y="112" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">① 検出</text><text x="105" y="134" text-anchor="middle" font-size="10.5" fill="#71717a">毎フレーム独立・ID なし</text><line x1="190" y1="116" x2="218" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="224,116 214,111 214,121" fill="#71717a"/><rect x="224" y="80" width="170" height="72" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="309" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#c2410c">② SORT で ID 付与</text><text x="309" y="134" text-anchor="middle" font-size="10.5" fill="#71717a">フレーム間で対応付け</text><line x1="394" y1="116" x2="422" y2="116" stroke="#71717a" stroke-width="2"/><polygon points="428,116 418,111 418,121" fill="#71717a"/><rect x="428" y="80" width="152" height="72" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="504" y="112" text-anchor="middle" font-size="14" font-weight="700" fill="#1d4ed8">③ 軌跡にルール適用</text><text x="504" y="134" text-anchor="middle" font-size="10.5" fill="#71717a">ID 別の中心点列</text><text x="300" y="204" text-anchor="middle" font-size="12" fill="#52525b">最後の段を差し替え → ① ライン通過 ／ ② 在室人数 ／ ③ 滞留・速度</text></svg><figcaption>3 つのユースケースに共通する<b>作り方のレシピ</b>です。<b>① 検出</b>（毎フレーム独立・ID なし）→ <b>② SORT で ID 付与</b>（フレーム間で対応付け）→ <b>③ ID 別の軌跡にアプリ固有のルールを当てる</b>、の 3 段で組みます。<b>最後の段を差し替える</b>だけで、<b>ライン通過カウント・在室人数・滞留/速度</b>など別アプリになります。ID を保つから同じ物体を<b>二重に数えずに済みます</b>。</figcaption></figure>

### ① ライン通過カウンター（`use_case.py`・動く出発点）

- **何に使うか**: 店舗の入店者数、横断歩道や通路の通行量、生産ラインの製品計数、道路の交通量
  調査。画面に引いた**仮想ライン**を物体が横切った回数を**方向別**に数えます。
- **作り方の要点**: 各 ID の中心点について、ライン（端点 `p1→p2`）に対する**符号付きの側**
  （外積の z 成分）を毎フレーム見て、**符号が反転したら 1 回横切った**と判定します。反転の
  向き（正側へ／負側へ）で「下り/上り」「入店/退店」を区別できます。追跡で ID を保つから、
  **同じ物体を二重カウントしない**のがポイントです。
- **注意**: 検出が途切れると ID が変わり、再カウントや取りこぼしが起きます（`max_age` を上げる、
  検出器を強くする）。ラインの位置・向きはシーンに合わせて調整します。合成入力では検出は
  輝度しきい値、実動画では背景差分(MOG2)を使う簡易検出なので、**本番は YOLO/torchvision 検出器に
  差し替える**と精度が上がります。

```bash
uv run python lectures/28_tracking/use_case.py
# → 合成 or data/28_tracking/ の動画で通過数を方向別に集計し、
#   モンタージュ/時系列グラフ/カウント JSON を outputs/28_tracking/ に保存
```

- **data 配置**: `data/28_tracking/` に**動画**（`*.mp4` / `*.avi` / `*.mov` / `*.mkv`）か
  **連番画像**（`*.png` / `*.jpg`）を置くと実入力で動きます（例: `data/28_tracking/street.mp4`）。
  無ければ合成シーン（壁で反射する 4 物体）で必ず完走します。
- **拡張アイデア**: 複数ライン／多角形ゾーンで「ゾーン内滞在数」を測る、方向別カウントの差分で
  **現在の在室人数**を出す、`detect` を小型検出器に替えて**クラス別**（人だけ/車だけ）に数える、
  通過時刻を記録して**時間帯別ヒストグラム**でピーク時間を可視化する、など。

### ② 在室人数モニタ（occupancy）

- **何に使うか**: 会議室・店舗・展示ブースの「今この空間に何人いるか」をリアルタイム表示。
- **作り方の要点**: ①のライン通過カウンターを**入口に 1 本**引き、`入店数 − 退店数` を逐次
  足し引きするだけ。方向別カウントがそのまま在室人数の増減になります。複数の出入口があるなら
  各ラインの符号を入口の向きに合わせて合算します。
- **注意**: 検出漏れで「入ったのに出た記録だけ残る」と負の人数になります。下限 0 でクリップし、
  一定間隔で**実数（基準値）にリセット**できる仕組みを足すと運用が安定します。

### ③ 滞留・速度の簡易推定（dwell / speed）

- **何に使うか**: レジ待ち行列の滞留検知、棚前の立ち止まり計測、車両のおおよその速度推定。
- **作り方の要点**: SORT が保つ **ID 別の軌跡（中心点列）** を使い、移動距離が小さいフレームが
  続いたら「滞留」、フレーム間の移動量×フレームレートで「速度」を推定します。追跡が ID を
  保つので、**同じ対象の時間方向の挙動**を測れるのが肝です。
- **注意**: 速度はピクセル単位なので、実寸が要るなら**ホモグラフィでの座標変換（俯瞰変換）**が
  必要です。短いオクルージョンで軌跡が途切れないよう `max_age` を確保し、外見特徴（`03`）で
  ID スイッチを抑えると計測が安定します。

---

## ▶ 動かし方

```bash
# 共有ユーティリティの自己テスト
uv run python lectures/28_tracking/_common.py
# 1) 単一物体トラッカ（OpenCV, hasattr ガード）
uv run python lectures/28_tracking/01_single_object_tracker.py
# 2) 自前 SORT（カルマン + ハンガリアン）
uv run python lectures/28_tracking/02_sort_from_scratch.py
# 3) 外見特徴で ID スイッチ抑制（DeepSORT の核）
uv run python lectures/28_tracking/03_kalman_appearance.py
# 4) MOT 評価指標を自前実装で検証
uv run python lectures/28_tracking/04_mot_metrics.py
# 章末ミニプロジェクト（検出→追跡→評価の統合）
uv run python lectures/28_tracking/mini_project.py
# 実践ユースケース（ライン通過カウンター。data/28_tracking/ に動画を置けば実入力）
uv run python lectures/28_tracking/use_case.py
# 演習（自己採点）と模範解答
uv run python lectures/28_tracking/exercises.py
uv run python lectures/28_tracking/exercises_solutions.py
```

出力（可視化・動画）は `outputs/28_tracking/` に保存されます。すべて CPU・合成データで完結し、
ネット接続もモデル重みも不要です（OpenCV トラッカに重みファイルは不要）。

---

> 参照ライブラリ（版）: opencv-python-headless 4.13 / numpy 2.x / scipy 1.x（任意）/
> torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11（本章では未使用）。
> 衝突依存（supervision/deep_sort_realtime/motmetrics/TrackEval）は実行経路では使わず、
> 概念紹介＋任意導入にとどめています。 — 2026-06