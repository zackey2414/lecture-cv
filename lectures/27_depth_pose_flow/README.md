# 27_depth_pose_flow: 単眼深度・姿勢/キーポイント・オプティカルフロー

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `metrics`（`pose` は概念のみ・任意）
>
> 参照ライブラリ: torch **2.12+cpu** / torchvision **0.27+cpu** / transformers **5.11** / opencv(headless) **4.13** / numpy **2.x**（2026-06 時点）

---

## 🎯 この章のゴール

1枚の画像、1人の人物、2枚の連続フレーム——この章では、これらから「**奥行き(depth)**」「**関節(pose/keypoints)**」「**動き(optical flow)**」という、シーンの3次元・運動構造に踏み込んだ情報を取り出せるようになることをゴールとします。具体的には、次の4点を AI 補助なしで自力で書けるところまで持っていきます。

- **単眼深度**: `transformers.pipeline('depth-estimation')` で Depth Anything V2 (Small) を動かし、出力が **相対(逆)深度**（値が大きいほど近い・絶対距離ではない）であることを理解し、可視化のための正規化と、評価のためのスケール合わせを行える。
- **姿勢/キーポイント**: torchvision `keypointrcnn_resnet50_fpn` で **COCO 17 点**を推定し、スケルトン（辺）の張り方と、検出器が「人を検出 → 関節を回帰」する2段構えを理解する。あわせて**合成図形では検出が発火しない domain gap** を体験する。
- **オプティカルフロー**: torchvision `raft_small` で密フローを推定し、RAFT 3 大前処理（**[-1,1] 正規化・8 の倍数サイズ・反復出力の最後を採用**）を踏める。古典法（Lucas-Kanade / Farneback）と対比し、輝度一定仮定の限界を語れる。
- **評価**: 深度 **AbsRel / RMSE / δ<1.25**、フロー **EPE**、姿勢 **OKS / PCK@0.2** を、合成 GT に対して numpy で実装・検算できる。

入力はすべて**合成生成**で（床・壁・箱・球の擬似室内、並進するテクスチャ画像対、腕を振る人型図形）、ネットに出るのは**モデル重みのダウンロードだけ**です。`data/27_depth_pose_flow/` に実画像を置けば、そちらを優先します。

---


## 0. 全体像 — なぜこの3つを1章にまとめるのか

深度・姿勢・フローは、一見バラバラなタスクに見えますが、いずれも「**画素そのもの**」ではなく「**画素が表す幾何・運動の量**」を回帰する密予測（dense/structured prediction）という共通の骨格を持ちます。分類や検出が「離散ラベル」を出すのに対し、ここで扱うのは深度値・関節座標・フローベクトルといった**連続量**で、評価も accuracy ではなく「真値との連続的なズレ」を測ります。だからこそ AbsRel・EPE・OKS のような**回帰系の指標**が主役になります。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="1枚の入力から深度・姿勢・フローという画素ごとの連続量を回帰する密予測の全体像" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="30" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">1 枚の入力 → 画素ごとの連続量を回帰（密予測）</text><rect x="40" y="118" width="120" height="120" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/><line x1="40" y1="206" x2="160" y2="206" stroke="#e4e4e7" stroke-width="1.5"/><rect x="60" y="166" width="34" height="40" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/><circle cx="126" cy="194" r="16" fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/><text x="100" y="262" text-anchor="middle" font-size="13" fill="#3f3f46">入力（画像/フレーム）</text><line x1="160" y1="178" x2="250" y2="178" stroke="#71717a" stroke-width="2"/><line x1="250" y1="178" x2="352" y2="100" stroke="#71717a" stroke-width="2"/><line x1="250" y1="178" x2="350" y2="176" stroke="#71717a" stroke-width="2"/><line x1="250" y1="178" x2="352" y2="252" stroke="#71717a" stroke-width="2"/><polygon points="360,94 355,104 349,96" fill="#71717a"/><polygon points="360,176 349,170 349,182" fill="#71717a"/><polygon points="360,258 349,256 355,248" fill="#71717a"/><rect x="360" y="58" width="250" height="72" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><rect x="372" y="70" width="18" height="48" fill="#fff7ed"/><rect x="390" y="70" width="18" height="48" fill="#ffedd5"/><rect x="408" y="70" width="18" height="48" fill="#f97316"/><rect x="426" y="70" width="18" height="48" fill="#ea580c"/><rect x="444" y="70" width="18" height="48" fill="#c2410c"/><text x="540" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">深度（近=明）</text><rect x="360" y="140" width="250" height="72" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><circle cx="400" cy="158" r="5" fill="#2563eb"/><line x1="400" y1="163" x2="400" y2="186" stroke="#2563eb" stroke-width="2"/><line x1="400" y1="168" x2="386" y2="180" stroke="#2563eb" stroke-width="2"/><line x1="400" y1="168" x2="414" y2="180" stroke="#2563eb" stroke-width="2"/><line x1="400" y1="186" x2="390" y2="200" stroke="#2563eb" stroke-width="2"/><line x1="400" y1="186" x2="410" y2="200" stroke="#2563eb" stroke-width="2"/><text x="520" y="181" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">姿勢 17 点 (x, y)</text><rect x="360" y="222" width="250" height="72" rx="6" fill="#fafafa" stroke="#e4e4e7" stroke-width="1.5"/><line x1="378" y1="244" x2="400" y2="244" stroke="#16a34a" stroke-width="2"/><polygon points="406,244 397,240 397,248" fill="#16a34a"/><line x1="378" y1="260" x2="400" y2="260" stroke="#16a34a" stroke-width="2"/><polygon points="406,260 397,256 397,264" fill="#16a34a"/><line x1="378" y1="276" x2="400" y2="276" stroke="#16a34a" stroke-width="2"/><polygon points="406,276 397,272 397,280" fill="#16a34a"/><text x="520" y="263" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">フロー (u, v)/画素</text></svg><figcaption>この章の3タスクはどれも<b>密予測（dense prediction）</b>——分類や検出のような<b>離散ラベル</b>ではなく、<b>画素ごと（または点ごと）の連続量</b>を回帰します。<code>深度</code>は奥行き値、<code>姿勢</code>は関節座標 (x, y)、<code>フロー</code>は移動ベクトル (u, v)。だから評価も accuracy ではなく、真値との<b>連続的なズレ</b>（AbsRel・EPE・OKS）で測ります。</figcaption></figure>

また、3つのタスクは応用上もよく一緒に使われます。たとえば AR・ロボティクス・スポーツ解析では「人がどこにいて(pose)、どれだけ手前にいて(depth)、次の瞬間どこへ動くか(flow)」を同時に知りたい場面が多くあります。本章の `mini_project.py` は、1本の合成クリップに深度→姿勢→フローを順に適用して1枚のレポートにまとめることで、まさにこの統合を実演します。

技術的な共通の注意点もあります。第一に、いずれも**学習時の前処理を一字一句守らないと無意味な出力になる**こと（深度の正規化、フローの [-1,1]・8の倍数、姿勢の正規化）。第二に、**CPU では小モデル＋低解像度＋ `inference_mode()`** を徹底しないと現実的な速度にならないことです。その意味で、この章は「前処理の勘所」を体に叩き込む章でもあります。

## 1. 単眼深度推定（Depth Anything V2）

**直感.** 人間は片目でも「手前/奥」が分かります。テクスチャの密度、ものの大きさ、遮蔽関係、陰影といった手がかりから脳が奥行きを推定しているからです。単眼深度モデルは、大量の画像とその深度の組から、この「1枚→奥行き」の写像を学んだものです。Depth Anything V2 は大規模な擬似ラベルで学習された強力な相対深度モデルで、Small 版なら CPU でも数秒で動きます。

**理論.** ここで決定的に重要なのは、単眼相対深度モデルの出力が**絶対距離ではない**ことです。Depth Anything が返す `predicted_depth` は**逆深度（disparity 的な量）**で、慣習として「**値が大きいほどカメラに近い**」という向きを持ちます。スケールとシフトも不定（affine-invariant）なので、「3.0 という値が 3 メートル」を意味しません。したがって、可視化の前には必ず正規化（min-max や percentile）が要りますし、GT と数値比較するなら**スケール合わせ**（中央値比や最小二乗での scale/shift フィット）が前提になります。なお、メートル単位が必要なら Metric 系のモデルを別途使います。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="単眼深度の出力は逆深度で近いほど値が大きく遠いほど小さい。絶対距離ではなくスケールとシフトが不定" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">単眼の出力 ＝ 逆深度（近いほど大）・絶対距離ではない</text><rect x="34" y="150" width="40" height="34" rx="3" fill="#3f3f46"/><polygon points="74,156 92,144 92,194 74,188" fill="#52525b"/><line x1="92" y1="170" x2="128" y2="176" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="92" y1="170" x2="340" y2="190" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><rect x="128" y="148" width="58" height="58" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/><text x="157" y="140" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">近い</text><rect x="206" y="92" width="40" height="118" fill="#ea580c"/><text x="226" y="84" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">逆深度 大</text><rect x="340" y="170" width="40" height="40" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="360" y="162" text-anchor="middle" font-size="13" font-weight="700" fill="#2563eb">遠い</text><rect x="414" y="164" width="40" height="46" fill="#f97316"/><text x="434" y="156" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">逆深度 小</text><line x1="196" y1="210" x2="470" y2="210" stroke="#71717a" stroke-width="1.5"/><line x1="30" y1="236" x2="600" y2="236" stroke="#71717a" stroke-width="2"/><polygon points="610,236 598,230 598,242" fill="#71717a"/><text x="300" y="258" text-anchor="middle" font-size="13" fill="#3f3f46">実距離 Z（カメラからの奥行き）→</text><text x="556" y="64" text-anchor="middle" font-size="14" font-weight="700" fill="#52525b">d ≈ 1 / Z</text></svg><figcaption>単眼相対深度モデルが返す <code>predicted_depth</code> は<b>逆深度</b>で、慣習として<b>カメラに近い画素ほど値が大きく</b>、遠いほど小さくなります。さらに<b>絶対スケールもシフトも不定</b>（affine-invariant）なので「3.0 ＝ 3 メートル」ではありません。可視化前には<b>正規化</b>、GT との数値比較前には<b>スケール合わせ</b>（中央値比や最小二乗の scale/shift）が必須です。</figcaption></figure>

**正準API.** transformers v5 では次が定石です（v5 で `AutoFeatureExtractor` は廃止、画像処理は torchvision バックエンドの fast 実装に統一）。

```python
from transformers import pipeline
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
out = pipe(pil_image)            # device 省略で CPU
inv_depth = out["predicted_depth"]   # torch.Tensor (H, W)  逆深度（近い=大）
depth_pil = out["depth"]             # 0-255 に正規化済みの PIL 画像（手早い可視化用）
```

低レベルに分解したいときは `AutoImageProcessor` + `AutoModelForDepthEstimation` を使い、`post_process_depth_estimation` で元解像度へ戻します。

**実装（`01_depth_anything.py`）.** 合成室内シーン（床・壁・箱・球、各領域に既知の GT 深度）を作り、pipeline で逆深度を推定 → percentile 正規化して magma カラーマップで可視化 → 入力と並べて保存します。評価では `1/inv_depth` で深度に戻し、**中央値スケール合わせ**してから AbsRel・RMSE・δ<1.25 を算出します（合成はモデルの学習分布外なので数値は参考値、定義の確認が目的）。ネット不通時はランプ状のフォールバック深度で最後まで通します。

<figure class="lec-fig"><svg viewBox="0 0 660 320" role="img" aria-label="深度モデルの出力predicted_depthは用途で処理が分かれる。可視化経路は正規化とカラーマップ、評価経路は深度化と中央値スケール合わせと指標" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="30" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">深度の出力をどう使う？ 可視化経路 と 評価経路</text><rect x="14" y="74" width="108" height="192" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><text x="68" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">モデル出力</text><text x="68" y="150" text-anchor="middle" font-size="9.5" fill="#3f3f46">predicted_depth</text><text x="68" y="174" text-anchor="middle" font-size="10.5" fill="#52525b">逆深度 (H×W)</text><text x="68" y="200" text-anchor="middle" font-size="10.5" font-weight="700" fill="#c2410c">近い = 大</text><text x="397" y="66" text-anchor="middle" font-size="12.5" font-weight="700" fill="#2563eb">可視化経路 — 見せるための処理</text><rect x="150" y="78" width="140" height="62" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="327" y="78" width="140" height="62" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="504" y="78" width="140" height="62" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="220" y="104" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">percentile 正規化</text><text x="220" y="124" text-anchor="middle" font-size="10.5" fill="#52525b">[0, 1] に圧縮</text><text x="397" y="104" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">magma</text><text x="397" y="124" text-anchor="middle" font-size="10.5" fill="#52525b">カラーマップ</text><text x="574" y="104" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">可視化画像</text><text x="574" y="124" text-anchor="middle" font-size="10" fill="#52525b">入力と並置で保存</text><text x="397" y="176" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">評価経路 — 測るための処理</text><rect x="150" y="188" width="140" height="62" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><rect x="327" y="188" width="140" height="62" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><rect x="504" y="188" width="140" height="62" rx="7" fill="#fafafa" stroke="#16a34a" stroke-width="2"/><text x="220" y="214" text-anchor="middle" font-size="11.5" font-weight="700" fill="#15803d">1/(inv+ε) で深度へ</text><text x="220" y="234" text-anchor="middle" font-size="10.5" fill="#52525b">逆深度を反転</text><text x="397" y="214" text-anchor="middle" font-size="10.5" font-weight="700" fill="#15803d">中央値スケール合わせ</text><text x="397" y="234" text-anchor="middle" font-size="10" fill="#52525b">scale を GT に一致</text><text x="574" y="214" text-anchor="middle" font-size="11.5" font-weight="700" fill="#15803d">深度の指標</text><text x="574" y="234" text-anchor="middle" font-size="10.5" fill="#52525b">AbsRel · RMSE · δ</text><line x1="124" y1="109" x2="144" y2="109" stroke="#71717a" stroke-width="2"/><polygon points="150,109 140,104 140,114" fill="#71717a"/><line x1="124" y1="219" x2="144" y2="219" stroke="#71717a" stroke-width="2"/><polygon points="150,219 140,214 140,224" fill="#71717a"/><line x1="292" y1="109" x2="321" y2="109" stroke="#71717a" stroke-width="2"/><polygon points="327,109 317,104 317,114" fill="#71717a"/><line x1="469" y1="109" x2="498" y2="109" stroke="#71717a" stroke-width="2"/><polygon points="504,109 494,104 494,114" fill="#71717a"/><line x1="292" y1="219" x2="321" y2="219" stroke="#71717a" stroke-width="2"/><polygon points="327,219 317,214 317,224" fill="#71717a"/><line x1="469" y1="219" x2="498" y2="219" stroke="#71717a" stroke-width="2"/><polygon points="504,219 494,214 494,224" fill="#71717a"/></svg><figcaption>単眼深度モデルの出力 <code>predicted_depth</code>（<b>逆深度</b>・近い=大）は、用途で処理が分かれます。<b>可視化経路</b>（見せる）は <b>percentile 正規化</b>で [0,1] に圧縮し magma カラーマップへ。<b>評価経路</b>（測る）は <code>1/(inv+ε)</code> で深度へ戻し、<b>中央値スケール合わせ</b>でスケールを GT に合わせてから <b>AbsRel・RMSE・δ</b> を計算します。<b>正規化（可視化用）とスケール合わせ（評価用）を混同しない</b>のが要点です。</figcaption></figure>

**落とし穴.** ①逆深度の向きを取り違えて near/far が反転する。②スケール合わせを忘れて AbsRel が極端に大きく出る（`04` の `abs_rel_before_align` と `abs_rel` の差で体感できます）。③`out["depth"]`（可視化済み PIL）を生の深度値と勘違いして指標計算に使う。

**実務の使い分け.** 相対深度で十分な用途（ボケ・背景合成・相対的な遠近の把握・3D風エフェクト）には Depth Anything V2 が最有力。測距・SLAM・占有格子など**メートルが要る**用途には Metric Depth 系か、ステレオ/LiDAR/ToF の併用を検討します。

## 2. 姿勢・キーポイント推定（keypoint R-CNN, COCO17）

**直感.** 姿勢推定は「人の体のどこに鼻・肩・肘・手首…があるか」を点で当てるタスクです。点同士を**スケルトン（辺）**で結ぶと棒人間になり、動作・ジェスチャ・転倒検知などの土台になります。COCO の標準は**17 点**（顔5＋上半身6＋下半身6）です。

**理論.** keypoint R-CNN は2段構えです。まず人を**検出**（bbox とスコア）し、各人の RoI から**17 関節を回帰**します。出力 `keypoints` は `(人数, 17, 3)` で、最後の 3 は `(x, y, score)`。COCO の点には**左右の区別**（その人にとっての左肩/右肩）があり、辺の張り方（肩-肘-手首、肩-腰、腰-膝-足首）が固定されています。トップダウン型（検出→関節）なので、**人が検出できなければ関節も出ません**——ここが本章最大の落とし穴に直結します。

<figure class="lec-fig"><svg viewBox="0 0 620 340" role="img" aria-label="COCO17点スケルトンの構造。顔5点・上半身6点・下半身6点を辺で結ぶ。出力は人数×17×3で各点x,y,score" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><rect x="64" y="42" width="148" height="290" rx="4" fill="none" stroke="#c2410c" stroke-width="2" stroke-dasharray="6 4"/><line x1="135" y1="64" x2="126" y2="55" stroke="#71717a" stroke-width="2.5"/><line x1="135" y1="64" x2="144" y2="55" stroke="#71717a" stroke-width="2.5"/><line x1="126" y1="55" x2="117" y2="60" stroke="#71717a" stroke-width="2.5"/><line x1="144" y1="55" x2="153" y2="60" stroke="#71717a" stroke-width="2.5"/><line x1="105" y1="108" x2="165" y2="108" stroke="#71717a" stroke-width="2.5"/><line x1="105" y1="108" x2="117" y2="60" stroke="#71717a" stroke-width="2.5"/><line x1="165" y1="108" x2="153" y2="60" stroke="#71717a" stroke-width="2.5"/><line x1="105" y1="108" x2="90" y2="156" stroke="#71717a" stroke-width="2.5"/><line x1="90" y1="156" x2="80" y2="202" stroke="#71717a" stroke-width="2.5"/><line x1="165" y1="108" x2="180" y2="156" stroke="#71717a" stroke-width="2.5"/><line x1="180" y1="156" x2="190" y2="202" stroke="#71717a" stroke-width="2.5"/><line x1="105" y1="108" x2="116" y2="190" stroke="#71717a" stroke-width="2.5"/><line x1="165" y1="108" x2="154" y2="190" stroke="#71717a" stroke-width="2.5"/><line x1="116" y1="190" x2="154" y2="190" stroke="#71717a" stroke-width="2.5"/><line x1="116" y1="190" x2="110" y2="252" stroke="#71717a" stroke-width="2.5"/><line x1="110" y1="252" x2="104" y2="312" stroke="#71717a" stroke-width="2.5"/><line x1="154" y1="190" x2="160" y2="252" stroke="#71717a" stroke-width="2.5"/><line x1="160" y1="252" x2="166" y2="312" stroke="#71717a" stroke-width="2.5"/><circle cx="135" cy="64" r="5" fill="#ea580c"/><circle cx="126" cy="55" r="5" fill="#ea580c"/><circle cx="144" cy="55" r="5" fill="#ea580c"/><circle cx="117" cy="60" r="5" fill="#ea580c"/><circle cx="153" cy="60" r="5" fill="#ea580c"/><circle cx="105" cy="108" r="5" fill="#2563eb"/><circle cx="165" cy="108" r="5" fill="#2563eb"/><circle cx="90" cy="156" r="5" fill="#2563eb"/><circle cx="180" cy="156" r="5" fill="#2563eb"/><circle cx="80" cy="202" r="5" fill="#2563eb"/><circle cx="190" cy="202" r="5" fill="#2563eb"/><circle cx="116" cy="190" r="5" fill="#16a34a"/><circle cx="154" cy="190" r="5" fill="#16a34a"/><circle cx="110" cy="252" r="5" fill="#16a34a"/><circle cx="160" cy="252" r="5" fill="#16a34a"/><circle cx="104" cy="312" r="5" fill="#16a34a"/><circle cx="166" cy="312" r="5" fill="#16a34a"/><text x="440" y="68" text-anchor="middle" font-size="16" font-weight="700" fill="#18181b">COCO 17 点スケルトン</text><text x="312" y="108" font-size="14" font-weight="700" fill="#18181b">出力 (N, 17, 3) = (x, y, score)</text><circle cx="320" cy="150" r="6" fill="#ea580c"/><text x="338" y="155" font-size="13" fill="#3f3f46">顔 5 点（鼻・目・耳）</text><circle cx="320" cy="180" r="6" fill="#2563eb"/><text x="338" y="185" font-size="13" fill="#3f3f46">上半身 6 点（肩・肘・手首）</text><circle cx="320" cy="210" r="6" fill="#16a34a"/><text x="338" y="215" font-size="13" fill="#3f3f46">下半身 6 点（腰・膝・足首）</text><text x="312" y="262" font-size="13" font-weight="700" fill="#c2410c">① 人を検出(bbox) → ② 17 関節を回帰</text></svg><figcaption>人体姿勢の標準は <b>COCO 17 点</b>（顔5＋上半身6＋下半身6）で、点を<b>スケルトン（辺）</b>で結ぶと棒人間になります。keypoint R-CNN は<b>トップダウン2段</b>——まず<b>人を検出</b>（bbox）し、各人の領域から<b>17 関節を回帰</b>——なので出力は <code>(N, 17, 3)</code>＝各点 <code>(x, y, score)</code>。<b>人が検出できなければ関節も出ません</b>（合成図形での未検出＝domain gap の正体）。</figcaption></figure>

**なぜ MediaPipe ではないのか.** 設計上は MediaPipe Pose Landmarker（人体33点・XNNPACK で CPU 実時間）も有力ですが、mediapipe は **numpy<2 ピンや protobuf の競合**を起こしやすく、本講座の numpy2 系環境と衝突します。そこで**実行経路は torchvision の keypoint R-CNN（COCO17）で完結**させ、MediaPipe は「概念＋任意導入」に留めます（`02` 末尾で import を `try/except` ガードし、未導入なら案内のみ）。導入したい場合は別環境で `uv add --group pose mediapipe`。

**正準API.**

```python
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=weights, progress=False).eval()
img = torch.from_numpy(rgb).permute(2,0,1).float()/255.0   # 正規化は内部で行うので [0,1] でよい
with torch.inference_mode():
    out = model([img])[0]   # keys: boxes, labels, scores, keypoints, keypoints_scores
print(weights.meta["keypoint_names"])   # COCO17 の順序
```

**実装（`02_pose_keypoints.py`）.** 腕を振る人型図形を4フレーム合成（**こちらが真の関節位置を知っている**）→ keypoint R-CNN を実行 → 検出があれば最高スコアの人のスケルトンを描画、無ければ GT 関節で代替描画 → モンタージュ保存。

**domain gap という最重要の落とし穴.** 実写の人で学習したモデルは「おもちゃの合成図形」をほぼ検出できません（本実装でも `top_score=0.00` で全フレーム未検出になります）。これはバグではなく、**学習分布(実写)とテスト分布(合成)のズレ=domain gap** の実演です。実務でも「自社データはモデルの学習分布と違う」ことが性能劣化の主因になります。`data/27_depth_pose_flow/` に実写人物画像を置けば、同じコードでモデルが発火します。

**実務の使い分け.** 多人数・遮蔽が多い → トップダウン型(keypoint R-CNN)は人数に比例して重いがオクルージョンに比較的強い。CPU 実時間・単人 → MediaPipe（別環境）。研究用の高精度 → HRNet/ViTPose 系。

## 3. オプティカルフロー（RAFT vs 古典）

**直感.** オプティカルフローは「連続2フレーム間で、各画素がどこへ動いたか」のベクトル場 `(u, v)` です。動画の動き解析・フレーム補間・追跡・行動認識の前段になります。

**理論.** 古典法は**輝度一定仮定**（同じ点はフレーム間で明るさを保つ）と**滑らかさ**に基づきます。ここから導かれる基本拘束は、1本の式に未知数が2つ（u, v）あるため**開口問題(aperture problem)**——エッジに沿った成分が決まらない——を生みます。そこで、近傍をまとめる（Lucas-Kanade: 疎、特徴点ごと）か、全画素を平滑化する（Farneback: 密）かで解きます。一方、深層の RAFT は**全ペア相関ボリューム + 反復更新(GRU)**で、大変位・無テクスチャ・遮蔽に強い密フローを出します。

<figure class="lec-fig"><svg viewBox="0 0 640 300" role="img" aria-label="開口問題。小さな窓からはエッジに垂直な法線成分しか観測できず真の動きが1つに定まらない" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">開口問題：小さな窓からは「動き」が1つに定まらない</text><path d="M 180,76 A 92 92 0 0 0 180,260 Z" fill="#e4e4e7"/><path d="M 180,76 A 92 92 0 0 1 180,260 Z" fill="#ffffff"/><circle cx="180" cy="168" r="92" fill="none" stroke="#3f3f46" stroke-width="2"/><line x1="180" y1="76" x2="180" y2="260" stroke="#18181b" stroke-width="3"/><line x1="262" y1="112" x2="262" y2="224" stroke="#d4d4d8" stroke-width="1.4" stroke-dasharray="4 3"/><line x1="180" y1="168" x2="262" y2="118" stroke="#71717a" stroke-width="2" stroke-dasharray="6 4"/><polygon points="262,118 251,118 257,128" fill="#71717a"/><line x1="180" y1="168" x2="262" y2="218" stroke="#71717a" stroke-width="2" stroke-dasharray="6 4"/><polygon points="262,218 251,218 257,208" fill="#71717a"/><line x1="180" y1="168" x2="262" y2="168" stroke="#ea580c" stroke-width="3"/><polygon points="266,168 255,163 255,173" fill="#ea580c"/><circle cx="180" cy="168" r="4" fill="#18181b"/><text x="300" y="110" font-size="13" font-weight="700" fill="#71717a">真の動き (u, v)？</text><text x="300" y="172" font-size="13" font-weight="700" fill="#c2410c">観測 = 法線成分のみ</text><text x="70" y="290" font-size="13" fill="#3f3f46">式 1・未知数 2 (u, v) → 近傍をまとめて解く（LK / Farneback / RAFT）</text></svg><figcaption><b>開口問題（aperture problem）</b>：小さな窓からエッジ（縦の輪郭）の動きを見ると、<b>エッジに垂直な成分（法線成分）しか観測できません</b>。エッジに沿う成分は見えないので、同じ見え方を与える<b>真の動き (u, v) は1つに定まりません</b>（図の破線はどれも同じ水平成分を持つ別々の動き）。輝度一定の基本拘束は<b>式1本・未知数2つ</b>のため、近傍をまとめる（Lucas-Kanade／Farneback）か相関で解く（RAFT）必要があります。</figcaption></figure>

**RAFT 3 大前処理（最重要）.** ①入力は **[-1,1] 正規化**（`weights.transforms()` がやる）。②高さ・幅は **8 の倍数**（内部で 1/8 に下げるため。さらに**最低 128px** ないと相関ピラミッドが作れずエラー）。③出力は**反復ごとのフローのリスト**で、採用するのは**最後 `flows[-1]`**。可視化は `torchvision.utils.flow_to_image`（色相=向き・彩度/明度=大きさ）。

**正準API.**

```python
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights, progress=False).eval()
ta, tb = weights.transforms()(img1, img2)   # [-1,1] へ
with torch.inference_mode():
    flows = model(ta, tb)        # 反復のリスト
flow = flows[-1]                 # (N, 2, H, W) 最後を採用
from torchvision.utils import flow_to_image
vis = flow_to_image(flow)        # uint8 RGB
```

**実装（`03_raft_optical_flow.py`）.** 並進する画像対（**全画素が一定 `(dx,dy)` で真フローが自明**）を合成し、RAFT・Farneback・Lucas-Kanade の **EPE** を比較します。易しい一定並進＋十分なテクスチャでは古典法でも当たりますが、大変位・無テクスチャ・遮蔽では RAFT の優位が出ます。

**落とし穴.** ①128px 未満を入れて `Feature maps are too small` で落ちる。②`transforms()` を通さず生の [0,255] を入れて精度が崩壊。③`flows`（リスト）をそのままテンソル扱いする。④`torchvision.io.read_video` は **0.26 で廃止**——動画 I/O は `cv2.VideoCapture` を使う（本講座は合成生成なので不要）。

## 4. 評価指標（深度 / フロー / 姿勢）

評価はモデルとは独立に、**合成 GT と既知の予測**で定義を検算するのが理解の近道です（`04_depth_flow_pose_metrics.py` はモデルを一切使いません）。

- **深度 AbsRel** = `mean(|pred-gt| / gt)`（相対誤差の平均・小さいほど良い）。**RMSE** = `sqrt(mean((pred-gt)^2))`。**δ<1.25** = 各画素で `max(pred/gt, gt/pred) < 1.25` となる割合（threshold accuracy・大きいほど良い）。相対深度は**スケール合わせ後**に測るのが鉄則。
- **フロー EPE(End-Point Error)** = `mean(||flow_pred - flow_gt||_2)`（各画素のベクトル差のユークリッド長の平均・px 単位）。
- **姿勢 OKS(Object Keypoint Similarity)** = `Σ exp(-d_i²/(2·s²·k_i²))·δ(v_i>0) / Σ δ(v_i>0)`、`k_i = 2σ_i`、`s² = area`。IoU の姿勢版で 1 が完全一致。**PCK@α** = `d_i <= α·ref_len` を満たす可視点の割合（PCK@0.2 が定番）。

<figure class="lec-fig"><svg viewBox="0 0 560 300" role="img" aria-label="フローのEPEは予測ベクトルと真ベクトルの差の長さ。端点間のユークリッド距離を全画素で平均する" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="280" y="28" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">EPE ＝ 予測フローと真フローの「ベクトル差の長さ」</text><line x1="120" y1="225" x2="290" y2="86" stroke="#16a34a" stroke-width="2.5"/><polygon points="290,86 279,89 285,96" fill="#16a34a"/><line x1="120" y1="225" x2="384" y2="178" stroke="#ea580c" stroke-width="2.5"/><polygon points="384,178 372,174 374,185" fill="#ea580c"/><line x1="290" y1="86" x2="384" y2="178" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/><line x1="337" y1="132" x2="430" y2="118" stroke="#d4d4d8" stroke-width="1.2"/><circle cx="120" cy="225" r="5" fill="#18181b"/><circle cx="290" cy="86" r="3.5" fill="#16a34a"/><circle cx="384" cy="178" r="3.5" fill="#ea580c"/><text x="110" y="246" text-anchor="middle" font-size="13" fill="#3f3f46">画素 p</text><text x="244" y="78" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">真フロー (u*, v*)</text><text x="398" y="178" font-size="13" font-weight="700" fill="#c2410c">予測フロー (u, v)</text><text x="436" y="114" font-size="13" font-weight="700" fill="#dc2626">EPE = ‖差‖₂</text></svg><figcaption>フローの <b>EPE（End-Point Error）</b>は、各画素で<b>予測ベクトルと真ベクトルの差の長さ</b>（ユークリッド距離 ‖f_pred − f_gt‖₂）を取り、全画素で平均した値（px 単位・小さいほど良い）。<b>u 成分・v 成分を別々に測るのではなく、2 次元ベクトルの差の長さ</b>である点に注意します。図の<b>赤い破線</b>が 1 画素ぶんの EPE です。</figcaption></figure>

OKS の per-keypoint 定数 σ は**小さい関節（目・鼻）ほど厳しく、大きい関節（腰）ほど緩い**よう COCO が定めています。そのため、同じ px ズレでも目の方が OKS を大きく下げます。これらの式は `exercises.py` で 1 問ずつ自力実装します。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

**統合課題.** 「奥行きのある背景の上を人型が右へ歩く」2フレームの合成クリップを作り、3タスクを1本のパイプラインに通して1枚の総合レポートにします。

- **ステージ1【深度】** 代表フレームを Depth Anything V2 で相対深度化 → カラーマップ可視化。
- **ステージ2【姿勢】** keypoint R-CNN で17点推定（未検出なら合成 GT 関節でスケルトン描画にフォールバック）→ GT に対する OKS。
- **ステージ3【フロー】** 2フレームを RAFT で密フロー化 → 矢印で重ね描き → **背景=0・人型領域=移動量** とした region-aware な合成 GT に対する EPE。

<figure class="lec-fig"><svg viewBox="0 0 640 320" role="img" aria-label="章末ミニプロジェクトのパイプライン。2フレーム合成クリップを入力に深度・姿勢・フローの3ステージを順に通し2×2モンタージュとJSONレポートにまとめる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="320" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">ミニプロジェクト — 1 本のパイプラインで 深度→姿勢→フロー</text><rect x="24" y="70" width="170" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="235" y="70" width="170" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="446" y="70" width="170" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="446" y="210" width="170" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="235" y="210" width="170" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="109" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">入力</text><text x="109" y="113" text-anchor="middle" font-size="10.5" fill="#52525b">2 フレームの</text><text x="109" y="129" text-anchor="middle" font-size="10.5" fill="#52525b">合成クリップ</text><text x="320" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">① 深度</text><text x="320" y="113" text-anchor="middle" font-size="10" fill="#52525b">Depth Anything V2</text><text x="320" y="129" text-anchor="middle" font-size="10.5" fill="#52525b">→ カラーマップ</text><text x="531" y="94" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">② 姿勢</text><text x="531" y="113" text-anchor="middle" font-size="10" fill="#52525b">keypoint R-CNN</text><text x="531" y="129" text-anchor="middle" font-size="10.5" fill="#52525b">→ OKS</text><text x="531" y="234" text-anchor="middle" font-size="13" font-weight="700" fill="#c2410c">③ フロー</text><text x="531" y="253" text-anchor="middle" font-size="10" fill="#52525b">RAFT (small)</text><text x="531" y="269" text-anchor="middle" font-size="10.5" fill="#52525b">→ EPE</text><text x="320" y="234" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">出力レポート</text><text x="320" y="253" text-anchor="middle" font-size="10.5" fill="#52525b">2×2 モンタージュ</text><text x="320" y="269" text-anchor="middle" font-size="10.5" fill="#52525b">+ JSON</text><line x1="196" y1="102" x2="229" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="235,102 225,97 225,107" fill="#71717a"/><line x1="407" y1="102" x2="440" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="446,102 436,97 436,107" fill="#71717a"/><line x1="531" y1="136" x2="531" y2="204" stroke="#71717a" stroke-width="2"/><polygon points="531,210 526,200 536,200" fill="#71717a"/><line x1="444" y1="242" x2="409" y2="242" stroke="#71717a" stroke-width="2"/><polygon points="405,242 415,237 415,247" fill="#71717a"/></svg><figcaption>章末ミニプロジェクトは <b>2 フレームの合成クリップ</b>を入口に、<b>① 深度</b>（Depth Anything V2）→ <b>② 姿勢</b>（keypoint R-CNN）→ <b>③ フロー</b>（RAFT small）の 3 ステージを<b>1 本のパイプライン</b>に順に通し、結果を最後に <code>2×2 モンタージュ</code>（入力/深度/姿勢/フロー）と <code>mini_report.json</code> にまとめます。どのモデルが落ちても止めず、できた範囲でレポートを出します（必ず exit 0）。</figcaption></figure>

出力は `mini_montage.png`（入力/深度/姿勢/フローの 2×2）と `mini_report.json`。どのモデルが落ちても止めず、できた範囲でレポートを出します（必ず exit 0）。発展として「人型を別物体に替える」「フレーム数を増やして時系列の OKS/EPE 推移を出す」「`data/` に実写を置いて keypoint R-CNN を発火させる」に挑戦してください。

---

## ✅ 到達チェックリスト

- [ ] `pipeline('depth-estimation')` で相対深度を取得し、出力が**逆深度（近い=大）・スケール不定**だと説明できる。
- [ ] 深度の**正規化（可視化用）**と**中央値スケール合わせ（評価用）**を区別して実装できる。
- [ ] keypoint R-CNN の出力 `(N,17,3)` を読み、COCO17 のスケルトンを描ける。
- [ ] 実写学習モデルが合成図形で**未検出になる domain gap** を説明できる。
- [ ] RAFT の **[-1,1]・8の倍数(最低128px)・最後の反復** という3前提を述べ、実装できる。
- [ ] 古典フロー（LK/Farneback）と RAFT を **EPE** で比較し、輝度一定仮定の限界を語れる。
- [ ] **AbsRel / RMSE / δ<1.25 / EPE / OKS / PCK** を numpy で実装し、検算できる（`exercises.py` 全 PASS）。
- [ ] mediapipe を**実行経路で使わず**ガード＋任意導入にする理由（numpy/protobuf 競合）を説明できる。

---

## ✍️ 演習問題

演習は `exercises.py` に TODO 形式で入っています。各 TODO を実装し `uv run python lectures/27_depth_pose_flow/exercises.py` を実行すると自己採点できます（`exercises_solutions.py` が解答）。

1. 深度マップを min-max で [0,1] に正規化する（`ex1_minmax_normalize` の TODO）。`max == min`（定数マップ）のときは 0 除算を避けて全て 0.0 を返す。
2. `value` 以上で `multiple`（既定 8）の倍数になる最小の整数を返す（`ex2_round_up_to_multiple` の TODO）。RAFT の「8 の倍数」サイズ要件のための切り上げで、既に倍数ならそのまま返す。
3. 深度の AbsRel = `mean(|pred-gt|/gt)` を計算する（`ex3_abs_rel` の TODO）。`gt` は `eps` で下限クリップしてから割る。
4. 深度の RMSE = `sqrt(mean((pred-gt)^2))` を計算する（`ex4_rmse` の TODO）。
5. 閾値正解率 δ<thr を返す（`ex5_delta_accuracy` の TODO）。各要素で `max(pred/gt, gt/pred) < thr` となる割合で、比をとる前に `eps` で下限クリップする。
6. 中央値スケール合わせ `pred*(median(gt)/median(pred))` で相対深度を GT のスケールへ寄せる（`ex6_align_scale_median` の TODO）。`median(pred)` が 0 付近なら `pred` をそのまま返す。
7. フローの平均端点誤差 EPE = `mean(sqrt(du^2+dv^2))` を返す（`ex7_endpoint_error` の TODO）。入力は最後の軸が `(u, v)` の形状。
8. 姿勢の OKS を計算する（`ex8_oks` の TODO）。per-keypoint の σ と物体スケール `area` で正規化した類似度で、可視点（`v>0`）だけを分母に数える（可視点が無ければ 0.0）。
9. 姿勢の PCK@α を返す（`ex9_pck` の TODO）。距離 `d_i` が `α*ref_len` 以内に収まった可視キーポイントの割合で、可視点が無ければ 0.0。

---

## ❓ 落とし穴・FAQ・デバッグ

- **Q. 深度の near/far が逆に見える.** A. `predicted_depth` は逆深度で「大きい=近い」。深度に戻すなら `1/(inv+eps)`、可視化の向きは `colorize_depth(..., invert=...)` で調整。
- **Q. AbsRel が異常に大きい.** A. 相対深度を**スケール合わせせず**に GT と比較している。中央値比 or 最小二乗の scale/shift を当てる。`04` の align 前後の差を参照。
- **Q. RAFT が `Feature maps are too small` で落ちる.** A. 入力が 128px 未満。`round_up_to_multiple(_, 8)` で 8 の倍数化し、最低 128px を確保。
- **Q. RAFT のフローがめちゃくちゃ.** A. `weights.transforms()` を通さず生 [0,255] を入れている／`flows`（リスト）の最後を取っていない。
- **Q. keypoint R-CNN が誰も検出しない.** A. 合成図形は学習分布外（domain gap）。`data/27_depth_pose_flow/` に実写人物を置く、または score 閾値を下げる。本講座は GT 関節フォールバックで学習体験を担保。
- **Q. mediapipe を入れたら numpy が壊れた.** A. 既知の競合。本講座の実行経路では使わない。試すなら**別 venv / 別グループ**で隔離。
- **Q. `torchvision.io.read_video` が無い.** A. 0.26 で廃止。動画 I/O は `cv2.VideoCapture` を使う。
- **Q. CPU で遅い.** A. Small/低解像度を使い、`model.eval()` + `torch.inference_mode()` + `torch.set_num_threads()` を徹底。初回はモデル DL 分だけ余計に時間がかかる。
- **デバッグの基本.** 形状（`(N,17,3)` / `(N,2,H,W)`）と値域（[-1,1] / [0,1] / 0-255）を `print` で常時確認。色は BGR(cv2) と RGB(torch/PIL/matplotlib) を取り違えない。

---

## 💡 実践ユースケース集

この章の「深度・姿勢・フロー」は、そのまま現実の小ツールになります。以下は**実応用の出発点**です（1 つ目は実際に動く `use_case.py` として同梱）。

### 1. ポートレート背景ぼかし（深度 bokeh）— 同梱 `use_case.py`

- **何に使うか**: スマホの「ポートレートモード」を 1 枚画像から再現。単眼深度で手前/奥を推定し、**ピント面から遠い画素ほど強くぼかす**ことで被写体を浮き立たせます。SNS アイコン・サムネ・商品写真の被写体強調に。
- **作り方の要点**: ① Depth Anything V2 (Small) で逆深度を推定（`out["predicted_depth"]` は入力と解像度が違うことがあるので**必ずリサイズ**）。② 逆深度を [0,1] 正規化し、分位点で「ピント面」を決める。③ ピント面から外れるほどボケ強度 α を 0→1 に上げ、**α を軽くぼかして輪郭のフェザリング**。④ `くっきり画像*(1-α) + 全面ぼかし*α` でアルファ合成。

<figure class="lec-fig"><svg viewBox="0 0 660 330" role="img" aria-label="深度ボケの4ステップ。逆深度推定からピント面決定、ボケ強度の算出とフェザリング、アルファ合成で背景をぼかす" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="32" text-anchor="middle" font-size="15" font-weight="700" fill="#18181b">深度ボケ（ポートレートモード）の 4 ステップ</text><rect x="24" y="70" width="176" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><rect x="242" y="70" width="176" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="460" y="70" width="176" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="460" y="210" width="176" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="242" y="210" width="176" height="64" rx="8" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/><rect x="24" y="210" width="176" height="64" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="112" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">入力</text><text x="112" y="117" text-anchor="middle" font-size="10.5" fill="#52525b">ポートレート画像</text><text x="330" y="92" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">① 逆深度推定</text><text x="330" y="110" text-anchor="middle" font-size="10" fill="#52525b">Depth Anything V2</text><text x="330" y="126" text-anchor="middle" font-size="10" fill="#52525b">（必ずリサイズ）</text><text x="548" y="92" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">② ピント面決定</text><text x="548" y="110" text-anchor="middle" font-size="10" fill="#52525b">[0,1] 正規化</text><text x="548" y="126" text-anchor="middle" font-size="10" fill="#52525b">→ 分位点で焦点</text><text x="548" y="232" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">③ ボケ強度 α</text><text x="548" y="250" text-anchor="middle" font-size="10" fill="#52525b">外れるほど 0→1</text><text x="548" y="266" text-anchor="middle" font-size="10" fill="#52525b">α をフェザリング</text><text x="330" y="232" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">④ アルファ合成</text><text x="330" y="250" text-anchor="middle" font-size="10" fill="#52525b">鮮明·(1-α)</text><text x="330" y="266" text-anchor="middle" font-size="10" fill="#52525b">+ ボケ·α</text><text x="112" y="238" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">出力</text><text x="112" y="257" text-anchor="middle" font-size="10.5" fill="#52525b">背景ぼかし画像</text><line x1="202" y1="102" x2="235" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="242,102 232,97 232,107" fill="#71717a"/><line x1="420" y1="102" x2="453" y2="102" stroke="#71717a" stroke-width="2"/><polygon points="460,102 450,97 450,107" fill="#71717a"/><line x1="548" y1="136" x2="548" y2="204" stroke="#71717a" stroke-width="2"/><polygon points="548,210 543,200 553,200" fill="#71717a"/><line x1="458" y1="242" x2="423" y2="242" stroke="#71717a" stroke-width="2"/><polygon points="418,242 428,237 428,247" fill="#71717a"/><line x1="240" y1="242" x2="205" y2="242" stroke="#71717a" stroke-width="2"/><polygon points="200,242 210,237 210,247" fill="#71717a"/></svg><figcaption>同梱 <code>use_case.py</code> の<b>深度ボケ</b>は、<b>① 逆深度推定</b>（Depth Anything V2・出力解像度が違うので<b>必ずリサイズ</b>）→ <b>② ピント面決定</b>（[0,1] 正規化し<b>分位点</b>で焦点面を選ぶ）→ <b>③ ボケ強度 α</b>（ピント面から外れる画素ほど α を 0→1 に上げ、α を軽くぼかして輪郭を<b>フェザリング</b>）→ <b>④ アルファ合成</b>（<code>鮮明·(1-α) + ぼかし·α</code>）の順に進みます。相対深度はスケール不定なので、焦点は「○m」ではなく<b>分位点</b>で決めるのが要点です。</figcaption></figure>

- **注意**: 相対深度はスケール不定なので「○m から先をぼかす」は不可、**分位点や正規化値で焦点を決める**。被写体の細い輪郭（髪・指）は深度の精度不足でボケが滲みやすい（フェザリングと後段マット処理で軽減）。点光源を“玉ボケ”にしたいなら Gaussian を**円盤カーネル**に替える。
- **実行 / データ / 拡張**:
  ```bash
  uv run python lectures/27_depth_pose_flow/use_case.py
  ```
  `data/27_depth_pose_flow/` に画像（`*.png/*.jpg`）を置くと先頭を自動使用（**人物が手前のポートレート推奨**）。無ければ合成室内シーンで動作（重み DL 失敗時も GT/ランプ深度でフォールバックし必ず exit 0）。出力は `lectures/27_depth_pose_flow/outputs/use_case_bokeh.png`（成果物）と `use_case_montage.png` / `use_case_bokeh.json`。**拡張**: `focus_pct` で焦点面を手前/奥に移動、`band` で被写界深度の薄さを調整、円盤カーネルで玉ボケ、ボケ量を多段量子化したレイヤ DoF、`data/` 全画像のバッチ加工 CLI 化、Tkinter スライダ対話（headless はファイル保存にフォールバック）。

### 2. 転倒・姿勢アラート（pose ベースの見守り）

- **何に使うか**: 介護・独居見守りや工場の安全監視で、人体キーポイントから「**転倒・しゃがみ込み・不自然な姿勢**」を検知して通知する。
- **作り方の要点**: keypoint R-CNN（または別環境の MediaPipe Pose）で 17/33 点を取り、**肩-腰の縦距離 vs 横距離の比**や胴体の傾き角、頭の高さの急落をフレーム間で監視。閾値を数フレーム連続で超えたらアラート（チャタリング防止）。
- **注意**: 実写学習モデルは**合成図形では発火しない**（本章で体験する domain gap）。現場カメラの画角・照明で必ず実データ検証を。トップダウン型は人数に比例して重いので、CPU なら低解像度＋フレームスキップ、プライバシー配慮（顔ぼかし・端末内処理）も設計に含める。

### 3. 動きヒートマップ / 群衆フロー可視化（optical flow ベース）

- **何に使うか**: 店舗・イベント・交通の動画から「**どこがよく動くか／人の流れの向き**」を可視化し、混雑・動線分析やレジ前の滞留検知に使う。
- **作り方の要点**: 連続フレームに Farneback（密・CPU 軽量）または RAFT(small) を適用し、フロー大きさを時間方向に累積して**動きヒートマップ**、フロー向きを色相にマップして**流れの方向場**を描く。背景差分と組み合わせると静止背景を除ける。
- **注意**: 古典フローは**輝度一定仮定**のため照明変化・影・無テクスチャ面で破綻しやすい。RAFT は頑健だが CPU では重く、**低解像度・8 の倍数サイズ・最低 128px・`flows[-1]` 採用**を守る。カメラが動くと全画素が流れるので、固定カメラ前提か別途カメラ運動補償が要る。

---

## 🚀 発展トピック・参考

- **Metric Depth**: 絶対距離が要るなら Depth Anything V2 Metric / ZoeDepth など。SLAM・占有格子・測距へ。
- **姿勢の高精度化**: HRNet・ViTPose、3D 姿勢（SMPL/メッシュ回帰）、多人数のボトムアップ型(OpenPose 系)。
- **フローの応用**: フレーム補間(RIFE)、動画安定化、追跡(第28回)・行動認識(第29回)の前段特徴。
- **任意導入（実行経路では未使用・ガード）**: `mediapipe`（Pose/Hand/Face Landmarker, CPU 実時間だが numpy/protobuf 競合に注意, `uv add --group pose mediapipe`）。
- **公式ドキュメント**: Depth Anything V2 (HF transformers model_doc) / torchvision optical_flow・detection / COCO keypoints 評価(OKS)。

---

## ▶ 動かし方

```bash
# 依存（CPU）: dl(torch/torchvision) + hf(transformers) + metrics は導入済み想定
uv sync --group dl --group hf --group metrics

# 各スクリプト（初回はモデル重みを自動 DL。以降はキャッシュ）
uv run python lectures/27_depth_pose_flow/01_depth_anything.py
uv run python lectures/27_depth_pose_flow/02_pose_keypoints.py
uv run python lectures/27_depth_pose_flow/03_raft_optical_flow.py
uv run python lectures/27_depth_pose_flow/04_depth_flow_pose_metrics.py
uv run python lectures/27_depth_pose_flow/mini_project.py

# 実践ユースケース: ポートレート背景ぼかし（深度 bokeh）。data/ に人物写真を置くと効果大
uv run python lectures/27_depth_pose_flow/use_case.py

# 演習（自己採点。TODO 未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/27_depth_pose_flow/exercises.py
uv run python lectures/27_depth_pose_flow/exercises_solutions.py
```

出力はすべて `lectures/27_depth_pose_flow/outputs/` に保存されます（`*.png` 可視化と `*.json` メトリクス）。実画像で試すときは `data/27_depth_pose_flow/` に画像を置くと自動で優先されます。なお `dpf_helpers.py` は合成データ生成・可視化・評価指標の共有ヘルパで、上記の各スクリプトが `import dpf_helpers as H` で利用します（単体実行はしません）。

---

> 参照ライブラリ＋版: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11**（opencv-headless 4.13・numpy 2.x）、2026-06 時点。
> 本章は headless(`cv2.imshow` 非使用)・CPU 前提・`model.eval()`+`torch.inference_mode()` で全スクリプトが exit 0 で完走します。