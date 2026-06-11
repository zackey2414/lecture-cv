# 27_depth_pose_flow: 単眼深度・姿勢/キーポイント・オプティカルフロー

> トラック: **深度・姿勢・動き** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf` `metrics`（`pose` は概念のみ・任意）
>
> 参照ライブラリ: torch **2.12+cpu** / torchvision **0.27+cpu** / transformers **5.11** / opencv(headless) **4.13** / numpy **2.x**（2026-06 時点）

---

## 🎯 この章のゴール

1枚の画像、1人の人物、2枚の連続フレーム。これらから「**奥行き(depth)**」「**関節(pose/keypoints)**」「**動き(optical flow)**」という、シーンの3次元・運動構造に踏み込んだ情報を取り出せるようになることがゴールです。具体的には次の4点を、AI 補助なしで自力で書けるところまで持っていきます。

- **単眼深度**: `transformers.pipeline('depth-estimation')` で Depth Anything V2 (Small) を動かし、出力が **相対(逆)深度**（値が大きいほど近い・絶対距離ではない）であることを理解し、可視化のための正規化と、評価のためのスケール合わせを行える。
- **姿勢/キーポイント**: torchvision `keypointrcnn_resnet50_fpn` で **COCO 17 点**を推定し、スケルトン（辺）の張り方と、検出器が「人を検出 → 関節を回帰」する2段構えを理解する。あわせて**合成図形では検出が発火しない domain gap** を体験する。
- **オプティカルフロー**: torchvision `raft_small` で密フローを推定し、RAFT 3 大前処理（**[-1,1] 正規化・8 の倍数サイズ・反復出力の最後を採用**）を踏める。古典法（Lucas-Kanade / Farneback）と対比し、輝度一定仮定の限界を語れる。
- **評価**: 深度 **AbsRel / RMSE / δ<1.25**、フロー **EPE**、姿勢 **OKS / PCK@0.2** を、合成 GT に対して numpy で実装・検算できる。

入力はすべて**合成生成**（床・壁・箱・球の擬似室内、並進するテクスチャ画像対、腕を振る人型図形）です。ネットに出るのは**モデル重みのダウンロードだけ**。`data/27_depth_pose_flow/` に実画像を置けばそちらを優先します。

---

## 本編

### 0. 全体像 — なぜこの3つを1章にまとめるのか

深度・姿勢・フローは、一見バラバラなタスクに見えますが、いずれも「**画素そのもの**」ではなく「**画素が表す幾何・運動の量**」を回帰する密予測（dense/structured prediction）という共通の骨格を持ちます。分類や検出が「離散ラベル」を出すのに対し、ここで扱うのは深度値・関節座標・フローベクトルといった**連続量**で、評価も accuracy ではなく「真値との連続的なズレ」を測ります。だからこそ AbsRel・EPE・OKS のような**回帰系の指標**が主役になります。

また3タスクは応用上もよく一緒に使われます。AR・ロボティクス・スポーツ解析では「人がどこにいて(pose)、どれだけ手前にいて(depth)、次の瞬間どこへ動くか(flow)」を同時に欲しがります。本章の `mini_project.py` は、1本の合成クリップに深度→姿勢→フローを順に適用して1枚のレポートにまとめる、まさにこの統合を実演します。

技術的な共通の注意点もあります。いずれも**学習時の前処理を一字一句守らないと無意味な出力になる**こと（深度の正規化、フローの [-1,1]・8の倍数、姿勢の正規化）、そして**CPU では小モデル＋低解像度＋ `inference_mode()`** を徹底しないと現実的な速度にならないこと。この章は「前処理の勘所」を体に叩き込む章でもあります。

### 1. 単眼深度推定（Depth Anything V2）

**直感.** 人間は片目でも「手前/奥」が分かります。テクスチャの密度、ものの大きさ、遮蔽関係、陰影といった手がかりから脳が奥行きを推定しているからです。単眼深度モデルは、大量の画像とその深度の組から、この「1枚→奥行き」の写像を学んだものです。Depth Anything V2 は大規模な擬似ラベルで学習された強力な相対深度モデルで、Small 版なら CPU でも数秒で動きます。

**理論.** ここで決定的に重要なのは、単眼相対深度モデルの出力が**絶対距離ではない**ことです。Depth Anything が返す `predicted_depth` は**逆深度（disparity 的な量）**で、慣習として「**値が大きいほどカメラに近い**」という向きを持ちます。スケールとシフトも不定（affine-invariant）なので、「3.0 という値が 3 メートル」を意味しません。だから可視化の前には必ず正規化（min-max や percentile）が要りますし、GT と数値比較するなら**スケール合わせ**（中央値比や最小二乗での scale/shift フィット）が前提になります。メートル単位が欲しければ Metric 系のモデルを別途使います。

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

**落とし穴.** ①逆深度の向きを取り違えて near/far が反転する。②スケール合わせを忘れて AbsRel が極端に大きく出る（`04` の `abs_rel_before_align` と `abs_rel` の差で体感できます）。③`out["depth"]`（可視化済み PIL）を生の深度値と勘違いして指標計算に使う。

**実務の使い分け.** 相対深度で十分な用途（ボケ・背景合成・相対的な遠近の把握・3D風エフェクト）には Depth Anything V2 が最有力。測距・SLAM・占有格子など**メートルが要る**用途には Metric Depth 系か、ステレオ/LiDAR/ToF の併用を検討します。

### 2. 姿勢・キーポイント推定（keypoint R-CNN, COCO17）

**直感.** 姿勢推定は「人の体のどこに鼻・肩・肘・手首…があるか」を点で当てるタスクです。点同士を**スケルトン（辺）**で結ぶと棒人間になり、動作・ジェスチャ・転倒検知などの土台になります。COCO の標準は**17 点**（顔5＋上半身6＋下半身6）です。

**理論.** keypoint R-CNN は2段構えです。まず人を**検出**（bbox とスコア）し、各人の RoI から**17 関節を回帰**します。出力 `keypoints` は `(人数, 17, 3)` で、最後の 3 は `(x, y, score)`。COCO の点には**左右の区別**（その人にとっての左肩/右肩）があり、辺の張り方（肩-肘-手首、肩-腰、腰-膝-足首）が固定されています。トップダウン型（検出→関節）なので、**人が検出できなければ関節も出ません**——ここが本章最大の落とし穴に直結します。

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

### 3. オプティカルフロー（RAFT vs 古典）

**直感.** オプティカルフローは「連続2フレーム間で、各画素がどこへ動いたか」のベクトル場 `(u, v)` です。動画の動き解析・フレーム補間・追跡・行動認識の前段になります。

**理論.** 古典法は**輝度一定仮定**（同じ点はフレーム間で明るさを保つ）と**滑らかさ**に基づきます。ここから出る基本拘束は1本の式に未知数2つ（u, v）で**開口問題(aperture problem)**——エッジに沿った成分が決まらない——を生むので、近傍をまとめる（Lucas-Kanade: 疎、特徴点ごと）か全画素を平滑化する（Farneback: 密）かで解きます。深層の RAFT は**全ペア相関ボリューム + 反復更新(GRU)**で、大変位・無テクスチャ・遮蔽に強い密フローを出します。

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

### 4. 評価指標（深度 / フロー / 姿勢）

評価はモデルとは独立に、**合成 GT と既知の予測**で定義を検算するのが理解の近道です（`04_depth_flow_pose_metrics.py` はモデルを一切使いません）。

- **深度 AbsRel** = `mean(|pred-gt| / gt)`（相対誤差の平均・小さいほど良い）。**RMSE** = `sqrt(mean((pred-gt)^2))`。**δ<1.25** = 各画素で `max(pred/gt, gt/pred) < 1.25` となる割合（threshold accuracy・大きいほど良い）。相対深度は**スケール合わせ後**に測るのが鉄則。
- **フロー EPE(End-Point Error)** = `mean(||flow_pred - flow_gt||_2)`（各画素のベクトル差のユークリッド長の平均・px 単位）。
- **姿勢 OKS(Object Keypoint Similarity)** = `Σ exp(-d_i²/(2·s²·k_i²))·δ(v_i>0) / Σ δ(v_i>0)`、`k_i = 2σ_i`、`s² = area`。IoU の姿勢版で 1 が完全一致。**PCK@α** = `d_i <= α·ref_len` を満たす可視点の割合（PCK@0.2 が定番）。

OKS の per-keypoint 定数 σ は**小さい関節（目・鼻）ほど厳しく、大きい関節（腰）ほど緩い**よう COCO が定めています。同じ px ズレでも目の方が OKS を大きく下げます。これらの式は `exercises.py` で 1 問ずつ自力実装します。

---

## 🛠 章末ミニプロジェクト（`mini_project.py`）

**統合課題.** 「奥行きのある背景の上を人型が右へ歩く」2フレームの合成クリップを作り、3タスクを1本のパイプラインに通して1枚の総合レポートにします。

- **ステージ1【深度】** 代表フレームを Depth Anything V2 で相対深度化 → カラーマップ可視化。
- **ステージ2【姿勢】** keypoint R-CNN で17点推定（未検出なら合成 GT 関節でスケルトン描画にフォールバック）→ GT に対する OKS。
- **ステージ3【フロー】** 2フレームを RAFT で密フロー化 → 矢印で重ね描き → **背景=0・人型領域=移動量** とした region-aware な合成 GT に対する EPE。

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

# 演習（自己採点。TODO 未実装でも exit 0）と模範解答（全 PASS）
uv run python lectures/27_depth_pose_flow/exercises.py
uv run python lectures/27_depth_pose_flow/exercises_solutions.py
```

出力はすべて `outputs/27_depth_pose_flow/` に保存されます（`*.png` 可視化と `*.json` メトリクス）。実画像で試すときは `data/27_depth_pose_flow/` に画像を置くと自動で優先されます。

---

> 参照ライブラリ＋版: **torch 2.12+cpu / torchvision 0.27+cpu / transformers 5.11**（opencv-headless 4.13・numpy 2.x）、2026-06 時点。
> 本章は headless(`cv2.imshow` 非使用)・CPU 前提・`model.eval()`+`torch.inference_mode()` で全スクリプトが exit 0 で完走します。