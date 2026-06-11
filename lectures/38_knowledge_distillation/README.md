# 36_knowledge_distillation: 知識蒸留の基礎(a) — 温度付きKD・特徴量蒸留(CPUトイ学習)

> トラック: **最適化・デプロイ** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `distill`

## 🎯 この章のゴール
teacher(凍結)→studentへ知識を移す蒸留の枠組みを理解し、温度TでソフトターゲットのKLにT^2スケールを掛けハードラベルCEとalpha結合するHinton蒸留、forward hookで中間特徴をMSEで合わせる特徴量蒸留(FitNets、射影層必須)を、CPUでMNIST/CIFAR-10の小モデルに対し学習ループを自分で書いて完結できる。

## 扱うトピック
- 蒸留の4本柱俯瞰(蒸留/量子化/プルーニング/低ランク)とteacher凍結
- ソフトターゲットと温度付きKLダイバージェンス(F.kl_div, reduction='batchmean', T^2スケール)
- ハードラベルCEとのalpha結合、レスポンス/特徴/関係ベース
- CPUトイKD(MNIST/CIFAR-10で素のstudent vs 蒸留studentの精度差)
- 特徴量蒸留(register_forward_hook+射影層nn.Linear/1x1conv+MSE+正規化)
- 学習ループ(forward→loss→backward→step)とevalループ

## 主要API
`F.log_softmax` / `F.softmax` / `F.kl_div` / `nn.KLDivLoss(reduction='batchmean')` / `F.cross_entropy` / `module.register_forward_hook` / `nn.MSELoss` / `torch.optim.AdamW`

## 評価方法
蒸留の効果を、同一studentアーキでの『素の教師あり学習』と『KD学習』のテストaccuracyを比較して定量化(暗黙知の移転)する。温度T・alphaを変えてaccuracy変化を観察し、特徴量蒸留追加による改善も測る。

## 完成物
小CNN teacher→より小さいstudentを温度付きKD+特徴蒸留でCPU学習し、素のstudentとのaccuracy差を出すトイ蒸留スクリプト一式。

## CPU / GPU メモ
MNIST/CIFAR級・小バッチ・少エポックでCPU完結。KLは入力=log_softmax(student)/ターゲット=softmax(teacher)の順序とreduction='batchmean'、T^2スケールを厳守。teacherはeval()+no_grad()。

## 予定スクリプト
- `01_hinton_kd_mnist.py`
- `02_feature_distill_fitnets.py`
- `03_student_vs_baseline.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `distill`）
