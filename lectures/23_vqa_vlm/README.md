# 23_vqa_vlm: VQAと軽量VLMによる画像理解・グラウンディング

> トラック: **マルチモーダル** ／ レベル: **中級** ／ 必要な依存グループ: `dl` `hf`

## 🎯 この章のゴール
チャット形式VLMのapply_chat_template→generate→batch_decodeの正準フローを身につけ、CPUで現実的なmoondream2(caption/query/detect/point)やSmolVLM2-256M/500MでVQA・グラウンディングを実行でき、VQA accuracyで評価し、大型Qwen2.5-VLは概念のみ整理する。

## 扱うトピック
- AutoModelForImageTextToTextとapply_chat_template(画像をcontentに埋め込む)
- moondream2(trust_remote_code/revision固定、caption/query/detect/point)
- SmolVLM2-256M/500MでのVQA、generate(max_new_tokens)
- Qwen2.5-VL(process_vision_info)は概念のみ
- BLIP-2/GITとの位置づけ整理
- VQA accuracy=min(#agree/3,1)の評価

## 主要API
`AutoModelForImageTextToText` / `processor.apply_chat_template` / `model.generate` / `processor.batch_decode` / `vikhyatk/moondream2` / `HuggingFaceTB/SmolVLM2-256M-Instruct`

## 評価方法
VQAの正答を、VQA v2方式のaccuracy=min(一致した人間回答数/3,1)を10人サブセットで平均して評価(小データで近似)し、exact-match/正規化一致も併用。グラウンディングはmoondreamのdetect/point出力とGTのIoU/距離で確認する。

## 完成物
画像について質問応答・物体ポインティングを行うVLMチャットスクリプトと、小VQAセットでaccuracyを出す評価コード。

## CPU / GPU メモ
CPUはmoondream2(~2B)/SmolVLM2-256M・500M、Qwen2.5-VL-7Bはメモリ十数GB・数分でGPU推奨と明記。moondreamはtrust_remote_code=True+revision固定、einops/timm依存。

## 予定スクリプト
- `01_moondream_vqa.py`
- `02_smolvlm_chat.py`
- `03_vqa_accuracy.py`

---
> ⚠️ この回はロードマップ上の**プレースホルダ**です。教材本体（解説＋実行コード＋演習）は順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl` `hf`）
