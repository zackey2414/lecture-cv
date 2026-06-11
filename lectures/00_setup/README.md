# 00_setup: 環境構築 — uv + Docker + CPU版PyTorch + HuggingFaceキャッシュ + device判定

> トラック: **環境構築** ／ レベル: **入門** ／ 必要な依存グループ: `dl`

## 🎯 この章のゴール
uvの依存グループとDocker(python:3.12-slim + libgl1/ffmpeg)で本講座の実行環境を再現でき、Linuxで巨大なCUDA版torchを避けてCPUホイールを入れる方法、HF_HOMEキャッシュの永続化、cpu/mps/cudaを自動判定するtorch.deviceの定石を全回で再利用できる形で確立する。

## 扱うトピック
- uv add --group の依存グループ運用とpyproject.toml/uv.lock
- PyTorch CPUインデックス([[tool.uv.index]] explicit=true + [tool.uv.sources])とmacのMPS
- torch.device自動判定とPYTORCH_ENABLE_MPS_FALLBACK
- HuggingFaceキャッシュ(HF_HOME)とDockerボリュームマウント、HF_HUB_OFFLINE
- opencv-python と opencv-python-headless の排他、headless運用の方針
- torch.set_num_threads / OMP_NUM_THREADS によるCPUスレッド調整

## 主要API
`torch.cuda.is_available` / `torch.backends.mps.is_available` / `torch.device` / `torch.set_num_threads` / `HF_HOME` / `uv add --group` / `[[tool.uv.index]]`

## 評価方法
—

## 完成物
device.py(cpu/mps/cuda判定とスレッド数設定を返す共通ユーティリティ)、CPU専用torchが入ったpyproject.toml/Dockerfile、import確認スクリプト一式。

## CPU / GPU メモ
既定でCPU。Linuxは https://download.pytorch.org/whl/cpu を明示しCUDA版を回避、macはuv add torchでCPU/MPS版が入る。GPUがある場合のみindexをcu126等へ差し替える注記を添える。

## 予定スクリプト
- `00_check_env.py`
- `01_device_util.py`
- `02_hf_cache_demo.py`

## いま使えるもの: 環境スモークテスト

教材本体は順次作成しますが、環境確認用の `check_env.py` は先行して用意しています。

```bash
uv run python lectures/00_setup/check_env.py
```

必須ライブラリ（numpy/opencv/pillow/matplotlib）と任意ライブラリ（torch/transformers/faiss）の導入状況・版・PyTorch のデバイス（cpu/mps/cuda）を表示します。

---
> ⚠️ この回の**教材本体**（device.py 等の解説＋演習）はロードマップ上のプレースホルダです。順次作成します。

> 依存追加の例: `uv add --group dl <packages>`（必要グループ: `dl`）
