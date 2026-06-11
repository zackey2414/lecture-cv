# CPU 既定（GPU不要・MacBook等でも動かせる構成）。
# GPUを使う場合は下のコメントを参照して nvidia/cuda ベースに差し替える。
FROM python:3.12-slim
WORKDIR /app

# OpenCV(cv2) の実行に必要な共有ライブラリと、動画処理用の ffmpeg を入れる。
#   libgl1, libglib2.0-0 : import cv2 時の libGL.so.1 / libgthread 不足を解消
#   ffmpeg               : 動画コーデック / VideoCapture, VideoWriter
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv をインストール（公式イメージからバイナリをコピー）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 依存定義だけ先にコピー（レイヤキャッシュを効かせる）。uv.lock はあれば使う。
COPY pyproject.toml .python-version ./
COPY uv.loc[k] ./

# プロジェクト内の仮想環境パスを優先
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# HuggingFace のモデルキャッシュをコンテナ内の固定パスへ
ENV HF_HOME=/app/.cache/huggingface

# 学習用なので主要トラックの依存(dl/hf/vector/metrics/aug)＋開発ツール(dev=既定)まで入れる。
# torch は Linux では CPU ホイール(pyproject の [tool.uv.sources])を取得する。
# 後半の専用トラック(face/anomaly/onnx 等)は、その回に到達してから個別に追加する。
RUN uv sync --group dl --group hf --group vector --group metrics --group aug

# ソース一式をコピー（compose でマウントするなら必須ではない）
COPY . .

# 学習中はコンテナを起こしておき、`docker compose exec` で uv run する運用。
CMD ["sleep", "infinity"]

# ===== GPU を使う場合の参考 =====
# 1) 先頭の FROM を CUDA ベースに変更:
#      FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
#    （python / pip を別途入れるか、uv の管理 python を利用）
# 2) pyproject の torch インデックスを CUDA 版へ、faiss は `--extra gpu` を使う。
# 3) docker-compose.yaml の deploy.resources(nvidia) のコメントを外す。
