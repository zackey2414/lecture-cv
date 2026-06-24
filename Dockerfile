# CPU 既定（GPU不要・MacBook等でも動かせる構成）。
# GPUを使う場合は末尾のコメントを参照して nvidia/cuda ベースに差し替える。
#
# 【方針】このイメージは「箱（OS + システムライブラリ + Python + uv）」だけを用意する。
# Python パッケージ（numpy / torch / opencv …）はイメージに焼き込まない。
# コンテナに入ってから uv で各回の環境を整える運用にする:
#     docker compose up -d --build
#     docker compose exec lecture-cv bash
#     （コンテナ内）uv sync            # or: uv sync --group dl --group hf ...
#     （コンテナ内）uv run python lectures/<id>/...
# 詳しい考え方は docs/docker-basics.md、手順は docs/getting-started.md を参照。
FROM python:3.12-slim
WORKDIR /app

# OpenCV(cv2) の実行に必要な共有ライブラリと、動画処理用の ffmpeg を入れる。
#   libgl1, libglib2.0-0 : import cv2 時の libGL.so.1 / libgthread 不足を解消
#   ffmpeg               : 動画コーデック / VideoCapture, VideoWriter
#   git, curl            : 一部パッケージの取得・補助ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv をインストール（公式イメージからバイナリをコピー）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# プロジェクト内の仮想環境(.venv)を使う。uv のキャッシュとリンク方式も固定する。
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# uv のダウンロードキャッシュ（compose でボリューム化 → 2 回目以降の uv sync が速い）
ENV UV_CACHE_DIR=/app/.cache/uv
# bind マウントを跨いだ hardlink 警告を避け、コピーで配置する
ENV UV_LINK_MODE=copy
# ベースイメージの Python 3.12 を使う（uv 管理 Python の余計なダウンロードを避ける）
ENV UV_PYTHON_PREFERENCE=only-system
# HuggingFace のモデルキャッシュをコンテナ内の固定パスへ（compose でボリューム化）
ENV HF_HOME=/app/.cache/huggingface

# ソース一式をコピー（compose で lectures/ などをマウントすれば編集が即反映される）。
# 依存(.venv)は焼き込まない：コンテナ内で uv sync して整える方針のため。
# .dockerignore により .venv/ .cache/ data/ outputs/ などは除外される。
COPY . .

# 学習中はコンテナを起こしておき、`docker compose exec lecture-cv bash` で入って
# `uv sync` → `uv run python ...` する運用。
CMD ["sleep", "infinity"]

# ===== GPU を使う場合の参考 =====
# 1) 先頭の FROM を CUDA ベースに変更:
#      FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
#    （python / pip を別途入れるか、uv の管理 python を利用）
# 2) pyproject の torch インデックスを CUDA 版へ、faiss は vector グループの faiss-cpu を
#    GPU 版（faiss-gpu-cuvs 等）に差し替える。
# 3) docker-compose.yaml の deploy.resources(nvidia) のコメントを外す。
