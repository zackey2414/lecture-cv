# 00_setup 参考 Dockerfile — 本講座を CPU・headless で再現するための最小構成。
#
# このファイルは「読んで理解する教材」であり、そのままビルドもできる参考実装。
# ねらい:
#   - python:3.12-slim をベースに、OpenCV(headless) と動画 I/O に必要な OS ライブラリだけ入れる
#   - uv で依存を再現インストール（uv.lock があれば --frozen で完全再現）
#   - PyTorch は CPU ホイール（pyproject の [[tool.uv.index]] pytorch-cpu）が引かれる
#   - HuggingFace の重みは HF_HOME=/cache/hf に集約し、ボリュームで永続化する
#
# ビルド例（プロジェクトルートで）:
#   docker build -f lectures/00_setup/Dockerfile -t lecture-cv:cpu .
# 実行例（HF キャッシュを永続化してネット再 DL を避ける）:
#   docker run --rm -v $PWD:/app -v hfcache:/cache/hf lecture-cv:cpu \
#     uv run python lectures/00_setup/mini_project.py

FROM python:3.12-slim

# OpenCV(headless) は GUI(Qt/X11) を要求しないが、libGL と glib は動的リンクで必要。
# ffmpeg は動画 I/O（cv2.VideoCapture）用。--no-install-recommends で肥大化を防ぐ。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv 本体（公式配布の静的バイナリをコピー）。pip より速く、lock で完全再現できる。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HuggingFace キャッシュを固定パスへ集約 → このパスをボリュームにすると再 DL を防げる。
ENV HF_HOME=/cache/hf \
    # オフライン/CI で動かすなら 1 にする（キャッシュ済みの重みだけを使う）。
    HF_HUB_OFFLINE=0 \
    # CPU スレッドの暴走（オーバーサブスクライブ）を抑える既定。実機コア数に合わせて上書きする。
    OMP_NUM_THREADS=4 \
    # 仮想環境を使わずシステムに入れる（コンテナでは venv 不要）。
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# 依存定義だけ先にコピーしてレイヤキャッシュを効かせる（ソース変更で再インストールしない）。
COPY pyproject.toml uv.lock* ./
# 深層の土台(dl)まで入れる。lock があれば --frozen で完全再現、無ければ通常解決。
RUN uv sync --group dl --no-dev || uv sync --group dl

# 残りのソースを配置。
COPY . .

# 既定コマンド: 環境まるごと検証ミニプロジェクトを走らせて健全性を確認する。
CMD ["uv", "run", "python", "lectures/00_setup/mini_project.py"]
