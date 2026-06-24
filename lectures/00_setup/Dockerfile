# 00_setup 参考 Dockerfile — 本講座を CPU・headless で再現するための最小構成。
#
# このファイルは「読んで理解する教材」であり、そのままビルドもできる参考実装。
# 方針はリポジトリルートの Dockerfile と同じ:
#   - python:3.12-slim をベースに、OpenCV(headless)/動画 I/O に要る OS ライブラリ＋uv だけ入れた「箱」を作る
#   - Python 依存（torch など）はイメージに焼き込まない。コンテナに入ってから uv sync で整える
#   - HuggingFace と uv のキャッシュは固定パスへ集約し、ボリュームで永続化する（再 DL 回避・再 sync 高速化）
#
# ビルド例（プロジェクトルートで）:
#   docker build -f lectures/00_setup/Dockerfile -t lecture-cv:cpu .
# 使い方（箱を起こして入り、中で uv sync して実行）:
#   docker run -d --name lecture-cv -v "$PWD:/app" lecture-cv:cpu
#   docker exec -it lecture-cv bash
#   （コンテナ内）uv sync && uv run python lectures/00_setup/mini_project.py
# 通常はリポジトリ同梱の docker-compose.yaml を使う（手順は docs/docker-basics.md / docs/getting-started.md）。

FROM python:3.12-slim
WORKDIR /app

# OpenCV(headless) は GUI(Qt/X11) を要求しないが、libGL と glib は動的リンクで必要。
# ffmpeg は動画 I/O（cv2.VideoCapture）用。git/curl は補助。--no-install-recommends で肥大化を防ぐ。
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv 本体（公式配布の静的バイナリをコピー）。pip より速く、lock で完全再現できる。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# プロジェクト内の .venv を使う。uv のキャッシュ/リンク方式・使う Python・HF キャッシュを固定。
#   UV_PYTHON_PREFERENCE=only-system : ベースイメージの Python 3.12 を使い、余計な DL を避ける
#   UV_LINK_MODE=copy                : bind マウント跨ぎの hardlink 警告を避ける
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    UV_CACHE_DIR=/app/.cache/uv \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    HF_HOME=/app/.cache/huggingface

# ソース一式を配置（依存 .venv は焼き込まない：コンテナ内で uv sync して整える）。
COPY . .

# 学習中はコンテナを起こしておき、`docker exec -it lecture-cv bash` で入って
# `uv sync` → `uv run python ...` する運用（依存はコンテナ内で uv が整える）。
CMD ["sleep", "infinity"]
