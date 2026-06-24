# Docker 入門 — この講座で Docker は何をして、uv と何が違うのか

> このページは **Docker をまったく触ったことがない人向け**に、「Docker が何をしているのか」「uv と役割がどう違うのか」「`Dockerfile` などのファイルがどう関係するのか」を、本講座の使い方に直結する形で説明します。難しいネットワークやオーケストレーションの話はしません。ゴールは **「`docker compose up` してコンテナに入り、`uv` で環境を整えて各回を動かす」までを、意味を理解したうえで自分でできる**ことです。

---

## 1. そもそも Docker は何の問題を解決する？

機械学習・CV のプロジェクトで誰もが踏むのが **「自分の PC では動くのに、別の環境では動かない」** です。OS が違う、システムライブラリが無い、Python のバージョンが違う——こうした差で、コードは正しいのに動かない、ということが起きます。

**Docker は「どの母艦（Mac / Windows / Linux）でも同じ Linux 環境を丸ごと再現する箱」を作る道具**です。箱の中身（OS・ライブラリ・Python）は常に同じなので、「環境が違うから動かない」が消えます。本講座でとくに効くのは、**Intel Mac ではネイティブに深層トラックの PyTorch を入れられない**（PyTorch が torch 2.3 以降の Intel Mac 向け配布を終了）という制約を、**コンテナの中は Linux なので回避できる**点です。だから本講座は「環境で迷ったら Docker」を推奨しています。

---

## 2. Docker と uv の責務 — 「箱」と「中身」

ここが最初の関門です。**Docker と uv は守備範囲がはっきり分かれています**。

| 担当すること | Docker | uv |
| --- | :---: | :---: |
| Linux OS そのもの（どの母艦でも同じ土台） | ✅ | |
| システムライブラリ（`libgl1` / `ffmpeg` など C のライブラリ） | ✅ | |
| Python 本体（3.12） | ✅（ベースイメージ） | |
| uv 自身（バイナリ） | ✅（イメージに入れる） | |
| **Python パッケージ（numpy / opencv / torch …）** | | ✅（`.venv` に入れる） |
| パッケージのバージョン固定（再現性） | | ✅（`uv.lock`） |
| ホスト OS からの隔離 | ✅ | |

ひとことで言えば：

- **Docker = 「マシン（箱）」を用意する係。** どの PC でも同じ Linux 環境を再現する。中身は OS・システムライブラリ・Python 本体・uv まで。
- **uv = 「その箱の中の Python パッケージ」を管理する係。** `pyproject.toml` / `uv.lock` を読んで、numpy・torch などを正確なバージョンで `.venv` に入れる。

<figure class="lec-fig"><svg viewBox="0 0 660 270" role="img" aria-label="Dockerとuvの責務分担。DockerはOS・システムライブラリ・Python本体・uvまでの箱を用意し、uvはその中にnumpyやtorchなどのPythonパッケージを.venvへ入れる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="330" y="26" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">守備範囲は「箱」と「中身」で分かれる</text><rect x="28" y="44" width="300" height="200" rx="12" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/><text x="178" y="72" text-anchor="middle" font-size="15" font-weight="800" fill="#1d4ed8">Docker ＝ 箱（マシン）を用意</text><rect x="50" y="86" width="256" height="30" rx="6" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/><text x="178" y="106" text-anchor="middle" font-size="12.5" fill="#18181b">Linux OS</text><rect x="50" y="122" width="256" height="30" rx="6" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/><text x="178" y="142" text-anchor="middle" font-size="12.5" fill="#18181b">システムライブラリ libgl1 / ffmpeg</text><rect x="50" y="158" width="256" height="30" rx="6" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/><text x="178" y="178" text-anchor="middle" font-size="12.5" fill="#18181b">Python 本体 3.12</text><rect x="50" y="194" width="256" height="30" rx="6" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/><text x="178" y="214" text-anchor="middle" font-size="12.5" fill="#18181b">uv 自身（バイナリ）</text><rect x="332" y="44" width="300" height="200" rx="12" fill="#fff7ed" stroke="#c2410c" stroke-width="2.5"/><text x="482" y="72" text-anchor="middle" font-size="15" font-weight="800" fill="#c2410c">uv ＝ 中身（パッケージ）を管理</text><rect x="354" y="86" width="256" height="30" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="482" y="106" text-anchor="middle" font-size="12.5" fill="#18181b">numpy / opencv / pillow</text><rect x="354" y="122" width="256" height="30" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="482" y="142" text-anchor="middle" font-size="12.5" fill="#18181b">torch / transformers / faiss</text><rect x="354" y="158" width="256" height="30" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="482" y="178" text-anchor="middle" font-size="12.5" fill="#18181b">バージョン固定（uv.lock）</text><rect x="354" y="194" width="256" height="30" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="482" y="214" text-anchor="middle" font-size="12.5" fill="#18181b">仮想環境 .venv に配置</text></svg><figcaption><b>Docker は「箱」</b>（OS・システムライブラリ・Python 本体・uv まで）を用意し、<b>uv は「中身」</b>（numpy・torch などの Python パッケージ）を <code>uv.lock</code> 通りに <code>.venv</code> へ入れます。本講座は、<b>箱は Docker が作り、中身はコンテナに入ってから uv で整える</b>役割分担にしています。</figcaption></figure>

### 料理のたとえ

- **Docker = キッチン**（建物・コンロ・ガス・水道）。どのキッチンでも同じ調理条件にする。
- **uv = レシピと食材の仕込み**（材料を正しいバージョンで揃えて下ごしらえ）。

同じキッチン（Docker）でも、食材を仕込まなければ（uv sync しなければ）料理はできません。逆に食材だけあってもキッチンが違えば（OS が違えば）同じ料理になりません。**両方そろって初めて、どこでも同じ結果**になります。

---

## 3. 登場するファイルとモノの関係

Docker まわりで出てくるファイルと「実体」を、流れで押さえます。

<figure class="lec-fig"><svg viewBox="0 0 680 300" role="img" aria-label="Dockerのファイルと実体の関係。Dockerfileをbuildするとイメージができ、composeでupするとコンテナが起動する。ホストのlecturesやキャッシュはボリュームでコンテナと共有され、pyprojectとuv.lockをコンテナ内でuv syncすると.venvができる" font-family="ui-sans-serif, system-ui, 'Noto Sans JP', sans-serif"><text x="340" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#18181b">ファイル → イメージ → コンテナ。ホストとはボリュームで共有</text><rect x="24" y="48" width="150" height="46" rx="8" fill="#fafafa" stroke="#52525b" stroke-width="2"/><text x="99" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#18181b">Dockerfile</text><text x="99" y="86" text-anchor="middle" font-size="11" fill="#52525b">箱の作り方</text><text x="207" y="66" text-anchor="middle" font-size="11" fill="#2563eb">build</text><line x1="174" y1="71" x2="240" y2="71" stroke="#2563eb" stroke-width="2"/><polygon points="246,71 236,66 236,76" fill="#2563eb"/><rect x="248" y="48" width="150" height="46" rx="8" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/><text x="323" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#1d4ed8">イメージ</text><text x="323" y="86" text-anchor="middle" font-size="11" fill="#52525b">焼いたテンプレ(読取専用)</text><text x="431" y="66" text-anchor="middle" font-size="11" fill="#16a34a">up</text><line x1="398" y1="71" x2="464" y2="71" stroke="#16a34a" stroke-width="2"/><polygon points="470,71 460,66 460,76" fill="#16a34a"/><rect x="472" y="48" width="184" height="46" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="2.5"/><text x="564" y="70" text-anchor="middle" font-size="13" font-weight="700" fill="#15803d">コンテナ</text><text x="564" y="86" text-anchor="middle" font-size="11" fill="#52525b">起動した実体(使い捨て可)</text><text x="232" y="118" text-anchor="middle" font-size="11" fill="#71717a">docker-compose.yaml が build と up の設定をまとめる</text><rect x="472" y="150" width="184" height="120" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"/><text x="564" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="#15803d">コンテナの中</text><rect x="490" y="182" width="148" height="30" rx="6" fill="#ffffff" stroke="#86efac" stroke-width="1.5"/><text x="564" y="202" text-anchor="middle" font-size="11.5" fill="#18181b">uv sync で .venv 生成</text><rect x="490" y="218" width="148" height="42" rx="6" fill="#ffffff" stroke="#86efac" stroke-width="1.5"/><text x="564" y="235" text-anchor="middle" font-size="11" fill="#18181b">/app/lectures, /app/.cache</text><text x="564" y="251" text-anchor="middle" font-size="10.5" fill="#52525b">＝ホストと共有(ボリューム)</text><rect x="24" y="150" width="180" height="120" rx="10" fill="#fff7ed" stroke="#c2410c" stroke-width="2"/><text x="114" y="172" text-anchor="middle" font-size="12.5" font-weight="700" fill="#c2410c">ホスト(あなたの PC)</text><rect x="42" y="182" width="144" height="26" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="114" y="200" text-anchor="middle" font-size="11" fill="#18181b">lectures/ data/ docs/</text><rect x="42" y="212" width="144" height="26" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="114" y="230" text-anchor="middle" font-size="11" fill="#18181b">pyproject / uv.lock</text><rect x="42" y="242" width="144" height="22" rx="6" fill="#ffffff" stroke="#fb923c" stroke-width="1.5"/><text x="114" y="258" text-anchor="middle" font-size="10.5" fill="#18181b">.cache/ (uv・HF)</text><line x1="204" y1="210" x2="468" y2="210" stroke="#c2410c" stroke-width="2" stroke-dasharray="6 4"/><text x="336" y="204" text-anchor="middle" font-size="11" fill="#c2410c">ボリュームで双方向に同期（編集も結果もキャッシュも共有）</text></svg><figcaption><code>Dockerfile</code> を <b>build</b> すると<b>イメージ</b>（焼いたテンプレ）ができ、<code>docker compose up</code> で<b>コンテナ</b>（起動した実体）になります。ホストの <code>lectures/</code> やキャッシュは<b>ボリューム</b>でコンテナと共有され、編集・実行結果・ダウンロード済みパッケージが双方向に同期します。<code>pyproject.toml</code>/<code>uv.lock</code> を<b>コンテナ内で <code>uv sync</code></b> すると <code>.venv</code> ができます。</figcaption></figure>

| ファイル / モノ | 役割 | 誰が使う |
| --- | --- | --- |
| `Dockerfile` | **箱の作り方**。ベース OS・`apt` で libgl1/ffmpeg・uv の設置を書いた手順書 | Docker（build 時） |
| `docker-compose.yaml` | **箱の起動方法**。どのフォルダを共有するか（ボリューム）・起動コマンドをまとめた設定 | Docker（up 時） |
| `.dockerignore` | build 時にイメージへ**コピーしないもの**の一覧（`.venv/` `.cache/` `data/` など） | Docker（build 時） |
| イメージ (image) | Dockerfile から焼いた**読み取り専用のテンプレ**。これを起動するとコンテナになる | Docker |
| コンテナ (container) | イメージを起動した**実体**。中に入って作業する。壊しても作り直せる | あなた |
| ボリューム (volume) | ホストとコンテナで**共有・永続化する棚**（`lectures/` やキャッシュ） | Docker / あなた |
| `pyproject.toml` | **食材リスト**。本体依存と依存グループ（dl/hf/vector …）の定義 | uv |
| `uv.lock` | 食材の**正確なバージョンを固定**したロックファイル（再現性） | uv |
| `.python-version` | 使う Python を **3.12** に固定 | uv |
| `.venv` | uv が作る**仕込み済みの食材置き場**（コンテナの中に作られる） | uv |

> 関係を 1 文で：**`Dockerfile`（作り方）→ イメージ（テンプレ）→ コンテナ（実体）**。コンテナの中で **`uv sync`（`pyproject.toml`/`uv.lock` を読む）→ `.venv`（中身）** ができ、ホストとは **ボリューム**でつながる。

---

## 4. 用語の整理 — イメージ / コンテナ / ボリューム

混同しやすい 3 語をはっきり分けます。

- **イメージ（image）**＝「**焼いたパンの型**」。`Dockerfile` から build して作る、変更しない読み取り専用のテンプレ。本講座では「OS + libgl1/ffmpeg + Python + uv」が入った軽量な箱。
- **コンテナ（container）**＝「**型から作った実物**」。イメージを `up` で起動した実体で、中に入って作業します。**使い捨てできる**のが利点で、壊れても `down` → `up` で作り直せます。
- **ボリューム（volume）**＝「**ホストと共有する棚**」。コンテナは使い捨てなので、消えてほしくないもの（あなたが書いた `lectures/`、ダウンロード済みのモデル/パッケージ）はボリュームでホスト側に置いて永続化します。なお本講座で使うのは正確には **bind mount**（ホストの実フォルダをそのまま共有する形）で、`docker-compose.yaml` の `volumes:` に書きます。`docker volume ls` に出る「名前付きボリューム」とは別物ですが、このページでは広義に「ボリューム」と呼びます。

これが本講座の設計に直結します。**`.venv`（中身）はコンテナの中**に作るので、コンテナを作り直すと消えます。でも **`.cache/`（ダウンロード済みファイル）はボリュームで永続化**しているので、作り直しても **`uv sync` は再ダウンロード無しで素早く**終わります（`UV_LINK_MODE=copy` のため、新しい `.venv` への展開・コピーは毎回少し走ります）。「使い捨てできるが、遅くはならない」を両立させる仕組みです。

---

## 5. この講座での使い方（手順）

本講座の Docker 運用は **「箱を起動 → 中に入る → uv で整える → 実行する」** の 4 ステップです。コマンドは 2 か所（**ホスト**と**コンテナの中**）で打ち分けます。

> **前提**: 母艦に **Docker** が入っていること（Mac/Windows は Docker Desktop、Linux は Docker Engine + Compose v2）。`docker compose version` が表示されれば OK（未導入なら [Docker 公式の入手ガイド](https://docs.docker.com/get-started/get-docker/) から）。以下のコマンドは **`docker-compose.yaml` のあるリポジトリのルート**（`lecture-cv/`）で実行します。

**① ホストで：箱をビルドして起動し、コンテナに入る**

```bash
docker compose up -d --build         # 箱をビルドして起動。-d はバックグラウンド起動（ターミナルがすぐ戻る）。初回だけ --build
docker compose exec lecture-cv bash  # コンテナに入る（lecture-cv は docker-compose.yaml の services 名。プロンプトが root@xxxx:/app# に変わる）
```

**② コンテナの中で：uv で環境を整えて実行する**

```bash
# ── ここからはコンテナの中（プロンプト例: root@xxxxxxxx:/app#）──
uv sync                                          # 画像基礎(00〜11)の依存をそろえる
uv run python lectures/00_setup/check_env.py     # 環境スモークテスト
uv run python lectures/01_image_basics/01_imread_imwrite.py

# 深層トラック(12 以降)に進むときは、その回に必要なグループを足す
uv sync --group dl --group hf
uv run python lectures/<id>/01_xxx.py            # <id> は各回フォルダ名（例 13_classification_transfer_learning）

exit                                             # コンテナから出る（コンテナは起動したまま）
```

ポイント：

- **どのグループを足すか**は各回ページ上部の「依存グループ」欄に書いてあります。`uv sync --group <name>` で足してから `uv run python ...`。
- **各回ページの「▶ 動かし方」に出てくる `uv sync ...` / `uv run python ...` は、Docker の場合①でコンテナに入った後、そのまま同じコマンドを打てば OK** です（コンテナ内では中身が同じなので、ネイティブ実行と区別なく使えます）。
- コンテナは `sleep infinity` で起こしっぱなしです。次回からは ① の `docker compose exec lecture-cv bash` で入り直すだけ（`up` 済みなら省略可）。

> 注意: `uv add` など **`uv.lock` を書き換える操作はホスト側で**行ってください。コンテナ内は依存定義（`pyproject.toml`/`uv.lock`）を単一ファイルでマウントしているため、lock の置換が失敗することがあります。コンテナ内では **`uv sync` / `uv run` に留める**のが安全です。

---

## 6. なぜ「依存はコンテナの中で uv sync」なのか（設計意図）

依存（torch など）を**イメージのビルド時に焼き込む**やり方もありますが、本講座は**コンテナに入ってから `uv sync` で整える**方式にしています。理由は 3 つ：

1. **uv の運用がネイティブと完全に同じになる。** ローカルでも Docker でも、環境構築は `uv sync` / `uv sync --group ...`、実行は `uv run python ...`。覚えることが 1 つで済み、「コンテナに入ったら uv をそのまま使う」だけになります。
2. **回ごとに必要なグループだけ足せる。** 重い深層依存を最初から全部入れず、到達した回で `uv sync --group dl --group hf` のように増やせます（本講座の「重い依存は到達してから」方針と一致）。
3. **イメージが軽量で再現性が高い。** 箱には OS・ライブラリ・uv だけを入れるので build が速く小さい。中身は `uv.lock` で固定されるので、いつ `uv sync` しても同じバージョンが入ります。

> 速度の心配は **キャッシュの永続化**で解消しています。`docker-compose.yaml` は uv のダウンロードキャッシュ（`./.cache/uv`）と HuggingFace のキャッシュ（`./.cache/huggingface`）をボリュームでホストに置くので、**初回の `uv sync` だけ DL が走り、2 回目以降（コンテナを作り直しても）は再ダウンロード無し**で素早く終わります（DL が無いだけで、新しい `.venv` への展開・コピーは走ります）。

---

## 7. よく使うコマンド早見表

| やりたいこと | コマンド（ホストで実行） |
| --- | --- |
| 箱をビルドして起動 | `docker compose up -d --build` |
| 起動（ビルド済み） | `docker compose up -d` |
| コンテナに入る | `docker compose exec lecture-cv bash` |
| （入った後）依存を整える | `uv sync` / `uv sync --group dl --group hf`（コンテナ内） |
| （入った後）各回を実行 | `uv run python lectures/<id>/...`（コンテナ内） |
| いま動いているか確認 | `docker compose ps` |
| ログを見る（※主プロセスが sleep のため通常ほぼ空。スクリプト出力は exec した端末側に出る） | `docker compose logs -f` |
| 止める（コンテナ削除） | `docker compose down` |
| 作り直す（Dockerfile を変えた） | `docker compose up -d --build` |
| 容量を掃除する | `docker system prune`（未使用イメージ等を削除。注意して実行） |

> 本講座の Docker は **`docker compose exec lecture-cv bash` でコンテナに入ってから、その中で `uv sync` / `uv run` を使う**のが基本です（連続して何本も動かす学習に向くため）。入った後は、各回ページの「▶ 動かし方」の `uv ...` コマンドをそのまま実行できます。

---

## 8. つまずいたら（FAQ）

**Q1. `docker compose up` で `COPY ... not found` などビルドが失敗する。**
最新を取り込んでから再実行してください（`git pull`）。`Dockerfile` が参照するファイル（`.python-version` など）はリポジトリに含まれている前提です。

**Q2. `docker compose exec` が「is not running」になる。**
先に `docker compose up -d` でコンテナを起動してください。`exec` は**起動中のコンテナ**に入るコマンドです。

**Q3. コンテナに入って `uv run` したら `ModuleNotFoundError`。**
その回に必要なグループを `uv sync --group <name>` で入れてからにします（各回ページの「依存グループ」欄を確認）。`uv sync`（無印）は画像基礎(00〜11)の本体依存（＋開発ツールの `dev` グループ）だけで、深層トラックの torch などは含みません。

**Q4. 毎回 `uv sync` でパッケージを再ダウンロードして遅い。**
`./.cache/uv` のボリュームが効いていない可能性があります。`docker-compose.yaml` の `volumes` に `./.cache/uv:/app/.cache/uv` があるか確認してください。効いていれば 2 回目以降は DL 無しで素早く終わります。

**Q5. `cv2.imshow` がコンテナで固まる / プロセスごと落ちる。**
コンテナには GUI が無く、本講座は `opencv-python-headless`（`imshow` 自体が無い）を使います。結果は `lectures/<id>/outputs/` にファイル保存して確認します（headless 方針）。詳しくは `00_setup` / `01_image_basics`。

**Q6. モデルが毎回ダウンロードされる。**
HuggingFace キャッシュ（`./.cache/huggingface`）のボリュームで永続化済みです。完全オフラインで回すなら、事前に DL した上で `HF_HUB_OFFLINE=1` を設定します。

**Q7. ディスクが足りない。**
未使用のイメージ・コンテナが溜まっている可能性。`docker system prune`（必要なら `-a`）で掃除します（消える対象を確認してから実行）。

**Q8. コンテナが作ったファイルがホストで root 所有になり、削除に sudo が要る。**
コンテナは root で動くため、`lectures/<id>/outputs/` の生成物や `.cache/` がホスト側で root 所有になることがあります。消すときは `docker compose exec lecture-cv rm ...`（コンテナ内から消す）か `sudo rm ...` で対処できます。気になる場合は `docker-compose.yaml` の service に `user: "${UID}:${GID}"` を足してホストの UID で動かす方法もあります。

**Q9. GPU を使いたい。**
本講座は CPU 既定です。GPU は `Dockerfile` 先頭の CUDA 化コメントと `docker-compose.yaml` の `deploy.resources`（nvidia）のコメントを外します（要 nvidia-container-toolkit）。`pick_device()`（`00_setup`）はそのまま `cuda` を拾います。

---

> 次へ：環境ができたら **[はじめ方](getting-started.html)** の「各回の進め方」に従って `00_setup` → `01_image_basics` と進みましょう。Docker と uv の使い分けを、手を動かしながら自分のものにしてください。

---

> 参照：Docker（v28 系）/ Docker Compose v2 / uv 0.10 系 / Python 3.12 — 2026-06
> （CPU 前提・headless OpenCV・Linux は CPU ホイール明示。コマンドは `docker compose`（v2 構文）に統一）
