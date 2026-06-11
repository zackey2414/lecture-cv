"""lecture-cv 教材閲覧サイト・ビルダー。

`lectures/<id>/README.md` と `.py` スクリプトを Markdown→HTML に事前レンダリングし、
`courses` と同系の意匠（紫グラデ hero・Noto Sans JP・JetBrains Mono・トラック→カード）で
オフラインでも開ける静的サイトを `site/` に生成する。

使い方:
    uv run --group site python tools/build_site.py
    # 生成後、site/index.html をブラウザで開く
"""

from __future__ import annotations

import html
import json
import pathlib
import re

import markdown
from pygments.formatters import HtmlFormatter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LECT = ROOT / "lectures"
SITE = ROOT / "site"
ASSETS = SITE / "assets"

LEVEL_CLASS = {"入門": "intro", "初級": "beginner", "中級": "intermediate", "上級": "advanced"}

SITE_CSS = """@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --p100:#f3e8ff;--p200:#e9d5ff;--p400:#c084fc;--p500:#a855f7;--p600:#9333ea;--p700:#7e22ce;--p800:#6b21a8;--p900:#581c87;
  --g50:#fafafa;--g100:#f4f4f5;--g200:#e4e4e7;--g300:#d4d4d8;--g500:#71717a;--g600:#52525b;--g700:#3f3f46;--g800:#27272a;--g900:#18181b;
  --radius:16px;--shadow:0 4px 24px rgba(0,0,0,.10);--t:.25s cubic-bezier(.4,0,.2,1)}
html{font-family:'Noto Sans JP','Hiragino Sans','Yu Gothic',sans-serif;color:var(--g800);background:var(--g50);line-height:1.8;-webkit-font-smoothing:antialiased}
code,pre,.mono{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace}
a{color:var(--p700)}
/* header */
.site-header{position:fixed;top:0;left:0;right:0;z-index:100;height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 20px;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--g200)}
.home-link{display:inline-flex;align-items:center;gap:10px;text-decoration:none;color:var(--g900);font-weight:800}
.logo-mark{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,var(--p600),var(--p400));color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px}
.logo-text small{display:block;font-size:10px;font-weight:500;color:var(--g500);letter-spacing:.3px}
.nav-roadmap{text-decoration:none;font-weight:700;font-size:14px;color:var(--p700);padding:6px 12px;border-radius:10px}
.nav-roadmap:hover{background:var(--p100)}
/* hero */
.hero{margin-top:56px;background:linear-gradient(135deg,var(--p900),var(--p600) 55%,var(--p400));color:#fff;text-align:center;padding:4.5rem 1.5rem 6rem}
.hero h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:900;letter-spacing:-.02em}
.hero p{margin-top:.6rem;font-size:1.1rem;opacity:.92}
.hero .hero-meta{font-size:.9rem;opacity:.8;margin-top:1rem}
/* index cards */
.cards{display:grid;gap:1.5rem;max-width:1100px;margin:-3rem auto 4rem;padding:0 1.5rem;position:relative;z-index:1}
.lang-section{background:#fff;border-radius:var(--radius);box-shadow:var(--shadow);border:1px solid var(--g200);padding:1.5rem 1.75rem 1.9rem}
.lang-header h2{font-size:1.3rem;font-weight:800;color:var(--g900);border-left:5px solid var(--p500);padding-left:.6rem}
.lang-levels{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem;margin-top:1.1rem}
.level-card{background:var(--g50);border-radius:12px;padding:1.1rem 1.25rem;text-decoration:none;color:inherit;border:1px solid var(--g200);transition:transform var(--t),box-shadow var(--t),border-color var(--t);display:flex;flex-direction:column;gap:.5rem}
.level-card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(0,0,0,.10);border-color:var(--p400);background:#fff}
.level-card h3{font-size:1rem;font-weight:700;color:var(--g900);display:flex;align-items:center;gap:.45rem;flex-wrap:wrap}
.level-card p{font-size:.85rem;color:var(--g600);flex:1}
.mod-num{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--p600);font-weight:700}
.card-foot{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;margin-top:.2rem}
.level-badge{display:inline-block;font-size:.68rem;font-weight:700;padding:.18em .6em;border-radius:6px;color:#fff;letter-spacing:.3px;white-space:nowrap}
.level-badge.intro{background:#22c55e}.level-badge.beginner{background:#3b82f6}.level-badge.intermediate{background:#f59e0b}.level-badge.advanced{background:#ef4444}
.status{font-size:.7rem;font-weight:700;padding:.15em .55em;border-radius:999px}
.status.done{background:#dcfce7;color:#15803d}.status.wip{background:var(--g100);color:var(--g500)}
.chip{font-size:.7rem;background:var(--p100);color:var(--p700);padding:.12em .5em;border-radius:6px;font-weight:600}
/* module page */
.container{max-width:880px;margin:56px auto 0;padding:2rem 1.4rem 4rem}
.mod-banner{background:linear-gradient(135deg,var(--p800),var(--p600));color:#fff;border-radius:var(--radius);padding:1.8rem 1.8rem;margin-bottom:2rem}
.mod-eyebrow{font-size:.85rem;opacity:.9;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.mod-banner h1{font-size:clamp(1.5rem,3.5vw,2rem);font-weight:900;margin:.5rem 0 .6rem;line-height:1.4}
.mod-goal{font-size:.95rem;opacity:.95;line-height:1.7}
.meta-table{width:100%;margin-top:1.1rem;border-collapse:collapse;background:rgba(255,255,255,.10);border-radius:10px;overflow:hidden;font-size:.85rem}
.meta-table th{text-align:left;width:7.5rem;padding:.5rem .8rem;color:#fff;opacity:.85;font-weight:600;vertical-align:top}
.meta-table td{padding:.5rem .8rem;color:#fff}
.meta-table .chip{background:rgba(255,255,255,.22);color:#fff}
/* content typography */
.content{font-size:1rem;color:var(--g800)}
.content h1,.content h2,.content h3,.content h4{font-weight:800;color:var(--g900);line-height:1.4;margin:2.2rem 0 .9rem;scroll-margin-top:70px}
.content h2{font-size:1.45rem;border-bottom:2px solid var(--p200);padding-bottom:.35rem}
.content h3{font-size:1.2rem;color:var(--p800)}
.content h4{font-size:1.05rem}
.content p{margin:.9rem 0}
.content ul,.content ol{margin:.8rem 0 .8rem 1.4rem}
.content li{margin:.3rem 0}
.content strong{color:var(--g900)}
.content a{text-decoration:underline}
.content blockquote{border-left:4px solid var(--p400);background:var(--p100);padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:1rem 0;color:var(--g700)}
.content table{border-collapse:collapse;width:100%;margin:1.2rem 0;font-size:.9rem;display:block;overflow-x:auto}
.content th,.content td{border:1px solid var(--g200);padding:.5rem .75rem;text-align:left;vertical-align:top}
.content thead th{background:var(--p100);color:var(--p800);font-weight:700}
.content tbody tr:nth-child(even){background:var(--g50)}
.content :not(pre)>code{background:var(--g100);color:#be185d;padding:.12em .4em;border-radius:5px;font-size:.86em}
.content pre,.codehilite,.highlight{background:#1e1b2e;border-radius:10px;padding:1rem 1.1rem;overflow-x:auto;margin:1.1rem 0;font-size:.84rem;line-height:1.6}
.content pre code,.codehilite code{background:none;color:#e7e3f4;padding:0}
.codehilite pre{background:none;padding:0;margin:0}
/* code source accordions */
.srcblock{margin:.7rem 0;border:1px solid var(--g200);border-radius:10px;overflow:hidden;background:#fff}
.srcblock>summary{cursor:pointer;padding:.7rem 1rem;font-weight:700;background:var(--g50);user-select:none}
.srcblock>summary:hover{background:var(--p100)}
.srcblock .highlight{margin:0;border-radius:0}
.muted{color:var(--g500)}
/* nav */
.prevnext{display:flex;justify-content:space-between;gap:1rem;max-width:880px;margin:1rem auto 3rem;padding:0 1.4rem}
.prevnext a{flex:1;text-decoration:none;background:#fff;border:1px solid var(--g200);border-radius:12px;padding:.9rem 1.1rem;font-weight:700;color:var(--g800);transition:border-color var(--t),transform var(--t)}
.prevnext a:hover{border-color:var(--p400);transform:translateY(-2px)}
.prevnext a:last-child{text-align:right}
.footer{text-align:center;padding:2.5rem 1rem;font-size:.85rem;color:var(--g500)}
@media(max-width:600px){.hero{padding:3.5rem 1rem 4rem}.cards{margin-top:-2rem}.container{padding:1.2rem 1rem 3rem}}
"""

# ----------------------------------------------------------------------------- helpers
def md_to_html(text: str) -> str:
    """Markdown 文字列を HTML へ（コードハイライト・表・TOC 対応）。"""
    return markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "codehilite", "sane_lists", "attr_list", "toc"],
        extension_configs={"codehilite": {"css_class": "codehilite", "guess_lang": False}},
    )


def highlight_py(code: str) -> str:
    """Python ソースを Pygments で `.highlight` ブロックへ。"""
    from pygments import highlight
    from pygments.lexers import PythonLexer

    return highlight(code, PythonLexer(), HtmlFormatter(cssclass="highlight"))


def ordered_scripts(d: pathlib.Path) -> list[pathlib.Path]:
    """番号付き → ヘルパ → exercises.py の順に .py を並べる。"""
    pys = [p for p in d.glob("*.py")]
    numbered = sorted(p for p in pys if p.name[:1].isdigit())
    ex = [p for p in pys if p.name == "exercises.py"]
    helpers = sorted(p for p in pys if not p.name[:1].isdigit() and p.name != "exercises.py")
    return numbered + helpers + ex


def page(title: str, body: str, *, rel: str = "") -> str:
    """共通レイアウト（固定ヘッダ＋本文）。"""
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{rel}assets/site.css">
<link rel="stylesheet" href="{rel}assets/pygments.css">
</head>
<body>
<header class="site-header">
  <a class="home-link" href="{rel}index.html">
    <span class="logo-mark">CV</span>
    <span class="logo-text">lecture-cv<small>Computer Vision を自分の血肉にする</small></span>
  </a>
  <a class="nav-roadmap" href="{rel}roadmap.html">ロードマップ</a>
</header>
{body}
</body>
</html>
"""


# ----------------------------------------------------------------------------- load data
data = json.loads((ROOT / "docs" / "curriculum.json").read_text())
modules = data["modules"]

for m in modules:
    d = LECT / m["id"]
    m["_dir"] = d
    m["_scripts"] = ordered_scripts(d) if d.is_dir() else []
    m["_authored"] = any(p.name not in ("__init__.py",) for p in m["_scripts"])
    m["_num"] = m["id"][:2]

authored = [m for m in modules if m["_authored"]]

# ----------------------------------------------------------------------------- index page
tracks: dict[str, list] = {}
for m in modules:
    tracks.setdefault(m["track"], []).append(m)

cards = []
for track, ms in tracks.items():
    items = []
    for m in ms:
        lv = LEVEL_CLASS.get(m["level"], "intro")
        status = '<span class="status done">公開</span>' if m["_authored"] else '<span class="status wip">準備中</span>'
        href = f'{m["id"]}.html'
        goal = html.escape((m.get("goal") or "")[:110]) + ("…" if len(m.get("goal") or "") > 110 else "")
        groups = "".join(f'<code class="chip">{html.escape(g)}</code>' for g in (m.get("needs_groups") or []))
        items.append(
            f'<a class="level-card" href="{href}">'
            f'<h3><span class="level-badge {lv}">{m["level"]}</span> '
            f'<span class="mod-num">{m["_num"]}</span> {html.escape(m["title"])}</h3>'
            f'<p>{goal}</p>'
            f'<div class="card-foot">{status}{groups}</div></a>'
        )
    cards.append(
        f'<section class="lang-section"><div class="lang-header"><div>'
        f'<h2>{html.escape(track)}</h2></div></div>'
        f'<div class="lang-levels">{"".join(items)}</div></section>'
    )

hero = f"""<header class="hero">
  <h1>lecture-cv</h1>
  <p>Computer Vision を「AI の補助なしで自力で書ける」まで叩き込む全{len(modules)}回ハンズオン講座</p>
  <p class="hero-meta">公開 {len(authored)} / {len(modules)} 回　・　CPU のみで完走　・　各回 解説＋実行コード＋演習</p>
</header>"""

index_body = hero + f'<main class="cards">{"".join(cards)}</main>' + \
    '<footer class="footer">lecture-cv ／ 設計時点: 2026-06 ／ 各回フッターにライブラリ版を明記</footer>'
SITE.mkdir(exist_ok=True)
(SITE / "index.html").write_text(page("lecture-cv 教材", index_body))

# ----------------------------------------------------------------------------- roadmap page
roadmap_md = (ROOT / "docs" / "roadmap.md").read_text()
roadmap_body = f'<main class="container content">{md_to_html(roadmap_md)}</main>'
(SITE / "roadmap.html").write_text(page("ロードマップ — lecture-cv", roadmap_body))

# ----------------------------------------------------------------------------- module pages
def strip_h1(md: str) -> str:
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip():
            return "\n".join(lines[i + 1 :]) if ln.startswith("# ") else md
    return md


for idx, m in enumerate(modules):
    lv = LEVEL_CLASS.get(m["level"], "intro")
    prev_m = modules[idx - 1] if idx > 0 else None
    next_m = modules[idx + 1] if idx + 1 < len(modules) else None

    readme = m["_dir"] / "README.md"
    readme_html = md_to_html(strip_h1(readme.read_text())) if readme.exists() else "<p>（準備中）</p>"

    # 各スクリプトの全文（折りたたみ）
    scripts_html = ""
    code_scripts = [p for p in m["_scripts"] if p.name != "exercises.py"]
    ex = [p for p in m["_scripts"] if p.name == "exercises.py"]
    if code_scripts:
        blocks = []
        for p in code_scripts:
            blocks.append(
                f'<details class="srcblock"><summary><code>{html.escape(p.name)}</code></summary>'
                f'{highlight_py(p.read_text())}</details>'
            )
        scripts_html = f'<h2>スクリプト全文</h2>{"".join(blocks)}'
    ex_html = ""
    if ex:
        ex_html = (
            f'<h2>演習</h2><details class="srcblock"><summary><code>exercises.py</code>'
            f'（TODO を埋めて自己採点）</summary>{highlight_py(ex[0].read_text())}</details>'
        )

    groups = "".join(f'<code class="chip">{html.escape(g)}</code>' for g in (m.get("needs_groups") or [])) or '<span class="muted">追加依存なし</span>'
    status = '公開' if m["_authored"] else '準備中（プレースホルダ）'
    meta_rows = (
        f'<tr><th>トラック</th><td>{html.escape(m["track"])}</td></tr>'
        f'<tr><th>レベル</th><td><span class="level-badge {lv}">{m["level"]}</span></td></tr>'
        f'<tr><th>依存グループ</th><td>{groups}</td></tr>'
        f'<tr><th>評価</th><td>{html.escape(m.get("evaluation") or "—")}</td></tr>'
        f'<tr><th>完成物</th><td>{html.escape(m.get("deliverable") or "—")}</td></tr>'
        f'<tr><th>状態</th><td>{status}</td></tr>'
    )
    banner = (
        f'<div class="mod-banner"><div class="mod-eyebrow">'
        f'<span class="level-badge {lv}">{m["level"]}</span> {html.escape(m["track"])} ／ 第{int(m["_num"])}回</div>'
        f'<h1>{html.escape(m["title"])}</h1>'
        f'<p class="mod-goal">🎯 {html.escape(m.get("goal") or "")}</p>'
        f'<table class="meta-table">{meta_rows}</table></div>'
    )

    nav = '<nav class="prevnext">'
    nav += f'<a href="{prev_m["id"]}.html">← {prev_m["_num"]} {html.escape(prev_m["title"][:24])}</a>' if prev_m else '<span></span>'
    nav += f'<a href="{next_m["id"]}.html">{next_m["_num"]} {html.escape(next_m["title"][:24])} →</a>' if next_m else '<span></span>'
    nav += '</nav>'

    body = f'<main class="container">{banner}<article class="content">{readme_html}{scripts_html}{ex_html}</article>{nav}</main>'
    (SITE / f'{m["id"]}.html').write_text(page(f'{m["_num"]} {m["title"]} — lecture-cv', body))

# ----------------------------------------------------------------------------- assets
ASSETS.mkdir(parents=True, exist_ok=True)
(ASSETS / "pygments.css").write_text(
    HtmlFormatter().get_style_defs(".codehilite") + "\n" + HtmlFormatter().get_style_defs(".highlight")
)
(ASSETS / "site.css").write_text(SITE_CSS)

print(f"built site/: {len(modules)} module pages ({len(authored)} authored) + index + roadmap")
