"""lecture-cv 教材閲覧サイト・ビルダー。

`lectures/<id>/README.md` と `.py` スクリプトを Markdown→HTML に事前レンダリングし、
ドキュメント風（左=全モジュール目次／中央=本文ブロック／右=ページ内目次）の
オフラインでも開ける静的サイトを `site/` に生成する。

使い方:
    uv run --group site python tools/build_site.py
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
PYGMENTS_STYLE = "monokai"  # 暗背景に整合する配色（コードの可読性）

# ブラウザ用 favicon（ヘッダーの CV ロゴと同一: オレンジ角丸 + 白「CV」）
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#f97316"/>'
    '<text x="16" y="22.5" text-anchor="middle" '
    'font-family="ui-sans-serif, system-ui, \'Segoe UI\', Roboto, sans-serif" '
    'font-size="15" font-weight="900" fill="#ffffff">CV</text></svg>'
)

SITE_CSS = """@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --p50:#fff7ed;--p100:#ffedd5;--p200:#fed7aa;--p400:#fb923c;--p500:#f97316;--p600:#ea580c;--p700:#c2410c;--p800:#9a3412;--p900:#7c2d12;
  --b50:#eff6ff;--b100:#dbeafe;--b200:#bfdbfe;--b400:#60a5fa;--b500:#3b82f6;--b600:#2563eb;--b700:#1d4ed8;--b800:#1e40af;--b900:#1e3a8a;
  --g50:#fafafa;--g100:#f4f4f5;--g200:#e4e4e7;--g300:#d4d4d8;--g500:#71717a;--g600:#52525b;--g700:#3f3f46;--g800:#27272a;--g900:#18181b;
  --code-bg:#282a36;--code-fg:#f8f8f2;
  --radius:16px;--shadow:0 4px 24px rgba(0,0,0,.10);--t:.2s cubic-bezier(.4,0,.2,1)}
html{font-family:'Noto Sans JP','Hiragino Sans','Yu Gothic',sans-serif;color:var(--g800);background:var(--g50);line-height:1.85;-webkit-font-smoothing:antialiased;scroll-behavior:smooth}
code,pre,.mono{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace}
a{color:var(--b700)}
img{max-width:100%}
/* ===== header ===== */
.site-header{position:fixed;top:0;left:0;right:0;z-index:100;height:56px;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:0 20px;background:rgba(255,255,255,.93);backdrop-filter:blur(8px);border-bottom:1px solid var(--g200)}
.home-link{display:inline-flex;align-items:center;gap:10px;text-decoration:none;color:var(--g900);font-weight:800;min-width:0;overflow:hidden}
.logo-mark{flex:none;width:30px;height:30px;border-radius:8px;background:var(--p500);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px}
.logo-text{display:flex;flex-direction:column;justify-content:center;min-width:0;overflow:hidden;white-space:nowrap;line-height:1.2}
.logo-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.logo-text small{display:block;font-size:10px;font-weight:500;color:var(--g500);letter-spacing:.3px;overflow:hidden;text-overflow:ellipsis}
.topnav{display:flex;gap:.3rem;flex:none}
.topnav a{text-decoration:none;font-weight:700;font-size:13.5px;color:var(--p700);padding:6px 11px;border-radius:10px;white-space:nowrap}
.topnav a:hover{background:var(--p100)}
/* ===== hero (index) ===== */
.hero{margin-top:56px;background:var(--p700);color:#fff;text-align:center;padding:4.5rem 1.5rem 6rem}
.hero h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:900;letter-spacing:-.02em}
.hero p{margin-top:.6rem;font-size:1.1rem;opacity:.92}
.hero .hero-meta{font-size:.9rem;opacity:.95;margin-top:1rem}
/* ===== index cards ===== */
.cards{display:grid;gap:1.5rem;max-width:1240px;margin:-3rem auto 4rem;padding:0 1.5rem;position:relative;z-index:1}
.lang-section{background:#fff;border-radius:var(--radius);box-shadow:var(--shadow);border:1px solid var(--g200);padding:1.5rem 1.75rem 1.9rem}
.lang-header h2{font-size:1.3rem;font-weight:800;color:var(--g900);border-left:5px solid var(--p500);padding-left:.6rem}
.lang-levels{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(290px,100%),1fr));gap:1rem;margin-top:1.1rem}
.level-card{background:var(--g50);border-radius:12px;padding:1.1rem 1.25rem;text-decoration:none;color:inherit;border:1px solid var(--g200);transition:transform var(--t),box-shadow var(--t),border-color var(--t);display:flex;flex-direction:column;gap:.5rem;min-width:0}
.level-card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(0,0,0,.10);border-color:var(--p400);background:#fff}
.level-card h3{font-size:.98rem;font-weight:700;color:var(--g900);display:flex;align-items:baseline;gap:.4rem;flex-wrap:wrap;min-width:0;overflow-wrap:anywhere;word-break:break-word;line-height:1.45}
.level-card p{font-size:.85rem;color:var(--g600);flex:1;overflow-wrap:anywhere}
.mod-num{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--p700);font-weight:700}
.card-foot{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;margin-top:.2rem}
.level-badge{display:inline-block;font-size:.68rem;font-weight:700;padding:.18em .6em;border-radius:6px;color:#fff;letter-spacing:.3px;white-space:nowrap}
.level-badge.intro{background:#15803d}.level-badge.beginner{background:#2563eb}.level-badge.intermediate{background:#b45309}.level-badge.advanced{background:#dc2626}
.status{font-size:.7rem;font-weight:700;padding:.15em .55em;border-radius:999px;white-space:nowrap}
.status.done{background:#dcfce7;color:#15803d}.status.wip{background:var(--g100);color:var(--g500)}
.chip{font-size:.7rem;background:var(--p100);color:var(--p700);padding:.12em .5em;border-radius:6px;font-weight:600;white-space:nowrap}
/* ===== index view toggle (ジャンル別 / 難易度別) ===== */
.view-toggle{display:flex;gap:.3rem;justify-content:center;width:max-content;max-width:94%;margin:-2.4rem auto 2.7rem;background:#fff;border:1px solid var(--g200);border-radius:999px;padding:.32rem;box-shadow:var(--shadow);position:relative;z-index:1}
.vt-btn{border:none;background:transparent;color:var(--g600);font-weight:700;font-size:.9rem;padding:.5rem 1.25rem;border-radius:999px;cursor:pointer;font-family:inherit;white-space:nowrap}
.vt-btn.active{background:var(--p600);color:#fff}
.cards-view{display:grid;gap:1.5rem}
.cards-view[hidden]{display:none}
.content a.rm-link{text-decoration:none}
.content a.rm-link code{cursor:pointer;background:var(--b50);color:var(--b800);white-space:nowrap}
.content a.rm-link:hover code{background:var(--p100);color:var(--p700)}
/* ===== 3-column layout (module/roadmap/graph) ===== */
.layout{display:grid;gap:1.6rem;max-width:1400px;margin:56px auto 0;padding:1.4rem 1.4rem 3rem;align-items:start}
.layout.has-left.has-right{grid-template-columns:240px minmax(0,1fr) 234px}
.layout.has-left:not(.has-right){grid-template-columns:240px minmax(0,1fr)}
.col-main{min-width:0}
.side-left,.side-right{position:sticky;top:72px;align-self:start;max-height:calc(100vh - 86px);overflow-y:auto;font-size:.82rem;scrollbar-width:thin}
.side-left::-webkit-scrollbar,.side-right::-webkit-scrollbar{width:7px}
.side-left::-webkit-scrollbar-thumb,.side-right::-webkit-scrollbar-thumb{background:var(--g200);border-radius:4px}
/* left: module nav */
.modnav{background:#fff;border:1px solid var(--g200);border-radius:14px;padding:.55rem .5rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.modnav-home{display:block;font-weight:800;color:var(--g900);text-decoration:none;padding:.4rem .6rem;border-radius:8px;font-size:.85rem}
.modnav-home:hover{background:var(--p100)}
.nav-track{font-size:.7rem;font-weight:800;color:var(--b700);margin:.75rem .45rem .2rem;letter-spacing:.02em;text-transform:none}
.nav-item{display:flex;align-items:flex-start;gap:.4rem;text-decoration:none;color:var(--g700);padding:.26rem .55rem;border-radius:7px;line-height:1.35;overflow-wrap:anywhere}
.nav-item:hover{background:var(--g100)}
.nav-item.active{background:var(--p100);color:var(--p800);font-weight:700}
.nav-item .dot{width:7px;height:7px;border-radius:50%;flex:none;margin-top:.42em}
.dot.intro{background:#16a34a}.dot.beginner{background:#2563eb}.dot.intermediate{background:#d97706}.dot.advanced{background:#dc2626}
.nav-num{font-family:'JetBrains Mono',monospace;color:var(--g500);font-size:.82em;flex:none}
/* right: in-page TOC */
.side-title{font-size:.76rem;font-weight:800;color:var(--b700);padding:.3rem .4rem .35rem;letter-spacing:.02em}
.side-right .toc,.toc-extra{font-size:.8rem}
.side-right .toc ul,.toc-extra{list-style:none;margin:0;padding:0}
.side-right .toc ul ul{padding-left:.75rem}
.side-right .toc li,.toc-extra li{margin:.08rem 0}
.side-right .toc a,.toc-extra a{display:block;text-decoration:none;color:var(--g600);padding:.18rem .45rem;border-radius:6px;border-left:2px solid transparent;overflow-wrap:anywhere}
.side-right .toc a:hover,.toc-extra a:hover{background:var(--g100);color:var(--p700);border-left-color:var(--p400)}
.toc-extra{border-top:1px solid var(--g200);margin-top:.4rem;padding-top:.3rem}
/* ===== module banner ===== */
.mod-banner{background:var(--p700);color:#fff;border-radius:var(--radius);padding:1.7rem 1.8rem;margin-bottom:1.4rem}
.mod-eyebrow{font-size:.85rem;opacity:.92;display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.mod-banner h1{font-size:clamp(1.4rem,3vw,1.95rem);font-weight:900;margin:.5rem 0 .6rem;line-height:1.45;overflow-wrap:anywhere}
.mod-goal{font-size:.95rem;opacity:.96;line-height:1.75}
.meta-table{width:100%;margin-top:1.1rem;border-collapse:collapse;background:rgba(0,0,0,.14);border-radius:10px;overflow:hidden;font-size:.85rem}
.meta-table th{text-align:left;width:7.5rem;padding:.5rem .8rem;color:#fff;font-weight:600;vertical-align:top}
.meta-table td{padding:.5rem .8rem;color:#fff;overflow-wrap:anywhere}
.meta-table .chip{background:rgba(0,0,0,.25);color:#fff}
.prereq-row a{display:inline-block;background:rgba(0,0,0,.25);color:#fff;padding:.1em .5em;border-radius:6px;text-decoration:none;font-size:.86em;margin:.1rem .2rem .1rem 0}
.mod-banner .muted{color:rgba(255,255,255,.9)}
/* ===== content blocks (courses-like cards) ===== */
.block{background:#fff;border:1px solid var(--g200);border-radius:14px;padding:1.2rem 1.6rem;margin:1.05rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.block>h2:first-child,.block>h3:first-child{margin-top:.15rem}
.content{font-size:1rem;color:var(--g800)}
.content h1,.content h2,.content h3,.content h4{font-weight:800;color:var(--g900);line-height:1.45;margin:1.6rem 0 .8rem;scroll-margin-top:72px;overflow-wrap:anywhere}
.content h2{font-size:1.4rem;color:var(--p800);border-bottom:2px solid var(--p100);padding-bottom:.35rem}
.content h3{font-size:1.17rem;color:var(--b700)}
.content h4{font-size:1.04rem}
.content p{margin:.85rem 0}
.content ul,.content ol{margin:.75rem 0 .75rem 1.4rem}
.content li{margin:.3rem 0}
.content strong{color:var(--p700)}
.content a{text-decoration:underline;text-underline-offset:2px;overflow-wrap:anywhere;word-break:break-word}
.content blockquote{border-left:4px solid var(--p400);background:var(--p50);padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:1rem 0;color:var(--g700)}
.content table{border-collapse:collapse;width:100%;margin:1.1rem 0;font-size:.9rem;display:block;overflow-x:auto}
.content th,.content td{border:1px solid var(--g200);padding:.5rem .75rem;text-align:left;vertical-align:top}
.content thead th{background:var(--b50);color:var(--b800);font-weight:700;white-space:nowrap}
.content tbody tr:nth-child(even){background:var(--g50)}
.content :not(pre)>code{background:var(--g100);color:var(--p700);padding:.12em .42em;border-radius:5px;font-size:.85em;overflow-wrap:anywhere}
/* ===== code blocks (dark, monokai) ===== */
.content pre,.codehilite,.highlight{border-radius:10px;padding:.9rem 1.05rem;overflow-x:auto;margin:1rem 0;font-size:.84rem;line-height:1.6;tab-size:4}
.content pre{background:var(--code-bg);color:var(--code-fg)}
.codehilite,.highlight{color:var(--code-fg)}
.content pre code,.codehilite code,.highlight code{background:none;padding:0;color:inherit}
.codehilite pre,.highlight pre{background:none;margin:0;padding:0;border-radius:0}
/* code source accordions */
.ex-intro{color:var(--g700);font-size:.92rem;margin:.1rem 0 .9rem;line-height:1.7}
.ex-intro b{color:var(--p700)}
.srcblock{margin:.65rem 0;border:1px solid var(--g200);border-radius:10px;overflow:hidden;background:#fff}
.srcblock>summary{cursor:pointer;padding:.65rem 1rem;font-weight:700;background:var(--g50);user-select:none;font-size:.9rem}
.srcblock>summary:hover{background:var(--p100)}
.srcblock .highlight{margin:0;border-radius:0}
.muted{color:var(--g500)}
/* graph helpers */
.content pre.mermaid,.mermaid{background:#fff;color:var(--g800);border:1px solid var(--g200);border-radius:12px;padding:1rem;overflow-x:auto;margin:1.1rem 0;text-align:center}
.reco-list{display:flex;flex-wrap:wrap;gap:.5rem;list-style:none;margin:1rem 0;padding:0}
.reco-list li{background:var(--g50);border:1px solid var(--g200);border-radius:999px;padding:.25rem .7rem;font-size:.82rem}
.reco-list a{text-decoration:none;color:var(--g800);display:inline-flex;align-items:center;gap:.35rem}
/* prev/next */
.prevnext{display:flex;justify-content:space-between;gap:1rem;margin:1.6rem 0 .5rem}
.prevnext a{flex:1;text-decoration:none;background:#fff;border:1px solid var(--g200);border-radius:12px;padding:.85rem 1.1rem;font-weight:700;color:var(--g800);transition:border-color var(--t),transform var(--t);overflow-wrap:anywhere}
.prevnext a:hover{border-color:var(--p400);transform:translateY(-2px)}
.prevnext a:last-child{text-align:right}
.footer{text-align:center;padding:2.5rem 1rem;font-size:.85rem;color:var(--g500)}
/* ===== figures (inline SVG diagrams) ===== */
figure.lec-fig{margin:1.4rem 0;padding:1rem 1.1rem .9rem;background:var(--g50);border:1px solid var(--g200);border-radius:12px;overflow-x:auto}
figure.lec-fig svg{max-width:100%;height:auto;display:block;margin:0 auto}
figure.lec-fig figcaption{margin-top:.7rem;font-size:.84rem;color:var(--g600);line-height:1.7;text-align:center}
figure.lec-fig figcaption b,figure.lec-fig figcaption strong{color:var(--p700)}
/* ===== responsive ===== */
@media(max-width:1120px){.layout.has-left.has-right{grid-template-columns:230px minmax(0,1fr)}.side-right{display:none}}
@media(max-width:860px){.layout{grid-template-columns:1fr!important;padding:1rem 1rem 2.5rem}.side-left{display:none}.block{padding:1.1rem 1.15rem}}
@media(max-width:600px){.hero{padding:3.5rem 1rem 4rem}.cards{margin-top:-2rem;padding:0 1rem}.mod-banner{padding:1.3rem 1.2rem}.logo-text small{display:none}.home-link{font-size:15px}.topnav{gap:.15rem}.topnav a{font-size:11.5px;padding:6px 8px}.site-header{padding:0 12px}}
@media(max-width:360px){.home-link{font-size:14px}.logo-mark{width:26px;height:26px;font-size:12px}.topnav a{font-size:11px;padding:5px 6px}.topnav{gap:.1rem}}
"""


# ----------------------------------------------------------------------------- markdown / code
def md_to_html(text: str) -> tuple[str, str]:
    """Markdown を (本文HTML, 目次HTML) に変換する。"""
    m = markdown.Markdown(
        extensions=["fenced_code", "tables", "codehilite", "sane_lists", "attr_list", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
            "toc": {"toc_depth": "2-3"},
        },
    )
    body = m.convert(text)
    # FAQ: 「**Q. …**」と次行の「A. …」が同じ段落に連結され同じ行に出てしまうので、
    # Q の直後に改行(<br>)を入れて A. を次の行から始める。
    body = re.sub(r"(<strong>Q\d*\..*?</strong>)\n(A\d*\. )", r"\1<br>\2", body)
    return body, getattr(m, "toc", "") or ""


def highlight_py(code: str) -> str:
    from pygments import highlight
    from pygments.lexers import PythonLexer

    return highlight(code, PythonLexer(), HtmlFormatter(cssclass="highlight"))


def blockify(html_content: str) -> str:
    """h2 セクションごとに <section class="block"> で包み、カード状に見せる。"""
    segs = re.split(r"(?=<h2[ >])", html_content)
    return "\n".join(f'<section class="block">{s}</section>' for s in segs if s.strip())


def ordered_scripts(d: pathlib.Path) -> list[pathlib.Path]:
    pys = list(d.glob("*.py"))
    numbered = sorted(p for p in pys if p.name[:1].isdigit())
    ex = [p for p in pys if p.name == "exercises.py"]
    helpers = sorted(p for p in pys if not p.name[:1].isdigit() and p.name != "exercises.py")
    return numbered + helpers + ex


def _short(title: str) -> str:
    """ナビ/グラフ用の短縮タイトル。区切りは em-dash のみ（Cluster-CLIP のハイフンは保持）。"""
    return title.split("—")[0].strip()


def page(title: str, body: str, *, rel: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<link rel="icon" type="image/svg+xml" href="{rel}assets/favicon.svg">
<link rel="stylesheet" href="{rel}assets/site.css">
<link rel="stylesheet" href="{rel}assets/pygments.css">
</head>
<body>
<header class="site-header">
  <a class="home-link" href="{rel}index.html">
    <span class="logo-mark">CV</span>
    <span class="logo-text"><span class="logo-name">lecture-cv</span><small>Computer Vision を自分の力にする</small></span>
  </a>
  <nav class="topnav">
    <a href="{rel}graph.html">学習順序グラフ</a>
    <a href="{rel}roadmap.html">ロードマップ</a>
  </nav>
</header>
{body}
</body>
</html>
"""


def lay(main_html: str, left: str = "", right: str = "") -> str:
    cls = "layout" + (" has-left" if left else "") + (" has-right" if right else "")
    return f'<div class="{cls}">{left}<main class="col-main">{main_html}</main>{right}</div>'


# ----------------------------------------------------------------------------- load data
data = json.loads((ROOT / "docs" / "curriculum.json").read_text())
modules = data["modules"]
for m in modules:
    d = LECT / m["id"]
    m["_dir"] = d
    m["_scripts"] = ordered_scripts(d) if d.is_dir() else []
    m["_authored"] = bool(m["_scripts"])
    m["_num"] = m["id"][:2]
authored = [m for m in modules if m["_authored"]]
modmap = {m["id"]: m for m in modules}
tracks: dict[str, list] = {}
for m in modules:
    tracks.setdefault(m["track"], []).append(m)


def sidebar_left(current_id: str | None) -> str:
    out = ['<aside class="side-left"><nav class="modnav">', '<a class="modnav-home" href="index.html">🏠 トップ（全体目次）</a>']
    for track, ms in tracks.items():
        out.append(f'<div class="nav-track">{html.escape(track)}</div>')
        for m in ms:
            lv = LEVEL_CLASS.get(m["level"], "intro")
            active = " active" if m["id"] == current_id else ""
            out.append(
                f'<a class="nav-item{active}" href="{m["id"]}.html">'
                f'<span class="dot {lv}"></span><span class="nav-num">{m["_num"]}</span>'
                f'<span>{html.escape(_short(m["title"]))}</span></a>'
            )
    out.append("</nav></aside>")
    return "".join(out)


# ----------------------------------------------------------------------------- index page
def _render_card(m: dict) -> str:
    lv = LEVEL_CLASS.get(m["level"], "intro")
    status = '<span class="status done">公開</span>' if m["_authored"] else '<span class="status wip">準備中</span>'
    goal = html.escape((m.get("goal") or "")[:108]) + ("…" if len(m.get("goal") or "") > 108 else "")
    groups = "".join(f'<code class="chip">{html.escape(g)}</code>' for g in (m.get("needs_groups") or []))
    return (
        f'<a class="level-card" href="{m["id"]}.html">'
        f'<h3><span class="level-badge {lv}">{m["level"]}</span>'
        f'<span class="mod-num">{m["_num"]}</span> {html.escape(m["title"])}</h3>'
        f'<p>{goal}</p><div class="card-foot">{status}{groups}</div></a>'
    )


def _section(title: str, ms: list) -> str:
    items = "".join(_render_card(m) for m in ms)
    return (
        f'<section class="lang-section"><div class="lang-header">'
        f'<h2>{html.escape(title)}</h2></div>'
        f'<div class="lang-levels">{items}</div></section>'
    )


# 表示① ジャンル（トラック）別
track_sections = "".join(_section(track, ms) for track, ms in tracks.items())
# 表示② 難易度別（入門→初級→中級→上級、各回は番号順）
LEVEL_ORDER = ["入門", "初級", "中級", "上級"]
by_level: dict[str, list] = {lvl: [] for lvl in LEVEL_ORDER}
for m in modules:
    by_level.setdefault(m["level"], []).append(m)
level_sections = "".join(
    _section(f"{lvl}（{len(by_level[lvl])}回）", sorted(by_level[lvl], key=lambda x: x["_num"]))
    for lvl in LEVEL_ORDER if by_level.get(lvl)
)
hero = f"""<header class="hero">
  <h1>lecture-cv</h1>
  <p>Computer Vision を「AI の補助なしで自力で書ける」まで叩き込む全{len(modules)}回ハンズオン講座</p>
  <p class="hero-meta">公開 {len(authored)} / {len(modules)} 回　・　CPU のみで完走　・　各回 解説＋実行コード＋演習</p>
</header>"""
view_toggle = (
    '<div class="view-toggle" role="tablist">'
    '<button class="vt-btn active" data-view="track" type="button">ジャンル別</button>'
    '<button class="vt-btn" data-view="level" type="button">難易度別</button>'
    "</div>"
)
toggle_script = (
    '<script>(function(){var bs=document.querySelectorAll(".vt-btn");'
    'bs.forEach(function(b){b.addEventListener("click",function(){'
    'bs.forEach(function(x){x.classList.remove("active")});b.classList.add("active");'
    'var v=b.getAttribute("data-view");'
    'document.getElementById("view-track").hidden=(v!=="track");'
    'document.getElementById("view-level").hidden=(v!=="level");});});})();</script>'
)
index_body = (
    hero
    + '<main class="cards">'
    + view_toggle
    + f'<div id="view-track" class="cards-view">{track_sections}</div>'
    + f'<div id="view-level" class="cards-view" hidden>{level_sections}</div>'
    + "</main>"
    + '<footer class="footer">lecture-cv ／ 設計時点: 2026-06 ／ 各回フッターにライブラリ版を明記</footer>'
    + toggle_script
)
SITE.mkdir(exist_ok=True)
(SITE / "index.html").write_text(page("lecture-cv 教材", index_body))

# ----------------------------------------------------------------------------- roadmap page
roadmap_html, _ = md_to_html((ROOT / "docs" / "roadmap.md").read_text())
# ロードマップ表中の各モジュールID（`<code>00_setup</code>`）を各ページへのリンクにする
for _mid in modmap:
    roadmap_html = roadmap_html.replace(
        f"<code>{_mid}</code>", f'<a class="rm-link" href="{_mid}.html"><code>{_mid}</code></a>'
    )
roadmap_body = lay(f'<article class="content">{blockify(roadmap_html)}</article>', left=sidebar_left(None))
(SITE / "roadmap.html").write_text(page("ロードマップ — lecture-cv", roadmap_body))

# ----------------------------------------------------------------------------- graph page
_depth_memo: dict[str, int] = {}


def _depth(mid: str) -> int:
    if mid in _depth_memo:
        return _depth_memo[mid]
    ps = [p for p in modmap[mid].get("prereqs", []) if p in modmap]
    _depth_memo[mid] = 0 if not ps else 1 + max(_depth(p) for p in ps)
    return _depth_memo[mid]


reco = sorted(modules, key=lambda m: (_depth(m["id"]), int(m["id"][:2])))
mer = [
    "flowchart LR",
    "  classDef intro fill:#dcfce7,stroke:#16a34a,color:#14532d;",
    "  classDef beginner fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;",
    "  classDef intermediate fill:#fef3c7,stroke:#d97706,color:#7c2d12;",
    "  classDef advanced fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;",
]
for m in modules:
    num = m["id"][:2]
    mer.append(f'  n{num}["{num} {_short(m["title"])[:18]}"]:::{LEVEL_CLASS.get(m["level"], "intro")}')
    mer.append(f'  click n{num} "{m["id"]}.html"')
for m in modules:
    for p in m.get("prereqs", []):
        if p in modmap:
            mer.append(f'  n{p[:2]} --> n{m["id"][:2]}')
mermaid_txt = "\n".join(mer)
reco_items = "".join(
    f'<li><a href="{m["id"]}.html"><span class="level-badge {LEVEL_CLASS.get(m["level"], "intro")}">'
    f'{m["level"]}</span>{m["id"][:2]} {html.escape(_short(m["title"]))}</a></li>'
    for m in reco
)


def _prereq_links(m) -> str:
    ps = [p for p in m.get("prereqs", []) if p in modmap]
    return " ・ ".join(f'<a href="{p}.html">{p[:2]}</a>' for p in ps) or "—"


table_rows = "".join(
    f'<tr><td><a href="{m["id"]}.html">{m["id"]}</a></td>'
    f'<td>{html.escape(m["track"])}</td><td>{_prereq_links(m)}</td></tr>'
    for m in modules
)
graph_main = f"""<article class="content">
<h1>学習順序グラフ（前提マップ）</h1>
<p>この講座は番号順の一本道ではなく、<strong>前提（prerequisite）でつながった有向グラフ（DAG）</strong>です。番号は安定した ID にすぎず、実際に学ぶ順番は下のグラフが正です。あとから回を足しても、各回の「前提」をたどれば学習の筋道が崩れません。</p>
<h2>依存グラフ</h2>
<p class="muted">ノードをクリックすると各回へ移動します（色＝レベル：緑=入門・青=初級・橙=中級・赤=上級）。オフラインでは描画されないことがあります（その場合は下の推奨順・前提表を参照）。</p>
<pre class="mermaid">{html.escape(mermaid_txt)}</pre>
<h2>推奨学習順（前提を満たす一例・トポロジカル順）</h2>
<ol class="reco-list">{reco_items}</ol>
<h2>前提一覧</h2>
<table><thead><tr><th>モジュール</th><th>トラック</th><th>前提</th></tr></thead><tbody>{table_rows}</tbody></table>
</article>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true, theme: 'neutral', flowchart: {{ useMaxWidth: false }} }});
</script>"""
(SITE / "graph.html").write_text(page("学習順序グラフ — lecture-cv", lay(graph_main, left=sidebar_left(None))))


# ----------------------------------------------------------------------------- module pages
def strip_h1(md: str) -> str:
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip():
            return "\n".join(lines[i + 1:]) if ln.startswith("# ") else md
    return md


EXNAMES = {"exercises.py", "exercises_solutions.py"}
for idx, m in enumerate(modules):
    lv = LEVEL_CLASS.get(m["level"], "intro")
    prev_m = modules[idx - 1] if idx > 0 else None
    next_m = modules[idx + 1] if idx + 1 < len(modules) else None

    readme = m["_dir"] / "README.md"
    if readme.exists():
        readme_html, toc_html = md_to_html(strip_h1(readme.read_text()))
    else:
        readme_html, toc_html = "<p>（準備中）</p>", ""

    code_scripts = [p for p in m["_scripts"] if p.name not in EXNAMES]
    ex_files = sorted(
        (p for p in m["_scripts"] if p.name in EXNAMES),
        key=lambda p: p.name != "exercises.py",
    )
    scripts_html = ""
    if code_scripts:
        det = "".join(
            f'<details class="srcblock"><summary><code>{html.escape(p.name)}</code></summary>'
            f"{highlight_py(p.read_text())}</details>"
            for p in code_scripts
        )
        scripts_html = f'<h2 id="scripts">スクリプト全文</h2>{det}'
    ex_html = ""
    if ex_files:
        det = "".join(
            f'<details class="srcblock"><summary><code>{html.escape(p.name)}</code> — '
            f'{"TODO を埋めて自己採点" if p.name == "exercises.py" else "模範解答（先に自分で解く）"}'
            f"</summary>{highlight_py(p.read_text())}</details>"
            for p in ex_files
        )
        ex_html = (
            f'<h2 id="exercises">演習</h2>'
            f'<p class="ex-intro">演習は下の <code>exercises.py</code> に <b>TODO 形式</b>で入っています。'
            f'開いて TODO を自分で埋め、<code>uv run python lectures/{m["id"]}/exercises.py</code> '
            f'を実行すると<b>自己採点</b>されます（<code>SHOW_SOLUTION=1</code> または '
            f'<code>exercises_solutions.py</code> で答え合わせ）。</p>'
            f"{det}"
        )

    # 右サイドの目次に スクリプト/演習 を追記
    extra = []
    if scripts_html:
        extra.append('<li><a href="#scripts">スクリプト全文</a></li>')
    if ex_html:
        extra.append('<li><a href="#exercises">演習</a></li>')
    extra_toc = f'<ul class="toc-extra">{"".join(extra)}</ul>' if extra else ""
    right = f'<aside class="side-right"><div class="side-title">このページの目次</div>{toc_html}{extra_toc}</aside>'

    groups = "".join(f'<code class="chip">{html.escape(g)}</code>' for g in (m.get("needs_groups") or [])) or '<span class="muted">追加依存なし</span>'
    prereq_html = " ".join(
        f'<a href="{p}.html">{p[:2]} {html.escape(modmap[p]["title"][:14])}</a>'
        for p in m.get("prereqs", []) if p in modmap
    ) or '<span class="muted">なし（最初から学べる）</span>'
    meta_rows = (
        f'<tr><th>前提</th><td class="prereq-row">{prereq_html}</td></tr>'
        f'<tr><th>トラック</th><td>{html.escape(m["track"])}</td></tr>'
        f'<tr><th>レベル</th><td><span class="level-badge {lv}">{m["level"]}</span></td></tr>'
        f'<tr><th>依存グループ</th><td>{groups}</td></tr>'
        f'<tr><th>評価</th><td>{html.escape(m.get("evaluation") or "—")}</td></tr>'
        f'<tr><th>完成物</th><td>{html.escape(m.get("deliverable") or "—")}</td></tr>'
        f'<tr><th>状態</th><td>{"公開" if m["_authored"] else "準備中（プレースホルダ）"}</td></tr>'
    )
    banner = (
        f'<div class="mod-banner"><div class="mod-eyebrow">'
        f'<span class="level-badge {lv}">{m["level"]}</span> {html.escape(m["track"])} ／ 第{int(m["_num"])}回</div>'
        f'<h1>{html.escape(m["title"])}</h1>'
        f'<p class="mod-goal">🎯 {html.escape(m.get("goal") or "")}</p>'
        f'<table class="meta-table">{meta_rows}</table></div>'
    )
    nav = '<nav class="prevnext">'
    nav += f'<a href="{prev_m["id"]}.html">← {prev_m["_num"]} {html.escape(_short(prev_m["title"])[:22])}</a>' if prev_m else "<span></span>"
    nav += f'<a href="{next_m["id"]}.html">{next_m["_num"]} {html.escape(_short(next_m["title"])[:22])} →</a>' if next_m else "<span></span>"
    nav += "</nav>"

    main = banner + f'<article class="content">{blockify(readme_html + scripts_html + ex_html)}</article>' + nav
    body = lay(main, left=sidebar_left(m["id"]), right=right)
    (SITE / f'{m["id"]}.html').write_text(page(f'{m["_num"]} {m["title"]} — lecture-cv', body))

# ----------------------------------------------------------------------------- assets
ASSETS.mkdir(parents=True, exist_ok=True)
fmt = HtmlFormatter(style=PYGMENTS_STYLE)
(ASSETS / "pygments.css").write_text(
    fmt.get_style_defs(".codehilite") + "\n" + fmt.get_style_defs(".highlight")
)
(ASSETS / "site.css").write_text(SITE_CSS)
(ASSETS / "favicon.svg").write_text(FAVICON_SVG)

print(f"built site/: {len(modules)} module pages ({len(authored)} authored) + index + roadmap + graph")
