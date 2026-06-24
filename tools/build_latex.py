"""lecture-cv 全教材を 1 つの LaTeX/PDF 資料にまとめるビルダー。

Markdown(README/docs) を pandoc で LaTeX 化し、各 README に埋め込まれたインライン SVG 図は
cairosvg で PDF 化して \\includegraphics に差し替える。日本語は xelatex + IPAex フォントで組む。

前提ツール（この環境で確認済み）:
    - .tools/pandoc-3.10/bin/pandoc（静的バイナリ）
    - xelatex（TeX Live）/ IPAexMincho・IPAexGothic・IPAGothic フォント
    - cairosvg（uv の一時環境で供給する）

使い方:
    uv run --with cairosvg python tools/build_latex.py            # 全教材
    uv run --with cairosvg python tools/build_latex.py --only 00_setup   # 1章だけ（検証用）
    uv run --with cairosvg python tools/build_latex.py --no-pdf   # .tex だけ（PDF は作らない）

出力: build/latex/lecture-cv.tex / lecture-cv.pdf / figs/*.pdf
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import subprocess
import sys

import cairosvg

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "latex"
FIGS = OUT / "figs"
PANDOC = ROOT / ".tools" / "pandoc-3.10" / "bin" / "pandoc"

# 前付け（docs）→ 本編（lectures 番号順）の順で 1 冊にする
FRONT_DOCS = [
    ROOT / "docs" / "getting-started.md",
    ROOT / "docs" / "docker-basics.md",
    ROOT / "docs" / "roadmap.md",
]

# 見出し等の絵文字は IPA フォントに字形が無く豆腐になるので除去する。
# 矢印(→←↓↑)・〜・○×・罫線などは IPA にあるので残す。
EMOJI_RE = re.compile("[\U0001f000-\U0001faff]|[✅❓⚠▶✍⬇⬆️]")

FIGURE_RE = re.compile(r'<figure class="lec-fig">(.*?)</figure>', re.DOTALL)
SVG_RE = re.compile(r"<svg.*?</svg>", re.DOTALL)
FIGCAP_RE = re.compile(r"<figcaption>(.*?)</figcaption>", re.DOTALL)
# SVG 内の Web フォント指定を IPAexGothic に寄せる（cairosvg が日本語を確実に描けるように）
FONTFAMILY_RE = re.compile(r'font-family="[^"]*"')


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)


def caption_to_latex(cap_html: str) -> str:
    """figcaption の HTML(<b>/<code>) を LaTeX に変換（特殊文字をエスケープ）。"""
    cap_html = html.unescape(cap_html)
    out: list[str] = []
    # <b>..</b> と <code>..</code> を順に処理。残りの素テキストはエスケープ。
    token = re.compile(r"<(b|code)>(.*?)</\1>", re.DOTALL)
    pos = 0
    for m in token.finditer(cap_html):
        out.append(latex_escape(re.sub(r"<[^>]+>", "", cap_html[pos : m.start()])))
        inner = re.sub(r"<[^>]+>", "", m.group(2))
        if m.group(1) == "b":
            out.append(r"\textbf{" + latex_escape(inner) + "}")
        else:
            out.append(r"\texttt{" + latex_escape(inner) + "}")
        pos = m.end()
    out.append(latex_escape(re.sub(r"<[^>]+>", "", cap_html[pos:])))
    return "".join(out).strip()


def process_markdown(md: str, slug: str) -> str:
    """1 ファイル分の Markdown を整形（図の差し替え・絵文字除去・タスクリスト平滑化）。"""
    fig_n = 0

    def repl_fig(m: re.Match) -> str:
        nonlocal fig_n
        block = m.group(1)
        svg_m = SVG_RE.search(block)
        if not svg_m:
            return ""
        fig_n += 1
        svg = FONTFAMILY_RE.sub('font-family="IPAexGothic, sans-serif"', svg_m.group(0))
        name = f"{slug}_{fig_n:02d}.pdf"
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(FIGS / name))
        cap_m = FIGCAP_RE.search(block)
        cap = caption_to_latex(cap_m.group(1)) if cap_m else ""
        # 生 LaTeX の figure をそのまま埋め込む（pandoc は raw LaTeX を通す）
        return (
            "\n\n\\begin{figure}[H]\\centering"
            f"\\includegraphics[width=0.85\\linewidth,height=0.42\\textheight,keepaspectratio]{{figs/{name}}}"
            f"\\caption{{{cap}}}\\end{{figure}}\n\n"
        )

    md = FIGURE_RE.sub(repl_fig, md)
    md = EMOJI_RE.sub("", md)
    md = re.sub(r"^(\s*[-*]) \[[ xX]\] ", r"\1 ", md, flags=re.MULTILINE)  # タスクリスト→普通の箇条書き
    return md


def collect_sources(only: list[str] | None) -> list[pathlib.Path]:
    lectures = sorted((ROOT / "lectures").glob("*/README.md"))
    if only:
        sel = set(only)
        lectures = [p for p in lectures if p.parent.name in sel]
        front = [p for p in FRONT_DOCS if p.stem in sel or p.parent.name in sel]
        return front + lectures
    return [p for p in FRONT_DOCS if p.exists()] + lectures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="対象を slug で限定（例: 00_setup 01_image_basics）")
    ap.add_argument("--no-pdf", action="store_true", help=".tex だけ生成（PDF を作らない）")
    args = ap.parse_args()

    if not PANDOC.exists():
        sys.exit(f"pandoc が見つかりません: {PANDOC}（README の手順で取得してください）")
    FIGS.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(args.only)
    parts: list[str] = []
    for src in sources:
        slug = src.parent.name if src.parent.name != "docs" else src.stem
        parts.append(process_markdown(src.read_text(encoding="utf-8"), slug))
    combined = "\n\n\\newpage\n\n".join(parts)
    combined_md = OUT / "combined.md"
    combined_md.write_text(combined, encoding="utf-8")

    header = OUT / "header.tex"
    header.write_text(
        "\\usepackage{graphicx}\n"  # 生 LaTeX の \includegraphics 用（pandoc は raw だと自動ロードしない）
        "\\usepackage{float}\n"
        "\\usepackage{fvextra}\n"
        "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\\\\{\\}}\n"
        "\\setlength{\\emergencystretch}{3em}\n",
        encoding="utf-8",
    )

    common = [
        str(PANDOC), str(combined_md),
        "--from", "markdown+raw_tex-implicit_figures",
        "--top-level-division=chapter",
        "--toc", "--toc-depth=2", "-N",
        "-V", "documentclass=report",
        "-V", "geometry:margin=20mm",
        "-V", "mainfont=IPAexMincho", "-V", "sansfont=IPAexGothic", "-V", "monofont=IPAGothic",
        "-V", "CJKmainfont=IPAexMincho", "-V", "CJKsansfont=IPAexGothic", "-V", "CJKmonofont=IPAGothic",
        "-V", "title=lecture-cv — Computer Vision ハンズオン全教材",
        "-V", "date=2026-06",
        "-V", "colorlinks=true", "-V", "linkcolor=NavyBlue", "-V", "toccolor=NavyBlue",
        "-H", str(header),
        "--pdf-engine=xelatex",
    ]
    out_tex = OUT / "lecture-cv.tex"
    subprocess.run(common + ["-s", "-o", str(out_tex)], check=True)
    print(f"[tex] {out_tex}")
    if not args.no_pdf:
        # PDF 化は figs/ のある出力ディレクトリで自前 xelatex（pandoc の一時dirだと相対 figs/ を見失う）。
        # TOC・章番号・相互参照の解決のため 2 パス回す。
        for i in (1, 2):
            r = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "lecture-cv.tex"],
                cwd=str(OUT), capture_output=True, text=True,
            )
            if r.returncode != 0:
                tail = "\n".join(r.stdout.splitlines()[-25:])
                sys.exit(f"xelatex 失敗 (pass {i}):\n{tail}")
        print(f"[pdf] {OUT / 'lecture-cv.pdf'}")


if __name__ == "__main__":
    main()
