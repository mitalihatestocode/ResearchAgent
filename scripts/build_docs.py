"""
Builds the submission documents (PDF + Word) from the Markdown sources:

    docs/REPORT.md       -> docs/submission/ResearchAgent_Short_Report.{pdf,docx}
    docs/TEST_REPORT.md  -> docs/submission/ResearchAgent_Test_Report.{pdf,docx}

Each gets a cover page with the group's details. Usage:

    python scripts/build_docs.py --members "Name One (Roll 01)" "Name Two (Roll 02)" \
        --institute "College name" --course "Course / Class"

PDF is printed by headless Google Chrome; .docx is converted by LibreOffice
(soffice). Both must be installed.
"""

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "submission")
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "",
]
REPO_URL = "https://github.com/mitalihatestocode/ResearchAgent"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1c1c1e; }
h1 { font-size: 20pt; margin: 0 0 6pt; letter-spacing: -0.3px; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; color: #0b3d91; border-bottom: 1px solid #d0d7e2; padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.8pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #c8ccd4; padding: 4pt 5pt; vertical-align: top; text-align: left; }
th { background: #eef2f8; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f2f2f7; padding: 0 2pt; border-radius: 3px; }
pre { background: #f2f2f7; padding: 8pt; border-radius: 6px; white-space: pre-wrap; font-size: 8.5pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; page-break-inside: avoid; }
td img { max-width: 100%; }
blockquote { border-left: 3px solid #007aff; margin: 6pt 0; padding: 2pt 10pt; color: #444; }
a { color: #0b57d0; text-decoration: none; }
.cover { height: 247mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; text-align: center; }
.cover .kicker { color: #007aff; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; font-size: 10pt; }
.cover h1 { font-size: 26pt; margin: 10pt 0; }
.cover .sub { font-size: 13pt; color: #3a3a3c; margin-bottom: 28pt; }
.cover table { width: 70%; margin: 0 auto; font-size: 10.5pt; }
.cover td { border: none; border-bottom: 1px solid #e5e5ea; padding: 6pt; }
.cover td:first-child { color: #6e6e73; width: 35%; }
.cover img { width: 78%; margin: 22pt auto 0; border: 1px solid #e5e5ea; }
"""


def cover_html(title: str, subtitle: str, args) -> str:
    rows = [
        ("Activity", "FA Activity: Agentic AI Application Implementation"),
        ("Domain", "Research Paper Analysis Agent"),
        ("Group members", "<br>".join(html.escape(m) for m in args.members)),
        ("Institute", html.escape(args.institute)),
        ("Course", html.escape(args.course)),
        ("Source code", f'<a href="{REPO_URL}">{REPO_URL}</a>'),
        ("Date", date.today().strftime("%d %B %Y")),
    ]
    table = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    arch = os.path.join(DOCS, "architecture.png")
    return (f'<div class="cover"><div class="kicker">Agentic AI Application</div><h1>{html.escape(title)}</h1>'
            f'<div class="sub">{html.escape(subtitle)}</div><table>{table}</table>'
            f'<img src="file://{arch}"></div>')


def _blank_line_before_lists(text: str) -> str:
    """Python-Markdown only starts a list after a blank line; GitHub doesn't need one."""
    out, in_code = [], False
    list_item = re.compile(r"^\s*([-*+]|\d+\.)\s+")
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if (not in_code and list_item.match(line) and out and out[-1].strip()
                and not list_item.match(out[-1]) and not out[-1].startswith((" ", "\t", "|"))):
            out.append("")
        out.append(line)
    return "\n".join(out)


def md_to_html(md_path: str) -> str:
    with open(md_path) as f:
        text = f.read()
    text = re.sub(r"^# .*\n", "", text, count=1)  # title moves to the cover page
    # The metadata block before the first horizontal rule duplicates the cover page
    head, sep, rest = text.partition("\n---\n")
    if sep and len(head) < 800:
        text = rest
    text = _blank_line_before_lists(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    base = os.path.dirname(md_path)

    def absolutize(match):
        attr, url = match.group(1), match.group(2)
        if re.match(r"^(https?:|mailto:|#|file:)", url):
            return match.group(0)
        path = os.path.normpath(os.path.join(base, url))
        if attr == "href" and not url.lower().endswith((".png", ".jpg", ".svg")):
            rel = os.path.relpath(path, ROOT)
            return f'{attr}="{REPO_URL}/blob/main/{rel}"'  # links to repo files open on GitHub
        return f'{attr}="file://{path}"'

    return re.sub(r'(src|href)="([^"]+)"', absolutize, body)


PAGE_WIDTH_PX = 600   # usable A4 width at 96 dpi with our margins
CELL_WIDTH_PX = 290   # half-width table cells (screenshot grid)
FONT = "font-family: Arial, Helvetica, sans-serif;"


def _word_friendly(page: str) -> str:
    from PIL import Image

    def size_img(match):
        tag, src = match.group(0), match.group(1)
        path = src.replace("file://", "")
        try:
            width, height = Image.open(path).size
        except OSError:
            return tag
        in_table = page.rfind("<td", 0, match.start()) > page.rfind("</td>", 0, match.start())
        target = CELL_WIDTH_PX if in_table else (440 if "architecture" in path and 'class="cover"' in page[:match.start()] and page.find("</div>", match.start()) < page.find("<h2") else PAGE_WIDTH_PX)
        target = min(target, width)
        return f'<img src="{src}" width="{target}" height="{round(height * target / width)}">'

    page = re.sub(r'<img\b[^>]*?\bsrc="([^"]+)"[^>]*>', size_img, page)
    page = page.replace("<body>", f'<body style="{FONT} font-size:10.5pt">')
    # Tables must fit the page width (LibreOffice ignores CSS widths) and use small type
    page = re.sub(r"<table>", '<table width="100%" border="1" cellpadding="3" cellspacing="0" style="font-size:8pt">', page)
    for tag in ("p", "li", "td", "th", "h1", "h2", "h3"):
        page = re.sub(rf"<{tag}(\s|>)", rf'<{tag} style="{FONT}"\1', page)
    return page


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if path and os.path.exists(path):
            return path
    sys.exit("Google Chrome/Chromium not found (needed for PDF export).")


def build(md_name: str, out_name: str, title: str, subtitle: str, args, workdir: str):
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{cover_html(title, subtitle, args)}"
            f"{md_to_html(os.path.join(DOCS, md_name))}</body></html>")
    html_path = os.path.join(workdir, out_name + ".html")
    with open(html_path, "w") as f:
        f.write(page)

    pdf_path = os.path.join(OUT, out_name + ".pdf")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
                   check=True, capture_output=True, timeout=120)

    # Word: LibreOffice ignores CSS image sizing and most fonts, so give it explicit
    # image dimensions (fit to the page, or to a table cell) and inline font styles.
    docx_html_path = os.path.join(workdir, out_name + ".docx.html")
    with open(docx_html_path, "w") as f:
        f.write(_word_friendly(page))
    soffice = shutil.which("soffice") or sys.exit("LibreOffice (soffice) not found (needed for .docx export).")
    subprocess.run([soffice, "--headless", "--convert-to", "docx:MS Word 2007 XML", "--outdir", workdir, docx_html_path],
                   check=True, capture_output=True, timeout=180)
    shutil.move(os.path.join(workdir, out_name + ".docx.docx"), os.path.join(OUT, out_name + ".docx"))
    print(f"built {pdf_path}\n      {os.path.join(OUT, out_name + '.docx')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=["[Student name 1 - roll no.]", "[Student name 2 - roll no.]"])
    parser.add_argument("--institute", default="[Institute name]")
    parser.add_argument("--course", default="[Course / class / division]")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        build("REPORT.md", "ResearchAgent_Short_Report",
              "ResearchAgent", "An Agentic AI System for Research Paper Analysis, Comparison and Research Gap Identification",
              args, workdir)
        build("TEST_REPORT.md", "ResearchAgent_Test_Report",
              "ResearchAgent: Test Report", "Automated, live-LLM, security and evaluation testing", args, workdir)


if __name__ == "__main__":
    main()
