#!/usr/bin/env python3
"""Build talk/mrs-talk-practice.html: the World as one self-contained file for practising on a phone (round 22).

    python3 talk/world3d/tools/build-practice.py

Every script, the stylesheet, the fonts and the figures are inlined (data URIs), so the file opens offline from a phone's
Downloads with no folder next to it; it starts in practice mode (keynotes under the World, tap left third = back).
"""
import base64
import hashlib
import re
from pathlib import Path

W = Path(__file__).resolve().parents[1]
OUT = W.parent / "mrs-talk-practice.html"
MIME = {".png": "image/png", ".woff2": "font/woff2"}


def data_uri(p: Path) -> str:
    return f"data:{MIME[p.suffix]};base64,{base64.b64encode(p.read_bytes()).decode()}"


def src_hash() -> str:
    """sha256 of index.html, css/* and js/* in sorted path order: tools/check-practice.mjs recomputes it to see a stale build."""
    h = hashlib.sha256()
    for f in sorted(["index.html"] + [f"css/{p.name}" for p in (W / "css").iterdir()] + [f"js/{p.name}" for p in (W / "js").iterdir()]):
        h.update((W / f).read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    html = (W / "index.html").read_text(encoding="utf-8")
    css = (W / "css/world.css").read_text(encoding="utf-8")
    css = re.sub(r"url\('\.\./(fonts/[^']+)'\)", lambda m: f"url('{data_uri(W / m.group(1))}')", css)
    html = html.replace('<link rel="stylesheet" href="css/world.css">', f"<style>{css}</style>")

    def script(m: re.Match) -> str:
        js = (W / m.group(1)).read_text(encoding="utf-8")
        js = re.sub(r"assets/[\w.-]+\.png", lambda a: data_uri(W / a.group(0)), js)
        return "<script>" + js.replace("</script", "<\\/script") + "</script>"

    html = re.sub(r'<script src="([^"]+)"></script>', script, html)
    html = html.replace("<head>", f'<head>\n<meta name="src-hash" content="{src_hash()}">\n<script>window.PRACTICE = 1;</script>', 1)
    left = re.findall(r'(?:src|href)="(?!data:|\$\{)[^"#]+"', html) + re.findall(r"assets/[\w.-]+\.png|\.\./fonts/", html)
    assert not left, f"still linked: {left[:5]}"
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
