"""Split the live MRS Deck (one 9.9 MB HTML file) into the editable talk/world/ source folder.

    python3 talk/world/tools/extract_deck.py                               # extract everything
    python3 talk/world/tools/extract_deck.py --only assets/narration.mp3   # restore the gitignored MP3

index.html + deck.css + one classic script per scene (js/) + the base64 blobs as real files (fonts/, assets/).
Every Deck line lands verbatim in exactly one output; only data URLs become relative paths. The scripts stay
classic <script src> files, not ES modules: they share one global lexical scope, so the bare engine globals
(scenes, show, applyUpTo, si, ...) that render/deck_shim.mjs drives survive, and the page still opens over file://.
Writes are all-or-nothing: absent files are written, identical ones skipped, and a file whose bytes differ
(an edited source file) stops the whole run unless --force.
"""

import argparse
import base64
import hashlib
import re
import sys
from pathlib import Path

DECK_SHA256 = "ad85ff7e2cf5a7d4b9566aed1ccbb7e5a78e1244c82633818c3d4cd83795ff98"  # talk/mrs-talk-animated.html @ 5d53f53
BLOBS = {  # sha256[:12] of the decoded bytes -> path under talk/world/
    "3c0c54927816": "fonts/P052-Regular.woff2",
    "aeb2af590263": "fonts/P052-Italic.woff2",
    "6ac694ec91a6": "assets/qr.png",
    "48fae48a9ebb": "assets/s08-fixed-order.png",
    "d5578281540f": "assets/s08-mrs.png",
    "083974b1f1e6": "assets/s11-fixed-order.png",
    "1ee79368b03d": "assets/s11-mrs.png",
    "bc8c0aed71a2": "assets/s12-figure.png",
    "227989c294ec": "assets/narration.mp3",
}
MARKER = re.compile(r"^/\* ={17} (?:SCENE (\d+) — (.+?)|engine) ={17} \*/$")
DATA_URL = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,([A-Za-z0-9+/=]+)")
STRICT = "'use strict';"
PROVENANCE = f"<!-- extracted from talk/mrs-talk-animated.html @ 5d53f53 sha256 {DECK_SHA256[:8]}… by tools/extract_deck.py -->"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def split_deck(text: str, names: dict[str, str]) -> dict[str, bytes]:
    """Deck text -> {relpath: bytes} for every output file. Pure; raises ValueError on any unexpected shape."""
    lines = text.split("\n")
    blobs: dict[str, bytes] = {}

    def unblob(chunk: list[str]) -> str:
        def sub(m: re.Match) -> str:
            raw = base64.b64decode(m.group(1))
            h = hashlib.sha256(raw).hexdigest()[:12]
            if h not in names:
                raise ValueError(f"unknown blob sha12={h} ({len(raw)} bytes): add it to BLOBS")
            blobs[names[h]] = raw
            return names[h]

        return DATA_URL.sub(sub, "\n".join(chunk) + "\n")

    style, style_end = lines.index("<style>"), lines.index("</style>")
    script, script_end = lines.index("<script>"), lines.index("</script>")
    if lines[script + 1] != STRICT:
        raise ValueError(f"script must open with {STRICT} (the whole inline script is strict)")
    marks = [(i, m) for i, ln in enumerate(lines) if script < i < script_end and (m := MARKER.match(ln))]
    scenes = [(i, m) for i, m in marks if m.group(1)]
    if not marks or marks[-1][1].group(1) or len(scenes) != len(marks) - 1:
        raise ValueError("expected SCENE markers followed by exactly one final engine marker")
    if [int(m.group(1)) for _, m in scenes] != list(range(1, len(scenes) + 1)):
        raise ValueError(f"scene numbers not 1..N: {[m.group(1) for _, m in scenes]}")

    bounds = [i for i, _ in marks] + [script_end]
    js = {"js/00-helpers.js": unblob(lines[script + 1 : bounds[0]])}
    for (i, m), end in zip(scenes, bounds[1:]):
        body = lines[i:end]
        while body and not body[-1].strip():
            body = body[:-1]
        if body[1] != "(function(){" or body[-1] != "})();":
            raise ValueError(f"SCENE {m.group(1)} is not a bare IIFE")
        js[f"js/s{int(m.group(1)):02d}-{slug(m.group(2))}.js"] = unblob([STRICT, *lines[i:end]])
    js["js/99-engine.js"] = unblob([STRICT, *lines[bounds[-2] : script_end]])

    # No DOCTYPE: the Deck has none, so it renders in quirks mode; adding one would change its layout.
    html = [PROVENANCE, *lines[1:style], '<link rel="stylesheet" href="deck.css">', *lines[style_end + 1 : script]]
    html += [f'<script src="{p}"></script>' for p in js]
    files = {"index.html": unblob(html), "deck.css": unblob(lines[style + 1 : style_end])}
    files = {k: v.encode() for k, v in {**files, **js}.items()}
    return {**files, **blobs}


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, default=here.parents[2] / "mrs-talk-animated.html")
    ap.add_argument("--out", type=Path, default=here.parents[1])
    ap.add_argument("--only", help="comma-separated paths under --out, e.g. assets/narration.mp3")
    ap.add_argument("--force", action="store_true", help="overwrite files whose bytes differ (edited sources)")
    a = ap.parse_args(argv)

    raw = a.deck.read_bytes()
    if (h := hashlib.sha256(raw).hexdigest()) != DECK_SHA256:
        print(f"sha256 mismatch: {a.deck} is {h[:12]}…, want {DECK_SHA256[:12]}… (commit 5d53f53)")
        return 1
    files = split_deck(raw.decode("utf-8"), BLOBS)
    if a.only:
        only = a.only.split(",")
        if unknown := [p for p in only if p not in files]:
            print(f"unknown --only path(s): {' '.join(unknown)}; known: {' '.join(sorted(files))}")
            return 2
        files = {p: files[p] for p in only}
    same = [p for p, b in files.items() if (a.out / p).is_file() and (a.out / p).read_bytes() == b]
    todo = [p for p in files if p not in same]
    if (differ := [p for p in todo if (a.out / p).exists()]) and not a.force:
        print(f"refusing: these differ from the Deck (edited?), nothing written; --force overwrites: {' '.join(differ)}")
        return 1
    for p in todo:
        (a.out / p).parent.mkdir(parents=True, exist_ok=True)
        (a.out / p).write_bytes(files[p])
    print(f"EXTRACT OK written={len(todo)} same={len(same)} files={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
