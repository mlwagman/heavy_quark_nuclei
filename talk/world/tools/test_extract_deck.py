"""Tests for extract_deck.py (ticket 02). Run: cd talk/world/tools && python3 -m pytest -q test_extract_deck.py"""

import base64
import hashlib
import re
from pathlib import Path

import pytest

from extract_deck import BLOBS, STRICT, main, split_deck

DECK = Path(__file__).resolve().parents[2] / "mrs-talk-animated.html"
real = pytest.mark.skipif(not DECK.exists(), reason="live Deck absent")
SCENE_FILES = [
    "s01-title", "s02-why-heavy-hadrons", "s03-pnrqcd", "s04-renormalon-problem", "s05-mrs", "s06-mu-r", "s07-qmc",
    "s08-quarkonium-tuning", "s09-mesons", "s10-baryons-vs-lattice", "s11-ratio-detmold", "s12-tetraquarks",
    "s13-what-did-mrs-buy", "s14-conclusions", "s15-references",
]

FONT = b"wOF2" + b"\x00" * 20
PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 20
MP3 = b"ID3" + b"\x02" * 20


def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def url(mime: str, b: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


MINI_NAMES = {sha12(FONT): "fonts/F.woff2", sha12(PNG): "assets/p.png", sha12(MP3): "assets/n.mp3"}


def mini_deck(scene2_first: str = "(function(){", engine_marker: bool = True, scene_nums: tuple = (1, 2)) -> str:
    s1, s2 = scene_nums
    lines = [
        '<!-- @dsCard group="Talk" -->',
        '<meta charset="utf-8">',
        "<title>t</title>",
        "<style>",
        f"  @font-face{{font-family:'P052';src:url({url('font/woff2', FONT)}) format('woff2');}}",
        "  body{margin:0}",
        "</style>",
        "",
        f'<div id="stage"><img src="{url("image/png", PNG)}" alt="qr"></div>',
        "",
        "<script>",
        "'use strict';",
        "const scenes=[];",
        f"/* ================= SCENE {s1} — title ================= */",
        "(function(){",
        "scenes.push({beats:[]});",
        "})();",
        "",
        f"/* ================= SCENE {s2} — mu'(r) / Two ================= */",
        scene2_first,
        f"const q='{url('image/png', PNG)}';",
        "})();",
        "",
    ]
    if engine_marker:
        lines.append("/* ================= engine ================= */")
    lines += [f"const AUD=new Audio('{url('audio/mpeg', MP3)}');", "</script>", ""]
    return "\n".join(lines)


def test_split_mini_deck():
    files = split_deck(mini_deck(), MINI_NAMES)
    assert sorted(files) == [
        "assets/n.mp3", "assets/p.png", "deck.css", "fonts/F.woff2", "index.html",
        "js/00-helpers.js", "js/99-engine.js", "js/s01-title.js", "js/s02-mu-r-two.js",
    ]
    assert (files["fonts/F.woff2"], files["assets/p.png"], files["assets/n.mp3"]) == (FONT, PNG, MP3)
    text = {k: v.decode() for k, v in files.items() if k.endswith((".html", ".css", ".js"))}
    assert all("base64" not in t for t in text.values())
    assert all(t.startswith("'use strict';\n") for k, t in text.items() if k.endswith(".js"))
    assert "url(fonts/F.woff2)" in text["deck.css"]
    assert "const q='assets/p.png';" in text["js/s02-mu-r-two.js"]
    assert "new Audio('assets/n.mp3')" in text["js/99-engine.js"]
    html = text["index.html"]
    assert "@dsCard" not in html and "<!doctype" not in html.lower()
    assert '<link rel="stylesheet" href="deck.css">' in html and 'src="assets/p.png"' in html
    order = [html.index(f'<script src="js/{n}"></script>') for n in ("00-helpers.js", "s01-title.js", "s02-mu-r-two.js", "99-engine.js")]
    assert order == sorted(order)


def test_unknown_blob_rejected():
    names = {k: v for k, v in MINI_NAMES.items() if v != "assets/p.png"}
    with pytest.raises(ValueError, match=f"sha12={sha12(PNG)}"):
        split_deck(mini_deck(), names)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"scene_nums": (1, 3)}, "scene numbers"),
        ({"engine_marker": False}, "engine marker"),
        ({"scene2_first": "(function(){ // x"}, "not a bare IIFE"),
    ],
)
def test_bad_markers_rejected(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        split_deck(mini_deck(**kwargs), MINI_NAMES)


@pytest.fixture(scope="module")
def real_files():
    return split_deck(DECK.read_text(encoding="utf-8"), BLOBS)


@real
def test_real_deck_layout(real_files):
    js = ["js/00-helpers.js", *(f"js/{s}.js" for s in SCENE_FILES), "js/99-engine.js"]
    assert set(real_files) == {"index.html", "deck.css", *BLOBS.values(), *js}
    assert len(real_files) == 28
    magic = {".woff2": b"wOF2", ".png": b"\x89PNG", ".mp3": b"ID3"}
    for path, data in real_files.items():
        for ext, head in magic.items():
            if path.endswith(ext):
                assert data.startswith(head), path
    text = {k: v.decode() for k, v in real_files.items() if not k.endswith(tuple(magic))}
    assert sum(len(v.encode()) for v in text.values()) < 120_000
    assert not [k for k, v in text.items() if "base64" in v or "<!doctype" in v.lower()]


@real
def test_real_deck_lossless(real_files):
    """Every Deck line (blobs swapped for their paths) lands in exactly one output, in Deck order."""
    deck = DECK.read_text(encoding="utf-8").split("\n")
    path_of = lambda m: BLOBS[sha12(base64.b64decode(m.group(1)))]  # noqa: E731
    wrappers = {0, deck.index("<style>"), deck.index("</style>"), deck.index("<script>"), deck.index("</script>")}
    expected = [re.sub(r"data:[a-z]+/[a-z0-9.+-]+;base64,([A-Za-z0-9+/=]+)", path_of, ln)
                for i, ln in enumerate(deck[:-1]) if i not in wrappers]

    def lines(p: str) -> list[str]:
        return real_files[p].decode().split("\n")[:-1]

    html = lines("index.html")
    link, tag0 = html.index('<link rel="stylesheet" href="deck.css">'), next(i for i, ln in enumerate(html) if ln.startswith("<script src="))
    tags = [ln.removeprefix('<script src="').removesuffix('"></script>') for ln in html[tag0:]]  # KeyError below if malformed
    js = [ln for p in tags for ln in (lines(p) if p == "js/00-helpers.js" else lines(p)[1:])]
    assert all(lines(p)[0] == STRICT for p in tags)
    assert html[1:link] + lines("deck.css") + html[link + 1 : tag0] + js == expected


def test_sha_pin(tmp_path, capsys):
    bad = tmp_path / "deck.html"
    bad.write_text(mini_deck(), encoding="utf-8")
    assert main(["--deck", str(bad), "--out", str(tmp_path / "w")]) == 1
    assert "sha256 mismatch" in capsys.readouterr().out
    assert not (tmp_path / "w").exists()


@pytest.fixture
def world(tmp_path, capsys):
    out = tmp_path / "world"
    assert main(["--out", str(out)]) == 0
    assert "EXTRACT OK written=28 same=0 files=28" in capsys.readouterr().out
    return out


@real
def test_rerun_is_noop(world, capsys):
    assert main(["--out", str(world)]) == 0
    assert "written=0 same=28" in capsys.readouterr().out


@real
def test_restores_only_missing(world, capsys):
    (world / "fonts/P052-Italic.woff2").unlink()
    assert main(["--out", str(world)]) == 0
    assert "written=1 same=27" in capsys.readouterr().out
    assert sha12((world / "fonts/P052-Italic.woff2").read_bytes()) == "aeb2af590263"


@pytest.fixture
def edited(world):
    """An edited scene file (ticket 08's job) next to a missing MP3 (a fresh checkout)."""
    js = world / "js/s14-conclusions.js"
    js.write_text(js.read_text(encoding="utf-8").replace("Thank you", "Merci"), encoding="utf-8")
    (world / "assets/narration.mp3").unlink()
    return world, js, js.read_bytes()


@real
def test_refuses_differing_atomically(edited, capsys):
    world, js, mine = edited
    assert main(["--out", str(world)]) == 1
    assert "js/s14-conclusions.js" in capsys.readouterr().out
    assert js.read_bytes() == mine and not (world / "assets/narration.mp3").exists()
    assert main(["--out", str(world), "--force"]) == 0
    assert "Merci" not in js.read_text(encoding="utf-8") and (world / "assets/narration.mp3").exists()


@real
def test_only_scopes(edited, capsys):
    world, js, mine = edited
    assert main(["--out", str(world), "--only", "assets/narration.mp3"]) == 0
    assert "written=1 same=0 files=1" in capsys.readouterr().out
    assert js.read_bytes() == mine and sha12((world / "assets/narration.mp3").read_bytes()) == "227989c294ec"


@real
def test_only_unknown_path(world, capsys):
    assert main(["--out", str(world), "--only", "assets/nope.mp3"]) == 2
    assert "assets/nope.mp3" in capsys.readouterr().out
