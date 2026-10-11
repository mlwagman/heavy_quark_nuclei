#!/usr/bin/env python3
"""Build talk/mrs-talk-world3d.zip: the talk as one folder, ready for Windows (round 24).

    python3 talk/world3d/tools/build-practice.py && python3 talk/world3d/tools/build-zip.py
    NODE_PATH=$(npm root -g) node talk/world3d/tools/check-zip.mjs

Everything sits in one folder, mrs-talk-world3d/, so "Extract all" on Windows gives a folder and not loose files; the README
has Windows line ends (CRLF) for Notepad. Inside: index.html, presenter.html, css, js, vendor, fonts, assets, the phone file.
"""
import zipfile
from pathlib import Path

W = Path(__file__).resolve().parents[1]
OUT = W.parent / "mrs-talk-world3d.zip"
TOP = "mrs-talk-world3d"
README = """MRS talk: the 3D World (offline, no install, no internet)

On Windows
1. Right-click mrs-talk-world3d.zip, choose "Extract all...", then "Extract".
   Do not open the pages from inside the zip: the browser cannot load the folders from there.
2. Open the extracted folder mrs-talk-world3d and double-click presenter.html.
   Chrome or Edge recommended; Firefox works too.
3. Click "Open the World". Move that window to the projector and press F for full screen.
4. Drive the talk from the presenter window or a clicker:
   Right / Space / PageDown = forward, Left / PageUp = back.
   In the World: O = overview map, N = full narration notes.
Without the presenter: double-click index.html.

Keep the folders (css, js, vendor, fonts, assets) next to the two .html files.

Practising on a phone: copy mrs-talk-practice.html (one file, no folder) to the phone and open it.
Tap the right two thirds of the screen = next, the left third = back.
"""


def main() -> None:
    files = [W / "index.html", W / "presenter.html"]
    for d in ("css", "js", "vendor", "fonts", "assets"):
        files += sorted(p for p in (W / d).rglob("*") if p.is_file())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{TOP}/README.txt", README.replace("\n", "\r\n"))
        for p in files:
            z.write(p, f"{TOP}/{p.relative_to(W).as_posix()}")
        z.write(W.parent / "mrs-talk-practice.html", f"{TOP}/mrs-talk-practice.html")
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(files) + 2} files)")


if __name__ == "__main__":
    main()
