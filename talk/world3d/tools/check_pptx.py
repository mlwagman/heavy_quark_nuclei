#!/usr/bin/env python3
"""Check talk/mrs-talk-world3d.pptx (round 22): one slide per beat that plays its beat's animation by itself.

    python3 talk/world3d/tools/check_pptx.py

FAIL unless: 75 slides at 16:9; on each, one video filling the slide, an H.264 clip as long as the beat's animation
(schedule.json tween + 0.3 s, within 0.25 s), set to start by itself when the slide opens (onBegin, playFrom) and not to
rewind; the slide's speaker notes carry every keynote of that beat (js/keys.js); a clip starts on the frame where the one
before it ended (sampled, mean grey difference under 4); and LibreOffice opens the file (75 pages in a PDF export).
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.util import Emu

W = Path(__file__).resolve().parents[1]
PPTX = W.parent / "mrs-talk-world3d.pptx"
SCHED = Path("/home/gofer/mrs_talk_video/world3d/schedule.json")
FF = "/tmp/w3d-bin/ffmpeg" if Path("/tmp/w3d-bin/ffmpeg").exists() else "ffmpeg"


def keys() -> list[list[dict]]:
    js = (W / "js/keys.js").read_text(encoding="utf-8")
    out = subprocess.run(["node", "-e", js + "\nprocess.stdout.write(JSON.stringify(KEYS))"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def frame(clip: Path, last: bool) -> np.ndarray:
    args = [FF, "-v", "error"] + (["-sseof", "-0.1"] if last else []) + ["-i", str(clip), "-frames:v", "1", "-vf", "scale=480:270,format=gray", "-f", "rawvideo", "-"]
    return np.frombuffer(subprocess.run(args, capture_output=True, check=True).stdout, np.uint8).astype(float)


def main() -> int:
    fails: list[str] = []
    if not PPTX.exists():
        print(f"FAIL {PPTX} missing")
        return 1
    prs = Presentation(str(PPTX))
    clicks = json.loads(SCHED.read_text())["clicks"]
    K = [b for sc in keys() for b in sc]
    if len(prs.slides) != 75 or (prs.slide_width, prs.slide_height) != (Emu(12192000), Emu(6858000)):
        fails.append(f"{len(prs.slides)} slides of {prs.slide_width}x{prs.slide_height} EMU, want 75 at 16:9")
    tmp = Path(tempfile.mkdtemp())
    clips = []
    for i, (slide, c) in enumerate(zip(prs.slides, clicks)):
        tag = c["id"]
        vids = [s for s in slide.shapes if s.element.xpath(".//a:videoFile")]
        if len(vids) != 1 or (vids[0].left, vids[0].top, vids[0].width, vids[0].height) != (0, 0, prs.slide_width, prs.slide_height):
            fails.append(f"{tag}: {len(vids)} videos, or not filling the slide")
            continue
        rid = vids[0].element.xpath(".//a:videoFile/@r:link")
        part = slide.part.related_part(rid[0])
        clip = tmp / f"{i:02d}.mp4"
        clip.write_bytes(part.blob)
        clips.append(clip)
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name,duration", "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout.strip().split(",")
        want = c["tween"] + 0.3
        if p[0] != "h264" or abs(float(p[1]) - want) > 0.25:
            fails.append(f"{tag}: clip {p}, want h264 of {want:.2f} s")
        t = slide.element.xpath("./p:timing")
        x = t[0].xml if t else ""
        if not (re.search(r'evt="onBegin"', x) and "playFrom(" in x and 'nodeType="afterEffect"' in x) or "rewind" in x:
            fails.append(f"{tag}: the video does not start by itself when the slide opens")
        tf = slide.notes_slide.notes_text_frame if slide.has_notes_slide else None
        notes = tf.text if tf else ""
        missing = [k for k in K[i]["k"] if k not in notes]
        if missing:
            fails.append(f"{tag}: keynotes missing from the notes: {missing}")
    for i in list(range(0, len(clips) - 1, 9)):
        d = np.abs(frame(clips[i], True) - frame(clips[i + 1], False)).mean()
        if d >= 4:
            fails.append(f"{clicks[i + 1]['id']}: starts {d:.1f} grey levels away from where {clicks[i]['id']} ended")
    pdf = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(PPTX)], capture_output=True, text=True, timeout=600)
    pages = subprocess.run(["pdfinfo", str(tmp / (PPTX.stem + ".pdf"))], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", pages)
    n = int(m.group(1)) if m else 0
    if n != 75:
        fails.append(f"LibreOffice: {n} pages ({pdf.stderr.strip()[:120]})")
    print(f"{PPTX.name}: {PPTX.stat().st_size / 1e6:.0f} MB, {len(prs.slides)} slides")
    for f in fails:
        print("FAIL " + f)
    print("PASS" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
