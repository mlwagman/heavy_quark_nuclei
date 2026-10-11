#!/usr/bin/env python3
"""Build talk/mrs-talk-world3d.pptx from the rendered video (round 22): one slide per beat (75), each playing that beat's
animation by itself when the slide opens and holding its last frame, so one click per beat as in the World. The keynotes of
the beat (js/keys.js) are the slide's speaker notes. The clips are silent, cut from mrs-talk-world3d.mp4 at each click.

    python3 talk/world3d/tools/build_pptx.py        then: python3 talk/world3d/tools/check_pptx.py
"""
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pptx import Presentation
from pptx.oxml import parse_xml
from pptx.util import Emu

W = Path(__file__).resolve().parents[1]
OUT = W.parent / "mrs-talk-world3d.pptx"
VIDEO = Path("/home/gofer/mrs_talk_video/world3d")
FPS = 30
B = Path("/tmp/w3d-bin")   # ffmpeg under another process name: a "pkill -x ffmpeg" elsewhere on the machine kills encoders
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
AUTO = f"""<p:seq xmlns:p="{P}" concurrent="1" nextAc="seek"><p:cTn id="3" dur="indefinite" nodeType="mainSeq"><p:childTnLst>
<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="3"/></p:cond></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="5" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="6" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="afterEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="7" dur="{{ms}}" fill="hold"/><p:tgtEl><p:spTgt spid="{{spid}}"/></p:tgtEl></p:cBhvr></p:cmd>
</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>
<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>
<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>"""


def ffmpeg() -> str:
    B.mkdir(exist_ok=True)
    link, wrap = B / "ffmpeg-w3d", B / "ffmpeg"
    if not link.exists():
        link.symlink_to("/usr/bin/ffmpeg")
    if not wrap.exists():
        wrap.write_text(f'#!/bin/sh\nexec {link} "$@"\n')
        wrap.chmod(0o755)
    return str(wrap)


def keys() -> list[dict]:
    js = (W / "js/keys.js").read_text(encoding="utf-8")
    out = subprocess.run(["node", "-e", js + "\nprocess.stdout.write(JSON.stringify(KEYS))"], capture_output=True, text=True, check=True)
    return [b for sc in json.loads(out.stdout) for b in sc]


def cut(ff: str, tmp: Path, i: int, c: dict) -> tuple[Path, Path]:
    clip, poster, n = tmp / f"{i:02d}.mp4", tmp / f"{i:02d}.png", round((c["tween"] + 0.3) * FPS)
    subprocess.run([ff, "-v", "error", "-y", "-ss", f"{c['frame'] / FPS:.4f}", "-i", str(VIDEO / "mrs-talk-world3d.mp4"), "-frames:v", str(n), "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)], check=True)
    subprocess.run([ff, "-v", "error", "-y", "-i", str(clip), "-frames:v", "1", str(poster)], check=True)
    return clip, poster


def main() -> None:
    ff, clicks, K = ffmpeg(), json.loads((VIDEO / "schedule.json").read_text())["clicks"], keys()
    tmp = Path(tempfile.mkdtemp())
    with ThreadPoolExecutor(4) as ex:
        media = list(ex.map(lambda ic: cut(ff, tmp, *ic), enumerate(clicks)))
    prs = Presentation()
    SW, SH = Emu(12192000), Emu(6858000)
    prs.slide_width, prs.slide_height = SW, SH
    for i, (c, (clip, poster)) in enumerate(zip(clicks, media)):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        mv = slide.shapes.add_movie(str(clip), Emu(0), Emu(0), SW, SH, poster_frame_image=str(poster), mime_type="video/mp4")
        root = slide.element.xpath("./p:timing/p:tnLst/p:par/p:cTn/p:childTnLst")[0]
        root.xpath("./p:video/p:cMediaNode/p:cTn")[0].set("id", "2")
        root.insert(0, parse_xml(AUTO.format(ms=round((c["tween"] + 0.3) * 1000), spid=mv.shape_id)))
        k = K[i]
        tf = slide.notes_slide.notes_text_frame
        assert tf is not None
        tf.text = f"{c['id']}\n" + "\n".join("• " + x for x in k["k"]) + (f"\non screen, no need to say: {k['show']}" if k.get("show") else "")
    prs.save(str(OUT))
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
