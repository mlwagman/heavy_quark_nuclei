#!/usr/bin/env python3
"""Python twin of the dataviz skill's validate_palette.js -- this box has no node.

Same thresholds, same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD transforms, same
OKLab Delta E x100 metric, so the numbers are comparable to the JS reference. Checks:

  2.  lightness band    OKLCH L inside the mode's band
  3.  chroma floor      OKLCH C >= 0.10 (below it a hue reads as gray)
  4.  CVD separation    min(protan, deutan) Delta E on the pairlist; >=8 target, 6-8 floor
                        (legal only with secondary encoding), <6 FAIL
  4b. normal-vision     worst unsimulated Delta E on the pairlist; >=15 hard gate
  5.  contrast          WCAG vs the chart surface; <3:1 is a WARN carrying the relief rule

Usage:
  validate_palette.py "#2a78d6,#eb6834,#1baf7a,#6250d6" [--mode light|dark] [--pairs all]
"""
import argparse
import math
import re
import sys

BAND = {'light': (0.43, 0.77), 'dark': (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0
DEFAULT_SURFACE = {'light': '#fcfcfb', 'dark': '#1a1a19'}

MACHADO = {
    'protan': ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    'deutan': ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    'tritan': ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}

# the shared whitespace set, per the JS note about trim()/strip() differing at the edges
WS = ' \t\n\v\f\r      　' + ''.join(
    chr(c) for c in range(0x2000, 0x200b))
HEX = re.compile(r'^#?[0-9a-fA-F]{6}$')


def hex2srgb(h):
    h = h.strip(WS).lstrip('#')
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def okhue(h):
    _, a, b = oklab_from_lin(lin(h))
    return math.degrees(math.atan2(b, a)) % 360


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate_ordinal(palette, mode='light', surface=None):
    """Ordered categories take a one-hue ramp, not categorical hues: the categorical checks
    FAIL a correct ramp by design (it spans the band and its light steps fall under the chroma
    floor).  These check that it reads AS a ramp."""
    surface = surface or DEFAULT_SURFACE[mode]
    rows, ok = [], True
    Ls = [oklch(c)[0] for c in palette]
    order = sorted(range(len(Ls)), key=lambda i: Ls[i])
    mono = order == list(range(len(Ls))) or order == list(reversed(range(len(Ls))))
    ok &= mono
    rows.append(('Lightness monotone', 'PASS' if mono else 'FAIL',
                 'steps read light->dark' if mono
                 else f'out of order - L {[round(l, 3) for l in Ls]}'))
    gaps = [abs(Ls[i + 1] - Ls[i]) for i in range(len(Ls) - 1)]
    thin = [(palette[i], palette[i + 1], round(g, 3))
            for i, g in enumerate(gaps) if g < ORDINAL_MIN_DL]
    ok &= not thin
    rows.append(('Adjacent dL', 'PASS' if not thin else 'FAIL',
                 f'too close: {thin}' if thin else f'all gaps >= {ORDINAL_MIN_DL}'))
    by_l = sorted(palette, key=lambda c: oklch(c)[0])
    lightest = by_l[-1] if mode == 'light' else by_l[0]
    cr = contrast(lightest, surface)
    ok &= cr >= ORDINAL_LIGHT_FLOOR
    rows.append(('Light-end contrast', 'PASS' if cr >= ORDINAL_LIGHT_FLOOR else 'FAIL',
                 f'{lightest} at {cr:.2f}:1 vs {surface}'
                 + ('' if cr >= ORDINAL_LIGHT_FLOOR else f' - below {ORDINAL_LIGHT_FLOOR}:1')))
    hues = [okhue(c) for c in palette]
    spread = max(hues) - min(hues)
    one = spread <= 35
    ok &= one
    rows.append(('Single hue', 'PASS' if one else 'FAIL',
                 f'hue spread {spread:.1f} deg'
                 + ('' if one else ' - a hue jump means this is categorical')))
    return ok, rows


def validate(palette, mode='light', surface=None, pairs='adjacent'):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    rows, ok = [], True

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    rows.append(('Lightness band', 'PASS' if not off else 'FAIL',
                 f'outside L {lo}-{hi}: {off}' if off else f'all {len(palette)} inside L {lo}-{hi}'))

    low = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not low
    rows.append(('Chroma floor', 'PASS' if not low else 'FAIL',
                 f'reads gray: {low}' if low else f'all {len(palette)} >= {CHROMA_FLOOR}'))

    n = len(palette)
    if pairs == 'all':
        plist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        plist = [(i, i + 1) for i in range(n - 1)]
    label = 'all-pairs' if pairs == 'all' else 'adjacent'

    worst = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j])
                 for k in ('protan', 'deutan') for i, j in plist), key=lambda t: t[0])
    tri = min(delta_e(palette[i], palette[j], 'tritan') for i, j in plist)
    state = 'PASS' if worst[0] >= CVD_TARGET else ('FLOOR' if worst[0] >= CVD_FLOOR else 'FAIL')
    ok &= state != 'FAIL'
    rows.append(('CVD separation', state,
                 f'worst {label} {worst[2]}<->{worst[3]} dE {worst[0]:.1f} ({worst[1]})'
                 f' - tritan {tri:.1f}'))

    wn = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in plist),
             key=lambda t: t[0])
    nstate = 'PASS' if wn[0] >= NORMAL_FLOOR else 'FAIL'
    ok &= nstate == 'PASS'
    rows.append(('Normal-vision floor', nstate,
                 f'worst {label} {wn[1]}<->{wn[2]} dE {wn[0]:.1f} (floor {NORMAL_FLOOR})'))

    cs = [(c, round(contrast(c, surface), 2)) for c in palette]
    weak = [x for x in cs if x[1] < CONTRAST_MIN]
    rows.append(('Contrast vs surface', 'PASS' if not weak else 'WARN',
                 f'below {CONTRAST_MIN}:1 (relief rule: direct labels or table): {weak}'
                 if weak else f'all >= {CONTRAST_MIN}:1 on {surface}'))
    return ok, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('palette')
    p.add_argument('--mode', default='light', choices=['light', 'dark'])
    p.add_argument('--surface', default=None)
    p.add_argument('--pairs', default='adjacent', choices=['adjacent', 'all'])
    p.add_argument('--ordinal', action='store_true',
                   help='validate as a one-hue ordinal ramp instead of categorical hues')
    a = p.parse_args()
    pal = [c.strip(WS) for c in a.palette.split(',') if c.strip(WS)]
    bad = [c for c in pal if not HEX.match(c)]
    if bad:
        sys.exit(f'not hex colors: {bad}')
    pal = ['#' + c.lstrip('#') for c in pal]
    ok, rows = (validate_ordinal(pal, a.mode, a.surface) if a.ordinal
                else validate(pal, a.mode, a.surface, a.pairs))
    w = max(len(r[0]) for r in rows)
    print(f'{len(pal)} colors, {"ordinal ramp" if a.ordinal else "categorical"}, '
          f'mode={a.mode}' + ('' if a.ordinal else f', pairs={a.pairs}')
          + f', surface={a.surface or DEFAULT_SURFACE[a.mode]}')
    for name, state, detail in rows:
        print(f'  {name:<{w}}  {state:<5}  {detail}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
