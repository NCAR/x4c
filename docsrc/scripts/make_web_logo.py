#!/usr/bin/env python
"""Generate the web-sized logo used by the sidebar and favicon.

docsrc/x4c-logo.png is a 2420x2416 / 625 KB master. The sidebar renders it at
roughly 300 px and the favicon far smaller, so shipping the master to every
page visitor wastes ~600 KB per load.

This writes docsrc/assets/x4c-logo-web.png at 600 px (2x the rendered size, so
still sharp on HiDPI) quantised to 256 colours, which is ~21 KB -- a 97%
reduction with no visible difference at display size. The master is left alone.

Run from anywhere:  python docsrc/scripts/make_web_logo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

DOCSRC = Path(__file__).resolve().parent.parent
SRC = DOCSRC / "x4c-logo.png"
DEST = DOCSRC / "assets" / "x4c-logo-web.png"

SIZE = 600      # 2x the ~300px sidebar render
COLORS = 256    # logo art, not photography -- 256 is indistinguishable here


def main() -> int:
    if not SRC.exists():
        print(f"missing source logo: {SRC}", file=sys.stderr)
        return 1

    DEST.parent.mkdir(exist_ok=True)
    im = Image.open(SRC).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    im.quantize(colors=COLORS, method=Image.FASTOCTREE).save(DEST, optimize=True)

    before = SRC.stat().st_size
    after = DEST.stat().st_size
    print(f"  {SRC.name:<20} {before // 1024:>5}K")
    print(f"  {DEST.name:<20} {after // 1024:>5}K   ({100 * (1 - after / before):.1f}% smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
