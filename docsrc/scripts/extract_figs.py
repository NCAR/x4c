#!/usr/bin/env python
"""Extract homepage gallery figures from the executed notebooks.

The notebooks ship pre-executed, so every figure already exists as base64 PNG
inside the .ipynb JSON. Jupyter Book 2 renders those inline but does not expose
them as files the homepage can link to, so we pull the ones the gallery needs
out to docsrc/assets/.

Figures are addressed by (notebook, index of the figure in document order),
which is stable across toolchains -- unlike a renderer's own cell numbering.
Each gallery notebook is also kept small and single-purpose (one feature, one
figure) precisely so that index rarely has to move.

Run from the repo root:  python docsrc/scripts/extract_figs.py
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path

DOCSRC = Path(__file__).resolve().parent.parent
NOTEBOOKS = DOCSRC / "notebooks"
ASSETS = DOCSRC / "assets"

# (output name, notebook, figure index in document order)
FIGURES = [
    ("gallery-isotopes.png",    "diags-variables-isotopes",     0),
    ("gallery-eof.png",         "core-analysis-eof",            0),
    ("gallery-ocean-land.png",  "diags-variables-ocean",        0),
    ("gallery-quickview.png",   "diags-quickview",               0),
    ("gallery-regrid.png",      "core-regridding-atm",          0),
    ("gallery-precip.png",      "diags-new_vars",                0),
    ("gallery-moc.png",         "core-visualization-vertical",  0),
    ("gallery-styles.png",      "core-visualization-styles",    0),
    ("gallery-seasonal.png",    "core-annualization-seasonal",  0),
    ("gallery-projections.png", "core-visualization-projections", 0),
    ("gallery-plev.png",        "core-plev",                     0),
]


def figures_in(nb_path: Path) -> list[bytes]:
    """Every PNG output in the notebook, in document order."""
    nb = json.loads(nb_path.read_text())
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                out.append(base64.b64decode(png))
    return out


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    failed = False

    for name, nb_name, index in FIGURES:
        nb_path = NOTEBOOKS / f"{nb_name}.ipynb"
        if not nb_path.exists():
            print(f"MISSING notebook: {nb_path}", file=sys.stderr)
            failed = True
            continue

        figs = figures_in(nb_path)
        if index >= len(figs):
            print(
                f"MISSING figure {index} in {nb_name} (has {len(figs)})",
                file=sys.stderr,
            )
            failed = True
            continue

        data = figs[index]
        (ASSETS / name).write_bytes(data)
        w, h = png_size(data)
        print(f"  {name:<26} {w:>5}x{h:<4} {len(data) // 1024:>4}K  <- {nb_name}[{index}]")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
