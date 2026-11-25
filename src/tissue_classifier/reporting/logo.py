"""Native vector SVG renderer and base64 loader for the bundled TOPO logotype."""

from __future__ import annotations

import base64
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TOPO_LOGO_SVG = ASSETS_DIR / "topo_logo.svg"
TOPO_ICON_SVG = ASSETS_DIR / "topo_icon.svg"

_TOKEN_RE = re.compile(r"[MLCZmlcz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TRANSLATE_RE = re.compile(r"translate\(\s*([-+]?\d*\.?\d+)(?:[\s,]+([-+]?\d*\.?\d+))?\s*\)")


def _parse_svg_path(d: str, tx: float = 0.0, ty: float = 0.0) -> MplPath:
    """Parse an SVG path string containing M, L, C, Z commands into a Matplotlib Path."""
    tokens = _TOKEN_RE.findall(d)
    verts: list[tuple[float, float]] = []
    codes: list[int] = []

    idx = 0
    cmd = ""
    start_pt = (0.0, 0.0)
    n = len(tokens)

    while idx < n:
        tok = tokens[idx]
        if tok.isalpha():
            cmd = tok
            idx += 1
            if cmd in ("Z", "z"):
                verts.append(start_pt)
                codes.append(MplPath.CLOSEPOLY)
                continue

        if cmd == "M":
            x, y = float(tokens[idx]) + tx, float(tokens[idx + 1]) + ty
            idx += 2
            start_pt = (x, y)
            verts.append((x, y))
            codes.append(MplPath.MOVETO)
            cmd = "L"  # subsequent coordinate pairs are implicit L
        elif cmd == "L":
            x, y = float(tokens[idx]) + tx, float(tokens[idx + 1]) + ty
            idx += 2
            verts.append((x, y))
            codes.append(MplPath.LINETO)
        elif cmd == "C":
            x1, y1 = float(tokens[idx]) + tx, float(tokens[idx + 1]) + ty
            x2, y2 = float(tokens[idx + 2]) + tx, float(tokens[idx + 3]) + ty
            x3, y3 = float(tokens[idx + 4]) + tx, float(tokens[idx + 5]) + ty
            idx += 6
            verts.extend([(x1, y1), (x2, y2), (x3, y3)])
            codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        else:
            raise ValueError(f"Unsupported SVG path command '{cmd}' in bundled TOPO asset")

    return MplPath(np.asarray(verts, dtype=np.float64), codes)


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def draw_topo_svg(
    fig: plt.Figure,
    rect: list[float],
    variant: str = "logo",
    include_subtitle: bool = True,
) -> plt.Axes:
    """Render the bundled TOPO vector SVG directly onto a Matplotlib figure.

    Parameters
    ----------
    fig:
        Target Matplotlib Figure.
    rect:
        [left, bottom, width, height] in figure-normalized coordinates.
    variant:
        "logo" for full icon + TOPO wordmark lockup, or "icon" for icon mark only.
    include_subtitle:
        Whether to draw the subtitle text below the TOPO wordmark.
    """
    svg_path = TOPO_LOGO_SVG if variant == "logo" else TOPO_ICON_SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb_raw = root.attrib.get("viewBox", "0 0 1020 370").split()
    vb_x, vb_y, vb_w, vb_h = (float(v) for v in vb_raw)

    ax = fig.add_axes(rect, frameon=False)
    ax.set_xlim(vb_x, vb_x + vb_w)
    ax.set_ylim(vb_y + vb_h, vb_y)  # SVG y increases downward
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.set_xticks([])
    ax.set_yticks([])

    # Physical scale factor (points per SVG user unit) for faithful stroke widths & text
    box_w_in = rect[2] * fig.get_figwidth()
    box_h_in = rect[3] * fig.get_figheight()
    scale_in = min(box_w_in / vb_w, box_h_in / vb_h)
    pts_per_unit = scale_in * 72.0

    def _walk(
        elem: ET.Element,
        tx: float,
        ty: float,
        style: dict[str, str],
    ) -> None:
        tag = _strip_ns(elem.tag)
        cur_style = dict(style)
        for key in (
            "fill",
            "stroke",
            "stroke-width",
            "stroke-linecap",
            "stroke-linejoin",
            "font-size",
            "font-weight",
        ):
            if key in elem.attrib:
                cur_style[key] = elem.attrib[key]

        cur_tx, cur_ty = tx, ty
        transform = elem.attrib.get("transform", "")
        if transform:
            m = _TRANSLATE_RE.search(transform)
            if m:
                cur_tx += float(m.group(1))
                cur_ty += float(m.group(2) or 0.0)

        if tag == "path":
            d = elem.attrib.get("d", "")
            if d:
                path = _parse_svg_path(d, cur_tx, cur_ty)
                fill = cur_style.get("fill", "none")
                stroke = cur_style.get("stroke", "none")
                sw = float(cur_style.get("stroke-width", "0"))
                cap = cur_style.get("stroke-linecap", "butt")
                join = cur_style.get("stroke-linejoin", "miter")
                patch = PathPatch(
                    path,
                    facecolor="none" if fill == "none" else fill,
                    edgecolor="none" if stroke == "none" else stroke,
                    linewidth=sw * pts_per_unit if stroke != "none" else 0.0,
                    capstyle=cap,
                    joinstyle=join,
                )
                ax.add_patch(patch)
        elif tag == "line":
            x1 = float(elem.attrib["x1"]) + cur_tx
            y1 = float(elem.attrib["y1"]) + cur_ty
            x2 = float(elem.attrib["x2"]) + cur_tx
            y2 = float(elem.attrib["y2"]) + cur_ty
            stroke = cur_style.get("stroke", "#1E2D3E")
            sw = float(cur_style.get("stroke-width", "1.0"))
            cap = cur_style.get("stroke-linecap", "butt")
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=stroke,
                linewidth=sw * pts_per_unit,
                solid_capstyle=cap,
            )
        elif tag == "text" and include_subtitle:
            txt = (elem.text or "").strip()
            if txt:
                x = float(elem.attrib.get("x", "0")) + cur_tx
                y = float(elem.attrib.get("y", "0")) + cur_ty
                fill = cur_style.get("fill", "#646B75")
                fs = float(cur_style.get("font-size", "19.5")) * pts_per_unit
                # Slight boost at very small sizes for crisp PDF readability while preserving alignment
                fs_render = max(fs * 1.18, 3.6)
                ax.text(
                    x,
                    y,
                    txt,
                    fontsize=fs_render,
                    color=fill,
                    fontfamily="sans-serif",
                    fontweight="normal",
                    va="baseline",
                    ha="left",
                )
        else:
            for child in elem:
                _walk(child, cur_tx, cur_ty, cur_style)

    _walk(root, 0.0, 0.0, {})
    return ax


def load_logo_svg_b64(variant: str = "logo") -> str:
    """Return base64-encoded SVG bytes for standalone HTML embedding."""
    svg_path = TOPO_LOGO_SVG if variant == "logo" else TOPO_ICON_SVG
    return base64.b64encode(svg_path.read_bytes()).decode("ascii")
