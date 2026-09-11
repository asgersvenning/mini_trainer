"""Compact standalone Matplotlib dendrogram SVGs without outlining labels."""

import hashlib
import re
from pathlib import Path

from matplotlib import rc_context

_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def compact_svg(svg: str) -> str:
    """Round path coordinates to 0.001 pt and share identical text styles.

    Matplotlib SVG coordinates use points. Rounding contributes at most 0.00071
    pt Euclidean error per point, including Bezier control points. Text content,
    positions, transforms, colors and font choices are preserved. Content-based
    CSS names avoid collisions when several exports are embedded in a document.
    """

    def path(match):
        def number(value):
            result = f"{float(value.group()):.3f}".rstrip("0").rstrip(".")
            return "0" if result == "-0" else result

        return match[1] + _NUMBER.sub(number, match[2]) + match[3]

    svg = re.sub(r'(<path\b[^>]*\bd=")([^"]*)(")', path, svg)
    styles = {}

    def text_style(match):
        style = match[1]
        name = "mt-text-" + hashlib.sha256(style.encode()).hexdigest()[:16]
        styles[name] = style
        return f'<text class="{name}"'

    svg = re.sub(r'<text style="([^"]*)"', text_style, svg)
    # Matplotlib gives every label a separate, otherwise empty group. Removing
    # these wrappers halves label DOM nodes without changing text or transforms.
    svg = re.sub(r'<g id="text_\d+">\s*(<text\b[^>]*>.*?</text>)\s*</g>', r"\1", svg, flags=re.DOTALL)
    if styles:
        css = "".join(f"text.{name}{{{style}}}" for name, style in styles.items())
        svg = svg.replace("</defs>", f'<style type="text/css">{css}</style></defs>', 1)
    return svg


def save_dendrogram_svg(figure, filename):
    """Export selectable labels and compact geometry/styles to a local SVG."""
    with rc_context({"svg.fonttype": "none"}):
        figure.savefig(filename, format="svg", bbox_inches="tight")
    path = Path(filename)
    path.write_text(compact_svg(path.read_text(encoding="utf-8")), encoding="utf-8")
