"""Validated LaTeX-to-MathML with Chromium/WebView2 presentation support.

This validates rendering syntax, not whether OCR transcribed the source correctly.
The caller must keep the unmodified source separately.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

MATH_NS = "http://www.w3.org/1998/Math/MathML"
MATH_BOX_CLASS = "comfortable-math-box"
VENDOR = Path(__file__).resolve().parent.parent / "vendor"
if VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


class MathRenderingError(ValueError):
    pass


def normalize_render_syntax(source: str) -> str:
    """Translate layout syntax only; never infer symbols, case, or punctuation."""
    text = source.strip()
    # The bundled converter optimizes a one-letter \mathrm group to a bare
    # Unicode mi, which MathML then renders in italics. Equivalent grouping
    # preserves the source's explicit upright style without changing symbols.
    text = re.sub(r'\\mathrm\{([A-Za-z])\}', lambda m: r'\mathrm{{'+m.group(1)+'}}', text)
    for left, right in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break
    # A row break followed by [2pt] is not an opening display-math delimiter.
    text = re.sub(
        r"(\\\\)\s*\[\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|em|ex|mm|cm)\s*\]",
        lambda match: match.group(1), text,
    )
    environments = {"aligned": "rl", "align": "rl", "align*": "rl",
                    "split": "rl", "gather": "c", "gather*": "c", "gathered": "c"}
    for name, columns in environments.items():
        text = text.replace(r"\begin{" + name + "}", r"\begin{array}{" + columns + "}")
        text = text.replace(r"\end{" + name + "}", r"\end{array}")
    text = re.sub(r"\\(?:bigl|Bigl|biggl|Biggl)\b", r"\\left", text)
    text = re.sub(r"\\(?:bigr|Bigr|biggr|Biggr)\b", r"\\right", text)
    text = re.sub(r"\\(?:nonumber|notag)\b", "", text)
    if r"\\" in text and r"\begin{" not in text:
        text = r"\begin{array}{l}" + text + r"\end{array}"
    return text


def _local(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _style(node: ET.Element, rule: str) -> None:
    old = node.get("style", "").strip().rstrip(";")
    node.set("style", (old + ";" if old else "") + rule)


def validate_mathml(markup: str) -> ET.Element:
    try:
        root = ET.fromstring(markup)
    except ET.ParseError as exc:
        raise MathRenderingError(f"MathML is not valid XML: {exc}") from exc
    if root.tag != "{" + MATH_NS + "}math":
        raise MathRenderingError("Expected a MathML math element with its namespace.")
    def visible(node: ET.Element) -> str:
        if _local(node) in {"annotation", "annotation-xml"}:
            return ""
        return (node.text or "") + "".join(visible(child) + (child.tail or "") for child in node)
    text = visible(root)
    if "\ufffd" in text or "\x00" in text:
        raise MathRenderingError("Invalid or replacement character in rendered math.")
    # The converter can succeed while emitting an unknown command as an mi.
    # Check non-letter commands too: a broken row break leaked '\\2pt]' before.
    if "\\" in text:
        raise MathRenderingError("A TeX command or backslash remains visible in MathML.")
    return root


def render_mathml(source: str, display: bool = False) -> str:
    try:
        from latex2mathml.converter import convert
    except ImportError as exc:
        raise MathRenderingError("latex2mathml is unavailable.") from exc
    normalized = normalize_render_syntax(source)
    try:
        root = validate_mathml(convert(normalized, display="block" if display else "inline"))
    except MathRenderingError:
        raise
    except Exception as exc:
        raise MathRenderingError(f"Could not render formula: {exc}") from exc
    parents = {child: parent for parent in root.iter() for child in parent}

    def ancestors(node: ET.Element):
        while node in parents:
            node = parents[node]
            yield node

    def display_style(node: ET.Element) -> bool:
        for item in [node, *ancestors(node)]:
            if item.get("displaystyle") in ("true", "false"):
                return item.get("displaystyle") == "true"
        return display

    for node in root.iter():
        tag = _local(node)
        if tag == "menclose" and "box" in node.get("notation", "").split():
            classes = node.get("class", "").split()
            if MATH_BOX_CLASS not in classes:
                node.set("class", " ".join(classes + [MATH_BOX_CLASS]))
        if tag == "mtd" and node.get("columnalign"):
            _style(node, "text-align:" + node.get("columnalign", "center"))
        if tag == "mtable":
            if display_style(node):
                node.set("displaystyle", "true")
            rowlines = node.get("rowlines", "").split()
            columnlines = node.get("columnlines", "").split()
            for ri, row in enumerate(node):
                for ci, cell in enumerate(row):
                    if ri and rowlines:
                        line = rowlines[min(ri - 1, len(rowlines) - 1)]
                        if line != "none":
                            _style(cell, "border-top:.06em " + ("dashed" if line == "dashed" else "solid") + " currentColor")
                    if ci < len(row) - 1 and columnlines:
                        line = columnlines[min(ci, len(columnlines) - 1)]
                        if line != "none":
                            _style(cell, "border-right:.06em " + ("dashed" if line == "dashed" else "solid") + " currentColor")
        if tag in ("msub", "msup", "msubsup") and len(node) and display_style(node):
            operator = "".join(node[0].itertext())
            if operator in {"∑", "∏", "⋃", "⋂", "lim", "limsup", "liminf", "max", "min", "inf", "sup"}:
                node.tag = "{" + MATH_NS + "}" + {"msub": "munder", "msup": "mover", "msubsup": "munderover"}[tag]
                node[0].set("movablelimits", "false")
        # Standard TeX capital Greek commands are upright. Explicit source
        # font variants (including inherited mstyle) take precedence.
        if tag == "mi" and len(node.text or "") == 1 and "Α" <= node.text <= "Ω":
            if not any(item.get("mathvariant") for item in [node, *ancestors(node)]):
                node.set("mathvariant", "normal")
        if tag == "mtext" and re.fullmatch(r"\((?:[A-D]|\d+)(?:\.\d+)+\)", node.text or ""):
            _style(node, "padding-inline-start:.9em")

    # Insert function spacing after its entire script group, not between the
    # function name and a logarithm base or exponent.
    functions = {"lg", "ln", "log", "sin", "cos", "tan", "arctan", "exp", "Pr", "gcd"}
    for node in list(root.iter()):
        name = node.text or ""
        if _local(node) not in ("mi", "mo") or name not in functions:
            continue
        if not re.search(r"\\" + re.escape(name) + r"\b", source):
            continue
        target = node
        parent = parents.get(node)
        if parent is not None and _local(parent) in ("msub", "msup", "msubsup", "munder", "mover", "munderover") and parent[0] is node:
            target, parent = parent, parents.get(parent)
        if parent is not None:
            parent.insert(list(parent).index(target) + 1, ET.Element("{" + MATH_NS + "}mspace", {"width": ".1667em"}))
    ET.register_namespace("", MATH_NS)
    markup = ET.tostring(root, encoding="unicode")
    validate_mathml(markup)
    return markup
