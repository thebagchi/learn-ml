"""Render Mermaid venn-beta syntax as matplotlib Venn diagrams.

Since mermaid.ink does not yet support the venn-beta diagram type
(added in Mermaid v11.12.3+), this module parses the same syntax and
draws equivalent 2-set or 3-set Venn diagrams using matplotlib patches.

Grammar parity with the official Mermaid jison grammar:
  https://github.com/mermaid-js/mermaid/blob/develop/packages/mermaid/src/diagrams/venn/parser/venn.jison
Syntax reference:
  https://mermaid.js.org/syntax/venn.html

Supported statements (case-insensitive keywords):
    venn-beta                        — required header
    title "My Title"                 — optional diagram title
    set A["Label"]                   — define a set (bracket label)
    set A["Label"]:20                — set with optional size hint
    set "Foo Bar"                    — quoted string identifier
    union A,B["Overlap"]             — overlap region of 2+ sets
    union A,B["Overlap"]:3           — union with size hint
    text a1["item"]                  — text node (indented → attaches to last set/union)
    text A,B myid["label"]           — text node with explicit target list
    style A fill:#ff6b6b             — CSS-like styling per target
    style A,B color:#333,fill:#eee   — comma-separated style fields
    %% comment                       — line comments

Supported style properties:
    fill, fill-opacity, stroke, stroke-width, color
    Values: #hex, rgb(), rgba(), bare identifiers, numbers

Use <br/> inside labels for line breaks.

Usage:
    from venn import venn

    venn('''
    venn-beta
      title "Sample Space"
      set A["Event A"]
      set B["Event B"]
      union A,B["A ∩ B"]
    ''')
"""

from __future__ import annotations

import os
import sys
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

import ply.lex as lex
import ply.yacc as yacc


# ---------------------------------------------------------------------------
# Material Design color schemes
# Each scheme defines: circles (up to 3), background, outline,
#                       label color, sub-text color
# Reference: https://m3.material.io/styles/color/static/baseline
# ---------------------------------------------------------------------------

class Color(Enum):
    """Material Design color schemes for Venn diagrams.

    Usage:
        from libs.venn import venn, Color
        venn(diagram_string, color_scheme=Color.OCEAN)
    """
    # ── Material Design 3 baseline ────────────────────────────────────────
    BASELINE = {
        "circles": ["#6750A4", "#625B71", "#7D5260"],   # primary, secondary, tertiary
        "bg_fill": "#FFFBFE",   "bg_stroke": "#79747E",  # surface, outline
        "label": "#1C1B1F",     "sub": "#49454F",        # on-surface, on-surface-variant
    }

    # ── Cool tones ────────────────────────────────────────────────────────
    OCEAN = {
        "circles": ["#1565C0", "#00838F", "#2E7D32"],   # blue 800, cyan 800, green 800
        "bg_fill": "#E3F2FD",   "bg_stroke": "#90CAF9",  # blue 50, blue 200
        "label": "#0D47A1",     "sub": "#1565C0",        # blue 900, blue 800
    }
    INDIGO = {
        "circles": ["#283593", "#1565C0", "#00695C"],   # indigo 800, blue 800, teal 800
        "bg_fill": "#E8EAF6",   "bg_stroke": "#9FA8DA",  # indigo 50, indigo 200
        "label": "#1A237E",     "sub": "#283593",        # indigo 900, indigo 800
    }
    TEAL = {
        "circles": ["#00695C", "#00838F", "#2E7D32"],   # teal 800, cyan 800, green 800
        "bg_fill": "#E0F2F1",   "bg_stroke": "#80CBC4",  # teal 50, teal 200
        "label": "#004D40",     "sub": "#00695C",        # teal 900, teal 800
    }

    # ── Warm tones ────────────────────────────────────────────────────────
    SUNSET = {
        "circles": ["#E65100", "#F9A825", "#AD1457"],   # orange 900, yellow 800, pink 800
        "bg_fill": "#FFF3E0",   "bg_stroke": "#FFCC80",  # orange 50, orange 200
        "label": "#BF360C",     "sub": "#E65100",        # deep-orange 900, orange 900
    }
    ROSE = {
        "circles": ["#AD1457", "#6A1B9A", "#C62828"],   # pink 800, purple 800, red 800
        "bg_fill": "#FCE4EC",   "bg_stroke": "#F48FB1",  # pink 50, pink 200
        "label": "#880E4F",     "sub": "#AD1457",        # pink 900, pink 800
    }
    EARTH = {
        "circles": ["#4E342E", "#33691E", "#E65100"],   # brown 800, l-green 900, orange 900
        "bg_fill": "#EFEBE9",   "bg_stroke": "#BCAAA4",  # brown 50, brown 200
        "label": "#3E2723",     "sub": "#4E342E",        # brown 900, brown 800
    }

    # ── Neutral / high-contrast ───────────────────────────────────────────
    SLATE = {
        "circles": ["#37474F", "#546E7A", "#78909C"],   # blue-grey 800/600/400
        "bg_fill": "#ECEFF1",   "bg_stroke": "#B0BEC5",  # blue-grey 50, blue-grey 200
        "label": "#263238",     "sub": "#37474F",        # blue-grey 900, blue-grey 800
    }
    MONOCHROME = {
        "circles": ["#212121", "#616161", "#9E9E9E"],   # grey 900/700/500
        "bg_fill": "#FAFAFA",   "bg_stroke": "#BDBDBD",  # grey 50, grey 400
        "label": "#212121",     "sub": "#424242",        # grey 900, grey 800
    }

    # ── Vibrant / presentation ────────────────────────────────────────────
    VIBRANT = {
        "circles": ["#D32F2F", "#1976D2", "#388E3C"],   # red 700, blue 700, green 700
        "bg_fill": "#FFFFFF",   "bg_stroke": "#BDBDBD",  # white, grey 400
        "label": "#212121",     "sub": "#616161",        # grey 900, grey 700
    }
    PASTEL = {
        "circles": ["#EF9A9A", "#90CAF9", "#A5D6A7"],   # red 200, blue 200, green 200
        "bg_fill": "#FFFFFF",   "bg_stroke": "#E0E0E0",  # white, grey 300
        "label": "#424242",     "sub": "#757575",        # grey 800, grey 600
    }
    NEON = {
        "circles": ["#AA00FF", "#00BFA5", "#FFD600"],   # purple A700, teal A700, yellow A700
        "bg_fill": "#F3E5F5",   "bg_stroke": "#CE93D8",  # purple 50, purple 200
        "label": "#4A148C",     "sub": "#6A1B9A",        # purple 900, purple 800
    }


# ---------------------------------------------------------------------------
# Active palette defaults (BASELINE)
# ---------------------------------------------------------------------------
_DEFAULT_SCHEME = Color.BASELINE
_d = _DEFAULT_SCHEME.value
_CIRCLE_COLORS = _d["circles"]
_BG_FILL = _d["bg_fill"]
_BG_STROKE = _d["bg_stroke"]
_LABEL_COLOR = _d["label"]
_SUB_COLOR = _d["sub"]
_FONT_FAMILY = "sans-serif"
_CIRCLE_ALPHA = 0.55
_CIRCLE_STROKE = "#FFFFFF"
_CIRCLE_LW = 2.5


def _style_val(styles, sid, prop, default):
    """Look up a CSS-like property for *sid* in the parsed styles dict."""
    return styles.get(sid, {}).get(prop, default)


def _draw_circle(ax, center, radius, fill, opacity, stroke, lw):
    """Draw one Mermaid-style circle with fill + subtle stroke."""
    rgba = to_rgba(fill, alpha=opacity)
    circ = plt.Circle(
        center, radius,
        facecolor=rgba,
        edgecolor=stroke,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(circ)
    return circ


def _put_label(ax, x, y, text, fontsize=11, color=_LABEL_COLOR, weight="bold"):
    """Place a primary label (set / union name)."""
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        color=color, fontfamily=_FONT_FAMILY,
        zorder=4,
    )


def _put_sub(ax, x, y, text, fontsize=9, color=_SUB_COLOR):
    """Place a secondary text node."""
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize, color=color,
        fontfamily=_FONT_FAMILY,
        zorder=4,
    )


def _draw_sample_space(ax, xlim, ylim, title, *, bg_fill=None,
                       bg_stroke=None, label_color=None):
    """Draw the outer 'S' sample-space rounded rectangle."""
    bg_fill = bg_fill or _BG_FILL
    bg_stroke = bg_stroke or _BG_STROKE
    label_color = label_color or _LABEL_COLOR
    x0, x1 = xlim
    y0, y1 = ylim
    w, h = x1 - x0, y1 - y0
    rect = patches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.06",
        facecolor=bg_fill,
        edgecolor=bg_stroke,
        linewidth=1.8,
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 0.08, y1 - 0.12, "S",
        fontsize=14, fontweight="bold",
        color=bg_stroke, fontfamily=_FONT_FAMILY,
        zorder=1,
    )
    pad = 0.15
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    if title:
        ax.set_title(
            title, fontsize=13, fontweight="semibold",
            color=label_color, fontfamily=_FONT_FAMILY,
            pad=12,
        )
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════
# PLY Lexer — mirrors the official Mermaid jison lexer (case-insensitive)
# ═══════════════════════════════════════════════════════════════════════════

tokens = (
    "VENN",
    "SET",
    "UNION",
    "TEXT",
    "STYLE",
    "TITLE",
    "BRACKET_LABEL",
    "NUMERIC",
    "HEXCOLOR",
    "RGBCOLOR",
    "RGBACOLOR",
    "IDENTIFIER",
    "STRING",
    "COMMA",
    "COLON",
    "NEWLINE",
)

# Track indent-based text attachment (mirrors jison indentMode)
_indent_mode = False
_current_sets = None


def _reset_lex_state():
    global _indent_mode, _current_sets
    _indent_mode = False
    _current_sets = None


# ── Simple tokens ─────────────────────────────────────────────────────────
t_COMMA = r","
t_COLON = r":"


# ── Ordered function-based tokens (longest match first) ──────────────────

def t_COMMENT(t):
    r"%%[^\n]*"
    pass  # discard comments


def t_NEWLINE(t):
    r"\n+"
    t.lexer.lineno += len(t.value)
    return t


def t_TITLE(t):
    r"(?i)title\s+[^\n]+"
    t.value = t.value.split(None, 1)[1].strip().strip('"')
    return t


def t_VENN(t):
    r"(?i)venn-beta"
    return t


def t_SET(t):
    r"(?i)(?<![A-Za-z0-9_])set(?![A-Za-z0-9_])"
    return t


def t_UNION(t):
    r"(?i)(?<![A-Za-z0-9_])union(?![A-Za-z0-9_])"
    return t


def t_TEXT(t):
    r"(?i)(?<![A-Za-z0-9_])text(?![A-Za-z0-9_])"
    return t


def t_STYLE(t):
    r"(?i)(?<![A-Za-z0-9_])style(?![A-Za-z0-9_])"
    return t


def t_BRACKET_LABEL(t):
    r'\["[^"]*"\]|\[[^\]"]+\]'
    # Strip outer [" ... "] or [ ... ]
    inner = t.value[1:-1]
    if inner.startswith('"') and inner.endswith('"'):
        inner = inner[1:-1]
    t.value = inner.strip()
    return t


def t_RGBACOLOR(t):
    r"rgba\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*\)"
    return t


def t_RGBCOLOR(t):
    r"rgb\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+\s*\)"
    return t


def t_NUMERIC(t):
    r"[+-]?(\d+(\.\d+)?|\.\d+)"
    t.value = float(t.value) if "." in t.value else int(t.value)
    return t


def t_HEXCOLOR(t):
    r"\#[0-9a-fA-F]{3,8}"
    return t


def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value[1:-1]
    return t


def t_IDENTIFIER(t):
    r"[A-Za-z_][A-Za-z0-9\-_]*"
    # Check if this is a keyword that slipped through (shouldn't with lookbehind)
    upper = t.value.upper()
    if upper == "SET":
        t.type = "SET"
    elif upper == "UNION":
        t.type = "UNION"
    elif upper == "TEXT":
        t.type = "TEXT"
    elif upper == "STYLE":
        t.type = "STYLE"
    return t


t_ignore = " \t"


def t_error(t):
    t.lexer.skip(1)


# Build lexer (suppress PLY's output file generation)
_lexer = lex.lex(errorlog=lex.NullLogger())


# ═══════════════════════════════════════════════════════════════════════════
# PLY Parser (LALR) — mirrors the official Mermaid jison grammar
# ═══════════════════════════════════════════════════════════════════════════

# Accumulator filled by parser actions (reset per parse call)
_parse_result: dict = {}


def _reset_parse_result():
    global _parse_result
    _parse_result = {
        "title": "",
        "sets": {},
        "set_order": [],
        "unions": {},
        "styles": {},
        "last_target": None,
    }


# ── Grammar rules (highest-level first) ──────────────────────────────────

def p_start(p):
    """start : opt_newlines VENN document"""
    pass


def p_opt_newlines_empty(p):
    """opt_newlines : """
    pass


def p_opt_newlines(p):
    """opt_newlines : opt_newlines NEWLINE"""
    pass


def p_document_empty(p):
    """document : """
    pass


def p_document_line(p):
    """document : document line"""
    pass


def p_line_newline(p):
    """line : NEWLINE"""
    pass


def p_line_statement(p):
    """line : statement"""
    pass


def p_line_statement_newline(p):
    """line : statement NEWLINE"""
    pass


# ── SET statements ────────────────────────────────────────────────────────

def p_statement_set(p):
    """statement : SET identifier"""
    sid = p[2]
    _add_set(sid, None, None)


def p_statement_set_label(p):
    """statement : SET identifier BRACKET_LABEL"""
    _add_set(p[2], p[3], None)


def p_statement_set_size(p):
    """statement : SET identifier COLON NUMERIC"""
    _add_set(p[2], None, p[4])


def p_statement_set_label_size(p):
    """statement : SET identifier BRACKET_LABEL COLON NUMERIC"""
    _add_set(p[2], p[3], p[5])


def _add_set(sid, label, size):
    r = _parse_result
    display = (label or sid).replace("<br/>", "\n")
    r["sets"][sid] = {"label": display, "size": size, "texts": []}
    if sid not in r["set_order"]:
        r["set_order"].append(sid)
    r["last_target"] = ("set", sid)


# ── UNION statements ─────────────────────────────────────────────────────

def p_statement_union(p):
    """statement : UNION multi_id_list"""
    _add_union(p[2], None, None)


def p_statement_union_label(p):
    """statement : UNION multi_id_list BRACKET_LABEL"""
    _add_union(p[2], p[3], None)


def p_statement_union_size(p):
    """statement : UNION multi_id_list COLON NUMERIC"""
    _add_union(p[2], None, p[4])


def p_statement_union_label_size(p):
    """statement : UNION multi_id_list BRACKET_LABEL COLON NUMERIC"""
    _add_union(p[2], p[3], p[5])


def _add_union(id_list, label, size):
    if len(id_list) < 2:
        raise SyntaxError("union requires at least 2 identifiers")
    uid = ",".join(id_list)
    display = (label or "").replace("<br/>", "\n")
    r = _parse_result
    r["unions"][uid] = {"label": display, "size": size, "texts": []}
    r["last_target"] = ("union", uid)


# ── TEXT statements ───────────────────────────────────────────────────────

# text id["label"]  (indented, attaches to last set/union)
def p_statement_text_id(p):
    """statement : TEXT identifier"""
    _add_text_to_last(str(p[2]))


def p_statement_text_id_label(p):
    """statement : TEXT identifier BRACKET_LABEL"""
    _add_text_to_last(p[3])


def p_statement_text_numeric(p):
    """statement : TEXT NUMERIC"""
    _add_text_to_last(str(p[2]))


# text with explicit target list: text A,B myid["label"]
def p_statement_text_list_id(p):
    """statement : TEXT multi_id_list identifier"""
    _add_text_explicit(p[2], str(p[3]))


def p_statement_text_list_id_label(p):
    """statement : TEXT multi_id_list identifier BRACKET_LABEL"""
    _add_text_explicit(p[2], p[4])


def p_statement_text_list_string(p):
    """statement : TEXT multi_id_list STRING"""
    _add_text_explicit(p[2], p[3])


def p_statement_text_list_string_label(p):
    """statement : TEXT multi_id_list STRING BRACKET_LABEL"""
    _add_text_explicit(p[2], p[4])


def p_statement_text_list_numeric(p):
    """statement : TEXT multi_id_list NUMERIC"""
    _add_text_explicit(p[2], str(p[3]))


def _add_text_to_last(txt):
    r = _parse_result
    txt = txt.replace("<br/>", "\n")
    lt = r["last_target"]
    if lt is None:
        return
    kind, key = lt
    target = r["sets"] if kind == "set" else r["unions"]
    if key in target:
        target[key]["texts"].append(txt)


def _add_text_explicit(id_list, txt):
    r = _parse_result
    txt = txt.replace("<br/>", "\n")
    uid = ",".join(id_list)
    # Try unions first, then first set in list
    if uid in r["unions"]:
        r["unions"][uid]["texts"].append(txt)
    elif id_list[0] in r["sets"]:
        r["sets"][id_list[0]]["texts"].append(txt)


# ── TITLE statement ───────────────────────────────────────────────────────

def p_statement_title(p):
    """statement : TITLE"""
    _parse_result["title"] = p[1]


# ── STYLE statements ─────────────────────────────────────────────────────

def p_statement_style(p):
    """statement : STYLE id_list styles_opt"""
    r = _parse_result
    for target_id in p[2]:
        if target_id not in r["styles"]:
            r["styles"][target_id] = {}
        for prop, val in p[3]:
            r["styles"][target_id][prop] = val


def p_styles_opt_single(p):
    """styles_opt : style_field"""
    p[0] = [p[1]]


def p_styles_opt_multi(p):
    """styles_opt : styles_opt COMMA style_field"""
    p[0] = p[1] + [p[3]]


def p_style_field(p):
    """style_field : IDENTIFIER COLON style_value"""
    p[0] = (p[1], p[3])


def p_style_value_string(p):
    """style_value : STRING"""
    p[0] = p[1]


def p_style_value_tokens(p):
    """style_value : value_tokens"""
    p[0] = " ".join(p[1])


def p_value_tokens_single(p):
    """value_tokens : value_token"""
    p[0] = [p[1]]


def p_value_tokens_multi(p):
    """value_tokens : value_tokens value_token"""
    p[0] = p[1] + [p[2]]


def p_value_token_id(p):
    """value_token : IDENTIFIER"""
    p[0] = p[1]


def p_value_token_numeric(p):
    """value_token : NUMERIC"""
    p[0] = str(p[1])


def p_value_token_hex(p):
    """value_token : HEXCOLOR"""
    p[0] = p[1]


def p_value_token_rgb(p):
    """value_token : RGBCOLOR"""
    p[0] = p[1]


def p_value_token_rgba(p):
    """value_token : RGBACOLOR"""
    p[0] = p[1]


# ── identifier lists ──────────────────────────────────────────────────────

# id_list: 1 or more identifiers (for style targets)
def p_id_list_single(p):
    """id_list : identifier"""
    p[0] = [p[1]]


def p_id_list_multi(p):
    """id_list : id_list COMMA identifier"""
    p[0] = p[1] + [p[3]]


# multi_id_list: 2 or more identifiers (for union, explicit text targets)
def p_multi_id_list_pair(p):
    """multi_id_list : identifier COMMA identifier"""
    p[0] = [p[1], p[3]]


def p_multi_id_list_append(p):
    """multi_id_list : multi_id_list COMMA identifier"""
    p[0] = p[1] + [p[3]]


# ── identifier (bare word or quoted string) ───────────────────────────────

def p_identifier_id(p):
    """identifier : IDENTIFIER"""
    p[0] = p[1]


def p_identifier_string(p):
    """identifier : STRING"""
    p[0] = p[1]


# ── Error rule ────────────────────────────────────────────────────────────

def p_error(p):
    if p:
        raise SyntaxError(
            f"venn-beta parse error at line {p.lineno}, token {p.type!r} = {p.value!r}"
        )
    raise SyntaxError("venn-beta parse error: unexpected end of input")


# Build parser once (suppress PLY file generation)
_parser = yacc.yacc(
    debug=False,
    write_tables=False,
    errorlog=yacc.NullLogger(),
    outputdir=os.devnull if sys.platform != "win32" else None,
)


# ── Public parse entry point ──────────────────────────────────────────────

def _parse(diagram: str):
    """Return (title, sets, set_order, unions, styles) from venn-beta text."""
    _reset_lex_state()
    _reset_parse_result()
    _lexer.lineno = 1
    _parser.parse(diagram.strip() + "\n", lexer=_lexer)
    r = _parse_result
    return r["title"], r["sets"], r["set_order"], r["unions"], r["styles"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def venn(diagram, color_scheme: Color | None = None, figsize: tuple[float, float] | None = None):
    """Parse and render a Mermaid venn-beta diagram with matplotlib.

    Args:
        diagram: Mermaid venn-beta syntax string.
        color_scheme: Optional ``Color`` enum member to override the default
            Material Design palette.  Per-set ``style`` directives in the
            diagram still take precedence.
        figsize: Optional ``(width, height)`` tuple in inches for the figure.
            Defaults to ``(9, 5.5)``.
    """
    title, sets, set_order, unions, styles = _parse(diagram)
    n = len(set_order)
    if n < 2 or n > 3:
        raise ValueError(f"venn() supports 2-3 sets, got {n}")

    # Resolve palette
    cs = (color_scheme or _DEFAULT_SCHEME).value
    circle_colors = cs["circles"]
    bg_fill = cs["bg_fill"]
    bg_stroke = cs["bg_stroke"]
    label_color = cs["label"]
    sub_color = cs["sub"]

    fig, ax = plt.subplots(figsize=figsize or (9, 5.5))
    fig.patch.set_facecolor("white")

    def _find_union(*sids):
        for key in unions:
            if set(key.split(",")) == set(sids):
                return unions[key]
        return None

    def _circ(sid, idx, center, radius):
        fill = _style_val(styles, sid, "fill", circle_colors[idx % len(circle_colors)])
        opacity = float(_style_val(styles, sid, "fill-opacity", str(_CIRCLE_ALPHA)))
        stroke = _style_val(styles, sid, "stroke", _CIRCLE_STROKE)
        lw = float(_style_val(styles, sid, "stroke-width", str(_CIRCLE_LW)))
        _draw_circle(ax, center, radius, fill, opacity, stroke, lw)

    def _color(sid):
        return _style_val(styles, sid, "color", label_color)

    # ── 2-set layout ──────────────────────────────────────────────────────
    if n == 2:
        a, b = set_order
        u = _find_union(a, b)
        has_union = u is not None

        r = 0.72 if has_union else 0.55
        sep = 0.58 if has_union else 1.20
        xa, xb = -sep / 2, sep / 2

        _circ(a, 0, (xa, 0), r)
        _circ(b, 1, (xb, 0), r)

        if has_union:
            # labels pushed outward from center
            _put_label(ax, xa - r * 0.42, 0.08, sets[a]["label"], color=_color(a))
            _put_label(ax, xb + r * 0.42, 0.08, sets[b]["label"], color=_color(b))
            if u["label"]:
                _put_label(ax, 0, 0.08, u["label"], fontsize=10, color=label_color)
            # text nodes
            for i, t in enumerate(sets[a]["texts"]):
                _put_sub(ax, xa - r * 0.42, -0.14 - i * 0.20, t, color=sub_color)
            for i, t in enumerate(sets[b]["texts"]):
                _put_sub(ax, xb + r * 0.42, -0.14 - i * 0.20, t, color=sub_color)
            for i, t in enumerate(u["texts"]):
                _put_sub(ax, 0, -0.14 - i * 0.20, t, color=sub_color)
        else:
            _put_label(ax, xa, 0.08, sets[a]["label"], color=_color(a))
            _put_label(ax, xb, 0.08, sets[b]["label"], color=_color(b))
            for i, t in enumerate(sets[a]["texts"]):
                _put_sub(ax, xa, -0.14 - i * 0.20, t, color=sub_color)
            for i, t in enumerate(sets[b]["texts"]):
                _put_sub(ax, xb, -0.14 - i * 0.20, t, color=sub_color)

        margin = 0.45
        box_x0 = min(xa - r, xb - r) - margin
        box_x1 = max(xa + r, xb + r) + margin
        box_y0 = -r - margin
        box_y1 = r + margin
        _draw_sample_space(ax, (box_x0, box_x1), (box_y0, box_y1), title,
                           bg_fill=bg_fill, bg_stroke=bg_stroke,
                           label_color=label_color)

    # ── 3-set layout ──────────────────────────────────────────────────────
    elif n == 3:
        r = 0.58
        positions = [(0, 0.28), (-0.34, -0.22), (0.34, -0.22)]
        label_offsets = [(0, 0.46), (-0.46, -0.42), (0.46, -0.42)]

        for i, sid in enumerate(set_order):
            _circ(sid, i, positions[i], r)
            lx = positions[i][0] + label_offsets[i][0]
            ly = positions[i][1] + label_offsets[i][1]
            _put_label(ax, lx, ly, sets[sid]["label"], color=_color(sid))
            for j, t in enumerate(sets[sid]["texts"]):
                _put_sub(ax, lx, ly - 0.20 * (j + 1), t, color=sub_color)

        for uid, uinfo in unions.items():
            parts = uid.split(",")
            idxs = [set_order.index(p) for p in parts if p in set_order]
            mx = sum(positions[i][0] for i in idxs) / len(idxs)
            my = sum(positions[i][1] for i in idxs) / len(idxs)
            if uinfo["label"]:
                _put_label(ax, mx, my, uinfo["label"], fontsize=10, color=label_color)
            for j, t in enumerate(uinfo["texts"]):
                _put_sub(ax, mx, my - 0.20 * (j + 1), t, color=sub_color)

        margin = 0.50
        all_x = [p[0] for p in positions]
        all_y = [p[1] for p in positions]
        box_x0 = min(all_x) - r - margin
        box_x1 = max(all_x) + r + margin
        box_y0 = min(all_y) - r - margin
        box_y1 = max(all_y) + r + margin
        _draw_sample_space(ax, (box_x0, box_x1), (box_y0, box_y1), title,
                           bg_fill=bg_fill, bg_stroke=bg_stroke,
                           label_color=label_color)

    plt.tight_layout()
    plt.show()
