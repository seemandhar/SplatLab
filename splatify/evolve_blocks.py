"""
EVOLVE-BLOCK — per-hook code regions for targeted mutation.

Inspired by AlphaEvolve's marked code regions. Each of the 9 GS-template
extension hooks becomes an independently evolvable block in a generated
method file. A mutation operator targets one (or several) block(s) and
rewrites *only* the body between its markers — the rest of the file
(imports, class scaffolding, unrelated hooks) is guaranteed untouched.

Marker format (Python-comment based, parseable by regex):

    # EVOLVE-BLOCK-START <hook_name>
    def <hook_name>(self, ...):
        ...body...
    # EVOLVE-BLOCK-END <hook_name>

The 9 hooks map 1:1 to the extension points in
splatify/gs_template/model.py. `HOOK_NAMES` is the canonical list; the
annotator finds each `def <hook>` in a generated model file and wraps
it with the markers if they're not already present.

Public API:
    HOOK_NAMES                          — canonical hook list
    annotate_method_file(path) -> int   — add missing markers in-place
    extract_block(source, hook) -> str  — read a block's body
    replace_block(source, hook, body)   — substitute a block's body
    list_blocks(source) -> list[str]    — which hooks currently exist
    guarded_diff_prompt(hooks) -> str   — instructions for mutator agents

This module is pure text manipulation; it never imports nerfstudio or
touches model code at runtime.
"""
from __future__ import annotations

import re
from pathlib import Path

# Canonical list of evolvable hooks. Order matches the comments in
# splatify/gs_template/model.py.
HOOK_NAMES: tuple[str, ...] = (
    "populate_modules",
    "get_outputs",
    "get_loss_dict",
    "compute_custom_losses",
    "split_gaussians",
    "get_gaussian_param_groups",
    "get_param_groups",
    "get_outputs_for_camera",
    "get_metrics_dict",
)

START_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)#\s*EVOLVE-BLOCK-START\s+(?P<hook>\w+)\s*$",
    re.MULTILINE,
)
END_MARKER_RE = re.compile(
    r"^[ \t]*#\s*EVOLVE-BLOCK-END\s+(?P<hook>\w+)\s*$",
    re.MULTILINE,
)

# Match a method definition at a given indentation.
def _method_def_re(hook: str) -> re.Pattern:
    return re.compile(
        r"(?P<leading>^|\n)(?P<indent>[ \t]*)def " + re.escape(hook)
        + r"\s*\(",
        re.MULTILINE,
    )


def list_blocks(source: str) -> list[str]:
    """Return the hook names that already have EVOLVE-BLOCK markers."""
    return [m.group("hook") for m in START_MARKER_RE.finditer(source)]


def _find_method_span(source: str, hook: str) -> tuple[int, int, str] | None:
    """Find (start, end, indent) byte offsets of a method body.

    Returns the offsets of the entire `def hook(...): ...` block up to
    but not including the next sibling or class-dedent. Indent is the
    leading whitespace of the `def` line.
    """
    m = _method_def_re(hook).search(source)
    if not m:
        return None
    def_start = m.start() + len(m.group("leading"))
    indent = m.group("indent")
    body_indent = indent + "    "

    lines = source[def_start:].splitlines(keepends=True)
    # First line = def; from the next non-blank line onward, a line at
    # a column <= indent (and non-blank, non-comment) ends the method.
    consumed = len(lines[0])
    for line in lines[1:]:
        stripped = line.strip()
        leading = len(line) - len(line.lstrip(" "))
        if stripped == "" or stripped.startswith("#"):
            consumed += len(line)
            continue
        if leading <= len(indent):
            break
        consumed += len(line)
    return def_start, def_start + consumed, indent


def annotate_method_file(path: Path) -> int:
    """Insert EVOLVE-BLOCK markers around every supported hook in-place.

    Returns the number of markers added. Idempotent: a second run on a
    fully annotated file returns 0.
    """
    path = Path(path)
    if not path.exists():
        return 0
    source = path.read_text()
    original = source
    already = set(list_blocks(source))

    # Work bottom-up so offsets remain valid after each insertion.
    edits: list[tuple[int, int, str]] = []
    for hook in HOOK_NAMES:
        if hook in already:
            continue
        span = _find_method_span(source, hook)
        if span is None:
            continue
        start, end, indent = span
        prefix = f"{indent}# EVOLVE-BLOCK-START {hook}\n"
        suffix = f"{indent}# EVOLVE-BLOCK-END {hook}\n"
        edits.append((start, end, prefix + source[start:end] + suffix))

    if not edits:
        return 0
    edits.sort(reverse=True)
    for start, end, new_text in edits:
        source = source[:start] + new_text + source[end:]

    if source != original:
        path.write_text(source)
    return len(edits)


def annotate_implementation(impl_dir: Path) -> dict[str, int]:
    """Annotate every model.py-like file under an implementation directory.

    Returns a mapping path -> markers-added. Used once at island-0
    creation so every subsequent mutation works against marked code.
    """
    impl_dir = Path(impl_dir)
    results: dict[str, int] = {}
    if not impl_dir.exists():
        return results
    for py in impl_dir.rglob("*.py"):
        # Skip tests / setup / __init__ noise.
        if py.name.startswith("test_") or py.name in {"setup.py", "__init__.py"}:
            continue
        try:
            added = annotate_method_file(py)
        except Exception:
            added = 0
        if added:
            results[str(py)] = added
    return results


def extract_block(source: str, hook: str) -> str | None:
    """Return the text between the markers for `hook`, or None."""
    pattern = re.compile(
        r"^[ \t]*#\s*EVOLVE-BLOCK-START\s+" + re.escape(hook) + r"\s*\n"
        r"(?P<body>.*?)"
        r"^[ \t]*#\s*EVOLVE-BLOCK-END\s+" + re.escape(hook) + r"\s*$",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(source)
    return m.group("body") if m else None


def replace_block(source: str, hook: str, new_body: str) -> str:
    """Substitute the body of a block. Returns new source.

    The new body should include its own leading indentation. If the
    block doesn't exist, `source` is returned unchanged.
    """
    pattern = re.compile(
        r"(?P<head>^[ \t]*#\s*EVOLVE-BLOCK-START\s+" + re.escape(hook) + r"\s*\n)"
        r"(?P<body>.*?)"
        r"(?P<tail>^[ \t]*#\s*EVOLVE-BLOCK-END\s+" + re.escape(hook) + r"\s*$)",
        re.MULTILINE | re.DOTALL,
    )
    return pattern.sub(
        lambda m: m.group("head") + new_body.rstrip("\n") + "\n" + m.group("tail"),
        source,
        count=1,
    )


def extract_all_blocks(source: str) -> dict[str, str]:
    """Return {hook: body} for every marked block in `source`."""
    out: dict[str, str] = {}
    for hook in HOOK_NAMES:
        body = extract_block(source, hook)
        if body is not None:
            out[hook] = body
    return out


def find_marked_files(impl_dir: Path) -> list[Path]:
    """All files in an implementation dir that contain any EVOLVE block."""
    impl_dir = Path(impl_dir)
    hits: list[Path] = []
    if not impl_dir.exists():
        return hits
    for py in impl_dir.rglob("*.py"):
        try:
            if "EVOLVE-BLOCK-START" in py.read_text():
                hits.append(py)
        except Exception:
            pass
    return hits


def guarded_diff_prompt(hooks: list[str]) -> str:
    """Instructions appended to a mutation agent's prompt.

    Tells the agent exactly which blocks it may touch and reminds it
    that anything outside the markers is off-limits. The `hooks` list
    specifies which ones are in scope for this mutation.
    """
    hooks_str = ", ".join(f"`{h}`" for h in hooks) if hooks else "any EVOLVE block"
    return (
        "\n\nEVOLVE-BLOCK PROTOCOL (strict):\n"
        f"- You may ONLY modify code between the markers of: {hooks_str}\n"
        "- Markers look like:\n"
        "    # EVOLVE-BLOCK-START <hook_name>\n"
        "    ... code ...\n"
        "    # EVOLVE-BLOCK-END <hook_name>\n"
        "- Do NOT remove or move the markers themselves.\n"
        "- Do NOT modify code outside any EVOLVE-BLOCK markers — this\n"
        "  includes imports, class declarations, unrelated hooks, and\n"
        "  pyproject.toml unless explicitly instructed.\n"
        "- If the change requires a new import, add it at the top of\n"
        "  the same file; that is the only exception.\n"
    )
