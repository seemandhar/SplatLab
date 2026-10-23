#!/usr/bin/env python3
"""Patch config.yaml from a JSON object on stdin.

Used by the SplatLab UI backend to seed a launch. Free-form idea text is
written into config.yaml's `idea:` field (the documented `/research` positional
fallback) so it never has to be injected into a shell/tmux command line.

Only a fixed whitelist of keys is touched; every other key in config.yaml is
preserved verbatim. Reads JSON from stdin, writes config.yaml in place.

Usage:  echo '{"idea": "...", "hours": 24}' | python3 patch_config.py /path/to/config.yaml
"""
import json
import sys

import yaml

ALLOWED = {
    "idea": str,
    "hours": int,
    "paper_url": str,
    "git_url": str,
    "model_preset": str,
    "max_budget_usd": (int, float, type(None)),
    "scenes": list,
}


def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    payload = json.load(sys.stdin)

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    for key, val in payload.items():
        if key not in ALLOWED:
            continue
        if val is None and key not in ("max_budget_usd",):
            continue
        if key == "hours":
            val = int(val)
        elif key == "scenes" and isinstance(val, list):
            val = [str(s) for s in val if str(s).strip()]
            if not val:
                continue
        elif key == "max_budget_usd":
            val = None if val in (None, "", 0) else float(val)
        else:
            val = val if val is None else str(val)
        # blank strings (paper_url / git_url cleared) -> drop, don't overwrite with ""
        if isinstance(val, str) and not val.strip() and key != "idea":
            cfg[key] = None
            continue
        cfg[key] = val

    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(json.dumps({"ok": True, "patched": [k for k in payload if k in ALLOWED]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
