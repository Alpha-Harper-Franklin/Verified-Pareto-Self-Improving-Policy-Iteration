#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_OFFICIAL_JSONL = "/root/autodl-tmp/datasets/official_power_v1/train.jsonl"
DEFAULT_PAT = r"(dcdc|dc-?dc|buck|boost|sepic|buck-?boost|converter)"
_NAME_RE = re.compile(r"^[RLCSD][A-Za-z0-9_]*$")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _as_text(prompt: str, inc: str, response_template: str) -> str:
    p = (prompt or "").rstrip()
    body = (inc or "").strip()
    if not p:
        p = "Generate a valid DC/power-related circuit in INC DSL. Output only INC lines."
    if not body.endswith("\n"):
        body += "\n"
    return p + "\n" + str(response_template) + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_jsonl", default=DEFAULT_OFFICIAL_JSONL)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--pattern", default=DEFAULT_PAT, help="Regex applied to source/inc/prompt (case-insensitive).")
    ap.add_argument("--max_rows", type=int, default=0, help="0 means no limit")
    ap.add_argument("--response_template", default="### Response:\n")
    ap.add_argument("--require_inc_prefix", action="store_true", help="Only keep samples whose inc lines all start with 'INC '.")
    ap.add_argument(
        "--require_rlcsd_only",
        action="store_true",
        help="Only keep samples whose every non-empty INC line is exactly 4 tokens and name starts with {R,L,C,S,D}.",
    )
    args = ap.parse_args()

    src = Path(args.official_jsonl)
    if not src.exists():
        raise SystemExit(f"missing official dataset: {src}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pat = re.compile(str(args.pattern), re.IGNORECASE)

    kept: List[Tuple[str, str]] = []
    n_total = 0
    n_match = 0
    n_bad = 0

    for r in _iter_jsonl(src):
        n_total += 1
        prompt = str(r.get("prompt") or "")
        inc = str(r.get("inc") or "")
        source = str(r.get("source") or "")

        blob = "\n".join([source, prompt, inc])
        if not pat.search(blob):
            continue
        n_match += 1

        if not inc.strip():
            n_bad += 1
            continue

        if bool(args.require_inc_prefix):
            ok = True
            for line in inc.splitlines():
                if not line.strip():
                    continue
                if not line.lstrip().startswith("INC "):
                    ok = False
                    break
            if not ok:
                n_bad += 1
                continue

        if bool(args.require_rlcsd_only):
            ok = True
            for line in inc.splitlines():
                s = (line or "").strip()
                if not s:
                    continue
                if not s.startswith("INC "):
                    ok = False
                    break
                toks = s.split()
                if len(toks) < 4:
                    ok = False
                    break
                name = toks[1]
                if not _NAME_RE.match(name or ""):
                    ok = False
                    break
            if not ok:
                n_bad += 1
                continue

        kept.append((prompt, inc))
        if int(args.max_rows) > 0 and len(kept) >= int(args.max_rows):
            break

    with out_path.open("w", encoding="utf-8") as f:
        for prompt, inc in kept:
            f.write(json.dumps({"text": _as_text(prompt, inc, args.response_template)}, ensure_ascii=False) + "\n")

    report = {
        "official_jsonl": str(src),
        "out_jsonl": str(out_path),
        "pattern": str(args.pattern),
        "n_total": int(n_total),
        "n_match": int(n_match),
        "n_bad": int(n_bad),
        "n_kept": int(len(kept)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
