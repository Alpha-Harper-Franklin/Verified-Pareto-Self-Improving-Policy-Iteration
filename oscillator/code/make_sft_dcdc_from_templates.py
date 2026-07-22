#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from dcdc_taskset import default_taskset
from dcdc_templates import templates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--template_variant", choices=["base", "full"], default="full")
    ap.add_argument("--repeat_per_task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tpl: Dict[str, str] = templates(str(args.template_variant))
    tasks = default_taskset()

    # Keep prompts consistent with evaluation.
    from eval_dcdc_family import build_prompt

    rows: List[str] = []
    for t in tasks:
        inc = str(tpl.get(t.family, "")).strip()
        if not inc:
            continue
        inc = inc + ("\n" if not inc.endswith("\n") else "")
        prompt = build_prompt(str(t.family), float(t.vin), float(t.vout))
        for _ in range(max(1, int(args.repeat_per_task))):
            rows.append(prompt + inc)

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for text in rows:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    report = {
        "out_jsonl": str(out_path),
        "template_variant": str(args.template_variant),
        "repeat_per_task": int(args.repeat_per_task),
        "n_tasks": int(len(tasks)),
        "n_rows": int(len(rows)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

