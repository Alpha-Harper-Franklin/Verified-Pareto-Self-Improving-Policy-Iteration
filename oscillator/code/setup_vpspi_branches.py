#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def _copy_adapter(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"anchor_adapter not found: {src}")
    if dst.exists():
        if not force:
            return
        shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor_adapter_src",
        default="/root/autodl-tmp/dcdc_family/runs/sft_anchor_official_plus_tpl_20260110_215415/sft_anchor/sft_final",
        help="Source SFT anchor adapter dir (will be copied).",
    )
    ap.add_argument("--out_root", default="/root/autodl-tmp/vpspi_multitype/branches")
    ap.add_argument("--families", default="amp,filter,osc")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(str(args.anchor_adapter_src)).resolve()
    out_root = Path(str(args.out_root)).resolve()
    families: List[str] = [x.strip().lower() for x in str(args.families).split(",") if x.strip()]
    if not families:
        raise SystemExit("no families")

    branch_map: Dict[str, str] = {}
    for fam in families:
        dst = out_root / fam / "anchor_adapter"
        _copy_adapter(src, dst, force=bool(args.force))
        branch_map[fam] = str(dst)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "branch_map.json").write_text(json.dumps(branch_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[OK] wrote", str(out_root / "branch_map.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
