from __future__ import annotations

from typing import Dict, List


def _join(lines: List[str]) -> str:
    return "\n".join(lines).strip() + "\n"


def templates(variant: str = "base") -> Dict[str, str]:
    v = (variant or "").strip().lower()
    if v in {"base", "simple", "v0"}:
        return {
            # Standard buck: vin->sw->L->out, diode to ground (anode=0, cathode=sw).
            "buck": _join(
                [
                    "INC S1 vin sw Sstd",
                    "INC D1 0 sw Dstd",
                    "INC L1 sw out 47u",
                    "INC C1 out 0 47u",
                ]
            ),
            # Standard boost: L from vin to sw, switch to gnd, diode to out.
            "boost": _join(
                [
                    "INC L1 vin sw 47u",
                    "INC S1 sw 0 Sstd",
                    "INC D1 sw out Dstd",
                    "INC C1 out 0 47u",
                ]
            ),
            # SEPIC: L1 (vin->sw), coupling cap (sw->n1), L2 (n1->0), diode (n1->out), switch (sw->0).
            "sepic": _join(
                [
                    "INC L1 vin sw 47u",
                    "INC C1 sw n1 1u",
                    "INC L2 n1 0 47u",
                    "INC S1 sw 0 Sstd",
                    "INC D1 n1 out Dstd",
                    "INC C2 out 0 47u",
                ]
            ),
            # Buck-Boost (cascaded buck->boost) with 2 gates.
            "buckboost": _join(
                [
                    "INC S1 vin sw1 Sstd1",
                    "INC D1 0 sw1 Dstd",
                    "INC L1 sw1 mid 47u",
                    "INC C1 mid 0 47u",
                    "INC L2 mid sw2 47u",
                    "INC S2 sw2 0 Sstd2",
                    "INC D2 sw2 out Dstd",
                    "INC C2 out 0 47u",
                ]
            ),
        }

    if v in {"full", "v1", "complex"}:
        # NOTE: these are "large-component-count" reference designs (>=20 elements) for the DC-DC power family.
        # They add realistic passive banks + RC snubbers/dampers but keep DC behavior close to the base topology.
        return {
            "buck": _join(
                [
                    # power stage
                    "INC S1 vin sw Sstd",
                    "INC D1 0 sw Dstd",
                    "INC L1 sw out 47u",
                    "INC C1 out 0 47u",
                    # input cap bank
                    "INC C2 vin 0 47u",
                    "INC C3 vin 0 10u",
                    "INC C4 vin 0 1u",
                    "INC C5 vin 0 0.1u",
                    # output cap bank
                    "INC C6 out 0 47u",
                    "INC C7 out 0 10u",
                    "INC C8 out 0 1u",
                    "INC C9 out 0 0.1u",
                    # switch snubbers (series RC -> capacitor blocks DC)
                    "INC R1 sw nsn1 5",
                    "INC C10 nsn1 0 1n",
                    "INC R2 sw nsn2 10",
                    "INC C11 nsn2 0 2n",
                    # output damping RCs
                    "INC R3 out nd1 0.2",
                    "INC C12 nd1 0 1u",
                    "INC R4 out nd2 0.5",
                    "INC C13 nd2 0 0.47u",
                    # cross-node RC (AC path only)
                    "INC R5 sw nd3 50",
                    "INC C14 nd3 out 100p",
                ]
            ),
            "boost": _join(
                [
                    # power stage
                    "INC L1 vin sw 47u",
                    "INC S1 sw 0 Sstd",
                    "INC D1 sw out Dstd",
                    "INC C1 out 0 47u",
                    # input cap bank
                    "INC C2 vin 0 47u",
                    "INC C3 vin 0 10u",
                    "INC C4 vin 0 1u",
                    "INC C5 vin 0 0.1u",
                    # output cap bank
                    "INC C6 out 0 47u",
                    "INC C7 out 0 10u",
                    "INC C8 out 0 1u",
                    "INC C9 out 0 0.1u",
                    # snubbers
                    "INC R1 sw nsn1 5",
                    "INC C10 nsn1 0 1n",
                    "INC R2 sw nsn2 10",
                    "INC C11 nsn2 0 2n",
                    # output damping RCs
                    "INC R3 out nd1 0.2",
                    "INC C12 nd1 0 1u",
                    "INC R4 out nd2 0.5",
                    "INC C13 nd2 0 0.47u",
                    # diode snubber (AC path only)
                    "INC R5 out nd3 50",
                    "INC C14 nd3 sw 100p",
                ]
            ),
            "sepic": _join(
                [
                    # power stage
                    "INC L1 vin sw 47u",
                    "INC C1 sw n1 1u",
                    "INC L2 n1 0 47u",
                    "INC S1 sw 0 Sstd",
                    "INC D1 n1 out Dstd",
                    "INC C2 out 0 47u",
                    # input cap bank
                    "INC C3 vin 0 47u",
                    "INC C4 vin 0 10u",
                    "INC C5 vin 0 1u",
                    "INC C6 vin 0 0.1u",
                    # output cap bank
                    "INC C7 out 0 47u",
                    "INC C8 out 0 10u",
                    "INC C9 out 0 1u",
                    "INC C10 out 0 0.1u",
                    # snubbers (avoid loading node n1; keep extras on sw/out only)
                    "INC R1 sw nsn1 5",
                    "INC C11 nsn1 0 1n",
                    "INC R2 sw nsn2 10",
                    "INC C12 nsn2 0 2n",
                    # output damping RCs
                    "INC R3 out nd1 0.2",
                    "INC C13 nd1 0 1u",
                    "INC R4 out nd2 0.5",
                    "INC C14 nd2 0 0.47u",
                    # cross-node RC (AC path only)
                    "INC R5 sw nd3 50",
                    "INC C15 nd3 out 100p",
                ]
            ),
            "buckboost": _join(
                [
                    # stage1 buck
                    "INC S1 vin sw1 Sstd1",
                    "INC D1 0 sw1 Dstd",
                    "INC L1 sw1 mid 47u",
                    "INC C1 mid 0 47u",
                    # stage2 boost
                    "INC L2 mid sw2 47u",
                    "INC S2 sw2 0 Sstd2",
                    "INC D2 sw2 out Dstd",
                    "INC C2 out 0 47u",
                    # input cap bank
                    "INC C3 vin 0 47u",
                    "INC C4 vin 0 10u",
                    "INC C5 vin 0 1u",
                    "INC C6 vin 0 0.1u",
                    # mid cap bank
                    "INC C7 mid 0 47u",
                    "INC C8 mid 0 10u",
                    # output cap bank
                    "INC C9 out 0 47u",
                    "INC C10 out 0 10u",
                    "INC C11 out 0 1u",
                    "INC C12 out 0 0.1u",
                    # snubbers (each switching node)
                    "INC R1 sw1 nsn1 5",
                    "INC C13 nsn1 0 1n",
                    "INC R2 sw2 nsn2 5",
                    "INC C14 nsn2 0 1n",
                    # damping RCs
                    "INC R3 out nd1 0.2",
                    "INC C15 nd1 0 1u",
                    "INC R4 mid nd2 0.2",
                    "INC C16 nd2 0 1u",
                ]
            ),
        }

    raise ValueError(f"unknown templates variant: {variant}")
