from __future__ import annotations

from typing import List

from dcdc_taskset import Task

# Oscillator sub-families inside the osc branch (single adapter, hard-routed at branch level).
OSC_FAMILY_LC = "osc_lc"
OSC_FAMILY_RC = "osc_rc"
OSC_FAMILY_RING = "osc_ring"
OSC_FAMILY_WIEN = "osc_wien"

# Reduced-difficulty curriculum: 5 frequency targets per sub-family.
OSC_VPP_HINT = 1.0

OSC_LC_FREQS_HZ = [1e4, 2e4, 5e4, 1e5, 2e5]
OSC_RC_FREQS_HZ = [1e3, 2e3, 5e3, 1e4, 2e4]
OSC_RING_FREQS_HZ = [1e5, 2e5, 5e5, 1e6, 2e6]
OSC_WIEN_FREQS_HZ = [1e3, 2e3, 5e3, 1e4, 2e4]


def default_taskset_osc() -> List[Task]:
    """Oscillator taskset (reduced difficulty).

    We reuse Task(family, vin, vout):
      - family: oscillator sub-family
      - vin:    target oscillation frequency (Hz)
      - vout:   vpp hint (V) (kept fixed; evaluator currently enforces only a minimum Vpp)

    Design choice:
      - Use 5 mid-range frequencies per family to increase early pass_CV rate.
    """

    out: List[Task] = []

    for f in OSC_LC_FREQS_HZ:
        out.append(Task(OSC_FAMILY_LC, float(f), float(OSC_VPP_HINT)))
    for f in OSC_RC_FREQS_HZ:
        out.append(Task(OSC_FAMILY_RC, float(f), float(OSC_VPP_HINT)))
    for f in OSC_RING_FREQS_HZ:
        out.append(Task(OSC_FAMILY_RING, float(f), float(OSC_VPP_HINT)))
    for f in OSC_WIEN_FREQS_HZ:
        out.append(Task(OSC_FAMILY_WIEN, float(f), float(OSC_VPP_HINT)))

    return out


# Guard set for non-regression: use ALL oscillator targets (20) to avoid duplicate/partial evals.
REDLINE_TASKS_OSC: List[Task] = list(default_taskset_osc())
