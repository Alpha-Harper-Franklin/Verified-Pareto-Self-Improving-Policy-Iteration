from __future__ import annotations

from typing import List

from dcdc_taskset import Task

# Sub-families inside the amplifier branch (single adapter, hard-routed at branch level).
AMP_FAMILY_OP2 = "amp_op2"      # two-stage op-amp (closed-loop)
AMP_FAMILY_RFPA = "amp_rfpa"    # RF / high-speed power amplifier driver (heavier load)

# Reduced-difficulty curriculum: fix gain, vary bandwidth only.
AMP_FIXED_GAIN_DB = 6.0

# 5-point bandwidth targets (Hz), chosen to prioritize easier/low-to-mid targets.
AMP_OP2_BW_HZ = [1e4, 3e4, 1e5, 3e5, 1e6]
AMP_RFPA_BW_HZ = [1e6, 2e6, 3e6, 5e6, 1e7]


def default_taskset_amp() -> List[Task]:
    """Amplifier taskset (reduced difficulty).

    We reuse the shared Task(family, vin, vout) schema:
      - family: amplifier sub-family (amp_op2 / amp_rfpa)
      - vin:    target closed-loop gain in dB (fixed to AMP_FIXED_GAIN_DB)
      - vout:   target -3dB bandwidth in Hz (5 point-spec targets)

    Global constraints (phase margin, power budget, load, core parameters) are
    family-specific and enforced by the evaluator in amp_eval_acop.py.

    Design choice:
      - This curriculum fixes gain to reduce combinatorial difficulty and
        concentrates learning on bandwidth control/stability.
    """

    out: List[Task] = []

    for bw in AMP_OP2_BW_HZ:
        out.append(Task(AMP_FAMILY_OP2, float(AMP_FIXED_GAIN_DB), float(bw)))

    for bw in AMP_RFPA_BW_HZ:
        out.append(Task(AMP_FAMILY_RFPA, float(AMP_FIXED_GAIN_DB), float(bw)))

    return out


# Guard set for non-regression: use ALL amplifier targets (10) to avoid duplicate/partial evals.
REDLINE_TASKS_AMP: List[Task] = list(default_taskset_amp())
