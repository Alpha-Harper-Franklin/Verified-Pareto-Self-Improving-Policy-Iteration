from __future__ import annotations

from typing import List

from dcdc_taskset import Task

# Filter sub-families inside the filter branch (single adapter, hard-routed at branch level).
FILTER_FAMILY_LPF = "filter_lpf"
FILTER_FAMILY_HPF = "filter_hpf"
FILTER_FAMILY_BPF = "filter_bpf"
FILTER_FAMILY_NOTCH = "filter_notch"

# Reduced-difficulty curriculum: use 5 frequency points and fix the secondary spec.
FILTER_FREQS_HZ = [1e4, 2e4, 3e4, 5e4, 1e5]
FILTER_FIXED_ATTEN_DB = 20.0
FILTER_FIXED_BPF_BW_RATIO = 0.20


def default_taskset_filter() -> List[Task]:
    """Filter taskset (reduced difficulty).

    We reuse the shared Task(family, vin, vout) schema:
      - family: filter sub-family
      - vin:    cutoff/center frequency (Hz)
      - vout:   sub-family dependent:
                * lpf/hpf: stopband attenuation target (dB) (fixed)
                * bpf:     target -3dB bandwidth (Hz) (derived from ratio)
                * notch:   notch depth target (dB) (fixed)

    Design choice:
      - We keep only 5 frequency targets and fix the secondary spec to the easiest
        setting to boost early positive sample rate.
    """

    out: List[Task] = []

    # LPF / HPF: cutoff + stopband attenuation (fixed).
    for fc in FILTER_FREQS_HZ:
        out.append(Task(FILTER_FAMILY_LPF, float(fc), float(FILTER_FIXED_ATTEN_DB)))
        out.append(Task(FILTER_FAMILY_HPF, float(fc), float(FILTER_FIXED_ATTEN_DB)))

    # BPF: center + bandwidth (derived).
    for f0 in FILTER_FREQS_HZ:
        out.append(Task(FILTER_FAMILY_BPF, float(f0), float(f0 * FILTER_FIXED_BPF_BW_RATIO)))

    # Notch: center + notch depth (fixed).
    for f0 in FILTER_FREQS_HZ:
        out.append(Task(FILTER_FAMILY_NOTCH, float(f0), float(FILTER_FIXED_ATTEN_DB)))

    return out


# Small redline set (guard) spanning extremes across sub-families.
REDLINE_TASKS_FILTER: List[Task] = [
    Task(FILTER_FAMILY_LPF, 1e4, FILTER_FIXED_ATTEN_DB),
    Task(FILTER_FAMILY_LPF, 1e5, FILTER_FIXED_ATTEN_DB),
    Task(FILTER_FAMILY_HPF, 1e4, FILTER_FIXED_ATTEN_DB),
    Task(FILTER_FAMILY_HPF, 1e5, FILTER_FIXED_ATTEN_DB),

    Task(FILTER_FAMILY_BPF, 1e4, 1e4 * FILTER_FIXED_BPF_BW_RATIO),
    Task(FILTER_FAMILY_BPF, 1e5, 1e5 * FILTER_FIXED_BPF_BW_RATIO),

    Task(FILTER_FAMILY_NOTCH, 1e4, FILTER_FIXED_ATTEN_DB),
    Task(FILTER_FAMILY_NOTCH, 1e5, FILTER_FIXED_ATTEN_DB),
]
