from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


FAMILY_ORDER = ["buck", "boost", "sepic", "buckboost"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def build_feature_vector(
    *,
    family: str,
    vin: float,
    vout: float,
    detail_before: Dict[str, Any],
    tunable_values: List[float],
    n_elems: Optional[float] = None,
) -> List[float]:
    fam = str(family or "").strip().lower()
    fam_oh = [1.0 if fam == f else 0.0 for f in FAMILY_ORDER]

    vout_f = max(1e-6, float(vout))
    vin_f = float(vin)

    ok = 1.0 if bool((detail_before or {}).get("ok", False)) else 0.0
    pass_cv = 1.0 if bool((detail_before or {}).get("pass_CV", False)) else 0.0
    pass_ce = 1.0 if bool((detail_before or {}).get("pass_CE", False)) else 0.0

    vavg = _safe_float((detail_before or {}).get("vavg", 0.0), 0.0)
    eff = _safe_float((detail_before or {}).get("eff", 0.0), 0.0)
    ripple = _safe_float((detail_before or {}).get("ripple", 0.0), 0.0)
    overshoot = _safe_float((detail_before or {}).get("overshoot", 0.0), 0.0)

    vavg_ratio = float(vavg / vout_f)
    cv_err = float(abs(vavg - vout_f) / vout_f)
    ripple_n = float(ripple / vout_f)

    if n_elems is None:
        n_elems = _safe_float((detail_before or {}).get("n_elems", 0.0), 0.0)
    n_elems_n = float(n_elems / 40.0)

    vin_n = float(vin_f / 24.0)
    vout_n = float(vout_f / 24.0)

    tun = list(tunable_values or [])[:4]
    while len(tun) < 4:
        tun.append(0.0)
    log_tun = [float(math.log(max(1e-12, abs(float(v))))) if float(v) > 0.0 else 0.0 for v in tun]

    return (
        fam_oh
        + [vin_n, vout_n]
        + [ok, pass_cv, pass_ce]
        + [vavg_ratio, cv_err, eff, ripple_n, float(overshoot)]
        + [n_elems_n]
        + log_tun
    )


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class RepairCorrector:
    model: _MLP
    x_mean: torch.Tensor
    x_std: torch.Tensor
    device: torch.device

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "RepairCorrector":
        p = Path(path)
        obj = torch.load(str(p), map_location="cpu")
        in_dim = int((obj.get("meta") or {}).get("in_dim") or 0)
        if in_dim <= 0:
            raise ValueError(f"Bad corrector checkpoint (missing in_dim): {p}")
        model = _MLP(in_dim=in_dim, out_dim=4, hidden=int((obj.get("meta") or {}).get("hidden") or 128))
        model.load_state_dict(obj["model"])
        dev = torch.device(device)
        model.to(dev)
        model.eval()
        x_mean = torch.tensor(obj["x_mean"], dtype=torch.float32, device=dev)
        x_std = torch.tensor(obj["x_std"], dtype=torch.float32, device=dev)
        return cls(model=model, x_mean=x_mean, x_std=x_std, device=dev)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / torch.clamp(self.x_std, min=1e-6)

    def suggest_scales(
        self,
        *,
        feature_vec: List[float],
        topk: int = 3,
        max_factor: float = 2.0,
    ) -> List[List[float]]:
        topk = max(1, int(topk))
        max_factor = float(max(1.01, float(max_factor)))
        clamp_log = float(math.log(max_factor))

        x = torch.tensor(feature_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dlog = self.model(self._norm(x)).squeeze(0).float().cpu().tolist()

        # Deterministic multi-try around the main prediction.
        scales_list: List[List[float]] = []
        multipliers = [1.0, 0.5, 1.5, -0.5, -1.0]
        for m in multipliers:
            if len(scales_list) >= topk:
                break
            scales: List[float] = []
            for v in dlog[:4]:
                v0 = float(max(-clamp_log, min(clamp_log, float(v) * float(m))))
                scales.append(float(math.exp(v0)))
            while len(scales) < 4:
                scales.append(1.0)
            scales_list.append(scales)

        return scales_list

