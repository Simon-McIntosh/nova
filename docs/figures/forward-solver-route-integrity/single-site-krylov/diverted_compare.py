"""Locate where the diverted rung first differs between the two Krylov bodies.

Reads the trace directories written by ``diverted_trace.py`` and writes
``diverted-move-mechanism.json``: both runs' per-trip residual traces, the
first differing trip and inner iteration, the per-call operator application
counts, and the first differing Krylov call and operator application with
its size in absolute, relative and last-place units.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def ulps(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ia, ib = a.view(np.int64), b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-(2**63)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2**63)) - ib, ib)
    return int(np.max(np.abs(ia - ib))) if a.size else 0


def difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    delta = np.abs(a - b)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return dict(
        max_abs=float(np.max(delta)) if delta.size else 0.0,
        max_rel_to_max=float(np.max(delta)) / scale if delta.size else 0.0,
        max_ulps=ulps(a, b),
        elements_different=int(np.count_nonzero(a != b)),
        size=int(a.size),
        norm_a=float(np.linalg.norm(a)),
    )


def first_mismatch(left, right, key=None):
    for index, (a, b) in enumerate(zip(left, right)):
        if (a if key is None else key(a)) != (b if key is None else key(b)):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


solver = {
    arm: json.loads((ROOT / f"{arm}-plain/solver.json").read_text())
    for arm in ("slice1", "exit")
}
receipt = dict(
    terminal={a: s["terminal_fixed_point_residual"] for a, s in solver.items()}
)


def trips(s):
    return s["production_telemetry"]["per_trip_residual_history"]


def inner(s):
    return s["production_telemetry"]["globalisation_decisions"]


receipt["per_trip_live_relative_residual"] = {
    a: [t["live_relative_residual"] for t in trips(s)] for a, s in solver.items()
}
receipt["per_inner_iteration"] = {a: inner(s) for a, s in solver.items()}
t1, t2 = (trips(solver[a]) for a in ("slice1", "exit"))
receipt["first_differing_trip"] = (
    None
    if (i := first_mismatch(t1, t2)) is None
    else dict(
        trip=i + 1,
        slice1=t1[i] if i < len(t1) else None,
        exit=t2[i] if i < len(t2) else None,
    )
)
n1, n2 = (inner(solver[a]) for a in ("slice1", "exit"))
i = first_mismatch(n1, n2)
if i is not None and i < min(len(n1), len(n2)):
    fields = [k for k in n1[i] if n1[i][k] != n2[i].get(k)]
    receipt["first_differing_inner_iteration"] = dict(
        iteration=n1[i]["iteration"],
        fields=fields,
        slice1={k: n1[i][k] for k in fields},
        exit={k: n2[i][k] for k in fields},
    )
else:
    receipt["first_differing_inner_iteration"] = i

traced = {}
for arm in ("slice1", "exit"):
    path = ROOT / f"{arm}-instrument"
    if not (path / "events.json").exists():
        continue
    summary = json.loads((path / "events.json").read_text())
    traced[arm] = dict(
        summary=summary,
        npz=np.load(path / "events.npz"),
        solver=json.loads((path / "solver.json").read_text()),
    )
if len(traced) == 2:
    s1, s2 = traced["slice1"]["summary"], traced["exit"]["summary"]
    receipt["instrumented_terminal"] = {
        a: t["solver"]["terminal_fixed_point_residual"] for a, t in traced.items()
    }

    def per_call(summary):
        counts, running = [], 0
        for kind, _ in summary["event_digests"]:
            if kind == "apply":
                running += 1
            else:
                counts.append(running)
                running = 0
        return counts

    c1, c2 = per_call(s1), per_call(s2)
    receipt["applications_per_call"] = dict(slice1=c1, exit=c2)
    receipt["calls"] = dict(slice1=len(c1), exit=len(c2))
    receipt["first_call_with_different_application_count"] = first_mismatch(c1, c2)
    e1, e2 = s1["event_digests"], s2["event_digests"]
    k = first_mismatch(e1, e2)
    receipt["first_differing_event_index"] = k
    if k is not None and k < min(len(e1), len(e2)):
        kind = e1[k][0]
        call_index = sum(1 for kd, _ in e1[:k] if kd == "call")
        z1, z2 = traced["slice1"]["npz"], traced["exit"]["npz"]
        first = dict(
            event=k,
            kind=kind,
            kind_exit=e2[k][0],
            krylov_call=call_index + 1,
            application_within_call=None,
        )
        if kind == "apply" and e2[k][0] == "apply":
            first["application_within_call"] = (
                sum(
                    1
                    for kd, _ in e1[:k][::-1][
                        : next(
                            (
                                j
                                for j, (kd2, _) in enumerate(e1[:k][::-1])
                                if kd2 == "call"
                            ),
                            k,
                        )
                    ]
                    if kd == "apply"
                )
                + 1
            )
            first["input"] = difference(z1[f"e{k}_input"], z2[f"e{k}_input"])
            first["output"] = difference(z1[f"e{k}_output"], z2[f"e{k}_output"])
        elif kind == "call" and e2[k][0] == "call":
            for name in ("residual_vector", "step", "unconditioned_step"):
                first[name] = difference(z1[f"e{k}_{name}"], z2[f"e{k}_{name}"])
            for name in (
                "achieved_reduction",
                "projected_condition",
                "qualification",
                "nonlinear_residual",
            ):
                first[name] = dict(
                    slice1=float(z1[f"e{k}_{name}"]), exit=float(z2[f"e{k}_{name}"])
                )
        receipt["first_differing_event"] = first
        # the Krylov call outcomes from the first differing call onward
        calls1 = [j for j, (kd, _) in enumerate(e1) if kd == "call"]
        calls2 = [j for j, (kd, _) in enumerate(e2) if kd == "call"]
        growth = []
        for n, (j1, j2) in enumerate(zip(calls1, calls2)):
            if j1 < k and j2 < k:
                continue
            growth.append(
                dict(
                    krylov_call=n + 1,
                    residual_vector=difference(
                        z1[f"e{j1}_residual_vector"], z2[f"e{j2}_residual_vector"]
                    ),
                    step=difference(z1[f"e{j1}_step"], z2[f"e{j2}_step"]),
                    qualification=[
                        int(z1[f"e{j1}_qualification"]),
                        int(z2[f"e{j2}_qualification"]),
                    ],
                    applications=[
                        c1[n] if n < len(c1) else None,
                        c2[n] if n < len(c2) else None,
                    ],
                )
            )
            if len(growth) >= 40:
                break
        receipt["krylov_calls_after_divergence"] = growth
OUT.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in receipt.items()
            if k
            not in (
                "per_inner_iteration",
                "applications_per_call",
                "krylov_calls_after_divergence",
            )
        },
        indent=1,
        default=str,
    )[:6000]
)
