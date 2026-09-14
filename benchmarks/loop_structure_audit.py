"""Audit which solve budgets the whole-cell certificate program carries.

The whole-cell certificate solve lowers one compiled program per mesh: the
fixed point is traced by ``ForwardProfile._accelerated_history_program`` on the
newton_krylov route with the certificate policy, then executed.  This benchmark
lowers and compiles that exact program on the CPU backend (no execution) and
records the compiled HLO instruction count at baseline and at a doubled value
of every budget the solve carries, one budget at a time.  A ratio near two
names a budget whose trip count is unrolled into the program; a ratio near one
names a scanned loop or a fixed-capacity structure.

Each budget lands its own part JSON as soon as its two compiles finish, so an
allocation expiry loses only the unfinished budgets.  A final pass assembles
the report table and the instruction-count figure from the parts.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.equilibrium import ForwardProfile
from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import ForwardFluxOperator, ForwardSource
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes

from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery

from benchmarks.solovev_certificate import (
    _case,
    _case_machine,
    _certificate_solve_request,
    _closed_form_current_target,
    _production_seed,
)

CASE_NAME = "weak-rotation-reactor-static"
CELLS = -300
MAXSIZE = 5  # production locator capacity resolves to max(5, 30) = 30
BACKEND = "cpu"

# Budgets carried by the certificate solve, with the source line that reads
# each value.  Lines are fixed-point reads; the policy fields live in
# solve_request.ForwardSolvePolicy.
GLOBAL_START = time.perf_counter()


def _wall() -> float:
    return time.perf_counter() - GLOBAL_START


BUDGETS = {
    "newton_steps": {
        "policy_field": "solve_request.py:236",
        "read_lines": ["fixed_point.py:1454-1455", "fixed_point.py:2194"],
        "baseline": 10,
    },
    "gmres_iterations": {
        "policy_field": "solve_request.py:237",
        "read_lines": ["fixed_point.py:1287-1288", "fixed_point.py:1796-1797"],
        "baseline": 30,
    },
    "warmup": {
        "policy_field": "solve_request.py:238",
        "read_lines": ["fixed_point.py:1455", "fixed_point.py:2153"],
        "baseline": 0,  # the certificate pins warmup to zero
    },
    "active_set_steps": {
        "policy_field": "solve_request.py:241",
        "read_lines": ["fixed_point.py:3837"],
        "baseline": 16,
    },
    "backtracking_factors": {
        "policy_field": "fixed_point.py:77 (module tuple)",
        "read_lines": ["fixed_point.py:979", "fixed_point.py:989"],
        "baseline": 6,
    },
    "model_rebuild_damping_trips": {
        "policy_field": "fixed_point.py:85 (module constant)",
        "read_lines": ["fixed_point.py:1337-1339"],
        "baseline": 6,
    },
    "topology_table_capacity": {
        "read_lines": ["forward_operator.py:85", "forward_operator.py:1486-1492"],
        "baseline": 30,  # max(5, _PRODUCTION_STATIONARY_POINT_CAPACITY)
    },
    "profile_node_count": {
        "read_lines": ["forward_operator.py:1459", "forward_operator.py:1867-1886"],
        "baseline": 300,
    },
}


def count_instructions(compiled) -> int:
    """Return the total HLO instruction count across every computation."""
    total = 0
    for module in compiled.runtime_executable().hlo_modules():
        for computation in module.computations():
            total += len(computation.instructions())
    return total


def _operator(source_case, machine, maxsize, exterior=None):
    grid_count = len(machine.node)
    wall_count = len(machine.wall_node)
    if exterior is not None:
        exterior = np.asarray(exterior)
        grid_exterior = exterior[:grid_count]
        wall_exterior = exterior[grid_count : grid_count + wall_count]
        sample_exterior = exterior[grid_count + wall_count :]
    else:
        grid_exterior = wall_exterior = sample_exterior = None
    source = ForwardSource(
        core=oracle_fixture.analytic_profile(source_case),
        boundary_pressure=0.0,
        boundary_field_function=source_case.boundary_f,
    )
    return ForwardFluxOperator(
        grid=oracle_fixture._target(
            machine.node,
            (
                machine.plasma_to_grid,
                machine.plasma_to_grid_r,
                machine.plasma_to_grid_z,
            ),
            Null2D.from_coordinates(
                machine.node.astype(np.float64),
                machine.interior_stencil,
                maxsize=maxsize,
            ),
            grid_exterior,
        ),
        wall=oracle_fixture._target(
            machine.wall_node,
            (
                machine.plasma_to_wall,
                machine.plasma_to_wall_r,
                machine.plasma_to_wall_z,
            ),
            Null1D(jnp.asarray(machine.wall_node, dtype=jnp.float64)),
            wall_exterior,
        ),
        sample=oracle_fixture._target(
            machine.sample_coordinates,
            (
                machine.plasma_to_sample,
                machine.plasma_to_sample_r,
                machine.plasma_to_sample_z,
            ),
            Null1D(jnp.asarray(machine.sample_coordinates, dtype=jnp.float64)),
            sample_exterior,
        ),
        source=source,
        external_current=jnp.ones(1),
        area=jnp.asarray(machine.area),
        polarity=1,
        moment_geometry=machine.moment_geometry,
    )


def build_machine(cells: int):
    carrier_case, source_case, exact = _case(CASE_NAME)
    machine = _case_machine(CASE_NAME, carrier_case, exact, cells)
    return source_case, exact, machine


def build_operator_and_profile(source_case, exact, machine, *, maxsize):
    """Return the production production chain up to the compiled program."""
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = oracle_fixture.exact_state(exact, coordinates)
    empty_operator = _operator(source_case, machine, maxsize=maxsize)
    exact_physical, fixture_exterior, _fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = _operator(
        source_case, machine, maxsize=maxsize, exterior=fixture_exterior
    )
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, mesh, newton_steps=recovery.NEWTON_STEPS)
    target_current, centroid, current_receipt = _closed_form_current_target(
        CASE_NAME, source_case, operator, exact_physical
    )
    seed, _branch, _seed_receipt = _production_seed(
        profile,
        CASE_NAME,
        target_current,
        centroid,
        current_receipt,
    )
    request = _certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"solovev:{CASE_NAME}:{len(machine.node)}",
    )
    initial_flux = request.seed_policy.resolve(profile, current=request.current)
    external = operator.external(None, None)
    return profile, request, initial_flux, external, int(len(machine.node))


def _compile_and_count(profile, request, initial_flux, external, options):
    program = profile._accelerated_history_program(
        request.route,
        requested_class=None,
        target_current=request.target_current,
        **options,
    )
    started = time.perf_counter()
    compiled = program.lower(initial_flux, external).compile()
    wall = time.perf_counter() - started
    return count_instructions(compiled), wall


@contextlib.contextmanager
def _patched(name, value):
    previous = getattr(fixed_point, name)
    setattr(fixed_point, name, value)
    try:
        yield
    finally:
        setattr(fixed_point, name, previous)


def measure_pair(
    part: dict,
    profile,
    request,
    initial_flux,
    external,
    base_options,
    doubled_options,
) -> dict:
    base_count, base_wall = _compile_and_count(
        profile, request, initial_flux, external, base_options
    )
    doubled_count, doubled_wall = _compile_and_count(
        profile, request, initial_flux, external, doubled_options
    )
    part.update(
        {
            "base_instructions": base_count,
            "doubled_instructions": doubled_count,
            "ratio": doubled_count / base_count if base_count else None,
            "base_compile_wall_s": base_wall,
            "doubled_compile_wall_s": doubled_wall,
        }
    )
    return part


CENSUS_FILES = (
    "nova/equilibrium/fixed_point.py",
    "nova/equilibrium/reduced_newton.py",
    "nova/equilibrium/forward.py",
    "nova/equilibrium/forward_operator.py",
)


def _census_markdown() -> str:
    """List every Python for/while loop and comprehension with its line and bound."""
    sections = []
    for path in CENSUS_FILES:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        rows = []
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.For,
                    ast.While,
                    ast.ListComp,
                    ast.GeneratorExp,
                    ast.DictComp,
                    ast.SetComp,
                ),
            ):
                continue
            enclosing = None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                    if (
                        parent.lineno
                        <= node.lineno
                        <= getattr(parent, "end_lineno", node.lineno)
                    ):
                        if enclosing is None or parent.lineno > enclosing.lineno:
                            enclosing = parent
            if isinstance(node, ast.For | ast.While):
                bound = (
                    ast.get_source_segment(source, node.iter)
                    if isinstance(node, ast.For)
                    else "condition"
                )
                rows.append(
                    f"| {node.lineno} | {type(node).__name__} | "
                    f"{enclosing.name if enclosing else '<module>'} | "
                    f"`{bound}` |"
                )
            else:
                generator = node.generators[0] if node.generators else None
                bound = (
                    ast.get_source_segment(source, generator.iter) if generator else ""
                )
                rows.append(
                    f"| {node.lineno} | {type(node).__name__} | "
                    f"{enclosing.name if enclosing else '<module>'} | "
                    f"`{bound}` |"
                )
        header = (
            f"\n### {path}\n\n"
            "| line | construct | function | bound iterable |\n"
            "|---|---|---|---|\n"
        )
        sections.append(header + "\n".join(rows))
    return (
        "## Python loop census inside traced functions\n\n"
        "Every Python `for`/`while` and comprehension in the four files, with "
        "the bound each iterates.  A `for` over `range(newton_steps)` or "
        "`range(active_set_steps)` is a budget-iteration unroll at trace time; "
        "a `for` over static structure (pairs, kernels, stencils) is fixed. "
        + "\n".join(sections)
    )


def assemble_report(parts: dict) -> str:
    rows = []
    for name in BUDGETS:
        part = parts.get(name)
        if part is None:
            rows.append(f"| {name} | -- | -- | -- | **not run** | -- | -- |")
            continue
        ratio = part.get("ratio")
        if isinstance(ratio, (int, float)):
            ratio_txt = f"{ratio:.3f}"
        elif part.get("error"):
            ratio_txt = "--"
        else:
            ratio_txt = part.get("ratio")
        verdict = part.get("verdict", "")
        if not verdict and part.get("error"):
            verdict = f"not measurable: {part['error']}"
        base = part.get("base_instructions")
        doubled = part.get("doubled_instructions")
        rows.append(
            f"| {name} | {part.get('baseline_value')} | "
            f"{part.get('doubled_value')} | {base} | {doubled} | "
            f"{ratio_txt} | {verdict} |"
        )
    header = (
        "| budget | baseline | doubled | base instrs | doubled instrs |"
        " ratio | verdict |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-node-count", action="store_true")
    parser.add_argument(
        "--budgets",
        default="",
        help="comma-separated budget names to measure (default: all); lets one "
        "job run each budget in a fresh process so LLVM section memory never "
        "accumulates across compiles",
    )
    parser.add_argument("--deadline-s", type=float, default=2800.0)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="assemble the report and figure from already-persisted parts",
    )
    args = parser.parse_args(argv)
    selected = set(args.budgets.split(",")) if args.budgets else None

    def want(name: str) -> bool:
        return selected is None or name in selected

    run_dir = args.out
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True

    if args.finalize_only:
        return _finalize(parts_dir, report_dir)

    probe = jax.jit(lambda x: x + 1.0)
    probe_count = count_instructions(probe.lower(jnp.zeros(3)).compile())
    print(f"PROBE hlo_instruction_count={probe_count}", flush=True)

    source_case, exact, machine = build_machine(CELLS)
    profile, request, initial_flux, external, node_count = build_operator_and_profile(
        source_case, exact, machine, maxsize=MAXSIZE
    )
    base_options = dict(request.policy.kernel_options())
    print(
        f"BASELINE node_count={node_count} route={request.route} "
        f"options={json.dumps({k: base_options[k] for k in sorted(base_options)})}",
        flush=True,
    )

    parts: dict[str, dict] = {}
    part_file = parts_dir / "baseline.json"
    if part_file.exists():
        parts["baseline"] = json.loads(part_file.read_text())
    else:
        count, wall = _compile_and_count(
            profile, request, initial_flux, external, base_options
        )
        parts["baseline"] = {
            "budget": "baseline",
            "route": request.route,
            "backend": BACKEND,
            "node_count": node_count,
            "base_instructions": count,
            "doubled_instructions": count,
            "ratio": 1.0,
            "base_compile_wall_s": wall,
            "doubled_compile_wall_s": wall,
            "base_options": base_options,
            "policy_source": "solve_request.py:232",
            "verdict": "baseline reference",
        }
        part_file.write_text(json.dumps(parts["baseline"], indent=2))

    base_count = parts["baseline"]["base_instructions"]

    def write_part(name: str, part: dict) -> None:
        part.setdefault("backend", BACKEND)
        part.setdefault("baseline_instructions", base_count)
        path = parts_dir / f"{name}.json"
        path.write_text(json.dumps(part, indent=2))
        parts[name] = part
        print(f"PART {name} ratio={part.get('ratio')} -> {path}", flush=True)

    # --- policy-stack budgets: one doubled value at a time, same 300-cell mesh
    for name, doubled, singular, origin in (
        ("newton_steps", 20, 10, "policy"),
        ("gmres_iterations", 60, 30, "policy"),
        ("active_set_steps", 32, 16, "policy"),
    ):
        if _wall() > args.deadline_s:
            print(f"DEADLINE before {name}", flush=True)
            break
        if not want(name):
            continue
        if (parts_dir / f"{name}.json").exists():
            continue
        dbl = {**base_options, name: doubled}
        part = {
            "budget": name,
            "baseline_value": singular,
            "doubled_value": doubled,
            "read_lines": BUDGETS[name]["read_lines"],
            "policy_field": BUDGETS[name]["policy_field"],
        }
        try:
            measure_pair(
                part, profile, request, initial_flux, external, base_options, dbl
            )
            part["verdict"] = "unrolled" if part["ratio"] >= 1.5 else "scanned_or_fixed"
        except Exception as exc:  # keep going when one budget fails
            part["error"] = f"{type(exc).__name__}: {exc}"
        write_part(name, part)

    # warmup: the certificate pins it to zero, so doubling is degenerate.
    if _wall() > args.deadline_s:
        print("DEADLINE before warmup", flush=True)
    elif not want("warmup"):
        pass
    elif not (parts_dir / "warmup.json").exists():
        part = {
            "budget": "warmup",
            "baseline_value": 0,
            "doubled_value": 0,
            "read_lines": BUDGETS["warmup"]["read_lines"],
            "policy_field": BUDGETS["warmup"]["policy_field"],
            "base_instructions": base_count,
            "doubled_instructions": base_count,
            "ratio": 1.0,
            "verdict": "degenerate_zero (pinned by the certificate)",
            "note": "a 0 -> 0 doubling is the same program; warmup is not active",
        }
        write_part("warmup", part)

    # fixed-point module budgets: doubled at trace time via a module patch.
    for name, doubled, origin in (
        ("backtracking_factors", 12, "module"),
        ("model_rebuild_damping_trips", 12, "module"),
    ):
        if _wall() > args.deadline_s:
            print(f"DEADLINE before {name}", flush=True)
            break
        if not want(name):
            continue
        if (parts_dir / f"{name}.json").exists():
            continue
        attr = (
            "_BACKTRACKING_FACTORS"
            if name == "backtracking_factors"
            else "_MODEL_REBUILD_DAMPING_TRIPS"
        )
        value = (
            tuple(1.0 / 2**i for i in range(doubled))
            if name == "backtracking_factors"
            else doubled
        )
        part = {
            "budget": name,
            "baseline_value": BUDGETS[name]["baseline"],
            "doubled_value": doubled,
            "read_lines": BUDGETS[name]["read_lines"],
            "policy_field": BUDGETS[name]["policy_field"],
        }
        try:
            with _patched(attr, value):
                count_doubled, wall_doubled = _compile_and_count(
                    profile, request, initial_flux, external, base_options
                )
            part.update(
                {
                    "base_instructions": base_count,
                    "doubled_instructions": count_doubled,
                    "ratio": count_doubled / base_count if base_count else None,
                    "base_compile_wall_s": parts["baseline"]["base_compile_wall_s"],
                    "doubled_compile_wall_s": wall_doubled,
                }
            )
            part["verdict"] = "unrolled" if part["ratio"] >= 1.5 else "scanned_or_fixed"
        except Exception as exc:
            part["error"] = f"{type(exc).__name__}: {exc}"
        write_part(name, part)

    # --- topology candidate table capacity: same mesh, larger locator
    if _wall() > args.deadline_s:
        print("DEADLINE before topology_table_capacity", flush=True)
    elif (
        want("topology_table_capacity")
        and not (parts_dir / "topology_table_capacity.json").exists()
    ):
        part = {
            "budget": "topology_table_capacity",
            "baseline_value": 30,
            "doubled_value": 60,
            "read_lines": BUDGETS["topology_table_capacity"]["read_lines"],
        }
        try:
            doubled_profile, doubled_request, doubled_flux, doubled_external, _n = (
                build_operator_and_profile(source_case, exact, machine, maxsize=60)
            )
            count_doubled, wall_doubled = _compile_and_count(
                doubled_profile,
                doubled_request,
                doubled_flux,
                doubled_external,
                dict(doubled_request.policy.kernel_options()),
            )
            part.update(
                {
                    "base_instructions": base_count,
                    "doubled_instructions": count_doubled,
                    "ratio": count_doubled / base_count if base_count else None,
                    "base_compile_wall_s": parts["baseline"]["base_compile_wall_s"],
                    "doubled_compile_wall_s": wall_doubled,
                }
            )
            part["verdict"] = "unrolled" if part["ratio"] >= 1.5 else "scanned_or_fixed"
        except Exception as exc:
            part["error"] = f"{type(exc).__name__}: {exc}"
        write_part("topology_table_capacity", part)

    # --- profile node count: a fresh, larger machine doubles the mesh
    if (
        _wall() > args.deadline_s
        and not (parts_dir / "profile_node_count.json").exists()
    ):
        print("DEADLINE before profile_node_count", flush=True)
    elif (
        want("profile_node_count")
        and not args.skip_node_count
        and not (parts_dir / "profile_node_count.json").exists()
    ):
        part = {
            "budget": "profile_node_count",
            "baseline_value": 300,
            "doubled_value": None,
            "read_lines": BUDGETS["profile_node_count"]["read_lines"],
        }
        # The oracle fixture only admits a discrete set of cell counts (its
        # grid has no complete hexagon generator at arbitrary radii).  Discover
        # the doubled mesh: build the candidate machines, keep the ones that
        # build, and measure the one whose node count is closest to 2x the
        # 300-cell baseline, so the ratio answers "does the program carry the
        # profile node count".
        try:
            base_node_count = int(parts["baseline"]["node_count"])
            part["baseline_value"] = f"300 ({base_node_count} nodes)"
            candidates = (400, 500, 1000, 2500)
            built: list[tuple[int, int]] = []  # (requested_cells, node_count)
            build_errors: dict[int, str] = {}
            for cells in candidates:
                if _wall() > args.deadline_s:
                    print(f"DEADLINE building {cells} cells", flush=True)
                    break
                try:
                    _src, _ex, machine = build_machine(cells)
                    built.append((cells, int(len(machine.node))))
                except Exception as exc:
                    build_errors[cells] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"CELLS {cells} unbuildable: {build_errors[cells]}",
                        flush=True,
                    )
            part["build_errors"] = build_errors
            if not built:
                part["error"] = (
                    "the oracle fixture builds no mesh other than the 300-cell "
                    "baseline for this case (no complete hexagon generator at "
                    "any doubled count)"
                )
            else:
                chosen, n_doubled = min(
                    built, key=lambda row: abs(row[1] - 2 * base_node_count)
                )
                part["doubled_value"] = f"{chosen} ({n_doubled} nodes)"
                part["doubled_node_count"] = n_doubled
                part["base_node_count"] = base_node_count
                part["buildable_cells"] = built
                src_c, ex_c, machine_c = build_machine(chosen)
                p_c, r_c, flux_c, ext_c, _n_c = build_operator_and_profile(
                    src_c, ex_c, machine_c, maxsize=MAXSIZE
                )
                count_c, wall_c = _compile_and_count(
                    p_c, r_c, flux_c, ext_c, dict(r_c.policy.kernel_options())
                )
                part.update(
                    {
                        "base_instructions": base_count,
                        "doubled_instructions": count_c,
                        "ratio": count_c / base_count if base_count else None,
                        "doubled_compile_wall_s": wall_c,
                        "base_compile_wall_s": parts["baseline"]["base_compile_wall_s"],
                    }
                )
                part["verdict"] = (
                    "data_size_carried" if part["ratio"] >= 1.5 else "scanned_or_fixed"
                )
        except Exception as exc:
            part["error"] = f"{type(exc).__name__}: {exc}"
        write_part("profile_node_count", part)

    return _finalize(parts_dir, report_dir)


def _finalize(parts_dir: Path, report_dir: Path) -> int:
    parts: dict[str, dict] = {}
    for path in sorted(parts_dir.glob("*.json")):
        parts[path.stem] = json.loads(path.read_text())
    table = assemble_report(parts)
    report = (
        "# Loop-structure audit of the whole-cell certificate solve\n\n"
        "Compiled (not executed) on the CPU backend (`JAX_PLATFORMS=cpu`) for "
        "the weak row at 300 cells.  Each budget row doubles that one budget "
        "against the certificate baseline (newton_steps=10, "
        "gmres_iterations=30, warmup=0, active_set_steps=16, capacity 30, "
        "300 nodes).\n\n" + table + "\n\n" + _census_markdown() + "\n"
    )
    (report_dir / "loop-audit.md").write_text(report)
    print(report, flush=True)
    _write_svg(parts, report_dir / "loop-audit.svg")
    return 0


def _write_svg(parts: dict, path: Path) -> None:
    labels = list(BUDGETS)
    bars = []
    for name in labels:
        part = parts.get(name)
        if not part:
            continue
        base = part.get("base_instructions")
        doubled = part.get("doubled_instructions")
        if base is None or doubled is None:
            continue
        bars.append((name, float(base), float(doubled), part.get("verdict", "")))
    width, height = 760, 460
    left, top, bottom = 150, 40, 410
    plot_width = width - left - 40
    plot_height = bottom - top
    group_w = plot_width / len(bars)
    bar_w = min(26.0, group_w * 0.28)
    max_bar = max(max(b, d) for _, b, d, _ in bars if b and d)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="14" '
        'font-weight="bold">Whole-cell certificate solve: HLO instruction count '
        "at baseline and doubled budget (300-cell weak row, CPU)</text>",
    ]
    for i, (name, base, doubled, verdict) in enumerate(bars):
        cx = left + (i + 0.5) * group_w
        hb = 5 + (base / max_bar) * (plot_height - 20)
        hd = 5 + (doubled / max_bar) * (plot_height - 20)
        lines.append(
            f'<rect x="{cx - bar_w - 2:.1f}" y="{bottom - hb:.1f}" '
            f'width="{bar_w:.1f}" height="{hb:.1f}" fill="#9db4d6"/>'
            f'<rect x="{cx + 2:.1f}" y="{bottom - hd:.1f}" '
            f'width="{bar_w:.1f}" height="{hd:.1f}" fill="#d68b8b"/>'
            f'<text x="{cx:.1f}" y="{bottom + 16}" text-anchor="end" '
            f'transform="rotate(-32 {cx:.1f} {bottom + 16})" '
            f'font-family="sans-serif" font-size="10">{name}</text>'
        )
        lines.append(
            f'<text x="{cx + 2:.1f}" y="{bottom - hd - 4:.1f}" '
            f'font-family="sans-serif" font-size="9" fill="#7a2d2d">'
            f"{int(doubled)}</text>"
        )
    lines.append(
        f'<text x="{left}" y="{bottom + 34}" font-family="sans-serif" '
        'font-size="10">blue = baseline, red = doubled. Ratio near 2 names an '
        "unrolled loop; near 1 a scanned or fixed structure.</text>"
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
