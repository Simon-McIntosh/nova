"""Census of instruction and temporary-byte attribution in the whole-cell solve.

Compiles the certificate whole-cell solve for one row (weak-rotation-reactor-
static) at two requested cell counts (300, 1000), writes the optimised HLO
text of the compiled program to the run directory, and attributes every
instruction and its output-buffer bytes to the innermost nova function named
in the metadata source location. The same census is taken of one application
of the flux map alone, so the solve-over-map overhead is a table (instruction
count and byte ratios) rather than a single figure.

Attribution rule: every HLO instruction carries an optional
``metadata={...}`` with an ``op_name`` path and a ``stack_frame_id``. The
stack frames table resolves that id to a source file, function name and line
(the python frame that created the op). An instruction is attributed to the
innermost nova function in that path - the resolved function name when the
source file lies under nova/, otherwise the trailing segment of the op_name
when that segment names a nova function, otherwise no-scope. Output-buffer
bytes are the byte size of the instruction's printed output shape.

Instruction count is the count of instruction lines across every computation
of the module as printed, which is the fence definition of the program size.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Iterable

import jax
import numpy as np

from benchmarks.solovev_certificate import (
    _case,
    _case_machine,
    _certificate_solve_request,
    _closed_form_current_target,
    _exact_state,
    _production_seed,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery

ROOT = Path(__file__).resolve().parents[1]

CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = (300, 1000)
REQUIRED_CELLS = REQUESTED_CELLS

# where this node lands its evidence (mirrors the dispatch write paths)
_RUN_DIR_ENV = "PROGRAM_SCOPE_CENSUS_RUN_DIR"


def _run_dir() -> Path:
    return Path(os.environ[_RUN_DIR_ENV])


_ELEMENT_BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "s32": 4,
    "u32": 4,
    "s64": 8,
    "u64": 8,
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "c64": 8,
    "c128": 16,
    "f8e4m3": 1,
    "f8e5m2": 1,
    "f8e4m3fn": 1,
    "f8e4m3fnuz": 1,
    "f8e5m2fnuz": 1,
    "f8e4m3b11fnuz": 1,
}


class _ShapeError(ValueError):
    pass


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i] == " ":
        i += 1
    return i


def _shape_bytes(text: str, i: int = 0) -> tuple[int | None, int]:
    """Parse one HLO output shape from ``text[i:]``.

    Returns ``(bytes, index_after_shape)``. ``None`` bytes means the shape
    could not be parsed (recorded as an unparsed shape, counted zero).
    """
    n = len(text)
    i = _skip_space(text, i)
    if i >= n:
        raise _ShapeError()
    if text[i] == "(":
        total = 0
        i += 1
        while True:
            i = _skip_space(text, i)
            if i >= n:
                raise _ShapeError()
            if text[i] == ")":
                return total, i + 1
            part, i = _shape_bytes(text, i)
            if part is None:
                total = 0
            else:
                total += part
            i = _skip_space(text, i)
            if i < n and text[i] == ",":
                i += 1
    for keyword in ("token", "opaque"):
        if text.startswith(keyword, i):
            j = text.find("]", i)
            if j < 0:
                raise _ShapeError()
            return 0, j + 1
    if text.startswith("optional", i):
        j = text.find("[", i) + 1
        return _shape_bytes(text, j)
    j = i
    while j < n and text[j] not in "[(":
        j += 1
    if j >= n or text[j] != "[":
        raise _ShapeError()
    element = text[i:j]
    j += 1
    dims: list[int] = []
    num = ""
    while j < n:
        ch = text[j]
        if ch == "]":
            if num:
                try:
                    dims.append(int(num))
                except ValueError:
                    pass
            j += 1
            break
        if ch == ",":
            if num:
                try:
                    dims.append(int(num))
                except ValueError:
                    pass
                num = ""
            j += 1
        elif ch.isdigit():
            num += ch
            j += 1
        elif ch == "{":
            depth = 1
            j += 1
            while j < n and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
        else:
            j += 1
    else:
        raise _ShapeError()
    if j < n and text[j] == "{":
        depth = 1
        j += 1
        while j < n and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
    count = 1
    for d in dims:
        count *= d
    return count * _ELEMENT_BYTES.get(element, 8), j


_INSTRUCTION_RE = re.compile(r"^\s*(ROOT\s+)?%[\w.]+ = ")


def _split_computations(text: str) -> list[tuple[str, bool, list[str]]]:
    """Split the printed module into computation blocks.

    Returns ``(computation_name, is_entry, instruction_lines)`` for every
    computation, including fused sub-computations and scan bodies. A block is
    opened by the lone line ending in ``{`` (the module header and the
    metadata tables never end in ``{``) and closed at brace depth zero.
    """
    blocks: list[tuple[str, bool, list[str]]] = []
    name = ""
    entry = False
    body: list[str] = []
    depth = 0
    open_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not open_block:
            # a computation header is the only non-instruction line ending
            # in an open brace (metadata tables and the module header end `}`)
            if not _INSTRUCTION_RE.match(stripped) and stripped.endswith("{"):
                entry = stripped.startswith("ENTRY ")
                header = stripped[6:].strip() if entry else stripped
                name = header.split("(", 1)[0].strip().lstrip("%")
                depth = 1
                open_block = True
            continue
        opens = line.count("{")
        closes = line.count("}")
        if depth > 0 or (closes and not opens):
            body.append(line)
        depth += opens - closes
        if depth <= 0:
            blocks.append((name, entry, body))
            name = ""
            entry = False
            body = []
            open_block = False
            depth = 0
    if open_block:
        blocks.append((name, entry, body))
    return blocks


_TABLE_NAMES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def _parse_tables(text: str) -> dict[str, dict[int, dict[str, Any]]]:
    """Parse the metadata tables from the head of the printed module."""
    tables: dict[str, dict[int, dict[str, Any]]] = {}
    lines = text.split("\n")
    section = None
    for line in lines:
        stripped = line.strip()
        if section is not None and re.match(r"^\d+ ", stripped):
            m = re.match(r"^(\d+) (.*)$", stripped)
            if m:
                tables[section][int(m.group(1))] = _parse_table_entry(m.group(2))
                continue
        if stripped in _TABLE_NAMES:
            section = stripped
            tables[section] = {}
            continue
        if section is not None and not re.match(r"^\d+ ", stripped):
            section = None
    return tables


def _parse_table_entry(body: str) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if body.startswith('"'):
        entry["value"] = body.strip().strip('"')
        return entry
    # brace-delimited key=value list
    start = body.find("{")
    if start < 0:
        return entry
    inner = body[start + 1 : body.rfind("}")] if "}" in body else body[start + 1 :]
    for piece in inner.split():
        if "=" not in piece:
            continue
        key, _, value = piece.partition("=")
        entry[key] = int(value) if value.lstrip("-").isdigit() else value
    return entry


def _resolve_frame(tables: dict[str, dict[int, dict[str, Any]]], frame_id: int | None):
    if not frame_id:
        return None
    frames = tables.get("StackFrames", {})
    locations = tables.get("FileLocations", {})
    function_names = tables.get("FunctionNames", {})
    files = tables.get("FileNames", {})
    frame = frames.get(frame_id)
    if not frame:
        return None
    location = locations.get(int(frame.get("file_location_id", 0)))
    if not location:
        return None
    function = function_names.get(int(location.get("function_name_id", 0)))
    source = files.get(int(location.get("file_name_id", 0)))
    return {
        "file": source.get("value") if source else None,
        "function": function.get("value") if function else None,
        "line": int(location.get("line", 0)),
    }


_OPNAME_RE = re.compile(r'op_name="([^"]*)"')
_STACKFRAME_RE = re.compile(r"stack_frame_id=(\d+)")
_SOURCE_FILE_RE = re.compile(r'source_file="([^"]*)"')
_SOURCE_LINE_RE = re.compile(r"source_line=(\d+)")


def _instruction_meta(line: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    body_match = re.search(r"metadata=\{(?P<body>.*?)\}\s*(?:,|$)", line)
    if body_match:
        body = body_match.group("body")
        m = _OPNAME_RE.search(body)
        if m:
            meta["op_name"] = m.group(1)
        m = _STACKFRAME_RE.search(body)
        if m:
            meta["stack_frame_id"] = int(m.group(1))
        m = _SOURCE_FILE_RE.search(body)
        if m:
            meta["source_file"] = m.group(1)
        m = _SOURCE_LINE_RE.search(body)
        if m:
            meta["source_line"] = int(m.group(1))
    return meta


def _parse_instruction(line: str):
    """Parse one instruction line into (name, shape_bytes, opcode, metadata)."""
    match = re.match(r"^\s*(ROOT\s+)?%([\w.]+) = ", line)
    if not match:
        return None
    rest = line[match.end() :]
    byte_size = None
    opcode = None
    try:
        byte_size, after = _shape_bytes(rest, 0)
        after = _skip_space(rest, after)
        op_match = re.match(r"([\w.]+)\(", rest[after:])
        opcode = (
            op_match.group(1) if op_match else rest[after:].split("(", 1)[0].split()[-1]
        )
    except _ShapeError, IndexError, ValueError:
        opcode = rest.split("(", 1)[0].rsplit(" ", 1)[-1]
    return {
        "id": match.group(2),
        "bytes": byte_size,
        "opcode": opcode or "?",
        "meta": _instruction_meta(line),
    }


def _census_module(text: str) -> dict[str, Any]:
    """Run the census over one compiled module's printed HLO text."""
    tables = _parse_tables(text)
    computations = _split_computations(text)
    records: list[dict[str, Any]] = []
    by_computation: dict[str, int] = {}
    while_directives: list[tuple[str, str]] = []
    conditional_directives: list[list[str]] = []
    while_count = 0
    conditional_count = 0
    for name, entry, lines in computations:
        count = 0
        for raw in lines:
            parsed = _parse_instruction(raw)
            if parsed is None:
                continue
            count += 1
            records.append(parsed)
            opcode = parsed["opcode"]
            if opcode == "while":
                while_count += 1
                body_m = re.search(r"body=%([\w.]+)", raw)
                cond_m = re.search(r"condition=%([\w.]+)", raw)
                while_directives.append(
                    (
                        body_m.group(1) if body_m else "",
                        cond_m.group(1) if cond_m else "",
                    )
                )
            elif opcode == "conditional":
                conditional_count += 1
                branch_m = re.search(
                    r"(?:branch_computations|branches)=\{([^}]*)\}", raw
                )
                if branch_m:
                    conditional_directives.append(
                        [
                            n.strip().lstrip("%")
                            for n in branch_m.group(1).split(",")
                            if n.strip()
                        ]
                    )
                else:
                    branch_m2 = re.search(
                        r"(?:branch_computations|branches)=%([\w.,]+)", raw
                    )
                    if branch_m2:
                        conditional_directives.append(
                            [
                                n.lstrip("%")
                                for n in branch_m2.group(1).split(",")
                                if n.strip()
                            ]
                        )
        by_computation[name] = count

    # scan bodies: computations reachable from while/conditional directives
    scan_body = set()
    for body_name, cond_name in while_directives:
        if body_name:
            scan_body.add(body_name)
        if cond_name:
            scan_body.add(cond_name)
    for branches in conditional_directives:
        scan_body.update(branches)
    # fixpoint: any computation called from a scan body is itself scanned
    calls: dict[str, set[str]] = {}
    for name, entry, lines in computations:
        calls[name] = set()
        for raw in lines:
            for m in re.finditer(r"calls=%([\w.]+)", raw):
                calls[name].add(m.group(1))
            for m in re.finditer(r"body=%([\w.]+)", raw):
                calls[name].add(m.group(1))
            for m in re.finditer(r"condition=%([\w.]+)", raw):
                calls[name].add(m.group(1))
            m = re.search(r"(?:branch_computations|branches)=\{([^}]*)\}", raw)
            if m:
                for n in m.group(1).split(","):
                    if n.strip():
                        calls[name].add(n.strip().lstrip("%"))
    changed = True
    while changed:
        changed = False
        for name in [c for c in calls if c in scan_body]:
            for target in calls.get(name, ()):
                if target not in scan_body:
                    scan_body.add(target)
                    changed = True

    total_instructions = len(records)
    total_bytes = 0
    unparsed = 0
    for r in records:
        if r["bytes"] is None:
            unparsed += 1
        else:
            total_bytes += r["bytes"]

    scanned_instructions = sum(by_computation.get(c, 0) for c in scan_body)
    scanned_bytes = 0
    for c in scan_body:
        for name, entry, lines in computations:
            if name == c:
                for raw in lines:
                    p = _parse_instruction(raw)
                    if p and p["bytes"] is not None:
                        scanned_bytes += p["bytes"]

    attributed: dict[str, dict[str, Any]] = {}
    no_metadata = 0
    no_nova_function = 0
    for r in records:
        meta = r["meta"]
        if not meta:
            no_metadata += 1
            _bump(attributed, "__no_scope__", r, {})
            continue
        frame = _resolve_frame(tables, meta.get("stack_frame_id"))
        function = None
        source = {
            "file": frame["file"] if frame else meta.get("source_file"),
            "line": frame["line"] if frame else meta.get("source_line"),
            "function": frame["function"] if frame else None,
        }
        if frame and "nova/" in (frame.get("file") or ""):
            function = frame["function"]
        if function is None:
            op_name = meta.get("op_name")
            if op_name:
                segments = [
                    s for s in op_name.split("/") if s and not s.startswith("jit(")
                ]
                if segments:
                    candidate = segments[-1]
                    if _looks_nova(candidate):
                        function = candidate
        if function is None:
            function = "__no_nova_function__"
            no_nova_function += 1
        _bump(attributed, function, r, source)

    return {
        "total_instructions": total_instructions,
        "total_bytes": total_bytes,
        "unparsed_shape_bytes": unparsed,
        "while_ops": while_count,
        "conditional_ops": conditional_count,
        "scan_instructions": scanned_instructions,
        "straight_line_instructions": total_instructions - scanned_instructions,
        "scan_bytes": scanned_bytes,
        "straight_line_bytes": total_bytes - scanned_bytes,
        "no_metadata_instructions": no_metadata,
        "no_nova_function_instructions": no_nova_function,
        "while_directives": [list(a) for a in while_directives],
        "computations": by_computation,
        "attributed": attributed,
    }


def _looks_nova(candidate: str) -> bool:
    # pure primitive / opcode words must not masquerade as functions in the
    # op-name fallback; anonymous primitives belong in the no-nova bucket
    if candidate in _PRIMITIVE_TOKENS:
        return False
    return True


_PRIMITIVE_TOKENS = {
    "abs",
    "add",
    "after_all",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "and",
    "argmax",
    "argmin",
    "atan2",
    "batch_norm",
    "bessel_i0",
    "bessel_i1",
    "bessel_j0",
    "bessel_j1",
    "bitcast",
    "bitcast_convert",
    "broadcast",
    "broadcast_in_dim",
    "ceil",
    "clamp",
    "collective_permute",
    "compare",
    "complex",
    "concatenate",
    "conditional",
    "const",
    "constant",
    "convert",
    "conv",
    "convolution",
    "cos",
    "cbrt",
    "divide",
    "dot",
    "dot_general",
    "dynamic_reshape",
    "dynamic_slice",
    "dynamic_update_slice",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fft",
    "floor",
    "gather",
    "get_dimension_size",
    "get_tuple_element",
    "imag",
    "infeed",
    "iota",
    "is_finite",
    "log",
    "log1p",
    "logistic",
    "map",
    "maximum",
    "mean",
    "minimum",
    "multiply",
    "neg",
    "not",
    "optimization_barrier",
    "or",
    "outfeed",
    "pad",
    "param",
    "population_count",
    "pow",
    "pred",
    "reduce",
    "reduce_precision",
    "reduce_scatter",
    "reduce_window",
    "rem",
    "reshape",
    "reverse",
    "rng",
    "round_nearest_afz",
    "rsqrt",
    "scatter",
    "select",
    "select_and_scatter",
    "send",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",
    "sign",
    "sin",
    "slice",
    "sort",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "transpose",
    "triangular_solve",
    "tuple",
    "while",
    "xor",
    "xla_call",
}


def _bump(
    attributed: dict[str, dict[str, Any]], key: str, record: dict, source: dict
) -> None:
    entry = attributed.setdefault(
        key,
        {"instructions": 0, "bytes": 0, "file": None, "line": None, "op_sample": None},
    )
    entry["instructions"] += 1
    if record["bytes"] is not None:
        entry["bytes"] += record["bytes"]
    if entry["file"] is None and source.get("file"):
        entry["file"] = source["file"]
    if entry["line"] is None and source.get("line"):
        entry["line"] = source["line"]
    if entry["op_sample"] is None:
        op_name = record["meta"].get("op_name")
        entry["op_sample"] = (
            record["opcode"] if not op_name else f"{record['opcode']}({op_name})"
        )


def _top(
    attributed: dict[str, dict[str, Any]], n: int, metric: str
) -> list[dict[str, Any]]:
    rows = []
    for function, entry in attributed.items():
        if function.startswith("__"):
            continue
        rows.append(
            {
                "function": function,
                "instructions": entry["instructions"],
                "bytes": entry["bytes"],
                "file": entry["file"],
                "line": entry["line"],
                "op_sample": entry["op_sample"],
            }
        )
    rows.sort(key=lambda r: -r[metric])
    return rows[:n]


def _build_profile(case_name: str, requested_cells: int):
    configure_dtypes()
    carrier_case, source_case, exact = _case(case_name)
    # the certificate and the oracle fixture index machines by a negative cell
    # count (dplasma<0 = filament count; positive dplasma is a linear spacing)
    machine = _case_machine(case_name, carrier_case, exact, -requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = _exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, fixture_exterior, _fixture_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty_operator, oracle_state
        )
    )
    operator = oracle_fixture.forward_operator(source_case, machine, fixture_exterior)
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(operator, mesh, newton_steps=recovery.NEWTON_STEPS)
    target_current, current_centroid, current_receipt = _closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    seed, requested_class, _seed_receipt = _production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    request = _certificate_solve_request(
        profile, seed, target_current, carrier_identity=case_name
    )
    return profile, operator, request, seed, target_current


def _compile_programs(profile, operator, request, seed, target_current, cells):
    """Return the two compiled programs (solve and one map application)."""
    external = operator.external(None, None)
    kernel = request.policy.kernel_options()
    solve_program = profile._accelerated_history_program(
        "newton_krylov",
        requested_class=None,
        target_current=target_current,
        **kernel,
    )
    started = perf_counter()
    solve_comp = solve_program.lower(seed, external).compile()
    solve_seconds = perf_counter() - started
    mapped = operator.flux_map(None, None, target_current)
    map_comp = jax.jit(mapped).lower(seed).compile()
    return solve_comp, map_comp, solve_seconds


def _census_compiled(comp, label, run_dir, case_name, cells):
    text = comp.as_text()
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    hlo_path = hlo_dir / f"{case_name}_{cells}c_{label}.hlo.txt"
    hlo_path.write_text(text, encoding="utf-8")
    census = _census_module(text)
    census["label"] = label
    census["cells"] = cells
    census["hlo_text_bytes"] = len(text)
    census["hlo_text_path"] = str(hlo_path)
    length = len(text)
    return census, length


def measure(case_name: str, cells: Iterable[int], run_dir: Path) -> dict[str, Any]:
    """Compile both programs at each cell count, census, and persist parts."""
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[Any, dict[str, Any]] = {}
    for requested_cells in cells:
        part_path = parts_dir / f"{case_name}_{requested_cells}c.json"
        if part_path.exists():
            persisted = json.loads(part_path.read_text(encoding="utf-8"))
            results[requested_cells] = persisted
            print(
                f"CENSUS_PERSISTED case={case_name} requested_cells={requested_cells}",
                flush=True,
            )
            continue
        profile, operator, request, seed, target_current = _build_profile(
            case_name, requested_cells
        )
        solve_comp, map_comp, solve_seconds = _compile_programs(
            profile, operator, request, seed, target_current, requested_cells
        )
        solve_census, solve_len = _census_compiled(
            solve_comp, "solve", run_dir, case_name, requested_cells
        )
        map_census, map_len = _census_compiled(
            map_comp, "map", run_dir, case_name, requested_cells
        )
        entry = {
            "case": case_name,
            "requested_cells": requested_cells,
            "compile_seconds": solve_seconds,
            "hlo_text_bytes_solve": solve_len,
            "hlo_text_bytes_map": map_len,
            "solve": solve_census,
            "map": map_census,
            "solve_over_map": {
                "instruction_ratio": (
                    solve_census["total_instructions"]
                    / map_census["total_instructions"]
                    if map_census["total_instructions"]
                    else None
                ),
                "byte_ratio": (
                    solve_census["total_bytes"] / map_census["total_bytes"]
                    if map_census["total_bytes"]
                    else None
                ),
                "solve_instructions": solve_census["total_instructions"],
                "map_instructions": map_census["total_instructions"],
                "solve_bytes": solve_census["total_bytes"],
                "map_bytes": map_census["total_bytes"],
            },
        }
        for key in ("solve", "map"):
            entry[key]["top30_instructions"] = _top(
                entry[key]["attributed"], 30, "instructions"
            )
            entry[key]["top30_bytes"] = _top(entry[key]["attributed"], 30, "bytes")
        part_path.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
        results[requested_cells] = entry
        print(
            f"CENSUS_PART_LANDED case={case_name} requested_cells={requested_cells} "
            f"solve_instr={solve_census['total_instructions']} "
            f"map_instr={map_census['total_instructions']}",
            flush=True,
        )
    return results


def _render_svg(entry: dict[str, Any], path: Path) -> None:
    """Render a treemap-like horizontal bar chart of instruction share at 1000 cells."""
    rows = _top(entry["solve"]["attributed"], 25, "instructions")
    total = entry["solve"]["total_instructions"]
    height = 40 + 26 * len(rows)
    width = 980
    bar_max = max((r["instructions"] for r in rows), default=1)
    body_x = 300
    plot_w = width - body_x - 20
    parts = []
    parts.append(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="DejaVu Sans, Arial, sans-serif" font-size="12">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        '<text x="14" y="24" font-size="15" font-weight="bold" fill="#1a1a1a">'
        "Solve instruction share by function (1000 cells)</text>"
    )
    parts.append(
        f'<text x="14" y="40" font-size="11" fill="#555">'
        f"{total} instructions total</text>"
    )
    for i, r in enumerate(rows):
        y = 56 + 26 * i
        frac = r["instructions"] / total if total else 0.0
        bar_w = max(2, int(plot_w * r["instructions"] / bar_max))
        hue = 20 + int(200 * frac)
        green = (180 - int(120 * frac)) if frac else 180
        parts.append(
            f'<rect x="{body_x}" y="{y}" width="{bar_w}" height="18" '
            f'fill="rgb({hue},{green},{255 - hue})" rx="2"/>'
        )
        label = r["function"]
        if len(label) > 34:
            label = label[:31] + "..."
        parts.append(
            f'<text x="{body_x - 10}" y="{y + 13}" text-anchor="end" '
            f'fill="#222">{label}</text>'
        )
        pct = frac * 100
        parts.append(
            f'<text x="{body_x + bar_w + 6}" y="{y + 13}" fill="#333">'
            f"{r['instructions']:,}  ({pct:.1f}%)</text>"
        )
        loc = ""
        if r.get("file"):
            loc = f"{Path(r['file']).name}:{r.get('line') or '?'}"
        parts.append(
            f'<text x="{width - 8}" y="{y + 13}" text-anchor="end" '
            f'fill="#777" font-size="10">{loc}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _fmt_bytes(value: int) -> str:
    if value >= 1 << 30:
        return f"{value / (1 << 30):.2f} GiB"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.2f} MiB"
    if value >= 1 << 10:
        return f"{value / (1 << 10):.2f} KiB"
    return f"{value} B"


def build_report(results: dict[int, dict[str, Any]]) -> str:
    lines: list[str] = []
    ap = lines.append
    ap("# Program scope census")
    ap("")
    ap(
        "Census of the compiled whole-cell certificate solve (weak-rotation-"
        "reactor-static row) at 300 and 1000 requested cells, and of one "
        "application of the flux map alone at the analytic flux."
    )
    ap("")
    ap("## Solve over one map application")
    ap("")
    ap(
        "| cells | solve instr | map instr | instr ratio | solve bytes "
        "| map bytes | byte ratio |"
    )
    ap("|---:|---:|---:|---:|---:|---:|---:|")
    for cells in REQUIRED_CELLS:
        entry = results[cells]
        s = entry["solve_over_map"]
        ap(
            f"| {cells} | {s['solve_instructions']:,} | {s['map_instructions']:,} "
            f"| {s['instruction_ratio']:.1f}x | {_fmt_bytes(s['solve_bytes'])} "
            f"| {_fmt_bytes(s['map_bytes'])} | {s['byte_ratio']:.1f}x |"
        )
    ap("")
    ap("## Structural summary (solve program)")
    ap("")
    for cells in REQUIRED_CELLS:
        entry = results[cells]
        census = entry["solve"]
        ap(f"### {cells} cells")
        total = census["total_instructions"]
        scanned = census["scan_instructions"]
        ap("")
        ap(
            f"- instructions: {total:,}  "
            f"(unparsed-shape: {census['unparsed_shape_bytes']})"
        )
        ap(
            f"- while ops: {census['while_ops']}  |  conditional ops: "
            f"{census['conditional_ops']}"
        )
        ap(
            f"- scanned bodies: {scanned:,} instructions "
            f"({scanned / total * 100:.1f}%)  |  straight-line: "
            f"{total - scanned:,} ({(total - scanned) / total * 100:.1f}%)"
        )
        no_nova = (
            census["no_metadata_instructions"] + census["no_nova_function_instructions"]
        )
        ap(f"- instructions with no nova scope: {no_nova:,}")
        ap(
            f"  (no metadata {census['no_metadata_instructions']:,}; "
            f"metadata but not nova {census['no_nova_function_instructions']:,})"
        )
        ap(
            f"- scan bytes: {_fmt_bytes(census['scan_bytes'])}  |  "
            f"straight-line bytes: {_fmt_bytes(census['straight_line_bytes'])}"
        )
        ap("")
        if census["while_directives"]:
            ap("while body/condition computation sizes:")
            ap("")
            ap("| body computation | instructions |")
            ap("|---:|---:|")
            sizes = {}
            for body_name, cond_name in census["while_directives"]:
                for n in (body_name, cond_name):
                    if n:
                        sizes[n] = census["computations"].get(n, 0)
            for n, size in sizes.items():
                ap(f"| {n} | {size:,} |")
            ap("")
    for kind in ("instructions", "bytes"):
        ap(f"## Top 30 functions by {kind}")
        ap("")
        ap("| cells | # | function | count | share | file:line |")
        ap("|---:|---:|---|---:|---:|---|")
        for cells in REQUIRED_CELLS:
            rows = results[cells]["solve"][f"top30_{kind}"]
            if kind == "instructions":
                total = results[cells]["solve"]["total_instructions"]
                for i, r in enumerate(rows, start=1):
                    loc = "-"
                    if r.get("file"):
                        loc = f"{Path(r['file']).name}:{r.get('line') or '?'}"
                    ap(
                        f"| {cells} | {i} | {r['function']} | {r['instructions']:,} "
                        f"| {r['instructions'] / total * 100:.1f}% | {loc} |"
                    )
            else:
                total = results[cells]["solve"]["total_bytes"]
                for i, r in enumerate(rows, start=1):
                    loc = "-"
                    if r.get("file"):
                        loc = f"{Path(r['file']).name}:{r.get('line') or '?'}"
                    ap(
                        f"| {cells} | {i} | {r['function']} | {_fmt_bytes(r['bytes'])} "
                        f"| {r['bytes'] / total * 100:.1f}% | {loc} |"
                    )
        ap("")
    for kind in ("instructions", "bytes"):
        ap(f"## Flux-map top 10 by {kind}")
        ap("")
        ap("| cells | # | function | count | share | file:line |")
        ap("|---:|---:|---|---:|---:|---|")
        for cells in REQUIRED_CELLS:
            rows = _top(results[cells]["map"]["attributed"], 10, kind)
            total = results[cells]["map"][
                "total_" + ("bytes" if kind == "bytes" else "instructions")
            ]
            for i, r in enumerate(rows, start=1):
                loc = "-"
                if r.get("file"):
                    loc = f"{Path(r['file']).name}:{r.get('line') or '?'}"
                if kind == "bytes":
                    value = _fmt_bytes(r["bytes"])
                else:
                    value = f"{r['instructions']:,}"
                share = (
                    f"{r[kind] / total * 100:.1f}%"
                    if "bytes" in r and r[kind] is not None
                    else f"{r['instructions'] / total * 100:.1f}%"
                )
                ap(f"| {cells} | {i} | {r['function']} | {value} | {share} | {loc} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Census of the compiled whole-cell solve program."
    )
    parser.add_argument("--case", default=CASE_NAME)
    parser.add_argument("--cells", type=int, nargs="+", default=list(REQUESTED_CELLS))
    parser.add_argument(
        "--run-dir",
        default=os.environ.get(_RUN_DIR_ENV, str(Path.cwd())),
    )
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    arguments = parser.parse_args()
    run_dir = Path(arguments.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    results = measure(arguments.case, arguments.cells, run_dir)
    entry = results.get(1000, results.get(list(results)[-1]))
    if arguments.figure_dir:
        figure_dir = Path(arguments.figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        _render_svg(entry, figure_dir / "scope-census-instruction-share.svg")
        print(
            f"CENSUS_FIGURE {figure_dir / 'scope-census-instruction-share.svg'}",
            flush=True,
        )
    report = build_report(results)
    report_dir = Path(arguments.report_dir) if arguments.report_dir else run_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "scope-census-index.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"CENSUS_REPORT {report_path}", flush=True)
    summary = {
        "case": arguments.case,
        "cells": {str(c): _summarize(results[c]) for c in results},
        "report": str(report_path),
    }
    print(json.dumps(summary, sort_keys=True, default=str), flush=True)


def _summarize(entry: dict[str, Any]) -> dict[str, Any]:
    s = entry["solve"]
    m = entry["map"]
    ratio = entry["solve_over_map"]
    return {
        "solve_instructions": s["total_instructions"],
        "map_instructions": m["total_instructions"],
        "instruction_ratio": ratio["instruction_ratio"],
        "solve_bytes": s["total_bytes"],
        "map_bytes": m["total_bytes"],
        "byte_ratio": ratio["byte_ratio"],
        "while_ops": s["while_ops"],
        "conditional_ops": s["conditional_ops"],
        "scan_share": s["scan_instructions"] / s["total_instructions"],
        "no_nova_scope": s["no_metadata_instructions"]
        + s["no_nova_function_instructions"],
        "top10_instructions": [
            {
                "function": r["function"],
                "instructions": r["instructions"],
                "file": r["file"],
                "line": r["line"],
            }
            for r in _top(s["attributed"], 10, "instructions")
        ],
    }


if __name__ == "__main__":
    main()
