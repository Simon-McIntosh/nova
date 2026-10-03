"""Source scanner for poloidal renderers that contour a scattered-node field
without the wall mask.

A field sampled at scattered nodes and interpolated onto a raster is finite
across the whole convex hull of those nodes, so a contour drawn from it crosses
a concave first wall unless the raster or the node triangulation is first
masked to the wall units. Two painter routes enforce that mask:
:func:`nova.media.poloidal.draw_scattered_contours` requires a ``wall`` and
masks the triangles that leave the vessel, and
:func:`nova.media.poloidal.draw_flux_contours` accepts an optional ``wall`` and
blanks every grid point outside it. A call that reaches a contouring routine on
scattered data while passing neither is drawing inventable field, because the
hull the interpolation fills is not the vessel.

This module finds those call sites. It is a read-only source scan: it parses
each file, tracks which names are bound to a scattered-field producer
(:data:`SCATTERED_PRODUCERS`) or to a raster already blanked to the wall
(:data:`BLANKED_PRODUCERS`), and reports every contouring call fed by the
former and not guarded by the latter.

A triangulated ``Axes.tricontour`` / ``tricontourf`` is scattered by
triangulation and is reported wherever it draws a figure; ``contour`` /
``contourf`` are reported only when an argument traces to a scattered producer,
because a regular-grid field needs no wall mask. A documented set of
non-renderer calls (:data:`NON_RENDERER_CALLS`) is exempt: those extract
contour polylines for measurement, reject every curve with a vertex outside the
wall, and close the figure without drawing it.

Records carry the repository-relative path so two same-named files in
different directories stay distinct. The CLI takes any list of files or
directories and exits nonzero when any single one holds an unguarded call, so
a repair batch can gate its own files without scanning the whole corpus.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Producers that build a field on scattered nodes and return a raster (or a
# coordinate set) finite across the node convex hull, not the vessel.
SCATTERED_PRODUCERS = frozenset(
    {
        "_raster_field",
        "_state_raster",
        "_raster",
        "_structured_flux",
        "LinearNDInterpolator",
        "griddata",
    }
)

# Producers that blank a scattered raster to the wall units before it is drawn.
BLANKED_PRODUCERS = frozenset({"_masked_field", "_blanked_field"})

PAINTER = "draw_flux_contours"
SCATTERED_PAINTER = "draw_scattered_contours"
TRIANGLE_CONTOURS = frozenset({"tricontour", "tricontourf"})
GRID_CONTOURS = frozenset({"contour", "contourf"})

# Calls that are not renderers of scattered data. Every entry is named by its
# repository-relative path and line, with the reason it is exempt; nothing is
# suppressed without a reason recorded here.
#
# ``benchmarks/plasma_cell_trip_panels.py:406`` triangulates the nodes only to
# extract candidate polylines: every curve with a vertex outside
# ``inside_wall_units`` is rejected at lines 412-417, the surviving curves are
# measured, and the temporary figure is closed at line 434 without being drawn.
# It is a probe, not a renderer, so no field reaches a reader from it.
#
# The mechanism-evidence renderer's analytic panel at 360 draws
# ``analytic_radius``/``analytic_height``/``analytic_full``, the exact analytic
# flux evaluated on a regular mesh; the panel needs no wall mask because the
# field is defined by a closed form rather than interpolated. It is suppressed
# because its value argument does not trace to a scattered producer through the
# alias in use, and removing the entry adds a record at 360 (verified by
# deleting it in a scratch copy: the tree reports 81 instead of 80). Its
# sibling at 511, drawing the interpolated solved field, is reported.
#
# The same file's analytic draw at 510 is on the record's own coordinates; the
# coordinates merely share the grid the solved field was interpolated onto, so
# the analytic panel is a regular-grid draw that needs no wall mask. Its
# sibling at 511 draws the interpolated solved field and is reported.
NON_RENDERER_CALLS = frozenset(
    {
        ("benchmarks/plasma_cell_trip_panels.py", 406),
        (
            "docs/figures/null-identification-authority/mechanism-evidence/render_mechanism_evidence.py",
            360,
        ),
        (
            "docs/figures/null-identification-authority/mechanism-evidence/render_mechanism_evidence.py",
            510,
        ),
    }
)

# ``draw_flux_contours`` declares ``wall`` as the ninth positional parameter
# (index 8, before ``**kwargs``), so a call with nine or more positional
# arguments hands the wall positionally and is guarded.
WALL_POSITION_INDEX = 8

DEFAULT_ROOTS = ("benchmarks", "docs/figures")

# Mask helpers whose call inside an expression proves the raster was blanked.
_MASK_FUNCTIONS = frozenset({"inside_wall_units", "_inside_wall_units"})


@dataclass(frozen=True)
class UnmaskedCall:
    """One unguarded scattered-field renderer call.

    ``path`` is repository-relative, so two same-named files in different
    directories are distinct records.
    """

    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}"


def _relative_path(path: Path) -> str:
    """Return ``path`` relative to the repository root, else as given."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _callee_name(func: ast.expr) -> str | None:
    """Return the trailing name of a call target (``a.b.f`` -> ``f``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _walk_expr(expr: ast.AST) -> Iterable[ast.AST]:
    """Yield every node in an expression subtree."""
    yield expr
    for child in ast.iter_child_nodes(expr):
        yield from _walk_expr(child)


def _flatten_targets(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_flatten_targets(element))
        return names
    return []


class _Scope:
    """A module or function scope and the names it binds."""

    def __init__(self, parent: "_Scope | None") -> None:
        self.parent = parent
        self.bindings: dict[str, list[ast.expr]] = {}
        self._scattered: dict[str, bool] = {}
        self._blanked: dict[str, bool] = {}
        self._resolving: set[str] = set()

    def bind(self, name: str, value: ast.expr) -> None:
        self.bindings.setdefault(name, []).append(value)

    def hierarchy(self) -> Iterable["_Scope"]:
        scope: _Scope | None = self
        while scope is not None:
            yield scope
            scope = scope.parent


class _Scanner:
    """Bind names per scope for one module, then report unguarded calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.relpath = _relative_path(path)
        self.aliases: dict[str, str] = {}
        self.module = _Scope(None)
        self.scopes: dict[ast.AST, _Scope] = {}
        self.node_scope: dict[int, _Scope] = {}
        self.func_scopes: dict[str, _Scope] = {}
        self._receipt_cache: dict[str, bool] = {}
        self._receipt_resolving: set[str] = set()
        self.key_flags: dict[str, str] = {}
        self.tree: ast.AST = ast.Module(body=[], type_ignores=[])

    # -- construction ---------------------------------------------------
    def _collect(self) -> None:
        source = self.path.read_text()
        tree = ast.parse(source, filename=str(self.path))
        self.tree = tree
        self._collect_aliases(tree)
        self._populate(tree, self.module)
        self._collect_keys(tree)

    def _collect_keys(self, tree: ast.AST) -> None:
        """Map dict key strings to the provenance of the value bound to them.

        A renderer often reads its raster out of a receipt dict built in
        another function (``solved = record["solved_grid"]``), so a subscript
        can only be classified once the key's producing expression is known.
        The registry records, for every string key found in a dict literal,
        whether its value expression reaches a scattered or blanked producer.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            scope = self.node_scope.get(id(node), self.module)
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    continue
                if self._expr_flag(scope, value, "scattered"):
                    self.key_flags.setdefault(key.value, "scattered")
                elif self._expr_flag(scope, value, "blanked"):
                    self.key_flags.setdefault(key.value, "blanked")

    def _collect_aliases(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = alias.name.split(".")[-1]

    def _populate(self, node: ast.AST, scope: _Scope) -> None:
        self.scopes[node] = scope
        self.node_scope[id(node)] = scope
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            self._record_assignment(node, scope)
        elif isinstance(node, ast.For):
            for name in _flatten_targets(node.target):
                scope.bind(name, node.iter)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                inner = _Scope(scope)
                self.scopes[child] = inner
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope.bind(child.name, ast.Constant(value=None))
                    self.func_scopes[child.name] = inner
                body = child.body if isinstance(child.body, list) else [child.body]
                for statement in body:
                    self._populate(statement, inner)
                if isinstance(child, ast.Lambda):
                    self._populate(child.body, inner)
                continue
            self._populate(child, scope)

    def _record_assignment(self, node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        else:  # AugAssign
            value = node.value
            targets = [node.target]
        if value is None:
            return
        for target in targets:
            for name in _flatten_targets(target):
                scope.bind(name, value)

    # -- classification -------------------------------------------------
    def _subscript_key(self, node: ast.Subscript) -> str | None:
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return index.value
        return None

    def _canonical(self, name: str) -> str:
        return self.aliases.get(name, name)

    def _producer_kind(self, expr: ast.expr) -> str | None:
        """Return ``scattered``/``blanked`` when expr reaches a producer call."""
        for node in _walk_expr(expr):
            if isinstance(node, ast.Call):
                name = _callee_name(node.func)
                if name is None:
                    continue
                name = self._canonical(name)
                if name in BLANKED_PRODUCERS:
                    return "blanked"
                if name in SCATTERED_PRODUCERS:
                    return "scattered"
        return None

    def _name_flag(self, scope: _Scope, name: str, kind: str) -> bool:
        name = self._canonical(name)
        for owner in scope.hierarchy():
            if name not in owner.bindings:
                continue
            if name in owner._resolving:
                return False
            owner._resolving.add(name)
            try:
                cache = owner._scattered if kind == "scattered" else owner._blanked
                if name not in cache:
                    cache[name] = any(
                        self._expr_flag(owner, value, kind)
                        for value in owner.bindings[name]
                    )
                return cache[name]
            finally:
                owner._resolving.discard(name)
        return False

    def _is_receipt(self, name: str) -> bool:
        """True when a module function builds or returns scattered data.

        A renderer often reads its raster from a receipt dict built in another
        function of the same module (``record = _build(...)`` then
        ``solved = record["solved_grid"]``), so a name bound to such a call is
        a scattered receipt and a subscript of it is scattered too.
        """
        scope = self.func_scopes.get(name)
        if scope is None:
            return False
        if name in self._receipt_cache:
            return self._receipt_cache[name]
        if name in self._receipt_resolving:
            return False
        self._receipt_resolving.add(name)
        try:
            result = any(
                self._expr_flag(scope, value, "scattered")
                for values in scope.bindings.values()
                for value in values
            )
            self._receipt_cache[name] = result
            return result
        finally:
            self._receipt_resolving.discard(name)

    def _expr_flag(self, scope: _Scope, expr: ast.expr, kind: str) -> bool:
        if self._producer_kind(expr) == kind:
            return True
        for node in _walk_expr(expr):
            if isinstance(node, ast.Name) and self._name_flag(scope, node.id, kind):
                return True
            if kind == "scattered" and isinstance(node, ast.Call):
                callee = _callee_name(node.func)
                if callee is not None and self._is_receipt(self._canonical(callee)):
                    return True
            if isinstance(node, ast.Subscript):
                if self.key_flags.get(self._subscript_key(node)) == kind:
                    return True
                if kind == "scattered" and self._expr_flag(scope, node.value, "scattered"):
                    return True
            if kind == "blanked" and isinstance(node, ast.Call):
                if _callee_name(node.func) == "where":
                    for arg in node.args:
                        for inner in _walk_expr(arg):
                            if isinstance(inner, ast.Name) and (
                                inner.id in _MASK_FUNCTIONS
                                or self._name_flag(scope, inner.id, "blanked")
                            ):
                                return True
        return False

    # -- scanning -------------------------------------------------------
    def scan(self) -> list[UnmaskedCall]:
        self._collect()
        found: list[UnmaskedCall] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                call = self._classify(node)
                if call is not None:
                    found.append(call)
        found.sort(key=lambda record: record.line)
        return found

    def _classify(self, node: ast.Call) -> UnmaskedCall | None:
        scope = self.node_scope.get(id(node), self.module)
        name = _callee_name(node.func)
        if name is None:
            return None
        resolved = self._canonical(name)

        if (self.relpath, node.lineno) in NON_RENDERER_CALLS:
            return None

        if resolved == SCATTERED_PAINTER:
            return None  # draw_scattered_contours requires a wall

        if resolved == PAINTER:
            if any(keyword.arg == "wall" for keyword in node.keywords):
                return None
            if len(node.args) > WALL_POSITION_INDEX:
                return None
            if any(self._expr_flag(scope, arg, "blanked") for arg in node.args):
                return None
            if any(self._expr_flag(scope, arg, "scattered") for arg in node.args):
                return UnmaskedCall(self.relpath, node.lineno, resolved)
            return None

        if name in TRIANGLE_CONTOURS:
            if any(self._expr_flag(scope, arg, "blanked") for arg in node.args):
                return None
            return UnmaskedCall(self.relpath, node.lineno, name)

        if name in GRID_CONTOURS:
            value = node.args[2] if len(node.args) > 2 else None
            if value is not None:
                if self._expr_flag(scope, value, "blanked"):
                    return None
                if self._expr_flag(scope, value, "scattered"):
                    return UnmaskedCall(self.relpath, node.lineno, name)
            return None

        return None

    def _scope_of(self, node: ast.AST) -> _Scope:
        return self.node_scope.get(id(node), self.module)


def _python_files(target: Path) -> Iterable[Path]:
    if target.is_dir():
        yield from sorted(target.rglob("*.py"))
    elif target.suffix == ".py":
        yield target


def scan_paths(paths: Sequence[str | Path]) -> list[UnmaskedCall]:
    """Return one record per unguarded scattered-field call under ``paths``."""
    records: list[UnmaskedCall] = []
    for raw in paths:
        target = Path(raw)
        for source in _python_files(target):
            try:
                records.extend(_Scanner(source).scan())
            except SyntaxError:
                continue
    records.sort(key=lambda record: (record.path, record.line))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_ROOTS))
    args = parser.parse_args(argv)
    records = scan_paths(args.paths)
    for record in records:
        print(record)
    return 1 if records else 0


if __name__ == "__main__":
    raise SystemExit(main())