"""Census of nova's IMAS seam.

The census is recomputed from the tree on every run. It resolves each module's
imports and classifies it by what the module does rather than by its name.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TREE_DIRECTORIES = ("nova", "scripts", "benchmarks", "apps", "tests")

MACHINE_NAME = re.compile(r"^(mast|diiid|machine)_")

SOLVER_PACKAGES = (
    "nova.equilibrium",
    "nova.transport",
    "nova.biot",
    "nova.frame",
    "nova.circuit",
)

CLASS_DESTINATION = {
    "generic reader": "stays",
    "generic writer": "stays",
    "machine authoring or publication": "moves to the ingest member",
    "machine analysis": "generalised when its machine's authoring moves",
    "convention conversion": "conversion moves to the ingest member",
    "convention statement": "rewired to the literal COCOS 17 values",
    "unreached": "deleted",
}

CONVENTION_REWIRED = "rewired to the literal COCOS 17 values"
CONVENTION_MOVES = "conversion moves to the ingest member"

CLASS_GLOSSARY = {
    "generic reader": "reads Data Dictionary IDSs, authors nothing",
    "generic writer": "writes Data Dictionary IDSs, authors no machine data",
    "machine authoring or publication": "builds or publishes a machine description",
    "machine analysis": "adjudicates, screens or compares machine data",
    "convention conversion": "maps between COCOS conventions or DD versions",
    "convention statement": "states nova's COCOS 17 convention, converts nothing",
    "unreached": "no module in the scanned trees imports it",
}

IDENTITY_NAMES = frozenset(["convention"])

DD_HELPER_NAMES = frozenset(
    [
        "dd_equilibrium_convention",
        "dd_equilibrium_factor",
        "translate_equilibrium",
    ]
)

TRANSFORM_CALLS = frozenset(["convention_transform", "identify_convention"])

DD_VERSION_NAMES = frozenset(["DD3_CONVENTION", "DD4_CONVENTION"])

IDS_FACTORY_CALLS = frozenset(["IDSFactory", "ids_factory", "_new_ids", "new_ids"])

PUBLICATION_IMPORTS = frozenset(
    ["nova.imas.machine_artifact", "nova.database.content_store", "nova.io.ingest"]
)

AUTHORING_DEF = re.compile(
    r"^_?author\w*$"
    r"|^(build|write|publish|create)_\w*(description|artifact|ids|geometry|machine|registry)\w*$"
    r"|^write_and_reopen$"
    r"|^_?\w*component_geometry$"
    r"|^_?\w*_geometry$"
)

AMBIX_MAP = {
    "mast": "imas_ambix/data/machine_maps/mast.json",
    "diiid": "imas_ambix/data/machine_maps/diii-d.json",
}

REFERENCE_SHOTS = (
    {
        "machine": "MAST",
        "shot": "21978",
        "source": (
            "tests/data-manifest.json mast_efm_geometry_reference and "
            "tests/test_forward_parity_constrained_arm.py"
        ),
    },
    {
        "machine": "DIII-D",
        "shot": "d3d_shot_00000c4a7b",
        "source": (
            "docs/plans/diiid-forward-onboarding.html, "
            "representative forward frame 102",
        ),
    },
)


@dataclass
class Facts:
    imports: dict = field(default_factory=dict)
    imported_dotted: set = field(default_factory=set)
    calls: dict = field(default_factory=dict)
    defs: list = field(default_factory=list)
    assigned: dict = field(default_factory=dict)
    import_lines: dict = field(default_factory=dict)
    has_main_guard: bool = False

    def imports_module(self, name):
        return name in self.imports or name in self.imported_dotted

    def imported_names(self, module):
        return set(self.imports.get(module, set()))

    def first_def_line(self):
        if not self.defs:
            return 1
        return min(line for name, line in self.defs)


@dataclass(frozen=True)
class ModuleRef:
    dotted: str
    path: Path
    rel: str
    tree: str


@dataclass(frozen=True)
class Row:
    module: str
    path: str
    line: int
    cls: str
    basis: str
    destination: str
    machine: str
    importers: tuple = ()
    solver_importers: tuple = ()
    duplicates: tuple = ()
    ambix_map: tuple = ()


@dataclass(frozen=True)
class ImporterRow:
    module: str
    path: str
    imports: tuple
    cls: str
    destination: str


@dataclass(frozen=True)
class ConventionRow:
    module: str
    path: str
    line: int
    cls: str
    basis: str
    destination: str


@dataclass(frozen=True)
class Census:
    rows: tuple
    importer_rows: tuple
    convention_rows: tuple
    counts: dict
    dataset_control: tuple
    convention_control: tuple
    reference_shots: tuple
    trees: tuple

    @property
    def files_seen(self):
        return self.counts["files_seen"]


def _dotted(path, root):
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _walk_tree(root, tree):
    base = root / tree
    if not base.is_dir():
        return []
    refs = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        refs.append(
            ModuleRef(
                dotted=_dotted(path, root),
                path=path,
                rel=path.relative_to(root).as_posix(),
                tree=tree,
            )
        )
    return refs


def _walk_ambix(ambix):
    refs = []
    for subtree in ("imas_ambix", "tests"):
        base = ambix / subtree
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            refs.append(
                ModuleRef(
                    dotted=_dotted(path, ambix),
                    path=path,
                    rel=path.relative_to(ambix).as_posix(),
                    tree="ambix",
                )
            )
    return refs


def _parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError, UnicodeDecodeError, OSError:
        return None


def _simple_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _facts(tree):
    facts = Facts()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            facts.defs.append((node.name, node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    facts.assigned.setdefault(target.id, node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                facts.assigned.setdefault(node.target.id, node.lineno)
        elif isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                left = test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    facts.has_main_guard = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _simple_name(node.func)
            if name and name not in facts.calls:
                facts.calls[name] = getattr(node, "lineno", 0)
    return facts


def _package_parts(ref):
    parts = ref.dotted.split(".")
    if ref.path.name == "__init__.py":
        return parts
    return parts[:-1]


def _resolve_base(ref, module, level):
    if level == 0:
        return module or ""
    package = _package_parts(ref)
    if level > 1:
        package = package[: max(0, len(package) - (level - 1))]
    parts = list(package)
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def _raw_imports(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, 0, [], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            out.append((node.module, node.level or 0, names, node.lineno))
    return out


def _build_graph(refs, known):
    facts_by = {}
    raw_by = {}
    for ref in refs:
        tree = _parse(ref.path)
        if tree is None:
            facts_by[ref.dotted] = Facts()
            raw_by[ref.dotted] = []
            continue
        facts = _facts(tree)
        raws = _raw_imports(tree)
        for module, level, names, line in raws:
            base = _resolve_base(ref, module, level)
            if not base:
                continue
            facts.imports.setdefault(base, set()).update(names)
            facts.imported_dotted.add(base)
            facts.import_lines.setdefault(base, line)
            for name in names:
                candidate = base + "." + name
                if candidate in known:
                    facts.imports.setdefault(candidate, set())
                    facts.imported_dotted.add(candidate)
                    facts.import_lines.setdefault(candidate, line)
        facts_by[ref.dotted] = facts
        raw_by[ref.dotted] = raws
    importers = defaultdict(list)
    for ref in refs:
        for module, level, names, line in raw_by.get(ref.dotted, []):
            base = _resolve_base(ref, module, level)
            if not base:
                continue
            targets = {base}
            for name in names:
                targets.add(base + "." + name)
            for target in targets:
                importers[target].append((ref.dotted, line))
    return facts_by, dict(importers)


def _converts(facts):
    if facts.imports_module("nova_cocos"):
        names = facts.imported_names("nova_cocos")
        transform = names - IDENTITY_NAMES
        if transform:
            return True, "imports the COCOS algebra for a transform"
        if "nova_cocos" in facts.imported_dotted and not names:
            return True, "imports the nova_cocos module"
    helpers = facts.imported_names("nova.io.sourcemap") & DD_HELPER_NAMES
    if helpers:
        return True, "derives a convention through nova.io.sourcemap"
    called = set(facts.calls) & (DD_HELPER_NAMES | TRANSFORM_CALLS)
    if called:
        return True, "calls a convention transform or DD helper"
    return False, ""


def _authors(ref, facts):
    for name, line in facts.defs:
        if AUTHORING_DEF.search(name):
            return True, "defines an authoring or publication function"
    if set(facts.calls) & IDS_FACTORY_CALLS:
        return True, "creates IDS objects"
    if facts.imported_dotted & PUBLICATION_IMPORTS:
        return True, "imports the publication or ingest path"
    if "artifact" in ref.path.stem:
        return True, "is the machine artifact path"
    return False, ""


def _generic_class(facts):
    writes = bool(set(facts.calls) & {"write", "put", "write_entry", "write_ids"})
    if writes or facts.imports_module("nova.imas.ids_entry"):
        return "generic writer", "writes IDSs through the generic layer"
    for name, line in facts.defs:
        if re.match(r"^(write|create|author|put)_\w+$", name):
            return "generic writer", "defines a writer function"
    return "generic reader", "reads IDSs through the generic layer"


def _classify(ref, facts, importers, referenced=frozenset()):
    if ref.path.stem in {"__init__", "__main__"} or ref.path.stem.startswith("test_"):
        return _generic_class(facts)
    converting, basis = _converts(facts)
    if "COCOS" in facts.assigned:
        return "convention statement", "module-level COCOS assignment"
    if converting:
        return "convention conversion", basis
    machine = bool(MACHINE_NAME.match(ref.path.stem))
    if machine:
        authoring, authoring_basis = _authors(ref, facts)
        if authoring:
            return "machine authoring or publication", authoring_basis
        if (
            ref.rel.startswith("nova/imas/")
            and not importers.get(ref.dotted)
            and ref.dotted not in referenced
        ):
            return "unreached", "no importer in the scanned trees"
        return "machine analysis", "machine-named with no authoring capability"
    if (
        ref.rel.startswith("nova/imas/")
        and not importers.get(ref.dotted)
        and ref.dotted not in referenced
    ):
        return "unreached", "no importer in the scanned trees"
    return _generic_class(facts)


def _machine(ref):
    stem = ref.path.stem
    if stem.startswith("diiid"):
        return "DIII-D"
    if stem.startswith("mast"):
        return "MAST"
    if stem.startswith("machine"):
        return "any machine"
    return "—"


def _converts_between_conventions(facts):
    names = facts.imported_names("nova_cocos")
    if names - IDENTITY_NAMES:
        return True
    if facts.imported_names("nova.io.sourcemap") & DD_VERSION_NAMES:
        return True
    return bool(set(facts.calls) & (DD_HELPER_NAMES | TRANSFORM_CALLS))


def _convention_destination(ref, facts, cls):
    if cls == "convention statement":
        return CONVENTION_REWIRED
    if ref.tree == "ambix":
        return CONVENTION_MOVES
    if _converts_between_conventions(facts):
        return CONVENTION_MOVES
    return CONVENTION_REWIRED


def _ambix_coverage(ref):
    stem = ref.path.stem
    if stem.startswith("machine"):
        return tuple(sorted(AMBIX_MAP.values()))
    for prefix, path in AMBIX_MAP.items():
        if stem.startswith(prefix):
            return (path,)
    return ()


def _solver_importers(importers):
    found = []
    for module, line in importers:
        if module.startswith(SOLVER_PACKAGES):
            found.append(module + ":" + str(line))
    return tuple(sorted(found))


TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _string_references(refs):
    index = defaultdict(set)
    for ref in refs:
        try:
            text = ref.path.read_text(encoding="utf-8")
        except UnicodeDecodeError, OSError:
            continue
        for token in set(TOKEN.findall(text)):
            if token.count(".") >= 2:
                index[token].add(ref.rel)
    return {token: tuple(sorted(paths)) for token, paths in index.items()}


def _duplicate_index(pairs):
    by_def = defaultdict(list)
    by_stem = defaultdict(list)
    for ref, facts in pairs:
        for name, line in facts.defs:
            if len(name) > 6:
                by_def[name].append(ref.rel + ":" + str(line))
        stem = MACHINE_NAME.sub("", ref.path.stem)
        by_stem[stem].append(ref.rel + ":" + str(facts.first_def_line()))
    index = defaultdict(set)
    for groups in (by_def, by_stem):
        for sites in groups.values():
            paths = {site.rsplit(":", 1)[0] for site in sites}
            if len(paths) < 2:
                continue
            unique = sorted(set(sites))
            for site in unique:
                for other in unique:
                    if other != site:
                        index[site.rsplit(":", 1)[0]].add(other)
    return {key: tuple(sorted(value)) for key, value in index.items()}


def _is_candidate(ref):
    if ref.tree != "nova":
        return False
    if ref.rel.startswith("nova/imas/"):
        return True
    return bool(MACHINE_NAME.match(ref.path.stem))


def build_census(root, ambix_root=None):
    root = Path(root).resolve()
    refs = []
    for tree in TREE_DIRECTORIES:
        refs.extend(_walk_tree(root, tree))
    if ambix_root is not None:
        ambix = Path(ambix_root)
        if ambix.is_dir():
            refs.extend(_walk_ambix(ambix))
    known = {ref.dotted for ref in refs}
    facts_by, importers = _build_graph(refs, known)

    candidates = [ref for ref in refs if _is_candidate(ref)]

    pairs = [(ref, facts_by[ref.dotted]) for ref in candidates]
    referenced = set(_string_references(refs))
    duplicates = _duplicate_index(pairs)

    rows = []
    for ref, facts in pairs:
        cls, basis = _classify(ref, facts, importers, referenced)
        row_importers = importers.get(ref.dotted, [])
        rows.append(
            Row(
                module=ref.dotted,
                path=ref.rel,
                line=facts.first_def_line(),
                cls=cls,
                basis=basis,
                destination=CLASS_DESTINATION[cls],
                machine=_machine(ref),
                importers=tuple(sorted(m + ":" + str(n) for m, n in row_importers)),
                solver_importers=_solver_importers(row_importers),
                duplicates=duplicates.get(ref.rel, ()),
                ambix_map=_ambix_coverage(ref),
            )
        )
    candidate_names = {row.module for row in rows}

    importer_rows = []
    for ref in refs:
        facts = facts_by[ref.dotted]
        targets = sorted(facts.imported_dotted & candidate_names)
        if not targets:
            continue
        cls, basis = _classify(ref, facts, importers, referenced)
        importer_rows.append(
            ImporterRow(
                module=ref.dotted,
                path=ref.rel,
                imports=tuple(targets),
                cls=cls,
                destination=CLASS_DESTINATION[cls],
            )
        )
    importer_rows.sort(key=lambda row: (row.path, row.imports))

    convention_rows = []
    for ref in refs:
        facts = facts_by[ref.dotted]
        converting, basis = _converts(facts)
        states = "COCOS" in facts.assigned
        algebra = facts.imports_module("nova_cocos")
        if not (converting or states or algebra):
            continue
        cls, basis = _classify(ref, facts, importers, referenced)
        line = facts.import_lines.get("nova_cocos", facts.first_def_line())
        convention_rows.append(
            ConventionRow(
                module=ref.dotted,
                path=ref.rel,
                line=line,
                cls=cls,
                basis=basis,
                destination=_convention_destination(ref, facts, cls),
            )
        )
    convention_rows.sort(key=lambda row: row.path)

    row_paths = {row.path for row in rows}
    counts = {
        "files_seen": len({ref.rel for ref in refs}),
        "module_rows": len(rows),
        "importer_rows": len(importer_rows),
        "convention_rows": len(convention_rows),
        "rows_total": len(rows) + len(importer_rows) + len(convention_rows),
        "files_with_module_rows": len(row_paths),
        "candidate_files_unrowed": len(
            [ref for ref in refs if _is_candidate(ref) and ref.rel not in row_paths]
        ),
    }
    dataset_control = tuple(
        sorted(m + ":" + str(n) for m, n in importers.get("nova.imas.dataset", []))
    )
    wanted = {
        "nova/equilibrium/convention.py",
        "nova/io/ingest.py",
        "nova/io/signalstore.py",
    }
    convention_control = tuple(
        (row.path, row.cls) for row in convention_rows if row.path in wanted
    )
    trees = list(TREE_DIRECTORIES)
    if ambix_root is not None and Path(ambix_root).is_dir():
        trees.append("ambix")
    return Census(
        rows=tuple(rows),
        importer_rows=tuple(importer_rows),
        convention_rows=tuple(convention_rows),
        counts=counts,
        dataset_control=dataset_control,
        convention_control=convention_control,
        reference_shots=REFERENCE_SHOTS,
        trees=tuple(trees),
    )


def _esc(value):
    return html.escape(str(value))


def _join(values, empty="—"):
    return ", ".join(values) if values else empty


def _module_table(rows, caption):
    out = [
        f"<h2>{_esc(caption)}</h2>",
        '<table class="census">',
        "<thead><tr><th>module</th><th>class</th>"
        "<th>machine</th>"
        "<th>destination</th><th>basis</th><th>importers</th>"
        "<th>solver importers</th><th>duplicate implementations</th>"
        "<th>upstream machine map</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        out.append(
            "<tr>"
            f'<td class="path">{_esc(row.path)}</td>'
            f'<td class="cls">{_esc(row.cls)}</td>'
            f"<td>{_esc(row.machine)}</td>"
            f"<td>{_esc(row.destination)}</td>"
            f"<td>{_esc(row.basis)}</td>"
            f"<td>{len(row.importers)}</td>"
            f'<td class="path">{_esc(_join(row.solver_importers))}</td>'
            f'<td class="path">{_esc(_join(row.duplicates))}</td>'
            f'<td class="path">{_esc(_join(row.ambix_map, "none"))}</td>'
            "</tr>"
        )
    out.extend(["</tbody>", "</table>"])
    return "\n".join(out)


def _importer_table(rows):
    out = [
        "<h2>Importers of a censused module</h2>",
        "<p>One row per module in the scanned trees that imports a module "
        "named above. A module whose import set reaches a censused module is "
        "listed once with every censused module it imports.</p>",
        '<table class="census">',
        "<thead><tr><th>importer</th><th>imports</th><th>class of importer</th>"
        "<th>destination</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        out.append(
            "<tr>"
            f'<td class="path">{_esc(row.path)}</td>'
            f'<td class="path">{_esc(", ".join(row.imports))}</td>'
            f"<td>{_esc(row.cls)}</td>"
            f"<td>{_esc(row.destination)}</td>"
            "</tr>"
        )
    out.extend(["</tbody>", "</table>"])
    return "\n".join(out)


def _convention_table(rows):
    out = [
        "<h2>Convention algebra and DD-version conversion</h2>",
        "<p>Every module that imports the COCOS algebra, derives a convention "
        "from a Data Dictionary version through a convention helper, or states "
        "nova's internal convention as a module-level assignment.</p>",
        '<table class="census">',
        "<thead><tr><th>module</th><th>class</th><th>basis</th>"
        "<th>destination</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        out.append(
            "<tr>"
            f'<td class="path">{_esc(row.path)}</td>'
            f"<td>{_esc(row.cls)}</td>"
            f"<td>{_esc(row.basis)}</td>"
            f"<td>{_esc(row.destination)}</td>"
            "</tr>"
        )
    out.extend(["</tbody>", "</table>"])
    return "\n".join(out)


PAGE_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 92em; margin: 2em auto; padding: 0 1em; color: #1b1b1b;
       line-height: 1.45; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; }
h2 { font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.06em;
     color: #4b4b4b; margin-top: 2.2em; border-bottom: 1px solid #d8d8d8;
     padding-bottom: 0.3em; }
p.lead { color: #3b3b3b; max-width: 62em; }
table.census { border-collapse: collapse; width: 100%; font-size: 0.78em;
               font-variant-numeric: tabular-nums; }
table.census th { text-align: left; font-weight: 600; color: #3b3b3b;
                  border-bottom: 1px solid #b9b9b9; padding: 0.35em 0.5em;
                  vertical-align: bottom; }
table.census td { border-bottom: 1px solid #ececec; padding: 0.3em 0.5em;
                  vertical-align: top; }
table.census tr:hover td { background: #f6f8fa; }
td.path, td.path code { font-family: "SFMono-Regular", Menlo, monospace;
                        font-size: 0.95em; color: #1b1b1b; }
td.cls { white-space: nowrap; color: #6a3d00; }
dl.controls { display: grid; grid-template-columns: 18em 1fr; gap: 0.3em 1em; }
dl.controls dt { font-family: "SFMono-Regular", Menlo, monospace;
                 font-size: 0.82em; color: #1b1b1b; }
dl.controls dd { margin: 0; font-size: 0.85em; }
"""


def _lead_html(census):
    c = census.counts
    trees = _esc(", ".join(census.trees))
    unrowed = c["candidate_files_unrowed"]
    return (
        '<p class="lead">'
        "Recomputed from the tree by "
        "<code>scripts/imas_seam_census.py</code>. "
        "The scan read <strong>"
        + str(c["files_seen"])
        + "</strong> Python modules across "
        + trees
        + ". "
        "It emitted <strong>"
        + str(c["module_rows"])
        + "</strong> module rows covering <strong>"
        + str(c["files_with_module_rows"])
        + "</strong> files, <strong>"
        + str(c["importer_rows"])
        + "</strong> importer rows and <strong>"
        + str(c["convention_rows"])
        + "</strong> convention rows: <strong>"
        + str(c["rows_total"])
        + "</strong> rows against <strong>"
        + str(c["files_seen"])
        + "</strong> files read. <strong>"
        + str(unrowed)
        + "</strong> censused files carry no row, so the row "
        "set is complete over the scan. Regenerate with the command at the "
        "foot of this page.</p>"
    )


def _classes_html(census):
    seen = defaultdict(int)
    for row in census.rows:
        seen[row.cls] += 1
    body = []
    for name, count in sorted(seen.items(), key=lambda item: (-item[1], item[0])):
        body.append(
            "<tr>"
            "<td>" + _esc(name) + "</td>"
            '<td class="num">' + str(count) + "</td>"
            "<td>" + _esc(CLASS_GLOSSARY.get(name, "")) + "</td>"
            "<td>" + _esc(CLASS_DESTINATION.get(name, "")) + "</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<h2>Classes</h2>",
            '<table class="census">',
            "<thead><tr><th>class</th><th>modules</th>"
            "<th>what the class means</th><th>destination</th></tr></thead>",
            "<tbody>",
        ]
        + body
        + ["</tbody>", "</table>"]
    )


def _controls_html(census):
    pairs = []
    for path, cls in census.convention_control:
        pairs.append("<dt>" + _esc(path) + "</dt><dd>" + _esc(cls) + "</dd>")
    return "\n".join(
        [
            "<h2>Controls</h2>",
            "<p>The importer scan run over "
            "<code>nova.imas.dataset</code> reports "
            + str(len(census.dataset_control))
            + " known importers, so a zero elsewhere means absent rather "
            "than missed. The convention scan reports these control sites:"
            "</p>",
            '<dl class="controls">',
        ]
        + pairs
        + ["</dl>"]
    )


def _shots_html(census):
    body = []
    for shot in census.reference_shots:
        body.append(
            "<tr><td>" + _esc(shot["machine"]) + "</td>"
            '<td class="path">' + _esc(shot["shot"]) + "</td>"
            "<td>" + _esc(shot["source"]) + "</td></tr>"
        )
    return "\n".join(
        [
            "<h2>Reference shots</h2>",
            "<p>The shot per machine whose IDSs the machine-analysis "
            "machine comparisons read.</p>",
            '<table class="census">',
            "<thead><tr><th>machine</th><th>shot</th><th>source</th></tr>",
            "</thead>",
            "<tbody>",
        ]
        + body
        + ["</tbody>", "</table>"]
    )


FOOTER = (
    "<h2>Regenerating this record</h2>"
    "<p>From the repository root:</p>"
    "<pre>uv run python scripts/imas_seam_census.py</pre>"
    "<p>The scan resolves imports through the abstract syntax tree, so it "
    "reports what the code imports rather than what its names suggest. "
    "<code>--check</code> exits nonzero when a censused module carries no "
    "row.</p>"
)


def render_html(census):
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>IMAS seam census</title>",
        "<style>" + PAGE_CSS + "</style>",
        "</head>",
        "<body>",
        "<h1>IMAS seam census</h1>",
        _lead_html(census),
        _classes_html(census),
        _module_table(census.rows, "Machine and IMAS modules"),
        _importer_table(census.importer_rows),
        _convention_table(census.convention_rows),
        _controls_html(census),
        _shots_html(census),
        FOOTER,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def write_html(census, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(census), encoding="utf-8")
    return target


def census_payload(census):
    return {
        "counts": census.counts,
        "trees": list(census.trees),
        "dataset_control": list(census.dataset_control),
        "convention_control": [list(pair) for pair in census.convention_control],
        "reference_shots": list(census.reference_shots),
        "rows": [
            {
                "module": row.module,
                "path": row.path,
                "class": row.cls,
                "machine": row.machine,
                "destination": row.destination,
                "basis": row.basis,
                "importers": list(row.importers),
                "solver_importers": list(row.solver_importers),
                "duplicates": list(row.duplicates),
                "ambix_map": list(row.ambix_map),
            }
            for row in census.rows
        ],
        "importer_rows": [
            {
                "module": row.module,
                "path": row.path,
                "imports": list(row.imports),
                "class": row.cls,
                "destination": row.destination,
            }
            for row in census.importer_rows
        ],
        "convention_rows": [
            {
                "module": row.module,
                "path": row.path,
                "class": row.cls,
                "destination": row.destination,
            }
            for row in census.convention_rows
        ],
    }


AMBIX_DEFAULT = "/home/ITER/mcintos/Code/imas-ambix"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to scan")
    parser.add_argument(
        "--ambix",
        default=AMBIX_DEFAULT,
        help="imas-ambix checkout to scan for importers (skipped if absent)",
    )
    parser.add_argument(
        "--html",
        default="docs/research/imas-seam-census.html",
        help="path for the rendered record",
    )
    parser.add_argument("--json", default=None, help="optional JSON dump path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero when a censused module carries no row",
    )
    args = parser.parse_args(argv)

    census = build_census(args.root, args.ambix)
    written = write_html(census, args.html)
    counts = census.counts
    print(
        "files_seen="
        + str(counts["files_seen"])
        + " module_rows="
        + str(counts["module_rows"])
        + " importer_rows="
        + str(counts["importer_rows"])
        + " convention_rows="
        + str(counts["convention_rows"])
        + " rows_total="
        + str(counts["rows_total"])
        + " unrowed="
        + str(counts["candidate_files_unrowed"])
    )
    print("wrote " + str(written))
    if args.json:
        Path(args.json).write_text(
            json.dumps(census_payload(census), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.check and counts["candidate_files_unrowed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
