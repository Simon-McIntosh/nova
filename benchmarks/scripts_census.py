"""Assign every Python file under ``benchmarks/`` and ``scripts/`` one verdict.

The population is enumerated through ``git ls-files`` -- the tracked-file
semantics ``nova.database.filepath.worktree_path_candidates`` already owns, one
enumeration over both trees and never a directory walk -- and a driver known to
be tracked must appear or the census fails: that positive control makes an empty
population a failure rather than a clean result.

A file receives exactly one verdict:

keep   -- imported by kept code or tests, run by a live gate, lane or test,
          owned by a plan in one of the owning sprints given on the command
          line (``--owning-sprint``, live or held), or the instrument this
          census is applied from;
delete -- reached by none of those. Each delete names every evidence record that
          cites it and the revision that record ran at, and the revision is
          proven to resolve. A cited driver whose record names no resolvable
          revision is refused and named rather than deleted on an unresolvable
          duplicate: citation;
merge  -- a byte-identical duplicate of one or more drivers. It folds into a
          twin that is itself a keep, so a duplicate with no kept twin is a
          delete rather than a merge.

Revisions resolve in one batched ``git cat-file`` pass and files are named
through :func:`nova.database.filepath.repository_relative`, so the census never
shells out to git once per file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from nova.database.filepath import compute_provenance, repository_relative

ROOT = Path(__file__).resolve().parents[1]
TREES = ("benchmarks", "scripts")
SELF = "benchmarks/scripts_census.py"
# The record this census writes; excluded from the evidence scan so the census
# never treats its own inventory as a citation of a run.
RECORD_SELF = "docs/evidence/fragments/code-reuse-and-hot-path/bench-scripts-census"

# A driver that is tracked and imported by tests/test_recovery_ladder_census.py:
# the positive control on enumeration and the subject of the negative control.
POSITIVE = "benchmarks/recovery_ladder_census.py"

# Directories that hold other directories rather than drivers. A plan or a
# record naming one names the tree's layout, not the work parked under it, so
# the name confers no ownership and cites nothing.
CONTAINERS = ("benchmarks", "scripts", "scripts/prototypes")

# This run's own machinery: the census driver, its test and its record. Each
# names drivers as fixtures and as the population, so counting any of them as a
# runner, an importer or a citation would let the census verify itself.
EXCLUDED = (SELF, "tests/test_scripts_census.py")

# The sprints whose plans own code are an input, not a property of this driver:
# a file any plan in one of them names in its own text is live or held work and
# is kept, because held work is parked, never discarded.

PATH_RE = re.compile(r"(?:benchmarks|scripts)/[A-Za-z0-9_./-]+?[.]py")
# An import statement, including one inside a probe string a test writes out and
# runs in a subprocess: the module name must start a line, as ``from`` does, so
# a Sphinx ``:mod:`` reference in a docstring does not read as a run.
IMPORT_STMT_RE = re.compile(
    r"^\s*(?:from|import)\s+((?:benchmarks|scripts)(?:\.[A-Za-z0-9_]+)+)"
)
IMPORT_CALL_RE = re.compile(
    r"import_module\(\s*[\"']((?:benchmarks|scripts)(?:\.[A-Za-z0-9_]+)+)[\"']"
)
QUOTED_PY_RE = re.compile(r"[\"']([A-Za-z0-9_./-]+?[.]py)[\"']")
# A path assembled from quoted segments, e.g. ``... / "benchmarks" / "x.py"``.
# This is how a test reaches a driver through ``importlib.util`` without ever
# writing the joined path, and matching the assembled form (not a bare
# basename) keeps a generic name like ``measure.py`` from matching 67 files.
ASSEMBLED_RE = re.compile(
    r"[\"'](benchmarks|scripts)[\"']((?:\s*/\s*[\"'][^\"']+[\"'])+)"
)
HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b")


class Refused(RuntimeError):
    """A census decision the evidence does not support; the entry is named."""


def git(args):
    done = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("git failed: " + " ".join(args) + "\n" + done.stderr)
    return done.stdout


def tracked():
    return [line for line in git(["ls-files"]).splitlines() if line]


def population(paths):
    files = sorted(
        p for p in paths if p.endswith(".py") and p.split("/", 1)[0] in TREES
    )
    if POSITIVE not in files:
        raise Refused(
            "the population returned nothing it is known to hold (%s), so this "
            "census proves nothing" % POSITIVE
        )
    return files


def module_names(path):
    stem = path[: -len(".py")]
    parts = stem.split("/")
    if parts[-1] == "__init__":
        return {".".join(parts[:-1])}
    return {stem.replace("/", ".")}


def imports_of(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    names = set()
    for node in imports_of_walk(tree):
        names.update(node)
    return names


def imports_of_walk(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found = {node.module}
            for alias in node.names:
                found.add(node.module + "." + alias.name)
            yield found


def _evaluated(node, constants):
    """The path an expression builds, or ``None`` when it is not a path at all.

    A prototype script puts a directory on ``sys.path`` and imports its
    neighbours as bare modules, so the directories on ``sys.path`` are read from
    the expressions that build them: ``Path(__file__).resolve().parent``,
    ``.parents[2]``, a literal appended with ``/``, and a module-level name
    bound to any of those.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Attribute):
        base = _evaluated(node.value, constants)
        if base is None:
            return None
        if node.attr == "parent":
            return str(Path(base).parent)
        return None
    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Attribute):
            return None
        if node.value.attr != "parents" or not isinstance(node.slice, ast.Constant):
            return None
        base = _evaluated(node.value.value, constants)
        if base is None:
            return None
        target = Path(base)
        for _ in range(int(node.slice.value)):
            target = target.parent
        return str(target)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _evaluated(node.left, constants)
        right = _evaluated(node.right, constants)
        if left is None or right is None:
            return None
        return str(Path(left) / right)
    if isinstance(node, ast.Call):
        name = _called_name(node.func)
        arguments = node.args
        if name in ("str", "Path", "resolve") and arguments:
            return _evaluated(arguments[0], constants)
        if name == "resolve" and isinstance(node.func, ast.Attribute):
            return _evaluated(node.func.value, constants)
        if name == "dirname" and arguments:
            inner = _evaluated(arguments[0], constants)
            return None if inner is None else str(Path(inner).parent)
        if name == "abspath" and arguments:
            return _evaluated(arguments[0], constants)
    return None


def _called_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def sys_path_roots(path, tree):
    """Directories this file adds to ``sys.path``, plus its own directory."""
    here = str(Path(path).parent)
    constants = {"__file__": path}
    roots = {here}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _evaluated(node.value, constants)
                if value is not None:
                    constants[target.id] = value
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            _record_path_call(node.value, constants, roots)
        for call in _calls_in(node):
            _record_path_call(call, constants, roots)
    return roots


def _calls_in(node):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            yield inner


def _record_path_call(call, constants, roots):
    """Keep the directory a ``sys.path`` call adds, resolved against the repo."""
    name = _called_name(call.func)
    if name not in ("insert", "append", "extend"):
        return
    if not isinstance(call.func, ast.Attribute):
        return
    owner = call.func.value
    if not isinstance(owner, ast.Attribute) or owner.attr != "path":
        return
    if not isinstance(owner.value, ast.Attribute):
        return
    arguments = call.args
    if name == "insert":
        arguments = arguments[1:]
    for argument in arguments:
        value = _evaluated(argument, constants)
        if value is not None:
            roots.add(value)


def _module_candidates(name, base_dirs):
    out = set()
    for base in base_dirs:
        stem = Path(base) / name if base else Path(name)
        out.add(str(stem) + ".py")
        out.add(str(Path(stem) / "__init__.py"))
    return out


def repo_relative(candidate):
    path = Path(candidate)
    if path.is_absolute():
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return None
    text = str(Path(*[part for part in path.parts if part not in ("", ".")]))
    return text


def sibling_targets(path, tree, popset):
    """Population files this script imports as bare modules from ``sys.path``.

    A prototype puts its own directory, or a directory it builds, on
    ``sys.path`` and then writes ``import panels`` or
    ``from records import ...``. Those names resolve against the directories the
    file itself puts on the path, never against the whole tree."""
    roots = sys_path_roots(path, tree)
    here = str(Path(path).parent)
    found = set()
    for node in ast.walk(tree):
        bases = []
        names = []
        if isinstance(node, ast.Import):
            bases = list(roots)
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = here
            for _ in range(max(node.level - 1, 0)):
                target = str(Path(target).parent)
            bases = [target] if node.level else list(roots)
            if node.module is None:
                names = [alias.name for alias in node.names]
            else:
                names = [node.module.split(".")[0]]
        for name in names:
            for candidate in _module_candidates(name, bases):
                relative = repo_relative(candidate)
                if relative in popset:
                    found.add(relative)
    return found


def ancestor_directories(pop):
    """Every directory holding a population file, the two tree roots excluded.

    A record naming ``scripts/`` or ``scripts/prototypes/`` names no particular
    driver, so only a directory at least one level below the two trees counts as
    citing or owning what it holds."""
    out = set()
    for path in pop:
        parts = path.split("/")[:-1]
        for depth in range(2, len(parts) + 1):
            out.add("/".join(parts[:depth]))
    return out


def files_under(pop, directories):
    mapping = defaultdict(list)
    for path in pop:
        parts = path.split("/")[:-1]
        for depth in range(2, len(parts) + 1):
            mapping["/".join(parts[:depth])].append(path)
    return mapping


def driver_directories(pop):
    """Directories that directly hold a tracked driver, so name held work."""
    return {str(Path(path).parent) for path in pop}


def owning_directories(pop):
    """Driver directories that are not containers of the two trees."""
    return {path for path in driver_directories(pop) if path not in CONTAINERS}


PATH_TOKEN_RE = re.compile(r"(?:benchmarks|scripts)/[A-Za-z0-9_./-]+")


def directory_tokens(text):
    return {token.rstrip("/") for token in PATH_TOKEN_RE.findall(text)}


def assembled_paths(text):
    out = set()
    for root, tail in ASSEMBLED_RE.findall(text):
        segments = [root] + re.findall(r"[\"']([^\"']+)[\"']", tail)
        out.add("/".join(segment for segment in segments if segment))
    return out


def references_in(text, popset, lookup):
    """Candidate drivers a piece of text reaches, in any of three forms.

    A full repository-relative path; a dotted module name starting an import
    statement or an ``importlib.import_module`` call (so a probe string a test
    writes out and runs is caught, while a Sphinx ``:mod:`` reference is not);
    or a path assembled from quoted segments. A bare basename is deliberately
    not matched, because a generic name like ``measure.py`` occurs in 67 files.
    """
    found = set(PATH_RE.findall(text)) & popset
    for name in IMPORT_STMT_RE.findall(text) + IMPORT_CALL_RE.findall(text):
        target = lookup.get(name)
        if target is not None and target in popset:
            found.add(target)
    found |= assembled_paths(text) & popset
    return found


def plan_texts(paths, owning):
    """Each owning plan's own text, keyed by plan slug."""
    out = {}
    for p in paths:
        if not p.startswith("docs/plans/") or not p.endswith(".html"):
            continue
        text = (ROOT / p).read_text(errors="replace")
        match = re.search(r'name="plan-sprint" content="(S\d+)"', text)
        if match and match.group(1) in owning:
            out[Path(p).stem] = text
    return out


#: Where a project's live crew pointers are kept. A queued or running node
#: declares the paths it will write, and those paths are held work whether or
#: not the plan's prose names them.
CREW_HOME = Path(
    os.environ.get("RECKON_CREW_HOME") or Path.home() / ".config/reckon/crew"
)


def repository_scope(entry):
    """A declared write path as ``benchmarks/``- or ``scripts/``-relative."""
    parts = [part for part in str(entry).strip().split("/") if part]
    for index, part in enumerate(parts):
        if part in TREES:
            return "/".join(parts[index:])
    return None


def live_plan_runs(plans):
    """Write paths the plans' live crew runs declare, keyed by plan slug.

    A run is read from its live pointer, which carries the node's declared
    write paths. Only runs of a plan whose text this census reads count, so
    the ownership stays the focused sprint's rather than any run's.
    """
    held = defaultdict(list)
    try:
        pointers = sorted((CREW_HOME / "live").glob("*.json"))
    except OSError:
        return held
    for pointer in pointers:
        try:
            data = json.loads(pointer.read_text(errors="replace"))
        except OSError, ValueError:
            continue
        if not isinstance(data, dict):
            continue
        node = data.get("node") if isinstance(data.get("node"), dict) else {}
        plan = str(data.get("plan") or node.get("plan") or "").strip()
        if plan not in plans:
            continue
        paths = node.get("write_paths")
        if not isinstance(paths, list):
            continue
        run = str(data.get("run_id") or pointer.stem)
        for entry in paths:
            scope = repository_scope(entry)
            if scope is not None:
                held[plan].append((run, scope))
    return held


def duplicates(pop):
    groups = defaultdict(list)
    for p in pop:
        digest = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        groups[digest].append(p)
    return [sorted(names) for names in groups.values() if len(names) > 1]


def batch_exists(queries):
    if not queries:
        return []
    done = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        cwd=str(ROOT),
        input="\n".join(queries) + "\n",
        capture_output=True,
        text=True,
    )
    lines = done.stdout.splitlines()
    if len(lines) != len(queries):
        raise RuntimeError(
            "git cat-file answered %d of %d" % (len(lines), len(queries))
        )
    answers = []
    for line in lines:
        fields = line.split()
        answers.append(bool(fields) and fields[-1] not in ("missing", "ambiguous"))
    return answers


def resolve(citations, texts):
    queries = []
    index = []
    for path, records in citations.items():
        for record in records:
            for token in sorted(set(HEX_RE.findall(texts[record]))):
                index.append((path, record, token))
                queries.append(token + ":" + path)
    answers = batch_exists(queries)
    out = defaultdict(list)
    ok = set()
    for (path, record, token), present in zip(index, answers):
        if present and (path, record) not in ok:
            out[path].append({"record": record, "revision": token, "resolves": True})
            ok.add((path, record))
    for path, records in citations.items():
        for record in records:
            if (path, record) not in ok:
                out[path].append(
                    {"record": record, "revision": None, "resolves": False}
                )
    return dict(out)


def live_test_targets(imports, lookup):
    got = set()
    for path, names in imports.items():
        if not path.startswith("tests/") or path in EXCLUDED:
            continue
        for name in names:
            target = lookup.get(name)
            if target is not None:
                got.add(target)
    return got


def census(force=None, owning=()):
    """The whole census, with ``owning`` naming the sprints whose plans own code."""
    paths = tracked()
    pop = population(paths)
    # ``paths`` is the whole tracked tree (importers, plans and records live
    # outside benchmarks/ and scripts/); the population is the two-tree subset.
    popset = set(pop)
    lookup = {}
    for p in pop:
        for name in module_names(p):
            lookup[name] = p

    sources = {}
    imports = {}
    trees = {}
    for p in paths:
        if not p.endswith(".py"):
            continue
        if p.split("/", 1)[0] not in ("nova", "tests", "apps", "benchmarks", "scripts"):
            continue
        try:
            text = (ROOT / p).read_text(errors="replace")
        except OSError:
            continue
        sources[p] = text
        imports[p] = imports_of(text)
        try:
            trees[p] = ast.parse(text)
        except SyntaxError:
            pass

    core = {
        p
        for p in imports
        if p.split("/", 1)[0] in ("nova", "tests", "apps") and p not in EXCLUDED
    }

    def targets(files):
        """Population files a set of files imports, by name or by sibling path."""
        got = set()
        for p in files:
            for name in imports[p]:
                t = lookup.get(name)
                if t is not None:
                    got.add(t)
            tree = trees.get(p)
            if tree is not None:
                got |= sibling_targets(p, tree, popset)
        return got

    reasons = defaultdict(set)
    for t in targets(core):
        reasons[t].add("imported_by_core")

    for p in paths:
        if p in popset or p.startswith("docs/") or p in EXCLUDED:
            continue
        live = p in ("tox.ini", "pyproject.toml", "conftest.py") or (
            p.startswith(("tests/", "scripts/", ".github/"))
            and p.endswith((".sh", ".py", ".yaml", ".yml", ".toml", ".cfg"))
        )
        if not live:
            continue
        text = sources.get(p)
        if text is None:
            try:
                text = (ROOT / p).read_text(errors="replace")
            except OSError:
                continue
        for t in references_in(text, popset, lookup):
            reasons[t].add("live_runner:" + p)

    plans = plan_texts(paths, set(owning))
    tree_dirs = ancestor_directories(pop)
    under = files_under(pop, tree_dirs)
    owning_dirs = owning_directories(pop)
    for t in pop:
        for slug in sorted(plans):
            if t in plans[slug]:
                reasons[t].add("plan_owned:" + slug)
    for slug in sorted(plans):
        for token in directory_tokens(plans[slug]):
            # A plan naming a driver directory owns everything under it, and a
            # plan naming one driver owns the directory that driver sits in:
            # the rest of that directory is the same held work. A container
            # holds no driver of its own, so naming it claims nothing.
            owned = set()
            if token in owning_dirs:
                owned.update(under[token])
            if token in popset:
                parent = str(Path(token).parent)
                if parent in owning_dirs:
                    owned.update(under.get(parent, []))
            for held in owned:
                reasons[held].add("plan_owned:" + slug)
    # A plan's live crew runs hold the paths they declare, including a node
    # queued behind an unlanded upstream. Plan prose alone cannot see that.
    live_runs = live_plan_runs(plans)
    for slug in sorted(live_runs):
        for run, scope in live_runs[slug]:
            if scope in popset:
                reasons[scope].add("held_run:" + run)
            elif scope in owning_dirs:
                for held in under[scope]:
                    reasons[held].add("held_run:" + run)

    for p in pop:
        if p.endswith("/__init__.py"):
            reasons[p].add("package_init")

    reasons[SELF].add("census_mechanism")

    dups = duplicates(pop)

    frontier = {p for p in popset if reasons.get(p) and p not in EXCLUDED}
    while frontier:
        nxt = {p for p in targets(frontier) if not reasons.get(p)}
        for t in nxt:
            reasons[t].add("imported_by_kept")
        frontier = nxt
    # Name the kept file each transitive import came from, so a keep a reader
    # cannot otherwise explain says which driver reaches it. The bare marker is
    # dropped only once a named importer replaces it: a file kept by nothing but
    # an import would otherwise lose its only reason in this pass.
    for p in pop:
        if not reasons.get(p) or p in EXCLUDED:
            continue
        for t in targets({p}):
            if t != p and reasons.get(t):
                reasons[t].add("imported_by_kept:" + p)
    for p in pop:
        if any(r.startswith("imported_by_kept:") for r in reasons[p]):
            reasons[p].discard("imported_by_kept")

    merge_into = {}
    for names in dups:
        kept = sorted(n for n in names if reasons.get(n))
        if not kept:
            continue
        for n in names:
            if n != kept[0] and not reasons.get(n):
                merge_into[n] = kept[0]

    rows = []
    for p in pop:
        if p in merge_into:
            rows.append(
                {
                    "path": p,
                    "verdict": "merge",
                    "merge_into": merge_into[p],
                    "reason": ["duplicate_of:" + merge_into[p]],
                }
            )
        elif reasons.get(p):
            rows.append({"path": p, "verdict": "keep", "reason": sorted(reasons[p])})
        else:
            rows.append({"path": p, "verdict": "delete", "reason": []})

    if force is not None:
        if force not in popset:
            raise Refused("forced entry is not in the population: " + force)
        held = sorted(reasons.get(force, []))
        if held:
            live = force in live_test_targets(imports, lookup)
            raise Refused(
                "delete refused: %s is a kept driver (%s%s) and cannot be deleted"
                % (force, ", ".join(held), ", imported by a live test" if live else "")
            )
        for row in rows:
            if row["path"] == force:
                row["verdict"] = "delete"
                row["reason"] = ["forced"]

    deletes = {r["path"] for r in rows if r["verdict"] == "delete"}
    citations = defaultdict(list)
    texts = {}
    for p in paths:
        if not p.startswith(("docs/evidence/", "docs/figures/")):
            continue
        if not p.endswith((".html", ".md", ".json", ".txt", ".svg")):
            continue
        # This census's own record lists every driver by name. It is an
        # inventory, not a record that a driver ran, so counting it as evidence
        # would let the census cite itself and change its own verdicts.
        if p.startswith(RECORD_SELF):
            continue
        try:
            text = (ROOT / p).read_text(errors="replace")
        except OSError:
            continue
        texts[p] = text
        hits = set(references_in(text, deletes, lookup))
        # A record that names a driver directory cites every driver under it.
        # A container names no work of its own, so it cites nothing.
        for token in directory_tokens(text) & owning_dirs:
            hits.update(held for held in under[token] if held in deletes)
        for hit in hits:
            if p not in citations[hit]:
                citations[hit].append(p)
    resolved = resolve(citations, texts)

    refused = []
    for row in rows:
        if row["verdict"] != "delete":
            continue
        cites = resolved.get(row["path"], [])
        row["evidence"] = cites
        if any(not cite["resolves"] for cite in cites):
            refused.append(row["path"])
            row["verdict"] = "keep"
            row["reason"] = ["cited_without_resolvable_revision"]
            continue

    counts = defaultdict(int)
    for row in rows:
        counts[row["verdict"]] += 1
    # Carry every verdict, so a reader sees merge=0 rather than a missing key.
    for verdict in ("keep", "delete", "merge"):
        counts.setdefault(verdict, 0)
    counts["total"] = len(rows)
    named = {p: repository_relative(ROOT / p) for p in pop}
    return {
        "revision": git(["rev-parse", "HEAD"]).strip(),
        "provenance": compute_provenance("numpy"),
        "trees": list(TREES),
        "owning_sprints": list(owning),
        "positive_control": POSITIVE,
        "negative_control": "forced delete of " + POSITIVE + " is refused",
        "live_runs": sorted({run for runs in live_runs.values() for run, _ in runs}),
        "counts": dict(counts),
        "reconciles": sum(counts[v] for v in ("keep", "delete", "merge"))
        == counts["total"],
        "refused_deletes": sorted(refused),
        "verdicts": [dict(r, path=named[r["path"]]) for r in rows],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    parser.add_argument("--force-delete", default=None)
    parser.add_argument(
        "--owning-sprint",
        action="append",
        required=True,
        metavar="SPRINT",
        help="a sprint whose plans own code; repeat for each one",
    )
    args = parser.parse_args(argv)
    payload = census(args.force_delete, args.owning_sprint)
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    print("CENSUS counts=%s" % payload["counts"])
    print("CENSUS reconciles=%s" % payload["reconciles"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
