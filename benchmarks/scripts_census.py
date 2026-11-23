"""Assign every Python file under ``benchmarks/`` and ``scripts/`` one verdict.

The population is enumerated through ``git ls-files`` -- the tracked-file
semantics ``nova.database.filepath.worktree_path_candidates`` already owns, one
enumeration over both trees and never a directory walk -- and a driver known to
be tracked must appear or the census fails: that positive control makes an empty
population a failure rather than a clean result.

A file receives exactly one verdict:

keep   -- imported by kept code or tests, run by a live gate, lane or test,
          owned by a plan in a focused sprint (S28-S32, live or held), or the
          instrument this census is applied from;
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

# A file any focused-sprint plan names in its own text is live or held work and
# is kept: held work is parked, never discarded.
OWNING = ("S28", "S29", "S30", "S31", "S32")

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


def plan_texts(paths):
    out = {}
    for p in paths:
        if not p.startswith("docs/plans/") or not p.endswith(".html"):
            continue
        text = (ROOT / p).read_text(errors="replace")
        match = re.search(r'name="plan-sprint" content="(S\d+)"', text)
        if match and match.group(1) in OWNING:
            out[Path(p).stem] = text
    return out


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
        if not path.startswith("tests/"):
            continue
        for name in names:
            target = lookup.get(name)
            if target is not None:
                got.add(target)
    return got


def census(force=None):
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

    core = {p for p in imports if p.split("/", 1)[0] in ("nova", "tests", "apps")}

    def targets(files):
        got = set()
        for p in files:
            for name in imports[p]:
                t = lookup.get(name)
                if t is not None:
                    got.add(t)
        return got

    reasons = defaultdict(set)
    for t in targets(core):
        reasons[t].add("imported_by_core")

    for p in paths:
        if p in popset or p.startswith("docs/"):
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

    plans = plan_texts(paths)
    for t in pop:
        for slug in sorted(plans):
            if t in plans[slug]:
                reasons[t].add("plan_owned:" + slug)

    for p in pop:
        if p.endswith("/__init__.py"):
            reasons[p].add("package_init")

    reasons[SELF].add("census_mechanism")

    dups = duplicates(pop)

    frontier = {p for p in popset if reasons.get(p)}
    while frontier:
        nxt = {p for p in targets(frontier) if not reasons.get(p)}
        for t in nxt:
            reasons[t].add("imported_by_kept")
        frontier = nxt

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
        for hit in references_in(text, deletes, lookup):
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
    counts["total"] = len(rows)
    named = {p: repository_relative(ROOT / p) for p in pop}
    return {
        "revision": git(["rev-parse", "HEAD"]).strip(),
        "provenance": compute_provenance("numpy"),
        "trees": list(TREES),
        "positive_control": POSITIVE,
        "negative_control": "forced delete of " + POSITIVE + " is refused",
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
    args = parser.parse_args(argv)
    payload = census(args.force_delete)
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
