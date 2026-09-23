"""Census of nova modules that import an extra-provided package at module level.

A package that only an optional extra of ``pyproject.toml`` provides is absent
from a default install, so a module-level import of it makes the importing
module unimportable without that extra. Imports guarded by a ``try`` whose
handler catches ``ImportError`` are excluded: that guard is the intended way to
keep an extra optional, and reporting it would drown the real findings.

Every module under ``nova/`` is parsed with ``ast``; the census records each
un-guarded module-level import whose top-level package is provided by at least
one extra and by no core dependency. The distribution-to-import-name mapping is
read from the installed environment, with a normalised-name fallback recorded
for any distribution that is not installed.

Run from the repository root:

    python benchmarks/optional_extra_import_census.py
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SOURCE_ROOT = REPO_ROOT / "nova"
OUTPUT_DIR = (
    REPO_ROOT / "docs" / "figures" / "forward-solve-api" / "optional-extra-imports"
)
REPORT_MD = OUTPUT_DIR / "report.md"
REPORT_JSON = OUTPUT_DIR / "report.json"
LOG_PATH = OUTPUT_DIR / "census.log"

POSITIVE_CONTROL = ("nova/geometry/section.py", 8, "vedo")

_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError"}


def normalise(distribution: str) -> str:
    """PEP 503 normalisation, so ``PyYAML``, ``pyyaml`` and ``Py_YAML`` agree."""
    return re.sub(r"[-_.]+", "-", distribution).lower()


def distribution_name(requirement: str) -> str | None:
    """The distribution name at the head of a PEP 508 requirement string."""
    match = _REQUIREMENT_NAME.match(requirement.strip())
    return match.group(1) if match else None


def distribution_module_map() -> dict[str, set[str]]:
    """Map each normalised distribution name to the import names it provides."""
    modules: dict[str, set[str]] = {}
    for module, distributions in packages_distributions().items():
        for distribution in distributions:
            modules.setdefault(normalise(distribution), set()).add(module)
    return modules


def provided_modules(
    requirements: list[str],
    module_map: dict[str, set[str]],
    unmapped: list[str],
) -> set[str]:
    """Import names provided by a list of requirement strings."""
    provided: set[str] = set()
    for requirement in requirements:
        name = distribution_name(requirement)
        if name is None:
            continue
        key = normalise(name)
        observed = module_map.get(key)
        if observed:
            provided |= observed
        else:
            unmapped.append(key)
            provided.add(key.replace("-", "_"))
    return provided


def exception_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple):
        names: list[str] = []
        for element in node.elts:
            names.extend(exception_names(element))
        return names
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def catches_import_error(try_node: ast.Try) -> bool:
    for handler in try_node.handlers:
        if handler.type is None:
            return True
        if _GUARD_EXCEPTIONS.intersection(exception_names(handler.type)):
            return True
    return False


def is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def module_level_imports(
    body: list[ast.stmt],
    guarded: bool,
    annotation_only: bool,
    found: list[tuple[ast.stmt, bool, bool]],
) -> None:
    """Collect module-level imports, descending compound statements but not
    function or class bodies. ``guarded`` tracks whether a try/except ImportError
    encloses the import; ``annotation_only`` tracks whether it sits under an
    ``if TYPE_CHECKING:`` block, where it is never executed."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append((node, guarded, annotation_only))
        elif isinstance(node, ast.Try | ast.TryStar):
            nested = guarded or catches_import_error(node)
            module_level_imports(node.body, nested, annotation_only, found)
            module_level_imports(node.orelse, guarded, annotation_only, found)
            module_level_imports(node.finalbody, guarded, annotation_only, found)
            for handler in node.handlers:
                module_level_imports(handler.body, guarded, annotation_only, found)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                module_level_imports(case.body, guarded, annotation_only, found)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        else:
            nested_annotation = annotation_only or (
                isinstance(node, ast.If) and is_type_checking_test(node.test)
            )
            for field, value in ast.iter_fields(node):
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, ast.stmt) for item in value)
                ):
                    module_level_imports(value, guarded, nested_annotation, found)


def imported_packages(node: ast.Import | ast.ImportFrom) -> list[tuple[str, int]]:
    """(top-level package, line) pairs for one import statement."""
    packages: list[tuple[str, int]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top:
                packages.append((top, node.lineno))
        return packages
    if node.level:
        return packages
    top = (node.module or "").split(".")[0]
    if top:
        packages.append((top, node.lineno))
    return packages


def scan_sources(extra_only: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
    findings: list[dict] = []
    guarded_hits: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collected: list[tuple[ast.stmt, bool, bool]] = []
        module_level_imports(tree.body, False, False, collected)
        for node, guarded, annotation_only in collected:
            for package, line in imported_packages(node):
                if package not in extra_only:
                    continue
                record = {
                    "module": relative,
                    "line": line,
                    "package": package,
                    "extras": extra_only[package],
                    "guarded": guarded,
                    "annotation_only": annotation_only,
                }
                if guarded:
                    guarded_hits.append(f"{relative}:{line} {package}")
                else:
                    findings.append(record)
    findings.sort(key=lambda item: (item["module"], item["line"], item["package"]))
    return findings, guarded_hits


def render_report(
    context: dict,
    findings: list[dict],
    guarded_hits: list[str],
) -> str:
    runtime = [item for item in findings if not item["annotation_only"]]
    annotations = [item for item in findings if item["annotation_only"]]
    lines = [
        "# Optional-extra module-level import census",
        "",
        "Modules under `nova/` that import, at module level, a package provided",
        "only by an optional extra of `pyproject.toml`. Every import listed is a",
        "module-level statement; imports guarded by a `try` whose handler catches",
        "`ImportError` are excluded, because that guard is the intended way to keep",
        "an extra optional.",
        "",
        "## Findings",
        "",
        "Each import here executes when the module is imported from a default",
        "install, where the extra is absent, so the importing module is",
        "unimportable. The extras column names every extra that provides the",
        "package.",
        "",
        "- Generated by: `python benchmarks/optional_extra_import_census.py`",
        f"- Source revision: `{context['revision']}`",
        f"- Core dependencies: {len(context['core_dependencies'])}",
        f"- Extras: {len(context['extras'])}",
        f"- Findings: {len(findings)} ({len(runtime)} at runtime, "
        f"{len(annotations)} under `if TYPE_CHECKING`)",
        f"- Guarded (excluded) hits: {len(guarded_hits)}",
        "",
    ]
    if runtime:
        lines += ["| Module | Line | Package | Extras |", "| --- | --- | --- | --- |"]
        for item in runtime:
            extras = ", ".join(item["extras"])
            lines.append(
                f"| `{item['module']}` | {item['line']} | `{item['package']}` "
                f"| {extras} |"
            )
    else:
        lines.append("_None._")
    lines += [
        "",
        "## Type-checking-only imports",
        "",
        "Module-level imports inside an `if TYPE_CHECKING:` block. They are listed",
        "because they are module-level statements, but they are never executed, so",
        "they do not break an import; the runtime findings above are the ones that",
        "do.",
        "",
    ]
    if annotations:
        lines += ["| Module | Line | Package | Extras |", "| --- | --- | --- | --- |"]
        for item in annotations:
            extras = ", ".join(item["extras"])
            lines.append(
                f"| `{item['module']}` | {item['line']} | `{item['package']}` "
                f"| {extras} |"
            )
    else:
        lines.append("_None._")
    lines += ["", "## Positive control", ""]
    control = context["positive_control"]
    state = "found" if control["found"] else "MISSING"
    lines.append(
        f"- `{control['module']}:{control['line']}` `{control['package']}` — {state}"
        " (a known un-guarded module-level import of the `test`/`viz` extra, so a"
        " census that does not list it is broken rather than clean)."
    )
    lines += ["", "## Extra-only packages observed", ""]
    observed = sorted({item["package"] for item in findings})
    if observed:
        lines += ["| Package | Provided by extras |", "| --- | --- |"]
        for package in observed:
            extras = ", ".join(context["extra_only_packages"][package])
            lines.append(f"| `{package}` | {extras} |")
    else:
        lines.append("_None._")
    lines += ["", "## Guarded hits (excluded)", ""]
    if guarded_hits:
        lines += [f"- `{hit}`" for hit in guarded_hits]
    else:
        lines.append("_None._")
    lines += [""]
    return "\n".join(lines)


def main() -> int:
    transcript: list[str] = []

    def emit(message: str) -> None:
        print(message)
        transcript.append(message)

    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision_label = revision.stdout.strip() or "unknown"
    emit(f"revision={revision_label} tree={REPO_ROOT}")
    emit("command=python benchmarks/optional_extra_import_census.py")

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    project_key = normalise(project["name"])
    core_dependencies: list[str] = list(project.get("dependencies", []))
    extras: dict[str, list[str]] = {
        name: list(requirements)
        for name, requirements in project.get("optional-dependencies", {}).items()
    }

    module_map = distribution_module_map()
    unmapped: list[str] = []

    core_modules = provided_modules(core_dependencies, module_map, unmapped)

    extra_modules: dict[str, list[str]] = {}
    extra_provider: dict[str, set[str]] = {}
    for extra_name, requirements in extras.items():
        self_referential = [
            requirement
            for requirement in requirements
            if distribution_name(requirement) is not None
            and normalise(distribution_name(requirement) or "") == project_key
        ]
        own = [
            requirement
            for requirement in requirements
            if requirement not in self_referential
        ]
        modules = provided_modules(own, module_map, unmapped)
        extra_modules[extra_name] = sorted(modules)
        for module in modules:
            extra_provider.setdefault(module, set()).add(extra_name)
        if self_referential:
            emit(
                f"skipped self-reference in extra {extra_name!r}: "
                f"{sorted(self_referential)}"
            )

    extra_only = {
        module: sorted(names)
        for module, names in extra_provider.items()
        if module not in core_modules
    }
    emit(f"core import names: {len(core_modules)}")
    emit(f"extra import names: {len(extra_provider)}")
    emit(f"extra-only import names: {len(extra_only)}")
    if unmapped:
        emit(
            "distribution->module mapping fell back to a normalised name for "
            f"{sorted(set(unmapped))}"
        )

    findings, guarded_hits = scan_sources(extra_only)
    for item in findings:
        emit(
            f"FINDING {item['module']}:{item['line']} {item['package']} "
            f"extras={','.join(item['extras'])}"
        )
    emit(f"findings: {len(findings)}")
    emit(f"guarded hits excluded: {len(guarded_hits)}")

    control_module, control_line, control_package = POSITIVE_CONTROL
    control_found = any(
        item["module"] == control_module
        and item["line"] == control_line
        and item["package"] == control_package
        for item in findings
    )
    emit(
        f"POSITIVE CONTROL {control_module}:{control_line} {control_package} "
        f"{'found' if control_found else 'MISSING'}"
    )

    context = {
        "revision": revision_label,
        "core_dependencies": core_dependencies,
        "extras": extras,
        "extra_only_packages": extra_only,
        "guarded_hits": guarded_hits,
        "unmapped_distributions": sorted(set(unmapped)),
        "positive_control": {
            "module": control_module,
            "line": control_line,
            "package": control_package,
            "found": control_found,
        },
    }

    report = {
        "generated_by": "benchmarks/optional_extra_import_census.py",
        "finding_count": len(findings),
        "guard_excluded_count": len(guarded_hits),
        "findings": findings,
        **context,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        render_report(context, findings, guarded_hits), encoding="utf-8"
    )
    LOG_PATH.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    emit(f"wrote {REPORT_MD.relative_to(REPO_ROOT)}")
    emit(f"wrote {REPORT_JSON.relative_to(REPO_ROOT)}")
    emit(f"wrote {LOG_PATH.relative_to(REPO_ROOT)}")

    if not control_found:
        emit("positive control missing: census is broken, not clean")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
