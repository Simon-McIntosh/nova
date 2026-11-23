"""The line-map driver refuses a map whose reach checks do not hold.

The map is evidence, and the properties that make it evidence are that a module
no route imports reads zero and that the public solve entry reads nonzero on
every route. Both are asserted here by running the driver against small
synthetic coverage reports, so a driver that only records those numbers fails
these tests instead of writing a plausible record. The stamp tests hold the
two revision checks, and the render test holds the fragment's closing tag.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "prototypes" / "hot_path_map"
DRIVER = SCRIPTS / "map.py"
STAMP = SCRIPTS / "stamp.py"
RENDER = SCRIPTS / "render.py"

MODULES = {
    "nova/pkg/forward.py": 6,
    "nova/pkg/reached.py": 4,
    "nova/pkg/control.py": 5,
}


def build_tree(tmp_path):
    for name, statements in MODULES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n" * statements)


def report(tmp_path, executed):
    """Write a coverage-shaped report; executed maps module name to lines."""
    files = {}
    for name, statements in MODULES.items():
        lines = sorted(executed.get(name, ()))
        files[str(tmp_path / name)] = dict(
            executed_lines=lines,
            missing_lines=[n for n in range(1, statements + 1) if n not in lines],
            summary=dict(
                covered_lines=len(lines), missing_lines=statements - len(lines)
            ),
        )
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(dict(files=files)))
    return path


def run(tmp_path, executed):
    forward = report(tmp_path, executed)
    output = tmp_path / "map.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--forward",
            str(forward),
            "--inverse",
            str(forward),
            "--root",
            str(tmp_path),
            "--revision",
            "test",
            "--tree",
            str(tmp_path),
            "--negative-control",
            "nova/pkg/control.py",
            "--public-entry",
            "nova/pkg/forward.py",
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    return completed, output


def test_clean_map_passes(tmp_path):
    build_tree(tmp_path)
    executed = {"nova/pkg/forward.py": [1, 2], "nova/pkg/reached.py": [1]}
    completed, output = run(tmp_path, executed)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text())
    assert payload["assertions"]["passed"] is True
    assert payload["negative_control"]["forward"]["executed"] == 0


def test_reached_negative_control_is_refused(tmp_path):
    build_tree(tmp_path)
    executed = {"nova/pkg/forward.py": [1, 2], "nova/pkg/control.py": [1, 2, 3]}
    completed, output = run(tmp_path, executed)
    assert completed.returncode == 1, completed.stdout
    assert "negative control" in completed.stdout
    assert "read 3 lines on the forward route" in completed.stdout
    payload = json.loads(output.read_text())
    assert payload["assertions"]["passed"] is False


def test_unreached_public_entry_is_refused(tmp_path):
    build_tree(tmp_path)
    completed, _ = run(tmp_path, {"nova/pkg/reached.py": [1]})
    assert completed.returncode == 1
    assert "read no executed line" in completed.stdout


def git(tree, *arguments):
    return subprocess.run(
        ["git", "-C", str(tree), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def initialise(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "tracked.py").write_text("x = 1\n")
    git(tmp_path, "add", "tracked.py")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        "one",
    )


def test_stamp_refuses_a_dirty_tree(tmp_path):
    initialise(tmp_path)
    clean = subprocess.run(
        [sys.executable, str(STAMP), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout.strip() == git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "tracked.py").write_text("x = 2\n")
    dirty = subprocess.run(
        [sys.executable, str(STAMP), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert dirty.returncode == 1
    assert "dirty" in dirty.stderr


def test_stamp_refuses_a_moved_head(tmp_path):
    initialise(tmp_path)
    moved = subprocess.run(
        [sys.executable, str(STAMP), str(tmp_path), "--expect", "0000"],
        capture_output=True,
        text=True,
    )
    assert moved.returncode == 1
    assert "HEAD moved" in moved.stderr


def test_fragment_closes_once_and_quotes_the_runner_line(tmp_path):
    entry = dict(
        module="nova/equilibrium/forward.py",
        total=6,
        measured=True,
        forward=dict(executed=3, percent=50.0, lines=[]),
        inverse=dict(executed=3, percent=50.0, lines=[]),
        union=dict(executed=3, percent=50.0, lines=[]),
    )
    control = dict(
        module="nova/pkg/control.py",
        total=5,
        measured=True,
        forward=dict(executed=0, percent=0.0, lines=[]),
        inverse=dict(executed=0, percent=0.0, lines=[]),
        union=dict(executed=0, percent=0.0, lines=[]),
    )
    payload = dict(
        revision="test",
        tree="tree",
        negative_control=dict(
            module="nova/pkg/control.py",
            measured=True,
            forward=control["forward"],
            inverse=control["inverse"],
        ),
        device_trace_entries=[],
        device_trace_contradictions=[],
        modules=[entry, control],
    )
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(payload))
    log = tmp_path / "gate.log"
    log.write_text("....\n1 passed in 12.00s\n")
    output = tmp_path / "fragment.html"
    completed = subprocess.run(
        [
            sys.executable,
            str(RENDER),
            "--map",
            str(map_path),
            "--output",
            str(output),
            "--job",
            "1337973",
            "--log",
            str(log),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    text = output.read_text()
    assert text.count("</html>") == 1
    assert text.rstrip().endswith("</html>")
    assert "1 passed in 12.00s" in text
    assert text.index("1 passed in 12.00s") < text.index("</section>")
