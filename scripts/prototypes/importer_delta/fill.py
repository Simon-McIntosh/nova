"""Render the delta fragment from the device tables, the run's facts and the template.

    fill.py --template <section fragment> --meta <facts file> --deltas <dir>
            --base-rows <receipts file> --target <file>

The template carries the section markup and the method; every count, every row
and every named file comes from the tables the gate built, so the prose counts
cannot drift from the rows beneath them. The facts file carries what the tables
do not: the two revisions, the trees each arm resolved over, the job ids and
nodes, and the paths a reader needs to reach the receipts.
"""

import argparse
import json
from pathlib import Path


def table_rows(path):
    data = json.loads(Path(path).read_text())
    return data["rows"] if isinstance(data, dict) else data


def one_row(row):
    cell = "<tr><td><code>" + row["file"] + "</code></td>"
    if row["state"] == "not-run":
        return cell + "<td>not run</td><td>not run</td><td>no verdict</td></tr>"
    if row["state"] == "head-pending":
        return (
            cell + "<td>" + str(row["base_count"]) + "</td><td>not run</td>"
            "<td>no verdict, head arm did not start</td></tr>"
        )
    if row["state"] == "head-other-revision":
        return (
            cell
            + "<td>"
            + str(row["base_count"])
            + "</td><td>another revision</td><td>no verdict, the recorded head "
            "arm measured <code>"
            + (row.get("measured_head_revision") or "an unstated revision")[:9]
            + "</code></td></tr>"
        )
    base = "no base arm" if row["head_only"] else str(row["base_count"])
    if not row["head_completed"]:
        return (
            cell
            + "<td>"
            + base
            + "</td><td>"
            + str(row["head_count"])
            + ", killed</td><td>unreadable</td></tr>"
        )
    if row["head_only"]:
        added = "no base arm"
    elif not row["added"]:
        added = "none"
    else:
        added = "<br>".join("<code>" + item + "</code>" for item in row["added"])
    return (
        cell
        + "<td>"
        + base
        + "</td><td>"
        + str(row["head_count"])
        + "</td><td>"
        + added
        + "</td></tr>"
    )


def table(rows, key):
    lines = [
        '<table id="importer-delta-' + key + '-table">',
        "<thead><tr><th>test file</th><th>base failures</th>"
        "<th>head failures</th><th>added at head</th></tr></thead><tbody>",
    ]
    lines.extend(one_row(row) for row in rows)
    lines.append("</tbody></table>")
    return "\n".join(lines)


def classify(rows):
    measured = [row for row in rows if row["state"] == "measured"]
    return (
        measured,
        [r for r in measured if not r["head_only"] and r["head_completed"]],
        [r for r in measured if not r["head_only"] and not r["head_completed"]],
        [r for r in measured if r["head_only"]],
        [r for r in rows if r["state"] != "measured"],
    )


def names(rows):
    return ", ".join("<code>" + row["file"] + "</code>" for row in rows)


def verdict(cpu, card, base_files):
    cpu_measured, cpu_pairs, cpu_killed, cpu_only, cpu_absent = classify(cpu)
    card_measured, card_pairs, card_killed, card_only, card_absent = classify(card)
    measured = cpu_measured + card_measured
    in_base = [row for row in measured if not row["head_only"]]
    pairs = cpu_pairs + card_pairs
    killed = cpu_killed + card_killed
    head_only = cpu_only + card_only
    added_cpu = sum(len(row["added"]) for row in cpu)
    added_card = sum(len(row["added"]) for row in card)
    parts = [
        "Added failure ids: <strong>" + str(added_cpu) + " on CPU</strong> and "
        "<strong>" + str(added_card) + " on the H200</strong>.",
        "This verdict covers "
        + str(len(in_base))
        + " of the "
        + str(base_files)
        + " test files in the base set. "
        + str(len(pairs))
        + " of those give a pair, "
        "a base arm and a head arm that both completed on the same device, and none of "
        "them adds a failure id: "
        + str(len(cpu_pairs))
        + " on CPU and "
        + str(len(card_pairs))
        + " on the H200.",
    ]
    if killed:
        parts.append(
            "Of the " + str(len(in_base)) + ", one ran a head arm that "
            "did not reach a summary line: "
            + names(killed)
            + ". That row states no verdict and is "
            "reported as unreadable below."
        )
    if cpu_absent or card_absent:
        parts.append(
            "The remaining "
            + str(len(cpu_absent) + len(card_absent))
            + " of the "
            + str(base_files)
            + " give no pair: "
            + "; ".join(describe(row) for row in cpu_absent + card_absent)
            + "."
        )
    if head_only:
        parts.append(
            "The "
            + str(len(head_only))
            + " test files the head adds, "
            + names(head_only)
            + ", have no base arm, are marked as such, and "
            + "lie outside the "
            + str(base_files)
            + "."
        )
    return " ".join(parts)


def describe(row):
    if row["state"] == "head-pending":
        return (
            "<code>" + row["file"] + "</code> has a base arm and a head arm that "
            "never started"
        )
    if row["state"] == "head-other-revision":
        return (
            "<code>"
            + row["file"]
            + "</code> has a base arm and a head arm at <code>"
            + (row.get("measured_head_revision") or "an unstated revision")[:9]
            + "</code>, which is not this head"
        )
    return "<code>" + row["file"] + "</code> has neither arm"


def arm_facts(row, which):
    text = []
    if row.get(which + "_summary"):
        text.append(row[which + "_summary"])
    if row.get(which + "_seconds") is not None:
        text.append("%d s" % int(row[which + "_seconds"]))
    if row.get(which + "_peak_kb"):
        text.append("%.0f GB peak" % (row.get(which + "_peak_kb") / 1048576.0))
    return ", ".join(text) if text else "nothing recorded"


def unreadable(cpu):
    dead = [
        row for row in cpu if row["state"] == "measured" and not row["head_completed"]
    ]
    if not dead:
        return "Every arm below reached its summary line."
    return (
        "Unreadable: "
        + "; ".join(
            "<code>"
            + row["file"]
            + "</code> base arm "
            + arm_facts(row, "base")
            + ", head arm "
            + arm_facts(row, "head")
            for row in dead
        )
        + ". A row with no head summary line states no verdict: no id was "
        "recorded for it, which is not the same as no id failing."
    )


def card_note(card, meta):
    walls = meta.get("cpu_walls_seconds") or {}
    order = [row for row in card if row["file"] in walls]
    parts = []
    if order:
        spell = ", ".join(
            "<code>"
            + row["file"].split("/")[-1]
            + "</code> "
            + str(walls[row["file"]])
            + " s"
            for row in sorted(order, key=lambda r: -walls[r["file"]])
        )
        parts.append(
            "These four files are the ones whose CPU arms run longest at "
            "the recorded revision: " + spell + "."
        )
    for job in (meta.get("jobs") or {}).get("h200", []):
        if job.get("held") is False:
            parts.append(
                "Job <code>"
                + job["id"]
                + "</code> was submitted and never started"
                + (": " + job["note"] if job.get("note") else "")
                + "."
            )
            continue
        parts.append(
            "Job <code>"
            + job["id"]
            + "</code> held one card on <code>"
            + job["node"]
            + "</code>"
            + (", " + job["note"] if job.get("note") else "")
            + "."
        )
    for row in card:
        if row["state"] == "head-pending":
            parts.append(
                "The base arm of <code>"
                + row["file"]
                + "</code> completed in "
                + str(int(row["base_seconds"]))
                + " s with "
                + str(row["base_count"])
                + " failure ids, and its head arm did "
                "not start."
            )
        elif row["state"] == "head-other-revision":
            parts.append(
                "The head arm recorded for <code>"
                + row["file"]
                + "</code> measured <code>"
                + (row.get("measured_head_revision") or "an unstated revision")[:9]
                + "</code> rather than the revision under test, so it is not this "
                "head's arm and the row carries no verdict."
            )
        elif row["state"] == "not-run":
            parts.append(
                "No arm of <code>" + row["file"] + "</code> ran to a receipt."
            )
    for job in meta.get("cancelled") or []:
        parts.append(
            "Job <code>" + job["id"] + "</code> " + job["note"].rstrip(".") + "."
        )
    parts.append(
        "A row reading not run in both failure cells has neither arm; a row "
        "whose head cell reads not run but whose base cell carries a count has "
        "a base arm and no head arm. Neither carries a verdict. An unmeasured "
        "row is left for a later dispatch rather than answered by the head arm "
        "of the other device, which is a different device measuring a "
        "different arm."
    )
    return '<p id="importer-delta-card-note">' + " ".join(parts) + "</p>"


def receipts(meta):
    items = [
        "<li>base revision <code>" + meta["base_revision"] + "</code>; head revision "
        "<code>" + meta["head_revision"] + "</code></li>",
        "<li>each arm resolved over an extracted tree of its own revision: base "
        "<code>"
        + meta["base_tree"]
        + "</code> at <code>"
        + meta["base_tree_revision"]
        + "</code>, head <code>"
        + meta["head_tree"]
        + "</code> at <code>"
        + meta["head_tree_revision"]
        + "</code>, both archived from that revision and reached through "
        "<code>PYTHONPATH</code> rather than from the working checkout</li>",
    ]
    for job in (meta.get("jobs") or {}).get("h200", []):
        items.append(
            "<li>H200 job <code>"
            + job["id"]
            + "</code> on <code>"
            + job["node"]
            + "</code></li>"
        )
    for job in (meta.get("jobs") or {}).get("cpu", []):
        items.append(
            "<li>CPU job <code>"
            + job["id"]
            + "</code> on <code>"
            + job["node"]
            + "</code></li>"
        )
    items.append(
        "<li>per-file logs, receipts and the device tables: <code>"
        + meta["receipts_dir"]
        + "</code></li>"
    )
    items.append(
        "<li>gate <code>bash " + meta["gate"] + " &lt;receipts-dir&gt; "
        "&lt;base-rows&gt; &lt;out-dir&gt; &lt;device-config&gt;...</code>, "
        "log <code>" + meta["gate_log"] + "</code>. It reads the rows and passes "
        "only when every declared file carries a completed pair on one device "
        "and no row adds an id: "
        + meta["gate_note"]
        + "</li>"
    )
    items.append(
        "<li>comparison controls <code>"
        + meta["control"]
        + "</code>, log <code>"
        + meta["control_log"]
        + "</code></li>"
    )
    return '\n<ul id="importer-delta-receipts">\n' + "\n".join(items) + "\n</ul>"


def earlier_rows(dirpath):
    if not dirpath:
        return []
    deltas = Path(dirpath)
    rows = []
    for name in ("delta-cpu.json", "delta-h200.json"):
        path = deltas / name
        if path.exists():
            rows.extend(table_rows(path))
    return rows


def earlier(dirpath, meta):
    rows = earlier_rows(dirpath)
    if not rows:
        return ""
    return table(rows, "earlier")


def earlier_note(dirpath, meta):
    rows = earlier_rows(dirpath)
    if not rows:
        return ""
    dead = [
        row
        for row in rows
        if not (row["state"] == "measured" and row["head_completed"])
    ]
    return (
        '<p id="importer-delta-earlier-note">These are the same files measured '
        "against the earlier head, <code>"
        + meta["earlier_head"]
        + "</code>, before its repair. "
        + str(len(dead))
        + " of these rows carry no head arm on the device that owns them, so the "
        "table is kept as a record of an incomplete head rather than as a "
        "verdict; the table above supersedes it. A row with no base arm is a file "
        "the head adds, and is marked as such.</p>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--deltas", required=True)
    parser.add_argument("--base-rows", required=True)
    parser.add_argument("--earlier-deltas")
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    deltas = Path(args.deltas)
    cpu = table_rows(deltas / "delta-cpu.json")
    card = table_rows(deltas / "delta-h200.json")
    meta = json.loads(Path(args.meta).read_text())
    base_files = len(table_rows(args.base_rows))

    text = Path(args.template).read_text()
    text = text.replace("VERDICT_LINE", verdict(cpu, card, base_files))
    text = text.replace("CPU_TABLE", table(cpu, "cpu"))
    text = text.replace("UNREADABLE_LINE", unreadable(cpu))
    text = text.replace("H200_NOTE", card_note(card, meta))
    text = text.replace("H200_TABLE", table(card, "card"))
    text = text.replace("EARLIER_TABLE", earlier(args.earlier_deltas, meta))
    text = text.replace("EARLIER_NOTE", earlier_note(args.earlier_deltas, meta))
    text = text.replace("RECEIPTS_BLOCK", receipts(meta))
    if "_LINE" in text or "_TABLE" in text or "_NOTE" in text or "BLOCK" in text:
        raise SystemExit("a placeholder was left unfilled")
    Path(args.target).write_text(text if text.endswith("\n") else text + "\n")
    print("wrote " + args.target)
    print("rows cpu=%d card=%d base_files=%d" % (len(cpu), len(card), base_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
