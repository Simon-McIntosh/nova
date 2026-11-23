"""Verify authored tables, raw receipt hashes, resource stops and accuracy cohorts."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.cells, self.cell = {}, None, None, None
        self.identity = None

    def handle_starttag(self, tag, attributes):
        if tag == "table":
            self.rows = []
            self.identity = dict(attributes).get("id")
        elif tag == "tr":
            self.cells = []
        elif tag in {"td", "th"}:
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.cells.append(self.cell.strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.cells)
        elif tag == "table":
            self.tables[self.identity] = self.rows
            self.rows = None


def verify_digest(path, expected):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path


def find(rows, key):
    matches = [row for row in rows if row[: len(key)] == key]
    assert len(matches) == 1, (key, matches)
    return matches[0]


def number(value):
    return "Unavailable" if value is None else f"{value:.9g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    document = Tables()
    document.feed(args.document.read_text())
    names = {
        "current": "Current",
        "plain": "Plain autodiff",
        "without_samples": "Without samples",
        "masked_samples": "Masked samples",
    }
    cost = document.tables["cost-table"][1:]
    assert len(cost) == len(summary["rows"]) == 20
    native = document.tables["native-table"][1:]
    native_count = 0
    for receipt in summary["rows"]:
        row = find(cost, [str(receipt["order"]), names[receipt["variant"]]])
        fields = (
            "stablehlo_bytes",
            "hlo_instructions",
            "packed_kernel_ops",
            "packed_kernel_instances",
            "temp_bytes",
        )
        expected = [
            "Unavailable" if receipt[field] is None else f"{receipt[field]:,}"
            for field in fields
        ]
        assert row[2:7] == expected, (row, expected)
        verify_digest(receipt["receipt_path"], receipt["receipt_sha256"])
        outcome = receipt.get("native_outcome")
        if outcome:
            native_count += 1
            verify_digest(outcome["time_log"], outcome["time_sha256"])
            verify_digest(outcome["index_path"], outcome["index_sha256"])
            actual = find(native, [names[receipt["variant"]]])
            stop = "RSS limit" if "RSS" in outcome["limit_reason"] else "Wall limit"
            assert actual[1:] == [
                f"{outcome['seconds']:.3f}",
                f"{outcome['time_peak_rss_kib'] / 2**20:.3f}",
                stop,
                str(outcome["exit"]),
                "Unavailable",
            ]
            assert outcome["exit"] != 0 and receipt["temp_bytes"] is None
            assert row[7] == stop
        else:
            assert row[7] == (
                "Complete" if receipt["phase"] == "complete" else "Lowering only"
            )
    assert len(native) == native_count == 4
    accuracy = document.tables["accuracy-table"][1:]
    compared = [row for row in summary["comparisons"] if row["state"] == "compared"]
    assert len(accuracy) == len(compared) == 12
    for comparison in compared:
        row = find(accuracy, [str(comparison["order"]), names[comparison["variant"]]])
        expected = [
            str(comparison["generic_targets"]),
            str(comparison["corner_targets"]),
            number(comparison["generic_max_relative"]),
            number(comparison["corner_max_relative"]),
            f"{comparison['finite_values']} / {comparison['total_values']}",
            comparison["candidate_mode"],
        ]
        assert row[2:] == expected, (row, expected)
        assert comparison["positive_control_relative"] > 0.009
        verify_digest(comparison["array_path"], comparison["array_sha256"])
        verify_digest(comparison["reference_path"], comparison["reference_sha256"])
        if comparison["corner_targets"] == 0:
            assert comparison["corner_max_relative"] is None
    print(
        f"RECORD_CHECK cost_rows={len(cost)} native_censored_rows={native_count} "
        f"accuracy_rows={len(compared)} receipt_and_array_digests=verified"
    )


if __name__ == "__main__":
    main()
