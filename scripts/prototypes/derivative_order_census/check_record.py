"""Check the authored cost table against saved measurement receipts."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.cells, self.cell = [], None, None, None

    def handle_starttag(self, tag, attributes):
        if tag == "table":
            self.rows = []
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
            self.tables.append(self.rows)
            self.rows = None


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
    cost = document.tables[0][1:]
    assert len(cost) == 20
    for receipt in summary["rows"]:
        row = next(
            row
            for row in cost
            if row[:2] == [str(receipt["order"]), names[receipt["variant"]]]
        )
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
        assert row[2:7] == expected, (
            receipt["variant"],
            receipt["order"],
            row[2:7],
            expected,
        )
        source = (
            args.summary.parent
            / "rows"
            / (f"derivative-{receipt['variant']}-{receipt['order']}.json")
        )
        source = Path(receipt.get("receipt_path", source))
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest() == receipt["receipt_sha256"]
        )
    accuracy = document.tables[1][1:]
    compared = [row for row in summary["comparisons"] if row["state"] == "compared"]
    for comparison in compared:
        row = next(
            row
            for row in accuracy
            if row[:2] == [str(comparison["order"]), names[comparison["variant"]]]
        )
        expected = [
            f"{comparison['generic_max_relative']:g}",
            f"{comparison['corner_max_relative']:g}",
            f"{comparison['finite_values']} / {comparison['total_values']}",
        ]
        assert row[2:5] == expected, (row, expected)
        assert comparison["positive_control_relative"] > 0.009
        source = (
            args.summary.parent
            / "rows"
            / (f"derivative-{comparison['variant']}-{comparison['order']}.npz")
        )
        source = Path(comparison.get("array_path", source))
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == comparison["array_sha256"]
        )
    print(
        f"RECORD_CHECK measured_cost_rows={len(summary['rows'])} "
        f"requested_cost_rows={len(cost)} accuracy_rows={len(compared)}"
    )


if __name__ == "__main__":
    main()
