"""Git-safe canonical form for the coil-metrology corpus.

This package transcodes measured coil-alignment workbooks into a
deterministic, content-addressable canonical form and back. The canonical
form of one workbook is a long-format CSV (one row per atomic measured value)
paired with a YAML sidecar recording provenance and the layout needed to
rebuild the workbook. Both are byte-stable, so a sha256 over the CSV is a
meaningful content address, and the round trip
workbook -> canonical -> workbook preserves every numeric value exactly.
"""
