"""Attribute optimized-HLO instructions to live-map application sites.

A site is the scope path in front of the first map-internal scope segment
(``compact_clipped_current_moments`` or ``read_qualification``), with any
trailing ``jit(evaluate)`` shared-body segment kept in the site name so a
shared request body is distinguishable from a direct map call.
"""

import collections
import json
import re
import sys

MARKERS = ("compact_clipped_current_moments", "read_qualification")
INSTRUCTION = re.compile(r'^\s*(?:ROOT\s+)?%[\w.-]+ = .*?op_name="([^"]*)"', re.M)


def site_of(op_name):
    parts = op_name.split("/")
    for index, part in enumerate(parts):
        if any(marker in part for marker in MARKERS):
            prefix = parts[:index]
            kind = "jvp" if part.startswith(("jvp(", "transpose(")) else "primal"
            if part.startswith("transpose("):
                kind = "transpose"
            return "/".join(prefix), kind
    return None, None


def census(path):
    text = open(path).read()
    total = len(re.findall(r"^\s*(?:ROOT\s+)?%[\w.-]+ = ", text, re.M))
    sites = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    for match in INSTRUCTION.finditer(text):
        site, kind = site_of(match.group(1))
        if site is not None:
            sites[site] += 1
            kinds[site][kind] += 1
    return total, sites, kinds


if __name__ == "__main__":
    path, output = sys.argv[1], sys.argv[2]
    total, sites, kinds = census(path)
    rows = [
        dict(
            site=site,
            instructions=count,
            shared_body="jit(evaluate)" in site,
            depth=site.count("while/body"),
            kinds=dict(kinds[site]),
        )
        for site, count in sites.most_common()
    ]
    mapped = sum(sites.values())
    shared = sum(r["instructions"] for r in rows if r["shared_body"])
    receipt = dict(
        hlo=path,
        optimized_instructions=total,
        map_site_count=len(rows),
        map_instructions=mapped,
        shared_body_sites=sum(r["shared_body"] for r in rows),
        shared_body_instructions=shared,
        direct_map_instructions=mapped - shared,
        rows=rows,
    )
    open(output, "w").write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}))
    for r in rows:
        print(
            r["instructions"],
            r["depth"],
            r["shared_body"],
            r["kinds"],
            r["site"][-110:],
        )
