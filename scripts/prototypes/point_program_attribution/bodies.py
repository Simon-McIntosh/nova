"""Count the repeated bodies in one lowered StableHLO module.

The module is parsed function by function, with value names scoped to their
function, so two functions that both use ``%0`` are not merged.

Three counts answer three different questions about the same module.

*calls*: how many times each outlined function body is instantiated, read from
the module's own ``call`` operations.  This is the count of repeated function
bodies.

*exact*: an operation's structural hash covers its opcode, its attributes and
the hash of every operation defining its operands, so two operations match only
when they compute the same expression from the same inputs.

*shape*: an operation's label at depth ``d`` covers its opcode and attributes
with operand identities removed, plus its operands' labels at depth ``d - 1``.
Two operations share a shape label when they are structurally identical bodies
whose leaves are merely different inputs, which is what says whether a per-term
body is emitted once or once per term.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

DEFINITION = re.compile(r"^\s*%([\w.$-]+) = (.*)$")
OPERAND = re.compile(r"%([\w.$-]+)")
LITERAL = re.compile(r"dense<[^>]*>|tensor<[^>]*>")
FUNCTION = re.compile(r"^\s*func\.func\s+(?:public\s+|private\s+)?@([\w.$-]+)\(")
CALL = re.compile(r"call @([\w.$-]+)\(")
FAMILY = re.compile(r"_\d+$")
DEPTHS = (2, 4, 6, 8, 10, 12, 16, 24)


def functions(text):
    """Yield (name, lines) for every function body in the module text."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = FUNCTION.match(lines[index])
        if match is None:
            index += 1
            continue
        start = index
        depth = 0
        while index < len(lines):
            depth += lines[index].count("{") - lines[index].count("}")
            index += 1
            if depth == 0:
                break
        yield match.group(1), lines[start:index]


def values(span):
    """Return ({value: body}, {value: [operands]}) with names scoped by caller."""
    bodies, operands = {}, {}
    for line in span:
        match = DEFINITION.match(line.rstrip())
        if match is None:
            continue
        name, body = match.groups()
        bodies[name] = body
        operands[name] = OPERAND.findall(body)
    return bodies, operands


def shape_label(body):
    return body.split(" ", 1)[0].split("(", 1)[0] + "|" + LITERAL.sub(
        "", OPERAND.sub("%", body).strip()
    )


def scoped(bodies, operands, prefix):
    return (
        {f"{prefix}::{name}": body for name, body in bodies.items()},
        {
            f"{prefix}::{name}": [f"{prefix}::{operand}" for operand in items]
            for name, items in operands.items()
        },
    )


def exact_hashes(bodies, operands, prefix):
    """Return {value: hash} and {value: subtree operation count}."""
    hashes, sizes = {}, {}
    hashes_local, sizes_local = {}, {}
    for name in bodies:
        stack, seen = [name], set()
        while stack:
            current = stack.pop()
            if current in seen or current not in bodies:
                continue
            seen.add(current)
            stack.extend(operands[current])
        sizes_local[name] = len(seen)
    for name in bodies:
        fingerprints = []
        for operand in operands[name]:
            if operand in bodies:
                fingerprints.append(hashes_local[operand])
            else:
                fingerprints.append("arg:" + operand)
        body = OPERAND.sub(
            lambda match: "<" + hashes_local.get(match.group(1), match.group(1)) + ">",
            bodies[name],
        )
        digest = hashlib.sha1((body + "|".join(fingerprints)).encode()).hexdigest()
        hashes_local[name] = digest
        hashes[f"{prefix}::{name}"] = digest
        sizes[f"{prefix}::{name}"] = sizes_local[name]
    return hashes, sizes


def parse_module(path):
    text = path.read_text()
    scopes = {}
    for name, span in functions(text):
        bodies, operands = values(span)
        scopes[name] = (bodies, operands)
    return text, scopes


def count_calls(text):
    called = Counter(CALL.findall(text))
    families = Counter()
    for name, sites in called.items():
        families[FAMILY.sub("", name)] += sites
    return {
        "call_sites": sum(called.values()),
        "bodies_called_once": sum(1 for sites in called.values() if sites == 1),
        "bodies_called_more_than_once": sum(1 for sites in called.values() if sites > 1),
        "most_call_sites_of_one_body": max(called.values(), default=0),
        "calls_by_family": families.most_common(12),
        "top_called_bodies": [
            {"body": name, "call_sites": sites} for name, sites in called.most_common(12)
        ],
    }


def count_function_bodies(scopes):
    classes = Counter()
    sizes = {}
    for name, (bodies, _) in scopes.items():
        fingerprint = "\n".join(
            OPERAND.sub("%", body) for body in bodies.values()
        )
        digest = hashlib.sha1(fingerprint.encode()).hexdigest()
        classes[digest] += 1
        sizes[digest] = len(bodies)
    return {
        "functions": len(scopes),
        "distinct_bodies_after_operand_erasure": len(classes),
        "bodies_emitted_more_than_once": sum(1 for count in classes.values() if count > 1),
        "most_emitted_one_body": max(classes.values(), default=0),
        "largest_body_operations": max(sizes.values(), default=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-size", type=int, default=50)
    args = parser.parse_args()
    receipts = []
    for path in args.module:
        text, scopes = parse_module(path)
        exact = {}
        sizes = {}
        for name, (bodies, _) in scopes.items():
            hashes, local = exact_hashes(*scopes[name], name)
            exact.update(hashes)
            sizes.update(local)
        receipts.append(
            {
                "module": str(path),
                "operations": sum(len(bodies) for bodies, _ in scopes.values()),
                "calls": count_calls(text),
                "function_bodies": count_function_bodies(scopes),
                "exact_expression_classes": len(set(exact.values())),
                "exact_expressions_repeated": sum(
                    1 for _, count in Counter(exact.values()).items() if count > 1
                ),
                "minimum_body_ranked_size": args.minimum_size,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipts, indent=2, default=str) + "\n")
    print(json.dumps(receipts, indent=2, default=str))


if __name__ == "__main__":
    main()