"""Attribute expanded JAX equations to the read's kernel call sites."""

from collections import Counter


def attribute_graph(graph):
    """Count call sites separately from loop iterations and shared bodies.

    A packed kernel is inlined into value_gradient, so that named boundary
    identifies each invocation. Source tracebacks classify each invocation's
    enclosed equations, including descendants of nested compiled functions.
    Nested construct totals overlap and are reported separately from the
    disjoint kernel-call partition.
    """
    calls = []
    hessians = []
    primitives = Counter()
    constructs = Counter()

    def trace_names(equation):
        traceback = equation.source_info.traceback
        if traceback is None:
            return []
        codes, _ = traceback.raw_frames()
        return [code.co_name for code in codes]

    def walk(value, path, stage=None):
        if hasattr(value, "jaxpr"):
            return walk(value.jaxpr, path, stage)
        if isinstance(value, dict):
            return sum(
                walk(item, path + "/" + key, stage) for key, item in value.items()
            )
        if isinstance(value, tuple | list):
            return sum(
                walk(item, path + f"/{i}", stage) for i, item in enumerate(value)
            )
        if not hasattr(value, "eqns"):
            return 0
        total = len(value.eqns)
        for index, equation in enumerate(value.eqns):
            name = str(equation.params.get("name", equation.primitive.name))
            location = path + f"/{index}:{name}"
            names = trace_names(equation)
            local_stage = stage
            for candidate in (
                "_sampled_flux",
                "_null_census",
                "_curvature_derivatives",
                "_wall_events",
            ):
                if stage is None and candidate in names:
                    local_stage = candidate
            primitives[equation.primitive.name] += 1
            nested = walk(equation.params, location, local_stage)
            total += nested
            if nested:
                constructs[name] += nested
            row = {
                "path": location,
                "stage": local_stage,
                "body_expanded_equations": nested,
                "call_with_body_equations": 1 + nested,
                "source_functions": names,
            }
            if name == "value_gradient":
                calls.append(row)
            if name == "evaluate" and "gradient_with_value" not in names:
                hessians.append(row)
        return total

    total = walk(graph, "read")
    for row in calls + hessians:
        row["body_share"] = row["body_expanded_equations"] / total
    return {
        "expanded_equations": total,
        "packed_kernel_invocations": len(calls),
        "hessian_invocations": len(hessians),
        "packed_kernel_call_sites": calls,
        "hessian_call_sites": hessians,
        "kernel_call_partition_equations": sum(
            row["call_with_body_equations"] for row in calls
        ),
        "top_nested_constructs_overlapping": constructs.most_common(12),
        "top_primitives_disjoint": primitives.most_common(12),
    }
