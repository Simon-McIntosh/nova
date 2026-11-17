"""Verify adapter discovery includes documentation and classifies every call."""

from pathlib import Path
import tempfile

from scripts.prototypes.diiid_adapter_current_refusal import caller_sweep as sweep


def main():
    record = sweep._inventory_record()
    corpus, _synthetic = sweep._inventory()
    documentation = (
        "docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py",
        "_diiid_state",
    )
    assert documentation in corpus, "MISSING corpus documentation caller"
    assert len(corpus) == 13
    assert len(record["call_sites"]) == len(record["counted"]) + len(record["excluded"])
    assert all(site["reason"] for site in record["excluded"])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = (
            "from example import complete_profile_current_adapter as adapt\n"
            "async def prepare():\n    return adapt(profile)\n"
        )
        for folder in (
            "docs/figures",
            "hidden/tools",
            ".cache/tools",
            ".venv",
            "worktrees",
        ):
            path = root / folder / "caller.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fixture)
        found = sweep._all_call_sites(root)
        assert {site["path"] for site in found} == {
            "docs/figures/caller.py",
            "hidden/tools/caller.py",
            ".cache/tools/caller.py",
        }
        assert all(
            site["function"] == "prepare" and site["line"] == 3 for site in found
        )
    # A narrowed discovery instrument must fail its documentation control.
    narrowed = {
        item for item in corpus if item[0].startswith(("benchmarks/", "scripts/"))
    }
    assert documentation not in narrowed
    print(
        "INVENTORY_PROBE: 13 corpus callers; documentation/alias/async/hidden-path "
        "controls passed; narrowed roots miss documentation"
    )


if __name__ == "__main__":
    main()
