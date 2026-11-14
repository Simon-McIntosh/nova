"""Reuse the slice profiler, retaining a metadata-free lowered program digest."""

from pathlib import Path


def main():
    driver = Path(__file__).parents[1] / "bank_h200_slice_profile/measure.py"
    source = driver.read_text()
    marker = "compile_seconds=perf_counter() - started,"
    assert source.count(marker) == 1
    source = source.replace(
        marker,
        marker
        + """
            lowered_sha256=hashlib.sha256(
                lowered.compiler_ir().operation.get_asm(enable_debug_info=False).encode()
            ).hexdigest(),""",
    )
    scope = dict(__file__=str(driver), __name__="__main__")
    exec(compile(source, str(driver), "exec"), scope)


if __name__ == "__main__":
    main()
