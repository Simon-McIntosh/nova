"""Reuse the trace process driver with a saved common fixture input."""

from pathlib import Path

from scripts.prototypes.bank_solve_trace_head import device


def main():
    execute = device.execute

    def capture_common(tree, revision, command, log, environment):
        command = [
            argument.replace(
                "from scripts.prototypes.bank_solve_trace.capture import main",
                "from scripts.prototypes.bank_solve_trace_four_read.capture "
                "import main",
            )
            for argument in command
        ]
        environment = dict(
            environment, NOVA_COMMON_SEEDS=str(log.parent / "common-seeds.npz")
        )
        if log.name.startswith("capture-"):
            destination = Path(command[-1])
            assert not list(destination.iterdir()), destination
        execute(tree, revision, command, log, environment)

    device.execute = capture_common
    device.main()


if __name__ == "__main__":
    main()
