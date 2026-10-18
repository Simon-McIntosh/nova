"""One case identity across trace names and adapter receipts."""


def case_key(name: str) -> str:
    return name.lower().replace(":", "-").replace("_", "-")
