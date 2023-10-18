import pytest

from nova.imas.coil import full_coil_name


@pytest.mark.parametrize(
    "identifier,full_name",
    [
        ("TCC_1-4", "Top Correction Coils, TCC-1 and TCC-4"),
        ("MCC_2-6", "Middle Correction Coils, MCC-2 and MCC-6"),
        ("BCC_9-2", "Bottom Correction Coils, BCC-9 and BCC-2"),
        ("CS1U", "Central Solenoid Module 1 Upper"),
        ("CS3L", "Central Solenoid Module 3 Lower"),
    ],
)
def test_full_coil_name(identifier, full_name):
    assert full_coil_name(identifier) == full_name


def test_full_coil_name_bad_id():
    with pytest.raises(NotImplementedError):
        full_coil_name("not_and_id")


if __name__ == "__main__":
    pytest.main([__file__])
