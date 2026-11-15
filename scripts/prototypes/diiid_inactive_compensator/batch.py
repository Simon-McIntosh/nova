"""Run the four booked-source rows in one allocation, retaining each receipt."""

from pathlib import Path

from scripts.prototypes.diiid_finite_edge_booking import batch as booked

booked.extent.HERE = Path(__file__).resolve().parent
completed = booked.extent.completed
booked.extent.completed = lambda folder: (
    completed(folder) and (folder / "trips.json").is_file()
)

if __name__ == "__main__":
    booked.extent.main()
