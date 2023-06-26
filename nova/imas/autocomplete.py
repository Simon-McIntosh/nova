"""Manage IMAS database auto-complete lists."""
import os

from nova.imas.database import Database


class AutoComplete(Database):
    """IMAS database auto-completer."""

    @staticmethod
    def _sort(_list: list[str], key=None):
        """Return interger sorted list."""
        return sorted(_list, key=int)

    @property
    def machine_list(self):
        """Return avalible machine list."""
        machine_path = os.path.join(self.home, "imasdb")
        machine_list = list(
            set([machine.lower() for machine in os.listdir(machine_path)])
        )
        return sorted(machine_list)

    @property
    def pulse_list(self):
        """Return avalible pulse list."""
        return self._sort(os.listdir(self.database_path), int)

    @property
    def run_list(self):
        """Return avalible run list."""
        pulse_path = os.path.join(self.database_path, str(self.pulse))
        return self._sort(os.listdir(pulse_path), int)

    def occurrence_list(self, name: str):
        """Return avalible occurence list."""
        files = [
            file.split("_")[-1].split(".")[0]
            for file in os.listdir(self.ids_path)
            if name in file
        ]
        files[files.index(name)] = "0"
        return self._sort(files, int)
