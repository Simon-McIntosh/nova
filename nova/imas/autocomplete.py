"""Manage IMAS database auto-complete lists."""
import os
import string

from nova.imas.database import IDS


class AutoComplete(IDS):
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

    @staticmethod
    def _listdir(path):
        """Return valid path list."""
        try:
            return os.listdir(path)
        except NotADirectoryError:
            return []

    def _isids(self, pulse_path, run):
        """Return named IDS status at pulse_path/run by checking for {name}.* entry."""
        ids_path = os.path.join(pulse_path, run)
        if self.name is None:
            return os.path.isdir(ids_path)
        return self.name in [name.split(".")[0] for name in self._listdir(ids_path)]

    def _filter(self, pulse_path):
        """Return filtered run list es."""
        if self.name is None:
            return os.listdir(pulse_path)
        return [
            run for run in self._listdir(pulse_path) if self._isids(pulse_path, run)
        ]

    @property
    def pulse_list(self):
        """Return avalible pulse list."""
        if not os.path.isdir(self.database_path):
            return []
        pulse_list = []
        for pulse in os.listdir(self.database_path):
            pulse_path = os.path.join(self.database_path, pulse)
            if self._filter(pulse_path):
                pulse_list.append(pulse)
        return self._sort(pulse_list, int)

    @property
    def run_list(self):
        """Return avalible run list."""
        pulse_path = os.path.join(self.database_path, str(self.pulse))
        if not os.path.isdir(pulse_path):
            return []
        return self._sort(self._filter(pulse_path), int)

    @property
    def occurrence_list(self):
        """Return avalible occurence list."""
        if not os.path.isdir(self.ids_path):
            return []
        files = os.listdir(self.ids_path)
        occurrence_list = [
            file.split(".")[0].lstrip(string.ascii_letters + "_") for file in files
        ]
        if self.name is not None:
            occurrence_list = [
                occurrence
                for occurrence, file in zip(occurrence_list, files)
                if self.name in file
            ]
        else:
            occurrence_list = list(set(occurrence_list))
        occurrence_list[occurrence_list.index("")] = "0"
        return self._sort(occurrence_list, int)


if __name__ == "__main__":
    attrs = {
        "pulse": 111001,
        "run": 202,
        "machine": "iter_md",
        "occurrence": 0,
        "user": "public",
        "name": "pf_active",
        "backend": "hdf5",
    }

    complete = AutoComplete(**attrs | {"name": "pf_active"})
    print(complete.pulse_list)
