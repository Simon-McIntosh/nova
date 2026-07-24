from os import path
from copy import deepcopy

import numpy as np
import shapely.geometry
import shapely.algorithms

from nova.definitions import root_dir
from nova.frame.columnar import is_list_like
from nova.frame.dataframe import DataFrame
from nova.utilities.IO import pythonIO
from nova.frame.coilset import CoilSet


class MachineData(CoilSet, pythonIO):
    """
    load ITER data and geometry.

    Data_for_study_of_ITER_plasma_magnetic_c_33NHXN_v3_15.xlsx
    Models_for_calculation_of_axisymmetric_c_XBQF5H_v2_2.xlsx
    """

    def __init__(self, read_txt=False, **kwargs):
        self.read_txt = read_txt
        self.directory = path.join(root_dir, "input/ITER")
        super().__init__(**kwargs)

    @staticmethod
    def append(data, x, z, rho, dt):
        """Append attributes to data."""
        for key, value in zip(["x", "z", "rho", "dt"], [x, z, rho, dt]):
            data[key].append(value)

    @staticmethod
    def orient(frame):
        """
        Orient a frame counter-clockwise.

        Parameters
        ----------
        frame : DataFrame
            Columnar frame with x and z coordinates.

        Returns
        -------
        frame : DataFrame
            Frame reversed to a ccw ring when its x, z boundary is clockwise.

        """
        columns = list(frame.columns)
        if frame.shape[0] > 2 and "x" in columns and "z" in columns:
            x = np.asarray(frame["x"], dtype=float)
            z = np.asarray(frame["z"], dtype=float)
            ring = shapely.geometry.LinearRing(np.column_stack([x, z]))
            if not ring.is_ccw:
                return DataFrame({col: np.asarray(frame[col])[::-1] for col in columns})
        return frame

    @staticmethod
    def _row_slice(frame, key):
        """Return a frame holding the sliced rows of every column."""
        return DataFrame({col: np.asarray(frame[col])[key] for col in frame.columns})

    def _sheet_rows(self, sheetname):
        """Return the worksheet rows as a list of value tuples, cached."""
        cache = self.__dict__.setdefault("_rows_cache", {})
        if sheetname not in cache:
            cache[sheetname] = list(self.f[sheetname].iter_rows(values_only=True))
        return cache[sheetname]

    def read_sheet(self, sheetname, skiprows, usecols, columns={}, nrows=None):
        """Read a worksheet region into a columnar frame.

        ``skiprows`` rows precede the header; ``usecols`` selects columns by
        zero-based position; rows with any empty selected cell are dropped.
        """
        rows = self._sheet_rows(sheetname)
        usecols = [int(col) for col in usecols]
        head = rows[skiprows]
        header = [head[col] if col < len(head) else None for col in usecols]
        body = rows[skiprows + 1 :]
        if nrows is not None:
            body = body[:nrows]
        records = []
        for row in body:
            values = [row[col] if col < len(row) else None for col in usecols]
            try:
                values = [float(value) for value in values]
            except TypeError, ValueError:
                continue  # a non-numeric or empty cell drops the row (dropna)
            records.append(values)
        arrays = {
            name: np.array([record[position] for record in records])
            for position, name in enumerate(header)
        }
        arrays = self._rename_columns(arrays, columns)
        return self.orient(DataFrame(arrays))

    @staticmethod
    def _rename_columns(arrays, columns):
        """Canonicalise geometry headers to x / z and the caller's overrides."""
        mapping = {"R, m": "x", "Z, m": "z", **columns}
        renamed = {}
        for name, column in arrays.items():
            if name is None:  # a selected column beyond the sheet extent
                continue
            key = name
            if any(token in name for token in (".1", ".2", "eff")):
                key = key.replace(".1", "").replace(".2", "").replace("eff", "")
            renamed[mapping.get(key, key)] = column
        if "h, mm" in renamed:  # convert the mm thickness column to metres
            renamed["h, m"] = np.asarray(renamed.pop("h, mm")) * 1e3
        return renamed

    def read_model(
        self,
        name,
        sheetname,
        skiprows,
        usecols,
        nrows=None,
        dt=0.06,
        ring=False,
        rho=None,
    ):
        """Read geometric model."""
        columns = {
            "R1(m)": "x1",
            "R2(m)": "x2",
            "Z1(m)": "z1",
            "Z2(m)": "z2",
            "Ω(Ohm)": "R",
        }
        model = self.read_sheet(
            sheetname, skiprows, usecols, nrows=nrows, columns=columns
        )
        if ring:  # triangular support and divertor rail
            ring_x = np.asarray(model["x"], dtype=float)
            ring_z = np.asarray(model["z"], dtype=float)
            model = DataFrame(
                {
                    "x1": np.array([ring_x[0]]),
                    "x2": np.array([ring_x[1]]),
                    "z1": np.array([ring_z[0]]),
                    "z2": np.array([ring_z[1]]),
                    "R": np.array([rho]),
                }
            )
        x1 = np.asarray(model["x1"], dtype=float)
        x2 = np.asarray(model["x2"], dtype=float)
        z1 = np.asarray(model["z1"], dtype=float)
        z2 = np.asarray(model["z2"], dtype=float)
        resistance = np.asarray(model["R"], dtype=float)
        data = {}
        _data = {var: [] for var in ["x", "z", "rho", "dt"]}
        _xz, index = np.array([0, 0]), 0
        for i in range(model.shape[0]):
            x = np.array([x1[i], x2[i]])
            z = np.array([z1[i], z2[i]])
            x_mean = x.mean()  # mean radius
            # segment length
            dL = np.linalg.norm(np.diff(np.array([x, z]), axis=1))
            # resistivity / thickness
            rho_segment = dL * resistance[i] / (2 * np.pi * x_mean)
            xz = np.array([x[0], z[0]])
            if i == 0 or not np.equal(_xz, xz).all():
                segment_name = f"S{index}{name}"
                if name == "cryo" and index in [35, 36]:
                    dt = 0.01
                else:
                    dt = 0.06
                data[segment_name] = deepcopy(_data)
                self.append(data[segment_name], x[0], z[0], dt * rho_segment, dt)
                index += 1
            _xz = np.array([x[1], z[1]])
            self.append(data[segment_name], x[1], z[1], dt * rho_segment, dt)

        if index == 1:  # drop sector index
            data = {name: data[segment_name]}
        if name == "cryo":
            for index, level in zip([35, 36], ["U", "L"]):
                data[f"{level}CTS"] = data.pop(f"S{index}{name}")
        for frame in data:
            data[frame] = self.orient(DataFrame(data[frame]))
        self.models[name] = data

    def load_models(self, **kwargs):
        """Load models from .pk file."""
        read_txt = kwargs.get("read_txt", self.read_txt)
        filepath = path.join(self.directory, "ITER_machine_models")
        if read_txt or not path.isfile(filepath + ".pk"):
            self.read_models()
            self.save_pickle(filepath, ["models"])
        else:
            self.load_pickle(filepath)

    def _open_workbook(self, filename):
        """Return a read-only, value-only workbook and reset the row cache."""
        import openpyxl

        self._rows_cache = {}
        self.filename = filename
        return openpyxl.load_workbook(
            path.join(self.directory, filename), read_only=True, data_only=True
        )

    def read_models(self):
        """Read model set."""
        self.models = {}
        self.f = self._open_workbook(
            "Models_for_calculation_of_axisymmetric_c_XBQF5H_v2_2.xlsx"
        )
        try:
            self.read_model(
                "vvin", "Conducting structures", 10, np.arange(1, 6), nrows=100
            )
            self.read_model(
                "vvout", "Conducting structures", 112, np.arange(1, 6), nrows=100
            )
            self.read_model(
                "cryo", "Conducting structures", 215, np.arange(1, 6), nrows=249
            )
            self.read_model(
                "trs",
                "Conducting structures",
                55,
                np.arange(10, 12),
                nrows=2,
                ring=True,
                rho=0.8,
            )
            self.read_model(
                "dir",
                "Conducting structures",
                67,
                np.arange(10, 12),
                nrows=2,
                ring=True,
                rho=0.9,
            )
        finally:
            self.f.close()

    def plot_models(self):
        """Plot geometrical models."""
        import matplotlib.pyplot as plt

        plt.set_aspect(1.1)
        for i, part in enumerate(self.models):
            for segment in self.models[part]:
                data = self.models[part][segment]
                plt.plot(data["x"], data["z"], f"C{i}")
        plt.axis("equal")
        plt.axis("off")

    def load_data(self, **kwargs):
        """Load machine data."""
        read_txt = kwargs.get("read_txt", self.read_txt)
        filepath = path.join(self.directory, "ITER_machine_data")
        if read_txt or not path.isfile(filepath + ".pk"):
            self.read_data()
            self.save_pickle(filepath, ["data"])
        else:
            self.load_pickle(filepath)

    def read_data(self):
        """Read geometric data."""
        self.data = {}
        self.f = self._open_workbook(
            "Data_for_study_of_ITER_plasma_magnetic_c_33NHXN_v3_15.xlsx"
        )
        try:
            self.data["separatrix"] = self.read_sheet("Target separatrix", 7, [2, 3])

            self.data["firstwall"] = self.read_sheet("FW & Divertor", 7, [1, 2])

            self.data["divertor"] = self.read_sheet("FW & Divertor", 7, [4, 5])

            self.data["SSring"] = self.read_sheet("VV & TS & DIR", 171, [1, 2], nrows=2)

            self.data["DIR"] = self.read_sheet("VV & TS & DIR", 183, [1, 2], nrows=2)

            self.data["VVin"] = self.read_sheet("VV & TS & DIR", 10, [1, 2], nrows=135)

            self.data["VVout"] = self.read_sheet("VV & TS & DIR", 10, [4, 5])

            self.data["cryostat"] = self._row_slice(
                self.read_sheet("Cryostat & CST", 8, np.arange(1, 5)), slice(None, 14)
            )

            self.data["cryostatCTS"] = self._row_slice(
                self.read_sheet("Cryostat & CST", 8, np.arange(1, 5)), slice(13, None)
            )

            for rib in range(4):
                self.data[f"cryostatR{rib + 1}"] = self.read_sheet(
                    "Cryostat & CST", 8 + rib * 5, np.arange(7, 11), nrows=2
                )

            self.data["upperCTS"] = self.read_sheet(
                "Cryostat & CST", 8, np.arange(12, 17)
            )
        finally:
            self.f.close()

    def plot_data(self, keys=None, ax=None, legend=False, **kwargs):
        """Plot geometric data."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if keys is not None:
            if not is_list_like(keys):
                keys = [keys]
        else:
            keys = self.data.keys()
        for key in keys:
            try:
                ax.plot(self.data[key]["x"], self.data[key]["z"], label=key, **kwargs)
            except KeyError:
                raise KeyError(key, self.data[key].columns)
        ax.axis("equal")
        ax.axis("off")
        if legend:
            ax.legend()

    def select_coilset(self, part_list=None):
        if not hasattr(self, "models"):
            self.load_models()
        if part_list is None:
            part_list = self.models.keys()
        if not is_list_like(part_list):
            part_list = part_list.replace("_", " ")
            part_list = part_list.split()
        part_list = [pl for pl in part_list if pl in self.models]
        part_list = list(np.unique(np.sort(part_list)))
        return part_list

    def load_coilset(self, **kwargs):
        read_txt = kwargs.get("read_txt", self.read_txt)
        filename = "ITER_coilset_"
        part_list = self.select_coilset(kwargs.get("part_list", None))
        filename += "_".join(part_list)
        filepath = path.join(self.directory, filename)
        if read_txt or not path.isfile(filepath + ".pk"):
            self.build_coilset(part_list)
            self.save_coilset(filename, directory=self.directory)
        else:
            CoilSet.load_coilset(self, filename, directory=self.directory)
        return self.coilset

    def build_coilset(self, part_list):
        for part in part_list:
            for segment in self.models[part]:
                frame = self.models[part][segment]
                self.shell.insert(
                    frame.x,
                    frame.z,
                    self.dshell,
                    frame.dt,
                    rho=frame.rho,
                    part=part,
                    name=segment,
                )


if __name__ == "__main__":
    machine = MachineData(dcoil=0.2, dshell=0, read_txt=False)
