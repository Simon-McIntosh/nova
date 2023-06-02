import numpy as np
from nova.plot_coilset import plot_coilset
import pandas as pd


class coil_class(dict):
    def __init__(self):
        super().__init__()
        self.set_attributes()
        self.initalise()
        self.pcs = plot_coilset(self)

    def set_attributes(self):
        self.attributes = {}
        self.attributes["required"] = ["x", "z", "dx", "dz"]
        self.attributes["optional"] = [
            "m",
            "R",
            "index",
            "Nt",
            "Nf",
            "tf",
            "material",
            "cross_section",
        ]
        self.attributes["default"] = [
            ("It", 0),
            ("R", 0),
            ("Nt", 1),
            ("Nf", 1),
            ("cross_section", "square"),
        ]
        self.attributes["inital"] = self.attributes["required"] + [
            key[0] for key in self.attributes["default"]
        ]

    def initalise(self):
        self["nC"] = 0
        self["index"] = {}
        self["coil"] = pd.DataFrame(columns=self.attributes["inital"])
        self["subcoil"] = pd.DataFrame()
        self["plasma"] = pd.DataFrame()
        self.coil = {}

    def add_coilset(self, coilset, label=None):
        nCo = self["nC"]
        for name in coilset["coil"]:
            x, z = coilset["coil"][name]["x"], coilset["coil"][name]["z"]
            dx, dz = coilset["coil"][name]["dx"], coilset["coil"][name]["dz"]
            It = coilset["coil"][name]["It"]
            oparg = {}  # optional keys
            for key in self.attributes["optional"]:
                if key in coilset["coil"][name]:
                    oparg[key] = coilset["coil"][name][key]
            self.add_coil(x, z, dx, dz, It, name=name, **oparg)
            if coilset["subcoil"]:
                Nf = coilset["coil"][name]["Nf"]
                for i in range(Nf):
                    subname = name + "_{}".format(i)
                    self["subcoil"][subname] = coilset["subcoil"][subname]
            else:
                self["subcoil"][name] = {"Nf": 1}
                self["subcoil"][name + "_0"] = coilset["coil"][name]
        if coilset["plasma"]:
            self.update_plasma(coilset)
        self.inherit_index(coilset, offset=nCo)
        if label:
            self["index"][label] = {
                "name": list(coilset["coil"].keys()),
                "index": np.arange(nCo, self["nC"]),
                "n": len(coilset["coil"]),
            }

    def add_coil(self, **kwargs):
        name = kwargs.get("name", f'Coil{self["nC"]:d}')
        self["nC"] += 1
        for key in self.attributes["required"]:
            if key in kwargs:
                self["coil"].loc[name, key] = kwargs.get(key)
            else:
                raise IndexError("required key: {key} not set in add_coil")
        rc = kwargs.get("rc", np.sqrt(kwargs["dx"] ** 2 + kwargs["dz"] ** 2) / 2)
        self["coil"].loc[name, "rc"] = rc
        for key in self.attributes["optional"]:  # optional keys
            if key in kwargs:
                self["coil"].loc[name, key] = kwargs.get(key)
        for keypair in self.attributes["default"]:  # set default values
            if pd.isnull(self["coil"].loc[name, keypair[0]]):
                self["coil"].loc[name, keypair[0]] = keypair[1]
        self["coil"] = self["coil"].astype({"Nf": int})

    def mesh_coils(self, dCoil=None):
        if dCoil is None:  # dCoil not set, use stored value
            if not hasattr(self, "dCoil"):
                self.dCoil = 0
        else:
            self.dCoil = dCoil
        self["subcoil"] = pd.DataFrame()
        if self.dCoil == 0:
            for name in self["coil"].index:
                subname = name + "_0"
                self["subcoil"].loc[subname] = self["coil"][name]
        else:
            for name in self["coil"].index:
                coil = self["coil"].loc[name]
                subcoil = self.mesh_coil(coil, self.dCoil, name)
                self["coil"].loc[name, "Nf"] = len(subcoil)
                self["subcoil"] = self["subcoil"].append(subcoil)

    @staticmethod
    def mesh_coil(coil, dCoil=None, name=None):
        xc, zc = coil["x"], coil["z"]
        Dx, Dz = abs(coil["dx"]), abs(coil["dz"])
        if dCoil is None:
            dCoil = np.max([Dx, Dz])
        elif dCoil == -1:  # mesh per-turn (inductance calculation)
            Nt = coil["Nt"]
            dCoil = (Dx * Dz / Nt) ** 0.5
        nx = int(np.ceil(Dx / dCoil))
        nz = int(np.ceil(Dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx, dz = Dx / nx, Dz / nz
        rc = np.sqrt(dx**2 + dz**2) / 4
        x = xc + np.linspace(dx / 2, Dx - dx / 2, nx) - Dx / 2
        z = zc + np.linspace(dz / 2, Dz - dz / 2, nz) - Dz / 2
        X, Z = np.meshgrid(x, z, indexing="ij")
        X, Z = np.reshape(X, (-1, 1))[:, 0], np.reshape(Z, (-1, 1))[:, 0]
        Nf = len(X)  # filament number
        If = coil["It"] / Nf
        cross_section = coil["cross_section"]
        if name is not None:
            index = [f"{name}_{i}" for i in range(Nf)]
        else:
            index = range(Nf)
        subcoil = pd.DataFrame(
            index=index, columns=["x", "z", "dx", "dz", "If", "rc", "cross_section"]
        )
        for i, (x, z) in enumerate(zip(X, Z)):
            subcoil.iloc[i] = [x, z, dx, dz, If, rc, cross_section]
        return subcoil

    def append_index(self, label, name, index=None):
        if index is None:
            index = self["nC"] - 1
        if hasattr(index, "__iter__"):
            n = len(index)
        else:
            n = 1
        if label in self["index"]:
            self["index"][label]["name"].append(name)
            self["index"][label]["index"].append(index)
            self["index"][label]["n"] += n
        else:
            if not isinstance(name, (list, tuple, np.ndarray)):
                name = [name]
            if not isinstance(index, (list, tuple, np.ndarray)):
                index = [index]
            self["index"][label] = {"name": name, "index": index, "n": n}

    def inherit_index(self, coilset, offset=0):
        if "index" in coilset:
            for label in coilset["index"]:
                if "n" in coilset["index"][label]:
                    if coilset["index"][label]["n"] > 0:
                        name = coilset["index"][label]["name"]
                        index = np.array(coilset["index"][label]["index"])
                        index += offset
                        self.append_index(label, name, index)

    def index_coilset(self):  # append coil-subcoil map in index
        subcoilset = list(self["subcoil"].keys())
        Nf = [self["coil"][name]["Nf"] for name in self["coil"]]
        for i, name in enumerate(self["coil"]):
            N = [
                np.sum(Nf[slice(None, i)], dtype=int),
                np.sum(Nf[slice(None, i + 1)], dtype=int),
            ]
            self["index"][name] = {
                "index": slice(*N),
                "n": Nf[i],
                "name": subcoilset[slice(*N)],
            }

    # def extract(self, **kwargs):

    def plot(self, **kwargs):
        self.pcs.plot(**kwargs)


if __name__ == "__main__":
    print("\nsee nova.coil_geom for usage examples")
