"""Build ITER coilset."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from nova.frame.coilset import CoilSet
from nova.frame.dataframe import DataFrame
from nova.frame.framedata import FrameData
from nova.frame.machinedata import MachineData
from nova.frame.turn import Turn


@dataclass
class VS3(FrameData):
    centroid: list[float, float] = field(
        default_factory=lambda: [0, 0]
    )  # coil geometric centroid
    theta: float = 0  # inclination angle
    width: float = 0.068  # winding pack width
    height: float = 0.064  # widning pack height
    degrees: bool = True

    def __post_init__(self):
        """Create turn center pattern."""
        self.pattern = np.zeros((4, 3))  # turn pattern
        for i, (ix, iz) in enumerate(zip([1, 1, -1, -1], [-1, 1, 1, -1])):
            self.pattern[i, 0] = ix * self.width / 2
            self.pattern[i, 2] = iz * self.height / 2
        rotation = Rotation.from_euler("y", self.theta, degrees=self.degrees)
        self.pattern = rotation.apply(self.pattern)
        self.pattern[:, 0] += self.centroid[0]
        self.pattern[:, 2] += self.centroid[1]
        self.turn = Turn(*self.frames)

    def insert(self, radius: list[float, float], **kwargs):
        """
        Insert turns to frame and subframe.

        Parameters
        ----------
        radius : list[float, float]
            Turn radius [inner, outer].

        label : str
            coil label

        Returns
        -------
        None.

        """
        turn_fraction = 1 - radius[0] / radius[1]
        self.turn.insert(
            self.pattern[:, 0], self.pattern[:, 2], radius[1], turn_fraction, **kwargs
        )

    """
    def insert_vs3_coils(self):
        co = 0.1065  # inner
        c1 = 0.14451  # outer
        rcs = np.array([co, c1]) / (2 * np.pi)
        acs_turn = np.pi * (rcs[1]**2 - rcs[0]**2)  # single turn cross-section
        d = 2 * rcs[1]  # turn diameter
        dt = 0.3  # skin_fraction
        dx_wp = 0.068  # winding pack width
        dz_wp = 0.064  # widning pack height
        self.geom = {}
        self.geom['LVS'] = {'x': 7.504, 'z': -2.495, 'dx': dx_wp, 'dz': dz_wp,
                            'theta': -37.8*np.pi/180, 'sign': 1,
                            'nturn': 4}
        self.geom['UVS'] = {'x': 5.81, 'z': 4.904, 'dx': dx_wp, 'dz': dz_wp,
                            'theta': 25.9*np.pi/180 + np.pi, 'sign': -1,
                            'nturn': -4}
        if not invessel:  # offset coils
            self.geom['UVS']['x'] += 1.7
            self.geom['UVS']['z'] -= 0
            self.geom['LVS']['x'] = self.geom['UVS']['x']
            self.geom['LVS']['z'] += -2.4

        xp = np.zeros((4, 2))  # coil pattern
        for i, (ix, iz) in enumerate(zip([1, 1, -1, -1], [-1, 1, 1, -1])):
            xp[i, 0] = ix*dx_wp/2
            xp[i, 1] = iz*dz_wp/2
        self.xc = {}  # coil centers
        for name in self.geom:  # add subcoils
            xc = np.dot(xp, rotate2D(self.geom[name]['theta'])[0])
            xc[:, 0] += self.geom[name]['x']
            xc[:, 1] += self.geom[name]['z']
            self.xc[name] = xc
            for i, x in enumerate(xc):
                subname = f'{name}{i}'
                self.coil.insert(
                    x[0], x[1], d, dt, delta=0, name=subname,
                    section='skin', turn='skin',
                    material='copper', part='VS3',
                    nturn=1)
                #R = resistivity_cu * 2 * np.pi * x[0] / acs_turn
                #m = density_cu * 2 * np.pi * x[0] * acs_turn
                #self.Loc[subname, 'R'] = R
                #self.Loc[subname, 'm'] = m

    def add_jacket(self, rcs=[0.0265, 0.0295]):
        acs_turn = np.pi * (rcs[1]**2 - rcs[0]**2)  # single turn cross-section
        d = 2*rcs[1]
        dt = 0.15  # turn_fraction
        Nf = 4
        for name in self.geom:
            for isub in range(Nf):
                subname = name+'{}'.format(isub)
                x_sub = self.subframe.at[subname, 'x']
                z_sub = self.subframe.at[subname, 'z']
                R = resistivity_ss * 2 * np.pi * x_sub / acs_turn
                m = density_ss * 2 * np.pi * x_sub * acs_turn
                jacket_name = f'{name}j{isub}'
                self.coil.insert(x_sub, z_sub, d, dt, name=jacket_name,
                                 section='skin', turn='skin',
                                 material='steel', delta=0, part='VS3j',
                                 active=False)
                #  R=R, m=m
        """


@dataclass
class ITERgeom(CoilSet):
    """Manage ITER geometry."""

    def __post_init__(self):
        """Insert default source."""
        super().__post_init__()
        self.metadata = {"source": "PCR"}
        # self.insert_poloidal_field_coils()
        # self.build_firstwall()

        for centroid, theta, name in zip(
            [[7.504, -2.495], [5.81, 4.904]], [-37.8, 25.9], ["L", "U"]
        ):
            vs3 = VS3(*self.frames, centroid=centroid, theta=theta)
            vs3.insert([0.01695, 0.02300], name=f"VS3{name}", part="vs3")
            vs3.insert(
                [0.0265, 0.0295],
                name=f"VS3{name}j",
                part="vs3j",
                active=False,
                link=True,
            )
        self.linkframe(["VS3L", "VS3U"], -1)

    def build_firstwall(self):
        machine = MachineData(dcoil=self.dcoil, read_txt=False)
        self.machine = machine

    def insert_poloidal_field_coils(self):
        """Insert poloidal field coils."""
        names, columns = self._parse_coil_table(self._poloidal_field_coils())
        part = ["cs" if "CS" in name else "pf" for name in names]
        data = DataFrame(columns, index=names)
        self.coil.insert(data, part=part, turn="hex")
        self.linkframe(["CS1U", "CS1L"])

    @staticmethod
    def _parse_coil_table(table):
        """Return coil names and named columns from a tab-delimited table.

        The header row labels seven geometry columns; each following row is a
        coil name followed by its values. Header labels collapse to their short
        form (``X, m`` -> ``x``) with resistance and turn count named
        explicitly and the section extents renamed to ``dl`` / ``dt``.
        """
        lines = [line.strip() for line in table.splitlines() if line.strip()]
        header = [field.strip() for field in lines[0].split("\t")]
        rename = {label: label.split(",")[0].lower() for label in header}
        rename["R, ohm"] = "R"
        rename["N,"] = "nturn"
        rename = {
            label: {"dx": "dl", "dz": "dt"}.get(name, name)
            for label, name in rename.items()
        }
        names, rows = [], []
        for line in lines[1:]:
            name, *values = [field.strip() for field in line.split("\t")]
            names.append(name)
            rows.append([float(value) for value in values])
        table = np.array(rows, dtype=float)
        columns = {rename[label]: table[:, i] for i, label in enumerate(header)}
        return names, columns

    def _poloidal_field_coils(self):
        """Return poloidal field coil geometrical data."""
        if self.metadata["source"] == "PCR":  # update, post 2012
            return """
                X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.6870	5.4640	0.7400	2.093	554	0.102	9.0e3
                CS2U	1.6870	3.2780	0.7400	2.093 	554	0.113	10.0e3
                CS1U	1.6870	1.0920	0.7400	2.093 	554	0.124	11.0e3
                CS1L	1.6870	-1.0720	0.7400	2.093 	554	0.124	11.0e3
                CS2L	1.6870	-3.2580	0.7400	2.093 	554	0.113	10.0e3
                CS3L	1.6870	-5.4440	0.7400	2.093 	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                """
        if self.metadata["source"] == "baseline":  # old
            return """
                X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.722	5.313	0.719	2.075	554	0.102	9.0e3
                CS2U	1.722	3.188	0.719	2.075	554	0.113	10.0e3
                CS1U	1.722	1.063	0.719	2.075	554	0.124	11.0e3
                CS1L	1.722	-1.063	0.719	2.075	554	0.124	11.0e3
                CS2L	1.722	-3.188	0.719	2.075	554	0.113	10.0e3
                CS3L	1.722	-5.313	0.719	2.075	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                """
        raise IndexError(f"source {self.metadata['source']} not found")


if __name__ == "__main__":
    coilset = ITERgeom(dcoil=0.25, dplasma=-150)
    coilset.plot()
