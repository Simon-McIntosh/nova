# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:27:36 2020
"""


class SimulationData:
    """
    container for simulation data

        target (dict): poloidal target coordinates and data
            target['targets'] (DataFrame):  target xz-coordinates
            target['Psi'] (DataFrame): poloidal flux
            target['Bx'] (DataFrame): radial field
            target['Bz'] (DataFrame): vertical field
            target['update'] (bool): update flag

        interaction (dict): coil grid / target interaction matrices (DataFrame)
            interaction['Psi']: poloidal flux interaction matrix
            interaction['Bx']: radial field interaction matrix
            interaction['Bz']: vertical field interaction matrix
    """

    # main class attributes
    _simulation_attributes = ["target", "grid", "interaction"]

    def __init__(self, target=None, grid=None, interaction=None, **kwargs):
        self._attributes += self._simulation_attributes
        self.target = self._initialize_target(target)
        self.grid = self._initialize_grid(grid, **kwargs)
        self.interaction = self._initialize_interaction(interaction)

    @staticmethod
    def _initialize_interaction(interaction=None):
        if interaction is None:
            interaction = {"Psi": DataFrame(), "Bx": DataFrame(), "Bz": DataFrame()}
        return interaction


class Rectangle(Vectors):
    def __init__(self, points):
        Vectors.__init__(self, points)

    def B(self, phi):
        return np.sqrt(self.rs**2 + self.r**2 - 2 * self.r * self.rs * np.cos(phi))

    def D(self, phi):
        return np.sqrt(self.gamma**2 + self.B(phi) ** 2)

    def G(self, phi):
        return np.sqrt(self.gamma**2 + self.r**2 * np.sin(phi) ** 2)

    def b1(self, phi):
        "beta 1"
        return (self.rs - self.r * np.cos(phi)) / self.G(phi)

    def b2(self, phi):
        "beta 2"
        return self.gamma / self.B(phi)

    def b3(self, phi):
        "beta 3"
        return (
            self.gamma
            * (self.rs - self.r * np.cos(phi))
            / (self.r * np.sin(phi) * self.D(phi))
        )

    def Jf(self, phi):
        "compute J intergrand"
        f = np.zeros(np.shape(phi))
        for i in range(f.shape[1]):
            f[:, i] = np.arcsinh(self.b1(phi[:, i]))
        return f

    def J(self, alpha, index=2):
        scheme = quadpy.line_segment.gauss_patterson(index)
        bounds = np.dot(
            np.array([[self.phi(0)], [self.phi(alpha)]]), np.ones((1, self.nI))
        )
        return scheme.integrate(self.Jf, bounds)

    def Cphi(self, alpha):
        return (
            0.5
            * self.gamma
            * self.a
            * (1 - self.k2 * np.sin(alpha) ** 2) ** 0.5
            * -np.sin(2 * alpha)
            - 1
            / 6
            * np.arcsinh(self.b2(alpha))
            * np.sin(2 * alpha)
            * (2 * self.r**2 * np.sin(2 * alpha) ** 2 + 3 * (self.rs**2 - self.r**2))
            - 1
            / 4
            * self.gamma
            * self.r
            * np.arcsinh(self.b1(alpha))
            * -np.sin(4 * alpha)
            - 1 / 3 * self.r**2 * np.arctan(self.b3(alpha))
            - np.cos(2 * alpha) ** 3
        )

    def flux(self):
        "calculate flux for rectangular coil section"
        Aphi = (
            self.Cphi(np.pi / 2)
            + self.gamma * self.r * self.J(np.pi / 2)
            + self.gamma
            * self.a
            / (6 * self.r)
            * (self.U * self.K - 2 * self.rs * self.E)
        )
        for p in range(3):
            Aphi += self.gamma / (6 * self.a * self.r)
        return np.zeros(len(self.r))

    """
    @property
    def U(self):
        if self._U is None:
            self._U = self.k2 * (4*self.gamma**2 + 3*self.rs**2 -
                                 5*self.r**2) / (4*self.r)
    """


def _extract_data(self, frame):
    data = {}
    for key in ["x", "z", "dx", "dz", "Nt"]:
        data[key] = getattr(frame, key)
    data["ro"] = self.gmr.calculate_self(data["dx"], data["dz"], frame.cross_section)
    return data


"""

# structured array
#fields; x, z, rms, turn_section,

"""


def assemble_source(self):
    self.nT = self.target.nC  # target number
    data = self._extract_data(self.source)
    self.source_m = {}
    for key in data:
        self.source_m[key] = np.dot(np.ones((self.nT, 1)), data[key].reshape(1, -1))


def assemble_target(self):
    self.nS = self.source.nC  # source filament number
    data = self._extract_data(self.target)
    self.target_m = {}
    for key in data:
        self.target_m[key] = np.dot(data[key].reshape(-1, 1), np.ones((1, self.nS)))


def assemble(self):
    self.assemble_source()
    self.assemble_target()
    # self.offset()  # transform turn-trun offset to geometric mean


def offset(self):
    "transform turn-trun offset to geometric mean"
    self.dL = np.array(
        [
            self.target_m["x"] - self.source_m["x"],
            self.target_m["z"] - self.source_m["z"],
        ]
    )
    self.Ro = np.exp((np.log(self.source_m["x"]) + np.log(self.target_m["x"])) / 2)
    self.dL_mag = np.linalg.norm(self.dL, axis=0)
    iszero = np.isclose(self.dL_mag, 0)  # self index
    self.dL_norm = np.zeros((2, self.nT, self.nS))
    self.dL_norm[:, ~iszero] = self.dL[:, ~iszero] / self.dL_mag[~iszero]
    self.dL_norm[0, iszero] = 1
    # self inductance index
    dr = (self.source_m["dx"] + self.source_m["dz"]) / 4  # mean radius
    idx = self.dL_mag < dr  # seperation < mean radius
    # mutual inductance offset
    if self.mutual_offset:  # mutual inductance offset
        nx = abs(self.dL_mag / self.source_m["dx"])
        nz = abs(self.dL_mag / self.source_m["dz"])
        mutual_factor = self.gmr.evaluate(nx, nz)
        mutual_adjust = (mutual_factor - 1) / 2
        for i, key in enumerate(["x", "z"]):
            offset = mutual_adjust[~idx] * self.dL[i][~idx]
            self._apply_offset(key, offset, ~idx)
    # self-inductance offset
    factor = (1 - self.dL_mag[idx] / dr[idx]) / 2
    ro = np.max([self.source_m["ro"][idx], self.target_m["ro"][idx]], axis=0)
    for i, key in enumerate(["x", "z"]):
        offset = factor * ro * self.dL_norm[i][idx]
        self._apply_offset(key, offset, idx)


def _apply_offset(self, key, offset, index):
    if key == "r":
        Ro_offset = np.exp(
            (
                np.log(self.source_m[key][index] - offset)
                + np.log(self.target_m[key][index] + offset)
            )
            / 2
        )
        shift = self.Ro[index] - Ro_offset  # gmr shift
    else:
        shift = np.zeros(np.shape(offset))
    self.source_m[key][index] -= offset + shift
    self.target_m[key][index] += offset - shift
    return shift


def locate(self):
    xt, zt = self.target_m["x"], self.target_m["z"]
    xs, zs = self.source_m["x"], self.source_m["z"]
    return xt, zt, xs, zs


# from simulation data
def update_interaction(self, coil_index=None, **kwargs):
    self.generate_grid(**kwargs)  # add | append data targets
    self.add_targets(**kwargs)  # re-generate grid on demand
    if coil_index is not None:  # full update
        self.grid["update"] = True and self.grid["n"] > 0
        self.target["update"] = True
        self.target["targets"]["update"] = True
    update_targets = self.grid["update"] or self.target["update"]
    if update_targets or coil_index is not None:
        if coil_index is None:
            coilset = self.coilset  # full coilset
        else:
            coilset = self.subset(coil_index)  # extract subset
        bs = biot_savart(source=coilset, mutual=False)  # load coilset
        if self.grid["update"] and self.grid["n"] > 0:
            bs.load_target(
                self.grid["x2d"].flatten(),
                self.grid["z2d"].flatten(),
                label="G",
                delim="",
                part="grid",
            )
            self.grid["update"] = False  # reset update status
        if self.target["update"]:
            update = self.target["targets"]["update"]  # new points only
            targets = self.target["targets"].loc[update, :]  # subset
            bs.load_target(
                targets["x"], targets["z"], name=targets.index, part="target"
            )
            self.target["targets"].loc[update, "update"] = False
            self.target["update"] = False
        M = bs.calculate_interaction()
        for matrix in M:
            if self.interaction[matrix].empty:
                self.interaction[matrix] = M[matrix]
            elif coil_index is None:
                drop = self.interaction[matrix].index.unique(level=1)
                for part in M[matrix].index.unique(level=1):
                    if part in drop:  # clear prior to concat
                        if part == "target":
                            self.interaction[matrix].drop(
                                points.index, level=0, inplace=True, errors="ignore"
                            )
                        else:
                            self.interaction[matrix].drop(part, level=1, inplace=True)
                self.interaction[matrix] = concat([self.interaction[matrix], M[matrix]])
            else:  # selective coil_index overwrite
                for name in coilset.coil.index:
                    self.interaction[matrix].loc[:, name] = M[matrix].loc[:, name]


def solve_interaction(self, plot=False, color="gray", *args, **kwargs):
    "generate grid / target interaction matrices"
    self.update_interaction(**kwargs)  # update on demand
    for matrix in self.interaction:  # Psi, Bx, Bz
        if not self.interaction[matrix].empty:
            # variable = matrix.lower()
            # index = self.interaction[matrix].index
            # value = np.dot(
            #        self.interaction[matrix].loc[:, self.coil.data.index],
            #        self.coil.data.Ic)
            # value = self.interaction[matrix].dot(self.Ic)
            np.dot(self.interaction[matrix].to_numpy(), self.Ic)
            # coil = DataFrame(value, index=index)  # grid, target
            """
            for part in coil.index.unique(level=1):
                part_data = coil.xs(part, level=1)
                part_dict = getattr(self, part)
                if 'n2d' in part_dict:  # reshape data to n2d
                    part_data = part_data.to_numpy()
                    part_data = part_data.reshape(part_dict['n2d'])
                    part_dict[matrix] = part_data
                else:
                    part_data = concat(
                            (Series({'t': self.t}), part_data),
                            sort=False)
                    part_dict[matrix] = concat(
                            (part_dict[matrix], part_data.T),
                            ignore_index=True, sort=False)
            """
    if plot and self.grid["n"] > 0:
        if self.grid["levels"] is None:
            levels = self.grid["nlevels"]
        else:
            levels = self.grid["levels"]
        QuadContourSet = plt.contour(
            self.grid["x2d"],
            self.grid["z2d"],
            self.grid["Psi"],
            levels,
            colors=color,
            linestyles="-",
            linewidths=1.0,
            alpha=0.5,
            zorder=5,
        )
        self.grid["levels"] = QuadContourSet.levels
        plt.axis("equal")
        # plt.quiver(self.grid['x2d'], self.grid['z2d'],
        #           self.grid['Bx'], self.grid['Bz'])

    """
    def index_part(self, M):
        M.loc[:, 'part'] = self.target.coil['part']
        M.set_index('part', append=True, inplace=True)
        return M

    def column_reduce(self, Mo):
        Mo = pd.DataFrame(Mo, index=self.target.subcoil.index,
                          columns=self.source.subcoil.index, dtype=float)
        Mcol = pd.DataFrame(index=self.target.subcoil.index,
                            columns=self.source.coil.index, dtype=float)
        for name in self.source.coil.index:  # column reduction
            index = self.source.coil.subindex[name]
            Mcol.loc[:, name] = Mo.loc[:, index].sum(axis=1)
        return Mcol

    def row_reduce(self, Mcol):
        Mrow = pd.DataFrame(columns=self.source.coil.index, dtype=float)
        if 'subindex' in self.target.coil.columns:
            #part = self.target.coil['part']
            for name in self.target.coil.index:  # row reduction
                index = self.target.coil.subindex[name]
                Mrow.loc[name, :] = Mcol.loc[index, :].sum(axis=0)
        else:
            #part = self.target.subcoil['part']
            Mrow = Mcol
        #Mrow['part'] = part
        #Mrow.set_index('part', append=True, inplace=True)
        return Mrow

    def calculate_inductance(self):
        self.colocate()  # set targets
        Mc = self.row_reduce(self.flux_matrix())  # line-current
        return Mc

    def calculate_interaction(self):
        self.assemble(offset=True)  # build interaction matrices
        M = {}
        M['Psi'] = self.flux_matrix()  # line-current interaction
        return M
    """


class self_inductance:
    """
    self-inductance methods for a single turn circular coil
    """

    def __init__(self, x, cross_section="circle"):
        self.x = x  # coil major radius
        self.cross_section = cross_section  # coil cross_section
        self.cross_section_factor = geometric_mean_radius.gmr_factor[self.cross_section]

    def minor_radius(self, L, bounds=(0, 1)):
        """
        inverse method, solve coil minor radius for given inductance

        Attributes:
            L (float): target inductance Wb
            bounds (tuple of floats): bounds fraction of major radius

        Returns:
            dr (float): coil minor radius
        """
        self.Lo = L
        r = minimize_scalar(
            self.flux_err,
            method="bounded",
            bounds=bounds,
            args=(self.Lo),
            options={"xatol": 1e-12},
        ).x
        gmr = self.x * r
        dr = gmr / self.cross_section_factor
        return dr

    def flux_err(self, r, *args):
        gmr = r * self.x
        L_target = args[0]
        L = self.flux(gmr)
        return (L - L_target) ** 2

    def flux(self, gmr):
        """
        calculate self-induced flux though a single-turn coil

        Attributes:
            a (float): coil major radius
            gmr (float): coil cross-section geometric mean radius

        Retuns:
            L (float): self inductance of coil
        """
        if self.x > 0:
            L = self.x * (
                (1 + 3 * gmr**2 / (16 * self.x**2)) * np.log(8 * self.x / gmr)
                - (2 + gmr**2 / (16 * self.x**2))
            )
        else:
            L = 0
        return biot_savart.mu_o * L  # Wb
