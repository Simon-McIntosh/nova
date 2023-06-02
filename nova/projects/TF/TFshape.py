"""
if 'config' in shape:

else:
    for name in ['dataname','filename']:
        if name in shape:
            setattr(self,name,shape.get(name))
"""

# self.rp = ripple(nTF=self.nTF,plasma={'sf':shape.get('sf',None)})


def set_bound(self, shape):
    self.bound = {}
    for side in ["internal", "external"]:
        R, Z = np.array([]), np.array([])
        if side == "internal" and "vessel" in shape:
            vv = shape["vessel"]  # add vessel loop
            R, Z = np.append(R, vv.R), np.append(Z, vv.Z)
        if "pf" in shape and hasattr(self, "setup"):
            Rpf, Zpf = shape["pf"].coil_corners(self.setup.coils[side])
            R, Z = np.append(R, Rpf), np.append(Z, Zpf)
        self.bound[side] = {"R": R, "Z": Z}
    # self.bound['ro_min'] = 4.35  # minimum TF radius
    if len(self.bound) == 0:
        errtxt = "\n"
        errtxt += "Require TF bounds input,"
        errtxt += "shape['vessel'] and or shape['pf'] + self.setup:\n"
        raise ValueError(errtxt)


def plot_bounds(self):
    for side, marker in zip(["internal", "external"], [".", "x"]):
        pl.plot(
            self.bound[side]["R"],
            self.bound[side]["Z"],
            marker,
            markersize=6,
            color=colors[9],
        )


def set_oppvar(self):  # set optimization bounds and normalize
    nopp = len(self.coil.oppvar)
    xo, self.bounds = np.zeros(nopp), np.zeros((nopp, 2))
    xnorm, bnorm = np.zeros(nopp), np.zeros((nopp, 2))
    for i, var in enumerate(self.coil.oppvar):
        xo[i] = self.coil.xo[var]["value"]
        self.bounds[i, :] = (self.coil.xo[var]["lb"], self.coil.xo[var]["ub"])
        xnorm[i] = (xo[i] - self.bounds[i, 0]) / (self.bounds[i, 1] - self.bounds[i, 0])
        bnorm[i, :] = [0, 1]
    return xnorm, bnorm


def get_oppvar(self, xnorm):
    xo = np.copy(xnorm)
    for i in range(len(xnorm)):
        xo[i] = xo[i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
    return xo


"""

# generate

            self.load_coil()  # load coil object
        if self.datatype == 'fit':
            tic = time.time()
            xnorm,bnorm = self.set_oppvar()  # normalized inputs
            xnorm = fmin_slsqp(self.fit,xnorm,
                               f_ieqcons=self.constraint_array,
                               bounds=bnorm,acc=0.02)
            xo = self.get_oppvar(xnorm)
            self.coil.set_input(xo=xo)  # TFcoil inner loop
            print('optimisation time {:1.1f}s'.format(time.time()-tic))
            print('noppvar {:1.0f}'.format(len(self.coil.oppvar)))
            # tf.save_loop here...



        x = self.coil.draw()
        self.get_coil_loops(x['r'],x['z'],profile='in')

        # self.rp.set_TFcoil(Rcl=self.Rcl,Zcl=self.Zcl,smooth=True)
        # print('ripple',self.rp.get_ripple())
"""


def plot_oppvar(self, eps=1e-2):
    xnorm, bnorm = self.set_oppvar()
    for var in self.coil.xo:
        self.coil.xo[var]["xnorm"] = (
            self.coil.xo[var]["value"] - self.coil.xo[var]["lb"]
        ) / (self.coil.xo[var]["ub"] - self.coil.xo[var]["lb"])
    data = pd.DataFrame(self.coil.xo).T
    data.reset_index(level=0, inplace=True)
    pl.figure(figsize=(8, 4))
    sns.set_color_codes("muted")
    sns.barplot(x="xnorm", y="index", data=data, color="b")
    sns.despine(bottom=True)
    pl.ylabel("")
    ax = pl.gca()
    ax.get_xaxis().set_visible(False)
    patch = ax.patches
    values = [self.coil.xo[var]["value"] for var in self.coil.xo]
    xnorms = [self.coil.xo[var]["xnorm"] for var in self.coil.xo]
    for p, value, xnorm, var in zip(patch, values, xnorms, self.coil.xo):
        x = p.get_width()
        y = p.get_y()
        if xnorm < eps or xnorm > 1 - eps:
            color = "r"
        else:
            color = "k"
        text = " {:1.3f}".format(value)
        if var not in self.coil.oppvar:
            text += "*"
        ax.text(x, y, text, ha="left", va="top", size="small", color=color)
    pl.plot(
        0.5 * np.ones(2),
        np.sort(ax.get_ylim()),
        "--",
        color=0.5 * np.ones(3),
        zorder=0,
        lw=1,
    )
    pl.plot(
        np.ones(2), np.sort(ax.get_ylim()), "-", color=0.5 * np.ones(3), zorder=0, lw=2
    )
    pl.xlim([0, 1])


def constraint_array(self, xnorm, ripple_limit=0.6):
    xo = self.get_oppvar(xnorm)  # de-normalize
    x = self.coil.draw(xo=xo)
    dot = np.array([])
    for side in ["internal", "external"]:
        dot = np.append(dot, self.dot_diffrence(x, side))
    self.get_coil_loops(x["r"], x["z"], profile="in")
    self.rp.set_TFcoil(Rcl=self.Rcl, Zcl=self.Zcl)
    max_ripple = self.rp.get_ripple()
    dot = np.append(dot, ripple_limit - max_ripple)
    return dot


def fit(self, xnorm):
    xo = self.get_oppvar(xnorm)  # de-normalize
    x = self.coil.draw(xo=xo)
    if self.objective == "L":  # coil length
        objF = geom.length(x["r"], x["z"], norm=False)[-1]
    else:  # coil volume
        objF = geom.loop_vol(x["r"], x["z"])
    return objF


def ripple(self, xnorm, ripple_limit=0.6):
    dsum = 0
    xo = self.get_oppvar(xnorm)  # de-normalize
    x = self.coil.draw(xo=xo)
    self.get_coil_loops(x["r"], x["z"], profile="in")
    self.rp.set_TFcoil(Rcl=self.Rcl, Zcl=self.Zcl, nTF=self.nTF)
    max_ripple = self.rp.get_ripple()
    if max_ripple > ripple_limit:
        dsum -= max_ripple - ripple_limit
    return dsum


def dot(self, xnorm):
    dsum = 0
    xo = self.get_oppvar(xnorm)  # de-normalize
    x = self.coil.draw(xo=xo)
    for side in ["internal", "external"]:
        dsum += self.dot_diffrence(x, side)
    return dsum


def dot_diffrence(self, x, side):
    Rloop, Zloop = x["r"], x["z"]  # inside coil loop
    switch = 1 if side == "internal" else -1
    if side == "external":
        Rloop, Zloop = geom.offset(Rloop, Zloop, self.dRcoil + 2 * self.dRsteel)
    nRloop, nZloop, Rloop, Zloop = geom.normal(Rloop, Zloop)
    R, Z = self.bound[side]["R"], self.bound[side]["Z"]
    # dsum = 0
    dot = np.zeros(len(R))
    for j, (r, z) in enumerate(zip(R, Z)):
        i = np.argmin((r - Rloop) ** 2 + (z - Zloop) ** 2)
        dr = [Rloop[i] - r, Zloop[i] - z]
        dn = [nRloop[i], nZloop[i]]
        dot[j] = switch * np.dot(dr, dn)
        # if dot < 0:
        #    dsum -= (dr[0]**2+dr[1]**2)
    return dot


"""
def amp_turns(self):
    if hasattr(self,'filename'):
        eqdsk = nova.geqdsk.read(self.filename)
        mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]
        self.Iturn = 2*np.pi*eqdsk['rcentr']*np.abs(eqdsk['bcentr'])/\
        (mu_o*self.nTF)
    else:
        self.Iturn = 0

def energy(self,plot=False,Jmax=7.2e7,Nseg=100,**kwargs):
    if 'Iturn' in kwargs:
        self.Iturn = kwargs['Iturn']
    elif 'nTF' in kwargs:
        self.nTF = kwargs['nTF']
        self.amp_turns()
    else:
        self.amp_turns()

    self.Acs = self.Iturn/Jmax
    R,Z = geom.rzSLine(self.Rcl,self.Zcl,npoints=Nseg)  # re-space
    Xo = np.zeros((Nseg,3))
    Xo[:,0],Xo[:,2] = R,Z
    theta = np.linspace(0,2*np.pi,self.nTF,endpoint=False)
    r = np.sqrt(self.Acs)
    nturn = 1
    neu = neumann()
    M = np.zeros((self.nTF))
    if plot:
        fig = pl.figure()
        ax = fig.gca(projection='3d')
    for i in np.arange(0,self.nTF):
        X = np.dot(Xo,geom.rotate(theta[i]))
        neu.setX(X)
        neu.setr(r)
        neu.setX_(Xo)
        M[i] = nturn**2*neu.calculate()
        if plot:
            ax.plot(np.append(X[:,0],X[0,0]),np.append(X[:,1],X[0,1]),
                    np.append(X[:,2],X[0,2]))
    self.Ecage = 0.5*self.Iturn**2*self.nTF*np.sum(M)
"""
