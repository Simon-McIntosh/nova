
@dataframe
class BiotPack(BiotMatrix, BiotAttributes):

    def __init__(self, source=None, target=None, **biot_attributes):
        CoilMatrix.__init__(self)
        BiotAttributes.__init__(self, **biot_attributes)
        self.source = BiotFrame(reduce=self.reduce_source)
        self.target = BiotFrame(reduce=self.reduce_target)
        self._nS, self._nT = 0, 0
        self.load_biotset(source, target)

    def load_biotset(self, source=None, target=None):
        if source is not None:
            self.source.add_coil(source)
        if target is not None:
            self.target.add_coil(target)

    def assemble_biotset(self):
        self.source.update_coilframe()
        self.target.update_coilframe()
        self.assemble()

    @property
    def nS(self):
        return self._nS

    @nS.setter
    def nS(self, nS):
        self._nS = nS
        self.target.nS = nS  # update target source filament number

    @property
    def nT(self):
        return self._nT

    @nT.setter
    def nT(self, nT):
        self._nT = nT
        self.source.nT = nT  # update source target filament number

    def assemble(self):
        self.nS = self.source.nC  # source filament number
        self.nT = self.target.nC  # target point number
        self.nI = self.source.nC*self.target.nC  # total number of interactions

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.source.x, self.source.z, 'C1o', label='source')
        ax.plot(self.target.x, self.target.z, 'C2.', label='target')
        ax.legend()
        ax.set_axis_off()
        ax.set_aspect('equal')

