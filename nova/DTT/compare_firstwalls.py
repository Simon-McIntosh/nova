import pylab as pl
from config import Setup
from streamfunction import SF
from radial_build import RB


for config in ['SN','SX','SFm','SFp','Xic']:
    setup = Setup(config)
    sf = SF(setup.filename)
    pl.plot(sf.rbdry,sf.zbdry,label=config)
    #rb = RB(setup,sf)
    #rb.firstwall(calc=False,plot=True)

pl.axis('equal')
pl.legend(loc=10)