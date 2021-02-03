
from nova.utilities.pyplot import plt

from nova.thermalhydralic.sultan.fluidresponse import FluidResponse
from nova.thermalhydralic.twente.twentedata import TwenteSource, TwentePost


sultan = FluidResponse('CSJA13', -1, 'Left')
#sultan.read_data()


twentesource = TwenteSource('CS_KAT', phase='virgin', index=0, binary=True)
twente = TwentePost(twentesource, binary=False)
poly = twente.fit_polynomial(2, plot=False)
twente.Qhys = 0.985*poly[0]
twente.plot()

frequency, gain = sultan.response(2)
plt.plot(frequency, gain, 'o-', label='fluid gain')

plt.plot(twente.frequency, twente.Prms)

plt.xscale('log')
plt.yscale('log')



#sultan.plot(2, dcgain_limit=1e8)