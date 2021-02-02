
import numpy as np

from nova.thermalhydralic.sultan.fluidresponse import FluidResponse
from nova.utilities.pyplot import plt


fluid = FluidResponse('CSJA_6', 0, 'Left')

omega, steadystate = fluid.response(2)

signal = 'unit'

steadystate *= (0.2*omega)**2  # raw signal

if signal == 'raw':
    plt.plot(omega / (2*np.pi), steadystate, '.-', label=r'raw: 1')
elif signal == 'MPAS':
    steadystate /= omega
    plt.plot(omega / (2*np.pi), steadystate, 'C1.-',
             label=r'MPAS: $\omega^{-1}$')
elif signal == 'unit':
    steadystate /= (0.2*omega)**2
    plt.plot(omega / (2*np.pi), steadystate, 'C2.-',
         label=r'unit input: $\alpha^{-2}\omega^{-2}$')

plt.xscale('log')
plt.xlabel('$f$ Hz')
plt.ylabel('$\dot{Q}$ W')
plt.despine()
plt.legend()