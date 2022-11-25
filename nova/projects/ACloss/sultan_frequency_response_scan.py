
from matplotlib.lines import Line2D

from nova.thermalhydralic.sultan.fluidresponse import FluidResponse
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.remotedata import FTPData
from nova.plot import plt

linestyle = {'ac0': '-', 'ac1': '--', 'ac2': ':'}
strand = {'CSJA_6': {'Left': 'Jastec', 'Right': 'Jastec'},
          'CSJA_7': {'Left': 'KAT', 'Right': 'KAT'},
          'CSJA_8': {'Left': 'KAT', 'Right': 'KAT'},
          'CSJA_9': {'Left': 'Jastec', 'Right': 'Jastec'},
          'CSJA_10': {'Left': 'Furukawa', 'Right': 'Furukawa'},
          'CSJA11': {'Left': 'Furukawa', 'Right': 'Furukawa'},
          'CSJA12': {'Left': 'KAT', 'Right': 'Jastec'},
          'CSJA13': {'Left': 'KAT', 'Right': 'Furukawa'}}
strandcolor = {'Jastec': 'C2', 'KAT': 'C0', 'Furukawa': 'C3'}

ftp = FTPData('')
for i, experiment in enumerate(ftp.listdir(select='CSJA')[2:]):
    for phase in Campaign(experiment).index:
        for side in ['Left', 'Right']:
            fluid = FluidResponse(experiment, phase, side)
            frequency, gain = fluid.response(2)
            strandname = strand[experiment][side]
            color = strandcolor[strandname]
            plt.plot(frequency, gain, f'C{i}', color=color,
                     ls=linestyle[phase])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\omega$ rads$^{-1}$')
plt.ylabel(r'$\dot{Q}$ W')
plt.despine()

plt.legend(handles=[
    Line2D([0], [0], color=strandcolor['KAT'], label='KAT'),
    Line2D([0], [0], color=strandcolor['Jastec'], label='Jastec'),
    Line2D([0], [0], color=strandcolor['Furukawa'], label='Furukawa'),
    Line2D([0], [0], ls='-', color='gray', label='virgin'),
    Line2D([0], [0], ls='--', color='gray', label='cycled')])
