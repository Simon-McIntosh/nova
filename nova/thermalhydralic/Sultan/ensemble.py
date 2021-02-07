
from nova.thermalhydralic.sultan.remotedata import FTPData
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.sultanspectrum import SultanSpectrum


class Ensemble:

    def __init__(self):
        ftp = FTPData('')
        for experiment in ftp.listdir(select='CSJA')[2:]:
            for phase in Campaign(experiment).index:
                for side in ['Left', 'Right']:
                    spectrum = SultanSpectrum(experiment, phase, side,
                                              reload=True)

if __name__ == '__main__':
    ensemble = Ensemble()