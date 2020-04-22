from nep.DINA.read_scenario import scenario_data
import pandas as pd
from amigo.time import clock
from os.path import join


d2 = scenario_data(read_txt=False)
coils = ['cs3u', 'cs2u', 'cs1', 'cs2l', 'cs3l',
         'pf1', 'pf2', 'pf3', 'pf4', 'pf5', 'pf6']
with pd.ExcelWriter(join(d2.directory, '../Pgrid.xlsx')) as writer:
    # with pd.HDFStore('Pgrid.h5') as hdf5:
    tick = clock(d2.nfolder, header='writing grid waveforms')
    for folder in d2.folders:
        d2.load_file(folder)
        cps_col = [col[1:] for col in d2.frame.columns.levels[0].values
                   if (col[0] == 'I'
                   and 'loop' not in col
                   and 're' not in col
                   and 'vv' not in col
                   and 'vc' not in col
                   and len(col) > 2
                   and len(col) < 6)]
        Pcps = 0
        for i, cps in enumerate(cps_col):
            try:
                Pcps += d2.frame[f'I{cps}'].values * d2.frame[f'V{cps}'].values
            except KeyError:
                print(d2.filename)
                Vcps = f'Vmc{coils.index(cps)+1}'
                Pcps += d2.frame[f'I{cps}'].values * d2.frame[Vcps].values
        d2.frame['Pcps', 'MW'] = Pcps
        columns = ['t', 'Pcps']
        if 'Paux' in d2.frame.columns.levels[0]:
            columns += ['Paux']
        columns += [col for col in d2.frame.columns.levels[0].values
                    if 'grid' in col]
        Pgrid = d2.frame[columns]
        # Pgrid.to_excel(writer, sheet_name=d2.filename)
        # hdf5[d2.filename] = Pgrid
        tick.tock()

'''
Pgrid variable only in 2015
at that time it was only the CPS power (sigma(Vi*Ii)).
Then we added in 2017 a more accurate Pgrid
considering the H&CD with their correct efficiency.
voltages and currents of PF/CS coils, and the value Paux?
'''