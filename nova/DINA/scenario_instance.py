from nep.DINA.read_scenario import scenario_data, read_scenario
from nep.coil_geom import PFgeom

cc = PFgeom(dCoil=0.5).cc

cc.add_coil(5, 1, 0.5, 0.5, dCoil=0.1, name='new')
cc.add_mpc(['PF1', 'new'], 1)  # link new to 'CS1L'

scn = read_scenario(folder=0, read_txt=False)

sd2 = scenario_data(folder=0, read_txt=False)

sd2.update(200)
cc.Ic = sd2.Ic


#scn.update_coilset(cc.coilset)

cc.plot(label=True, current=True, unit='A')
