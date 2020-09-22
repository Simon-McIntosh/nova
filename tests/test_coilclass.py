from numpy import allclose

from nova.electromagnetic.coilclass import CoilClass 
from nova.electromagnetic.coilgeom import PFgeom

def test_dina_scenario_filename():
    cc = CoilClass(dCoil=0.15, n=1e3, expand=0.25, nlevels=51,
                   current_update='active')
    
    cc.update_coilframe_metadata('coil', additional_columns=['R'])
    cc.scenario_filename = '15MA DT-DINA2016-01_v1.1'
    assert cc.scenario['filename'] == '15MA_DT-DINA2016-01_v1.1'
    
def test_IM_field(plot=False):
    # build ITER coilset
    cc = CoilClass(dCoil=0.25, dField=0.5)
    cc.coilset = PFgeom(VS=False, dCoil=cc.dCoil, source='PCR').coilset
    cc.field.add_target(cc.coil, ['CS', 'PF'], dField=cc.dField)
    cc.field.solve()
    # load DINA scenario
    cc.filename = '15MA DT-DINA2020-04'
    cc.scenario = 'IM'
    # retreve DINA field data
    vector = cc.d3.vector  
    field_index = [index for index in vector.index if index[:2] == 'B_']
    vector = vector.loc[field_index]
    vector.rename(index={index: index[2:].upper() for index in vector.index},
                  inplace=True)
    if plot:
        cc.plot(True)
        cc.field.plot()
    assert allclose(vector.values, 
                    cc.field.frame.loc[vector.index, 'B'].values,
                    atol=0.25)

if __name__ == '__main__':
    
    test_IM_field(plot=True)
    

'''
    x, z, dx = 5.5, -5, 2
    dz = 2*dx
    cc.add_coil(x+1, z+2, dx, dz, name='PF1', part='PF', Ic=-15e6,
                cross_section='square', turn_section='circle', 
                turn_fraction=0.75, Nt=15, dCoil=-1)

    cc.add_coil(x+3, z, dx, dz, name='PF3', part='PF', Ic=-25e6,
                cross_section='square', turn_section='circle', 
                turn_fraction=0.95, Nt=27, dCoil=-1, power=False)
    
    cc.add_coil(x+6, z, dx, dz, name='PF2', part='PF', Ic=-5e6,
                cross_section='circle', turn_section='skin', 
                turn_fraction=0.95, skin_fraction=0.5, 
                Nt=27, dCoil=-1, power=True)
    
    cc.add_coil(x+6, z+4, dx, dz, name='PF6', part='PF', Ic=-5e6,
                cross_section='ellipse', turn_section='rectangle', 
                turn_fraction=1, Nt=157, dCoil=-1, power=False)
    
    cc.add_mpc(['PF3', 'PF1'], 1)  # link coils
    
    cc.drop_coil('PF3')
    #print(cc.current_update)
    #print(cc.subcoil.current_update)
    
    cc.Ic = 6
    cc.current_update = 'passive'
    cc.Ic = -12
    
    cc.current_update = 'plasma'
    cc.scenario = 100
    
    print(cc.coil['Ic'], cc.coil.Ic)
    print(cc.coil.current_update)
    
    cc.plot(True)
    
    cc.grid.generate_grid()
    cc.grid.plot_flux()
'''