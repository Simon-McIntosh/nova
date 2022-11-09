import pytest
from numpy import allclose

@pytest.mark.skip('Field calculation for coil perimiters not currently available.')
def test_IM_field(plot=False):
    # build ITER coilset
    '''
    cc = CoilClass(dCoil=0.25, dField=0.5)
    cc.coilset = PFgeom(VS=False, dCoil=cc.dCoil, source='PCR').coilset
    cc.biot_instances = 'field'
    cc.field.add_coil(cc.coil, ['CS', 'PF'], dField=cc.dField)
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
    return cc
    '''


if __name__ == '__main__':

    pytest.main([__file__])
    #cc = test_IM_field(plot=True)
