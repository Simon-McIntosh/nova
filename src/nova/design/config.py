import numpy as np
import collections
from nova.DEMOxlsx import DEMO
import nova
import os.path


def select(base={'TF': 'dtt', 'eq': 'SN'}, nTF=18, update=True, **kwargs):
    config = {'nTF': nTF}
    config['eq_base'] = base['eq']
    config['TF_base'] = base['TF']
    if 'nCS' in kwargs:
        config['nCS'] = kwargs.get('nCS')
    if 'nPF' in kwargs:
        config['nPF'] = kwargs.get('nPF')
    name = base.get('name', base.get('eq'))
    config['TF'] = '{}_{}'.format(name, base['TF'])
    if 'nCS' in kwargs and 'nPF' in kwargs:
        config['eq'] = '{:s}_{:s}_{:d}TF_{:d}PF_{:d}CS'.format(
            name, base['TF'], nTF, config['nPF'], config['nCS'])
    else:
        config['eq'] = '{:s}_{:s}_{:d}TF'.format(name, base['TF'], nTF)
    setup = Setup(base['eq'])
    if update:  # update eqdsk filename
        setup.filename = setup.eqdir + config['eq'] + '.eqdsk'
    return config, setup


def nova_path(directory, file=''):
    sep = os.path.sep
    root = sep.join(nova.__file__.split(sep)[:-2])
    path = os.path.join(root, directory)+sep
    path = os.path.join(path, file)
    return path


class Setup(object):

    # preferance for all eqdsks should be stored in Nova/eqdsk/ or subdirs
    # overide with eqdsk keyword
    def __init__(self, configuration='', **kwargs):
        eqdir = nova_path('eqdsk')
        self.eqdir = kwargs.get('eqdir', eqdir)
        if 'subdir' in kwargs:
            self.eqdir += kwargs['subdir']+'/'
        self.configuration = configuration
        self.set_defaults()
        self.update(configuration)
        self.backfill()

    def set_defaults(self):
        self.filename = ''
        self.dataname = ''
        self.targets = collections.OrderedDict()  # targets structure (ordered)
        self.targets['default'] = {}
        self.targets['default']['L2D'] = 0.8
        self.targets['default']['open'] = False
        self.targets['default']['graze'] = 1.5 * np.pi / 180
        self.targets['default']['dPlate'] = 0.5
        self.targets['default']['dR'] = 0
        self.firstwall = {}  # initalise firstwall data structure
        self.firstwall['dXfw'] = 0.225
        self.firstwall['psi_n'] = 1.07
        self.firstwall['div_ex'] = 0.25  # lower divertor flux offset
        self.firstwall['dx_div'] = 0.5  # divertor thickness
        self.firstwall['bb_gap'] = 0.05  # divertor-blanket gap
        self.firstwall['vv_gap'] = 0.0  # extra divertor-vessel gap
        self.firstwall['trim'] = [0.75, 0.7]
        self.firstwall['flux_fit'] = True
        self.firstwall['conformal'] = False
        self.build = {}  # initalise build data structure
        self.build['tfw'] = 0.1  # first wall thickness
        self.build['tBBsupport'] = 0.1  # blanket support
        self.build['BS'] = np.array(
            [0.78 - 0.2, 1.304 - 0.2])  # blanket+sheild #PMI
        self.build['BS'] -= (self.build['tfw'] + self.build['tBBsupport'])
        BBfrac = np.array([1, 1])
        # demo = DEMO()  # load from xls baseline file
        # demo.blanket_thickness()  # min/max
        self.build['BB'] = [0.77531, 1.30027]
        # self.build['BB'] = list(BBfrac*self.build['BS'])  # blanket (in/out)
        self.build['sheild'] = list(
            (1 - BBfrac) * self.build['BS'])  # sheilding
        self.build['sheild_connect'] = [0, 1]
        self.build['Dsheild'] = []  # wrap sheild around divertor [0,1]
        # if Dsheild [], base gap divertor-sheild
        self.build['sheild_base'] = -1
        self.build['VV'] = [0.587, 1.2]  # vacumn vessel thickness (in/out)
        self.build['VV'] = [0.5, 1.2]  # vacumn vessel thickness (in/out)
        self.TF = {}
        self.TF['opp'] = 'L'  # L==length, V==volume
        # coils structure
        self.coils = {'internal': {'id': [], 'dR': [], 'dZ': []},
                      'external': {'id': [], 'dR': [], 'dZ': []}}

    def backfill(self):
        for key in list(self.targets)[1:]:  # backfill defaults
            for default in list(self.targets['default']):
                if default not in self.targets[key]:
                    self.targets[key][default] = \
                        self.targets['default'][default]

    def update(self, configuration):  # update
        self.configuration = configuration
        if configuration == 'DEMO_SF':
            self.dataname = 'SFm'
            # self.filename = '/Equil_AR3d1_16coils_SFminus_v4_2015' +\
            #    '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            # self.filename = '/2015_SFminus_eqdsk_2MCQRZ_v1_0_IDM.eqdsk'
            self.targets['default']['dPlate'] = 0.35  # target plate length
            self.targets['inner1'] = {
                'L2D': [1.1 + 0.52], 'open': True, 'dR': 0}
            self.targets['inner2'] = {
                'L2D': [1.2 - 0.08], 'open': False, 'dR': -1}
            self.targets['outer1'] = {
                'L2D': [1.65 - 0.2], 'open': False, 'dR': -1}
            self.targets['outer2'] = {
                'L2D': [1.1 + 0.3], 'open': True, 'dR': 0}
            self.firstwall['div_ex'] = 0.2
            self.firstwall['trim'] = [0.75, 0.7]  # trim fraction (in/out)
            self.coils['external']['id'] = list(range(0, 16))  # all external
            self.TF['opp'] = 'L'

        if configuration == 'SFm':
            self.dataname = 'SFm'
            self.filename = '/Equil_AR3d1_16coils_SFminus_v4_2015' +\
                '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            # self.filename = '/2015_SFminus_eqdsk_2MCQRZ_v1_0_IDM.eqdsk'
            self.targets['default']['dPlate'] = 0.35  # target plate length
            self.targets['inner1'] = {
                'L2D': [1.1 + 0.52], 'open': True, 'dR': 0}
            self.targets['inner2'] = {
                'L2D': [1.2 - 0.08], 'open': False, 'dR': -1}
            self.targets['outer1'] = {
                'L2D': [1.65 - 0.7], 'open': False, 'dR': -1}
            self.targets['outer2'] = {
                'L2D': [1.1 + 0.15], 'open': True, 'dR': 0}
            self.firstwall['div_ex'] = 0.18
            self.firstwall['trim'] = [0.75, 0.7]  # trim fraction (in/out)
            self.coils['external']['id'] = list(range(0, 16))  # all external
            self.TF['opp'] = 'L'

        elif configuration == 'SFp':
            self.dataname = 'SFp'
            self.filename = '/Equil_AR3d1_16coils_SFplus_v4_2015' +\
                '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            self.targets['default']['dPlate'] = 0.35  # target plate length
            self.targets['inner1'] = {'L2D': [1.1], 'open': True, 'dR': 0.1}
            self.targets['inner2'] = {'L2D': [1.2], 'open': False, 'dR': -1}
            self.targets['outer1'] = {'L2D': [1.65], 'open': False, 'dR': -1}
            self.targets['outer2'] = {'L2D': [1.1], 'open': True, 'dR': 0.1}
            self.firstwall['div_ex'] = 0.18
            self.firstwall['trim'] = [0.75, 0.7]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.22, 1]
            self.TF['opp'] = 'L'
            self.coils['external']['id'] = list(range(0, 16))

        elif configuration == 'SX_old':
            self.dataname = 'SX8'
            self.filename = '/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3.eqdsk'
            #self.filename = '/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            #self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3_new.eqdsk'
            self.filename = '/2015_SX_ext_coils_eqdk_2MK6XX_v1_0.eqdsk'  # IDM
            #self.filename = '/Equil_AR3d1_16coils_bt_1d03li_0d8_Ipl_'
            #self.filename += '20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter3_v5_old.eqdsk'
            self.targets['inner'] = {'L2D': [1.1]}
            self.targets['outer'] = {'L2D': [3.55]}
            self.firstwall['div_ex'] = 1
            self.firstwall['trim'] = [0.71, 0.74]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.15, 0.9]
            self.coils['external']['id'] = list(range(10, 16))

        elif configuration == 'SX':
            self.dataname = 'SXex'
            #self.filename = '/SXex.eqdsk'
            self.filename = 'SXMOF.eqdsk'
            self.targets['inner'] = {'L2D': [1.1]}
            self.targets['outer'] = {'L2D': [3.55]}
            self.firstwall['div_ex'] = 1.75
            self.firstwall['trim'] = [0.63, 0.68]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.2, 0.9]
            self.build['sheild_base'] = 0
            self.targets['inner'] = {'L2D': [1.0], 'dR': 0.15, 'dPlate': 0.25}
            self.targets['outer'] = {'L2D': [5.615], 'dPlate': 1.0}

        elif configuration == 'DEMO_SX':
            self.dataname = 'DEMO_SXe'
            self.filename = 'DEMO_SX.eqdsk'
            self.targets['inner'] = {'L2D': [1.1]}
            self.targets['outer'] = {'L2D': [3.55]}
            self.firstwall['div_ex'] = 1.75
            self.firstwall['trim'] = [0.63, 0.68]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.2, 0.9]
            self.build['sheild_base'] = 0
            self.targets['inner'] = {'L2D': [1.0], 'dR': 0.15, 'dPlate': 0.25}
            self.targets['outer'] = {'L2D': [5.615], 'dPlate': 1.0}

        elif configuration == 'X':
            self.dataname = 'X'
            self.filename = '/Equil_AR3d1_16coils_v3_bt_1d03li_'
            self.filename += '0d8_Ipl_20d25_XDext_v4_NOEDDY_SOF_FINAL3_v1.eqdsk'
            self.targets['default']['dPlate'] = 0.2  # target plate length
            self.targets['inner'] = {'L2D': [1.1]}
            self.targets['outer'] = {'L2D': [3.55]}
            self.firstwall['div_ex'] = 0.2
            self.firstwall['trim'] = [0.81, 0.81]  # trim fraction (in/out)
            self.coils['external']['id'] = list(range(10, 16))
            self.targets['inner'] = {'L2D': 0.95, 'dR': -1}
            self.targets['outer'] = {'L2D': 1.92, 'graze': 1.0 * np.pi / 180}

        elif configuration == 'Xic':
            self.dataname = 'X'
            self.filename = '/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF_v2.eqdsk'
            self.filename = '/2015_XD_eqdk_2MJ4JW_v2_1.eqdsk'
            self.targets['default']['dPlate'] = 0.2  # target plate length
            self.targets['inner'] = {'L2D': 1.1, 'dR': 0}
            self.targets['outer'] = {'L2D': 2.44}
            self.firstwall['div_ex'] = 0.6
            self.firstwall['trim'] = [0.75, 0.75]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.1, 1]
            self.build['sheild_base'] = 0
            self.coils['internal']['id'] = [11, 12]

        elif configuration == 'SN_old':
            self.dataname = 'SND'
            self.filename = '/2015_SN_eqdk_2MG9A4_v1_0.eqdsk'
            self.filename = '/2015_SN_eqdk_2MG9A4_v1_0_IDM.eqdsk'
            self.targets['inner'] = {'L2D': [1.1]}
            self.targets['outer'] = {'L2D': [3.55]}
            self.firstwall['div_ex'] = 0.25
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.coils['external']['id'] = [0, 4]
            self.targets['inner'] = {'L2D': 0.6}
            self.targets['outer'] = {'L2D': 0.65}

        elif configuration == 'SN2015_SOF':
            self.dataname = 'DEMO_FW'
            self.filename = 'Equil_AR3d1_2015_04_v2_SOF_CSred_fine_fi_2N246U_v1_0.eqdsk'
            self.firstwall['div_ex'] = 0.5
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}
            self.targets['outer'] = {'L2D': 1.36}

        elif configuration == 'SN2015_EOF':
            self.dataname = 'DEMO_FW'
            self.filename = 'Equil_AR3d1_2015_04_v2_EOF_CSred_fine_fi_2NDUNR_v1_0.eqdsk'
            self.firstwall['div_ex'] = 0.75
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}
            self.targets['outer'] = {'L2D': 1.36}

        elif configuration == 'DEMO_SN':
            self.dataname = 'DEMO_SOF'
            self.filename = 'Equil_AR3d1_2015_04_v2_SOF_CSred_fine_final.eqdsk'
            self.firstwall['div_ex'] = 0.75
            # self.firstwall['trim'] = [0.88,0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.1}  # 0.6
            self.targets['outer'] = {'L2D': 1.45}  # 0.65

        elif configuration == 'DEMO_SN_SOF':
            self.dataname = 'DEMO_SOF'
            self.filename = 'Equil_AR3d1_2015_04_v2_SOF_CSred_fine_final.eqdsk'
            self.firstwall['div_ex'] = 0.75
            # self.firstwall['trim'] = [0.88,0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}  # 0.6
            self.targets['outer'] = {'L2D': 1.36}  # 0.65

        elif configuration == 'DEMO_SN_EOF':
            self.dataname = 'DEMO_EOF'
            self.filename = 'Equil_AR3d1_2015_04_v2_EOF_CSred_fine_final.eqdsk'
            self.firstwall['div_ex'] = 0.75
            # self.firstwall['trim'] = [0.88,0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}  # 0.6
            self.targets['outer'] = {'L2D': 1.36}  # 0.65

        elif configuration == 'SN':
            self.dataname = 'DEMO_SN'
            self.filename = 'Equil_AR3d1_2016_01_v1_SN_SOF_baseline_2016_opt.eqdsk'
            self.firstwall['div_ex'] = 0.5  # 0.25
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}  # 0.6
            self.targets['outer'] = {'L2D': 1.1}  # 0.65

        elif configuration == 'SN2017_SOF':
            self.dataname = 'DEMO_SN2014'
            self.filename = 'Equil_2016_DTT1_SN_2017_01_11PF_SOF_Ipl_20d25MA'
            self.filename += '_bt_1d03_li_0d8_R0_8d76m_final.eqdsk'
            self.firstwall['div_ex'] = 0.25  # 0.25
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}  # 0.6
            self.targets['outer'] = {'L2D': 1.36}  # 0.65

        elif configuration == 'SN2017_EOF':
            self.dataname = 'DEMO_SN2014'
            #self.filename = 'Equil_2016_DTT1_SN_2017_01_11PF_EOF_Ipl_20d25MA'
            #self.filename += '_bt_1d03_li_0d8_R0_8d76m_v2.eqdsk'
            self.filename = 'Equil_2016_DTT1_SN_2017_01_11PF_EOF_Ipl_20d25MA'
            self.filename += '_bt_1d03_li_0d8_R0_8d76m_opt_v4.eqdsk'
            self.firstwall['div_ex'] = 0.25  # 0.25
            self.firstwall['trim'] = [0.88, 0.95]  # trim fraction (in/out)
            self.targets['inner'] = {'L2D': 1.0}  # 0.6
            self.targets['outer'] = {'L2D': 1.36}  # 0.65

        elif configuration == 'DTT_SN':
            self.dataname = 'DTT_SN'
            #self.filename = 'Equil_2016_DTT1_SN_2017_01_11PF_EOF_Ipl_20d25MA'
            #self.filename += '_bt_1d03_li_0d8_R0_8d76m_v2.eqdsk'
            #self.filename = 'Equil_AR3d1_SN_v2_2015_09_bt_1d03li_0d8_'
            #self.filename += 'Ipl_20d25_EOF.eqdsk'
            self.filename = '2015_SND_eqdk_2MG9A4_v1_0.eqdsk'
            self.firstwall['div_ex'] = 0.6
            self.targets['inner'] = {'L2D': 1.0}
            self.targets['outer'] = {'L2D': 1.36}

        elif configuration == 'DEMO_SNc':
            self.dataname = 'DEMO_SN'
            self.filename = 'Equil_AR3d1_SN_v2_2015_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            self.firstwall['div_ex'] = 0.25
            self.targets['inner'] = {'L2D': 0.6}
            self.targets['outer'] = {'L2D': 0.65}
            self.targets['default']['dPlate'] = 0.25

        elif configuration == 'DN':
            self.dataname = 'DN'
            self.firstwall['div_ex'] = 0.65
            self.targets['inner'] = {'L2D': 1.0}
            self.targets['outer'] = {'L2D': 1.36}
            self.filename = 'Equil_FT_R0_9d197_k95_1d59_delta95_0d33_final_opt.eqdsk'
        else:
            self.filename = configuration + '.eqdsk'
        self.filename = self.eqdir + self.filename
