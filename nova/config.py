import numpy as np
import collections

class Setup(object):
    
    def __init__(self,configuration=''):
        self.configuration = configuration
        self.set_defaults()
        self.update(configuration)
        self.backfill()

    def set_defaults(self):
        self.filename = ''
        self.dataname = ''
        self.targets = collections.OrderedDict()  # targets structure (ordered)
        self.targets['default'] = {}
        self.targets['default']['L2D'] = 2
        self.targets['default']['open'] = False
        self.targets['default']['graze'] = 1.5*np.pi/180
        self.targets['default']['dPlate'] = 0.5
        self.targets['default']['dR'] = 0
        self.firstwall = {}  # initalise firstwall data structure
        self.firstwall['dRfw'] = 0.25
        self.firstwall['div_ex'] = 0.18
        self.firstwall['trim'] = [0.75,0.7]
        self.build = {}  # initalise build data structure
        self.build['tfw'] = 0.1  # first wall thickness
        self.build['tBBsupport'] = 0.1  # blanket support
        self.build['BS'] = np.array([0.78-0.2,1.304-0.2])  # blanket+sheild #PMI
        self.build['BS'] -= (self.build['tfw']+self.build['tBBsupport'])
        BBfrac = np.array([1,1])
        self.build['BB'] = list(BBfrac*self.build['BS'])  # blanket (in/out) 
        self.build['sheild'] = list((1-BBfrac)*self.build['BS'])  # sheilding
        self.build['sheild_connect']=[0,1]
        self.build['Dsheild'] =[]  # wrap sheild around divertor [0,1] 
        self.build['sheild_base'] = -1  # if Dsheild [], base gap divertor-sheild
        self.build['VV'] = [0.587,1.2]  # vacumn vessel thickness (in/out)
        self.TF = {}
        self.TF['opp'] = 'L'  # L==length, V==volume
        self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},  # coils structure
                      'external':{'id':[],'dR':[],'dZ':[]}}
        
    def backfill(self):
        for key in list(self.targets)[1:]:  # backfill defaults
            for default in list(self.targets['default']):
                if default not in self.targets[key]:
                    self.targets[key][default] = self.targets['default'][default]
    
    def update(self,configuration):  # update 
        self.configuration = configuration
        if configuration == 'SFm':
            self.dataname = 'SFm'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_SFminus_v4_2015'+\
            '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            #self.filename = '../eqdsk/2015_SFminus_eqdsk_2MCQRZ_v1_0_IDM.eqdsk'
            self.targets['default']['dPlate'] = 0.35 # target plate length
            self.targets['inner1'] = {'L2D':[1.1+0.52],'open':True,'dR':0}
            self.targets['inner2'] = {'L2D':[1.2-0.08],'open':False,'dR':-1}
            self.targets['outer1'] = {'L2D':[1.65-0.7],'open':False,'dR':-1}
            self.targets['outer2'] = {'L2D':[1.1+0.15],'open':True,'dR':0}  
            self.firstwall['div_ex'] = 0.18
            self.firstwall['trim'] = [0.75,0.7]  # trim fraction (in/out)
            self.coils['external']['id'] = list(range(0,16))  # all external
            self.TF['opp'] = 'L'
            
        elif configuration == 'SFp':
            self.dataname = 'SFp'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_SFplus_v4_2015'+\
            '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            self.targets['default']['dPlate'] = 0.35 # target plate length
            self.targets['inner1'] = {'L2D':[1.1],'open':True,'dR':0.1}
            self.targets['inner2'] = {'L2D':[1.2],'open':False,'dR':-1}
            self.targets['outer1'] = {'L2D':[1.65],'open':False,'dR':-1}
            self.targets['outer2'] = {'L2D':[1.1],'open':True,'dR':0.1}  
            self.firstwall['div_ex'] = 0.18
            self.firstwall['trim'] = [0.75,0.7]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.22,1]
            self.TF['opp'] = 'L'
            self.coils['external']['id'] = list(range(0,10)) 
            
        if configuration == 'SX':
            self.dataname = 'SX8'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3.eqdsk'
            #self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            #self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3_new.eqdsk'
            self.filename = '../eqdsk/2015_SX_ext_coils_eqdk_2MK6XX_v1_0.eqdsk'  # IDM
            #self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_Ipl_'
            #self.filename += '20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter3_v5_old.eqdsk'
            self.targets['inner'] = {'L2D':[1.1]}
            self.targets['outer'] = {'L2D':[3.55]} 
            self.firstwall['div_ex'] = 1
            self.firstwall['trim'] = [0.71,0.74]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.15,0.9]
            self.coils['external']['id'] = list(range(10,16)) 
        
        elif configuration == 'SXex':
            self.dataname = 'SXex'
            self.filename = '../eqdsk/SXex.eqdsk'
            self.targets['inner'] = {'L2D':[1.1]}
            self.targets['outer'] = {'L2D':[3.55]} 
            self.firstwall['div_ex'] = 1.3
            self.firstwall['trim'] = [0.63,0.68]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.2,0.9]
            self.build['sheild_base'] = 0
            self.targets['inner'] = {'L2D':[1.0],'dR':0.15,'dPlate':0.25}
            self.targets['outer'] = {'L2D':[5.615],'dPlate':1.0}
                
        elif configuration == 'X':
            self.dataname = 'X'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_v3_bt_1d03li_'
            self.filename += '0d8_Ipl_20d25_XDext_v4_NOEDDY_SOF_FINAL3_v1.eqdsk'
            self.targets['default']['dPlate'] = 0.2  # target plate length
            self.targets['inner'] = {'L2D':[1.1]}
            self.targets['outer'] = {'L2D':[3.55]} 
            self.firstwall['div_ex'] = 0.2
            self.firstwall['trim'] = [0.81,0.81]  # trim fraction (in/out)
            self.coils['external']['id'] = list(range(10,16)) 
            self.targets['inner'] = {'L2D':0.95,'dR':-1}
            self.targets['outer'] = {'L2D':1.92,'graze':1.0*np.pi/180}

        
        elif configuration == 'Xic':
            self.dataname = 'X'
            self.filename = '../eqdsk/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF_v2.eqdsk'
            self.filename = '../eqdsk/2015_XD_eqdk_2MJ4JW_v2_1.eqdsk'
            self.targets['default']['dPlate'] = 0.2  # target plate length
            self.targets['inner'] = {'L2D':1.1,'dR':0}
            self.targets['outer'] = {'L2D':2.44}
            self.firstwall['div_ex'] = 0.6
            self.firstwall['trim'] = [0.75,0.75]  # trim fraction (in/out)
            self.build['sheild_connect'] = [0.1,1]
            self.build['sheild_base'] = 0
            self.coils['internal']['id'] = [11,12] 
                                   
        elif configuration == 'SN':
            self.dataname = 'SND'
            self.filename = '../eqdsk/2015_SN_eqdk_2MG9A4_v1_0.eqdsk'
            self.filename = '../eqdsk/2015_SN_eqdk_2MG9A4_v1_0_IDM.eqdsk'
            self.targets['inner'] = {'L2D':[1.1]}
            self.targets['outer'] = {'L2D':[3.55]} 
            self.firstwall['div_ex'] = 0.25
            self.firstwall['trim'] = [0.88,0.95]  # trim fraction (in/out)
            self.coils['external']['id'] = [0,4]
            self.targets['inner'] = {'L2D':0.6}
            self.targets['outer'] = {'L2D':0.65}

                