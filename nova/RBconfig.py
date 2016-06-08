import numpy as np
import collections


def eqconfig(config=''):

        name = {}
        name['file'] = ''
        name['data'] = ''
        
        targets = collections.OrderedDict()  # targets data structure (ordered)
        targets['default'] = {}
        targets['default']['L2D'] = 2
        targets['default']['open'] = False
        targets['default']['graze'] = 1.5*np.pi/180
        targets['default']['dPlate'] = 0.5
        targets['default']['dR'] = 0.1

        firstwall = {}  # initalise firstwall data structure
        firstwall['dRfw'] = 0.25
        firstwall['div_ex'] = 0.18
        firstwall['trim'] = [0.75,0.7]
        
        coils = {'internal':{'id':[],'dR':[],'dZ':[]},  # coils data structure
                 'external':{'id':list(range(0,16)),'dR':[],'dZ':[]}}
                
        if config == 'SFm':
            name['file'] = '../eqdsk/Equil_AR3d1_16coils_SFminus_v4_2015'+\
            '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'
            #name['file'] = '../eqdsk/2015_SFminus_eqdsk_2MCQRZ_v1_0_IDM.eqdsk'
            name['data'] = 'SFm'
            
            targets['default']['dPlate'] = 0.35 # target plate length
            targets['inner1'] = {'L2D':[1.1+0.52],'open':True,'dR':0.0}
            targets['inner2'] = {'L2D':[1.2-0.08],'open':False,'dR':-1}
            targets['outer1'] = {'L2D':[1.65-0.7],'open':False,'dR':-1}
            targets['outer2'] = {'L2D':[1.1+0.15],'open':True,'dR':0.0}  
            
            firstwall['div_ex'] = 0.18
            firstwall['trim'] = [0.75,0.7]  # trim fraction (in/out)
            
            coils['external']['id'] = list(range(0,16))
            '''
            self.TFopp = 'V'
            self.sheild_connect=[0,1]  # 0.3,0.95
            #self.Dsheild =[0,0.84]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            '''

            coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(0,16)),'dR':[],'dZ':[]}} 
                     

        for key in list(targets)[1:]:
            print(key)
        return name,targets,firstwall
        
        
'''
                             
        
    
        self.conformal = True
        self.dSOL=0.005  # [m]
        self.Nsol=51
        self.dRfw = 0.25  # [m]
        #self.div_ex = 0.3
        self.TFopp = 'V'  # TF shape optimisation 'L'==length, 'V'==volume
        self.sheild_base = -1
        self.tfw = 0.1  # first wall thickness
        self.tBBsupport = 0.1  # blanket support
        #self.BS = np.array([1.075,1.56])  # blanket+sheild (in/out)
        self.BS = np.array([0.78-0.2,1.304-0.2])  # blanket+sheild (in/out) #PMI xls
        self.BS -= (self.tfw+self.tBBsupport)
        self.BBfrac = np.array([0.8,0.65])
        self.BBfrac = np.array([1,1])
        self.BB = list(self.BBfrac*self.BS)  # breeding blanket thickness (in/out)
        self.sheild = list((1-self.BBfrac)*self.BS)  # neutron sheilding
        #self.sheild = [0.2,0.6]
        #self.VV = 0.32  # vacumn vessel thickness (in/out)
        self.VV = [0.587,1.2]  # vacumn vessel thickness (in/out)
        
        self.graze = 1.5*np.pi/180  # toroidal grazing angle
        self.dPlate = 0.2  # target plate length
        self.nTF = 18  # number of TF coils
        self.Jmax = 30e6  # 15e6  # max current density
        self.trim = [0.93,0.81]  # trim fraction (in/out)
        self.config = config
        
        self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                      'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
        self.targets = collections.OrderedDict()
        
        self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.1}
        self.targets['outer'] = {'L2D':[3.55],'open':False,'graze':self.graze,'dR':0.1}
            
                
        if config == 'vde':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 1
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.15,0.9]
            self.sheild_base = -1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)
            self.filename = '../eqdsk/vde.eqdsk'
            self.dataname = 'SX7'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                          'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[3.55],'open':False,'graze':self.graze,'dR':0.1}
            
        if config == 'SX8_IDM':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 1
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.15,0.9]
            self.sheild_base = -1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)

            self.filename = '../eqdsk/2015_SX_ext_coils_eqdk_2MK6XX_v1_0.eqdsk'  # IDM

            self.dataname = 'SX8_IDM'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,
                                     'graze':self.graze,'dR':0}
            self.targets['outer'] = {'L2D':[3.55],'open':False,
                                     'graze':self.graze,'dR':0} 
                                     
        if config == 'SX8':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 1
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.15,0.9]
            self.sheild_base = -1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3.eqdsk'
            
            #self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            #self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v3_new.eqdsk'
            self.filename = '../eqdsk/2015_SX_ext_coils_eqdk_2MK6XX_v1_0.eqdsk'  # IDM
            #self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_Ipl_'
            #self.filename += '20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter3_v5_old.eqdsk'

            self.dataname = 'SX8'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,
                                     'graze':self.graze,'dR':0}
            self.targets['outer'] = {'L2D':[3.55],'open':False,
                                     'graze':self.graze,'dR':0}   

        if config == 'SX7':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 1
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.15,0.9]
            self.sheild_base = -1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter5_v2.eqdsk'
            self.dataname = 'SX7'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[3.55],'open':False,'graze':self.graze,'dR':0.1}

        if config == 'SX6':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 1
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.15,0.9]
            self.sheild_base = -1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter4_v3.eqdsk'
            self.dataname = 'SX6'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(10,16)),'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[3.55],'open':False,'graze':self.graze,'dR':0.1}


        if config == 'SX5':
            self.dRfw = 0.25  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 0.3
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.2,1]
            self.sheild_base = 0.1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.71,0.74]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter3_v5.eqdsk'
            self.dataname = 'SX5'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':[],'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.105}
            self.targets['outer'] = {'L2D':[4.15],'open':False,'graze':self.graze,'dR':0.105}

        elif config == 'SX4':
            self.dRfw = 0.4  # [m]
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 0.1
            self.dPlate = 0.25  # target plate length
            self.sheild_connect=[0.25,1]
            self.sheild_base = 0.1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.715,0.72]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_iter2.eqdsk'
            self.dataname = 'SX4'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':[],'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.1],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[4.15],'open':False,'graze':self.graze,'dR':0.8}

        elif config == 'SX3':
            self.TFopp = 'L'
            self.inflate = 0
            self.div_ex = 0.35
            self.dPlate = 0.2  # target plate length
            self.sheild_connect=[0.3,1]
            self.sheild_base = 0.1
            self.Dsheild =[0,1]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.76,0.72]  # trim fraction (in/out)
            self.filename = '../eqdsk/Equil_AR3d1_16coils_bt_1d03li_0d8_I'
            self.filename += 'pl_20d25_SX_on_SFgeom_NOEDDY_EOF_fine_v2.eqdsk'

            self.dataname = 'SX3'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':[],'dR':[],'dZ':[]}}

            self.targets['inner'] = {'L2D':[1.5],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[3],'open':False,'graze':self.graze,'dR':0.3}

        elif config == 'SX2':
            self.TFopp = 'L'
            self.inflate = 0.125
            self.div_ex = 0.2
            self.dPlate = 0.4  # target plate length
            self.sheild_connect=[0.25,0.9]
            self.sheild_base = 0.1
            self.Dsheild =[0.045,0.68]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.58,0.48]  # trim fraction (in/out)
            self.filename = '../eqdsk/redtosx4_1.eqdsk'
            self.dataname = 'SX2'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':[],'dR':[],'dZ':[]}}
            if inside:
                self.dataname += 'inside'
                self.targets['inner'] = {'L2D':[3],'open':False,'graze':self.graze}
                self.targets['outer'] = {'L2D':[5.5],'open':True,'graze':self.graze}
            else:
                self.targets['inner'] = {'L2D':[0.7],'open':False,'graze':self.graze,'dR':0.1}
                self.targets['outer'] = {'L2D':[6.45],'open':False,'graze':self.graze,'dR':0.3}

        elif config == 'SX':
            self.inflate = 0.2
            self.TFopp = 'V'
            self.div_ex = 2
            self.dPlate = 0.5  # target plate length
            self.sheild_connect=[0.25,0.9]
            self.Dsheild =[0.04,0.58]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.62,0.59]  # trim fraction (in/out)
            self.filename = '../eqdsk/Super-X_equilibrium_2MC43F_v1_0.eqdsk'
            self.dataname = 'SX'
            #self.coils = {'internal':{'id':[11,12,12],'dR':[0.6],'dZ':[]},
            #         'external':{'id':list(range(0,11))+[13],'dR':[],'dZ':[]}}
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                          'external':{'id':[],'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.4],'open':False,
                                     'graze':self.graze,'dR':0}
            self.targets['outer'] = {'L2D':[5.2],'open':False,
                                     'graze':self.graze,'dR':0}
                
        elif config == 'SXex':
            self.inflate = 0
            self.TFopp = 'L'
            self.div_ex = 1.3
            #self.dPlate = 5.5  # target plate length
            self.sheild_connect=[0.2,0.9]
            self.Dsheild =[]  # inner outer 0-1
            self.sheild_base = 0  # 0.25
            self.trim = [0.63,0.68]  # trim fraction (in/out)
            #self.filename = '../eqdsk/Super-X_equilibrium_2MC43F_v1_0.eqdsk'
            self.filename = './plot_data/SXex.eqdsk'
            self.dataname = 'SXex'
            #self.coils = {'internal':{'id':[11,12,12],'dR':[0.6],'dZ':[]},
            #         'external':{'id':list(range(0,11))+[13],'dR':[],'dZ':[]}}
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                          'external':{'id':[],'dR':[],'dZ':[]}}
            self.targets['inner'] = {'L2D':[1.0],'open':False,
                                     'graze':self.graze,'dR':0.15,
                                     'dPlate':0.25}
            self.targets['outer'] = {'L2D':[5.615],'open':False,  # 5.607
                                     'graze':self.graze,'dR':0,
                                     'dPlate':1.0}  # 7.5
                
        elif config == 'X':
            self.dPlate = 0.2  # target plate length
            self.div_ex = 0.2
            self.Dsheild =[0.12,0.6]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            #self.trim = [0.62,0.65]  # trim fraction (in/out)
            self.trim = [0.81,0.81]  # trim fraction (in/out)
            self.sheild_connect=[0,1]
            self.sheild_base = -1
            #self.filename = '../eqdsk/Equil_AR3d1_bt_1d03li_0d8_Ipl_'+\
            #'20d25_XD_N_2LAT7R_v1_0.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF_v2.eqdsk'
            self.filename = '../eqdsk/2015_XD_eqdk_2MJ4JW_v2_1.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_v3_bt_'
            self.filename += '1d03li_0d8_Ipl_20d25_SF_NOEDDY_SOF_FINAL3_v9.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_v3_bt_1d03li_'
            self.filename += '0d8_Ipl_20d25_SF_NOEDDY_SOF_FINAL3_v12.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_v3_bt_1d03li_0d8_'
            self.filename += 'Ipl_20d25_XDext_NOEDDY_SOF_FINAL3_v11.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_16coils_v3_bt_1d03li_'
            self.filename += '0d8_Ipl_20d25_XDext_v4_NOEDDY_SOF_FINAL3_v1.eqdsk'
            self.dataname = 'X'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},  # 12  0.85*2
                     'external':{'id':range(10,16),'dR':[],'dZ':[]}}  # list(range(0,13))
            self.targets['inner'] = {'L2D':[0.95],'open':False,  #0.85
                                     'graze':self.graze,'dR':-1}
            self.targets['outer'] = {'L2D':[1.92],'open':False,  # 1.9,1.95
                                     'graze':1.0*np.pi/180,'dR':0} #2.433 386
                                     
                                     
                                     
        elif config == 'Xex':
            self.dPlate = 0.2  # target plate length
            self.div_ex = 0.1
            self.Dsheild =[0.12,0.6]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            #self.trim = [0.62,0.65]  # trim fraction (in/out)
            self.trim = [0.78,0.7]  # trim fraction (in/out)
            self.sheild_connect=[0,1]
            self.sheild_base = 0.05
            #self.filename = '../eqdsk/Equil_AR3d1_bt_1d03li_0d8_Ipl_'+\
            #'20d25_XD_N_2LAT7R_v1_0.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF.eqdsk'
            self.filename = '../eqdsk/Equil_AR3d1_XD_2015_Invess_5d0MA_EOF_v2.eqdsk'
            self.filename = '../eqdsk/2015_XD_eqdk_2MJ4JW_v2_1.eqdsk'
            self.dataname = 'X'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},  # 12  0.85*2
                     'external':{'id':[],'dR':[],'dZ':[]}}  # list(range(0,13))
            self.targets['inner'] = {'L2D':[0.8],'open':False,'graze':self.graze,'dR':0.1}
            self.targets['outer'] = {'L2D':[3.12],'open':False,'graze':self.graze,'dR':0.11} #2.433
                        
        elif config == 'SN':
            self.TFopp = 'L'
            self.dPlate = 0.5  # target plate length
            self.div_ex = 0.25
            self.trim = [0.88,0.95]  # trim fraction (in/out)
            self.Dsheild =[0.2,0.55]  # inner outer 0-1
            self.Dsheild =[]
            self.sheild_connect=[0,1]
            self.sheild_base = -1
            #self.filename = '../eqdsk/Equil_AR3d1_bt_1d03li_0d8_Ipl'+\
            #'_20d25_EOF_2M56U8_v1_0.eqdsk'
            self.filename = '../eqdsk/2015_SN_eqdk_2MG9A4_v1_0.eqdsk'
            self.filename = '../eqdsk/2015_SN_eqdk_2MG9A4_v1_0_IDM.eqdsk'
            self.dataname = 'SND'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':[0,4],'dR':[],'dZ':[]}}  #list(range(0,11))
            self.targets['inner'] = {'L2D':[0.6],'open':False,
                                     'graze':self.graze,'dR':0.0}
            self.targets['outer'] = {'L2D':[0.65],'open':False,
                                     'graze':self.graze,'dR':0.0}
        
        elif config == 'SF':
            self.filename = '../eqdsk/Equil_AR3d1_bt_1d03li_0d8_Ipl_'+\
            '20d25_26co_2L2R9C_v1_0.eqdsk'
            self.dataname = 'SFD'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(0,10)),'dR':[],'dZ':[]}} 
            self.targets['inner1'] = {'L2D':[2],'open':False,'graze':self.graze}
            self.targets['inner2'] = {'L2D':[2],'open':False,'graze':self.graze}
            self.targets['outer1'] = {'L2D':[2],'open':False,'graze':self.graze}
            self.targets['outer2'] = {'L2D':[2],'open':False,'graze':self.graze} 
        


        elif config == 'SFp':
            self.TFopp = 'V'
            self.div_ex = 0.18
            self.dPlate = 0.35  # target plate length
            self.sheild_connect=[0.22,1]
            #self.Dsheild =[0,0.84]  # inner outer 0-1
            self.Dsheild =[]  # inner outer 0-1
            self.trim = [0.75,0.75]  # trim fraction (in/out)
            
            self.filename = '../eqdsk/Equil_AR3d1_16coils_SFplus_v4_2015'+\
            '_09_bt_1d03li_0d8_Ipl_20d25_SOF.eqdsk'

            self.dataname = 'SFp'
            self.coils = {'internal':{'id':[],'dR':[],'dZ':[]},
                     'external':{'id':list(range(0,10)),'dR':[],'dZ':[]}} 
            self.targets['inner1'] = {'L2D':[1.1],'open':True,'graze':self.graze,'dR':0.1}
            self.targets['inner2'] = {'L2D':[1.2],'open':False,'graze':self.graze,'dR':-1}
            self.targets['outer1'] = {'L2D':[1.65],'open':False,'graze':self.graze,'dR':-1}
            self.targets['outer2'] = {'L2D':[1.1],'open':True,'graze':self.graze,'dR':0.1}  
                    
        keys = ['conformal','dSOL','Nsol','dRfw','div_ex','TFopp',
                'sheild_base','tfw','tBBsupport','BS','BBfrac','BB',
                'sheild','VV','graze','dPlate','nTF','Jmax','trim',
                'config','targets']
        config = {}
        for key in keys:  # package as dict
            config[key] = getattr(self,key)
        print(config)
        return config         
'''        