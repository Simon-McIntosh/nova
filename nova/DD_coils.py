from png_tools import data_mine

file = 'DD3'  # DD1,DD2,DD3
path = './config_data/'+file+'/'

coil = []
if file == 'DD1':
    coil = ['CS3L','CS3LB','CS2L','CS1','CS2U','CS3U',
            'P1','P2','P3','P4','P5','P5B','P6','P6B']
    I = {'CS3U':1.26, 'CS2U':-2.25, 'CS1':-41.73, 'CS2L':2.65,
         'CS3L':-0.34, 'P1':5.79, 'P2':-7.77, 'P3':1.30,
         'P4':-10.91, 'P6':6.18, 'P5':-3.79, 'P6B':0.65,
         'P5B':3.78, 'CS3LB':-3.96}  # negated CS3LB coil
    xlim = [4,16]
    ylim = [-10,8]
elif file == 'DD2':
    coil = ['CS3L','CS3LB','CS2L','CS1','CS2U','CS3U',
            'P1','P2','P3','P4','P5','P5B','P6','P6B','P6C']
    I = {'CS3U':1.26, 'CS2U':-2.25, 'CS1':-41.73, 'CS2L':1.53,
         'CS3L':-0.12, 'P1':5.28, 'P2':-7.52, 'P3':0.98,
         'P4':-10.69, 'P6':6.6, 'P5':-3.66, 'P6B':0.12,
         'P5B':3.23, 'CS3LB':-4.52, 'P6C':1.08}  # negated CS3LB coil
    xlim = [4,16]
    ylim = [-10,8]    
elif file == 'DD3':
    coil = ['CS3L','CS3LB','CS2L','CS1','CS2U','CS3U',
            'P1','P2','P3','P4','P5','P5B','P6','P6B','P6C',
            'PS1', 'PS2','PS3','PS4', 'PS5']
    I = {'CS2U':3.24, 'CS3LB':-21.83, 'P4':-15.03, 'P1':34.42,
         'PS3':1.97, 'P3':6.41, 'P2':-12.8, 'P5':-4.79,
         'CS3U':-0.73, 'P6':10.77, 'CS3L':29.54, 'PS1':6.91,
         'PS4':16.18, 'CS1':6.78, 'PS5':1.95, 'CS2L':0.67, 'P5B':0.5,
         'PS2':1.05, 'P6B':-0.12, 'P6C':-3.03}
    xlim = [5,15]
    ylim = [-12,8] 

f = open(path+'coilID.vi', 'w')
for name in coil:
    f.write(name+'\n')
f.write('\n')
f.close()

f = open(path+'current.vi', 'w')
for name in sorted(I.keys()):
    f.write(name+' '+str(I[name])+'\n')
f.write('\n')
f.close()

'''
data_ID = 'plasma'  # sol-LCFS-centre
data_mine(path, file, data_ID, xlim, ylim)

data_ID = 'coils'  # specify order
data_mine(path, file, data_ID, xlim, ylim)

data_ID = 'structure'  # TF coils inner-outer, divertor (optional)
data_mine(path, file, data_ID, xlim, ylim)
'''