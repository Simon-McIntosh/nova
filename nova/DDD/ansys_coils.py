import shelve
import math

path = './coil_data/'
file = 'Double_Decker_Coils'

ansys_file = 'DDcoils.txt'
f = open(path+ansys_file, 'w')  # open file

coil_names = ['P1','P2','P3','P4','PS1','P6C','PS3','PS4','CS3LB',
              'P6','P5','PS2','PS5','P5B','P6B']
coil_current = {'P1':25.19, 'P2':-14.06, 'P3':6.33, 'P4':-15.45, 'P6':10.8, 
               'P5':-4.8, 'P6B':-0.1, 'CS3LB':-30.32, 'P5B':0.49, 'PS1':6.33, 
               'PS2':1.06, 'PS3':0.28, 'PS4':13.03, 'PS5':1.93, 'P6C':-3.42}
 
Jmax = 2e7  # [A/m^2]

data = shelve.open(path+file+'_'+'coils')
for key, name in zip(range(1,16), coil_names):
    key = 'data-'+str(key)
    D = 2*(abs(coil_current[name])*1e6/(Jmax*math.pi))**0.5
    f.write(name+'_x = '+'{:1.4f}'.format(data[key][0,0]*1e3)+'\n') 
    f.write(name+'_y = '+'{:1.4f}'.format((data[key][0,1]+15)*1e3)+'\n')
    f.write(name+'_D = '+'{:1.4f}'.format(D*1e3)+'\n')
    f.write('\n')
data.close()

divertor_names = ['TL', 'TU']
edge = ['min', 'max']
data = shelve.open(path+file+'_'+'divertor')
for key, name in zip(data.keys(), divertor_names):
    for i, loc in enumerate(edge):
        f.write(name+'_x_'+loc+' = '+'{:1.4f}'.format(data[key][i,0]*1e3)+'\n') 
        f.write(name+'_y_'+loc+' = '+'{:1.4f}'.format((data[key][i,1]+15)*1e3)+'\n')
    f.write('\n')
data.close()


f.close()




'''
P1_D = 100
P1_x = 6000
P1_y = 20500
P2_D = 100
P2_x = 14000
P2_y = 18000
P3_D = 100
'''