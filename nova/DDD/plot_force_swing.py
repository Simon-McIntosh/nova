import pylab as pl
import csv
path = './coil_data/'

swing = 1
force_file = 'DDswing_'+str(swing)+'.txt'
force = {}

f = open(path+force_file)  
data = csv.reader(f, delimiter=' ')
for row in data:
    if len(row) > 0:
        if 'Fx' in row[0]: 
            name = row[0].replace('_Fx','')
            force[name] = {}
            force[name]['Fx'] = []
            force[name]['Fz'] = []
            
f.close()

for swing in range(1,5):
    force_file = 'DDswing_'+str(swing)+'.txt'
    f = open(path+force_file)  
    data = csv.reader(f, delimiter=' ')
    for row in data:
        if len(row) > 0:
            if 'Fx' in row[0]: 
                name = row[0].replace('_Fx','')
                force[name]['Fx'].append(float(row[2]))
            elif 'Fz' in row[0]:
                name = row[0].replace('_Fz','')
                force[name]['Fz'].append(float(row[2]))

pl.figure(figsize=(9,6))               
coils = ['P6','P5','PS2','PS5','P5B','P6B']            
for coil in coils:
    pl.plot(force[coil]['Fz']) 
pl.ylim([-0.7e8,1.2e8])

          
