import numpy as np
import pylab as pl
import pickle
from Jscript_functions import JSfunc
from radial_build import RB
from feild_calc import solCalc

config = 'DD3'
with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)

with open('./plot_data/'+config+'_Supports.pkl', 'rb') as input:
    P6support = pickle.load(input)
    P5support = pickle.load(input)
    

    
f = open('JSfiles/DD.js', 'w')
JS = JSfunc(f)
funcID = {'coil':[],'pancake':[],
                       'case':[],'loop':[],
                       'insulation':[]}
skID = {'coil':[],'pancake':[],
                       'case':[],'loop':[],
                       'insulation':[]}
psID = {'loop':[]}
XYbp = JS.get_plane('XY')

rb = RB(geom,sf,config,Np=400)
plot.set_keys(['P6','P5','PS2','P6B','P5B','PS2','PS5'])  # internal coils

#rb.internal_coils(['P6','P5','PS2','P6B','P5B','PS2','PS5'])

funcID['coil'].append(JS.open_function())  # coils
skID['coil'].append(JS.start_sketch(name='coil'))
for name in plot.coil_keys: 
    JS.square(rb.geom.coil[name]['r'],rb.geom.coil[name]['z'],
              rb.geom.coil[name]['rc'])
JS.end_sketch()
JS.close_function() 
JS.sketch(XYbp,funcID['coil'][-1])

funcID['coil'].append(JS.open_function())  # jackets
skID['coil'].append(JS.start_sketch(name='jackets'))
for name in plot.coil_keys: 
    a = np.sqrt(np.pi)*rb.geom.coil[name]['rc']
    if name is 'PS5':
        b = 0.15
    else:
        b=0
    JS.rectangle(rb.geom.coil[name]['r'],rb.geom.coil[name]['z'],
              [a,a+0.1])
    JS.rectangle(rb.geom.coil[name]['r'],rb.geom.coil[name]['z'],
              [a+b+0.15,a+0.1+0.15])
JS.end_sketch()
JS.close_function() 
JS.sketch(XYbp,funcID['coil'][-1])

funcID['coil'].append(JS.open_function())  # cut
skID['coil'].append(JS.start_sketch(name='cut'))
for name in plot.coil_keys: 
    a = np.sqrt(np.pi)*rb.geom.coil[name]['rc']
    if name is 'PS5':
        b = 0.15
    else:
        b=0
    JS.rectangle(rb.geom.coil[name]['r'],rb.geom.coil[name]['z'],
              [a+b+0.15,a+0.1+0.15])
JS.end_sketch()
JS.close_function() 
JS.sketch(XYbp,funcID['coil'][-1])

funcID['coil'].append(JS.open_function())  # spar
skID['coil'].append(JS.start_sketch(name='Spars'))
for ID,support in zip(['P6','P5'],[P6support,P5support]):
    for level in ['top','bottom']:
        JS.spline(skID['coil'][-1],support[level]['rI'],support[level]['zI'])
    for ends in [0,-1]:
        points = [[support['top']['rI'][ends],support['top']['zI'][ends]],
                  [support['bottom']['rI'][ends],support['bottom']['zI'][ends]]]
        JS.line(points)
JS.end_sketch()
JS.close_function() 
JS.sketch(XYbp,funcID['coil'][-1])

funcID['coil'].append(JS.open_function())  # caps
skID['coil'].append(JS.start_sketch(name='Caps'))
for ID,support in zip(['P6','P5'],[P6support,P5support]):
    for level in ['top','bottom']:
        JS.spline(skID['coil'][-1],support[level]['r'],support[level]['z'])
        JS.spline(skID['coil'][-1],support[level]['rI'],support[level]['zI'])
        for ends in [0,-1]:
            points = [[support[level]['r'][ends],support[level]['z'][ends]],
                  [support[level]['rI'][ends],support[level]['zI'][ends]]]
            JS.line(points)    
JS.end_sketch()
JS.close_function() 
JS.sketch(XYbp,funcID['coil'][-1])



JS.end_script()