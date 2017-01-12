import pylab as pl
from nova.config import select
from nova.streamfunction import SF
from nova.radial_build import RB
from nova.coils import PF
import numpy as np
import amigo.geom as geom
from nova.DEMOxlsx import DEMO
from amigo.IO import trim_dir
from openpyxl import load_workbook
from nova.DEMOxlsx import cluster_points

import seaborn as sns
rc = {'figure.figsize':[7,7*16/12],'savefig.dpi':125, # 
      'savefig.jpeg_quality':200,'savefig.pad_inches':0.1,
      'lines.linewidth':1.5}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2',5)


config,setup = select(base={'TF':'dtt','eq':'DEMO_FW_SOF'},nTF=18)
sf = SF(setup.filename)  
pf = PF(sf.eqdsk)
pf.plot(coils=pf.coil,label=True,plasma=False,current=True) 
levels = sf.contour(plot_vac=False)
rb = RB(setup,sf)
rb.firstwall(plot=True,debug=False,color=color[1])
rb.trim_sol()


filename = trim_dir('../../Data/')+'CATIA_FW.xlsx'
wb = load_workbook(filename=filename,read_only=True,data_only=True)
ws = wb[wb.get_sheet_names()[0]]

FW = {}
for col,var in zip([5,6],['r','z']):
    row = ws.columns[col]
    FW[var] = np.zeros(len(row)-1)
    for i,r in enumerate(row[1:]):
        try:
            FW[var][i] = 1e-3*float(r.value)  # m
        except:
            break
r,z = geom.pointloop(FW['r'],FW['z'],ref='min')
pl.plot(r[:-1],z[:-1])

       
'''
self.parts = OrderedDict()  # component parts    
part,loop = [],[]
for row in ws.columns:
    new_part = get_label(row[0].value,part)
    if new_part:
        self.parts[part[-1]] = OrderedDict()
    if len(part) == 0:
        continue
    new_loop = get_label(row[1].value,loop,force=new_part,part=part[-1])
    p,l = part[-1],loop[-1]
    if new_loop:
        self.parts[p][l] = OrderedDict()
'''
                
'''
config,setup = select(base={'TF':'dtt','eq':'DEMO_FW_EOF'},nTF=18)
sf = SF(setup.filename)  
sf.contour(plot_vac=False,levels=levels)
rb = RB(setup,sf)
rb.firstwall(plot=True,debug=False)
rb.trim_sol()
'''



'''
nTF = 18#config['nTF']
profile = Profile(config['TF'],family='S',part='TF',
                  nTF=nTF,obj='L',load=True)


shp = Shape(profile,nTF=nTF,obj='L',eqconf=config['eq'],ny=3)
#shp.add_vessel(rb.segment['vessel_outer'])
#shp.loop.set_l({'value':0.8,'lb':0.75,'ub':1.8})  # 1/tesion
#shp.loop.xo['lower'] = {'value':0.7,'lb':0.5,'ub':1}  # vertical shift
#shp.minimise(ripple=True,verbose=True)
shp.update()
tf = TF(profile,sf=sf)
tf.fill()



demo = DEMO()
demo.fill_part('Blanket',alpha=1)
demo.fill_part('Vessel',alpha=1)
demo.fill_part('TF_Coil',alpha=1)

pl.tight_layout()


sf.eqwrite(pf,config=config['eq'],CREATE=True)

pkl = PKL(config['eq'],directory='../../Movies/')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])
for i,(flux,S) in enumerate(zip(inv.swing['flux'],['SOF','MOF','EOF'])):
    inv.fix_flux(flux)  # swing
    inv.solve_slsqp()
    inv.eq.gen_opp()
    sf.eqwrite(inv.eq.pf,config=config['eq']+'_{}'.format(S),CREATE=True)


data = {}
for loop in ['first_wall','blanket','vessel_inner','vessel_outer']:
    data[loop] = {}
    for var in rb.segment[loop]:
        data[loop][var] = list(rb.segment[loop][var])
for loop,label in zip(['in','out'],['TF_inner','TF_outer']):
    data[label] = {}
    for var in tf.x[loop]:
        data[label][var] = list(tf.x[loop][var])
datadir = trim_dir('../../../Data/') 
with open(datadir+'{}.json'.format(config['eq']),'w') as f:
    json.dump(data,f,indent=4)

'''