from blueprint.postPROCESS import postPROCESS, plot
from nova.config import nova_path
from nova.DEMOxlsx import DEMO

'''
Blnkoth 0.982 OUT.dat: PROCESS output
o   n_TF=16
o   B_t @ R_0 = 4.89 T
o   Winding radial thickness (m)                                             (thkwp)                   4.322E-01  OP
o   Winding width 1 (m)                                                      (wwp1)                    1.409E+00  OP
·
    SN-A31-k165.xslx: Christian’s excel of DEMO
·   SN-A31-k165 Smoothed.xlsx: Sam Merriman’s initial nova output
o   Smoothing of the plasma shape
o   TF coil exclusion coords (I think)
o   2D shape (not structurally solved)
'''

file = nova_path('Inputs', file='blnkoth_0.982_OUT.DAT')
f = open(file, 'r')


#p = postPROCESS(f)
#plot(p)

demo = DEMO()


sep = demo.parts['Plasma']['out']
pl.plot(sep['x'], sep['z'])


demo.fill_part('Vessel')
demo.fill_part('Blanket')
demo.fill_part('Plasma')

AL0SHQST a4