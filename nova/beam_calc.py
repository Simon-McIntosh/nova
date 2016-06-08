from extract import extract_geom
from cross_coil import Fcoil


config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 0.4  # node spaceing [m]
#geom.plasma_coils(I, delta)

Cname = 'P5'

Nseg = 18  # number of DEMO segments
deltaz = 5e-3
E = 200e9  # SS [Pa]
L = 15-geom.coil[Cname]['r']
Fr,Fz = Fcoil(geom.coil, Cname)
W = Fz/Nseg

I = abs(W)*L**3/(3*E*deltaz)
h = 1  # [m]
b = 12*I/h**3
print(b)