import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from amigo.addtext import linelabel
from read_dina import timeconstant

path = os.path.join(class_dir(nep), '../Data/LTC/')

# data_mine(path, 'VS3_discharge', [10, 10.1], [0, 80e3])
# data_mine(path, 'VS3_discharge_main_report', [10, 10.125], [0, 70e3])


points = data_load(path, 'VS3_discharge', date='2018_02_28')[0]
# points = data_load(path, 'VS3_discharge_main_report', date='2018_06_25')[0]

to, Io = points[0]['x'], points[0]['y']  # bare conductor
td, Id = points[1]['x'], points[1]['y']  # jacket + vessel

tc = timeconstant(to, Io, trim_fraction=0)

Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(1)  # bare single fit
tc.load(td, Id)  # replace with coupled discharge curve
Io_d, tau_d, tfit_d, Ifit_d = tc.nfit(3)

plt.plot(1e3*(to-to[0]), 1e-3*Io, 'C0-', label='bare conductor')
txt_o = timeconstant.ntxt(Io_o/Io[0], tau_o)
plt.plot(1e3*(tfit_o-to[0]), 1e-3*Ifit_o, 'C1--', label='exp fit '+txt_o)
plt.plot(1e3*(td-td[0]), 1e-3*Id, 'C2-',
         label='conductor + passive structures')
txt_d = timeconstant.ntxt(Io_d/Id[0], tau_d)
plt.plot(1e3*(tfit_d-td[0]), 1e-3*Ifit_d, 'C3--', label='exp fit '+txt_d)
plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I$ kA')
plt.legend()

plt.figure()
points = data_load(path, 'VS3_discharge', date='2018_02_28')[0]
to, Io = points[0]['x'], points[0]['y']  # bare conductor
tc = timeconstant(to, Io, trim_fraction=0)
Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(1)  # bare single fit
txt_o = timeconstant.ntxt(Io_o/Io[0], tau_o)
plt.plot(1e3*(to-to[0]), 1e-3*Io, 'C0-', label='28_02 ' + txt_o)

points = data_load(path, 'VS3_discharge_main_report', date='2018_06_25')[0]
to, Io = points[0]['x'], points[0]['y']  # bare conductor
tc = timeconstant(to, Io, trim_fraction=0)
Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(1)  # bare single fit
txt_o = timeconstant.ntxt(Io_o/Io[0], tau_o)
plt.plot(1e3*(to-to[0]), 1e-3*Io, 'C1-', label='25_06 ' + txt_o)
plt.legend()
plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I$ kA')
plt.legend()


plt.figure()
points = data_load(path, 'VS3_discharge', date='2018_02_28')[0]
to, Io = points[1]['x'], points[1]['y']
tc = timeconstant(to, Io, trim_fraction=0)
Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(3)  # bare single fit
txt_o = timeconstant.ntxt(Io_o/Io[0], tau_o)
plt.plot(1e3*(to-to[0]), 1e-3*Io, 'C0-', label='28_02 ' + txt_o)

points = data_load(path, 'VS3_discharge_main_report', date='2018_06_25')[0]
to, Io = points[1]['x'], points[1]['y']
tc = timeconstant(to, Io, trim_fraction=0)
Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(3)  # bare single fit
txt_o = timeconstant.ntxt(Io_o/Io[0], tau_o)
plt.plot(1e3*(to-to[0]), 1e-3*Io, 'C1-', label='25_06 ' + txt_o)
plt.legend()
plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I$ kA')
plt.legend()
