
import imas

import numpy as np


uri = 'imas:hdf5?user=public;version=3;shot=160400;run=10;database=iter'

equilibrium = imas.equilibrium()
equilibrium.ids_properties.homogeneous_time = 1
equilibrium.time = np.array([0])

pulse_schedule = imas.pulse_schedule()
pulse_schedule.ids_properties.homogeneous_time = 1
pulse_schedule.time = np.array([0])

db_entry = imas.DBEntry()
db_entry.create(uri=uri)
#db_entry.put(equilibrium)#, occurrence=0)
#db_entry.put(pulse_schedule, occurrence=0)
db_entry.close()

'''
db_entry = imas.DBEntry()
db_entry.create(uri=uri)
db_entry.put(pulse_schedule)#, occurrence=0)
db_entry.close()
'''
