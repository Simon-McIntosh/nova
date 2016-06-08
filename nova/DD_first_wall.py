from png_tools import data_mine

file = 'SDX'  
path = './config_data/'+file+'/'

data_ID = 'first_wall_tmp'
data_mine(path, file, data_ID, [5,15], [-12,8])
