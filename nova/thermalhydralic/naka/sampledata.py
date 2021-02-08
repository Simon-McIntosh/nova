
import pandas

from nova.thermalhydralic.naka.nakadata import NakaData

naka = NakaData(2015)
#print(naka.select('AC'))

name = naka.locate(0)
naka.download()

#
'''
naka.select_download('AC')
shot = naka.shot_index[284]
file = naka.locate(284, files='1-H')[0]
dataframe = pandas.read_csv(file, skiprows=7)
columns = {}
for name in dataframe.columns:
    columns[name] = name.replace('(sec)', '')
    columns[name] = columns[name].replace('phy(', '').replace(')', '')
dataframe.rename(columns=columns, inplace=True)
dataframe.dropna(inplace=True, axis=1)

print(dataframe.shape)
'''
