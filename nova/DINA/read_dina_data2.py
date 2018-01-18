import pandas as pd
from os import listdir
from os.path import isfile, join
import nep
from amigo.IO import class_dir
from amigo.pyplot import plt


path = 'C:/Users/mcintos/Downloads/'
file = '15MA-DINA2016-02_Data2.txt'

path = "C:/Users/mcintos/Documents/Work/Code/NeP/\
Scenario_database/15MA DT-DINA2010-03-v2/TEXT data/"
file = '15MA-DINA2010-03-v2_Data2.TXT'


class dina:

    def __init__(self, scenario):
        self.data_base = join(class_dir(nep), '../Scenario_database/')
        self.locate(scenario)

    def locate(self, scenario):
        folder = join(join(self.data_base, scenario), 'TEXT data')
        self.files = {}

        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            fl = f.lower()
            index = fl.find('data')  # find data files
            if index >= 0:
                name = fl[index:].replace('.txt','').replace('data', 'd')
                self.files[name] = f


        print(self.files)


if __name__ == '__main__':

    scenario = '15MA DT-DINA2010-03-v2'
    dn = dina(scenario)

    data = pd.read_csv(path+file, sep='\t', usecols=range(0, 60))

    columns = {}
    for c in list(data):
        columns[c] = c.split(',')[0]
    data = data.rename(index=str, columns=columns)

    t = data['t'].get_values()
    Ip = data['Ip'].get_values()

    print(Ip.max())
    iFT = Ip > 15


    plt.plot(t[iFT][0], Ip[iFT][0],'o')
    plt.plot(t[iFT][-1], Ip[iFT][-1],'o')
    plt.plot(t, Ip)
    plt.despine()
    plt.show()


    data.plot('t', ['Ip', 'Ipf6', 'Ipf5', 'Ipf4', 'Ipf3', 'Ipf2', 'Ipf1'])
    plt.ylabel(r'current $I$ MA')
    plt.despine()
    plt.show()

    print(list(data))
