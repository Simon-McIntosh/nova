
from nova.electromagnetic.vde import VDE


if __name__ == '__main__':

    folder = 'VDE_UP_slow'
    vde = VDE(folder=folder, read_txt=False)

    vde.update(-3)
    vde.loc['plasma', 'nturn'] = 0

    vde.plot()
