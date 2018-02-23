from nep.DINA.read_dina import dina
from nova.streamfunction import SF


class read_eqdsk:

    def __init__(self, database_folder='eqdsk', file='burn'):
        self.dina = dina(database_folder)
        self.read_file(file=file)

    def read_file(self, file):
        filename = self.dina.locate_file(file)
        self.sf = SF(filename=filename)

    def plot(self):
        self.sf.contour()
        print(self.sf.cpasma)

if __name__ == '__main__':

    eqdsk = read_eqdsk()
    eqdsk.plot()



