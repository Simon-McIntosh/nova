import nep
from amigo.IO import class_dir
from os.path import join, isfile, isdir
from os import listdir


class dina:

    def __init__(self, database_folder):
        self.get_directory(database_folder)
        self.get_folders()

    def get_directory(self, database_folder):
        self.directory = join(class_dir(nep), '../Scenario_database')
        if database_folder is not None:
            self.directory = join(self.directory, database_folder)

    def get_folders(self):
        folders = [f for f in listdir(self.directory)]
        self.folders = sorted(folders)
        self.nf = len(self.folders)

    def select_folder(self, folder):  # folder entered as string, index or None
        if isinstance(folder, int):  # index (int)
            if folder > self.nf-1:
                txt = '\nfolder index {:d} greater than '.format(self.folder)
                txt += 'folder number {:d}'.format(self.nf)
                raise IndexError(txt)
            folder = self.folders[folder]
        elif isinstance(folder, str):
            if folder not in self.folders:
                txt = '\nfolder {} '.format(folder)
                txt += 'not found in {}'.format(self.directroy)
                raise IndexError(txt)
        elif folder is None:
            folder = self.directory
        else:
            raise ValueError('folder required as int, str or None')
        return join(self.directory, folder)

    def locate_file(self, file_type, folder):
        folder = self.select_folder(folder)
        ext = file_type.split('.')[-1].lower()
        if ext in ['xls', 'qda', 'txt']:  # data*.*
            file_type = file_type.split('.')[0].lower()
            for subfolder in listdir(folder):
                subfolder = join(folder, subfolder)
                if isdir(subfolder):
                    files = [f for f in listdir(subfolder) if
                             isfile(join(subfolder, f))]
                    folder_ext = files[0].split('.')[-1].lower()
                    if ext == folder_ext:
                        folder = subfolder
                        break
                    else:
                        files = []
            if not files:
                raise IndexError('file {} not found'.format(file_type))
        else:
            files = [f for f in listdir(folder) if isfile(join(folder, f))]
        file = [f for f in files if file_type in f.lower()]
        if len(file) == 0:
            txt = '\nfile key {} not found '.format(file_type)
            txt += 'in: \n{}'.format(files)
            raise IndexError(txt)
        else:
            file = file[0]
        return join(folder, file)


if __name__ == '__main__':

    dina = dina('operations')
    filename = dina.locate_file('data3.qda', folder=0)



