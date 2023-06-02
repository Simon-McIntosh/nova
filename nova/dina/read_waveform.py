from os import listdir, sep
from os.path import join, isfile, isdir
from datetime import datetime
import string
import fnmatch

import numpy as np
import pandas as pd

from nova.definitions import root_dir
from nova.utilities.IO import pythonIO, readtxt
from nova.utilities.time import clock
from nova.utilities.qdafile import QDAfile
import nova


class read_waveform(pythonIO):
    def __init__(self, database_folder=None, read_txt=True):
        """
        Kwargs:
            database_folder (str): name of database folder
            read_txt (bool): read / reread source text files
        """
        pythonIO.__init__(self)  # python read/write
        self.set_directory(database_folder)
        self.read_txt = read_txt

    def set_directory(self, database_folder):
        self.database_folder = database_folder
        self.get_directory()
        self.get_folders()

    def get_directory(self):
        self.directory = join(root_dir, "data/DINA")
        if self.database_folder is not None:
            self.directory = join(self.directory, self.database_folder)

    def get_folders(self):
        self.folders = [f for f in listdir(self.directory) if f[0] != "."]
        self.nfolder = len(self.folders)
        files = [f for f in listdir(self.directory) if isfile(f)]
        self.files = sorted(files)
        self.nfile = len(self.files)

    def select_folder(self, folder):  # folder entered as string, index or None
        if isinstance(folder, int):  # index (int)
            if folder < 0:
                folder += self.nfolder
            if folder > self.nfolder - 1:
                txt = "\nfolder index {:d} greater than ".format(folder)
                txt += "folder number {:d}".format(self.nfolder)
                raise IndexError(txt)
            folder = self.folders[folder]
        elif isinstance(folder, str):
            if folder not in self.folders:
                folder = folder.split("_")  # remove leading underscore
                folder = "_".join([" ".join(folder[:2]), "_".join(folder[2:])])
                if folder not in self.folders:
                    txt = "\nfolder {} ".format(folder)
                    txt += "not found in {}".format(self.directory)
                    raise IndexError(txt)
        elif folder is None:
            folder = self.directory
        else:
            raise ValueError("folder required as int, str or None")
        return join(self.directory, folder)

    def locate_file_type(self, file, file_type, folder):
        file_types = ["txt", "qda"]
        if file_type in file_types:
            file_types.remove(file_type)
            file_types = [file_type, *file_types]
        else:
            file_types = [file_type]
        filepath = None
        for file_type in file_types:
            try:
                filepath = self.locate_file(f"{file}.{file_type}", folder=folder)
                break
            except IndexError:
                pass
        if filepath is None:
            raise FileNotFoundError()
        return filepath, file_type

    def locate_folder(self, file, folder, file_type="txt"):
        print(file, folder, file_type)
        filepath, file_type = self.locate_file_type(file, file_type, folder)
        print(filepath, file_type)
        self.filename = filepath.split(sep)[-3].replace(" ", "_")
        self.folder = folder
        self.filepath = sep.join(filepath.split(sep)[:-1]) + sep
        return join(self.filepath, self.filename), file_type

    def locate_file(self, file_type, folder=None):
        if self.nfolder == 0:
            folder = None
        folder = self.select_folder(folder)
        print("folder", folder)
        ext = file_type.split(".")[-1].lower()
        if ext in ["xls", "qda", "txt"]:  # *.*
            files = []
            file_type = file_type.split(".")[0].lower()
            subfolders = listdir(folder)
            for subfolder in subfolders:
                subfolder = join(folder, subfolder)
                if isdir(subfolder):
                    files = [
                        f for f in listdir(subfolder) if isfile(join(subfolder, f))
                    ]
                    folder_ext = [file.split(".")[-1].lower() for file in files]
                    print(folder_ext)
                    if ext in folder_ext:
                        folder = subfolder
                        break
                    else:
                        files = []
            if not files:
                raise IndexError(
                    f"file {file_type}.{ext} not found\n"
                    f"dir: {join(folder, subfolder)}"
                )
        else:
            file_type = file_type.split(".")[0]
            files = [f for f in listdir(folder) if isfile(join(folder, f))]
        # files = [f for f in files if file_type.lower() in f.lower()]
        files = [
            f
            for f in files
            if fnmatch.fnmatch(f.lower().split(".")[0], file_type.lower())
        ]
        if len(files) == 0:
            txt = "\nfile key {} not found ".format(file_type)
            txt += "in: \n{}".format(files)
            raise IndexError(txt)
        try:
            file = [f for f in files if ext in f.lower()][0]
        except IndexError:
            raise IndexError("ext {} not found in {}".format(ext, files))
        self.folder = folder
        self.file = file
        return join(folder, file)

    def load_folder(self):
        """
        load / reload all files from specified database folder
        """
        nfolder = self.nfolder
        tick = clock(
            nfolder,
            header=f"loading {nfolder} scenarios from "
            f"folder: {self.database_folder}",
        )
        for folder in range(nfolder):
            self.load_file(folder, read_txt=True, verbose=False)
            tick.tock()

    def multiindex(self, data, columns, units, dropnan=False, dataframe=True):
        data.columns = pd.MultiIndex.from_tuples(
            [(columns[c], u) for c, u in zip(columns, units)], names=["name", "unit"]
        )
        data_keys = list(data.keys())
        for var in data_keys:
            if len(data[var]) == 0 or np.isnan(data[var]).all():
                data.pop(var)
        if dropnan:
            data = data.dropna(axis=0)  # remove NaN values
        if not dataframe:
            data = data.to_dict(orient="list")
        return data


class read_corsica(read_waveform):
    "read corsica wavefrom data"

    def __init__(self, *args, **kwargs):
        read_waveform.__init__(self, *args, **kwargs)

    def read_file(self):
        filename = self.locate_file("T_.txt", folder=-1)
        data = pd.DataFrame()
        comments, units = {}, []
        nz_index = 0
        with readtxt(filename) as f:
            f.trim("ncol", index=0)
            ncol = f.readnumber()
            f.skiplines(1)
            nt = f.readnumber()
            while True:
                try:
                    label = f.readline(split=True, string=True, sep=":")
                    variable = label[0].strip()
                    note = label[1].split()
                    comment = " ".join(note[:-1])
                    comments[variable] = comment
                    unit = note[-1].replace("[", "").replace("]", "")
                    units.append(unit)
                    if variable == "<nz>(t)":
                        variable = variable.replace(
                            "(", f"{string.ascii_letters[nz_index]}("
                        )
                        nz_index += 1
                        num = f.readline(split=True, string=True)[2::3]
                        num = [float(n.replace("D", "E").replace(",", "")) for n in num]
                        comment += f" ({num[0]}, {num[1]})"
                    if variable == "Ncoils":  # read PF / CS coil currents
                        nC = f.readnumber()
                    else:
                        data[variable] = f.readblock()
                except ValueError:
                    try:
                        label = f.readline(split=False, string=True)
                        if "nottt available" in label:  # valiable not avalible
                            continue
                        else:  # not implemented
                            raise NotImplementedError(f"read error for line: {label}")
                    except:
                        break
            data.rename(
                columns={c: c.replace("(t)", "") for c in data.columns}, inplace=True
            )
            data.rename(
                columns={c: c.replace(",t)", ")") for c in data.columns}, inplace=True
            )

            current = {
                c: f'I{c.replace("current", "").strip()}'
                for c in data.columns
                if "current" in c
            }
            data.rename(columns=current, inplace=True)
            kappa = {
                c: c.replace("Triangularity", "kappa")
                .strip()
                .replace("_Lower", "L")
                .replace("_Upper", "U")
                for c in data.columns
                if "Triangularity" in c
            }
            data.rename(columns=kappa, inplace=True)
            elongation = {
                c: c.replace("Elongation", "dell").strip()
                for c in data.columns
                if "Elongation" in c
            }
            data.rename(columns=elongation, inplace=True)
            data.rename(columns={"Timebase": "t"}, inplace=True)


class read_dina(read_waveform):
    date_switch = datetime.strptime("2016-02", "%Y-%m")

    def __init__(self, *args, **kwargs):
        read_waveform.__init__(self, *args, **kwargs)

    def get_folders(self):
        read_waveform.get_folders(self)
        if self.database_folder == "scenarios":
            self.folders = sorted(
                self.folders,
                key=lambda x: f'{x.split("-")[1]}_'
                f'{x.split("-")[2]}_'
                f'{x.split("-")[0]}',
            )

    def get_folder_array(self, exclude=[]):
        dtype = [
            ("name", "U25"),
            ("year", int),
            ("mode", "U25"),
            ("month", int),
            ("version", "U25"),
        ]
        folder_array = np.ones(self.nfolder, dtype=dtype)
        for i in range(self.nfolder):
            folder_array[i]["name"] = self.folders[i]
            folder_array[i]["mode"] = self.folders[i].split("DINA")[0][:-1]
            timestamp = self.folders[i].split("DINA")[-1]
            folder_array[i]["year"] = int(timestamp.split("-")[0])
            timestamp = "".join(timestamp.split("-")[1:])
            folder_array[i]["month"] = int(timestamp[:2])
            folder_array[i]["version"] = timestamp[2:].replace("_", "")
        folder_array.sort(order=["year", "month", "version"])
        if exclude:
            index = [name not in exclude for name in folder_array["name"]]
            folder_array = folder_array[index]
        self.folder_array = folder_array
        self.folders = self.folder_array["name"]

    def locate_folder(self, file, folder, file_type="txt"):
        filepath, file_type = read_waveform.locate_folder(self, file, folder, file_type)
        self.date = datetime.strptime(
            self.filename.split("DINA")[-1].split("_")[0][:7], "%Y-%m"
        )
        return filepath, file_type

    def read_csv(self, filename, split="", dropnan=True, dataframe=True):
        data = pd.read_csv(filename, delimiter="\t", na_values="NAN")
        columns = {}
        units = []
        for c in list(data):
            uo = ""
            if split:
                c_split = c.split(split)
                co = c_split[0]
                co = co.replace(" or ", "_or_")
                co = co.replace(" ", "")
                if len(c_split) == 2:
                    uo = c_split[1].replace(" ", "")
            if co in [key.split(split)[0] for key in columns]:
                co += "_extra"  # seperate duplicates
            columns[c] = co
            units.append(uo)
        data = self.multiindex(
            data, columns, units, dropnan=dropnan, dataframe=dataframe
        )
        return data

    def read_qda(self, filename, split="", dropnan=True, dataframe=True):
        qdafile = QDAfile(filename)
        data = pd.DataFrame()
        columns = {}
        units = []
        for i, (var, nrow) in enumerate(zip(qdafile.headers, qdafile.rows)):
            uo = ""
            var = var.decode()
            if nrow > 0:
                var_split = var.split(",")
                co = var_split[0]
                co = co.replace(" or ", "_or_")
                co = co.replace(" ", "")
                columns[var] = co
                if len(var_split) == 2:
                    uo = var_split[1].replace(" ", "")
                data[columns[var]] = np.array(qdafile.data[i, :])
                units.append(uo)
        data = self.multiindex(
            data, columns, units, dropnan=dropnan, dataframe=dataframe
        )
        return data


if __name__ == "__main__":
    # corsica = read_corsica('corsica')
    # corsica.read_file()

    dina = read_dina("scenarios")
    dina.load_file(-1)
    # filename = dina.locate_file('data2.txt', folder=1)
