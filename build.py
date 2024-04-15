"""Build IMAS Access Layer."""

import glob
import os
from pathlib import Path
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
import shutil
import subprocess
import zipfile


class CMakeExtension(Extension):
    """Extend setuptools Extension to build cmake source."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class BuildExt(build_ext):
    """Build IMAS al-python extension."""

    inplace: bool = True

    @property
    def cfg(self):
        """Return cfg attribute."""
        if self.debug:
            return "Debug"
        return "Release"

    def build_extension(self, ext):
        """Overwrite build_ext.build_extension to cmake al-python."""
        env = os.environ.copy()
        ext_full_path = self.get_ext_fullpath(ext.name)
        self.build_lib = Path(ext_full_path).parent.resolve()
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.build_lib}",
            f"-DCMAKE_BUILD_TYPE={self.cfg}",
        ]
        build_args = ["--config", self.cfg, "--", "-j4"]
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

    def _copy(self):
        """Copy imas and al-lowlevel wheels to project nova/imas/."""
        imas_dir = Path(self.build_temp).parent.parent / "imas"
        if imas_dir.is_dir():
            shutil.rmtree(imas_dir)
        imas_dir.mkdir()
        for wheel in glob.glob("**/*.whl", root_dir=self.build_temp, recursive=True):
            name = f"{Path(wheel).name.split('-')[0]}.whl"
            shutil.copy(Path(self.build_temp) / wheel, imas_dir / name)

    def unzip(self):
        """Unzip al-python wheels to project root."""
        root_dir = Path(self.build_temp).parent.parent
        for wheel in glob.glob("**/*.whl", root_dir=self.build_temp, recursive=True):
            with zipfile.ZipFile(Path(self.build_temp) / wheel, "r") as whl:
                members = [name for name in whl.namelist() if "dist-info" not in name]
                whl.extractall(root_dir, members)

    def copy(self):
        """Copy shared libraries to venv/lib -> update LD_LIBRARY_PATH to include."""
        library_dir = Path(os.environ["VIRTUAL_ENV"]) / "lib"
        shutil.copytree(self.build_lib, library_dir, dirs_exist_ok=True)


def build():
    """Build IMAS python access layer."""
    ext_modules = [CMakeExtension("al_python", sourcedir="../al-python")]
    distribution = Distribution({"ext_modules": ext_modules})
    cmd = BuildExt(distribution)
    cmd.ensure_finalized()
    cmd.run()
    cmd.unzip()
    cmd.copy()

    """
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        print(output, relative_extension)
        if not os.path.exists(output):
            continue

    shutil.copyfile(output, relative_extension)
    mode = os.stat(relative_extension).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(relative_extension, mode)
    """


if __name__ == "__main__":
    if Path("../al-python").is_dir():
        build()
