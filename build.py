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

    def __init__(self, name: str, sourcedir: str = "", **kwargs) -> None:
        super().__init__(name, sources=[], **kwargs)
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

    def copy_whl(self):
        """Copy imas and al-lowlevel wheels to project nova/imas/."""
        dist_dir = Path(self.build_temp).parent.parent / "dist"
        dist_dir.mkdir(exist_ok=True)
        for wheel in glob.glob("**/*.whl", root_dir=self.build_temp, recursive=True):
            shutil.copy(Path(self.build_temp) / wheel, dist_dir / Path(wheel).name)

    def unzip(self):
        """Unzip al-python wheels to project root."""
        root_dir = Path(self.build_temp).parent.parent
        for wheel in glob.glob("**/*.whl", root_dir=self.build_temp, recursive=True):
            with zipfile.ZipFile(Path(self.build_temp) / wheel, "r") as whl:
                members = [name for name in whl.namelist() if "dist-info" not in name]
                whl.extractall(root_dir, members)

    def copy_lib(self):
        """Copy shared libraries to venv/lib -> update LD_LIBRARY_PATH to include."""
        lib_dir = Path(self.build_temp).parent.parent / "lib"
        shutil.copytree(self.build_lib, lib_dir, dirs_exist_ok=True)


def build():
    """Build IMAS python access layer."""
    ext_modules = [
        CMakeExtension(
            "al_python", sourcedir="../al-python", runtime_library_dirs=["lib"]
        )
    ]
    distribution = Distribution({"name": "al_python", "ext_modules": ext_modules})
    distribution.package_dir = {"imas": "al_python"}

    cmd = BuildExt(distribution)
    cmd.ensure_finalized()
    cmd.run()
    cmd.unzip()
    cmd.copy_lib()
    cmd.copy_whl()

    print(cmd.get_outputs())


if __name__ == "__main__":
    if Path("../al-python").is_dir():
        build()
        """
        # awaiting https://github.com/python-poetry/poetry/issues/5983
        python = f"{os.environ['VIRTUAL_ENV']}/bin/python"
        for module in ["imas", "al_lowlevel"]:
            subprocess.check_call(
                [python, "-m", "pip", "install", "--find-links", "dist/", module]
            )
        """
