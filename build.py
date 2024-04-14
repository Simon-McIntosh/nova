"""Build IMAS Access Layer."""

import os
from pathlib import Path
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

import shutil

import subprocess


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

    def copy(self):
        """Copy build_temp to build/imas."""
        imas_temp = Path(self.build_temp) / "imas"
        imas_module = Path(self.build_temp).parent.parent / "imas"
        if imas_module.is_dir():
            shutil.rmtree(imas_module)
        shutil.copytree(imas_temp, imas_module)
        imas_lib = imas_module / "lib"
        if imas_lib.is_dir():
            shutil.rmtree(imas_lib)
        shutil.copytree(self.build_lib, imas_lib)


def build():
    """Build IMAS python access layer."""
    ext_modules = [CMakeExtension("imas", sourcedir="../al-python")]

    distribution = Distribution({"ext_modules": ext_modules})
    # distribution.package_dir = {"": "extended"}
    cmd = BuildExt(distribution)
    cmd.ensure_finalized()
    cmd.run()
    cmd.copy()

    """
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        print("output", output)
        print("re", relative_extension)
        print(os.listdir())
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)
    """


if __name__ == "__main__":
    build()
