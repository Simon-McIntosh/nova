from setuptools import setup, find_packages

setup(name='nova',
      version='1.0.0',
      description='equlibrium tools',
      url='https://git.iter.org/projects/SCEN/repos/nova/',
      author='Simon McIntosh',
      author_email='simon.mcintosh@iter.org',
      license='MIT',
      packages=find_packages(), 
      ext_modules=[cc.distutils_extension()],
      zip_safe=False)
