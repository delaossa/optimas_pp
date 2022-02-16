from setuptools import setup, find_packages

from libe_opt_postproc.__version__ import __version__

# Read long description
with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines()]


# Main setup command
setup(name='libE_opt_postproc',
      version=__version__,
      author='',
      author_email='',
      description='Post processing scripts for libE_opt',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/delaossa/libE_opt_postproc',
      license='',
      packages=find_packages('.'),
      install_requires=read_requirements(),
      scripts=['libe_opt_postproc/plot_history.py',
               'libe_opt_postproc/plot_model.py'],
      platforms='any',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent"],
      )
