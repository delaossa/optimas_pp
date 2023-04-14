from setuptools import setup, find_packages

from libe_opt_postproc.__version__ import __version__

# Read long description
with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines()]


# Main setup command
setup(name='optimas_pp',
      version=__version__,
      author='',
      author_email='',
      description='Post processing scripts for Optimas',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/delaossa/optimas_pp.git',
      license='',
      packages=find_packages('.'),
      install_requires=read_requirements(),
      scripts=['optimas_pp/plot_history.py',
               'optimas_pp/plot_model.py',
               'optimas_pp/show_sims.py'],
      platforms='any',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent"],
      )
