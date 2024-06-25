import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
      name = "ifmta",
      version = "0.1",
      author = "François Leroux",
      author_email = "francois.leroux@gmail.com",
      description = "ifmta is a pedagogical python project about ifta algorithms for computer generated holograms computations. It has been initiated by François Leroux in 2024 at IMT Atlantique, Brest, France.",
      #license = read('LICENSE.txt'),
      keywords = "optics, holography, Iterative Fourier Transform Algorithm",
      #url = "https://github.com/...",
      install_requires = install_requires,
      packages = [
                  'ifmta'
                 ],
    package_dir = {'ifmta': 'ifmta'},
    data_files = [
                  ('', ['README.md', 'requirements.txt'])
                 ],
    classifiers = [
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Programming Language :: Python',
                   #'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
                  ],
    python_requires = '>=3.11.7',
)
