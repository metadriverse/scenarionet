# Please don't change the order of following packages!
import os
import sys
from os import path

from setuptools import setup, find_namespace_packages  # This should be place at top!

ROOT_DIR = os.path.dirname(__file__)


def is_mac():
    return sys.platform == "darwin"


def is_win():
    return sys.platform == "win32"


assert sys.version_info.major == 3 and 6 <= sys.version_info.minor < 12, \
    "python version >= 3.6, <3.12 is required"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
packages = find_namespace_packages(
    exclude=("docs", "docs.*", "documentation", "documentation.*", "build.*"))
print("We will install the following packages: ", packages)

""" ===== Remember to modify the EDITION at first ====="""
version = "0.0.1"

install_requires = [
    "numpy>=1.21.6, <=1.24.2",
    "matplotlib",
    "pandas",
    "tqdm",
    "metadrive-simulator",
]

setup(
    name="scenarionet",
    python_requires='>=3.6, <3.12',  # do version check with assert
    version=version,
    description="Scalable Traffic Scenario Management System",
    url="https://github.com/metadriverse/ScenarioNet",
    author="MetaDrive Team",
    author_email="quanyili0057@gmail.com, pzh@cs.ucla.edu",
    packages=packages,
    install_requires=install_requires,
    # extras_require={
    #     "cuda": cuda_requirement,
    #     "nuplan": nuplan_requirement,
    #     "waymo": waymo_requirement,
    #     "all": nuplan_requirement + cuda_requirement
    # },
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)

"""
How to publish to pypi?  Noted by Zhenghao in Dec 27, 2020.

0. Rename version in setup.py

1. Remove old files and ext_modules from setup() to get a clean wheel for all platforms in py3-none-any.wheel
    rm -rf dist/ build/ documentation/build/ scenarionet.egg-info/ docs/build/

2. Rename current version to X.Y.Z.rcA, where A is arbitrary value represent "release candidate A". 
   This is really important since pypi do not support renaming and re-uploading. 
   Rename version in setup.py 

3. Get wheel
    python setup.py sdist bdist_wheel

4. Upload to test channel
    twine upload --repository testpypi dist/*

5. Test as next line. If failed, change the version name and repeat 1, 2, 3, 4, 5.
    pip install --index-url https://test.pypi.org/simple/ scenarionet

6. Rename current version to X.Y.Z in setup.py, rerun 1, 3 steps.

7. Upload to production channel 
    twine upload dist/*

"""
