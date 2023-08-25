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
    "metadrive-simulator>=0.4.1.1",
    "geopandas",
    "yapf==0.30.0",
    "shapely"
]

doc = [
    "sphinxemoji",
    "sphinx",
    "sphinx_rtd_theme",
]

train_requirement = [
    "ray[rllib]==1.0.0",
    # "torch",
    "wandb==0.12.1",
    "aiohttp==3.6.0",
    "gymnasium",
    "tensorflow",
    "tensorflow_probability"]

setup(
    name="scenarionet",
    python_requires='>=3.8',  # do version check with assert
    version=version,
    description="Scalable Traffic Scenario Management System",
    url="https://github.com/metadriverse/ScenarioNet",
    author="MetaDrive Team",
    author_email="quanyili0057@gmail.com, pzh@cs.ucla.edu",
    packages=packages,
    install_requires=install_requires,
    extras_require={
        "train": train_requirement,
        "doc": doc
    },
    include_package_data=True,
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)

