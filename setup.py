import os
import os.path as op
from setuptools import PEP420PackageFinder
from distutils.core import setup

ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")

setup(
    name="subhramaniyan-t-package",
    version="0.1",
    author="Subhrmaniyan",
    author_email="subhram@gmail.com",
    package_dir={"": "src"},
    description="TAMLEP",
    packages=PEP420PackageFinder.find(where=str(SRC))
)
