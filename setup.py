""" Setup
"""
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

setup(
    name="uuv-optim",
    version=__version__,
    description="A design optimization study of underwater vehicle using Bayesian optimization and deep learning based surrogate model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vardhah/UUV-design-optimization",
    author="Umesh Timalsina, Harsh Vardhan",
    author_email="umesh.timalsina@vanderbilt.edu",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="cfd surrogate modeling bayesian optimization",
    packages=["uuv_optim"],
    include_package_data=True,
    install_requires=["torch >= 1.4", "torchvision"],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["uuv-optim = uuv_optim.__main__:run"]},
)
