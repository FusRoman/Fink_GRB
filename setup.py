#!/usr/bin/env python

from setuptools import setup, find_packages
import fink_grb


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="fink-grb",
    version=fink_grb.__version__,
    description="Correlation of Fink alerts with notices from gamma ray monitors",
    author="Roman Le Montagner",
    author_email="roman.le-montagner@ijclab.in2p3.fr",
    url="https://github.com/FusRoman/Fink_GRB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"fink_grb": ["conf/fink_grb.conf"]},
    install_requires=[
        "fink-utils>=0.8.0",
        "docopt>=0.6.2",
        "terminaltables>=3.1.10",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "astropy==4.3.1",
        "gcn-kafka>=0.1.2",
        "importlib-resources==5.9.0",
        "pyarrow==9.0.0",  # WARNING: Version upper than the fink-broker version (pyarrow==4.0.1)
        "pyspark==3.3.0",
        "scipy==1.7.3",
        "voevent-parse==1.0.3",
        "fastavro==1.5.1",
        "healpy==1.16.1",
    ],
    entry_points={"console_scripts": ["fink_grb=fink_grb.fink_grb_cli:main"]},
    license="Apache-2.0 License",
    platforms="Linux Debian distribution",
    project_urls={
        "Source": "https://github.com/FusRoman/Fink_GRB",
    },
    python_requires=">=3.7",
)
