#!/usr/bin/env python
import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="NomTransformerOCR",
    version="0.0.1",
    description="OCR for Nom (Han-Nom) script using CoMER-based transformer",
    author="",
    author_email="",
    url="",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    packages=find_packages(),
)
