from setuptools import setup, find_packages

with open("Readme.md", "r") as fh:
    long_description = fh.read()

setup (
    name = "farmersdashboard",
    packages = find_packages(include=['farmersdashboard', 'farmersdashboard.*']),
    version = '0.1.0',
    author = 'Peter Hanappe',
    description = "Python package for the ROMI's Farmer's Dashboard.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/romi/farmers-dashboard-analysis/",
    install_requires=[],
    python_requires = '>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)"
    ]
)
