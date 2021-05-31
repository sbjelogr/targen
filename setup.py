import setuptools
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="targen",
    version="0.2.1",
    description="Tools for generating targets for ML experiments. Documentation link: https://sbjelogr.github.io/targen/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Sandro Bjelogrlic",
    author_email="sandro.bjelogrlic@gmail.com",
    license="Open Source",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(
        exclude=["notebooks"]
    ),
    install_requires=[
        "scikit-learn>=0.22.2",
        "pandas>=0.25",
        "matplotlib>=3.1.1",
        "scipy>=1.4.0",
        "joblib>=0.13.2",

    ],
    url="https://github.com/sbjelogr/targen",
    zip_safe=False,
)
