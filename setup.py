from setuptools import setup

import versioneer

requirements = [
    # package requirements go here
    "pandas",
    "numpy",
    "statsmodels",
]

setup(
    name="pstatmodel",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Statistical model for precipitacion",
    license="MIT",
    author="Gerardo Rivera Tello",
    author_email="grivera@igp.gob.pe",
    url="https://github.com/DangoMelon/pstatmodel",
    packages=["pstatmodel"],
    entry_points={"console_scripts": ["pstatmodel=pstatmodel.cli:cli"]},
    install_requires=requirements,
    keywords="pstatmodel",
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
