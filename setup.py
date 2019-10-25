from setuptools import setup, find_packages


# allows to get version via python setup.py --version
__version__ = "0.3.0"


setup(
    name="edflow",
    version=__version__,
    description="Logistics for Deep Learning",
    url="https://github.com/pesser/edflow",
    author="Mimo Tilbich et al.",
    author_email="patrick.esser@iwr.uni-heidelberg.de, johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=[
        "pyyaml",
        "tqdm",
        "Pillow",
        "chainer",
        "numpy",
        "scikit-image",
        "pandas",
        "psutil",
        "deprecated",
        "fastnumbers",
    ],
    extras_require={
        "docs": ["sphinx >= 1.4", "sphinx_rtd_theme", "numpy"],
        "test": ["pytest", "pytest-cov", "coveralls"],
    },
    zip_safe=False,
    scripts=[
        "edflow/edflow",
        "edflow/edcache",
        "edflow/edlist",
        "edflow/edeval",
        "edflow/edsetup",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
