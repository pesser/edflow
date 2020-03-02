from setuptools import setup, find_packages

# allows to get version via python setup.py --version
__version__ = "dev"

install_requires = [
    "pyyaml",
    "tqdm",
    "Pillow < 7.0.0",
    "chainer",
    "numpy",
    "pandas",  # for csv dataset and eval pipeline
]

install_full = [  # for extra functionality
    "streamlit > 0.49",  # for edexplore
    "psutil",  # for edlist
    "scipy>=1.4.1",  # pinned dependency of scikit-image; 1.4.1 fixes https://github.com/scipy/scipy/issues/11237
    "scikit-image",  # for ssim in image_metrics.py
    "black",  # for formatting of code
    "matplotlib",  # for plot_datum
    "flowiz",  # for visualizing flow with streamlit
    "wandb",  # for `--wandb_logging True`
    "tensorboard",  # for `--tensorboard_logging True`
]
install_docs = [  # for building the documentation
    "sphinx >= 1.4",
    "sphinx_rtd_theme",
    "better-apidoc",
]
install_test = install_full + [  # for running the tests
    "pytest",
    "pytest-cov",
    "coveralls",
    "coverage < 5.0",  # TODO pinned dependency of coveralls; see https://github.com/coveralls-clients/coveralls-python/issues/203
]
extras_require = {"full": install_full, "docs": install_docs, "test": install_test}

long_description = """Reduce boilerplate code for your ML projects. TensorFlow
and PyTorch. [Documentation](https://edflow.readthedocs.io/)"""


setup(
    name="edflow",
    version=__version__,
    description="Logistics for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pesser/edflow",
    author="Mimo Tilbich et al.",
    author_email="patrick.esser@iwr.uni-heidelberg.de, johannes.haux@iwr.uni-heidelberg.de",
    license="MIT",
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
    scripts=[
        "edflow/edflow",
        "edflow/edcache",
        "edflow/edlist",
        "edflow/edeval",
        "edflow/edsetup",
        "edflow/edexplore",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
