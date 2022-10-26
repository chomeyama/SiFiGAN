# -*- coding: utf-8 -*-

"""Setup Source-Filter HiFiGAN Library."""

import os
import sys
from distutils.version import LooseVersion

import pip
from setuptools import find_packages, setup

if LooseVersion(sys.version) < LooseVersion("3.8"):
    raise RuntimeError(
        "sifigan requires Python>=3.8, " "but your Python is {}".format(sys.version)
    )
if LooseVersion(pip.__version__) < LooseVersion("21.0.0"):
    raise RuntimeError(
        "pip>=21.0.0 is required, but your pip is {}. "
        'Try again after "pip install -U pip"'.format(pip.__version__)
    )

requirements = {
    "install": [
        "wheel",
        "torch>=1.9.0",
        "torchaudio>=0.8.1",
        "setuptools>=38.5.1",
        "librosa>=0.8.0",
        "soundfile>=0.10.2",
        "tensorboardX>=2.2",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "h5py>=2.10.0",
        "pyworld>=0.2.12",
        "sprocket-vc",
        "protobuf<=3.19.0",
        "hydra-core>=1.2",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
}
entry_points = {
    "console_scripts": [
        "sifigan-extract-features=sifigan.bin.extract_features:main",
        "sifigan-compute-statistics=sifigan.bin.compute_statistics:main",
        "sifigan-train=sifigan.bin.train:main",
        "sifigan-decode=sifigan.bin.decode:main",
        "sifigan-anasyn=sifigan.bin.anasyn:main",
        "sifigan-param-count=sifigan.bin.param_count:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
setup(
    name="sifigan",
    version="0.1",
    url="http://github.com/chomeyama/SourceFilterHiFiGAN",
    author="Reo Yoneyama",
    author_email="yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp",
    description="Source-Filter HiFiGAN implementation",
    long_description_content_type="text/markdown",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    license="MIT License",
    packages=find_packages(include=["sifigan*"]),
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=[
        "Programming Language :: Python :: 3.9.5",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
