#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io, os, re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


def get_version(package):
    """Return package version as listed in `__version__` in `__init__.py`."""
    with open(os.path.join(package, "__init__.py")) as f:
        init_py = f.read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


extras_require = {
    "shellcomplete": ["click_completion"],
    "tensorflow": ["tensorflow~=2.0", "tensorflow-probability~=0.10"],
    "jax": ["jax~=0.1,>0.1.72", "jaxlib~=0.1,>0.1.51"],
}
extras_require["backends"] = sorted(
    set(extras_require["tensorflow"] + extras_require["jax"])
)


setup(
    name="anabel",
    version=get_version("./src/anabel"),
    description="An end to end differentiable finite element framework.",
    long_description=f"{read('README.md')}\n{read('CHANGELOG.md')}",
    long_description_content_type="text/markdown",
    author="Claudio M. Perez",
    author_email="claudio_perez@berkeley.edu",
    url="https://claudioperez.github.io/anabel",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    project_urls={
        "Changelog": "https://github.com/claudioperez/anabel/blob/master/CHANGELOG.md",
        "Issue Tracker": "https://github.com/claudioperez/anabel/issues",
    },
    keywords=[],
    python_requires=">=3.7",
    install_requires=[
        "jax", "jaxlib", "numpy", "pandoc", "matplotlib", "scipy", "anon", "meshio"
    ],
    extras_require=extras_require,
)
