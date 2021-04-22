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
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

def get_version(package):
    """Return package version as listed in `__version__` in `__init__.py`."""
    with open(os.path.join(package, '__init__.py')) as f:
        init_py = f.read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


extras_require = {
    'shellcomplete': ['click_completion'],
    'tensorflow':    ['tensorflow~=2.0', 'tensorflow-probability~=0.10'],
    'jax': ['jax~=0.1,>0.1.72', 'jaxlib~=0.1,>0.1.51']
}
extras_require['backends'] = sorted(
    set(extras_require['tensorflow'] + extras_require['jax'])
)


setup(
    name='emme',
    version=get_version("./src/emme"),
    description='Object oriented finite element analysis.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.md')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.md'))
    ),
    long_description_content_type= 'text/markdown',
    author='Claudio M. Perez',
    author_email='claudio_perez@berkeley.edu',
    url='https://github.com/claudioperez/emme',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        'Changelog': 'https://github.com/claudioperez/emme/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/claudioperez/emme/issues',
    },
    keywords=[
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandoc",
        "matplotlib",
        "scipy"
    ],
    extras_require=extras_require,
)
