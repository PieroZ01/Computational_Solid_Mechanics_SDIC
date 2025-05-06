from setuptools import setup, find_packages

meta = {}
with open("meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = 'Small Library for Computational Solid Mechanics'
URL = 'https://github.com/PieroZ01/Computational_Solid_Mechanics_SDIC'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'Computational Solid Mechanics'

REQUIRED = [
    'numpy', 'matplotlib'
]

LDESCRIPTION = (
    "This package is a small library to solve some simple problems in "
    "computational solid mechanics, namely: springs, linear bars and plane "
    "trusses. It is designed to be easy to use and understand; "
    "tutorials are provided along with the code. The package was "
    "developed in 2025 as part of the course 'Computational Solid Mechanics', "
    "which is part of the Master's degree in Scientific and Data-Intensive "
    "Computing at the University of Trieste, Italy."
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: Production/Stable',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    zip_safe=False,
)
