from distutils.core import setup
from setuptools import find_packages

setup(
    name='pulse',
    version='0.1-beta',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://github.com/sebastianbernasek/pulse',
    license='MIT',
    description='Package for simulating pulse response.',
    long_description=open('README.md').read(),
    install_requires=[
        "sobol >= 0.9",
        "sobol-seq >= 0.1.2",
        "matplotlib >= 2.0.0",
        "scipy >= 1.1.0"],
)
