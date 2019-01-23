from distutils.core import setup
from setuptools import find_packages

setup(
    name='gram',
    version='1.0',
    author='Sebastian Bernasek',
    author_email='sebastian@u.northwestern.com',
    packages=find_packages(exclude=('tests',)),
    scripts=[],
    url='https://github.com/sebastianbernasek/gram',
    license='MIT',
    description='Package for simulating pulse response under gene regulatory and metabolic perturbations.',
    long_description=open('README.md').read(),
    install_requires=[
        "sobol >= 0.9",
        "sobol-seq >= 0.1.2",
        "matplotlib >= 2.0.0",
        "scipy >= 1.1.0"],
)
