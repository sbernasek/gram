# GRaM


GRaM Overview
=============

This repository and its dependencies contain all of the code necessary to reproduce our study of the relationship between Gene Regulation and Metabolism (GRaM). The code offers two primary functions:

  1. Running the simulations described in our manuscript. In general, these simulations probe pulse response sensitivity to gene regulatory network (GRN) perturbations under a variety of different metabolic conditions. *GRaM* provides the tools necessary to set up each of these simulations and analyze the resultant dynamics. The simulations themselves are performed by our stochastic simulation package, [GeneSSA](https://github.com/sebastianbernasek/genessa).
  2. Quantifying Yan protein expression dynamics in the Drosophila eye. Yan level measurements were extracted from confocal microscopy data using [FlyEye Silhouette](FLYEYE SILHOUETTE), our macOS platform for eye cell segmentation and annotation. The annotated `.silhouette` files are available in our [data repository](DATA REPOSITORY). We analyzed these measurements using [FlyEye Analysis](FLYEYE ANALYSIS), our newly released pipeline for analyzing FlyEye Silhouette data.


Reproducibility
===============

This repository contains all of the code used to generate the content in our manuscript. However, please note that the vast majority of our figures are based on large-scale simulations that were executed on a high performance computing cluster. These simulations are therefore not well suited for reproduction on a personal computer. It is consequently not possible to provide a simple "one click" means to directly reproduce all of our results because the execution platforms will differ considerably between users. Instead, we have provided a [Jupyter notebook](https://github.com/sebastianbernasek/GRaM/tree/master/notebooks) that walks the user through the steps necessary to set up and execute each type of simulation that appears in our manuscript. While we leave it to the user to design a means to execute these simulations en masse, we have also made our own results comprehensively available via our [data repository](DATA REPOSITORY). We have provided an additional series of [Jupyter notebooks](NOTEBOOKS) demonstrating how each of our figures were generated from these results.


Installation
============

Before attempting to install *GRaM*, we suggest creating a clean virtual environment and installing all necessary dependencies first. If you intend to reproduce our numerical simulations, you will first need to compile and install our stochastic simulation package, [GeneSSA](https://github.com/sebastianbernasek/genessa).


System Requirements
-------------------

 - Python 3.6+
 - [Scipy](https://www.scipy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Matplotlib](https://matplotlib.org/)
 - [GeneSSA](https://github.com/sebastianbernasek/genessa)
 - [FlyEye Analysis](file:///Users/Sebi/Documents/grad_school/research/flyeye/flyeye/docs/index.html)


Install GRaM
------------

Download the [latest distribution](https://github.com/sebastianbernasek/genessa/archive/v0.1.tar.gz).

The simplest method is to install it via ``pip``:

    pip install gram-0.1.tar.gz


GRaM Modules
============

GRaM consists of several modules:

  * ``gram.models`` provides templates for each type of GRN discussed in our manuscript.

  * ``gram.simulation`` provides methods for performing pulse response simulations.

  * ``gram.analysis`` provides methods for comparing expression dynamics between simulations, e.g. evaluating "error frequency".

  * ``gram.sweep`` provides methods for constructing and executing a parameter sweep of each type of model.

  * ``gram.figures`` provides templates for each type of figure that appears in our manuscript.



Supporting Data
===============

  * [Simulation Results](DATA REPOSITORY) - complete output from each of our simulations.

  * [Yan Expression](DATA REPOSITORY) - segmented and annotated $YanYFP$ and $Yan^{\delta miR7}YFP$ eye discs from animals raised under normal metabolic conditions and in animals subject to IPC ablation.


Example Usage
-------------

For examples detailing the usage of our stochastic simulation software, please see [GeneSSA](https://github.com/sebastianbernasek/genessa).

For examples demonstrating the analysis of protein expression in the Drosophila eye, please see [FlyEye Analysis](file:///Users/Sebi/Documents/grad_school/research/flyeye/flyeye/docs/index.html).


Additional Questions
--------------------

Please contact the [Amaral Lab] with any questions regarding GRaM, GeneSSA, or the FlyEye suite. We will make an effort to get back to you within a reasonable timeframe.
