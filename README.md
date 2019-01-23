Overview
========

This repository and its dependencies contain all of the code necessary to reproduce our study of the relationship between Gene Regulation and Metabolism (GRaM). The code offers two primary functions:

  1. Running the simulations described in our manuscript. In general, these simulations probe pulse response sensitivity to gene regulatory network (GRN) perturbations under a variety of different metabolic conditions. *GRaM* provides the tools necessary to set up each of these simulations and analyze the resultant dynamics. The simulations themselves are performed by our stochastic simulation package, [GeneSSA](https://github.com/sebastianbernasek/genessa).
  2. Quantifying Yan protein expression dynamics in the Drosophila eye. Yan level measurements were extracted from confocal microscopy data using [FlyEye Silhouette](FLYEYE SILHOUETTE), our macOS platform for eye cell segmentation and annotation. The annotated `.silhouette` files are available in our [data repository](DATA REPOSITORY). We analyzed these measurements using [FlyEye Analysis](https://github.com/sebastianbernasek/flyeye), our pipeline for analyzing FlyEye Silhouette data.


Supporting Data
===============

  * [Simulation Results](DATA REPOSITORY) - complete output from each of our simulations.

  * [Yan Expression](DATA REPOSITORY) - segmented and annotated $YanYFP$ and $Yan^{\delta miR7}YFP$ eye discs from animals raised under normal metabolic conditions and in animals subject to IPC ablation.


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


Reproducing our Results
=======================

This repository contains all of the code used to generate the content in our manuscript. However, please note that the vast majority of our figures are based on large-scale simulations that were executed on a high performance computing cluster. These simulations are therefore not well suited for reproduction on a personal computer. While we have provided the scripts necessary to reproduce all of our results, their direct execution would be impractical.

As an alternative, we have provided the output from all of our [completed simulations](DATA REPOSITORY) along with a series of [Jupyter notebooks](https://github.com/sebastianbernasek/GRaM/tree/master/notebooks) that walk the user through the steps necessary to analyze these results and reproduce each of our figures. The notebooks also provide users with an opportunity to set up and execute each type of simulation that appears in our manuscript. We leave it to the user to design a means to execute these simulations en masse.


Package Contents
================

The ``gram`` package consists of a set of python modules, scripts, and notebooks that walk the user through reproducing all of our simulation results. Their contents are as follows.


Modules
-------

The GRaM package consists of several modules:

  * ``gram.models`` provides templates for each type of GRN discussed in our manuscript.

  * ``gram.simulation`` provides methods for performing pulse response simulations.

  * ``gram.analysis`` provides methods for comparing expression dynamics between simulations, e.g. evaluating "error frequency".

  * ``gram.sweep`` provides methods for constructing and executing a parameter sweep of each type of model.

  * ``gram.figures`` provides templates for each type of figure that appears in our manuscript.


Scripts
-------

The GRaM package contains several python scripts in ``gram/scripts``. Those that may prove helpful include:

  * ``run_pairs.py`` executes a series of pairwise simulations between different types of repressors acting upon different levels of gene expression.

  * ``build_sweep.py`` initializes a parameter sweep of a specified model.

  * ``run_simulation.py`` runs an individual ``ConditionSimulation``.

  * ``run_batch.py`` runs a batch of ``ConditionSimulation`` instances.



Jupyter Notebooks
-----------------

  * ``Parameter Sweeps (Figs. S1-S4).ipynb`` walks the user through conducting a parameter sweep and visualizing the results. This notebook facilitates direct reproduction of Figures S1-S4 in our manuscript.

  * ``Repressor Pairs (Figs. 4G, 7A, and S2H).ipynb`` walk the user through conducting and analyzing simulations in which one of a pair of repressors is removed. This notebook facilitates direct reproduction of Figures 4G, 7A, and S2H in our manuscript.

  * ``Yan Expression Dynamics (Fig. 5).ipynb`` walks the user through our exploration of YanYFP expression dynamics in silico and in vivo. This notebooks enables reproduction of Figure 5 in our manuscript.

  * ``YanACT Simulations (Figs. 6A,B).ipynb`` walks the user through simulating an example scenario in which all repressors are removed. This notebooks enables reproduction of Figures 6A and 6B in our manuscript.



Additional Resources
====================


Examples
--------

For examples detailing the usage of our stochastic simulation software, please see [GeneSSA](https://github.com/sebastianbernasek/genessa).

For examples demonstrating the analysis of protein expression in the Drosophila eye, please see [FlyEye Analysis](https://github.com/sebastianbernasek/flyeye).


Contact
-------

Please contact the [Amaral Lab](https://amaral.northwestern.edu/) with any questions regarding GRaM, GeneSSA, or the FlyEye suite. We will make an effort to get back to you within a reasonable timeframe.
