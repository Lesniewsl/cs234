# cs234
CS 234 - Reinforcement Learning Project

This repository hosts the core files used for the CS 234 project paper "Practical Improvements on the Warfarin Dosing Process Through Stochastic Linear Bandits". In particular it contains:

- the data pre-processing code (all xxx_data.py)
- the code for baseline metrics simulation (baseline.py)
- the code for the LinUCB algorithm structure, based on Chu et al. (2011), along with a test run (linucb.py)
- the code for the Lasso bandit algorithm structure, based on Bastani and Bayati (2015), along with a test run (lasso.py)
- an framework used to generate the paper's plots, along with an test example generating LinUCB regret vs. fixed dosing regret (plot.py)

For conciseness purposes, this repository does not contain the various modifications and back-and-forth code changes referenced in the paper (e.g. modification of the LinUCB reward function). However, every experiment conducted in the paper is based entirely on these core files.
