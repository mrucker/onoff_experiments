# Infinite Action Bandits with Data Reuse

This repo contains two example projects of the [CappedIGW algorithm](https://arxiv.org/abs/2302.08551).

The first project is in the `demo` directory. This project was made specifically for new users and contains:
  1. Two demonstration notebooks that make it easy to play with CappedIGW on real world datasets
  2. Simplified and documented implementations of CappedIGW and the Betting Martingale Normalization procedure.

The second project is in the `paper` directory. This project contains code to reproduce the results in the published paper. This code is harder to read than the implementations in the `demo` directory and also includes a working implementation of the algorithm to adaptively choose $\tau$.

### Getting Started With The Demo Project:

To play with the experiments in the `demo` directory follow these steps:
  1. Download this repo to your local machine
  2. Make sure you have python installed
  3. On the command line run `pip install notebook`
  4. On the command line navigate to your download of the repo
  5. On the command line run `jupyter notebook`
  6. From your web browser open either of the notebook files in `demo`

### Getting Started With The Paper Project:

To run the experiments in the `paper` directory follow these steps:
  1. Download this repo to your local machine
  2. Make sure you have python installed
  3. On the command line navigate to your download of the repo
  4. On the command line run `pip -r ./paper/requirements.txt`
  5. On the command line run `python ./paper/run_online.py`
  6. On the command line run `python ./paper/run_offline.py`
  7. Open `plots.ipynb` to create plots from the paper
