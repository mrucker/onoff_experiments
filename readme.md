# Continuous Action Bandits with Data Reuse

This project evaluates the performance of a new large action CB algorithm.

This algorithm has competitive online performance while also generating data exhaust that is more useful for off-policy learning.

The original paper describing the algorithms can be found at <paper_url_to_be_added>.

## Repo Structure:

+ ./outcomes -- a directory that experimental results will be written to
+ ./notebooks/paper.ipynb -- a Jupyter notebook which recreates the plots in the paper
+ ./run_online.py -- This will run the online experiments and generate the data exhaust.
+ ./run_offline.py -- This will run the offline experiments (run the online experiments first to generate data).

## Dependencies
+ Pytorch    (see the pytorch website)
+ Scipy      (pip install scipy)
+ Matplotlib (pip install matplotlib)
+ Numpy      (pip install numpy)
+ Coba       (pip install coba==6.2.6)
