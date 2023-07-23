import torch
import coba as cb

from itertools import repeat

from tasks    import MyEvaluator
from oracles  import MlpArgmin, ArgminPlusDispersion
from learners import LargeActionLearner

n_processes = 8
out_file    = "offline.gz"

if n_processes > 1:
    torch.set_num_threads(1)

if __name__ == "__main__":

    log = cb.Environments.from_save("./outcomes/online.zip").shuffle(seed=1)
    lrn = [
        LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .004 , 500, IPS=True, v=2),
        LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .0075, 750, IPS=False,v=2)
    ]

    cb.Experiment(log,lrn,MyEvaluator(True)).run(log)
