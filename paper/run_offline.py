import torch
import coba as cb

from tasks    import MyEvaluator
from oracles  import MlpArgmin, ArgminPlusDispersion
from learners import LargeActionLearner

import sys

n_processes = 8 if len(sys.argv) == 1 else int(sys.argv[1])
out_file    = "offline.gz"

if n_processes > 1:
    torch.set_num_threads(1)

if __name__ == "__main__":

    print(f"RUNNING offline with {n_processes} processes")

    log = cb.Environments.from_save("online.zip").shuffle(seed=1)
    lrn = [
        LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .004 , 500, IPS=True, v=2),
        LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .0075, 750, IPS=False,v=2)
    ]

    cb.Experiment(log,lrn,MyEvaluator()).run(out_file,processes=n_processes)
