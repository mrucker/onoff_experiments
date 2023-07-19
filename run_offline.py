import torch
import coba as cb

from itertools import repeat

from tasks    import MyEvaluator
from oracles  import MlpArgmin, ArgminPlusDispersion
from learners import LargeActionLearner

class MyEnv:
    def __init__(self,env):
        self._env   = env
        self._ident = (self._env.params.get('openml_data'),self._env.params.get('openml_task'),self._env.params['shuffle1'],self._env.params['shuffle2'])
        self._hash  = hash(self._ident)
    @property
    def params(self):
        params = dict(self._env.params)
        params.pop('learner')
        return params
    def read(self):
        return self._env.read()
    def __hash__(self) -> int:
        return self._hash
    def __eq__(self, o):
        return self._ident == o._ident

class MyLrn:
    def __init__(self,lrn,param) -> None:
        self._lrn   = lrn
        self._param = param
    
    @property
    def params(self):
        return {**self._lrn.params, **self._param}

    def learn(self,*args,**kwargs):
        self._lrn.learn(*args,**kwargs)

    def predict(self,*args,**kwargs):
        return self._lrn.predict(*args,**kwargs)

#We control threads/processes explicitly in coba.
#Because of this Pytorch's implicit threading can soak our cores.
torch.set_num_threads(1)

if __name__ == "__main__":

    logs = cb.Environments.from_save("./outcomes/online.zip")

    ips_learner = LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .004 , 500, IPS=True, v=2)
    dir_learner = LargeActionLearner(None, ArgminPlusDispersion(argminblock=MlpArgmin()), .0075, 750, IPS=False,v=2)

    logs_old    = cb.Environments([e for e in logs if e.params['learner']['sampler']=='old'                                        ])
    logs_new_2  = cb.Environments([e for e in logs if e.params['learner']['sampler']=='new' and e.params['learner']['k_inf'] == 2])
    logs_new_4  = cb.Environments([e for e in logs if e.params['learner']['sampler']=='new' and e.params['learner']['k_inf'] == 4])
    logs_new_24 = cb.Environments([e for e in logs if e.params['learner']['sampler']=='new' and e.params['learner']['k_inf'] == 24])

    logs_old    = list(map(MyEnv,logs_old   .shuffle(n=1)))
    logs_new_2  = list(map(MyEnv,logs_new_2 .shuffle(n=1)))
    logs_new_4  = list(map(MyEnv,logs_new_4 .shuffle(n=1)))
    logs_new_24 = list(map(MyEnv,logs_new_24.shuffle(n=1)))

    pairs = []

    pairs.extend(zip(repeat(MyLrn(dir_learner,{"sampler":"old",            })), logs_old))
    pairs.extend(zip(repeat(MyLrn(ips_learner,{"sampler":"new","k_inf":"2" })), logs_new_2))
    pairs.extend(zip(repeat(MyLrn(ips_learner,{"sampler":"new","k_inf":"4" })), logs_new_4))
    pairs.extend(zip(repeat(MyLrn(ips_learner,{"sampler":"new","k_inf":"24"})), logs_new_24))

    pairs = sorted(pairs, key=lambda pair: hash(pair[1]))

    config = {"processes":8}
    log = "./outcomes/offline.gz"
    cb.Experiment(pairs, evaluation_task=MyEvaluator(True)).config(**config).run(log)
