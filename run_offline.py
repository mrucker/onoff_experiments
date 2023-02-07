import coba as cb

from itertools import repeat

from tasks    import MyEvaluator
from oracles  import MlpArgmax, ArgmaxPlusDispersion
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

if __name__ == "__main__":

    logs = cb.Environments.from_save("./outcomes/online.zip")

    ips_learner = LargeActionLearner(None, ArgmaxPlusDispersion(argmaxblock=MlpArgmax()), .004 , 500, IPS=True, v=2)
    dir_learner = LargeActionLearner(None, ArgmaxPlusDispersion(argmaxblock=MlpArgmax()), .0075, 750, IPS=False,v=2)

    logs_old            = cb.Environments([e for e in logs if e.params['learner']['sampler']=='old'                                        ])
    logs_new_one_fourth = cb.Environments([e for e in logs if e.params['learner']['sampler']=='new' and e.params['learner']['k_inf'] == 1/4])

    logs_old            = list(map(MyEnv,logs_old           .shuffle(n=1)))
    logs_new_one_fourth = list(map(MyEnv,logs_new_one_fourth.shuffle(n=1)))

    pairs = []

    pairs.extend(zip(repeat(MyLrn(dir_learner,{"sampler":"old",              })), logs_old))
    pairs.extend(zip(repeat(MyLrn(ips_learner,{"sampler":"new","k_inf":"1/4" })), logs_new_one_fourth))

    pairs = sorted(pairs, key=lambda pair: hash(pair[1]))

    config = {"processes":8}
    log = "./outcomes/offline.gz"
    cb.Experiment(pairs, evaluation_task=MyEvaluator(True)).config(**config).run(log)
