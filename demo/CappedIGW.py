import math
from typing import Callable

from AbstractClasses import ReferencePolicy, LossPredictor
from AnytimeNormalizedSampler import AnytimeNormalizedSampler

#the unused arguments below (denoted by _=None) are a requirement
#of the experiment orchestration library, coba, and can be ignored
#in practice.

class CappedIGW:
    def __init__(self, *,
        mu:ReferencePolicy,
        fhat:LossPredictor,
        tau:float = 10,
        gamma_sched: Callable[[int],float] = math.sqrt,
        kappa_infty:float = 100,
        alpha:float = 0.05,
        delta_beta:float = 1e-2) -> None:
        self.gamma_sched = gamma_sched
        self.sampler     = AnytimeNormalizedSampler(tau,mu,kappa_infty,alpha,delta_beta)
        self.fhat        = fhat
        self.t           = 1

    @property
    def params(self):
        return {**self.sampler.params, **self.fhat.params}

    def predict(self, contexts, _=None):
        #The dict wrapping our prediction is a simple format hint for our experimentation harness, coba.
        return {'action_prob':[self.sampler.sample(c, self.fhat, self.gamma_sched(self.t)) for c in contexts] }

    def learn(self, contexts, actions, losses, _=None):
        self.t += len(contexts)
        self.fhat.learn(contexts, actions, losses)
