import math
from typing import Callable

from AbstractClasses import ReferencePolicy, LossPredictor
from AnytimeNormalizedSampler import AnytimeNormalizedSampler

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

    def predict(self, contexts, _):
        return [self.sampler.sample(c, self.fhat, self.gamma_sched(self.t)) for c in contexts]

    def learn(self, contexts, _1, actions, losses, _2):
        self.t += len(contexts)
        self.fhat.learn(contexts, actions, losses)
