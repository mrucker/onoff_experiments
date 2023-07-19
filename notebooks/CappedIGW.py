import math

from AbstractClasses import ReferencePolicy, RewardPredictor, GammaScheduler
from AnytimeNormalizedSampler import AnytimeNormalizedSampler

from coba.primitives import Batch

class CappedIGW:
    def __init__(self, *,
        mu:ReferencePolicy,
        fhat:RewardPredictor,
        tau:float = 100,
        gamma_scheduler: GammaScheduler = lambda t: math.sqrt(t), 
        kappa_infty:float=24,
        alpha:float=0.05,
        delta_beta:float=1e-2) -> None:
        self.gamma_scheduler = gamma_scheduler
        self.sampler         = AnytimeNormalizedSampler(tau,mu,kappa_infty,alpha,delta_beta)
        self.fhat            = fhat
        self.t               = 1

    @property
    def params(self):
        return {**self.sampler.params, **self.fhat.params}

    def predict(self, context, _):
        if isinstance(context,Batch):
            return [self.sampler.sample(c, self.fhat, self.gamma_scheduler(self.t)) for c in context]
        else:
            return self.sampler.sample(context, self.fhat, self.gamma_scheduler(self.t))

    def learn(self, contexts, _1, actions, rewards, _2):
        self.t += len(contexts) if isinstance(contexts,Batch) else 1
        self.fhat.learn(contexts, actions, rewards)