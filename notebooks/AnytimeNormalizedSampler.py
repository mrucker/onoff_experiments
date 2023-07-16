
from more_itertools import chunked
from math import log
from typing import Callable, Any

from AbstractClasses import ReferencePolicy
from BettingMartingale import BettingMartingale

class AnytimeNormalizedSampling:

    def __init__(self,tau: float, mu: ReferencePolicy, kappa_infty: float, batch_size:int, alpha:float) -> None:

        assert tau >= 1
        assert kappa_infty >= 1

        self.tau         = tau
        self.mu          = mu
        self.kappa_infty = kappa_infty
        self.alpha       = alpha
        self.batch_size  = batch_size

    @property
    def params(self):
        return {'tau': self.tau, 'kappa': self.kappa_infty, 'alpha': self.alpha, **self.mu.params}

    def sample(self, contexts, fhat:Callable[[Any],float], gamma:float):
        return zip(*map(lambda c:self._sample_one(c,fhat,gamma),contexts))
        
    def _sample_one(self,context,fhat,gamma):
        n_max = 1+int(10*self.tau*log(gamma/self.alpha))
        sampler = chunked(self.mu.sample(context), self.batch_size)

        for n in range(1+n_max//self.batch_size):

            batch = next(sampler)
            fhats = fhat(context,batch)



        beta_martingale = 

        pass
        #return (action,density)

        