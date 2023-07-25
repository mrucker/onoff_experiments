from math import log
from typing import Tuple, Any

import numpy

from AbstractClasses import ReferencePolicy, LossPredictor
from BettingMartingale import BettingMartingale

class AnytimeNormalizedSampler:

    def __init__(self, tau: float, mu: ReferencePolicy, kappa_infty: float, alpha:float, delta_beta:float):

        assert tau >= 1
        assert kappa_infty >= 1
        assert delta_beta >= 0

        self.tau         = tau
        self.mu          = mu
        self.kappa_infty = kappa_infty
        self.alpha       = alpha
        self.delta_beta  = delta_beta

    @property
    def params(self):
        return {
            'tau': self.tau,
            'kappa': self.kappa_infty,
            'alpha': self.alpha,
            'delta_beta': self.delta_beta,
            **self.mu.params
        }

    def sample(self, context, fhat:LossPredictor, gamma:float) -> Tuple[Any,float]:
        sampler = self.mu.sample(context)
        beta = self._calculate_beta(context, fhat, gamma, sampler)

        for actions in sampler:
            fhats        = numpy.array(fhat.predict(context,actions))
            accept_probs = 1/(1+gamma*numpy.clip(fhats-beta,a_min=0,a_max=None))
            accept       = accept_probs >= numpy.random.rand(len(actions))

            if accept.any():
                accept_index = accept.nonzero()[0][0]
                return actions[accept_index], self.tau*accept_probs[accept_index]

    def _calculate_beta(self, context, fhat, gamma, sampler):
        g = lambda z: self.tau/(1+gamma*max(z,0))
        N = 1+int(10*self.tau*log(gamma/self.alpha))

        martingale = BettingMartingale(
            g=g,
            beta_min=(1-self.tau)/gamma,
            beta_max=1,
            delta_beta=self.delta_beta,
            kappa=self.kappa_infty,
            g_max=self.tau,
            alpha=self.alpha
        )

        n = 0
        for batch in sampler:
            fhats = fhat.predict(context,batch)

            for f in fhats:
                n += 1
                martingale.addobs(f)

            l,h = martingale.getci()
            if l + self.delta_beta > h: break
            if n > N: break

        return min(martingale.getci())

if __name__ == '__main__':
    import numpy as np
    import timeit

    def test_once(gen,fails):
        class UniformReferencePolicy:
            def sample(self,context):
                while True:
                    yield gen.uniform(0,1,size=(100,))

        tau = gen.uniform(50,75)
        class Fhat:
            def predict(self, _context, actions):
                return np.array(2 * tau * actions > 1, dtype=float)
        fhat = Fhat()
        gamma = gen.uniform(2,10)
        kappa_infty = 24
        alpha = .05
        delta_beta = 1e-2

        ns = AnytimeNormalizedSampler(tau,UniformReferencePolicy(),kappa_infty,alpha,delta_beta)
        densities = [ns.sample(None,fhat,gamma)[1] for _ in range(1000)]

        mean_density = np.mean(densities)
        if not (1/kappa_infty < mean_density and mean_density < 1):
            fails.append(1)

    gen = np.random.default_rng(45)
    number = 10
    fails = []
    avruntime = timeit.timeit(lambda: test_once(gen, fails), number=number) / number
    print({ 'fail_rate': len(fails) / number, 'average runtime (ms)': round(1000 * avruntime, 1) })
