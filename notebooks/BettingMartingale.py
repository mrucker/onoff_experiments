import timeit
from math import log1p,log
import scipy.optimize as so

#TODO: Memoize the logwealth between beta changes to make the short circuit O(1)

class BettingMartingale:
    def __init__(self, *, g, beta_min, beta_max, delta_beta, kappa, g_max, alpha):

        assert delta_beta > 0
        assert beta_min < beta_max
        assert kappa > 1
        assert 0 < alpha and alpha < 1
        assert g_max > 1

        #g is assumed to be non-negative, which is true for CappedIGW

        self.g = g
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta_beta = delta_beta
        self.kappa = kappa
        self.g_max = g_max
        self.alpha = alpha

        self.t = 0
        self.alist = []
        self.lamlist = [0]
        self.nulist = [0]
        self.lamgrad = 0
        self.nugrad = 0
        self.betaminus = beta_min
        self.betaplus = beta_max
        self.thres = -log(alpha/2)
        self.lammin = 0
        self.lammax = 1/2*1/(self.g_max-1)
        self.lamD = self.lammax - self.lammin
        self.lamG = max([1/(1+self.lammin), 2*(self.g_max-1)])
        self.numin = 0
        self.numax = self.kappa/2
        self.nuD = self.lammax - self.lammin
        self.nuG = max([2/self.kappa , 2*(self.g_max-1/self.kappa)])

        self._stale = False

    def betlowercs(self):
        g = self.g(self.alist[-1], self.betaminus)
        nabla = (1 - g) / (1 + self.lamlist[-1] * (1 - g))
        self.lamgrad += nabla**2
        b = 1 / (4 * self.lamG * self.lamD)
        epsilon = 1 / (b**2 * self.lamD**2)

        ytp1 = self.lamlist[-1] + nabla / (b * (epsilon + self.lamgrad))
        xtp1 = max(self.lammin, min(self.lammax, ytp1))
        self.lamlist.append(xtp1)

    def betuppercs(self):
        g = self.g(self.alist[-1], self.betaplus)
        nabla = (g - 1/self.kappa) / (1 + self.nulist[-1] * (g - 1/self.kappa))
        self.nugrad += nabla**2
        b = 1 / (4 * self.nuG * self.nuD)
        epsilon = 1 / (b**2 * self.nuD**2)

        ytp1 = self.nulist[-1] + nabla / (b * (epsilon + self.nugrad))
        xtp1 = max(self.numin, min(self.numax, ytp1))
        self.nulist.append(xtp1)

    def addobs(self, a):
        self._stale = True

        self.alist.append(a)
        self.t += 1

        self.betlowercs()
        self.betuppercs()

    def uppercswealth(self, beta):
        s = 0
        for bet,a in zip(self.nulist,self.alist):
            s += log1p(bet*(self.g(a,beta)-1/self.kappa))
        return s

    def _updateuppercs(self):
        maxbeta = self.betaplus-self.delta_beta

        if maxbeta <= self.betaminus:
            return
        
        maxbetawealth = self.uppercswealth(maxbeta)
        if maxbetawealth < self.thres:
            return

        minbeta = self.beta_min
        minbetawealth = self.uppercswealth(minbeta)
        if minbetawealth > self.thres:
            self.betaplus = self.beta_min
            return

        res = so.root_scalar(
            f = lambda beta: self.uppercswealth(beta) - self.thres, 
            xtol=self.delta_beta/10, 
            method = 'brentq', 
            bracket = [minbeta, maxbeta])
        assert res.converged, res
        self.betaplus = res.root

    def lowercswealth(self, beta):
        s = 0
        for bet,a in zip(self.lamlist,self.alist):
            s += log1p(bet*(1-self.g(a,beta)))
        return s

    def _updatelowercs(self):
        minbeta = self.betaminus + self.delta_beta

        if minbeta >= self.betaplus:
            return

        minbetawealth = self.lowercswealth(minbeta)
        if minbetawealth < self.thres:
            return

        maxbeta = self.beta_max
        maxbetawealth = self.lowercswealth(maxbeta)
        if maxbetawealth > self.thres:
            self.betaminus = self.beta_max
            return

        res = so.root_scalar(
            f = lambda beta: self.lowercswealth(beta)-self.thres, 
            xtol=self.delta_beta/10,
            method = 'brentq',
            bracket = [ minbeta, maxbeta ])
        assert res.converged, res
        self.betaminus = res.root

    def getci(self):
        if self._stale:
            self._updatelowercs()
            self._updateuppercs()
            self._stale = False
        return self.betaminus, self.betaplus

if __name__ == '__main__':
    import numpy as np

    def test_once(gen):

        tau = gen.uniform(2,1_000)
        fhat = lambda a: 1 if 2*a*tau > 1 else 0
        integrate = lambda tau,beta,gamma : (2 * tau - 1) / (2 + 2 * gamma * (1 - beta)) + 1 / (2 + 2 * gamma * max(0, -beta))
        gamma = gen.uniform(2,1_000)
        kappa_infty = 24
        beta_min = (1-tau)/gamma
        beta_max = 1
        g_max = tau
        alpha = .05
        n_max = 1+int(10*tau*log(gamma/alpha))
        delta_beta =  1e-2

        g = lambda a,beta: tau/(1+gamma*max(fhat(a)-beta,0))
        cs = BettingMartingale(g=g, beta_min=beta_min, beta_max=beta_max, delta_beta=delta_beta, kappa=kappa_infty, g_max=g_max, alpha=alpha)

        for n in range(n_max):
            a = gen.random()
            cs.addobs(a)
            if n & (n-1) == 0:
                l,h = cs.getci()
                #print((n_max, n, l, integrate(tau,l,gamma), h,integrate(tau,h,gamma)))
                if l + delta_beta > h: break

        l,h = cs.getci()
        beta_hat = (l+h)/2
        integral = integrate(tau,beta_hat,gamma)
        assert 1/kappa_infty < integral and integral < 1, (integral, tau, beta_hat, gamma)

    gen = np.random.default_rng(2)

    print(f"test pass took {timeit.timeit(lambda: test_once(gen), number=10)} seconds")
