import timeit
from math import log1p,log
import scipy.optimize as so

class LowerMartingale:
    def __init__(self, g, beta_min, beta_max, delta_beta, kappa, alpha):
        from collections import defaultdict
        from math import log
        
        assert beta_min < beta_max
        assert delta_beta > 0
        assert kappa > 1
        assert 0 < alpha < 1
        
        self.g = g
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta_beta = delta_beta
        self.kappa = kappa
        self.alpha = alpha
        
        self.stats = defaultdict(lambda: [0, 0, 0])
        self.curnu = 0
        self.phicurnu = self._phi(self.curnu)
        self.maxnu = (1/2) * kappa
        self.curbeta = beta_max
        self.thres = -log(alpha)
        self._updatecurbetatest(max(beta_max - delta_beta, beta_min))
        self.betaminlogwealth = 0
        
    def addobs(self, fa):
        from math import ceil
        fu = int(ceil(fa / self.delta_beta))
        self.stats[fu][0] += 1
        self.stats[fu][1] += self.curnu
        self.stats[fu][2] += self.phicurnu
        
        glmin = self.g(self.delta_beta * fu - self.beta_min)
        self.betaminlogwealth += self.curnu * (glmin - 1/self.kappa) - self.phicurnu * (glmin - 1/self.kappa)**2
        
        gltest = self.g(self.delta_beta * fu - self.curbetatest)
        self.curbetatestlogwealth += self.curnu * (gltest - 1/self.kappa) - self.phicurnu * (gltest - 1/self.kappa)**2
        self.curbetatestnum += (gltest - 1/self.kappa)
        self.curbetatestdenom += (gltest - 1/self.kappa)**2
        
        if self.curbetatestnum <= 0:
            self.curnu = 0
        elif self.curbetatestnum <= self.maxnu * (self.curbetatestnum + self.curbetatestdenom):
            self.curnu = self.curbetatestnum / (self.curbetatestnum + self.curbetatestdenom)
        else:
            self.curnu = self.maxnu
            
        self.phicurnu = self._phi(self.curnu)

    def getbeta(self):
        if self.curbetatestlogwealth > self.thres:
            self._updatecs()
        return self.curbeta
    
    def _phi(self, x):
        from math import log1p
        return -x - log1p(-x)
        
    def _updatecurbetatest(self, beta):
        logw, num, denom = 0, 0, 0
        for fu, stats in self.stats.items():
            gl = self.g(self.delta_beta * fu - beta)
            logw += stats[1] * (gl - 1/self.kappa) - stats[2] * (gl - 1/self.kappa)**2
            num += stats[0] * (gl - 1/self.kappa)
            denom += stats[0] * (gl - 1/self.kappa)**2
            
        self.curbetatest = beta
        self.curbetatestlogwealth = logw
        self.curbetatestnum = num
        self.curbetatestdenom = denom       
    
    def _updatecs(self):
        from scipy import optimize as so
        if self.curbetatest <= self.beta_min:
            return
        
        maxbeta = self.curbetatest
        maxbetawealth = self.curbetatestlogwealth
        if maxbetawealth < self.thres:
            return
        
        minbeta = self.beta_min
        minbetawealth = self.betaminlogwealth
        if minbetawealth >= self.thres:
            self.curbeta = self.beta_min
            self._updatecurbetatest(self.beta_min)
            return
        
        def logwealth(beta):
            logw = 0
            for fu, stats in self.stats.items():
                gl = self.g(self.delta_beta * fu - beta)
                logw += stats[1] * (gl - 1/self.kappa) - stats[2] * (gl - 1/self.kappa)**2
            return logw
                        
        res = so.root_scalar(
            f = lambda beta: logwealth(beta)-self.thres, 
            xtol = self.delta_beta/10,
            method = 'brentq',
            bracket = [ minbeta, maxbeta ])
        assert res.converged, res
        self.curbeta = res.root
        self._updatecurbetatest(max(self.curbeta - self.delta_beta, self.beta_min))

class UpperMartingale:
    def __init__(self, g, beta_min, beta_max, delta_beta, g_max, alpha):
        from collections import defaultdict
        from math import log
        
        assert beta_min < beta_max
        assert delta_beta > 0
        assert g_max >= 1
        assert 0 < alpha < 1
        
        self.g = g
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta_beta = delta_beta
        self.g_max = g_max
        self.alpha = alpha
        
        self.stats = defaultdict(lambda: [0, 0, 0])
        self.curnu = 0
        self.phicurnu = self._phi(self.curnu)
        self.maxnu = (1/2) / (g_max - 1)
        self.curbeta = beta_min
        self.thres = -log(alpha)
        self._updatecurbetatest(min(beta_min + delta_beta, beta_max))
        self.betamaxlogwealth = 0
        
    def addobs(self, fa):
        fl = int(fa / self.delta_beta)
        self.stats[fl][0] += 1
        self.stats[fl][1] += self.curnu
        self.stats[fl][2] += self.phicurnu
        
        gumax = self.g(self.delta_beta * fl - self.beta_max)
        self.betamaxlogwealth += self.curnu * (1 - gumax) - self.phicurnu * (1 - gumax)**2
        
        gutest = self.g(self.delta_beta * fl - self.curbetatest)
        self.curbetatestlogwealth += self.curnu * (1 - gutest) - self.phicurnu * (1 - gutest)**2
        self.curbetatestnum += (1 - gutest)
        self.curbetatestdenom += (1 - gutest)**2
        
        if self.curbetatestnum <= 0:
            self.curnu = 0
        elif self.curbetatestnum <= self.maxnu * (self.curbetatestnum + self.curbetatestdenom):
            self.curnu = self.curbetatestnum / (self.curbetatestnum + self.curbetatestdenom)
        else:
            self.curnu = self.maxnu
            
        self.phicurnu = self._phi(self.curnu)

    def getbeta(self):
        if self.curbetatestlogwealth > self.thres:
            self._updatecs()
        return self.curbeta
    
    def _phi(self, x):
        from math import log1p
        return -x - log1p(-x)
        
    def _updatecurbetatest(self, beta):
        logw, num, denom = 0, 0, 0
        for fl, stats in self.stats.items():
            gu = self.g(self.delta_beta * fl - beta)
            logw += stats[1] * (1 - gu) - stats[2] * (1 - gu)**2
            num += stats[0] * (1 - gu)
            denom += stats[0] * (1 - gu)**2
            
        self.curbetatest = beta
        self.curbetatestlogwealth = logw
        self.curbetatestnum = num
        self.curbetatestdenom = denom       
    
    def _updatecs(self):
        from scipy import optimize as so
        if self.curbetatest >= self.beta_max:
            return
        
        minbeta = self.curbetatest
        minbetawealth = self.curbetatestlogwealth
        if minbetawealth < self.thres:
            return
        
        maxbeta = self.beta_max
        maxbetawealth = self.betamaxlogwealth
        if maxbetawealth >= self.thres:
            self.curbeta = self.beta_max
            self._updatecurbetatest(self.beta_max)
            return
        
        def logwealth(beta):
            logw = 0
            for fl, stats in self.stats.items():
                gu = self.g(self.delta_beta * fl - beta)
                logw += stats[1] * (1 - gu) - stats[2] * (1 - gu)**2
            return logw
                
        res = so.root_scalar(
            f = lambda beta: logwealth(beta)-self.thres, 
            xtol = self.delta_beta/10,
            method = 'brentq',
            bracket = [ minbeta, maxbeta ])
        assert res.converged, res
        self.curbeta = res.root
        self._updatecurbetatest(min(self.curbeta + self.delta_beta, self.beta_max))

class BettingMartingale:
    def __init__(self, *, g, beta_min, beta_max, delta_beta, g_max, kappa, alpha):
        self.lmart = LowerMartingale(g, beta_min, beta_max, delta_beta/2, kappa=kappa, alpha=alpha/2)
        self.umart = UpperMartingale(g, beta_min, beta_max, delta_beta/2, g_max=g_max, alpha=alpha/2)
        
    def addobs(self, fa):
        self.lmart.addobs(fa)
        self.umart.addobs(fa)
        
    def getci(self):
        return [ self.umart.getbeta(), self.lmart.getbeta() ]

if __name__ == '__main__':
    import numpy as np

    def test_once(gen,fails):

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
        delta_beta =  1e-3

        g = lambda z: tau/(1+gamma*max(z,0))
        cs = BettingMartingale(g=g, beta_min=beta_min, beta_max=beta_max, delta_beta=delta_beta, kappa=kappa_infty, g_max=g_max, alpha=alpha)

        for n in range(n_max):
            a = gen.random()
            cs.addobs(fhat(a))
            l,h = cs.getci()
            if l + delta_beta > h: break

        l,h = cs.getci()
        beta_hat = l
        integral = integrate(tau,beta_hat,gamma)

        if not (1/kappa_infty < integral and integral < 1):
            fails.append((integral, tau, beta_hat, gamma))

    gen = np.random.default_rng(45)
    number = 1000
    fails = []
    avruntime = timeit.timeit(lambda: test_once(gen, fails), number=number) / number
    print({ 'fail_rate': len(fails) / number, 'average runtime (ms)': round(1000 * avruntime, 1) })
