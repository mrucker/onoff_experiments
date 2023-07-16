import torch
import coba as cb
import numpy as np
import scipy.optimize as so

from itertools import count
from normalizers import BettingNormCS

class CorralSmoothIGW:

    def __init__(self, eta, gzero, gscale, tau_min, tau_max, nalgos):

        self.lamb     = 1
        self.args     = (eta,gzero,gscale.__name__,tau_min,tau_max,nalgos)
        self.eta      = eta / nalgos
        self.gamma    = gzero
        self.gzero    = gzero
        self.gscale   = gscale
        self.taus     = torch.Tensor(np.geomspace(tau_min, tau_max, nalgos))
        self.invpalgo = torch.Tensor([self.taus.shape[0]] * self.taus.shape[0])
        self.T        = 0

    @property
    def params(self):
        return {"sampler":"old", **dict(zip(['eta','gzero','gscale','tau_min','tau_max','n_taus','v2'],self.args))}

    def update(self, algo, invprop, reward):

        reward = reward.round(decimals=8)
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward

        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)
        newinvpalgo = torch.scatter_reduce(input=self.invpalgo, dim=0, index=algo, src=weightedlosses, reduce='sum')

        # just do this calc on the cpu
        invp = newinvpalgo.cpu().numpy()
        invp += 1 - np.min(invp)
        Zlb = 0
        Zub = 1

        while (np.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2 

        root, res = so.brentq(lambda z: 1 - np.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res

        self.invpalgo = torch.Tensor(invp + root, device=self.invpalgo.device)

    def sample(self, fhatstar, ahatstar, fhat, X):

        #This is a "batch" of data.. So we need to pick a base for each of the N items in the batch
        N = fhatstar.shape[0]

        #select a base algorithm N times according to the current corral distribution
        #interestingly, this appears to be considerably faster than using random.choices
        algo = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False).sample((N,))

        invpalgo = self.invpalgo.gather(dim=0,index=algo).unsqueeze(1)
        tau      = self.taus.gather(dim=0,index=algo).unsqueeze(1)
        self.T += 1 

        self.gamma = self.gscale(self.gzero,self.gamma,self.T)

        #Sampling once from our base measure for each example in the batch.
        #Our base measure in this case is the uniform distrubtion in [0,1].
        arando = torch.rand(size=(N, 1), device=X.device)
        fhatrando = fhat(X, arando)    
        probs = tau / (tau + self.gamma*(fhatstar-fhatrando))        
        unif  = torch.rand(size=(N, 1), device=X.device)

        #shouldexplore is a Bernoulli random variable 
        #with mean m_t(a) for each action in the batch
        shouldexplore = (unif <= probs).long()

        #This is ahatstar when shouldexplore is 0 otherwise it is a_rng
        selected_action = (ahatstar + shouldexplore * (arando - ahatstar)).squeeze(1).tolist()
        selected_density = [ p[0].item() if e else None for p,e in zip(probs,shouldexplore) ]

        return selected_action, selected_density, algo, invpalgo

class CorralCappedIGW:

    def __init__(self, eta, gzero, gscale, tau_min, tau_max, nalgos, kappa_infty=1):

        self.lamb        = 1
        self.args        = (eta,gzero,gscale.__name__,tau_min,tau_max,nalgos,kappa_infty)
        self.eta         = eta / nalgos
        self.gamma       = gzero
        self.gzero       = gzero
        self.gscale      = gscale
        self.taus        = torch.Tensor(np.geomspace(tau_min, tau_max, nalgos))
        self.invpalgo    = torch.Tensor([self.taus.shape[0]] * self.taus.shape[0])
        self.T           = 0
        self.kappa_infty = kappa_infty

    @property
    def params(self):
        return {"sampler":"new", **dict(zip(['eta','gzero','gscale','tau_min','tau_max','n_taus','k_inf'],self.args))}

    def update(self, algo, invprop, reward):

        reward = reward.round(decimals=8)
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward

        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)
        newinvpalgo = torch.scatter_reduce(input=self.invpalgo, dim=0, index=algo, src=weightedlosses, reduce='sum')

        # just do this calc on the cpu
        invp = newinvpalgo.cpu().numpy()
        invp += 1 - np.min(invp)
        Zlb = 0
        Zub = 1

        while (np.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2 

        root, res = so.brentq(lambda z: 1 - np.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res

        self.invpalgo = torch.Tensor(invp + root, device=self.invpalgo.device)

    def sample(self, fhatstar, ahatstar, fhat, X):

        #This is a "batch" of data.. So we need to pick a base for each of the N items in the batch
        N = fhatstar.shape[0]

        algos    = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False).sample((N,))
        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1), dim=1, index=algos.unsqueeze(1))
        tau      = torch.gather(input=self.taus.unsqueeze(0).expand(N, -1), dim=1, index=algos.unsqueeze(1))

        selected_actions  = []
        selected_density  = []

        self.T += 1 

        self.gamma = self.gscale(self.gzero,self.gamma,self.T)
        
        for i,t,x,A,F in zip(count(),tau,X,ahatstar,fhatstar):
            t = t.item()

            infinium = (fhat(x,0) if A > .5 else fhat(x,1)).item()
            supremum = F.item()

            if self.kappa_infty == 1:
                beta = self.find_beta_montecarlo(fhat,self.gamma,x,t,infinium,supremum,1000,1)
            else:
                beta = self.find_beta_martingale(fhat,self.gamma,x,t,supremum)

            m_t        = lambda a: t/(self.lamb+self.gamma*torch.clamp((supremum-fhat(x,a))-beta,min=0))
            normalizer = m_t(A).item()

            actions   = self.base_action_sampler(5000)
            densities = m_t(actions).squeeze()
            keep      = np.random.rand() < densities/normalizer

            if not keep.any():
                selected_actions.append(A.item())
                selected_density.append(None)
            else:
                index = keep.nonzero()[0].item()
                selected_actions.append(actions[index].item())
                selected_density.append(densities[index].item())

        return selected_actions, selected_density, algos, invpalgo

    def base_action_sampler(self, n):
        return torch.rand(size=(n,1))

    def find_beta_montecarlo(self,fhat,gamma,x,tau,inf,sup,n,target):
        f = sup-fhat(x.repeat(n,1), torch.arange(0,1,1/n).unsqueeze(1)).cpu().numpy()
        B = lambda beta: target-tau/n*np.sum(1/(self.lamb+gamma*np.clip(f-beta,0,None)))
        beta, res = so.brentq(B, -tau/gamma, (sup-inf), full_output=True)
        assert res.converged, f"We failed to find a root: {res}."

        return beta

    def find_beta_martingale(self,fhat,gamma,x,tau,sup):

        clip_min = np.core.umath.maximum

        alpha = .05
        g     = lambda f,beta: tau/clip_min(f-beta,self.lamb)
        cs    = BettingNormCS(g=g, tau=tau, gamma=gamma, alpha=alpha, lb=1/self.kappa_infty)

        T  = 10_000 # we shouldn't ever hit this but just in case...
        fs = []

        def batched_base_action_sampler():
            while True:
                yield from fhat(x,self.base_action_sampler(100)).squeeze().tolist()

        for t,f in zip(range(T),batched_base_action_sampler()):

            fs.append(f)
            cs.addobs(1+gamma*(f))

            if t % 10 == 0:
                cs.updatelowercs()
                cs.updateuppercs()
                l, u = cs.getci()
                if l > u: break

        print(t)

        if 'samples' not in cb.CobaContext.learning_info:
            cb.CobaContext.learning_info['samples'] = []
        cb.CobaContext.learning_info['samples'].append(t)
        
        return min(u,l) + torch.rand(size=(1,)).item()*abs(u-l)
