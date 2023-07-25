from math import sqrt

import torch
import coba as cb

from oracles import ArgminPlusDispersion

class LargeActionLearner:
    def __init__(self, sampler, fhat: ArgminPlusDispersion, initlr, tzero, IPS=False, v=1):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.sampler   = sampler
        self.fhat      = fhat
        self.initlr    = initlr
        self.tzero     = tzero
        self.opt       = None
        self.scheduler = None
        self.IPS       = IPS
        self.v         = v
        self.t         = 0

    @property
    def params(self):
        samp_params = self.sampler.params if self.sampler else {}
        return {**samp_params, **self.fhat.params, "lr":self.initlr, "tz":self.tzero, "weight":self.IPS, "v": self.v }

    def initialize(self,in_features):
        torch.manual_seed(1)
        self.fhat.in_features(in_features)
        self.opt = torch.optim.Adam((p for p in self.fhat.parameters() if p.requires_grad ), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:sqrt(self.tzero)/sqrt(self.tzero+t))

    def predict(self, context, actions):
        context = torch.tensor(context,dtype=torch.float32)
        if not self.opt: self.initialize(context.shape[1])

        with torch.no_grad():
            if self.sampler is None:
                actions = self.fhat.argmin(context).flatten().tolist()
                density = [None]*len(actions)
                return actions,density
            else:            
                ahatstar = self.fhat.argmin(context)
                fhatstar = self.fhat(context,ahatstar)
                actions, dense, algo, invpalgo = self.sampler.sample(self.fhat, context, fhatstar, ahatstar)
                fhats = self.fhat(context,torch.tensor(actions).unsqueeze(1)).flatten()

                cb.CobaContext.learning_info["ahatgreedy"] = ahatstar.flatten().tolist()
                cb.CobaContext.learning_info["fhatgreedy"] = fhatstar.flatten().tolist()
                cb.CobaContext.learning_info["fhatpolicy"] = fhats.tolist()

                return actions, dense, {'algo':algo, 'invpalgo':invpalgo}

    def learn(self, context, actions, action, reward, probs, **kwargs):
        self.t+= 1
        action  = torch.tensor(action).unsqueeze(1)
        context = torch.tensor(context,dtype=torch.float32)
        reward  = torch.tensor(reward).unsqueeze(1)

        if kwargs: algo, invpalgo = kwargs['algo'],kwargs['invpalgo']

        if not self.opt: self.initialize(context.shape[1])

        weights = 1 if self.IPS else torch.tensor([min(1./p,5) if p else 5 for p in probs]).unsqueeze(1)

        for _ in range(self.v):
            self.opt.zero_grad()
            score = self.fhat(context, action)
            loss  = self.loss(score, (1-reward))*weights
            loss.mean().backward()
            self.opt.step()

        self.scheduler.step()

        cb.CobaContext.learning_info['regressor_loss'] = loss.flatten().tolist()

        if self.sampler:
            with torch.no_grad():
                self.sampler.update(algo, invpalgo, reward)
