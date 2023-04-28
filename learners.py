import coba as cb
import torch
from math import sqrt

class LargeActionLearner:
    def __init__(self, sampler, model: torch.nn.Module, initlr, tzero, IPS=False, v=1):

        self.loss = torch.nn.MSELoss(reduction='none')

        self.sampler   = sampler
        self.model     = model
        self.initlr    = initlr
        self.tzero     = tzero
        self.opt       = None
        self.scheduler = None
        self.IPS       = IPS
        self.v         = v

    @property
    def params(self):
        samp_params = self.sampler.params if self.sampler else {}
        return {**samp_params, **self.model.params, "lr":self.initlr, "tz":self.tzero, "weight":self.IPS, "v": self.v }

    def initialize(self,in_features):
        torch.manual_seed(1)
        self.model.in_features(in_features)
        self.opt = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad ), lr=self.initlr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda t:sqrt(self.tzero)/sqrt(self.tzero+t))

    def predict(self, context, actions):
        context = torch.tensor(context,dtype=torch.float32)
        if not self.opt: self.initialize(context.shape[1])

        with torch.no_grad():
            ahatstar = self.model.argmax(context)
            fhatstar = self.model.max().expand(len(actions),-1)

            cb.CobaContext.learning_info["greedy"] = ahatstar.flatten().tolist()

            if self.sampler:
                actions, dense, algo, invpalgo = self.sampler.sample(fhatstar, ahatstar, self.model, context)
                kwargs = {'algo':algo, 'invpalgo':invpalgo}
            else:
                actions = ahatstar.squeeze(1).tolist()
                dense   = [None]*len(context)
                kwargs  = {}

            return actions, dense, kwargs

    def learn(self, context, actions, action, reward, probs, **kwargs):
        action  = torch.tensor(action).unsqueeze(1)
        context = torch.tensor(context,dtype=torch.float32)
        reward  = torch.tensor(reward).unsqueeze(1)

        if not self.opt: self.initialize(context.shape[1])

        weights = torch.tensor([min(1./p,5) if p and self.IPS else 5 for p in probs]).unsqueeze(1)

        for _ in range(self.v):
            self.opt.zero_grad()
            score = self.model(context, action)
            loss  = self.loss(score, reward)*weights
            loss.mean().backward()
            self.opt.step()

        self.scheduler.step()

        if self.sampler:
            with torch.no_grad():
                self.sampler.update(kwargs['algo'], kwargs['invpalgo'], reward)
