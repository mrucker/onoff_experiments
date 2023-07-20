import warnings
from typing import Sequence

import torch
import coba as cb

from AbstractClasses import RewardPredictor
from CauchyNetwork import CauchyNetwork

class MyRewardPredictor(RewardPredictor):
    def __init__(self, *, numrff:int, sigma:float, in_features:int, opt_factory, sched_factory) -> None:
        self._cauchy_network = CauchyNetwork(numrff,sigma,in_features)
        self.loss            = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.opt             = opt_factory(self._cauchy_network.parameters())
        self.scheduler       = sched_factory(self.opt)

    @property
    def params(self):
        return self._cauchy_network.params

    #one context many actions (add more documentation)
    def predict(self, context: Sequence[float], actions: Sequence[float]) -> Sequence[float]:
        with torch.no_grad():
            context = torch.tensor(context).unsqueeze(0).expand(len(actions),-1)
            actions = 10*torch.tensor(actions).unsqueeze(1)
            return self._cauchy_network.reward(torch.cat([context,actions],dim=1))

    #these are triples in parallel arrays (TODO: cleanup documentation)
    def learn(self, 
              contexts: Sequence[Sequence[float]],
              actions : Sequence[float], 
              rewards : Sequence[float]) -> None:

        with torch.no_grad():
            contexts = torch.tensor(contexts)
            actions  = 10*torch.tensor(actions).unsqueeze(1)

            X = torch.cat([contexts,actions],dim=1)
            y = torch.tensor(rewards)

        self.opt.zero_grad()
        pred = self._cauchy_network.pre_logits(X)
        loss = self.loss(pred.squeeze(1),y)
        loss.mean().backward()
        self.opt.step()
        #ignore false warning when multiprocessing
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            self.scheduler.step()

        with torch.no_grad():
            cb.CobaContext.learning_info['reward_prediction_loss'] = loss.tolist()
            optimal_loss = self.loss(torch.logit(y),y)
            cb.CobaContext.learning_info['reward_prediction_regret'] = (loss-optimal_loss).tolist()
