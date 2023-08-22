from AbstractClasses import LossPredictor
from typing import Sequence
import torch

class LetCatLossPredictor(LossPredictor):
    class LogisticRegressor(torch.nn.Module):
        def __init__(self, in_features:int):
            super().__init__()
            self.linear  = torch.nn.Linear(in_features=in_features, out_features=1)
            self.sigmoid = torch.nn.Sigmoid()

        @property
        def params(self):
            return {'type':'logistic'}
    
        def pre_logits(self, X):
            return self.linear(X)

        def predictions(self, X):
            return self.sigmoid(self.pre_logits(X))

    def __init__(self, *, set_size:int, opt_factory, sched_factory, params={}) -> None:
        self._regressor = self.LogisticRegressor(4*set_size)
        self.loss       = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.opt        = opt_factory(self._regressor.parameters())
        self.scheduler  = sched_factory(self.opt)
        self.y_sum      = 0
        self.t          = 0
        self._params    = params

    @property
    def params(self):
        return {**self._regressor.params, **self._params}

    def _features(self,x,a):
        features = []
        for e in a:
            i = e[0]['instruction']

            x_task = 'first' if 'first' in x else 'second' if 'second' in x else 'last'
            e_task = 'first' if 'first' in i else 'second' if 'second' in i else 'last'

            x_name = x[x.find('"')+1:x.rfind('"')]
            e_name = i[i.find('"')+1:i.rfind('"')]

            x_first,x_last = x_name.split(' ', 2) if ' ' in x_name else (x_name, '')
            e_first,e_last = e_name.split(' ', 2) if ' ' in e_name else (e_name, '')

            same_task  = int(x_task==e_task)
            same_first = int(x_first==e_first)
            same_last  = int(x_last==e_last)
            similarity = e[1]

            features.extend([same_task,same_first,same_last,similarity])
        return features

    def predict(self, context: str, actions) -> Sequence[float]:
        with torch.no_grad():
            X = torch.tensor([ self._features(context,action) for action in actions])
            # NB: must be transferred to cpu so AnytimeNormalizedSampler.py can use it
            return self._regressor.predictions(X).to('cpu').numpy()

    def learn(self, contexts: torch.Tensor, actions: Sequence[float], losses: Sequence[float]) -> None:
        self.t += 1
        
        with torch.no_grad():
            X = torch.tensor([self._features(context,action) for context,action in zip(contexts,actions)])            
            y = torch.tensor(losses).float()
            self.y_sum += torch.mean(y).item()
        
        self.opt.zero_grad()
        yhat = self._regressor.pre_logits(X).squeeze(1)
        loss = self.loss(yhat,y)
        loss.mean().backward()

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.opt.step()
            if self.scheduler:
                self.scheduler.step()

        with torch.no_grad():
            import coba
            
            y_av = self.y_sum/self.t
            best_const_loss = 0 if y_av <= 0 or y_av >= 1 else self.loss(torch.logit(y_av*torch.ones_like(y)),y)
            coba.CobaContext.learning_info['loss_prediction_loss'] = loss.tolist()
            coba.CobaContext.learning_info['loss_prediction_regret'] = (loss-best_const_loss).tolist()