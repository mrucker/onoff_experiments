import json
import warnings
from collections import defaultdict
from typing import Sequence, Callable

import torch
import coba as cb
from sentence_transformers import SentenceTransformer

from AbstractClasses import ReferencePolicy, LossPredictor
from LetCatEnvironment import LetCatEnvironment
from CappedIGW import CappedIGW

model = SentenceTransformer('all-MiniLM-L6-v2')

take = 10_000
n_processes = 16
n_samples = 59
log = 'out.log.gz'
device = 'cpu'

if n_processes > 1:
    torch.set_num_threads(1)

torch.set_default_device(device)

def embedder(items):
    if isinstance(items,str):
        return torch.nn.functional.normalize(model.encode([items],convert_to_tensor=True)).to(device)
    if isinstance(items,list):
        items = [ i['instruction'] for i in items]
        return torch.nn.functional.normalize(model.encode(items,convert_to_tensor=True)).to(device)
    raise AssertionError()

def stratum(item):
    text = item if isinstance(item,str) else item['instruction']
    return 'first' if 'first' in text else 'second' if 'second' in text else 'last'

class RandomizedSimilarity(ReferencePolicy):
    def __init__(self,
        embedder: Callable,
        examples: Sequence,
        ex_embeddings: Sequence,
        batch_size: int,
        temperature:float,
        set_size:int,
        stratum: Callable = lambda item: 1,
        preselect:int = 500) -> None:
        self._embedder = embedder
        self._batch_size = batch_size
        self._temperature = temperature
        self._set_size = set_size
        self._stratum = stratum
        self._preselect = preselect

        self._strata_examples = defaultdict(list)
        self._strata_embeddings = defaultdict(list)
        for example,embedding in zip(examples,ex_embeddings):
            self._strata_examples[stratum(example)].append(example)
            self._strata_embeddings[stratum(example)].append(embedding)
        self._strata_examples.default_factory = None
        self._strata_embeddings.default_factory = None

        for stratum,embeddings in self._strata_embeddings.items():
            self._strata_embeddings[stratum] = torch.stack(embeddings)

    @property
    def params(self):
        return {'temp':self._temperature, 'sampler':'RandomizedSimilarity', 'n_strata':len(self._strata_examples)}

    def sample(self, context):
        torch.set_default_device('cpu')
        with torch.no_grad():
            context_stratum = self._stratum(context)

            embeddings = self._strata_embeddings[context_stratum]
            examples   = self._strata_examples[context_stratum]

            embedded_context = self._embedder(context)
            all_similarities = embedded_context @ embeddings.T
            top_similarities = torch.topk(all_similarities,k=self._preselect)
            similarities     = top_similarities.values
            original_indices = top_similarities.indices

            gumbel = torch.distributions.gumbel.Gumbel(0,1)
            gumbel_shape = torch.Size([self._batch_size, similarities.shape[1]])

            while True:
                gumbels = gumbel.sample(gumbel_shape)*self._temperature
                topks   = torch.topk(similarities+gumbels,self._set_size,dim=1).indices

                yield [ [ (examples[original_indices[0,i]],similarities[0,i].item()) for i in row] for row in topks ]

class MyLossPredictor(LossPredictor):
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

        def loss(self, X):
            return self.sigmoid(self.pre_logits(X))

    def __init__(self, *, set_size:int, opt_factory, sched_factory, params={}) -> None:
        self._regressor = MyLossPredictor.LogisticRegressor(4*set_size)
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

            x_first,x_last = x_name.split(' ')
            e_first,e_last = e_name.split(' ')

            same_task  = int(x_task==e_task)
            same_first = int(x_first==e_first)
            same_last  = int(x_last==e_last)
            similarity = e[1]

            features.extend([same_task,same_first,same_last,similarity])
        return features

    def predict(self, context: str, actions) -> Sequence[float]:
        with torch.no_grad():
            X = torch.tensor([ self._features(context,action) for action in actions])
            return self._regressor.loss(X)

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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.opt.step()
            self.scheduler.step()

        with torch.no_grad():
            y_av = self.y_sum/self.t
            best_const_loss = 0 if y_av <= 0 or y_av >= 1 else self.loss(torch.logit(y_av*torch.ones_like(y)),y)
            cb.CobaContext.learning_info['loss_prediction_loss'] = loss.tolist()
            cb.CobaContext.learning_info['loss_prediction_regret'] = (loss-best_const_loss).tolist()

class FewShotFixedStrategy:
    def __init__(self, sampler:ReferencePolicy) -> None:
        self._sampler = sampler
    @property
    def params(self):
        return self._sampler.params
    def predict(self, context, actions):
        if isinstance(context,cb.Batch): raise Exception()
        action, prob = next(self._sampler.sample(context))[0],None
        return action, prob
    def learn(self, context, actions, action, reward, probs, **kwargs):
        pass

class ZeroShotStrategy:
    def predict(self, context, actions):
        if isinstance(context,cb.Batch): raise Exception()
        return [],None
    def learn(self, context, actions, action, reward, probs, **kwargs):
        pass

def uniform(low,high,n) -> torch.Tensor:
    return torch.distributions.uniform.Uniform(low,high).sample([n])

def loguniform(low, high, n) -> torch.Tensor:
    return uniform(*torch.tensor([low,high]).log(),n).exp()

def generate_random_hypers(n):
    lrs  = loguniform(1e-3,1e1,n).tolist()
    tzs  = loguniform(1e-1,1e4,n).tolist()
    gtzs = loguniform(1e-5,1e2,n).tolist()

    yield from zip(lrs,tzs,gtzs)

with open('LetCatTrain.jsonl',mode='rb') as f:
    examples = [ json.loads(line) for line in f ][:5000]
ex_embeddings = embedder(examples)

rs_00_1  = RandomizedSimilarity(embedder, examples, ex_embeddings, batch_size=1 , temperature=.00, set_size=3)
rs_05_30 = RandomizedSimilarity(embedder, examples, ex_embeddings, batch_size=30, temperature=.05, set_size=3)

env = cb.Environments(LetCatEnvironment()).take(take).batch(1)
val = cb.OnPolicyEvaluator()
lrn = [ FewShotFixedStrategy(rs_00_1) ]

for lr,tz,gtz in generate_random_hypers(n_samples):
    fhat = MyLossPredictor(
        set_size=3,
        opt_factory=lambda params: torch.optim.Adam(params,lr=lr),
        sched_factory=lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda t:(1+t/tz)**(-.5)),
        params = {'lr':lr, 'tz':tz, 'gtz':gtz}
    )
    lrn.append(CappedIGW(mu=rs_05_30, fhat=fhat, gamma_sched=lambda t:(1+t/gtz)**(0.5)))

cb.Experiment(env,lrn).run(log,processes=n_processes)
