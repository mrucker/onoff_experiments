import coba
import torch
from collections import defaultdict
from typing import Sequence, Callable
from AbstractClasses import ReferencePolicy

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

class FewShotFixedStrategy:
    def __init__(self, sampler:ReferencePolicy) -> None:
        self._sampler = sampler
    @property
    def params(self):
        return self._sampler.params
    def predict(self, context, actions):
        if isinstance(context,coba.Batch): raise Exception()
        action = next(self._sampler.sample(context))[0]
        return action
    def learn(self, context, action, reward, _):
        pass

class ZeroShotStrategy:
    def predict(self, context, actions):
        if isinstance(context,coba.Batch): raise Exception()
        return []
    def learn(self, context, action, reward, _):
        pass