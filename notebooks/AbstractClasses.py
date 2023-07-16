from typing import Iterable
from abc import ABC

class ReferencePolicy(ABC):

    @property
    def params(self):        
        return {}

    @abstractmethod
    def sample(self, context, n_actions:int):
        pass

class Regressor(ABC):

    @property
    def params(self):        
        return {}

    @abstractmethod
    #one context many actions (add more documentation)
    def predict(self, context, actions) -> Iterable[float]:
        pass

    @abstractmethod
    #these are triples in paralelle arrays (TODO: cleanup documentation)
    def learn(self, contexts, actions, rewards):
        pass