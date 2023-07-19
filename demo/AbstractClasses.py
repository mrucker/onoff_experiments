from typing import Generator, Mapping, Sequence, Callable, Any
from abc import ABC, abstractmethod

GammaScheduler = Callable[[int],float] #e.g., lambda t: sqrt(t)

class ReferencePolicy(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        return {}

    @abstractmethod
    def sample(self, context) -> Generator[Sequence[Any],None,None]:
        pass

class RewardPredictor(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        return {}

    @abstractmethod
    #one context many actions (add more documentation)
    def predict(self, context, actions) -> Sequence[float]:
        pass

    @abstractmethod
    #these are triples in parallel arrays (TODO: cleanup documentation)
    def learn(self, contexts, actions, rewards) -> None:
        pass
