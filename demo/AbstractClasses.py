from typing import Generator, Mapping, Sequence, Callable, Any
from abc import ABC, abstractmethod

class ReferencePolicy(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        return {}

    @abstractmethod
    def sample(self, context) -> Generator[Sequence[Any],None,None]:
        pass

class LossPredictor(ABC):

    @property
    def params(self) -> Mapping[str,Any]:
        return {}

    @abstractmethod
    #one context many actions (add more documentation)
    def predict(self, context, actions) -> Sequence[float]:
        pass

    @abstractmethod
    #three parallel arrays (TODO: cleanup documentation)
    def learn(self, contexts, actions, rewards) -> None:
        pass
