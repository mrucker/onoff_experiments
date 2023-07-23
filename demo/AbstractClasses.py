from typing import Generator, Mapping, Sequence, Callable, Any
from abc import ABC, abstractmethod

class ReferencePolicy(ABC):
    """The reference policy we sample from (mu in the original paper)."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the reference policy (used for documentation only)."""
        return {}

    @abstractmethod
    def sample(self, context) -> Generator[Sequence[Any],None,None]:
        """Generate batches of actions from the reference policy.

        Remarks:
            This should be a generator that returns batches 
            one at a time for as long as we ask for samples.
        """
        pass

class LossPredictor(ABC):
    """A class that learns/predicts losses for context-action pairs."""

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the loss predictor (used for documentation only)."""
        return {}

    @abstractmethod
    def predict(self, context: Any, actions: Sequence[Any]) -> Sequence[float]:
        """Predict the losses for a given context and actions.

        Args:
            context: A single context
            actions: A sequence of actions

        Returns:
            Returns a sequence of loss predictions. The 
            length of return should equal len(actions).
        """
        pass

    @abstractmethod
    def learn(self, contexts: Sequence[Any], actions: Sequence[Any], losses: Sequence[float]) -> None:
        """Learn to better predict losses for context and actions.

        Args:
            contexts: A sequence of contexts
            actions: A sequence of actions
            losses: Observed losses for each context-action pair

        Remarks:
            The arguments are parallel arrays so that
            len(contexts)==len(actions)==len(losses).

        """
