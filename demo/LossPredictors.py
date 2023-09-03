from AbstractClasses import LossPredictor
from typing import Sequence
import torch

class LetCatLossPredictor(LossPredictor):
    def __init__(self, device):
        from LLM import PeftGPT2Classifier
        from peft import IA3Config, TaskType

        super().__init__()
        peft_config = IA3Config(task_type=TaskType.CAUSAL_LM, fan_in_fan_out=True)
        self._regressor = PeftGPT2Classifier(1, peft_config).to(device)
        self.t = 0
        self.y_sum = 0

    def _stringify(self, x, a):
        prepend = '\n'.join(f"Instruction: {v[0]['instruction']} Answer: {v[0]['answer']}" for v in a)
        return f"{prepend}\nInstruction: {x} Answer: "

    def predict(self, context: str, actions) -> Sequence[float]:
        with torch.no_grad():
            X = [ self._stringify(context, action) for action in actions ]
            # NB: must be transferred to cpu so AnytimeNormalizedSampler.py can use it
            return self._regressor.predict(X).to('cpu').numpy()

    def learn(self, contexts: torch.Tensor, actions: Sequence[float], losses: Sequence[float]) -> None:
        self.t += 1

        with torch.no_grad():
            X = [ self._stringify(context,action) for context,action in zip(contexts,actions) ]
            y = torch.tensor(losses).float().unsqueeze(1)
            self.y_sum += torch.mean(y).item()

        loss = self._regressor.learn(X, y)

        with torch.no_grad():
            import coba
            import torch.nn.functional as F

            y_av = self.y_sum/self.t
            best_const_loss = 0 if y_av <= 0 or y_av >= 1 else F.binary_cross_entropy(y_av*torch.ones_like(y), y).item()
            coba.CobaContext.learning_info['loss_prediction_loss'] = [loss]*len(losses)
            coba.CobaContext.learning_info['loss_prediction_regret'] = [loss-best_const_loss]*len(losses)
