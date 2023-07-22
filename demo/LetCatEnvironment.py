import json
from collections import Counter
from typing import Iterable, Sequence

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

class LetCatEnvironment:

    class Reward:

        def __init__(self, test, choices:Sequence[str], tokenizer, model):
            self._test      = test
            self._choices   = choices
            self._tokenizer = tokenizer
            self._model     = model

        def eval(self, action):

            prompt = ""
            for example in action:
                prompt += f"Problem: {example[0]['instruction']}\nSolution: {example[0]['answer']}\n\n"
            prompt += f"Problem: {self._test['instruction']}\nSolution: "

            responses = [ (self._logprobs_from_prompt(prompt+c),c) for c in self._choices ]
            best = max(responses)[1]
            return int(best!=self._test['answer'])

        def _logprobs_from_prompt(self, prompt):
            encoded = self._tokenizer(prompt, return_tensors="pt").to("cpu")
            input_ids = encoded["input_ids"]
            output = self._model(input_ids=input_ids)
            shift_labels = input_ids[..., 1:].contiguous()
            shift_logits = output.logits[..., :-1, :].contiguous()
            log_prob = 0

            for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
                    log_prob += torch.nn.functional.log_softmax(logit, dim=0)[label_id].item()

            return log_prob

    def __init__(self, n_choices:int=4, model:str = 'gpt2', seed:int = 1) -> None:
        self._n_choices = n_choices
        self._model     = model
        self._rng       = np.random.default_rng(seed)

    @property
    def params(self):
        return {'data':'LetCat', 'n_alts':self._n_choices, 'model': self._model}

    def read(self) -> Iterable[dict]:

        tokenizer = AutoTokenizer.from_pretrained(self._model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self._model)
        model.config.pad_token_id = model.config.eos_token_id

        with open('LetCatTest.jsonl', 'r') as f:
            tests = [json.loads(line) for line in f]
            counts = Counter([t['answer'] for t in tests])
            freqs = {k:v/counts.total() for k,v in counts.items()}

            _answers,_freqs = zip(*freqs.items())

            for test in tests:
                prompt = test['instruction']
                answer = test['answer']

                alts = list(self._rng.choice(_answers, self._n_choices+1, replace=False, p=_freqs))
                if answer in alts: alts.remove(answer)
                alts.append(answer)

                best_const_reward = int(answer==max([(freqs[a],a) for a in alts])[1])

                reward = LetCatEnvironment.Reward(test,alts[-self._n_choices:],tokenizer,model)
                yield { 'context': prompt, 'actions': [], 'rewards': reward, 'best_const': best_const_reward}
