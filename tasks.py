import copy

from statistics import mean
from coba.experiments import SimpleEvaluation
from coba.environments.filters import Unbatch, Batch

from coba.random import CobaRandom
from coba.environments.filters import Unbatch, Batch

class MyEvaluator:

    def __init__(self, epochs=False) -> None:
        self._epochs = epochs

    def process(self, learner, interactions):
        
        rng = CobaRandom(1)
        interactions = list(interactions)

        def reshuffle(examples):
            return list(Batch(int(.05*len(examples))).filter(rng.shuffle(list(Unbatch().filter(examples)))))

        def train(learner,examples):
            list(SimpleEvaluation(learn=True,predict=False).process(learner,examples))
            return copy.deepcopy(learner)

        def validate(learner,examples):
            return mean([mean(i['reward']) for i in SimpleEvaluation(learn=False,predict=True).process(learner,examples)])

        if not self._epochs:
            for interaction, result in zip(interactions,SimpleEvaluation().process(learner,interactions)):
                if 'greed_act' in result:
                    result['greed_rwd'] = mean(interaction['rewards'].eval(result.pop('greed_act')))
                yield result
        
        else:
            trn_end = int(len(interactions)*.8)
            tst_beg = int(len(interactions)*.9)

            trn = interactions[:trn_end]
            val = interactions[trn_end:tst_beg]
            tst = interactions[tst_beg:]

            old_lrn = learner
            old_val = validate(learner,reshuffle(val))

            new_lrn = train   (learner,reshuffle(trn))
            new_val = validate(learner,reshuffle(val))

            while new_val > old_val:
                old_lrn = new_lrn
                old_val = new_val

                new_lrn = train   (learner,reshuffle(trn))
                new_val = validate(learner,reshuffle(val)+reshuffle(val)+reshuffle(val)+reshuffle(val))

            yield from SimpleEvaluation('reward',learn=False,predict=True).process(old_lrn,tst)
