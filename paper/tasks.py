import copy
from statistics import mean
import coba as cb

class MyEvaluator:

    def evaluate(self, environment, learner):
        rng = cb.CobaRandom(1)
        env = cb.Environments(environment).materialize()
        env_length = len(list(env[0].read()))

        def reshuffle(_env: cb.Environments):
            n_batches = .05*len(list(_env[0].read()))
            return _env.unbatch().shuffle(seed=rng.randint(0,10_000)).batch(8)

        def train(_env,learner):
            list(cb.OffPolicyEvaluator(predict=False).evaluate(_env[0],learner))
            return copy.deepcopy(learner)

        def validate(_env,learner):
            return mean([i['reward'] for i in cb.OffPolicyEvaluator(learn=False).evaluate(_env[0],learner)])

        trn_end = int(env_length*.8)
        tst_beg = int(env_length*.9)

        trn = env.slice(None   ,trn_end)
        val = env.slice(trn_end,tst_beg).batch(1)
        tst = env.slice(tst_beg,None   ).batch(1)

        old_lrn = learner
        old_val = validate(val           ,learner)
        new_lrn = train   (reshuffle(trn),learner)
        new_val = validate(val           ,learner)
        
        while new_val > old_val:
            old_lrn = new_lrn
            old_val = new_val

            new_lrn = train   (reshuffle(trn), learner)
            new_val = validate(val           , learner)

        yield from cb.OnPolicyEvaluator(learn=False).evaluate(tst[0],old_lrn)
