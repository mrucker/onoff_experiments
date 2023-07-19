#Our test harness, coba, uses a reward convention but our paper uses a loss convention
#So we assign our loss function to 'rewards' and expect a loss in CappedIGW.
class MakeLosses:
    class Loss:
        def __init__(self, label):
            self._label = label
        def eval(self, action):
            assert 0 <= action and action <= 1
            return abs(action-self._label)

    def filter(self,interactions):

        interactions = list(interactions)
        labels = [i['rewards']._label for i in interactions]

        min_l = min(labels)
        max_l = max(labels)

        for interaction,label in zip(interactions,labels):
            new = interaction.copy()
            new['rewards'] = MakeLosses.Loss((label-min_l)/(max_l-min_l))
            yield new
