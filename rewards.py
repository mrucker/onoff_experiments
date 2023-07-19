import coba as cb

class ScaledRewards:

    class Reward:
        def __init__(self, label):
            self._label = label
        def eval(self, action):
            return 1-abs(action-self._label)

    def filter(self,interactions):

        interactions = list(interactions)
        labels = [i['rewards']._label for i in interactions]

        min_l = min(labels)
        max_l = max(labels)

        for interaction,label in zip(interactions,labels):
            new = interaction.copy()
            new['rewards'] = ScaledRewards.Reward((label-min_l)/(max_l-min_l))
            yield new
