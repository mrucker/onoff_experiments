#Our test harness, coba, uses a reward convention but our paper uses a loss convention
#So, we assign a loss function to 'rewards' which is what CappedIGW will receive.
import torch

class MakeLosses:
    class LambdaLoss:
        def __init__(self,loss):
            self.eval = loss
    def filter(self,interactions):
        interactions = list(interactions)
        min_l = min([i['rewards']._label for i in interactions])
        max_l = max([i['rewards']._label for i in interactions])
        for i in interactions:
            scaled_label = (i['rewards']._label-min_l)/(max_l-min_l)
            i['rewards'] = MakeLosses.LambdaLoss(lambda a: abs(a-scaled_label))
            yield i

class Tensorize:
    def filter(self,interactions):
        for interaction in interactions:
            interaction['context'] = torch.tensor(interaction['context'])
            yield interaction