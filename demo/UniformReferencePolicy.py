import torch 
from AbstractClasses import ReferencePolicy

class UniformReferencePolicy(ReferencePolicy):
    def __init__(self, batch_size:int = 100):
        self._batch_size = batch_size

    def sample(self, context):
        while True:
            yield torch.rand(self._batch_size).tolist()
