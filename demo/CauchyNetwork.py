import math
import torch

class CauchyNetwork(torch.nn.Module):
    def __init__(self, numrff:int, sigma:float, in_features:int):
        super().__init__()
        self.args    = (numrff, sigma)
        self.rffW    = torch.nn.Parameter(torch.empty(in_features, numrff).cauchy_(sigma=sigma), requires_grad=False)
        self.rffb    = torch.nn.Parameter((2 * torch.pi * torch.rand(numrff)), requires_grad=False)
        self.sqrtrff = torch.nn.Parameter(torch.Tensor([math.sqrt(numrff)]), requires_grad=False)
        self.linear  = torch.nn.Linear(in_features=numrff, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    @property
    def params(self):
        return {"nrff": self.args[0], "sigma": self.args[1]}

    def pre_logits(self, X):
        with torch.no_grad():
            rff = (torch.matmul(X, self.rffW) + self.rffb).cos() / self.sqrtrff
        return self.linear(rff)
    
    def loss(self, X):
        return self.sigmoid(self.pre_logits(X))
    