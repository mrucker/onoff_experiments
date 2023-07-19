import torch

class ArgminPlusDispersion(torch.nn.Module):

    def __init__(self, argminblock):
        super().__init__()
        self.argmin  = argminblock
        self.powers  = torch.tensor([1.,1.5,2.],requires_grad=False)
        self.weights = torch.nn.Parameter(torch.ones((len(self.powers)*2,1)))
        self.relu    = torch.nn.ReLU()
        self.min_val = torch.nn.Parameter(torch.zeros(1))

    def in_features(self, in_features):
        self.argmin.in_features(in_features)

    @property
    def params(self):
        return {"APD":4,**self.argmin.params}

    def forward(self, Xs, As):

        diff = (self.argmin(Xs)-As).view(-1,1)
        pos  = self.relu(diff)
        neg  = self.relu(-diff)

        return self.min_val+torch.cat([pos.pow(self.powers),neg.pow(self.powers)],dim=1)@self.weights.abs()

class LinearArgmin(torch.nn.Module):
    def in_features(self, in_features):
        self.linear = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    @property
    def params(self):
        return {"argmax":"linear"}

    def forward(self, Xs):
        return self.sigmoid(self.linear(Xs))

class MlpArgmin(torch.nn.Module):
    def in_features(self, in_features):
        self.layers = torch.nn.Sequential(
          torch.nn.Linear(in_features=in_features, out_features=in_features),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(in_features=in_features, out_features=in_features),
          torch.nn.LeakyReLU(),
          torch.nn.Linear(in_features=in_features, out_features=1)
        )
        self.sigmoid = torch.nn.Sigmoid()
    
    @property
    
    def params(self):
        return {"argmax":"MLP"}
    
    def forward(self, Xs):
        return self.sigmoid(self.layers(Xs))
