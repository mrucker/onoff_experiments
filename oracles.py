import torch

class ArgmaxPlusDispersion(torch.nn.Module):

    def __init__(self, argmaxblock):
        super().__init__()
        self.argmax  = argmaxblock
        self.powers  = torch.tensor([1.,1.5,2.],requires_grad=False)
        self.weights = torch.nn.Parameter(torch.ones((len(self.powers)*2,1)))
        self.max_val = torch.nn.Parameter(torch.ones(1))
        self.relu    = torch.nn.ReLU()

    def in_features(self, in_features):
        self.argmax.in_features(in_features)

    @property
    def params(self):
        return {"APD":4,**self.argmax.params}

    def max(self):
        return self.max_val

    def forward(self, Xs, As):

        diff = (self.argmax(Xs)-As).view(-1,1)
        pos  = self.relu(diff)
        neg  = self.relu(-diff)

        return self.max_val-torch.cat([pos.pow(self.powers),neg.pow(self.powers)],dim=1)@self.weights.abs()

class LinearArgmax(torch.nn.Module):
    def in_features(self, in_features):
        self.linear = torch.nn.Linear(in_features=in_features, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    @property
    def params(self):
        return {"argmax":"linear"}

    def forward(self, Xs):
        return self.sigmoid(self.linear(Xs))

class MlpArgmax(torch.nn.Module):
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
