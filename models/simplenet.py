import torch
import torch.nn.functional as F


class SimpleNet(torch.nn.Module):
    def __init__(self, hidden_dim = 20, num_classes=10):
        super(SimpleNet, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_dim)   # hidden layer
        # self.bn = torch.nn.BatchNorm1d(args.hidden_dim)
        self.hidden1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.predict = torch.nn.Linear(hidden_dim, num_classes)   #output layer
        self.repr_dim = hidden_dim
        self.output_dim = num_classes

    def forward(self, x):
        x = F.relu(self.hidden(x))
        #x = self.bn(x)
        x = F.relu(self.hidden1(x))
        #x = self.bn(x)
        x = self.predict(x)
        # x = (F.tanh(x) * 1/2 + 1/2)*(self.ub - self.lb) + self.lb
        return x

    def get_representation(self, x):
        """
        Given input, returns representation
        "z" vector.
        """
        with torch.no_grad():
            # x = x.clone()
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
            return x