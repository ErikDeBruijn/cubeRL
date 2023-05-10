import torch.nn as nn
import torch.nn.functional as F


class RubiksCubeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RubiksCubeNet, self).__init__()

        # Define the layers of the network
        print("input_dim: ", input_dim)
        print("output_dim: ", output_dim)
        # Define the layers of the network
        self.hidden_layer1 = nn.Linear(input_dim, 256)
        self.hidden_layer2 = nn.Linear(256, 512)
        self.hidden_layer3 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x
