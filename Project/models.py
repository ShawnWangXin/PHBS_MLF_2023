import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, d_feat, hidden_size = 32, num_layers=2, dropout=0.3):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.num_layers = num_layers
        self.gru = nn.GRU(d_feat, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.d_feat)
        out, _ = self.gru(x)  # 前向传播
        out = self.fc(out[:, -1, :])
        return out

class MLP(nn.Module):
    def __init__(self, d_feat, hidden_size = 64, output_size = 1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_feat, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.squeeze(out)
        return out