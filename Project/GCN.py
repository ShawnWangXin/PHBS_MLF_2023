import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda:0'
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        adjacency = adjacency + torch.eye(adjacency.size(0)).to(device)
        d = torch.diag(torch.pow(torch.sum(adjacency, dim=1), -0.5))  # 用来归一化，度的平方根的倒数
        adjacency = torch.mm(torch.mm(d, adjacency), d)

        # 进行线性变换并与邻接矩阵相乘
        support = self.linear(x)
        output = torch.mm(adjacency, support)
        return output

class AttentionModule(nn.Module):
    def __init__(self, hidden_size, n_layers):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, layers_output):
        # layers_output: a list of layer outputs, each is [N, hidden_size]
        x = torch.stack(layers_output, dim=1)  # [N, hidden_size, n_layers]
        Q = self.query(x)  # [N, H,hidden_size]
        K = self.key(x)  # [N, H,, hidden_size]
        V = self.value(x)  # [N, H, hidden_size]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))  # [N, H, H]
        attn_scores = F.softmax(attn_scores, dim=-1)  # [N, T, T]

        output = torch.matmul(attn_scores, V)  # [N, H, hidden_size]
        output = torch.sum(output, dim=1).squeeze()
        return output

class GCN(nn.Module):
    def __init__(self, d_feat, num_layers, hidden_size = 128, dropout = 0.2, nclass=1):
        super(GCN, self).__init__()
        self.d_feat = d_feat
        self.gc_layers = nn.ModuleList([GCNLayer(hidden_size, hidden_size) for i in range(num_layers)]) # d_feat if i == 0 else 
        self.attention = AttentionModule(hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, nclass)
        self.dropout = dropout
        self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )


    def forward(self, x, adjacency):
        x_hidden = x.reshape(len(x), self.d_feat, -1) # [N, F, T]      
        x_hidden = x_hidden.permute(0, 2, 1) # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)  # 用RNN为每个股票编码，但是这里用到的是节点特征而不是embedding
        x_hidden = x_hidden[:, -1, :] # [N,F] (281,64)
        layer_outputs = []
        for gc_layer in self.gc_layers:
            x_hidden = F.relu(gc_layer(x_hidden, adjacency))
            x_hidden = F.dropout(x_hidden, self.dropout, training=self.training)
            layer_outputs.append(x_hidden)
        x_hidden = self.attention(layer_outputs)
        y = self.fc(x_hidden).squeeze()
        return y
