
import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, Linear, GCNConv,global_mean_pool, GINEConv, GATv2Conv
import torch.nn as nn

class GNN(torch.nn.Module):
    def __init__(self, model_name, in_channels, hidden_channels, out_channels, num_layers, heads):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.embedding = Linear(20, 10)
        in_channels -= 10
        self.hidden_dim = hidden_channels
        self.model_name = model_name
        d=1
        if model_name == "GraphSAGE":
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='add'))
            self.batch_norms.append(BatchNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.batch_norms.append(BatchNorm(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr = 'add'))
        
        elif model_name == "GCN":
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(BatchNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        elif model_name == "GAT":
            self.convs.append(GATv2Conv(in_channels, hidden_channels,heads, edge_dim=2))
            self.batch_norms.append(BatchNorm(hidden_channels*heads))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels*heads, hidden_channels,heads, edge_dim=2))
                self.batch_norms.append(BatchNorm(hidden_channels*heads))
            self.convs.append(GATv2Conv(hidden_channels*heads, out_channels, heads=1, concat=False, edge_dim=2))
        elif model_name == 'GINE':
            self.leakyrelu = nn.LeakyReLU(0.2)
            for _ in range(num_layers-1):
                self.convs.append(GINEConv(nn.Sequential(nn.Linear(in_channels, hidden_channels),  
                                self.leakyrelu,
                                nn.BatchNorm1d(hidden_channels),
                                nn.Linear(hidden_channels, hidden_channels), 
                                self.leakyrelu), edge_dim = 8
                                )) 
                self.batch_norms.append(BatchNorm(hidden_channels))
                in_channels = hidden_channels     
            self.convs.append(GINEConv(nn.Sequential(nn.Linear(in_channels, out_channels),  
                                self.leakyrelu,
                                nn.BatchNorm1d(out_channels),
                                nn.Linear(out_channels, out_channels), 
                                self.leakyrelu), edge_dim = 8
                                )) 
        self.classifier = Linear(out_channels, 16) 

    def forward(self, x, edge_index,edge_attr,  batch, pos=None):
        #dissimilarity = (edge_attr[: , 0] - 33)/(2*19)+1/2
        #edge_weight = 1-dissimilarity #Add edge weight
        embedding = self.embedding(x[:, :20])
        out = torch.cat((embedding, x[:, 20:]), axis = 1)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            if self.model_name == "GINE" or self.model_name == "GAT":
                out = conv(out, edge_index, edge_attr = edge_attr)
            else:
                out = conv(out, edge_index)
                out = batch_norm(out)
                out = F.relu(out)
                out = F.dropout(out, p=0.2, training=self.training)
        out = self.convs[-1](out, edge_index)
        out = global_mean_pool(out, batch=batch)
        out = self.classifier(out)
        return out
    