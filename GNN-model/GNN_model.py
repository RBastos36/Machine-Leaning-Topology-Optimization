# Author: Ricardo A. O. Bastos
# Created: June 2025


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, MessagePassing


class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type='GCN'):
        super(GNNBlock, self).__init__()
        
        # Select the appropriate graph convolutional layer
        if conv_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels)
        elif conv_type == 'GraphConv':
            self.conv = GraphConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        
        # Batch normalization for graph data
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph convolution
        x = self.conv(x, edge_index)
        
        # Apply batch normalization
        x = self.bn(x)
        
        # Apply ReLU activation
        x = F.relu(x)
        
        return x


class TopologyOptimizationGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, conv_type='GCN', num_layers=3):
        super(TopologyOptimizationGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolutional layers
        self.gnn_blocks = nn.ModuleList()
        
        # First GNN layer after input embedding
        self.gnn_blocks.append(GNNBlock(hidden_dim, hidden_dim, conv_type))
        
        # Additional GNN layers
        for _ in range(num_layers - 1):
            self.gnn_blocks.append(GNNBlock(hidden_dim, hidden_dim, conv_type))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.input_embedding(x)
        x = F.relu(x)
        
        # Apply GNN blocks
        for gnn_block in self.gnn_blocks:
            x = gnn_block(x, edge_index, batch)
            
        # Final output layer to predict displacement
        x = self.output_layer(x)
        
        # Node-level prediction: shape [num_nodes, 2] for x and y displacements
        return x


class DeepTopologyOptimizationGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=(32, 64, 128, 64, 32), output_dim=2, conv_type='GraphConv'):
        super(DeepTopologyOptimizationGNN, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # Encoder GNN blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_blocks.append(
                GNNBlock(hidden_dims[i], hidden_dims[i+1], conv_type)
            )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.input_embedding(x)
        
        # Encoder path
        for block in self.encoder_blocks:
            x = block(x, edge_index, batch)
            
        # Output layer
        x = self.output_layer(x)
        
        return x


class EdgeFeatureGNN(nn.Module):
    """
    GNN model that also considers edge features like distances between nodes.
    Useful for physical simulations where spatial relationships matter.
    """
    class EdgeConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='mean')
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels + 1, out_channels),  # +1 for edge feature (distance)
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            
        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
            
        def message(self, x_i, x_j, edge_attr):
            # x_i: features of target nodes [num_edges, in_channels]
            # x_j: features of source nodes [num_edges, in_channels]
            # edge_attr: edge features [num_edges, edge_features]
            
            # Concatenate source, target features and edge features
            message = torch.cat([x_i, x_j, edge_attr], dim=1)
            
            # Apply MLP
            return self.mlp(message)
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, num_layers=3):
        super(EdgeFeatureGNN, self).__init__()
        
        # Input embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Edge convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(self.EdgeConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def compute_edge_features(self, pos, edge_index):
        """Compute edge features (distances between connected nodes)"""
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], p=2, dim=1).view(-1, 1)
        return dist
        
    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        
        # Compute edge features (distances)
        edge_attr = self.compute_edge_features(pos, edge_index)
        
        # Node feature embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Apply edge convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Final prediction
        x = self.output_layer(x)
        
        return x
