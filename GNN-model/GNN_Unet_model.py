import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, GraphConv


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


class GraphUNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=(32, 64, 128), output_dim=2, conv_type='GCN', pool_ratios=(0.8, 0.6)):
        super(GraphUNet, self).__init__()
        
        # Ensure proper lengths for our U-Net structure
        assert len(hidden_dims) == len(pool_ratios) + 1, "Number of hidden dimensions should be one more than pool ratios"
        
        self.num_layers = len(hidden_dims)
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # Encoder path
        self.down_gnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Create encoder layers with pooling
        for i in range(len(hidden_dims) - 1):
            self.down_gnns.append(GNNBlock(hidden_dims[i], hidden_dims[i+1], conv_type))
            self.pools.append(TopKPooling(hidden_dims[i+1], ratio=pool_ratios[i]))
            
        # Bottleneck layer
        self.bottleneck = GNNBlock(hidden_dims[-1], hidden_dims[-1], conv_type)
        
        # Decoder path
        self.up_gnns = nn.ModuleList()
        
        # Create decoder layers with upsampling
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Input is features from previous layer + skip connection
            self.up_gnns.append(GNNBlock(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1], conv_type))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (e.g., for a single graph), create a batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial embedding
        x = self.input_embedding(x)
        x = F.relu(x)
        
        # Store features for skip connections
        xs = [x]
        edge_indices = [edge_index]
        batches = [batch]
        
        # Encoder path with pooling
        for i in range(len(self.down_gnns)):
            x = self.down_gnns[i](x, edge_index, batch)
            xs.append(x)
            
            # Apply pooling
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
            edge_indices.append(edge_index)
            batches.append(batch)
        
        # Bottleneck
        x = self.bottleneck(x, edge_index, batch)
        
        # Decoder path with skip connections
        for i in range(len(self.up_gnns)):
            # Get features from encoder path
            skip_x = xs[-(i+2)]  # Get corresponding skip connection
            skip_edge_index = edge_indices[-(i+2)]
            skip_batch = batches[-(i+2)]
            
            # Handle different graph sizes (unpooling)
            # We need to map back to the original node set in each layer
            x = self._unpool_nodes(x, skip_x, batch, skip_batch)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_x], dim=1)
            
            # Apply GNN block
            x = self.up_gnns[i](x, skip_edge_index, skip_batch)
            
            # Update for next iteration
            edge_index = skip_edge_index
            batch = skip_batch
        
        # Final output layer
        x = self.output_layer(x)
        
        return x
        
    def _unpool_nodes(self, x, target_x, batch, target_batch):
        """
        Upsampling operation for graph data.
        Maps features from a pooled graph back to the original node set.
        """
        # Get device
        device = x.device
        
        # Initialize output tensor with zeros
        out = torch.zeros(target_x.size(0), x.size(1), device=device)
        
        # For each batch, we'll map the pooled nodes back to the original graph
        for b in torch.unique(target_batch):
            # Find indices in the pooled graph for this batch
            pooled_idx = (batch == b).nonzero(as_tuple=True)[0]
            
            # Find indices in the original graph for this batch
            orig_idx = (target_batch == b).nonzero(as_tuple=True)[0]
            
            if len(pooled_idx) > 0 and len(orig_idx) > 0:
                # Simple mapping: distribute features uniformly
                # For each node in the original graph, we assign the average of pooled features
                num_orig = len(orig_idx)
                num_pool = len(pooled_idx)
                
                # If there are more pooled nodes than original (shouldn't happen)
                # just take the first ones
                if num_pool > num_orig:
                    pooled_idx = pooled_idx[:num_orig]
                    num_pool = num_orig
                
                # Calculate how many original nodes each pooled node maps to
                nodes_per_pool = max(1, num_orig // num_pool)
                
                # Map pooled features back to original nodes
                for i, p_idx in enumerate(pooled_idx):
                    start_idx = i * nodes_per_pool
                    end_idx = min((i + 1) * nodes_per_pool, num_orig)
                    
                    if start_idx < end_idx:
                        out[orig_idx[start_idx:end_idx]] = x[p_idx]
        
        return out


class DeepGraphUNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=(32, 64, 128, 256), output_dim=2, 
                 conv_type='GCN', pool_ratios=(0.8, 0.7, 0.6)):
        super(DeepGraphUNet, self).__init__()
        
        # Ensure we have proper lengths for our U-Net structure
        assert len(hidden_dims) == len(pool_ratios) + 1, "Number of hidden dimensions should be one more than pool ratios"
        
        self.num_layers = len(hidden_dims)
        
        # Input embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # Encoder path
        self.down_gnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Create encoder layers with pooling
        for i in range(len(hidden_dims) - 1):
            self.down_gnns.append(GNNBlock(hidden_dims[i], hidden_dims[i+1], conv_type))
            self.pools.append(TopKPooling(hidden_dims[i+1], ratio=pool_ratios[i]))
            
        # Bottleneck layer
        self.bottleneck = GNNBlock(hidden_dims[-1], hidden_dims[-1], conv_type)
        
        # Decoder path
        self.up_gnns = nn.ModuleList()
        
        # Create decoder layers with upsampling
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Input is features from previous layer + skip connection
            self.up_gnns.append(GNNBlock(hidden_dims[i] + hidden_dims[i-1], hidden_dims[i-1], conv_type))
            
        # Output layer with a bit more complexity
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.Linear(hidden_dims[0] // 2, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (e.g., for a single graph), create a batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial embedding
        x = self.input_embedding(x)
        
        # Store features for skip connections
        xs = [x]
        edge_indices = [edge_index]
        batches = [batch]
        
        # Encoder path with pooling
        for i in range(len(self.down_gnns)):
            x = self.down_gnns[i](x, edge_index, batch)
            xs.append(x)
            
            # Apply pooling
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
            edge_indices.append(edge_index)
            batches.append(batch)
        
        # Bottleneck
        x = self.bottleneck(x, edge_index, batch)
        
        # Decoder path with skip connections
        for i in range(len(self.up_gnns)):
            # Get features from encoder path
            skip_x = xs[-(i+2)]  # Get corresponding skip connection
            skip_edge_index = edge_indices[-(i+2)]
            skip_batch = batches[-(i+2)]
            
            # Handle different graph sizes (unpooling)
            x = self._unpool_nodes(x, skip_x, batch, skip_batch)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_x], dim=1)
            
            # Apply GNN block
            x = self.up_gnns[i](x, skip_edge_index, skip_batch)
            
            # Update for next iteration
            edge_index = skip_edge_index
            batch = skip_batch
        
        # Final output layer
        x = self.output_layer(x)
        
        return x
        
    def _unpool_nodes(self, x, target_x, batch, target_batch):
        """
        More advanced upsampling for graph data that preserves structure
        """
        # Get device
        device = x.device
        
        # Initialize output tensor with zeros
        out = torch.zeros(target_x.size(0), x.size(1), device=device)
        
        # For each batch, we'll map the pooled nodes back to the original graph
        for b in torch.unique(target_batch):
            # Find indices in the pooled graph for this batch
            pooled_idx = (batch == b).nonzero(as_tuple=True)[0]
            
            # Find indices in the original graph for this batch
            orig_idx = (target_batch == b).nonzero(as_tuple=True)[0]
            
            if len(pooled_idx) == 0 or len(orig_idx) == 0:
                continue
                
            # Get the pooled features for this batch
            pooled_features = x[pooled_idx]
            
            # Calculate assignment weights based on feature similarity
            # This helps distribute features in a more meaningful way
            if len(pooled_idx) < len(orig_idx):
                # We need to map fewer pooled nodes to more original nodes
                # Use a simple replication strategy with some learned weighting
                
                # Simple approach: replicate features to match original size
                repeat_factor = (len(orig_idx) + len(pooled_idx) - 1) // len(pooled_idx)
                repeated_features = pooled_features.repeat_interleave(repeat_factor, dim=0)
                
                # Trim to match original size
                if repeated_features.size(0) > len(orig_idx):
                    repeated_features = repeated_features[:len(orig_idx)]
                
                # Assign to original nodes
                out[orig_idx[:repeated_features.size(0)]] = repeated_features
            else:
                # We have more or equal pooled nodes than original nodes
                # Just take the first ones or average them
                out[orig_idx] = pooled_features[:len(orig_idx)]
        
        return out
