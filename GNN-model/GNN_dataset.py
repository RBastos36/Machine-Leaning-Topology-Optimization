# Author: Ricardo A. O. Bastos
# Created: June 2025


import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric.data as geom_data
from torch_geometric.data import Batch
import h5py
import json
import numpy as np
from tqdm import tqdm


class FEMGNN_Dataset(Dataset):
    def __init__(self, hdf5_path, json_split_path, split='train', transform=None, stats=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.stats = stats

        # Load split information
        with open(json_split_path, 'r') as f:
            split_data = json.load(f)

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Split must be one of 'train', 'validation', 'test', got {split}")

        self.samples = split_data[split]
        self.h5_file = None

    def _load_h5_dataset(self, path):
        """Helper function to load a dataset from HDF5 file"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
        return np.array(self.h5_file[path])

    def normalize_features(self, features, is_input=True):
        """Normalize features using pre-computed statistics"""
        if self.stats is None:
            return features

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means'])
        stds = torch.tensor(self.stats[stats_key]['stds'])

        # Expand dimensions for proper broadcasting with node features
        if len(features.shape) == 2:  # [num_nodes, num_features]
            means = means.view(1, -1)
            stds = stds.view(1, -1)

        return (features - means) / (stds + 1e-8)  # Add epsilon for numerical stability

    def denormalize_features(self, features, is_input=True):
        """Denormalize features back to original scale"""
        if self.stats is None:
            return features

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means'])
        stds = torch.tensor(self.stats[stats_key]['stds'])

        # Expand dimensions for proper broadcasting
        if len(features.shape) == 2:  # [num_nodes, num_features]
            means = means.view(1, -1)
            stds = stds.view(1, -1)

        return (features * stds) + means

    def __len__(self):
        return len(self.samples)

    def matrix_to_graph(self, domain, fixed_x, fixed_y, loads_x, loads_y, disp_x=None, disp_y=None):
        """
        Convert grid-based FEM data to a graph representation.
        
        Args:
            domain: Binary matrix indicating material presence
            fixed_x, fixed_y: Matrices indicating fixed boundary conditions
            loads_x, loads_y: Matrices indicating applied loads
            disp_x, disp_y: Displacement matrices (if available)
            
        Returns:
            torch_geometric.data.Data: Graph representation of the FEM problem
        """
        height, width = domain.shape
        
        # 1. Create node features
        # For each valid node (where domain=1), store its features
        valid_nodes = []
        node_features = []
        node_targets = []
        
        for i in range(height):
            for j in range(width):
                # Only include nodes where material exists (domain=1)
                if domain[i, j] == 1:
                    # Store node position (will be used for edge construction)
                    valid_nodes.append((i, j))
                    
                    # Node features: fixed_x, fixed_y, loads_x, loads_y
                    # Add 1 to handle padded domain (as in the CNN version)
                    features = [
                        1.0,  # domain value (always 1 for valid nodes)
                        fixed_x[i, j],
                        fixed_y[i, j],
                        loads_x[i, j],
                        loads_y[i, j]
                    ]
                    node_features.append(features)
                    
                    # Node targets (if available)
                    if disp_x is not None and disp_y is not None:
                        targets = [disp_x[i, j], disp_y[i, j]]
                        node_targets.append(targets)
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)  # [num_nodes, 5]
        
        # 2. Create edges based on mesh connectivity
        # Connect each node to its neighbors (up, down, left, right, and diagonals)
        edges_src = []
        edges_dst = []
        
        # Map from (i, j) to node index
        node_indices = {pos: idx for idx, pos in enumerate(valid_nodes)}
        
        # Define potential neighbors (8-connectivity: horizontal, vertical, and diagonal)
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # up, down, left, right
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
        ]
        
        for node_idx, (i, j) in enumerate(valid_nodes):
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                
                # Check if neighbor is within bounds and is a valid node
                if 0 <= ni < height and 0 <= nj < width and (ni, nj) in node_indices:
                    neighbor_idx = node_indices[(ni, nj)]
                    
                    # Add edge in both directions (undirected graph)
                    edges_src.append(node_idx)
                    edges_dst.append(neighbor_idx)
        
        # Create edge_index tensor [2, num_edges]
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        
        # 3. Prepare the final graph
        graph_data = geom_data.Data(
            x=x,
            edge_index=edge_index
        )
        
        # Add targets if available
        if disp_x is not None and disp_y is not None:
            graph_data.y = torch.FloatTensor(node_targets)  # [num_nodes, 2]
        
        # Add pos for visualization if needed
        graph_data.pos = torch.tensor([[j, i] for i, j in valid_nodes], dtype=torch.float)
        
        return graph_data

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load input matrices
        domain = self._load_h5_dataset(sample['inputs']['domain'])
        fixed_x = self._load_h5_dataset(sample['inputs']['fixed_x'])
        fixed_y = self._load_h5_dataset(sample['inputs']['fixed_y'])
        loads_x = self._load_h5_dataset(sample['inputs']['loads_x'])
        loads_y = self._load_h5_dataset(sample['inputs']['loads_y'])

        # Load output matrices
        disp_x = self._load_h5_dataset(sample['outputs']['displacement_x'])
        disp_y = self._load_h5_dataset(sample['outputs']['displacement_y'])

        # Convert to graph
        graph = self.matrix_to_graph(domain, fixed_x, fixed_y, loads_x, loads_y, disp_x, disp_y)
        
        # Apply normalization if statistics are available
        if self.stats is not None:
            graph.x = self.normalize_features(graph.x, is_input=True)
            graph.y = self.normalize_features(graph.y, is_input=False)

        return graph

    def __del__(self):
        """Cleanup: close HDF5 file if open"""
        if self.h5_file is not None:
            self.h5_file.close()


def collate_graphs(batch):
    """Custom collate function for batching graphs"""
    return Batch.from_data_list(batch)


def calculate_dataset_statistics(hdf5_path, json_split_path, batch_size=32):
    # Create a temporary dataset without normalization
    dataset = FEMGNN_Dataset(hdf5_path, json_split_path, split='train', transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    
    # Initialize variables for mean and variance calculation
    n_input_features = 5  # domain, fixed_x, fixed_y, loads_x, loads_y
    n_output_features = 2  # disp_x, disp_y
    
    input_sum = torch.zeros(n_input_features)
    input_squared_sum = torch.zeros(n_input_features)
    output_sum = torch.zeros(n_output_features)
    output_squared_sum = torch.zeros(n_output_features)
    
    total_nodes = 0
    
    print("Calculating dataset statistics...")
    for batch in tqdm(loader):
        num_nodes = batch.x.size(0)
        total_nodes += num_nodes
        
        # Update sums for inputs (shape: [num_nodes, n_input_features])
        input_sum += batch.x.sum(dim=0)
        input_squared_sum += (batch.x ** 2).sum(dim=0)
        
        # Update sums for outputs (shape: [num_nodes, n_output_features])
        output_sum += batch.y.sum(dim=0)
        output_squared_sum += (batch.y ** 2).sum(dim=0)
    
    # Calculate means
    input_means = input_sum / total_nodes
    output_means = output_sum / total_nodes
    
    # Calculate standard deviations
    input_stds = torch.sqrt((input_squared_sum / total_nodes) - input_means ** 2)
    output_stds = torch.sqrt((output_squared_sum / total_nodes) - output_means ** 2)
    
    # Create statistics dictionary
    stats = {
        'inputs': {
            'means': input_means.tolist(),
            'stds': input_stds.tolist(),
            'feature_names': ['domain', 'fixed_x', 'fixed_y', 'loads_x', 'loads_y']
        },
        'outputs': {
            'means': output_means.tolist(),
            'stds': output_stds.tolist(),
            'feature_names': ['displacement_x', 'displacement_y']
        },
        'metadata': {
            'total_nodes': total_nodes
        }
    }
    
    # Print summary
    print("\nDataset Statistics:")
    print("\nInput features:")
    for name, mean, std in zip(stats['inputs']['feature_names'], 
                              stats['inputs']['means'], 
                              stats['inputs']['stds']):
        print(f"{name:10s}: mean = {mean:8.4f}, std = {std:8.4f}")
    
    print("\nOutput features:")
    for name, mean, std in zip(stats['outputs']['feature_names'], 
                              stats['outputs']['means'], 
                              stats['outputs']['stds']):
        print(f"{name:10s}: mean = {mean:8.4f}, std = {std:8.4f}")
    
    return stats


def load_fem_gnn_data(hdf5_path, json_split_path, split='train', stats=None, batch_size=32, shuffle=True):
    dataset = FEMGNN_Dataset(
        hdf5_path,
        json_split_path,
        split=split,
        stats=stats
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_graphs,
        num_workers=4
    )
    
    return loader, dataset
