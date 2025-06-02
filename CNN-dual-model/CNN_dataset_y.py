# Author: Ricardo A. O. Bastos
# Created: June 2025


import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import numpy as np
from tqdm import tqdm


class FEMDataset(Dataset):
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

    def normalize_tensor(self, tensor, is_input=True):
        """Normalize tensor using pre-computed statistics"""
        if self.stats is None:
            return tensor

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means']).view(-1, 1, 1)
        stds = torch.tensor(self.stats[stats_key]['stds']).view(-1, 1, 1)

        return (tensor - means) / (stds + 1e-8)  # Add epsilon for numerical stability

    def denormalize_tensor(self, tensor, is_input=True):
        """Denormalize tensor back to original scale"""
        if self.stats is None:
            return tensor

        stats_key = 'inputs' if is_input else 'outputs'
        means = torch.tensor(self.stats[stats_key]['means']).view(-1, 1, 1)
        stds = torch.tensor(self.stats[stats_key]['stds']).view(-1, 1, 1)

        return (tensor * stds) + means

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load input matrices (5 channels)
        domain = self._load_h5_dataset(sample['inputs']['domain'])
        fixed_x = self._load_h5_dataset(sample['inputs']['fixed_x'])
        fixed_y = self._load_h5_dataset(sample['inputs']['fixed_y'])
        loads_x = self._load_h5_dataset(sample['inputs']['loads_x'])
        loads_y = self._load_h5_dataset(sample['inputs']['loads_y'])

        # Pad domain matrix to match the size of other matrices
        padded_domain = np.zeros((domain.shape[0] + 1, domain.shape[1] + 1))
        padded_domain[:-1, :-1] = domain

        # Stack inputs into a single 5-channel tensor
        input_data = np.stack([padded_domain, fixed_x, fixed_y, loads_x, loads_y])

        # Load output matrix (1 channels)
        disp_y = self._load_h5_dataset(sample['outputs']['displacement_y'])

        # Stack outputs into a single 1-channel tensor
        output_data = np.stack([disp_y])

        # Convert to PyTorch tensors
        input_tensor = torch.FloatTensor(input_data)
        output_tensor = torch.FloatTensor(output_data)

        # Apply normalization if statistics are available
        if self.stats is not None:
            input_tensor = self.normalize_tensor(input_tensor, is_input=True)
            output_tensor = self.normalize_tensor(output_tensor, is_input=False)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)

        return input_tensor, output_tensor

    def __del__(self):
        """Cleanup: close HDF5 file if open"""
        if self.h5_file is not None:
            self.h5_file.close()


def calculate_dataset_statistics(hdf5_path, json_split_path, batch_size=32):
    """
    Calculate channel-wise mean and std statistics for the training dataset.

    Args:
        hdf5_path (str): Path to HDF5 file
        json_split_path (str): Path to JSON split file
        batch_size (int): Batch size for processing

    Returns:
        dict: Dictionary containing statistics for inputs and outputs
    """
    # Create a temporary dataset without normalization
    dataset = FEMDataset(hdf5_path, json_split_path, split='train', transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables for online mean and variance calculation
    n_samples = 0
    input_sum = torch.zeros(5)  # 5 input channels
    input_squared_sum = torch.zeros(5)
    output_sum = torch.zeros(1)  # 1 output channels
    output_squared_sum = torch.zeros(1)

    print("Calculating dataset statistics...")
    for inputs, outputs in tqdm(loader):
        batch_size = inputs.size(0)
        n_samples += batch_size

        # Update sums for inputs (shape: [batch_size, 5, height, width])
        input_sum += inputs.sum(dim=(0, 2, 3))  # Sum across batch, height, and width
        input_squared_sum += (inputs ** 2).sum(dim=(0, 2, 3))

        # Update sums for outputs (shape: [batch_size, 2, height, width])
        output_sum += outputs.sum(dim=(0, 2, 3))
        output_squared_sum += (outputs ** 2).sum(dim=(0, 2, 3))

    # Calculate means
    input_means = input_sum / (n_samples * inputs.size(2) * inputs.size(3))
    output_means = output_sum / (n_samples * outputs.size(2) * outputs.size(3))

    # Calculate standard deviations
    input_stds = torch.sqrt(
        (input_squared_sum / (n_samples * inputs.size(2) * inputs.size(3)))
        - input_means ** 2
    )
    output_stds = torch.sqrt(
        (output_squared_sum / (n_samples * outputs.size(2) * outputs.size(3)))
        - output_means ** 2
    )

    # Create statistics dictionary
    stats = {
        'inputs': {
            'means': input_means.tolist(),
            'stds': input_stds.tolist(),
            'channel_names': ['domain', 'fixed_x', 'fixed_y', 'loads_x', 'loads_y']
        },
        'outputs': {
            'means': output_means.tolist(),
            'stds': output_stds.tolist(),
            'channel_names': ['displacement_x', 'displacement_y']
        },
        'metadata': {
            'n_samples': n_samples,
            'spatial_dims': [inputs.size(2), inputs.size(3)]
        }
    }

    # Print summary
    print("\nDataset Statistics:")
    print("\nInput channels:")
    for name, mean, std in zip(stats['inputs']['channel_names'],
                               stats['inputs']['means'],
                               stats['inputs']['stds']):
        print(f"{name:10s}: mean = {mean:8.4f}, std = {std:8.4f}")

    print("\nOutput channels:")
    for name, mean, std in zip(stats['outputs']['channel_names'],
                               stats['outputs']['means'],
                               stats['outputs']['stds']):
        print(f"{name:10s}: mean = {mean:8.4f}, std = {std:8.4f}")

    return stats


def load_fem_data(hdf5_path, json_split_path, split='train', stats=None):
    """
    Helper function to load FEM data with statistics-based normalization

    Args:
        hdf5_path (str): Path to HDF5 file
        json_split_path (str): Path to JSON split file
        split (str): One of 'train', 'validation', or 'test'
        stats (dict, optional): Statistics for normalization

    Returns:
        FEMDataset object
    """
    dataset = FEMDataset(
        hdf5_path,
        json_split_path,
        split=split,
        stats=stats
    )
    return dataset


# Example usage
if __name__ == "__main__":
    # Calculate statistics from training data
    stats = calculate_dataset_statistics('cantilever-diagonal_dataset.h5', 'dataset_split.json')

    # Create datasets with normalization
    train_dataset = load_fem_data('cantilever-diagonal_dataset.h5', 'dataset_split.json',
                                  split='train', stats=stats)
    val_dataset = load_fem_data('cantilever-diagonal_dataset.h5', 'dataset_split.json',
                                split='validation', stats=stats)
    test_dataset = load_fem_data('cantilever-diagonal_dataset.h5', 'dataset_split.json',
                                 split='test', stats=stats)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)