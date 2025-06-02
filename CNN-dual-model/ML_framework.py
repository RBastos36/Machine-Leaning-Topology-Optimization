# Author: Ricardo A. O. Bastos
# Created: June 2025


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py

# Import U-Net model
from CNN_model_Unet_node_level import TopologyOptimizationCNN


class SingleInstanceTester:
    def __init__(self, model, model_path, stats=None, device=None):
        # Determine the best available device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model
        self.model_path = model_path
        self.stats = stats
        print(f'Using device: {self.device}')

    def load_model(self):
        """Load the trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        # Set weights_only=False to maintain compatibility with older PyTorch versions
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Move model to the selected device
        self.model = self.model.to(self.device)
        print(f"Model loaded from {self.model_path} (trained for {checkpoint['epoch_idx'] + 1} epochs)")
        return checkpoint['metrics']

    def normalize_input(self, input_tensor):
        """Normalize input tensor based on training statistics"""
        if self.stats is None:
            return input_tensor

        means = torch.tensor(self.stats['inputs']['means']).to(self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats['inputs']['stds']).to(self.device).view(-1, 1, 1)
        return (input_tensor - means) / stds

    def denormalize_output(self, output_tensor):
        """Denormalize output tensor back to physical units"""
        if self.stats is None:
            return output_tensor

        means = torch.tensor(self.stats['outputs']['means']).to(self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats['outputs']['stds']).to(self.device).view(-1, 1, 1)
        return (output_tensor * stds) + means

    def predict_single_instance(self, input_tensor):
        """Make a prediction for a single input tensor"""
        # Ensure model is in eval mode
        self.model.eval()

        # Ensure input is a tensor with correct shape [1, 5, H, W]
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        # Add batch dimension if needed
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Ensure we have 5 channels
        assert input_tensor.shape[1] == 5, f"Input tensor should have 5 channels, got {input_tensor.shape[1]}"

        # Move to device
        input_tensor = input_tensor.to(self.device)

        # Normalize if stats are available
        if self.stats is not None:
            input_tensor = self.normalize_input(input_tensor)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)

        # Denormalize if stats are available
        if self.stats is not None:
            output = self.denormalize_output(output)

        return output

    def set_plotting_style(self):
        """Set consistent plotting style for visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161'])
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans',
                                           'sans-serif']
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16

    def visualize_inputs(self, input_tensor):
        """Visualize the 5 input channels"""
        self.set_plotting_style()

        # Make sure input is a tensor and on CPU for visualization
        if isinstance(input_tensor, torch.Tensor):
            input_data = input_tensor.cpu().squeeze().numpy()
        else:
            input_data = np.asarray(input_tensor).squeeze()

        # Create names for each channel
        channel_names = [
            'Domain (Padded)',
            'X Load',
            'Y Load',
            'X Boundary Condition',
            'Y Boundary Condition'
        ]

        # Create colormaps for different types of data
        cmaps = ['gray', 'inferno', 'inferno', 'Blues', 'Blues']

        # Create a figure with 5 subplots
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=150)
        fig.suptitle('Input Channels', fontsize=16, fontweight='bold', y=1.05)

        # Plot each channel
        for i in range(5):
            im = axes[i].imshow(input_data[i], cmap=cmaps[i])
            axes[i].set_title(channel_names[i])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("input_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig("input_visualization.svg", bbox_inches='tight')
        plt.close()

        print("Input visualization saved as PNG and SVG")

    def visualize_prediction(self, prediction):
        """Visualize the 2-channel prediction"""
        self.set_plotting_style()

        # Make sure prediction is a numpy array and on CPU for visualization
        if isinstance(prediction, torch.Tensor):
            pred_data = prediction.cpu().squeeze().numpy()
        else:
            pred_data = np.asarray(prediction).squeeze()

        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        fig.suptitle('Displacement Prediction', fontsize=16, fontweight='bold', y=1.05)

        # Plot x-displacement
        im0 = axes[0].imshow(pred_data[0], cmap='viridis')
        axes[0].set_title("X Displacement")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=9)

        # Plot y-displacement
        im1 = axes[1].imshow(pred_data[1], cmap='plasma')
        axes[1].set_title("Y Displacement")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        plt.tight_layout()
        plt.savefig("prediction_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig("prediction_visualization.svg", bbox_inches='tight')
        plt.close()

        print("Prediction visualization saved as PNG and SVG")

        # Return the numerical min/max values
        return {
            'x_min': float(pred_data[0].min()),
            'x_max': float(pred_data[0].max()),
            'y_min': float(pred_data[1].min()),
            'y_max': float(pred_data[1].max())
        }

    def create_combined_visualization(self, input_tensor, prediction):
        """Create a combined visualization of inputs and predictions"""
        self.set_plotting_style()

        # Make sure tensors are numpy arrays and on CPU for visualization
        if isinstance(input_tensor, torch.Tensor):
            input_data = input_tensor.cpu().squeeze().numpy()
        else:
            input_data = np.asarray(input_tensor).squeeze()

        if isinstance(prediction, torch.Tensor):
            pred_data = prediction.cpu().squeeze().numpy()
        else:
            pred_data = np.asarray(prediction).squeeze()

        # Create figure with subplots for both input and output
        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.suptitle('FEM Prediction Pipeline', fontsize=18, fontweight='bold', y=0.98)

        # Define grid for subplots
        gs = plt.GridSpec(2, 5, height_ratios=[1, 1], figure=fig)

        # Input channel names and colormaps
        channel_names = [
            'Domain (Padded)',
            'X Load',
            'Y Load',
            'X Boundary Condition',
            'Y Boundary Condition'
        ]
        cmaps = ['gray', 'inferno', 'inferno', 'Blues', 'Blues']

        # Plot input channels
        for i in range(5):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(input_data[i], cmap=cmaps[i])
            ax.set_title(channel_names[i])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add a divider with text
        ax_divider = fig.add_subplot(gs[1, :2])
        ax_divider.text(0.5, 0.5, "CNN Model\nPredicts Displacements",
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=1'))
        ax_divider.axis('off')

        # Plot prediction outputs
        ax_x = fig.add_subplot(gs[1, 2:4])
        im_x = ax_x.imshow(pred_data[0], cmap='viridis')
        ax_x.set_title("X Displacement")
        ax_x.set_xticks([])
        ax_x.set_yticks([])
        plt.colorbar(im_x, ax=ax_x, fraction=0.046, pad=0.04)

        ax_y = fig.add_subplot(gs[1, 4])
        im_y = ax_y.imshow(pred_data[1], cmap='plasma')
        ax_y.set_title("Y Displacement")
        ax_y.set_xticks([])
        ax_y.set_yticks([])
        plt.colorbar(im_y, ax=ax_y, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjust for suptitle
        plt.savefig("combined_visualization.png", dpi=300, bbox_inches='tight')
        plt.savefig("combined_visualization.svg", bbox_inches='tight')
        plt.close()

        print("Combined visualization saved as PNG and SVG")

    def benchmark_performance(self, input_tensor, num_iterations=100):
        """Benchmark inference performance on the given device"""
        print(f"Benchmarking inference performance on {self.device}...")

        # Ensure input is a tensor
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        # Add batch dimension if needed
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Move to device
        input_tensor = input_tensor.to(self.device)

        # Normalize if needed
        if self.stats is not None:
            input_tensor = self.normalize_input(input_tensor)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_tensor)

        # Measure time
        import time
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(input_tensor)

        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        print(f"Average inference time: {avg_time:.2f} ms per sample")
        return avg_time


def create_sample_input(shape=(64, 64)):
    """Create a sample input tensor with 5 channels for testing purposes"""
    # Channel 1: Domain (padded) - binary mask of the domain
    domain = np.ones(shape)

    # Channel 2-3: Loads in x and y directions
    # Creating point loads at specific locations
    load_x = np.zeros(shape)
    load_y = np.zeros(shape)
    load_x[shape[0] // 2, shape[1] - 1] = 1.0  # Point load on right edge

    # Channel 4-5: Boundary conditions in x and y directions
    # Fixed on the left edge
    bc_x = np.zeros(shape)
    bc_y = np.zeros(shape)
    bc_x[:, 0] = 1.0  # Fixed in x on left edge
    bc_y[:, 0] = 1.0  # Fixed in y on left edge

    # Stack channels
    input_tensor = np.stack([domain, load_x, load_y, bc_x, bc_y], axis=0)
    return torch.tensor(input_tensor, dtype=torch.float32)


def load_input_from_hdf5(hdf5_path, sample_idx=0):
    """Load a specific input sample from the HDF5 dataset"""
    with h5py.File(hdf5_path, 'r') as f:
        # Assuming the dataset structure has 'inputs' key with shape [n_samples, 5, height, width]
        inputs = f['inputs'][sample_idx]
        # If you also want to compare with ground truth:
        # outputs = f['outputs'][sample_idx]
        return torch.tensor(inputs, dtype=torch.float32)  # , torch.tensor(outputs, dtype=torch.float32)


def main():
    # Paths
    model_path = 'topology_Unet_model.pkl'
    hdf5_path = '../dataset-creation/cantilever-diagonal_dataset.h5'

    # Choose device
    # Set to None for auto-detection
    device = None

    # Load statistics if available
    with open('dataset-stats.json', 'r') as f:
        stats = json.load(f)

    # Initialize model
    model = TopologyOptimizationCNN()

    # Initialize tester
    tester = SingleInstanceTester(
        model=model,
        model_path=model_path,
        stats=stats,
        device=device
    )

    # Load trained model
    metrics = tester.load_model()

    # Choose whether to create a synthetic input or load from HDF5
    use_synthetic = True

    if use_synthetic:
        # Create a synthetic input tensor (5 channels)
        input_tensor = create_sample_input(shape=(181, 61))
        print(f"Created synthetic input tensor with shape: {input_tensor.shape}")
    else:
        # Load a specific sample from the HDF5 dataset
        input_tensor = load_input_from_hdf5(hdf5_path, sample_idx=0)
        print(f"Loaded input tensor from HDF5 with shape: {input_tensor.shape}")

    # Visualize input
    tester.visualize_inputs(input_tensor)

    # Run a quick benchmark to measure inference speed
    avg_time = tester.benchmark_performance(input_tensor, num_iterations=100)

    # Make prediction
    prediction = tester.predict_single_instance(input_tensor)
    print(f"Generated prediction with shape: {prediction.shape}")
    print(prediction[0].cpu().numpy())

    # Visualize prediction and get min/max values
    value_ranges = tester.visualize_prediction(prediction)
    print("Displacement ranges:")
    print(f"  X-displacement: {value_ranges['x_min']:.6f} to {value_ranges['x_max']:.6f}")
    print(f"  Y-displacement: {value_ranges['y_min']:.6f} to {value_ranges['y_max']:.6f}")

    # Create combined visualization
    tester.create_combined_visualization(input_tensor, prediction)


if __name__ == "__main__":
    main()