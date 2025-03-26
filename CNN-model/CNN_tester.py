#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import h5py

from CNN_dataset import FEMDataset, calculate_dataset_statistics
from CNN_model_simple_node_level import TopologyOptimizationCNN


class ModelTester:
    def __init__(self, model, test_loader, model_path, stats=None):
        self.model = model
        self.test_loader = test_loader
        self.model_path = model_path
        self.stats = stats
        self.device = 'cpu'
        print(f'Using device: {self.device}')

    def load_model(self):
        """Load the trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        # Set weights_only=False to maintain compatibility with older PyTorch versions
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {self.model_path} (trained for {checkpoint['epoch_idx'] + 1} epochs)")
        return checkpoint['metrics']

    def denormalize_tensor(self, tensor):
        """Denormalize tensor back to physical units"""
        if self.stats is None:
            return tensor

        means = torch.tensor(self.stats['outputs']['means']).to(self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats['outputs']['stds']).to(self.device).view(-1, 1, 1)
        return (tensor * stds) + means

    def evaluate(self):
        """Evaluate the model on test data"""
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        x_loss = 0.0
        y_loss = 0.0
        mse_loss = torch.nn.MSELoss()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating model'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)

                # Calculate losses
                loss = mse_loss(predictions, targets)
                total_loss += loss.item() * inputs.size(0)

                x_loss += mse_loss(predictions[:, 0, :, :], targets[:, 0, :, :]).item() * inputs.size(0)
                y_loss += mse_loss(predictions[:, 1, :, :], targets[:, 1, :, :]).item() * inputs.size(0)

                # Store for visualization
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        # Calculate average loss
        avg_loss = total_loss / len(self.test_loader.dataset)
        avg_x_loss = x_loss / len(self.test_loader.dataset)
        avg_y_loss = y_loss / len(self.test_loader.dataset)

        print(f"Test Loss: {avg_loss:.6f}")
        print(f"X-displacement Loss: {avg_x_loss:.6f}")
        print(f"Y-displacement Loss: {avg_y_loss:.6f}")

        # Return concatenated predictions and targets for visualization
        return torch.cat(all_predictions), torch.cat(all_targets)

    def set_plotting_style(self):
        """Set consistent plotting style for all visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')
        # Define a color-blind friendly palette
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

    def visualize_predictions(self, predictions, targets, num_samples=5):
        """Visualize model predictions against ground truth with enhanced styling"""
        self.set_plotting_style()

        if self.stats is not None:
            # Denormalize to physical units
            predictions = self.denormalize_tensor(predictions)
            targets = self.denormalize_tensor(targets)

        # Select random samples to visualize
        indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

        # Use consistent colormaps for all visualizations
        cmap_x = 'viridis'
        cmap_y = 'plasma'

        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

            # Create a common title for the figure
            fig.suptitle(f'Displacement Prediction Sample {i + 1}', fontsize=16, fontweight='bold', y=0.98)

            # Get min and max values for consistent color scaling
            x_min = min(targets[idx, 0].min().item(), predictions[idx, 0].min().item())
            x_max = max(targets[idx, 0].max().item(), predictions[idx, 0].max().item())
            y_min = min(targets[idx, 1].min().item(), predictions[idx, 1].min().item())
            y_max = max(targets[idx, 1].max().item(), predictions[idx, 1].max().item())

            # X-displacement
            im0 = axes[0, 0].imshow(targets[idx, 0].numpy(), cmap=cmap_x, vmin=x_min, vmax=x_max)
            axes[0, 0].set_title("Ground Truth - X Displacement")
            axes[0, 0].set_xticks([])  # Hide ticks for cleaner look
            axes[0, 0].set_yticks([])
            cbar0 = plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            cbar0.ax.tick_params(labelsize=9)

            im1 = axes[0, 1].imshow(predictions[idx, 0].numpy(), cmap=cmap_x, vmin=x_min, vmax=x_max)
            axes[0, 1].set_title("Prediction - X Displacement")
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=9)

            # Y-displacement
            im2 = axes[1, 0].imshow(targets[idx, 1].numpy(), cmap=cmap_y, vmin=y_min, vmax=y_max)
            axes[1, 0].set_title("Ground Truth - Y Displacement")
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=9)

            im3 = axes[1, 1].imshow(predictions[idx, 1].numpy(), cmap=cmap_y, vmin=y_min, vmax=y_max)
            axes[1, 1].set_title("Prediction - Y Displacement")
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            cbar3 = plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=9)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)  # Adjust for suptitle
            plt.savefig(f"prediction_sample_{i}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"prediction_sample_{i}.svg", bbox_inches='tight')
            plt.close()

        print(f"Saved {len(indices)} visualization samples as PNG and SVG files")

    def analyze_error_distribution(self, predictions, targets):
        """Analyze the error distribution across the test set with enhanced styling"""
        self.set_plotting_style()

        if self.stats is not None:
            # Denormalize to physical units
            predictions = self.denormalize_tensor(predictions)
            targets = self.denormalize_tensor(targets)

        # Calculate absolute errors
        x_errors = torch.abs(predictions[:, 0, :, :] - targets[:, 0, :, :]).flatten()
        y_errors = torch.abs(predictions[:, 1, :, :] - targets[:, 1, :, :]).flatten()

        # Get color palette from style
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Plot error histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Add figure title
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)

        # X-displacement errors
        bins = np.linspace(0, max(x_errors.max().item(), y_errors.max().item()), 50)

        ax1.hist(x_errors.numpy(), bins=bins, alpha=0.8, color=colors[0], edgecolor='white', linewidth=0.8)
        ax1.set_title('X-Displacement Error Distribution')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Add mean and median lines
        x_mean = x_errors.mean().item()
        x_median = x_errors.median().item()
        ax1.axvline(x_mean, color=colors[2], linestyle='-', linewidth=2, label=f'Mean: {x_mean:.4f}')
        ax1.axvline(x_median, color=colors[3], linestyle='--', linewidth=2, label=f'Median: {x_median:.4f}')
        ax1.legend()

        # Y-displacement errors
        ax2.hist(y_errors.numpy(), bins=bins, alpha=0.8, color=colors[1], edgecolor='white', linewidth=0.8)
        ax2.set_title('Y-Displacement Error Distribution')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Add mean and median lines
        y_mean = y_errors.mean().item()
        y_median = y_errors.median().item()
        ax2.axvline(y_mean, color=colors[2], linestyle='-', linewidth=2, label=f'Mean: {y_mean:.4f}')
        ax2.axvline(y_median, color=colors[3], linestyle='--', linewidth=2, label=f'Median: {y_median:.4f}')
        ax2.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust for suptitle
        plt.savefig("error_distribution.png", dpi=300, bbox_inches='tight')
        plt.savefig("error_distribution.svg", bbox_inches='tight')
        plt.close()

        # Print statistics
        print(f"X-Displacement Error - Mean: {x_mean:.6f}, Median: {x_median:.6f}, Max: {x_errors.max():.6f}")
        print(f"Y-Displacement Error - Mean: {y_mean:.6f}, Median: {y_median:.6f}, Max: {y_errors.max():.6f}")

        return x_errors, y_errors

    def plot_training_history(self, metrics):
        """Plot the training history from saved metrics with enhanced style"""
        self.set_plotting_style()

        # Get colors from the consistent palette
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Create figure with better resolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Add figure title
        fig.suptitle('Training History', fontsize=16, fontweight='bold', y=0.98)

        epochs = range(1, len(metrics['train_loss']) + 1)

        # Plot total loss with improved styling
        ax1.plot(epochs, metrics['train_loss'], '-', color=colors[0], linewidth=2,
                 label='Training Loss', marker='o', markersize=4)
        ax1.plot(epochs, metrics['val_loss'], '-', color=colors[1], linewidth=2,
                 label='Validation Loss', marker='s', markersize=4)

        # Fill area between curves for visual emphasis
        ax1.fill_between(epochs, metrics['train_loss'], metrics['val_loss'],
                         alpha=0.1, color=colors[1])

        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(frameon=True, fontsize=10, loc='upper right',
                   facecolor='white', edgecolor='gray')

        # Plot component losses with improved styling
        ax2.plot(epochs, metrics['x_loss'], '-', color=colors[2], linewidth=2,
                 label='X-Displacement Loss', marker='o', markersize=4)
        ax2.plot(epochs, metrics['y_loss'], '-', color=colors[3], linewidth=2,
                 label='Y-Displacement Loss', marker='s', markersize=4)

        # Fill area between curves for visual emphasis
        ax2.fill_between(epochs, metrics['x_loss'], metrics['y_loss'],
                         alpha=0.1, color=colors[3])

        ax2.set_title('Component-wise Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(frameon=True, fontsize=10, loc='upper right',
                   facecolor='white', edgecolor='gray')

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust for suptitle

        # Save with higher quality
        plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
        plt.savefig("training_history.svg", bbox_inches='tight')
        plt.close()


def main():
    # Paths
    hdf5_path = '../dataset-creation/cantilever-diagonal_dataset.h5'
    json_split_path = '../dataset-creation/dataset_split_stratified.json'
    model_path = 'models/topology_cnn_model.pkl'

    # Parameters
    batch_size = 32

    # Calculate dataset statistics for normalization
    print("Loading dataset statistics...")
    #stats = calculate_dataset_statistics(hdf5_path, json_split_path, batch_size)

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = FEMDataset(
        hdf5_path=hdf5_path,
        json_split_path=json_split_path,
        split='test',
        #stats=stats
    )

    # Create test data loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = TopologyOptimizationCNN()

    # Initialize tester
    tester = ModelTester(
        model=model,
        test_loader=test_loader,
        model_path=model_path,
        #stats=stats
    )

    # Load trained model
    metrics = tester.load_model()

    # Plot training history
    tester.plot_training_history(metrics)

    # Evaluate model
    predictions, targets = tester.evaluate()

    # Visualize predictions
    tester.visualize_predictions(predictions, targets, num_samples=5)

    # Analyze error distribution
    tester.analyze_error_distribution(predictions, targets)


if __name__ == "__main__":
    main()