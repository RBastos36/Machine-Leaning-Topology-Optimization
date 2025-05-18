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
        """Visualize model predictions against ground truth and error maps, saving each separately."""
        self.set_plotting_style()

        if self.stats is not None:
            predictions = self.denormalize_tensor(predictions)
            targets = self.denormalize_tensor(targets)

        indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

        for i, idx in enumerate(indices):
            pred = predictions[idx]
            target = targets[idx]
            error = pred - target

            # Common min/max for better comparison
            x_min = min(target[0].min().item(), pred[0].min().item())
            x_max = max(target[0].max().item(), pred[0].max().item())
            y_min = min(target[1].min().item(), pred[1].min().item())
            y_max = max(target[1].max().item(), pred[1].max().item())

            error_x_min = error[0].min().item()
            error_x_max = error[0].max().item()
            error_y_min = error[1].min().item()
            error_y_max = error[1].max().item()

            ## === Ground Truth Figure ===
            fig_gt, axes_gt = plt.subplots(2, 1, figsize=(8, 10), dpi=150)
            fig_gt.suptitle(f'Ground Truth Displacements - Sample {i + 1}', fontsize=16, fontweight='bold', y=0.98)

            im_gt_x = axes_gt[0].imshow(target[0].numpy(), cmap='seismic', vmin=x_min, vmax=x_max)
            axes_gt[0].set_title("Ground Truth - X Displacement")
            axes_gt[0].set_xticks([])
            axes_gt[0].set_yticks([])
            cbar_gt_x = plt.colorbar(im_gt_x, ax=axes_gt[0], fraction=0.046, pad=0.04)
            cbar_gt_x.ax.tick_params(labelsize=9)

            im_gt_y = axes_gt[1].imshow(target[1].numpy(), cmap='seismic', vmin=y_min, vmax=y_max)
            axes_gt[1].set_title("Ground Truth - Y Displacement")
            axes_gt[1].set_xticks([])
            axes_gt[1].set_yticks([])
            cbar_gt_y = plt.colorbar(im_gt_y, ax=axes_gt[1], fraction=0.046, pad=0.04)
            cbar_gt_y.ax.tick_params(labelsize=9)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(f"sample_{i}_ground_truth.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"sample_{i}_ground_truth.svg", bbox_inches='tight')
            plt.close(fig_gt)

            ## === Prediction Figure ===
            fig_pred, axes_pred = plt.subplots(2, 1, figsize=(8, 10), dpi=150)
            fig_pred.suptitle(f'Predicted Displacements - Sample {i + 1}', fontsize=16, fontweight='bold', y=0.98)

            im_pred_x = axes_pred[0].imshow(pred[0].numpy(), cmap='seismic', vmin=x_min, vmax=x_max)
            axes_pred[0].set_title("Prediction - X Displacement")
            axes_pred[0].set_xticks([])
            axes_pred[0].set_yticks([])
            cbar_pred_x = plt.colorbar(im_pred_x, ax=axes_pred[0], fraction=0.046, pad=0.04)
            cbar_pred_x.ax.tick_params(labelsize=9)

            im_pred_y = axes_pred[1].imshow(pred[1].numpy(), cmap='seismic', vmin=y_min, vmax=y_max)
            axes_pred[1].set_title("Prediction - Y Displacement")
            axes_pred[1].set_xticks([])
            axes_pred[1].set_yticks([])
            cbar_pred_y = plt.colorbar(im_pred_y, ax=axes_pred[1], fraction=0.046, pad=0.04)
            cbar_pred_y.ax.tick_params(labelsize=9)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(f"sample_{i}_prediction.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"sample_{i}_prediction.svg", bbox_inches='tight')
            plt.close(fig_pred)

            ## === Error Figure ===
            fig_err, axes_err = plt.subplots(2, 1, figsize=(8, 10), dpi=150)
            fig_err.suptitle(f'Displacement Error Maps - Sample {i + 1}', fontsize=16, fontweight='bold', y=0.98)

            im_err_x = axes_err[0].imshow(error[0].numpy(), cmap='seismic', vmin=error_x_min, vmax=error_x_max)
            axes_err[0].set_title("Error - X Displacement (Prediction - Ground Truth)")
            axes_err[0].set_xticks([])
            axes_err[0].set_yticks([])
            cbar_err_x = plt.colorbar(im_err_x, ax=axes_err[0], fraction=0.046, pad=0.04)
            cbar_err_x.ax.tick_params(labelsize=9)

            im_err_y = axes_err[1].imshow(error[1].numpy(), cmap='seismic', vmin=error_y_min, vmax=error_y_max)
            axes_err[1].set_title("Error - Y Displacement (Prediction - Ground Truth)")
            axes_err[1].set_xticks([])
            axes_err[1].set_yticks([])
            cbar_err_y = plt.colorbar(im_err_y, ax=axes_err[1], fraction=0.046, pad=0.04)
            cbar_err_y.ax.tick_params(labelsize=9)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig(f"sample_{i}_error.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"sample_{i}_error.svg", bbox_inches='tight')
            plt.close(fig_err)

        print(f"Saved {len(indices)} samples: ground truth, prediction, and error maps as PNG and SVG files each.")

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