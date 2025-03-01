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
from CNN_model_node_level import TopologyOptimizationCNN


class ModelTester:
    def __init__(self, model, test_loader, model_path, stats=None):
        self.model = model
        self.test_loader = test_loader
        self.model_path = model_path
        self.stats = stats
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

    def visualize_predictions(self, predictions, targets, num_samples=5):
        """Visualize model predictions against ground truth"""
        if self.stats is not None:
            # Denormalize to physical units
            predictions = self.denormalize_tensor(predictions)
            targets = self.denormalize_tensor(targets)

        # Select random samples to visualize
        indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # X-displacement
            im0 = axes[0, 0].imshow(targets[idx, 0].numpy(), cmap='viridis')
            axes[0, 0].set_title("Ground Truth - X Displacement")
            plt.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(predictions[idx, 0].numpy(), cmap='viridis')
            axes[0, 1].set_title("Prediction - X Displacement")
            plt.colorbar(im1, ax=axes[0, 1])

            # Y-displacement
            im2 = axes[1, 0].imshow(targets[idx, 1].numpy(), cmap='viridis')
            axes[1, 0].set_title("Ground Truth - Y Displacement")
            plt.colorbar(im2, ax=axes[1, 0])

            im3 = axes[1, 1].imshow(predictions[idx, 1].numpy(), cmap='viridis')
            axes[1, 1].set_title("Prediction - Y Displacement")
            plt.colorbar(im3, ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig(f"prediction_sample_{i}.png")
            plt.close()

        print(f"Saved {len(indices)} visualization samples as PNG files")

    def analyze_error_distribution(self, predictions, targets):
        """Analyze the error distribution across the test set"""
        if self.stats is not None:
            # Denormalize to physical units
            predictions = self.denormalize_tensor(predictions)
            targets = self.denormalize_tensor(targets)

        # Calculate absolute errors
        x_errors = torch.abs(predictions[:, 0, :, :] - targets[:, 0, :, :]).flatten()
        y_errors = torch.abs(predictions[:, 1, :, :] - targets[:, 1, :, :]).flatten()

        # Plot error histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(x_errors.numpy(), bins=50, alpha=0.7, color='blue')
        ax1.set_title('X-Displacement Error Distribution')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')

        ax2.hist(y_errors.numpy(), bins=50, alpha=0.7, color='red')
        ax2.set_title('Y-Displacement Error Distribution')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig("error_distribution.png")
        plt.close()

        # Print statistics
        print(
            f"X-Displacement Error - Mean: {x_errors.mean():.6f}, Median: {x_errors.median():.6f}, Max: {x_errors.max():.6f}")
        print(
            f"Y-Displacement Error - Mean: {y_errors.mean():.6f}, Median: {y_errors.median():.6f}, Max: {y_errors.max():.6f}")

        return x_errors, y_errors

    def plot_training_history(self, metrics):
        """Plot the training history from saved metrics"""
        epochs = range(len(metrics['train_loss']))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot total loss
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot component losses
        ax2.plot(epochs, metrics['x_loss'], 'g-', label='X-Displacement Loss')
        ax2.plot(epochs, metrics['y_loss'], 'm-', label='Y-Displacement Loss')
        ax2.set_title('Component-wise Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()


def main():
    # Paths
    hdf5_path = 'cantilever-diagonal_dataset.h5'
    json_split_path = 'dataset_split.json'
    model_path = 'models/topology_cnn_checkpoint.pkl'

    # Parameters
    batch_size = 32

    # Calculate dataset statistics for normalization
    print("Loading dataset statistics...")
    stats = calculate_dataset_statistics(hdf5_path, json_split_path, batch_size)

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = FEMDataset(
        hdf5_path=hdf5_path,
        json_split_path=json_split_path,
        split='test',
        stats=stats
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
        stats=stats
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