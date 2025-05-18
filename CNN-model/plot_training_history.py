import os
import torch
import matplotlib.pyplot as plt
from CNN_model_simple_node_level import TopologyOptimizationCNN


class ModelHistoryPlotter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cpu'
        print(f'Using device: {self.device}')

    def load_model(self):
        """Load the trained model from checkpoint and return the training metrics."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        print(f"Model loaded from {self.model_path} (trained for {checkpoint['epoch_idx'] + 1} epochs)")
        return checkpoint['metrics']

    def set_plotting_style(self):
        """Set consistent plotting style for all visualizations."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161'])
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['axes.labelsize'] = 16  # Axis label size
        plt.rcParams['axes.titlesize'] = 14  # Title size
        plt.rcParams['xtick.labelsize'] = 12  # X tick size
        plt.rcParams['ytick.labelsize'] = 12  # Y tick size
        plt.rcParams['legend.fontsize'] = 12  # Legend font size
        plt.rcParams['axes.titleweight'] = 'bold'

    def plot_training_history(self, metrics):
        """Plot the training history from saved metrics."""
        self.set_plotting_style()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        epochs = range(1, len(metrics['train_loss']) + 1)

        # First figure: Model Loss
        fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)

        ax1.plot(epochs, metrics['train_loss'], '-', color=colors[0], linewidth=2, label='Training Loss', marker='o',
                 markersize=4)
        ax1.plot(epochs, metrics['val_loss'], '-', color=colors[1], linewidth=2, label='Validation Loss', marker='s',
                 markersize=4)
        ax1.fill_between(epochs, metrics['train_loss'], metrics['val_loss'], alpha=0.1, color=colors[1])
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(frameon=True, fontsize=10, loc='upper right', facecolor='white', edgecolor='gray')

        plt.tight_layout()
        fig1.savefig("model_loss.svg")
        plt.show()

        # Second figure: Component-wise Loss
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)

        ax2.plot(epochs, metrics['x_loss'], '-', color=colors[2], linewidth=2, label='X-Displacement Loss', marker='o',
                 markersize=4)
        ax2.plot(epochs, metrics['y_loss'], '-', color=colors[3], linewidth=2, label='Y-Displacement Loss', marker='s',
                 markersize=4)
        ax2.fill_between(epochs, metrics['x_loss'], metrics['y_loss'], alpha=0.1, color=colors[3])
        ax2.set_title('Component-wise Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(frameon=True, fontsize=10, loc='upper right', facecolor='white', edgecolor='gray')

        plt.tight_layout()
        fig2.savefig("component_loss.svg")
        plt.show()


def main():
    model_path = 'models/topology_Unet_strided_model.pkl'
    plotter = ModelHistoryPlotter(model_path)
    metrics = plotter.load_model()
    plotter.plot_training_history(metrics)


if __name__ == "__main__":
    main()
