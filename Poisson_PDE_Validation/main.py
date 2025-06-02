# Author: Ricardo A. O. Bastos
# Created: June 2025


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import PoissonDataset
from model_Unet_node_level import TopologyOptimizationCNN
from trainer import TopologyTrainer


def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 150

    # Paths
    model_save_path = 'models/topology_Unet_model_Poisson.pkl'

    # Create datasets
    print("Creating datasets...")
    train_dataset = PoissonDataset(num_samples=10000, grid_size=64)
    validation_dataset = PoissonDataset(num_samples=2000, grid_size=64)

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    # Initialize model
    model = TopologyOptimizationCNN()

    # Initialize trainer
    trainer = TopologyTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_path=model_save_path,
        load_model=False  # Set to True to resume training
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Show the final training plots
    plt.show()


if __name__ == "__main__":
    main()
