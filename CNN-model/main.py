#!/usr/bin/env python3


from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from CNN_dataset import FEMDataset, calculate_dataset_statistics
from CNN_model_simple_node_level import TopologyOptimizationCNN
from CNN_trainer import TopologyTrainer


def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 40

    # Paths
    hdf5_path = '../dataset-creation/cantilever-diagonal_dataset.h5'
    json_split_path = '../dataset-creation/dataset_split.json'
    model_save_path = 'models/topology_cnn_model.pkl'

    # Calculate dataset statistics for normalization
    print("Calculating dataset statistics...")
    stats = calculate_dataset_statistics(hdf5_path, json_split_path, batch_size)

    # Create datasets
    print("Creating datasets...")
    train_dataset = FEMDataset(
        hdf5_path=hdf5_path,
        json_split_path=json_split_path,
        split='train',
        stats=stats
    )

    validation_dataset = FEMDataset(
        hdf5_path=hdf5_path,
        json_split_path=json_split_path,
        split='validation',
        stats=stats
    )

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
        load_model=False,  # Set to True to resume training
        stats=stats
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Show the final training plots
    plt.show()


if __name__ == "__main__":
    main()
