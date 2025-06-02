# Author: Ricardo A. O. Bastos
# Created: June 2025


import os
import matplotlib.pyplot as plt

from GNN_dataset import FEMGNN_Dataset, calculate_dataset_statistics, load_fem_gnn_data
from GNN_model import TopologyOptimizationGNN, DeepTopologyOptimizationGNN, EdgeFeatureGNN
from GNN_trainer import TopologyGNNTrainer


def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    hidden_dim = 64
    num_layers = 4
    conv_type = 'GCN'  # Options: 'GCN', 'GAT', 'GraphConv'
    model_type = 'standard'  # Options: 'standard', 'deep', 'edge'

    # Paths
    hdf5_path = 'cantilever-diagonal_dataset.h5'
    json_split_path = 'dataset_split.json'
    model_save_path = 'models/topology_gnn_checkpoint.pt'
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Calculate dataset statistics for normalization
    print("Calculating dataset statistics...")
    stats = calculate_dataset_statistics(hdf5_path, json_split_path, batch_size)

    # Create data loaders with normalization
    print("Creating data loaders...")
    train_loader, train_dataset = load_fem_gnn_data(
        hdf5_path, 
        json_split_path,
        split='train', 
        stats=stats,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader, val_dataset = load_fem_gnn_data(
        hdf5_path, 
        json_split_path,
        split='validation', 
        stats=stats,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader, test_dataset = load_fem_gnn_data(
        hdf5_path, 
        json_split_path,
        split='test', 
        stats=stats,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model based on chosen type
    print(f"Initializing {model_type} GNN model with {conv_type} convolutions...")
    
    if model_type == 'standard':
        model = TopologyOptimizationGNN(
            input_dim=5,  # domain, fixed_x, fixed_y, loads_x, loads_y
            hidden_dim=hidden_dim,
            output_dim=2,  # displacement_x, displacement_y
            conv_type=conv_type,
            num_layers=num_layers
        )
    elif model_type == 'deep':
        model = DeepTopologyOptimizationGNN(
            input_dim=5,
            hidden_dims=(32, 64, 128, 64, 32),
            output_dim=2,
            conv_type=conv_type
        )
    elif model_type == 'edge':
        model = EdgeFeatureGNN(
            input_dim=5,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Initialize trainer
    load_model = os.path.isfile(model_save_path)
    trainer = TopologyGNNTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        model_path=model_save_path,
        load_model=load_model,
        stats=stats
    )

    # Train the model
    print("Starting training...")
    metrics = trainer.train()
    
    # Show the final training plots
    plt.show()


if __name__ == "__main__":
    main()
