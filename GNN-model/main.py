#!/usr/bin/env python3

import os
import json
import torch
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
        load_model=load_model,  # Will load if the file exists
        stats=stats
    )

    # Train the model
    print("Starting training...")
    metrics = trainer.train()

    # # Evaluate on test set
    # print("Evaluating model on test set...")
    # test_results = trainer.evaluate(test_loader)
    #
    # # Save test results
    # results_path = os.path.splitext(model_save_path)[0] + '_results.json'
    # with open(results_path, 'w') as f:
    #     json.dump(test_results, f, indent=2)
    #
    # print(f"Test results saved to {results_path}")
    #
    # # Visualization of a few examples from test set (optional)
    # visualize_examples(model, test_loader, stats, num_examples=3)
    
    # Show the final training plots
    plt.show()


# def visualize_examples(model, test_loader, stats, num_examples=3):
#     """Visualize a few examples from the test set"""
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     model.eval()
#
#     # Create a figure for visualization
#     fig, axes = plt.subplots(num_examples, 2, figsize=(10, 4*num_examples))
#
#     # Get a few examples from the test set
#     shown = 0
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = batch.to(device)
#
#             # Predict displacements
#             predictions = model(batch)
#
#             # Denormalize if using stats
#             if stats is not None:
#                 means = torch.tensor(stats['outputs']['means']).to(device)
#                 stds = torch.tensor(stats['outputs']['stds']).to(device)
#
#                 pred_physical = (predictions * stds) + means
#                 target_physical = (batch.y * stds) + means
#             else:
#                 pred_physical = predictions
#                 target_physical = batch.y
#
#             # Visualize the examples
#             for i in range(min(batch.num_graphs, num_examples - shown)):
#                 if shown >= num_examples:
#                     break
#
#                 # Get node indices for this graph
#                 batch_mask = batch.batch == i
#                 nodes_pos = batch.pos[batch_mask].cpu().numpy()
#                 true_disp = target_physical[batch_mask].cpu().numpy()
#                 pred_disp = pred_physical[batch_mask].cpu().numpy()
#
#                 # Original mesh with true displacements
#                 ax = axes[shown, 0]
#                 ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], c='blue', alpha=0.5, s=10)
#                 ax.quiver(nodes_pos[:, 0], nodes_pos[:, 1], true_disp[:, 0], true_disp[:, 1],
#                           color='red', scale=0.5, width=0.003)
#                 ax.set_title(f'Example {shown+1}: True Displacements')
#                 ax.set_aspect('equal')
#
#                 # Original mesh with predicted displacements
#                 ax = axes[shown, 1]
#                 ax.scatter(nodes_pos[:, 0], nodes_pos[:, 1], c='blue', alpha=0.5, s=10)
#                 ax.quiver(nodes_pos[:, 0], nodes_pos[:, 1], pred_disp[:, 0], pred_disp[:, 1],
#                           color='green', scale=0.5, width=0.003)
#                 ax.set_title(f'Example {shown+1}: Predicted Displacements')
#                 ax.set_aspect('equal')
#
#                 shown += 1
#
#             if shown >= num_examples:
#                 break
#
#     plt.tight_layout()
#     plt.savefig('gnn_visualization.png')
#     print("Visualization saved to gnn_visualization.png")


if __name__ == "__main__":
    main()
