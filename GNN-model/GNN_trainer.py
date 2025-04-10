import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore, Style
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


class TopologyGNNTrainer:
    def __init__(self, model, train_loader, validation_loader, learning_rate, num_epochs,
                 model_path, load_model=False, stats=None):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.stats = stats

        # Using MSE Loss since this is a regression problem
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(Fore.BLUE + 'Device is ' + self.device + Style.RESET_ALL)

        # Enable interactive mode for matplotlib
        plt.ion()

        # Setup two matplotlib figures - one for total loss, one for separate x/y displacement losses
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))

        self.ax1.set_title('Total Training Loss', fontweight="bold")
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('MSE Loss')

        self.ax2.set_title('Component-wise Loss', fontweight="bold")
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('MSE Loss')

        # Initialize empty plots
        self.handles = {
            'total': self.ax1.plot([], [], '-b', label='Train')[0],
            'total_val': self.ax1.plot([], [], '-r', label='Validation')[0],
            'x': self.ax2.plot([], [], '-g', label='X-displacement')[0],
            'y': self.ax2.plot([], [], '-m', label='Y-displacement')[0]
        }

        self.ax1.legend()
        self.ax2.legend()

        self.model_path = model_path
        self.load_model = load_model if os.path.isfile(model_path) else False

    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to physical units"""
        if self.stats is None:
            return predictions

        means = torch.tensor(self.stats['outputs']['means']).to(self.device)
        stds = torch.tensor(self.stats['outputs']['stds']).to(self.device)
        
        return (predictions * stds) + means

    def compute_component_losses(self, predicted, target):
        """Compute separate losses for x and y displacement components"""
        # Split the predicted and target tensors into x and y components
        x_pred, y_pred = predicted[:, 0], predicted[:, 1]
        x_target, y_target = target[:, 0], target[:, 1]
        
        # Calculate individual losses
        x_loss = nn.MSELoss()(x_pred, x_target)
        y_loss = nn.MSELoss()(y_pred, y_target)
        
        # Option to calculate loss in physical space (if stats are available)
        if self.stats is not None:
            # For tracking physical errors, not used in training
            pred_physical = self.denormalize_predictions(predicted)
            target_physical = self.denormalize_predictions(target)
            
            # x_error_physical = nn.MSELoss()(pred_physical[:, 0], target_physical[:, 0])
            # y_error_physical = nn.MSELoss()(pred_physical[:, 1], target_physical[:, 1])

        return x_loss, y_loss

    def draw(self, metrics):
        epochs = list(range(len(metrics['train_loss'])))

        # Update data for all plots
        self.handles['total'].set_data(epochs, metrics['train_loss'])
        self.handles['total_val'].set_data(epochs, metrics['val_loss'])
        self.handles['x'].set_data(epochs, metrics['x_loss'])
        self.handles['y'].set_data(epochs, metrics['y_loss'])

        # Reset the view limits for both axes
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()

        # Add a small margin to y-axis to prevent plots from hitting the top
        self.ax1.margins(y=0.1)
        self.ax2.margins(y=0.1)

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # To ensure the plot is displayed properly
        plt.pause(0.1)

    def train(self):
        if self.load_model:
            # NOTE: Using weights_only=False when loading models to handle PyTorch 2.6+ compatibility.
            checkpoint = torch.load(self.model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            start_epoch = checkpoint['epoch_idx']
            metrics = checkpoint['metrics']
        else:
            start_epoch = 0
            metrics = {
                'train_loss': [],
                'val_loss': [],
                'x_loss': [],
                'y_loss': []
            }

        self.model.to(self.device)

        # Draw initial empty plot
        self.draw(metrics)

        for epoch_idx in range(start_epoch, self.num_epochs):
            print(f'Starting epoch {epoch_idx}')

            # Training
            self.model.train()
            train_losses = []
            epoch_x_losses = []
            epoch_y_losses = []

            for batch in tqdm(self.train_loader, desc=f'Training epoch {epoch_idx}'):
                batch = batch.to(self.device)
                
                # Predict displacements
                predictions = self.model(batch)
                
                # Calculate loss
                loss = self.loss(predictions, batch.y)
                
                # Calculate component-wise losses
                x_loss, y_loss = self.compute_component_losses(predictions, batch.y)
                epoch_x_losses.append(x_loss.item())
                epoch_y_losses.append(y_loss.item())
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in tqdm(self.validation_loader, desc=f'Validating epoch {epoch_idx}'):
                    batch = batch.to(self.device)
                    
                    # Predict displacements
                    predictions = self.model(batch)
                    
                    # Calculate loss
                    loss = self.loss(predictions, batch.y)
                    val_losses.append(loss.item())

            # Update metrics
            metrics['train_loss'].append(np.mean(train_losses))
            metrics['val_loss'].append(np.mean(val_losses))
            metrics['x_loss'].append(np.mean(epoch_x_losses))
            metrics['y_loss'].append(np.mean(epoch_y_losses))

            # Draw updated plot
            self.draw(metrics)

            # Print current losses
            print(f'Epoch {epoch_idx} - Train Loss: {metrics["train_loss"][-1]:.6f}, '
                  f'Validation Loss: {metrics["val_loss"][-1]:.6f}')
            print(f'X-displacement Loss: {metrics["x_loss"][-1]:.6f}, '
                  f'Y-displacement Loss: {metrics["y_loss"][-1]:.6f}')

            # Save model checkpoint
            # torch.save({
            #     'epoch_idx': epoch_idx + 1,
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            #     'metrics': metrics
            # }, self.model_path)

            if (len(metrics['val_loss']) <= 1) or (metrics['val_loss'][-1] == min(metrics['val_loss'])):
                # save the model
                torch.save({
                    'epoch_idx': epoch_idx + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics
                }, self.model_path)
                print(Fore.MAGENTA + "Model saved!" + Style.RESET_ALL)
            else:
                print(Fore.MAGENTA + "Model not saved to prevent Overfitting!" + Style.RESET_ALL)

            # Early stopping could be implemented here
            # if epoch_idx > min_epochs and metrics['val_loss'][-1] > metrics['val_loss'][-2]:
            #     print(f'Early stopping at epoch {epoch_idx}')
            #     break

        print(Fore.GREEN + f'Training completed after {self.num_epochs} epochs' + Style.RESET_ALL)
        plt.ioff()  # Turn off interactive mode
        
        return metrics
    
    def evaluate(self, test_loader):
        """Evaluate model on test set and compute metrics"""
        self.model.to(self.device)
        self.model.eval()
        
        test_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating on test set'):
                batch = batch.to(self.device)
                
                # Predict displacements
                predictions = self.model(batch)
                
                # Calculate loss
                loss = self.loss(predictions, batch.y)
                test_losses.append(loss.item())
                
                # Store predictions and targets for additional metrics
                if self.stats is not None:
                    # Denormalize if using stats
                    pred_physical = self.denormalize_predictions(predictions)
                    target_physical = self.denormalize_predictions(batch.y)
                    all_predictions.append(pred_physical.cpu())
                    all_targets.append(target_physical.cpu())
                else:
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch.y.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate overall MSE
        test_mse = np.mean(test_losses)
        
        # Calculate component-wise MSE
        x_mse = nn.MSELoss()(all_predictions[:, 0], all_targets[:, 0]).item()
        y_mse = nn.MSELoss()(all_predictions[:, 1], all_targets[:, 1]).item()
        
        # Calculate RMSE
        test_rmse = np.sqrt(test_mse)
        x_rmse = np.sqrt(x_mse)
        y_rmse = np.sqrt(y_mse)
        
        # Calculate Mean Absolute Error
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        x_mae = torch.mean(torch.abs(all_predictions[:, 0] - all_targets[:, 0])).item()
        y_mae = torch.mean(torch.abs(all_predictions[:, 1] - all_targets[:, 1])).item()
        
        # Calculate maximum error
        max_error = torch.max(torch.abs(all_predictions - all_targets)).item()
        
        # Print results
        print(Fore.CYAN + "\nTest Set Evaluation Results:" + Style.RESET_ALL)
        print(f"Overall MSE: {test_mse:.6f}")
        print(f"Overall RMSE: {test_rmse:.6f}")
        print(f"Overall MAE: {mae:.6f}")
        print(f"Maximum Error: {max_error:.6f}")
        print("\nComponent-wise Results:")
        print(f"X-displacement - RMSE: {x_rmse:.6f}, MAE: {x_mae:.6f}")
        print(f"Y-displacement - RMSE: {y_rmse:.6f}, MAE: {y_mae:.6f}")
        
        results = {
            "mse": test_mse,
            "rmse": test_rmse, 
            "mae": mae,
            "max_error": max_error,
            "x_rmse": x_rmse,
            "y_rmse": y_rmse,
            "x_mae": x_mae,
            "y_mae": y_mae
        }
        
        return results
