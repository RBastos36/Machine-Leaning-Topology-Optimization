import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from colorama import Fore, Style
import matplotlib.pyplot as plt


class TopologyTrainer:
    def __init__(self, model, train_loader, validation_loader, learning_rate, num_epochs,
                 model_path, load_model, stats=None):
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

        # Setup two matplotlib figures - one for total loss, one for separate x/y displacement losses
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self.ax1.set_title('Total Training Loss', fontweight="bold")
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('MSE Loss')
        
        self.ax2.set_title('Component-wise Loss', fontweight="bold")
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('MSE Loss')
        
        self.handles = {'total': None, 'total_val': None, 'x': None, 'y': None}

        self.model_path = model_path
        self.load_model = load_model if os.path.isfile(model_path) else False

    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to physical units"""
        if self.stats is None:
            return predictions

        means = torch.tensor(self.stats['outputs']['means']).to(self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats['outputs']['stds']).to(self.device).view(-1, 1, 1)
        return (predictions * stds) + means

    def compute_component_losses(self, predicted, target):
        """Compute separate losses for x and y displacement components"""
        # Option 1: Calculate loss in normalized space (current behavior)
        x_loss = nn.MSELoss()(predicted[:, 0, :, :], target[:, 0, :, :])
        y_loss = nn.MSELoss()(predicted[:, 1, :, :], target[:, 1, :, :])

        # Option 2: Calculate loss in physical space (if stats are available)
        if self.stats is not None:
            # Denormalize predictions and targets
            pred_physical = self.denormalize_predictions(predicted)
            target_physical = self.denormalize_predictions(target)

            # Calculate physical errors (these could be stored separately)
            # x_error_physical = nn.MSELoss()(pred_physical[:, 0, :, :], target_physical[:, 0, :, :])
            # y_error_physical = nn.MSELoss()(pred_physical[:, 1, :, :], target_physical[:, 1, :, :])

        return x_loss, y_loss


    def draw(self, metrics):
        epochs = range(len(metrics['train_loss']))
        
        # Update total loss plot
        if self.handles['total'] is None:
            self.handles['total'], = self.ax1.plot(epochs, metrics['train_loss'], '-b', label='Train')
            self.handles['total_val'], = self.ax1.plot(epochs, metrics['val_loss'], '-r', label='Validation')
            self.ax1.legend()
        else:
            self.handles['total'].set_data(epochs, metrics['train_loss'])
            self.handles['total_val'].set_data(epochs, metrics['val_loss'])
        
        # Update component loss plot
        if self.handles['x'] is None:
            self.handles['x'], = self.ax2.plot(epochs, metrics['x_loss'], '-g', label='X-displacement')
            self.handles['y'], = self.ax2.plot(epochs, metrics['y_loss'], '-m', label='Y-displacement')
            self.ax2.legend()
        else:
            self.handles['x'].set_data(epochs, metrics['x_loss'])
            self.handles['y'].set_data(epochs, metrics['y_loss'])

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        plt.draw()
        plt.pause(0.1)


    def train(self):
        if self.load_model:
            checkpoint = torch.load(self.model_path)
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

        for epoch_idx in range(start_epoch, self.num_epochs):
            print(f'Starting epoch {epoch_idx}')
            
            # Training
            self.model.train()
            train_losses = []
            epoch_x_losses = []
            epoch_y_losses = []
            
            for inputs, targets in tqdm(self.train_loader, desc=f'Training epoch {epoch_idx}'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(inputs)
                loss = self.loss(predictions, targets)
                
                x_loss, y_loss = self.compute_component_losses(predictions, targets)
                epoch_x_losses.append(x_loss.item())
                epoch_y_losses.append(y_loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for inputs, targets in tqdm(self.validation_loader, desc=f'Validating epoch {epoch_idx}'):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    predictions = self.model(inputs)
                    loss = self.loss(predictions, targets)
                    val_losses.append(loss.item())
            
            # Update metrics
            metrics['train_loss'].append(np.mean(train_losses))
            metrics['val_loss'].append(np.mean(val_losses))
            metrics['x_loss'].append(np.mean(epoch_x_losses))
            metrics['y_loss'].append(np.mean(epoch_y_losses))
            
            print(f'Epoch {epoch_idx} - Train Loss: {metrics["train_loss"][-1]:.6f}, '
                  f'Val Loss: {metrics["val_loss"][-1]:.6f}')
            print(f'X-displacement Loss: {metrics["x_loss"][-1]:.6f}, '
                  f'Y-displacement Loss: {metrics["y_loss"][-1]:.6f}')
            
            # Save model
            self.save_model(epoch_idx, metrics)
            
            # Update plots
            self.draw(metrics)


    def save_model(self, epoch_idx, metrics):
        print(f'Saving model to {self.model_path}... ', end='')
        torch.save({
            'epoch_idx': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, self.model_path)
        print('Saved.')
