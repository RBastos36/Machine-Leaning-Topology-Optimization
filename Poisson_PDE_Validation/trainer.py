# Author: Ricardo A. O. Bastos
# Created: June 2025


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

        # Enable interactive mode for matplotlib
        plt.ion()

        # Setup two matplotlib figures - one for total loss, one for separate x/y displacement losses
        self.fig, (self.ax1) = plt.subplots(1, 1, figsize=(6, 5))

        self.ax1.set_title('Total Training Loss', fontweight="bold")
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('MSE Loss')

        # Initialize empty plots
        self.handles = {
            'total': self.ax1.plot([], [], '-b', label='Train')[0],
            'total_val': self.ax1.plot([], [], '-r', label='Validation')[0],
        }

        self.ax1.legend()

        self.model_path = model_path
        self.load_model = load_model if os.path.isfile(model_path) else False

    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to physical units"""
        if self.stats is None:
            return predictions

        means = torch.tensor(self.stats['outputs']['means']).to(self.device).view(-1, 1, 1)
        stds = torch.tensor(self.stats['outputs']['stds']).to(self.device).view(-1, 1, 1)
        return (predictions * stds) + means


    def draw(self, metrics):
        epochs = list(range(len(metrics['train_loss'])))

        # Update data for all plots
        self.handles['total'].set_data(epochs, metrics['train_loss'])
        self.handles['total_val'].set_data(epochs, metrics['val_loss'])

        # Reset the view limits
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Add a small margin to y-axis to prevent plots from hitting the top
        self.ax1.margins(y=0.1)

        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # To ensure the plot is displayed properly
        plt.pause(0.1)

    def train(self):
        if self.load_model:
            # NOTE: Using weights_only=False when loading models to handle PyTorch 2.6+ compatibility.
            # In PyTorch 2.6, the default changed from False to True, causing loading errors with
            # models saved in earlier versions.
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
            }

        self.model.to(self.device)

        # Draw initial empty plot
        self.draw(metrics)

        for epoch_idx in range(start_epoch, self.num_epochs):
            print(f'Starting epoch {epoch_idx}')

            # Training
            self.model.train()
            train_losses = []

            for inputs, targets in tqdm(self.train_loader, desc=f'Training epoch {epoch_idx}'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(inputs)
                loss = self.loss(predictions, targets)

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


            print(f'Epoch {epoch_idx} - Train Loss: {metrics["train_loss"][-1]:.6f}, '
                  f'Val Loss: {metrics["val_loss"][-1]:.6f}')


            if (len(metrics['val_loss']) <= 1) or (metrics['val_loss'][-1] == min(metrics['val_loss'])):
                # save the model
                self.save_model(epoch_idx, metrics)
                print(Fore.MAGENTA + "Model saved!" + Style.RESET_ALL)
            else:
                print(Fore.MAGENTA + "Model not saved to prevent Overfitting!" + Style.RESET_ALL)

            # Update plots
            self.draw(metrics)

    def save_model(self, epoch_idx, metrics):
        print(f'Saving model to {self.model_path}... ', end='')
        torch.save({
            'epoch_idx': epoch_idx + 1,  # Save next epoch to resume from
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, self.model_path)
        print('Saved.')