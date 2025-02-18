import torch
from torch import nn
import lightning as L
from torchmetrics.classification import BinaryAccuracy
from model.inception_block import InceptionBlock

# For testing purposes
from data.signal_data import SignalDataModule
import numpy as np
import time

class SignalFeatureModel(nn.Module):
    """SignalFeatureModel as proposed in DeepSignal by Ni et al. 2019"""
    def __init__(self, c_out=64, bias_mode=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=c_out, kernel_size=7, stride=2, padding=3, bias=bias_mode),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=c_out, out_channels=c_out*2, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=c_out*2),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_out*2, out_channels=c_out*4, kernel_size=3, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=c_out*4),
            nn.ReLU(),
        )

        inception_x3_1 = [InceptionBlock(c_in=c_out*4)] + 2 * [InceptionBlock()]
        inception_x5 = 5 * [InceptionBlock()]
        inception_x3_2 = 3 * [InceptionBlock()]

        self.body = nn.Sequential(
            *inception_x3_1,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *inception_x5,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *inception_x3_2,
            nn.AvgPool1d(kernel_size=7, stride=1, padding=1),
        )

        self.head = nn.Flatten()
        
        self.net = nn.Sequential(self.stem, self.body, self.head)

    def forward(self, x):
        return self.net(x)

class LitSignalFeatureModel(L.LightningModule):
    """Lightning wrapper for SignalFeatureModel used for development only."""
    def __init__(self, dropout=0, lr=1e-3, linear_dimensions=(3600, 3600)):
        super().__init__()
        self.model = SignalFeatureModel(dropout=dropout, linear_dimensions=linear_dimensions)
        self.loss_fn = nn.BCELoss()
        self.acc = BinaryAccuracy(threshold=0.5)
        self.lr = lr
        self.reset_activation_history = True

    def __save_activations(self, layer_alias):
        '''Hook function to save activations and corresponding labels'''
        def hook_fn(model, input, output):
            # module: nn.Module, input: Tensor, output: Tensor
            self.activations[layer_alias].append(output.detach().numpy(force=True))
        return hook_fn

    def remove_hooks(self):
        self.hook_handle_first_linear.remove()

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        X, y, _ = batch
        pred = self.model(X.unsqueeze(1))
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch
        pred = self.model(X.unsqueeze(1))
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("val_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        X, y, idx = batch
        # Initialize dict and register hook only during first test step
        if self.reset_activation_history == True:
            self.labels = []
            self.data_indices = []
            self.activations = {"first_linear": []}
            self.hook_handle_first_linear = self.model.net[2][1].register_forward_hook(self.__save_activations("first_linear"))
            self.reset_activation_history = False
        
        # Save labels and data indices
        self.labels.append(y.numpy(force=True))
        self.data_indices.append(idx.numpy(force=True))

        # Perform test step logic
        pred = self.model(X.unsqueeze(1))
        loss = self.loss_fn(pred, y.unsqueeze(-1))
        acc = self.acc(pred, y.unsqueeze(-1))
        self.log("test_acc", acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)