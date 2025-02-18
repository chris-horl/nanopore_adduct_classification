import torch
from torch import nn

class InceptionBlock(nn.Module):
    """Implemented according to DeepSignal by Ni et al. 2019.
    
    Concept of inception block based on Szegedy et al. 2015.

    Args:
        c_in: Number of input channels
        filter_size: Smallest common multiple of number of output channels
        bias_mode: Bool flag to toggle bias on/off
    """

    def __init__(self, c_in: int=240, filter_size: int=16, bias_mode=False):
        super(InceptionBlock, self).__init__()

        self.b1_1x1conv = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=3*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size),
            nn.ReLU()
        )

        self.b2_1x3conv = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=2*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=2*filter_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*filter_size, out_channels=3*filter_size, kernel_size=3, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size),
            nn.ReLU()
        )

        self.b3_1x5conv = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=2*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=2*filter_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*filter_size, out_channels=3*filter_size, kernel_size=5, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size),
            nn.ReLU()
        )

        self.b4_1x3conv = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=2*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*filter_size, out_channels=4*filter_size, kernel_size=3, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=4*filter_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*filter_size, out_channels=3*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size),
        )

        self.b4_res = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=3*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size)
        )

        self.b5_1x3maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1), 
            nn.Conv1d(in_channels=c_in, out_channels=3*filter_size, kernel_size=1, stride=1, padding="same", bias=bias_mode),
            nn.BatchNorm1d(num_features=3*filter_size),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.cat((self.b1_1x1conv(x),
                          self.b2_1x3conv(x),
                          self.b3_1x5conv(x),
                          # merge b4 subbranches into residual 1x3 conv branch
                          nn.functional.relu((self.b4_1x3conv(x) + self.b4_res(x))),
                          self.b5_1x3maxpool(x)),
                         dim=1)