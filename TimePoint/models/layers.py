import torch.nn as nn
from TimePoint.models.wtconv1d import WTConv1d


class ConvBlock1D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, norm=nn.BatchNorm1d, act=nn.ReLU, padding=1):  
        super(ConvBlock1D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride,  padding=padding),
            norm(c_out),
            act(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class WTConvBlock1D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, norm=nn.BatchNorm1d, act=nn.ReLU, wt_levels=3):
        super(WTConvBlock1D, self).__init__()
        self.layer = nn.Sequential(
            WTConv1d(c_in, c_in, kernel_size=kernel_size, wt_levels=wt_levels, stride=stride),
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=1, padding=0),
            norm(c_out),
            act()
        )
    def forward(self, x):
        return self.layer(x)