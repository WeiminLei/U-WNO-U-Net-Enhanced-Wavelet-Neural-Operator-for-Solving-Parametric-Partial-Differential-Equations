import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv1d

torch.manual_seed(0)
np.random.seed(0)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0),
        )
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, out_channel):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        x = self.layers(x)
        act = nn.LeakyReLU(0.1, inplace=True)
        return act(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        #self.layer = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                               #kernel_size=1, stride=1)
        self.layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x, feature_map):
        #up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(x)
        return torch.cat((out, feature_map), dim=1)


class UNet_1d(nn.Module):
    def __init__(self, in_channels):
        super(UNet_1d, self).__init__()
        # self.a = self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.down1 = DownSample(in_channels)
        self.down2 = DownSample(in_channels)
        self.conv2 = Conv_Block(in_channels, in_channels)
        self.down3 = DownSample(in_channels)
        self.conv3 = Conv_Block(in_channels, in_channels)
        self.up1 = UpSample(in_channels, in_channels)
        #self.conv4 = Conv_Block(2*in_channels, in_channels)
        self.up2 = UpSample(2*in_channels, in_channels)
        self.up3 = UpSample(2*in_channels, in_channels)
        self.out = nn.Conv1d(2*in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        R1 = self.down1(x)
        R2 = self.conv2(self.down2(R1))
        R3 = self.conv3(self.down3(R2))
        O2 = self.up1(R3, R2)
        O1 = self.up2(O2,R1)
        O0 = self.up3(O1,x)
        return self.out(O0)
# %%
""" The forward operation """


class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(WNO1d, self).__init__()

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding

        self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        self.unet = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channel, self.width)  # input channel is 2: (a(x), x)
        for i in range(self.layers):
            self.conv.append(WaveConv1d(self.width, self.width, self.level, self.size, self.wavelet))
            self.w.append(nn.Conv1d(self.width, self.width, 1))
            self.unet.append(UNet_1d(self.width))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)  # Shape: Batch * x * Channel
        x = x.permute(0, 2, 1)  # Shape: Batch * Channel * x
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])
        r = x
        for index, (convl, wl, unetl) in enumerate(zip(self.conv, self.w, self.unet)):
            x = convl(x + r) + wl(x) + unetl(x)
            if index != self.layers - 1:  # Final layer has no activation
                x = F.mish(10 * self.a * x)  # Shape: Batch * Channel * x

        if self.padding != 0:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)  # Shape: Batch * x * Channel
        x = F.gelu(self.fc1(x))  # Shape: Batch * x * Channel
        x = self.fc2(x)  # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%
""" Model configurations """

PATH = 'data/train_IC2.npz'
ntrain = 900
ntest = 100

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 50  # weight-decay step size
gamma = 0.5  # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 3  # lavel of wavelet decomposition
width = 96  # uplifting dimension
layers = 4  # no of wavelet layers

h = 40  # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 2  # (a(x), x) for this case

# %%
""" Read data """

# Data is of the shape (number of samples, grid size)
data = np.load(PATH)
x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx

x_data = u_train[:, 0, :]  # N x nx, initial solution
y_data = u_train[:, -2, :]  # N x nx, final solution

x_data = torch.tensor(x_data)
y_data = torch.tensor(y_data)

x_train = x_data[:ntrain, :]
y_train = y_data[:ntrain, :]
x_test = x_data[-ntest:, :]
y_test = y_data[-ntest:, :]

x_train = x_train[:, :, None]
x_test = x_test[:, :, None]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

model = torch.load('model/UWNO_advection_time_independent')
print(model)
print(count_params(model))
model.eval()
myloss = LpLoss(size_average=False)

# %%
pred = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
pred = []
test_e = []
with torch.no_grad():
    index = 0
    for x, y in test_loader:
        # test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x)
        test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_e.append(test_l2 / batch_size)
        pred.append(out)
        print("Batch-{}, Test-loss-{:0.8e}".format(index, test_l2 / batch_size))
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))
print('Mean Error:', 100 * torch.mean(test_e).numpy(), '%')

# %%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16

figure1 = plt.figure(figsize=(12, 6))
plt.subplots_adjust(hspace=0.4)
for i in range(y_test.shape[0]):
    if i % 20 == 1:
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, 1, 40), x_test[i, :].numpy(),linewidth=2)
        plt.title('I.C.')
        plt.xlabel('$\\ x$', fontsize=15)
        plt.ylabel('$\\ u(x,0)$', fontsize=15)
        plt.grid(True)
        plt.xticks()
        plt.yticks()
        plt.margins(0)

        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, 1, 40), y_test[i, :].numpy(),linewidth=2, label='Exact')
        plt.plot(np.linspace(0, 1, 40), pred[i, :].cpu().numpy(), '-.',linewidth=2, label='Prediction')
        plt.title('Solution')
        plt.legend(['Exact', 'Prediction'], ncol=2, loc=3, fontsize=15)
        plt.xlabel('$\\ x$', fontsize=15)
        plt.ylabel('$\\ u(x,1)$', fontsize=15)
        plt.grid(True)
        plt.xticks()
        plt.yticks()
        plt.margins(0)

# plt.savefig('figures/EX_4.6.1_1.pdf',bbox_inches='tight')
plt.show()