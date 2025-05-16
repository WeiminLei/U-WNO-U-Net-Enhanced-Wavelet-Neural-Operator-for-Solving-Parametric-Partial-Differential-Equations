import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv1d
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0),
        )
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, out_channel):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm1d(out_channel),
        )
        # self.a = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, x):
        x = self.layers(x)
        act = nn.LeakyReLU(0.1, inplace=True)
        return act(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        # self.layer = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
        # kernel_size=1, stride=1)
        self.layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x, feature_map):
        # up = F.interpolate(x, scale_factor=2, mode='nearest')
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
        # self.conv4 = Conv_Block(2*in_channels, in_channels)
        self.up2 = UpSample(2 * in_channels, in_channels)
        self.up3 = UpSample(2 * in_channels, in_channels)
        self.out = nn.Conv1d(2 * in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        R1 = self.down1(x)
        R2 = self.conv2(self.down2(R1))
        R3 = self.conv3(self.down3(R2))
        O2 = self.up1(R3, R2)
        O1 = self.up2(O2, R1)
        O0 = self.up3(O1, x)
        return self.out(O0)

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
            if index != self.layers - 1:
                x = convl(x + r) + wl(x) + unetl(x)
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
ntest = 100
T = 39
step = 1
batch_size = 1
# %%
data = np.load('data/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
x_test = u_test[:ntest, :-1, :]  # N x nx
y_test = u_test[:ntest, 1:, :]  # one step ahead,
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
x_test = x_test.permute(0, 2, 1)
y_test = y_test.permute(0, 2, 1)

# model
model = torch.load('model/UWNO_advection_time_dependent')
print(model)
print(count_params(model))
# model.eval()
myloss = LpLoss(size_average=False)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
prediction = []
test_e = []
with torch.no_grad():
    index = 0
    for xx, yy in test_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        prediction.append(pred)
        test_l2_step = loss.item()
        test_l2_batch = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        test_e.append(test_l2_step / batch_size / (T / step))
        index += 1

        print("Batch-{}, Test-loss-step-{:0.6e}, Test-loss-batch-{:0.6e}".format(
            index, test_l2_step / batch_size / (T / step), test_l2_batch / batch_size))

prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy(), '%')

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15

figure1 = plt.figure(figsize=(18, 14))
figure1.text(0.05, 0.15, '\n Error: $\ |u -\hat{u}|$', rotation=90, color='k', fontsize=15)
figure1.text(0.05, 0.33, '\n Prediction: $\ \hat{u}(x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.05, 0.54, '\n Exact: $\ u (x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.06, 0.75, 'I.C.: $\ u (x,0)$', rotation=90, color='k', fontsize=15)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(y_test.shape[0]):
    if value % 23 == 4 and value != 4:
        print(value)
        plt.subplot(4, 4, index + 1)
        plt.plot(np.linspace(0, 1, 39), x_test[value, 0, :], linewidth=2, color='blue')
        plt.title('IC-{}'.format(index + 1), color='b', fontsize=15)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ u(x,0)$')
        plt.margins(0)
        ax = plt.gca()
        ratio = 0.9
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        plt.subplot(4, 4, index + 1 + 4)
        plt.imshow(y_test[value, :, :], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.xlabel('$\ x$')
        plt.ylabel('$\ t$')
        plt.colorbar(fraction=0.045)

        plt.subplot(4, 4, index + 1 + 8)
        plt.imshow(prediction[value, :, :].cpu(), cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.xlabel('$\ x$')
        plt.ylabel('$\ t$')
        plt.colorbar(fraction=0.045)

        plt.subplot(4, 4, index + 1 + 12)
        plt.imshow(np.abs(y_test[value, :, :] - prediction[value, :, :].cpu().numpy()), cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.xlabel('$\ x$')
        plt.ylabel('$\ t$')
        plt.colorbar(fraction=0.045)

        plt.margins(0)
        index = index + 1
# plt.savefig('figures/EX_4.6.2_1.pdf', bbox_inches='tight')
plt.show()