import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pandas as pd
from timeit import default_timer

from matplotlib import gridspec

from utils import *
from wavelet_convolution import WaveConv1d

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

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        self.unet = nn.ModuleList()
        self.a = nn.Parameter(torch.FloatTensor([0.1]))

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


""" Model configurations """

PATH = 'data/burgers_data_512_51.mat'
ntest = 100

sub = 1  # subsampling rate
h = 512  # total grid size divided by the subsampling rate
# grid_range = 1
# in_channel = 21  # input channel is 21: (20 for a(x,t1-t20), 1 for x)

T_in = 20  # No of initial temporal-samples
T = 30  # No of prediction steps
step = 1  # Look-ahead step size
batch_size = 1
# %%
""" Read data """
dataloader = MatReader(PATH)
data = dataloader.read_field('sol')  # N x Nx x Nt

x_test = data[-ntest:, ::sub, :T_in]
y_test = data[-ntest:, ::sub, T_in:T_in + T]

x_test = x_test.reshape(ntest, h, T_in)


""" The model definition """
model = torch.load('model/UWNO_burgers_time_dependent')
print(model)
print(count_params(model))
model.eval()
# %%
""" Prediction """
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
myloss = LpLoss(size_average=False)
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

        print("Batch-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, test_l2_step / batch_size / (T / step), test_l2_batch / batch_size))

prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy(), '%')


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 15


figure3 = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(4, 4)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

xtest = torch.cat([x_test, y_test.cpu()], axis=-1)
xpred = torch.cat([x_test, prediction.cpu()], axis=-1)
val = 8

#  Exact
ax1 = figure3.add_subplot(gs[0, :])
exact_image = plt.imshow(xtest.cpu()[val, :, :], extent=[0, 1, -1, 1], cmap='bwr', aspect='auto', origin='lower')
# plt.colorbar(aspect=15, pad=0.015)
plt.title('Exact', fontsize=15, loc='left')
plt.xlabel('$\\ t$')
plt.ylabel('$\\ x$')
plt.axvline(0.4, linewidth=1, color='w')

cbar_ax = figure3.add_axes([0.91, 0.73, 0.005, 0.15])
plt.colorbar(exact_image, cax=cbar_ax)

#  Prediction
ax2 = figure3.add_subplot(gs[1, :])
prediction_image = plt.imshow(xpred.cpu()[val, :, :], extent=[0, 1, -1, 1], cmap='bwr', aspect='auto', origin='lower')
# plt.colorbar(aspect=15, pad=0.015)
plt.title('Prediction', fontsize=15, loc='left')
plt.xlabel('$\\ t$')
plt.ylabel('$\\ x$')
plt.axvline(0.4, linewidth=1, color='w')

cbar_ax = figure3.add_axes([0.91, 0.522, 0.005, 0.15])
plt.colorbar(prediction_image, cax=cbar_ax)


ax3 = figure3.add_subplot(gs[2, :])
error_image = plt.imshow(torch.abs(xtest.cpu()[val, :, :] - xpred.cpu()[val, :, :]),
                         extent=[0, 1, -1, 1], cmap='bwr', aspect='auto', origin='lower', vmax=4e-3)
# plt.colorbar(aspect=15, pad=0.015)
plt.title('Error', fontsize=15, loc='left')
plt.axvline(x=0.4, color='w', linewidth=1)
plt.xlabel('$\\ t$')
plt.ylabel('$\\ x$')

cbar_ax = figure3.add_axes([0.91, 0.315, 0.005, 0.15])
plt.colorbar(error_image, cax=cbar_ax)

slices = [20, 27, 38 , 49]
for i, slice_val in enumerate(slices):
    ax = figure3.add_subplot(gs[3, i])
    plt.plot(xtest.cpu()[val, :, slice_val], 'r',linewidth=3, label='Exact')
    plt.plot(xpred.cpu()[val, :, slice_val], 'k-.', linewidth=3, label='Prediction')
    plt.title('$\\ t = {}s$'.format(0.01 * slice_val * 2), fontsize=15)
    plt.xlabel('$\\ x$')
    plt.ylabel('$\\ u(x,t)$')
    plt.margins(0)

plt.legend(['Exact', 'Prediction'], ncol=2, bbox_to_anchor=(0.5, -0.3))

# plt.savefig('figures/EX_1.2_2.pdf', bbox_inches='tight')
plt.show()
