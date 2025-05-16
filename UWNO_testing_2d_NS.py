import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils import *
from timeit import default_timer
from wavelet_convolution import WaveConv2d

torch.manual_seed(0)
np.random.seed(0)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
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
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x, feature_map):
        #up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(x)
        return torch.cat((out, feature_map), dim=1)


class UNet_2d(nn.Module):
    def __init__(self, in_channels):
        super(UNet_2d, self).__init__()

        self.down1 = DownSample(in_channels)
        self.down2 = DownSample(in_channels)
        self.conv2 = Conv_Block(in_channels, in_channels)
        self.down3 = DownSample(in_channels)
        self.conv3 = Conv_Block(in_channels, in_channels)
        self.up1 = UpSample(in_channels, in_channels)
        self.up2 = UpSample(2 * in_channels, in_channels)
        self.up3 = UpSample(2 * in_channels, in_channels)
        self.out = nn.Conv2d(2 * in_channels, in_channels, 3, padding=1)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(2*in_channels, in_channels, kernel_size=3,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        R1 = self.down1(x)
        R2 = self.conv2(self.down2(R1))
        R3 = self.conv3(self.down3(R2))
        O2 = self.up1(R3, R2)
        O1 = self.up2(O2,R1)
        O0 = self.up3(O1, x)
        return self.out(O0)

""" The forward operation """

class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(WNO2d, self).__init__()

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

        self.fc0 = nn.Linear(self.in_channel, self.width)  # input channel is 3: (a(x, y), x, y)
        for i in range(self.layers):
            self.conv.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
            self.unet.append(UNet_2d(self.width))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)  # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)  # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        r = x
        for index, (convl, wl,unetl) in enumerate(zip(self.conv, self.w, self.unet)):
            x = convl(x+r) + wl(x) +unetl(x)
            if index != self.layers - 1:  # Final layer has no activation
                x = F.gelu(10*self.a*x)  # Shape: Batch * Channel * x * y

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # Shape: Batch * x * y * Channel
        x = F.gelu(self.fc1(x))  # Shape: Batch * x * y * Channel
        x = self.fc2(x)  # Shape: Batch * x * y * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


""" Model configurations """

PATH = 'data/ns_V1e-3_N5000_T50.mat'
ntrain = 1000
ntest = 100

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 50  # weight-decay step size
gamma = 0.5  # weight-decay rate

wavelet = 'db4'  # wavelet basis function
level = 3  # lavel of wavelet decomposition
width = 26  # uplifting dimension
layers = 4  # no of wavelet layers

sub = 1  # subsampling rate
h = 64  # total grid size divided by the subsampling rate
grid_range = [1, 1]
in_channel = 12  # input channel is 12: (10 for a(x,t1-t10), 2 for x)

T_in = 10
T = 10  # No of prediction steps
step = 1  # Look-ahead step size

""" Read data """

reader = MatReader(PATH)
data = reader.read_field('u')
train_a = data[:ntrain, ::sub, ::sub, :T_in]
train_u = data[:ntrain, ::sub, ::sub, T_in:T + T_in]

test_a = data[-ntest:, ::sub, ::sub, :T_in]
test_u = data[-ntest:, ::sub, ::sub, T_in:T + T_in]

train_a = train_a.reshape(ntrain, h, h, T_in)
test_a = test_a.reshape(ntest, h, h, T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=batch_size, shuffle=False)

""" The model definition """
model = torch.load('model/UWNO_navier_stokes2d')
print(model)

print(count_params(model))
myloss = LpLoss(size_average=False)
# model.eval()
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
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        prediction.append(pred.cpu())
        test_l2_step = loss.item()
        test_l2_batch = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        test_e.append(test_l2_step)
        index += 1

        print("Batch-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, test_l2_step / batch_size / (T / step), test_l2_batch/batch_size))

prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy() / batch_size / (T / step), '%')

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

figure1 = plt.figure(figsize=(18, 14))
figure1.text(0.06, 0.15, '\n Error: $\ |\omega -\hat{\omega }|$', rotation=90, color='k', fontsize=15)
figure1.text(0.06, 0.34, '\n Prediction: $\ \hat{\omega }(x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.06, 0.54, '\n Exact: $\ \omega (x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.07, 0.75, 'I.C.: $\ \omega_{0} (x,y)$', rotation=90, color='k', fontsize=15)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(test_u.shape[-1]):
    if value % 3 == 0:
        plt.subplot(4, 4, index + 1)
        plt.imshow(test_a.numpy()[88, :, :, 0], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.title('$\ t={}s$'.format(value + 10), color='k', fontsize=15)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 4)
        plt.imshow(test_u[88, :, :, value], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 8)
        plt.imshow(prediction[88, :, :, value], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 12)
        plt.imshow(np.abs(test_u[88, :, :, value] - prediction[88, :, :, value])/10, cmap='bwr', extent=[0, 1, 0, 1],vmax=8e-3,
                   interpolation='Gaussian')
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.colorbar(fraction=0.045, format='%.0e')

        plt.margins(0)
        index = index + 1
# plt.savefig('figures/EX_4.4_1.pdf')
plt.show()