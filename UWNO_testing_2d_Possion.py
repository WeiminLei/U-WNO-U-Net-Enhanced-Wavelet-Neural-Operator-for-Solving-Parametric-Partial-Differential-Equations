import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv2d

torch.manual_seed(0)
np.random.seed(0)

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
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
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
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
        self.conv2 = Conv_Block(in_channels,in_channels)
        self.down3 = DownSample(in_channels)
        self.conv3 = Conv_Block(in_channels, in_channels)
        self.up1 = UpSample(in_channels, in_channels)
        self.up2 = UpSample(2 * in_channels, in_channels)
        self.up3 = UpSample(2 * in_channels, in_channels)
        self.out = nn.Conv2d(2 * in_channels, in_channels, 3, padding=1)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, in_channels, kernel_size=3,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        R1 = self.down1(x)
        R2 = self.conv2(self.down2(R1))
        R3 = self.conv3(self.down3(R2))
        O2 = self.up1(R3, R2)
        O1 = self.layer(O2)
        O1 = torch.cat((O1, R1), dim=1)
        O0 = self.up3(O1, x)
        return self.out(O0)
# %%
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
            self.w.append(nn.Conv2d(self.width, self.width, 1, stride=1))
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
        for index, (convl, wl, unetl) in enumerate(zip(self.conv, self.w, self.unet)):
            x = convl(x+r) + wl(x) + unetl(x)
            if index != self.layers - 1:  # Final layer has no activation
                x = F.mish(10*self.a*x)  # Shape: Batch * Channel * x * y

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
PATH = 'data/u_sol_poissons.mat'

ntrain = 1000
ntest = 100

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 50  # weight-decay step size
gamma = 0.5  # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 4  # lavel of wavelet decomposition
width = 64  # uplifting dimension
layers = 4  # no of wavelet layers

sub = 5
h = int(((421 - 1) / sub) + 1)  # total grid size divided by the subsampling rate
grid_range = [1, 1]  # The grid boundary in x and y direction
in_channel = 3  # (a(x, y), x, y) for this case

""" Read data """
reader = MatReader(PATH)
x_train = reader.read_field('mat_sd')[:ntrain, ::sub, ::sub][:, :h, :h]
y_train = reader.read_field('sol')[:ntrain, ::sub, ::sub][:, :h, :h]

# reader.load_file(PATH_Test)
x_test = reader.read_field('mat_sd')[-ntest:, ::sub, ::sub][:, :h, :h]
y_test = reader.read_field('sol')[-ntest:, ::sub, ::sub][:, :h, :h]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain, h, h, 1)
x_test = x_test.reshape(ntest, h, h, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)
# %%
model = torch.load('model/UWNO_poss')

model.eval()
print(count_params(model))
print(model)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()
pred = []
test_e = []
with torch.no_grad():
    index = 0
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(batch_size, h, h)
        out = y_normalizer.decode(out)
        pred.append(out.cpu())

        test_l2 += myloss(out, y).item()
        test_e.append(test_l2 / batch_size)

        print("Batch-{}, Loss-{}".format(index, test_l2 / batch_size))
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy(), '%')

""" Plotting """
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

figure1 = plt.figure(figsize=(18, 14))
figure1.text(0.06, 0.15, '\n Error: $|u-\hat{u}|$', rotation=90, color='k', fontsize=15)
figure1.text(0.06, 0.33, '\n Prediction: $\ \hat{u}(x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.06, 0.55, '\n Exact: $\ u(x,y)$', rotation=90, color='k', fontsize=15)
figure1.text(0.07, 0.71, 'Source function: $\ f(x,y)$', rotation=90, color='k', fontsize=15)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(y_test.shape[0]):
    if value % 29 == 9 :
        # print(value)
        plt.subplot(4, 4, index + 1)
        plt.imshow(x_test[value, :, :, 0], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.title('$\ f(x,y)-{}$'.format(index + 1), color='k', fontsize=15)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 4)
        plt.imshow(y_test[value, :, :], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 8)
        plt.imshow(pred[value, :, :], cmap='bwr', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('$\ x$')
        plt.ylabel('$\ y$')
        plt.xticks()
        plt.yticks()

        plt.subplot(4, 4, index + 1 + 12)
        plt.imshow(np.abs(pred[value, :, :] - y_test[value, :, :]), cmap='bwr', extent=[0, 1, 0, 1],vmax=3e-3,
                   interpolation='Gaussian')
        plt.xlabel('$\ x$')
        plt.ylabel('$\ x$')
        plt.colorbar(fraction=0.045, format='%.0e')

        plt.margins(0)
        index = index + 1
plt.show()
# plt.savefig('figures/fig12.pdf', bbox_inches='tight')