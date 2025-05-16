
"""
This code is for 2-D Possion equation.
"""

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


class UWNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(UWNO2d, self).__init__()


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


# %%
""" Model configurations """

PATH = 'data/u_sol_poissons.mat'
# PATH_Test = 'data/piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest = 100

batch_size = 20
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

# %%
""" Read data """
reader = MatReader(PATH)
x_train = reader.read_field('mat_sd')[:ntrain, ::sub, ::sub][:, :h, :h]
y_train = reader.read_field('sol')[:ntrain, ::sub, ::sub][:, :h, :h]

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
""" The model definition """
model = UWNO2d(width=width, level=level, layers=layers, size=[h, h], wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range, padding=1).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
y_normalizer.to(device)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, h, h)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer.step()

        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(batch_size, h, h)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.6e}, Train-L2-{:0.6e}, Test-L2-{:0.6e}"
          .format(ep, t2 - t1, train_mse, train_l2, test_l2))
l2_error = torch.cat((train_loss.unsqueeze(0),test_loss.unsqueeze(0)),dim=0).numpy().T
l2 = pd.DataFrame(l2_error)
l2.to_excel('results/1uwnoPossL2.xlsx')
# %%
""" Prediction """
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

        test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        test_e.append(test_l2 / batch_size)

        print("Batch-{}, Loss-{}".format(index, test_l2 / batch_size))
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy(), '%')

# %%
""" Plotting """
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

figure1 = plt.figure(figsize=(18, 14))
figure1.text(0.04, 0.17, '\n Error', rotation=90, color='purple', fontsize=20)
figure1.text(0.04, 0.34, '\n Prediction', rotation=90, color='green', fontsize=20)
figure1.text(0.04, 0.57, '\n Truth', rotation=90, color='red', fontsize=20)
figure1.text(0.04, 0.75, 'Permeability \n field', rotation=90, color='b', fontsize=20)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(y_test.shape[0]):
    if value % 26 == 1:
        plt.subplot(4, 4, index + 1)
        plt.imshow(x_test[value, :, :, 0], cmap='rainbow', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.title('a(x,y)-{}'.format(index + 1), color='b', fontsize=20, fontweight='bold')
        plt.xlabel('x', fontweight='bold')
        plt.ylabel('y', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        plt.subplot(4, 4, index + 1 + 4)
        plt.imshow(y_test[value, :, :], cmap='rainbow', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x', fontweight='bold')
        plt.ylabel('y', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        plt.subplot(4, 4, index + 1 + 8)
        plt.imshow(pred[value, :, :], cmap='rainbow', extent=[0, 1, 0, 1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x', fontweight='bold')
        plt.ylabel('y', fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        plt.subplot(4, 4, index + 1 + 12)
        plt.imshow(np.abs(pred[value, :, :] - y_test[value, :, :]), cmap='jet', extent=[0, 1, 0, 1],
                   interpolation='Gaussian')
        plt.xlabel('x', fontweight='bold')
        plt.ylabel('y', fontweight='bold')
        plt.colorbar(fraction=0.045, format='%.0e')

        plt.margins(0)
        index = index + 1
plt.savefig( 'figures/EX_2_1.pdf')
plt.show()

# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/UWNO_poss')
scipy.io.savemat('results/uwno_results_poss.mat', mdict={'x_test': x_test.cpu().numpy(),
                                                         'y_test': y_test.cpu().numpy(),
                                                         'pred': pred.cpu().numpy(),
                                                         'test_e': test_e.cpu().numpy()})