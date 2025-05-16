"""
-- This code is for 1-D Burger's equation (time-independent problem).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv1d

torch.manual_seed(0)
np.random.seed(0)

# %%
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


class UWNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(UWNO1d, self).__init__()

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
            x = convl(x + r) + wl(x) + unetl(x)
            if index != self.layers - 1:  # Final layer has no activation
                x = F.mish(10*self.a*x)  # Shape: Batch * Channel * x

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

PATH = 'data/burgers_data_R10.mat'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50  # weight-decay step size
gamma = 0.5  # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 8  # lavel of wavelet decomposition
width = 64  # uplifting dimension
layers = 4  # no of wavelet layers

sub = 2 ** 3  # subsampling rate
h = 2 ** 13 // sub  # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 2  # (a(x), x) for this case

# %%
""" Read data """

dataloader = MatReader(PATH)
x_data = dataloader.read_field('a')[:, ::sub]
y_data = dataloader.read_field('u')[:, ::sub]

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
# %%
""" The model definition """
model = UWNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()  # l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            test_mse += F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    test_mse /= ntest

    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    print('Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.8e}, Train-L2-{:0.8e}, Test-L2-{:0.8e}'
          .format(ep, t2 - t1, train_mse, train_l2, test_l2))
l2_error = torch.cat((train_loss.unsqueeze(0), test_loss.unsqueeze(0)), dim=0).numpy().T
l2 = pd.DataFrame(l2_error)
l2.to_excel('results/uwnoBurgersL2.xlsx')
# %%
""" Prediction """
pred = []
test_e = []
with torch.no_grad():
    index = 0
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x)
        test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_e.append(test_l2 / batch_size)
        pred.append(out)
        print("Batch-{}, Test-loss-{:0.8e}".format(index, test_l2 / batch_size))
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))
print('Mean Error:', 100 * torch.mean(test_e).numpy())

# %%
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

colormap = plt.cm.jet
colors = [colormap(i) for i in np.linspace(0, 1, 5)]

""" Plotting """
figure7 = plt.figure(figsize=(10, 4), dpi=300)
index = 0
for i in range(y_test.shape[0]):
    if i % 20 == 1:
        plt.plot(y_test[i, :].cpu().numpy(), color=colors[index], label='Actual')
        plt.plot(pred[i, :].cpu().numpy(), '--', color=colors[index], label='Prediction')
        index += 1
plt.legend(ncol=5)
plt.grid(True)
plt.margins(0)
# plt.savefig('figures/EX_4_1_1_1.pdf')
plt.show()
# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/UWNO_burgers')
scipy.io.savemat('results/uwno_results_burgers.mat', mdict={'x_test': x_test.cpu().numpy(),
                                                           'y_test': y_test.cpu().numpy(),
                                                           'pred': pred.cpu().numpy(),
                                                           'test_e': test_e.cpu().numpy()})