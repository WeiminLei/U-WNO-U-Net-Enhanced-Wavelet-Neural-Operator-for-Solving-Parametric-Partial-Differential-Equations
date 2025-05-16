
"""
This code is for 1-D wave advection equation (time-dependent problem).
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


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout(0),
        )

    def forward(self, x):
        return self.layers(x)


class DownSample(nn.Module):
    def __init__(self, out_channel):
        super(DownSample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm1d(out_channel),
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
                x = convl(x+r) + wl(x) + unetl(x)
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

""" Model configurations """
PATH_train = 'data/train_IC2.npz'
PATH_test = 'data/test_IC2.npz'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50  # weight-decay step size
gamma = 0.5  # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 3  # lavel of wavelet decomposition
width = 80  # uplifting dimension
layers = 4  # no of wavelet layers

sub = 1  # subsampling rate
h = 40  # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 40  # input channel is 21: (20 for a(x,t1-t20), 1 for x)

T = 39  # No of prediction steps
step = 1  # Look-ahead step size


""" Read data """

data = np.load(PATH_train)
x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
x_train = u_train[:ntrain, :-1, :]  # N x nx
y_train = u_train[:ntrain, 1:, :]  # one step ahead,
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = x_train.permute(0, 2, 1)
y_train = y_train.permute(0, 2, 1)

data = np.load(PATH_test)
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
x_test = u_test[:ntest, :-1, :]  # N x nx
y_test = u_test[:ntest, 1:, :]  # one step ahead,
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
x_test = x_test.permute(0, 2, 1)
y_test = y_test.permute(0, 2, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)


""" The model definition """
model = UWNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_batch = 0
    for xx, yy in train_loader:
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

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_batch += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_batch = 0
    with torch.no_grad():
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

            test_l2_step += loss.item()
            test_l2_batch += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_loss[ep] = train_l2_step / ntrain / (T / step)
    test_loss[ep] = test_l2_step / ntest / (T / step)

    t2 = default_timer()
    scheduler.step()
    print(
        'Epoch-{}, Time-{:0.4f}, Train-L2-Batch-{:0.6e}, Train-L2-Step-{:0.6e}, Test-L2-Batch-{:0.6e}, Test-L2-Step-{:0.6e}'
        .format(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_batch / ntrain,
                test_l2_step / ntest / (T / step),
                test_l2_batch / ntest))
l2_error = torch.cat((train_loss.unsqueeze(0),test_loss.unsqueeze(0)),dim=0).numpy().T
l2 = pd.DataFrame(l2_error)
l2.to_excel('results/uwnoAdvectionTimeL2.xlsx')

""" Prediction """
prediction = []
test_e = []
with torch.no_grad():
    index = 0
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_batch = 0
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
        test_l2_batch = myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()

        test_e.append(test_l2_step / len(test_loader) / (T / step))
        index += 1

        print("Batch-{}, Test-loss-step-{:0.6e}, Test-loss-batch-{:0.6e}".format(
            index, test_l2_step / len(test_loader) / (T / step), test_l2_batch / len(test_loader)))

prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))
print('Mean Testing Error:', 100 * torch.mean(test_e).numpy(), '%')


""" Plotting """
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

xtest = torch.cat([x_test, y_test.cpu()], axis=-1)
xpred = torch.cat([x_test, prediction.cpu()], axis=-1)

sample = 16
""" Solution and colocation points """
fig = plt.figure(figsize=(12, 5), dpi=100)
plt.subplots_adjust(wspace=0.3)
plt.subplot(1, 2, 1)
plt.imshow(xtest[sample, ...].cpu().numpy(), interpolation='nearest', cmap='rainbow',
           extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(aspect=15, pad=0.015)
plt.title('Ground Truth', fontsize=20)  # font size doubled
plt.axvline(x=0.25, color='w', linewidth=1)
plt.axvline(x=0.50, color='w', linewidth=1)
plt.axvline(x=0.75, color='w', linewidth=1)
plt.xlabel(r'$t$', size=12)
plt.ylabel(r'$x$', size=12)

plt.subplot(1, 2, 2)
plt.imshow(xpred[sample, ...].cpu().numpy(), interpolation='nearest', cmap='rainbow',
           extent=[0, 1, -1, 1], origin='lower', aspect='auto')
plt.colorbar(aspect=15, pad=0.015)
plt.title('Prediction', fontsize=20)  # font size doubled
plt.axvline(x=0.25, color='w', linewidth=1)
plt.axvline(x=0.50, color='w', linewidth=1)
plt.axvline(x=0.75, color='w', linewidth=1)
plt.xlabel(r'$t$', size=12)
plt.ylabel(r'$x$', size=12)
plt.savefig( 'figure/EX_4_6_2_1.png')
plt.show()


""" Solution at slices """
fig = plt.figure(figsize=(14, 5), dpi=100)
fig.subplots_adjust(wspace=0.4)
slices = [12, 25, 38]
x = torch.linspace(-1, 1, h)

sample = 16
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(x, xtest[sample, :, slices[i]], 'b-', linewidth=2, label='Exact')
    plt.plot(x, xpred[sample, :, slices[i]], 'r--', linewidth=2, label='Prediction')
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$')
    plt.title('$t = {}$'.format(0.01 * slices[i] * 2), fontsize=15)
    plt.axis('square')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.25)
    if i == 1:
        plt.legend(frameon=False, ncol=2, bbox_to_anchor=(1, -0.15))
# plt.savefig( 'figure/EX_4_6_2_2.pdf')

plt.show()


"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/UWNO_advection_time_dependent')
scipy.io.savemat('results/uwno_results_advection_time_dependent.mat', mdict={'x_test': x_test.cpu().numpy(),
                                                                            'y_test': y_test.cpu().numpy(),
                                                                            'pred': pred.cpu().numpy(),
                                                                            'test_e': test_e.cpu().numpy()})