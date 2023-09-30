"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import os
import sys
from model_ol import *
from utility import UnitGaussianNormalizer
from Adam import Adam
from tensorboardX import SummaryWriter
import mat73
import statistics
import numpy as np
import torch
from timeit import default_timer
import argparse
from scipy.io import savemat

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12, help='')
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--M',  type=int, default=500, help="number of dataset")
parser.add_argument('--device', type=int, default=1, help="index of cuda device")
parser.add_argument('--state', type=str, default='eval', help="evaluation ot training")
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--mode_path', type=str, default='')
parser.add_argument('--path_model', type=str, default='model/FNO_grid/FNOg_2500_cw32_m12_lr0.001-100-0.5_nolizTrue.model', help="path of model for testing")
cfg = parser.parse_args()

print(sys.argv)
device = torch.device('cuda:' + str(cfg.device))

batch_size = 32
M = cfg.M
width = cfg.width
modes = cfg.modes
N = 256
ntrain = M
ntest = 1
s = N
learning_rate = 0.001
epochs = 1000
step_size = 100
gamma = 0.5

prefix = "/home/jcy02/wang/dataset/Poisson/"
data = mat73.loadmat(prefix + "Poisson_Triangular_2000.mat")

inputs = data['f']
outputs = data['u_field']


# function to evaluate
N = 256
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
gridx = x.reshape(1, N).repeat(N, axis=0)
gridy = y.reshape(N, 1).repeat(N, axis=1)
f = -32 * np.pi**2 * np.multiply(np.sin(4 * np.pi * gridx),  np.sin(4 * np.pi * gridy))
u = -np.multiply(np.sin(4 * np.pi * gridx), np.sin(4 * np.pi * gridy))
u = u / 1e6
f = f/1e6

x_train = torch.from_numpy(np.reshape(inputs[:ntrain, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(outputs[:ntrain, :, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(f, -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(u, -1).astype(np.float32))

x_normalizer = UnitGaussianNormalizer(x_train, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)

x_train = x_train.reshape(ntrain,s,s,1)
y_train = y_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)
y_test = y_test.reshape(ntest,s,s,1)


################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).to(device)
string =  str(ntrain) + "_cw" + str(width) + "_m" + str(modes) + "_lr" + str(learning_rate) + "-" + str(step_size) + "-" + str(gamma) +  '_noliz' + str(cfg.noliz)

if cfg.state=='train':
    path = "training/FNO_grid/FNOg_"+string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)

    path_model = "model/FNO_grid/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)

else: # eval
    path = "predictions/FNO_grid/"
    if not os.path.exists(path):
        os.makedirs(path)
    if (cfg.path_model):
        model_state_dict = torch.load(cfg.path_model, map_location=device)
        model.load_state_dict(model_state_dict)
    else:
        model_state_dict = torch.load("model/FNO_grid/FNOg_" + string + ".model", map_location=device)
        model.load_state_dict(model_state_dict)
    epochs = 1
    batch_size = 1
    predict = torch.zeros(y_test.reshape(ntest, -1).shape).to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
x_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    if cfg.state=='train':
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x).reshape(batch_size_, -1)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_,-1), y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        train_l2/= ntrain

        writer.add_scalar("train/error", train_l2, ep)

    average_relative_error = 0
    error_list = []
    ite = -1
    with torch.no_grad():
        for x, y, in test_loader:
            ite += 1
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            out = model(x).reshape(batch_size_, -1)
            out = y_normalizer.decode(out)
            # predict[ite:ite+1, :] = out.clone()
            y = y_normalizer.decode(y)
            if ep % 10 == 0:
                y = y.reshape(batch_size_, -1)
                norms = torch.norm(y, dim=1)
                error = y - out
                relative_error = torch.norm(error, dim=1) / norms
                if cfg.state == 'eval':
                    error_list.append(relative_error.item())
                average_relative_error += torch.sum(relative_error)
    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        print(f"Average Relative Test Error : {ep }{average_relative_error: .6e} ")

    if cfg.state=='eval':
        yb = y.reshape(-1)
        ye = out.reshape(-1)
        a = torch.matmul(ye.T, yb) / torch.matmul(ye.T, ye)
        error = torch.norm(yb - a * ye)/torch.norm(yb)
        savemat('predictions/FNO_grid/FNO_final.mat', {'predictions': ye.detach().cpu().numpy()})

if cfg.state=='train':
    torch.save(model.state_dict(), 'model/FNO_grid/FNOg_' + string + '.model')