import os
import sys
import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from model_ol import *
from utility import MinMaxNormalizer
import scipy.io as io
from Adam import Adam
import mat73
import torch.nn.functional as F
import argparse
import sys
from scipy.io import savemat
import statistics

# Construct function f : -32 * pi^2 * sin(4 * pi * x) * sin(4 * pi * y)
# True solution u = sin (4 * pi * x) * sin(4 * pi * y)

# path_git = '/home/jcy02/wang/Poisson_learning/model/GIT/GIT_2500_dpca_200-200_l3_act_gelu_dw512_cw32_lr0.001-500-0.5_nolizTrue_nogrid.model'
# path_fno = '/home/jcy02/wang/Poisson_learning/model/FNO_grid/FNOg_2500_cw32_m12_lr0.001-100-0.5_nolizTrue.model'

import os
import sys
import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from model_ol import *
from utility import MinMaxNormalizer
import scipy.io as io
from Adam import Adam
import mat73
import torch.nn.functional as F
import argparse
import sys
from scipy.io import savemat
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--c_width', type=int, default=32, help='')
parser.add_argument('--d_width', type=int, default=512)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--dim_PCA', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--device', type=int, default=0, help="index of cuda device")
parser.add_argument('--state', type=str, default='eval')
parser.add_argument('--path_model', type=str, default='model/GIT/GIT_2500_dpca_200-200_l3_act_gelu_dw512_cw32_lr0.001-500-0.5_nolizTrue_nogrid.model', help="path of model for testing")
cfg = parser.parse_args()

print(sys.argv)
device = torch.device('cuda:' + str(cfg.device))


# parameters
ntrain = cfg.M
ntest = cfg.M
layer = 4
width = cfg.c_width
d_width = cfg.d_width
batch_size = 64
learning_rate = 0.001
num_epoches = 5000
ep_predict = 10
step_size = 500
gamma = 0.5

# load data
prefix = "/home/jcy02/wang/dataset/Poisson/"
data = mat73.loadmat(prefix + "Poisson_Triangular_2000.mat")

inputs = data['f']
outputs = data['u_field']

inputs = np.transpose(inputs, (1, 2, 0))
outputs = np.transpose(outputs, (1, 2, 0))


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
print('f ',f.shape)
print('u ',u.shape)
print(f[0, 0])
print(f[127, 127])
print(u[0, 0])
print(u[127, 127])


# PCA
train_inputs = np.reshape(inputs[:, :, :ntrain], (-1, ntrain))
input_norm = np.linalg.norm(train_inputs, axis=0)
factor = np.mean(input_norm)
# test_inputs = np.reshape(f/np.linalg.norm(f) * factor, (-1, 1))
test_inputs = np.reshape(f, (-1, 1))
Ui, Si, Vi = np.linalg.svd(train_inputs, full_matrices=False)
en_f = 1 - np.cumsum(Si) / np.sum(Si)
r_f = np.argwhere(en_f < cfg.eps)[0, 0]
if r_f>cfg.dim_PCA:
    r_f = cfg.dim_PCA

Uf = Ui[:, :r_f]
f_hat = np.matmul(Uf.T, train_inputs)
print(f_hat.shape)
print(Ui.shape)
print(Uf.shape)
print(test_inputs.shape)
f_hat_test = np.matmul(Uf.T, test_inputs)
# np.linalg.norm(np.matmul(Uf, f_hat_test) - test_inputs)
x_train = torch.from_numpy(f_hat.T.astype(np.float32))
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))

train_outputs = np.reshape(outputs[:, :, :ntrain], (-1, ntrain))
# test_outputs = np.reshape(u/np.linalg.norm(f) * factor, (-1, 1))
test_outputs = np.reshape(u, (-1, 1))
Uo, So, Vo = np.linalg.svd(train_outputs, full_matrices=False)
en_g = 1 - np.cumsum(So) / np.sum(So)
r_g = np.argwhere(en_g < cfg.eps)[0, 0]
if r_g>cfg.dim_PCA:
    r_g = cfg.dim_PCA
Ug = Uo[:, :r_g]
g_hat = np.matmul(Ug.T, train_outputs)
g_hat_test = np.matmul(Ug.T, test_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32))
y_test = torch.from_numpy(g_hat_test.T.astype(np.float32))
# test_outputs = torch.from_numpy(test_outputs).to(device)

# normalization
x_normalizer = MinMaxNormalizer(x_train, -1, 1, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = MinMaxNormalizer(y_train, -1, 1, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test).to(device)
y_normalizer.cuda()

print("Input #bases : ", r_f, " output #bases : ", r_g)

################################################################################
#      Training and evaluation
################################################################################

model = GIT(r_f, d_width, width, r_g)
string = str(ntrain) + '_dpca_' + str(r_f) + '-' + str(r_g) + '_l' + str(layer) + '_act_gelu' + '_dw' + str(d_width) + '_cw' + str(width) + '_lr' + str(learning_rate) + '-' + str(step_size) + '-' + str(gamma)+ '_noliz' + str(cfg.noliz) + '_nogrid'
# path to save model
if cfg.state=='train':
    path = 'training/GIT/GIT_' + string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)

    path_model = "model/GIT/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)
else:
    path = "predictions/GIT/"
    if not os.path.exists(path):
        os.makedirs(path)
    if (cfg.path_model):
        model_state_dict = torch.load(cfg.path_model, map_location=device)
        model.load_state_dict(model_state_dict)
    else:
        model_state_dict = torch.load('model/GIT/GIT_' + string +  '.model', map_location=device)
        model.load_state_dict(model_state_dict)
    num_epoches = 1
    batch_size = 1

# data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, test_outputs.T),batch_size=batch_size, shuffle=False)

model = model.float()

if torch.cuda.is_available():
    model = model.to(device)

# model loss
criterion = LpLoss(size_average=False)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING
for ep in range(num_epoches):

# Validation
    with torch.no_grad():
        
        
        x = x_test.to(device)
        out = model(x)
        print('test error ', torch.norm(y_test-out)/torch.norm(y_test))
        out = y_normalizer.decode(out).detach().cpu().numpy()
        
        
        # y_test = y_test.detach().cpu().numpy()
        y_test_pred = np.matmul(Ug, out.T)
        norms = np.linalg.norm(test_outputs, axis=1)
        error = test_outputs - y_test_pred
        relative_error = np.linalg.norm(error, axis=1) / norms
        average_relative_error = np.sum(relative_error)

    
        
        print(f"Average Relative Error of original PCA: {ep } {average_relative_error: .6e}")
        
    if cfg.state=='eval':
        output = np.matmul(Ug, output.T).T.reshape(-1, 1)
        yb = y_test_pred.reshape(-1)
        ye = output.reshape(-1)
        a = torch.matmul(ye.T, yb) / torch.matmul(ye.T, ye)
        error = torch.norm(yb - a * ye) / torch.norm(yb)
        savemat('predictions/GIT/GIT_final.mat', {'predictions': ye.detach().cpu().numpy()})
        



