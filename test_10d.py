import argparse
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
from torch.optim.lr_scheduler import MultiStepLR
from functools import reduce
from siren_pytorch import SirenNet
import math, random,sys
import datetime

def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.__stdout__
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() 
#        self.close()
    def flush(self):
        self.log.flush()  
import torch

import os

################
# Arguments
################
parser = argparse.ArgumentParser(description='Algorithm for AI4S_CG2309')

parser.add_argument('--seed', type=int, default=1234, help='Random initialization.')

#Model Params
parser.add_argument('--N_dim', type=int, default=10, help='Dimension of the problem')
parser.add_argument('--sign', type=int, default=1, help='The sign of the right hand side')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--dim_hidden', type=int, default=32, help='Number of node of hidden layer')
parser.add_argument('--dim_out', type=int, default=3, help='Number of node of output layer')

#train params
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--use_lr_scheduler', type=str, default="True", help='Use a learning rate scheduler')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay Rate of Learning Rate Scheduler')
parser.add_argument('--decay_steps', type=float, default=100, help='Decay Steps of Learning Rate Scheduler')
parser.add_argument('--lambda_ic', type=float, default=500**2, help='Lambda for boundary condition')
parser.add_argument('--lambda_f', type=float, default=1., help='Lambda for residual loss')
parser.add_argument('--max_iter', type=int, default=2000, help='Max iterations')

#sampler_params
parser.add_argument('--N_f', type=int, default=30000, help='Number of collocation points to sample for training')
parser.add_argument('--N_eval', type=int, default=10000, help='Number of collocation points to sample for evaluation')

parser.add_argument('--results_dir', type=str, default="./results/", help='save_base_path')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

args = parser.parse_args()


set_seed(args.seed)

# CUDA support
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu_id}')
else:
    device = torch.device('cpu')

print("Device Initialized.")


#Results dir
# path_base = f"Seed_{args.seed}_layers_{args.num_layers}_dim_hidden_{args.dim_hidden}_dim_out_{args.dim_out}/N_f_{args.N_f}_sign_{args.sign}"

# if not os.path.exists(path_base):
#     os.makedirs(path_base)

N_dim = args.N_dim
# define the network
class Multip(torch.nn.Module):
    def __int__(self):
        super(Multip,self).__init__()
    def forward(self,x_list):
        # x_last_col = torch.split(x,1,-1)
        # out = reduce((lambda x, y: x*y),x_list)
        out = x_list[0]
        for i in range(1,len(x_list)):
            out *= x_list[i]
        # x = torch.stack(x_list,-1)
        # out = torch.prod(x,-1)
        out = torch.sum(out,dim=1,keepdim=True)
        return out

# We define a seperatable PINNs network to investigate the solution

class FunNet(torch.nn.Module):
    def __init__(self, N_dim,num_layers,dim_hidden,dim_out):
        super(FunNet, self).__init__()

        self.N_dim = N_dim
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        print(f"Initializing a default SirenNet for Dim:{self.N_dim}")
        
        # write a for loop to contain the network
        self.mod = torch.nn.ModuleList()
        for i in range(self.N_dim):
            self.dnn_end = SirenNet(
                dim_in = 1,                        # input dimension, ex. 2d coor
                dim_hidden = self.dim_hidden,                  # hidden dimension
                dim_out = self.dim_out,                       # output dimension, ex. rgb value
                num_layers = self.num_layers,                    # number of layers
                final_activation = torch.nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
                w0 = 10,
                w0_initial = 10                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )
            # self.dnn_end = DNN(layers) #.to(device)
            self.dnn_end = self.dnn_end.to(device)
            self.mod.append(self.dnn_end)
        self.last_layer = Multip()
        self.last_layer = self.last_layer.to(device)
        
    def forward(self,x):
        x_out_list = []
        for i in range(self.N_dim):
            x_out_list.append(self.mod[i](x[:,i:i+1]))
        out = self.last_layer(x_out_list)
        return out
    
class PhysicsInformedNN():
    def __init__(self, nu, N_dim = 4,num_layers=2,dim_hidden=32,dim_out=3,sign=1.0):
        
        self.nu = nu
        self.N_dim = N_dim
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.sign = sign
        self.dnn = FunNet(N_dim=self.N_dim,num_layers=self.num_layers,dim_hidden=self.dim_hidden,dim_out=self.dim_out)
        self.dnn = self.dnn.to(device)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(),lr=0.001,betas=(0.9,0.999))
        self.scheduler = MultiStepLR(self.optimizer, milestones=[500, 700, 1000, 1500],gamma=0.5)
        
    def net_u(self, x, t=None):  
        u = self.dnn(x)
        return u
    def net_u_x(self, x,t=None):
        x = x.clone().detach().requires_grad_(True)
        u = self.net_u(x,t)
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph= True,
            create_graph= True
        )[0]
        return u_x
    def laplacian(self, xs, f, create_graph=False, keep_graph=None, return_grad=False):
        xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)
        ys = f(xs_flat.view_as(xs))
#         print(ys.shape)
        (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
        ones = torch.ones_like(ys_g)
        (dy_dxs,) = torch.autograd.grad(ys_g, xs_flat, ones, create_graph=True,retain_graph=True)
        ones2 = torch.ones_like(dy_dxs[...,0])
        lap_ys = [
            torch.autograd.grad(
                dy_dxi, xi, ones2, retain_graph=True, create_graph=create_graph
            )[0]
            for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
        ]
        if not (create_graph if keep_graph is None else keep_graph):
            ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
        result = lap_ys, ys
        if return_grad:
            result += (dy_dxs.detach().view_as(xs),)
        return result
    
    def net_f(self, x, t=None):
        """ The pytorch autograd version of calculating residual """
        u_pp, u, u_p = self.laplacian(x,self.net_u,create_graph=True, return_grad=True)
        if t is not None:
            u_t = torch.autograd.grad(
                u, t, 
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
            )[0]

        nabla_t = u_pp[0][:,None]
        for i in range(1,self.N_dim):
            nabla_t += u_pp[i][:,None]

        # define the rhs term
        rhst = torch.prod(torch.sin(self.nu*math.pi*x),dim=1,keepdim=True)
        f = nabla_t  - self.sign*self.nu**2*self.N_dim*math.pi**2*rhst
        return f
    
            
    def predict(self, X):
        X = torch.abs(X)-0.5*torch.floor(2*torch.abs(X))
        x = X.clone().detach().requires_grad_(True)
        # x = torch.tensor(X, requires_grad=True).float().to(device)
        self.dnn.eval()
        u = self.net_u(x)
        f = self.net_f(x)
        u = u.detach()
        f = f.detach()
        return u, f

    def predict_u(self, X):
        X = torch.abs(X)-0.5*torch.floor(2*torch.abs(X))
        x = X.clone().detach().requires_grad_(True)
        self.dnn.eval()
        u = self.net_u(x)
        u = u.detach()
        return u

nu = 4

# Test path
#Results dir
path_base = f"{args.results_dir}/DIM{args.N_dim}"

if not os.path.exists(path_base):
    os.makedirs(path_base)

# Fix the test collection points
import re
# do the evaluate
left_end = 0
right_end = 1.0
N_eval = args.N_eval
# Doman bounds
lb = np.array(N_dim*[left_end]) #X_star.min(0)
ub = np.array(N_dim*[right_end]) #X_star.max(0)  
X_f_eval = lb + (ub-lb)*lhs(N_dim, N_eval)
X_star = torch.tensor(X_f_eval).float().to(device)
data = -torch.prod(torch.sin(nu*math.pi*X_star),dim=1,keepdim=True)
u_star = data.flatten()[:,None]   

path = './checkpoints/DIM10/Seed_1234_layers_4_dim_hidden_32_dim_out_5.pth'

res = re.search(r"_layers_(\d)_dim_hidden_(\d+)_dim_out_(\d)",path)
Out = [int(x) for x in [res.group(i) for i in range(1,4)]]

# generate the model
model = PhysicsInformedNN(nu, N_dim = args.N_dim,num_layers=Out[0],dim_hidden=Out[1],dim_out=Out[2],sign=args.sign)
model.dnn.load_state_dict(torch.load(path))
model.dnn.eval()

# 
# num_params = sum(param.numel() for param in model.dnn.parameters())
# print(num_params)



# with torch.no_grad():
u_pred, f_pred = model.predict(X_star)

error_u = torch.linalg.norm(u_star-u_pred,2)/torch.linalg.norm(u_star,2)
# print('Error u: %e' % (error_u))                     
Error = torch.abs(data.flatten()[:,None] - u_pred)
sys.stdout = Logger(path_base+'/results.txt')
print(path)
print(
    'Test_error (Rand. 30000): %.5e' % (error_u.item())
)

# in the following, we do the visulaization
# EVALUATION code

left_end = 0
right_end = 1.0
N_part = 4        
input_samp = [torch.linspace(left_end,right_end,N_part).to(device) for x in range(N_dim)]
input_ind = torch.meshgrid(*input_samp)
X_star_4 = torch.stack([ele.flatten() for ele in input_ind],-1)
data_4 = -torch.prod(torch.sin(nu*math.pi*X_star_4),dim=1,keepdim=True)
u_star_4 = data_4.flatten()[:,None] 

with torch.no_grad():
        u_pred_4 = model.predict_u(X_star_4)

error_u = torch.linalg.norm(u_star_4-u_pred_4,2)/torch.linalg.norm(u_star_4,2)
# print('Error u: %e' % (error_u))                     
Error = torch.abs(data.flatten()[:,None] - u_pred)
sys.stdout = Logger(path_base+'/results.txt')
print(path)
print(
    'Test_error (Meshgrid 4^10): %.5e' % (error_u.item())
)

# EVALUATION code
left_end = 0
right_end = 1.0
N_part = 100        
input_samp = [torch.linspace(left_end,right_end,N_part).to(device) for x in range(2)]
input_ind = torch.meshgrid(*input_samp)

# Sampling for indices
ind_sub = [0,1]
ori_list = N_dim*[1.0/8*torch.ones_like(input_ind[0])]
for i, idx in  enumerate(ind_sub):
    ori_list[idx] = input_ind[i]
X_star_2 = torch.stack([ele.flatten() for ele in ori_list],-1)
data_2 = reduce((lambda x,y: torch.sin(nu*math.pi*x)*torch.sin(nu*math.pi*y)),input_ind)
# data = data/(N_dim*math.pi**2*nu**2)
data_2 = -1*data_2
u_star_2 = data_2.flatten()[:,None]   

u_pred_2 = model.predict_u(X_star_2)

error_u = torch.linalg.norm(u_star_2-u_pred_2,2)/torch.linalg.norm(u_star_2,2)
sys.stdout = Logger(path_base+'/results.txt')
print('Error u: %e' % (error_u))                     

# U_pred = griddata(X_star, u_pred.flatten(), input_ind, method='cubic')
Error_2 = torch.abs(data_2.flatten()[:,None] - u_pred_2)

""" The aesthetic setting has changed. """
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

####### Row 0: u(x) ##################    
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(Error_2.cpu().numpy().reshape(data_2.shape), interpolation='nearest', cmap='rainbow', 
            extent=[0.0, 1.0, 0.0, 1.0], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

# ax.plot(
#     X_u_train[:,1], 
#     X_u_train[:,0], 
#     'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
#     markersize = 4,  # marker size doubled
#     clip_on = False,
#     alpha=1.0
# )

line = np.linspace(0.0, 1.0, 2)[:,None]
ax.plot(0.1*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.3*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.8*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$x_1$', size=20)
ax.set_ylabel('$x_2$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

# plt.show()
path_save_base = path_base

if not os.path.exists(path_save_base):
    os.makedirs(path_save_base)

plt.savefig('{}.png'.format(os.path.join(path_save_base,'out_err')), bbox_inches='tight', pad_inches=0)
plt.clf()

####### Row 0: u(x) ##################    
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(u_pred_2.cpu().numpy().reshape(data_2.shape), interpolation='nearest', cmap='rainbow', 
            extent=[0.0, 1.0, 0.0, 1.0], 
            origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15) 

# ax.plot(
#     X_u_train[:,1], 
#     X_u_train[:,0], 
#     'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
#     markersize = 4,  # marker size doubled
#     clip_on = False,
#     alpha=1.0
# )

line = np.linspace(0.0, 1.0, 2)[:,None]
ax.plot(0.1*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.3*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.8*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$x_1$', size=20)
ax.set_ylabel('$x_2$', size=20)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.9, -0.05), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)
ax.set_title('$u(x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

# plt.show()
path_save_base = path_base

if not os.path.exists(path_save_base):
    os.makedirs(path_save_base)

plt.savefig('{}.png'.format(os.path.join(path_save_base,'out')), bbox_inches='tight', pad_inches=0)
plt.clf()


    ####### Row 1: u(t,x) slices ################## 

""" The aesthetic setting has changed. """
x = input_samp[0].cpu().numpy()
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,data_2.cpu().numpy()[15,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,u_pred_2.cpu().numpy().reshape(data_2.shape)[15,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x_2$')
ax.set_ylabel('$u(x)$')  
ax.set_title('$x_1=0.1$', fontsize = 15)
# ax.axis('square')
ax.set_xlim([-0.1,1.1])
# ax.set_ylim([-0.1,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x,data_2.cpu().numpy()[30,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,u_pred_2.cpu().numpy().reshape(data_2.shape)[30,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x_2$')
ax.set_ylabel('$u(x)$')
# ax.axis('square')
ax.set_xlim([-0.1,1.1])
# ax.set_ylim([-0.1,1.1])
ax.set_title('$x_1 = 0.3$', fontsize = 15)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=5, 
    frameon=False, 
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,data_2.cpu().numpy()[79,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,u_pred_2.cpu().numpy().reshape(data_2.shape)[79,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x_2$')
ax.set_ylabel('$u(x)$')
# ax.axis('square')
ax.set_xlim([-0.1,1.1])
# ax.set_ylim([-1.1,1.1])    
ax.set_title('$x_1 = 0.8$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

# plt.show()
plt.savefig('{}.png'.format(os.path.join(path_save_base,'out_profile')), bbox_inches='tight', pad_inches=0)
plt.clf()



                