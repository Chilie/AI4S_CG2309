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
parser.add_argument('--N_dim', type=int, default=2, help='Dimension of the problem')
parser.add_argument('--sign', type=int, default=1, help='The sign of the right hand side')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--dim_hidden', type=int, default=32, help='Number of node of hidden layer')
parser.add_argument('--dim_out', type=int, default=3, help='Number of node of output layer')

#train params
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--use_lr_scheduler', type=str, default="True", help='Use a learning rate scheduler')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay Rate of Learning Rate Scheduler')
parser.add_argument('--decay_steps', type=float, default=100, help='Decay Steps of Learning Rate Scheduler')
parser.add_argument('--lambda_ic', type=float, default=500**2, help='Lambda for boundary condition')
parser.add_argument('--lambda_f', type=float, default=1., help='Lambda for residual loss')
parser.add_argument('--max_iter', type=int, default=20000, help='Max iterations')

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
path_base = f"./exp/DIM2_Seed_{args.seed}_layers_{args.num_layers}_dim_hidden_{args.dim_hidden}_dim_out_{args.dim_out}/N_f_{args.N_f}_sign_{args.sign}"

if not os.path.exists(path_base):
    os.makedirs(path_base)

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

nu = 4
left_end = 0
right_end = 0.5
N_f = args.N_f

# Doman bounds
lb = np.array(N_dim*[left_end]) 
ub = np.array(N_dim*[right_end])    

X_f_train = lb + (ub-lb)*lhs(N_dim, N_f)
X_f = torch.tensor(X_f_train, requires_grad=True).float().to(device)

N_f_res = int(N_f/(2*N_dim))
# Doman bounds
lb_res = np.array((N_dim-1)*[left_end]) 
ub_res = np.array((N_dim-1)*[right_end])    

X_u_train_res = lb_res + (ub_res-lb_res)*lhs((N_dim-1), N_f_res)
for idd in range(N_dim):
    z1= np.hstack((X_u_train_res[:,:idd], left_end*np.ones_like(X_u_train_res[:,0:1]), X_u_train_res[:,idd:]))
    z2 = np.hstack((X_u_train_res[:,:idd], right_end*np.ones_like(X_u_train_res[:,0:1]), X_u_train_res[:,idd:]))
    z_local = np.vstack([z1,z2]) # the boundary
    if idd == 0:
        X_u_train = z_local
    else:
        X_u_train = np.vstack([X_u_train,z_local])
X_u = torch.tensor(X_u_train, requires_grad=True).float().to(device)
gamma = torch.tensor(0.0).to(device)
lamb = args.lambda_ic
for iter in range(2000):
    if iter == 0:
        model = PhysicsInformedNN(nu, N_dim = args.N_dim,num_layers=args.num_layers,dim_hidden=args.dim_hidden,dim_out=args.dim_out,sign=args.sign)

    model.dnn.train()

    model.scheduler.step(iter)
    model.optimizer.zero_grad()
    f_pred = model.net_f(X_f)
    u_pred = model.net_u(X_u)
    
    loss_f = torch.mean(f_pred ** 2)
    loss_b = torch.mean(lamb*u_pred**2)
        
    loss = loss_f + loss_b 

    loss.backward()

    model.optimizer.step()
    # Add the sampling
    #Evolutionary ReSampling
    if iter >0 and iter %1 ==0: # 10
        with torch.no_grad():
            weit_f = torch.norm(X_f-1/4,dim=0,keepdim=True)
            gate_f = 1-torch.sigmoid(10*(weit_f-gamma))
            fitness = torch.abs(f_pred)*gate_f
            mask = fitness > fitness.mean()
            X_f_old = X_f[mask.squeeze()]
            X_f_new = lb + (ub-lb)*lhs(N_dim, N_f - torch.sum(mask))
            X_f_new = torch.tensor(X_f_new).float().to(device)
            X_f = torch.cat((X_f_old, X_f_new), dim=0)
            # X_f = torch.tensor(X_f, requires_grad=True).float().to(device)
            X_f = X_f.to(device).clone().detach().requires_grad_(True)

            fitness = torch.abs(u_pred)
            mask = fitness <= fitness.mean()

            N_f_res = int(torch.sum(mask)/(2*N_dim))+1
            # Doman bounds
            lb_res = np.array((N_dim-1)*[left_end]) 
            ub_res = np.array((N_dim-1)*[right_end])    

            X_u_train_res = lb_res + (ub_res-lb_res)*lhs((N_dim-1), N_f_res)
            for idd in range(N_dim):
                z1= np.hstack((X_u_train_res[:,:idd], left_end*np.ones_like(X_u_train_res[:,0:1]), X_u_train_res[:,idd:]))
                z2 = np.hstack((X_u_train_res[:,:idd], right_end*np.ones_like(X_u_train_res[:,0:1]), X_u_train_res[:,idd:]))
                z_local = np.vstack([z1,z2]) # the boundary
                if idd == 0:
                    X_u_new = z_local
                else:
                    X_u_new = np.vstack([X_u_new,z_local])
            X_u_new = torch.tensor(X_u_new).float().to(device)
            mask = fitness > fitness.mean()
            X_u_old = X_u[mask.squeeze()]
            X_u_train = torch.cat((X_u_old, X_u_new), dim=0)[:N_f,:]
            # X_u = torch.tensor(X_u_train, requires_grad=True).float().to(device)
            X_u = X_u_train.to(device).clone().detach().requires_grad_(True)
            
            # update gamma
            loss_f_g = torch.mean(f_pred ** 2*weit_f)
            gradient = torch.exp(-20.0 * loss_f_g)
            gradient = gradient if gradient <= 1e-1 else 1e-1
            gamma += 1e-4 * gradient
    if iter >0 and iter % 1 == 0:
        sys.stdout = Logger(path_base+'/results.txt')
        print(
            'Time:-',datetime.datetime.now(),'Iter %d, Loss: %.5e, Loss_f: %.5e, Loss_b: %.5e' % (iter, loss.item(), loss_f.item(),loss_b.item())
        )

    if iter >0 and (iter+1) % 200 == 0:
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

        u_pred, f_pred = model.predict(X_star)

        error_u = torch.linalg.norm(u_star-u_pred,2)/torch.linalg.norm(u_star,2)
        # print('Error u: %e' % (error_u))                     
        Error = torch.abs(data.flatten()[:,None] - u_pred)
        sys.stdout = Logger(path_base+'/results.txt')
        print(
            'Iter %d, Test_error: %.5e' % (iter, error_u.item())
        )

        
    

        """ The aesthetic setting has changed. """
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)

        ####### Row 0: u(x) ##################    
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)

        h = ax.imshow(Error.cpu().numpy().reshape(data.shape), interpolation='nearest', cmap='rainbow', 
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

        ax.set_xlabel('$x$_1', size=20)
        ax.set_ylabel('$x$_2', size=20)
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

        plt.savefig('{}.png'.format(os.path.join(path_base,'out_iter%d' %(iter,))), bbox_inches='tight', pad_inches=0)
        plt.clf()
    if iter >0 and (iter+1) % 200 ==0:
        # save the model
        save_path = os.path.join(path_base,'step_%d.pth'%(iter,))
        torch.save(model.dnn.state_dict(),save_path)
            