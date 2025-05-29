import time
import sys
import torch

from torch import nn, Tensor

# sys.path.append('./')
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler, LinearVPScheduler, VPScheduler, CosineScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper, gradient
from torch.distributions.multivariate_normal import MultivariateNormal
# visualization
import matplotlib.pyplot as plt
from matplotlib import cm

import pytorch_warmup as warmup

# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):

    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
    
    return data.float()

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )
    

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


def train(lr = 0.001, 
          batch_size = 128,
          iterations = 101,
          print_every = 20,
          hidden_dim = 512, 
          step_size=0.01,
          reg=1, 
          seed=42,
          freeze=True,
          path="OT"):

    # velocity field model init
    torch.manual_seed(seed)
    vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device) 

    # instantiate an affine path object
    if path == "OT":
        path = AffineProbPath(scheduler=CondOTScheduler())
    elif path == "VP":
        path = AffineProbPath(scheduler=LinearVPScheduler())
    else:
        raise ValueError("Invalid path type")

    # init optimizer
    optim = torch.optim.Adam(vf.parameters(), lr=lr) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.999)
    warmup_scheduler = warmup.LinearWarmup(optimizer=optim, warmup_period=10)
    # train
    start_time = time.time()
    loss_lst = []
    loss_diff_lst = []
    loss_der_lst = []
    best_ll = -1e10
    best_det = -1e10
    for i in range(iterations):
        optim.zero_grad() 

        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        x_1 = inf_train_gen(batch_size=batch_size, device=device) # sample data
        x_0 = torch.randn_like(x_1).to(device)

        # sample time (user's responsibility)
        t = torch.rand(x_1.shape[0]).to(device)

        # sample probability path
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        # flow matching l2 loss
        x_tilde = path_sample.x_t
        x_tilde.requires_grad_(True)
        vec_tilde = vf(x_tilde, path_sample.t)
        loss = torch.pow(vec_tilde - path_sample.dx_t, 2).mean() 
        z = (torch.randn_like(x_tilde).to(x_tilde.device) < 0) * 2.0 - 1.0
        d_log_p_t = path_sample.d_log_p_t
        u_t = path_sample.u_t
        d_u_t = path_sample.d_u_t
        sigma_t = path_sample.sigma_t

        vec_tilde_dot_z = torch.einsum("ij,ij->i", vec_tilde.flatten(start_dim=1), z.flatten(start_dim=1))
        grad_vec_tilde_dot_z = gradient(vec_tilde_dot_z, x_tilde, create_graph=True)
        
        if freeze:
            right_vec = d_log_p_t * vec_tilde.detach() - u_t * d_log_p_t - d_u_t
        else:
            right_vec = d_log_p_t * vec_tilde - u_t * d_log_p_t - d_u_t
        grad_vec_tilde = gradient(vec_tilde, x_tilde, create_graph=True, grad_outputs=torch.ones_like(x_tilde))
        eps_diff = (grad_vec_tilde + right_vec) * sigma_t
        

        # eps_diff = (grad_vec_tilde_dot_z + right_vec*z) * sigma_t
        loss_diff = torch.pow(eps_diff, 2).mean()

        div = torch.einsum("ij,ij->i", grad_vec_tilde_dot_z.flatten(start_dim=1), z.flatten(start_dim=1))
        if freeze:
            right_tr = torch.einsum("ij,ij->i", d_log_p_t.flatten(start_dim=1), z.flatten(start_dim=1)) *\
                        torch.einsum("ij,ij->i", vec_tilde.detach().flatten(start_dim=1), z.flatten(start_dim=1)) - \
                        torch.einsum("ij,ij->i", u_t.flatten(start_dim=1), z.flatten(start_dim=1))*\
                        torch.einsum("ij,ij->i", d_log_p_t.flatten(start_dim=1), z.flatten(start_dim=1)) -\
                        torch.einsum("ij,ij->i", d_u_t.flatten(start_dim=1), torch.pow(z, 2).flatten(start_dim=1))
        else:
            right_tr = torch.einsum("ij,ij->i", d_log_p_t.flatten(start_dim=1), z.flatten(start_dim=1)) *\
                        torch.einsum("ij,ij->i", vec_tilde.flatten(start_dim=1), z.flatten(start_dim=1)) - \
                        torch.einsum("ij,ij->i", u_t.flatten(start_dim=1), z.flatten(start_dim=1))*\
                        torch.einsum("ij,ij->i", d_log_p_t.flatten(start_dim=1), z.flatten(start_dim=1)) -\
                        torch.einsum("ij,ij->i", d_u_t.flatten(start_dim=1), torch.pow(z, 2).flatten(start_dim=1))
        
        eps_der = (div + right_tr) * sigma_t.mean(dim=1)   
        # eps_der2 = torch.einsum("ij,ij->i", (vec_tilde - path_sample.dx_t).flatten(start_dim=1), torch.pow(z, 2).flatten(start_dim=1))
        eps_der2 = (vec_tilde - path_sample.dx_t) * x_1.shape[-1]
        loss_der = torch.pow(eps_der, 2).mean()  
        loss_der2 = torch.pow(eps_der2, 2).mean()
        loss_der += loss_der2
        # loss_diff = torch.zeros_like(loss_der)     

        ## regularize eps_diff
        loss_sum = loss + reg * (loss_diff+loss_der)
        # loss_sum = loss + reg * loss_der 
        loss_lst.append(loss.item())
        loss_diff_lst.append(loss_diff.item())
        loss_der_lst.append(loss_der.item())

        # optimizer step
        loss_sum.backward() # backward
        optim.step() # update
        
        with warmup_scheduler.dampening():
            lr_scheduler.step()

        # log loss
        if (i+1) % print_every == 0:
            elapsed = time.time() - start_time
            loss_mean = sum(loss_lst)/len(loss_lst)
            loss_diff_mean = sum(loss_diff_lst)/len(loss_diff_lst)
            loss_der_mean = sum(loss_der_lst)/len(loss_der_lst)
            lr = optim.param_groups[0]['lr']

            # sample with likelihood
            T = torch.tensor([1., 0.])  # sample times
            T = T.to(device=device)

            # source distribution is a gaussian
            gaussian_log_density = MultivariateNormal(torch.zeros(2, device=device), torch.eye(2, device=device)).log_prob
            wrapped_vf = WrappedModel(vf)
            solver = ODESolver(velocity_model = wrapped_vf)
            # compute log likelihood with unbiased hutchinson estimator, average over num_acc
            num_acc = 10
            log_p_acc = 0
            log_det_acc = 0
            source_log_p_acc = 0
            for _ in range(num_acc):
                x_1 = inf_train_gen(batch_size=batch_size, device=device)
                _, log_p, log_det = solver.compute_likelihood(x_1=x_1, method='euler', 
                                                              step_size=step_size, time_grid=torch.tensor([1.0, 0.0]), exact_divergence=False, 
                                                              log_p0=gaussian_log_density)
                log_det_acc += log_det
                source_log_p = log_p - log_det
                source_log_p_acc += source_log_p
                log_p_acc += log_p
            log_p_acc /= num_acc
            log_det_acc /= num_acc
            source_log_p_acc /= num_acc
            # log_p_acc = torch.exp(log_p_acc).mean().item()
            log_p_acc = log_p_acc.mean().item()
            source_log_p_acc = source_log_p_acc.mean().item()
            log_det_acc = log_det_acc.mean().item()
            
            if best_ll < log_p_acc:
                best_ll = log_p_acc
                best_det = log_det_acc
                if reg > 0:
                    torch.save(vf.state_dict(), 'vf_2d_best_reg_{}.pth'.format(int(reg*1000)))
                    print('save model at iter {}'.format(i+1))

            print('| iter {:6d} | {:5.2f} ms/step | lr {:8.6f} | loss {:8.3f} | loss diff {:8.3f} | loss der {:8.3f} | ll {:8.3f} '.format(i+1, elapsed*1000/print_every, 
                                                                                                                                           lr, loss_mean, 
                                                                                                                                           loss_diff_mean, 
                                                                                                                                           loss_der_mean,
                                                                                                                                           log_p_acc)) 
            start_time = time.time()
            loss_lst = []
            loss_diff_lst = []
    if reg == 0:
        torch.save(vf.state_dict(), 'vf_2d_best_reg_{}.pth'.format(int(reg*1000)))
        print('save model at iter {}'.format(i+1))

    return best_ll, log_p_acc, best_det, log_det_acc


def test(step_size=0.01, batch_size = 500000, 
         eps_time = 1e-2, 
         hidden_dim = 128, num_T=3, reg=1):

    vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device) 
    vf.load_state_dict(torch.load('vf_2d_best_reg_{}.pth'.format(int(reg*1000))))
    wrapped_vf = WrappedModel(vf)
    norm = cm.colors.Normalize(vmax=50, vmin=0)


    T = torch.linspace(0, 1, num_T)  # sample times
    T = T.to(device=device)
    x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
    solver = ODESolver(velocity_model = wrapped_vf)  # create an ODESolver class
    sol = solver.sample(time_grid=T, x_init=x_init, method='euler', 
                        step_size=step_size, 
                        return_intermediates=True)  # sample from the model


    sol = sol.cpu().numpy()
    T = T.cpu()

    fig, axs = plt.subplots(1, 1, figsize=(16, 16))

    i = num_T-1
    H= axs.hist2d(sol[i,:,0], sol[i,:,1], 200, range=((-5,5), (-5,5)), cmap='copper')
    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    _ = axs.hist2d(sol[i,:,0], sol[i,:,1], 200, range=((-5,5), (-5,5)), norm=norm, cmap='copper')
    axs.set_aspect('equal')
    axs.axis('off')
    axs.set_title('t= %.2f' % (T[i]), fontsize=40)
    
    plt.tight_layout()
    plt.savefig('cfm_on_checkerboard_reg_{}.png'.format(int(reg*1000)))

    # plot ground truth
    x_1 = inf_train_gen(batch_size=batch_size, device=device)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    H = ax.hist2d(x_1[:,0].cpu().numpy(), x_1[:,1].cpu().numpy(), 200, range=((-5,5), (-5,5)), cmap='copper')
    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    _ = ax.hist2d(x_1[:,0].cpu().numpy(), x_1[:,1].cpu().numpy(), 200, range=((-5,5), (-5,5)), norm=norm, cmap='copper')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ground Truth', fontsize=40)
    plt.tight_layout()
    plt.savefig('ground_truth.png')

if __name__ == "__main__":
    tunning = False
    bs_ = 512
    lr_ = 1e-3
    ite_ = 30001
    freeze = True
    path_ = "OT"
    reg_v = 0.1
    best_ll, _, best_det, _ = train(lr=lr_, batch_size=bs_,
                                iterations=ite_, print_every=200, 
                                hidden_dim=512,
                                reg=reg_v, path=path_)
    print('best ll: ', best_ll)
    print('best det: ', best_det)
    test(hidden_dim=512, reg=reg_v)

