# -*- coding: utf-8 -*-
import copy
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from common_problem import (f, g, f_x, f_w, g_x, g_w,
                            g_x_xhat_w, x_star_alpha)


################################################################################
#
#  Bilevel Optimization Toy Example
#
#  min_{x,w} f(x, w)
#  s.t. x = argmin_x g(x, w)
#
#  f_x = df/dx
#  f_w = df/dw
#  g_x = dg/dx
#  g_w = dg/dw
#
#  xhat = x^{(k)}, where k is steps of inner optimization
#  inner_opt means optimizer of xhat
#  opt means optimizer of [x, w]
#
################################################################################

LOWER = -100
UPPER = 100

def bilevel_descent_bome(x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta):
    xs, ws, fs, gs, xhats, times = [], [], [], [], [], []

    gg = []

    xhat = copy.deepcopy(x)
    x_opt = torch.optim.SGD([x], lr=x_lr)
    w_opt = torch.optim.SGD([w], lr=w_lr)
    xhat_opt = torch.optim.SGD([xhat], lr=xhat_lr)

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    n_params_x = x.numel()
    n_params_w = w.numel()
    df = torch.zeros(n_params_x+n_params_w).to(x.device)
    dg = torch.zeros(n_params_x+n_params_w).to(x.device)

    t = 0

    for i in range(maxIter):
        t0 = time.time()
        xhat.data = x.data.clone()
        for j in range(k):
            xhat_opt.zero_grad()
            xhat.grad = g_x(xhat, w).data
            xhat_opt.step()
            xhat.data.clamp_(LOWER, UPPER)

        xhats.append(xhat.data.clone().view(-1).cpu())
        g_gap = (g(x, w) - g(xhat, w)).data.clone()
        gg.append(g(x,w).view(-1).clone().cpu())
        
        # prepare gradients 
        fx = f_x(x, w)
        fw = f_w(x, w)

        loss, gx, gw_minus_gw_k = g_x_xhat_w(x, xhat, w)

        df[:n_params_x].copy_(fx.view(-1).clone())
        dg[:n_params_x].copy_(gx.view(-1).clone())
        df[n_params_x:].copy_(fw.view(-1).clone())
        dg[n_params_x:].copy_(gw_minus_gw_k.clone())

        norm_dq = dg.norm().pow(2)
        dot = df.dot(dg)

        d = df + F.relu(eta - dot/(norm_dq+1e-8)) * dg

        w_opt.zero_grad()
        x_opt.zero_grad()
        x.grad = d[:n_params_x].data.view(x.shape).clone()
        w.grad = d[n_params_x:].data.view(w.shape).clone()
        w_opt.step()
        x_opt.step()
        x.data.clamp_(LOWER, UPPER)
        t1 = time.time()
        t += t1-t0

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())
        times.append(t)

    res = { 
        'x'   : torch.vstack(xs),
        'xhat': torch.vstack(xhats),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
        't'  : torch.Tensor(times).view(-1),
        'gg' : torch.vstack(gg).view(-1),
    }
    return res 


def BSG_1(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500):
    xs, ws, fs, gs = [], [], [], []
    gg = []
    times = []

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))
    w_opt = torch.optim.SGD([w], lr=w_lr)
    x_opt = torch.optim.SGD([x], lr=x_lr)

    t = 0
    for i in range(maxIter):

        t0 = time.time()
        g0 = g(x,w).data.clone()
        for it in range(k):
            x_opt.zero_grad()
            x.grad = g_x(x, w).data
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)
        g_gap = g0 - g(x,w).data.clone()
        gg.append(g(x,w).data.clone().view(-1).cpu())

        # prepare gradients 
        fx = f_x(x, w)
        fw = f_w(x, w)
        gx = g_x(x, w)
        gw = g_w(x, w)

        w_opt.zero_grad()
        w.grad = (fw - fx.view(-1).dot(gx.view(-1)) / (gx.view(-1).dot(gx.view(-1))+1e-8) * gw).data
        w_opt.step()
        t1 = time.time()
        t += t1-t0

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())
        times.append(t)

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
        't'   : torch.Tensor(times).view(-1),
        'gg'  : torch.Tensor(gg).view(-1),
    }
    return res 


def calculate_g_gap(x, w, lr, k):
    x_ = copy.deepcopy(x)
    g0 = g(x, w).data.clone()
    for j in range(k):
        x_.data = x_.data - lr * g_x(x_, w).data.clone()
        x_.data.clamp_(LOWER, UPPER)
    gnow = g(x_, w).data.clone()
    return g0 - gnow
    

def BVFSM(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500, l2_reg=0.1, ln_reg=1):
    xs, ws, fs, gs = [], [], [], []
    times = []
    gg = []

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))

    z_l2_reg_coef = l2_reg
    y_l2_reg_coef = l2_reg
    y_ln_reg_coef = ln_reg
    decay_rate = 1.1

    z = copy.deepcopy(x)
    w_opt = torch.optim.SGD([w], lr=w_lr)
    z_opt = torch.optim.SGD([z], lr=xhat_lr)
    x_opt = torch.optim.SGD([x], lr=x_lr)

    t = 0
    for i in range(maxIter):
        g_gap = calculate_g_gap(x, w, xhat_lr, k)

        t0 = time.time()
        reg_decay_rate = 1 / (math.log(decay_rate * (maxIter+1)))

        for it in range(k):
            z_opt.zero_grad()
            loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            loss_z.backward()
            z_opt.step()
            z.data.clamp_(LOWER, UPPER)

        for it in range(k):
            x_opt.zero_grad()
            loss_x = g(x, w)
            loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
            log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + loss_z.detach() - loss_x + 1e-4)
            loss_x = f(x, w) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
            loss_x.backward()
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)

        gg.append(g(x,w).data.clone().cpu().view(-1))

        w_opt.zero_grad()
        loss_x = g(x, w)
        loss_z = g(z, w) + z_l2_reg_coef * reg_decay_rate * z.norm(2).pow(2)
        log_barrier = -y_ln_reg_coef * reg_decay_rate * torch.log(loss_x.detach() + loss_z - loss_x + 1e-4)
        loss_w = f(x, w) + log_barrier + y_l2_reg_coef * reg_decay_rate * x.norm(2).pow(2)
        loss_w.backward()
        w_opt.step()

        t1 = time.time()
        t += t1-t0

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())
        times.append(t)

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
        't'   : torch.Tensor(times).view(-1),
        'gg'  : torch.vstack(gg).view(-1),
    }
    return res 


def penalty(x, w, x_lr, w_lr, xhat_lr, k, maxIter=500, lmbd_g=0.1, eps=0.1, gamma=0.1):
    xs, ws, fs, gs = [], [], [], []
    times = []
    gg = []

    xs.append(x.data.clone().view(-1))
    ws.append(w.data.clone().view(-1))
    t = 0

    def penalty_gx(x, w, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x,w), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w) + (nu_k * gx).mean() + lmbd_g * g(x, w) + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_x = torch.autograd.grad(loss, x, allow_unused=True)[0]
        return grad_x, grad_x.norm().detach().cpu().item()

    def penalty_gw(x, w, gamma_k, nu_k):
        gx = torch.autograd.grad(g(x,w), x, create_graph=True, allow_unused=True)[0]
        loss = f(x, w) + (nu_k * gx).mean() + 0.5 * gamma_k * gx.norm(2).pow(2)
        grad_w = torch.autograd.grad(loss, w, allow_unused=True)[0]
        return grad_w, grad_w.norm().detach().cpu().item(), gx

    lmbd_g = lmbd_g
    eps    = eps
    gamma  = gamma
    nu     = torch.ones_like(x) * 1e-4

    c_gamma = 1.1
    c_eps = 0.9
    c_lmbd = 0.9

    x_opt = torch.optim.SGD([x], lr=x_lr)
    w_opt = torch.optim.SGD([w], lr=w_lr)

    for i in range(maxIter):

        g_gap = calculate_g_gap(x, w, xhat_lr, k)
        t0 = time.time()
        for j in range(k):
            x_opt.zero_grad()
            grad, gx_norm = penalty_gx(x, w, gamma, nu)
            x.grad = grad.data
            x_opt.step()
            x.data.clamp_(LOWER, UPPER)

        grad, gw_norm, gx = penalty_gw(x, w, gamma, nu)
        w_opt.zero_grad()
        w.grad = grad.data
        w_opt.step()

        gg.append(g(x,w).data.clone().view(-1).cpu())

        if gx_norm**2 + gw_norm**2 < eps**2:
            gamma *= c_gamma
            eps *= c_eps
            lmbd_g *= c_lmbd
            nu += gx.detach().data * gamma
            w_lr *= 0.9
            x_lr *= 0.9
            w_opt = torch.optim.SGD([w], lr=w_lr)
            x_opt = torch.optim.SGD([x], lr=x_lr)
            print("update gamma and eps", gamma, eps)

        t1 = time.time()

        xs.append(x.data.clone().view(-1).cpu())
        ws.append(w.data.clone().view(-1).cpu())
        fs.append(f(x,w).data.clone().view(-1).cpu())
        gs.append(g_gap.clone().view(-1).cpu())
        t += t1-t0
        times.append(t)

    res = { 
        'x'   : torch.vstack(xs),
        'w'   : torch.vstack(ws),
        'f'   : torch.vstack(fs).view(-1),
        'g'   : torch.vstack(gs).view(-1),
        't'   : torch.Tensor(times).view(-1),
        'gg'  : torch.Tensor(gg).view(-1),
    }
    return res


def plotme(res, xcts0, xcts, name): 
    plt.figure()
    fig, axs = plt.subplots(1,3, figsize=(15, 5))

    # plot f traj
    axs[0].plot(res['f'], label='f', color='b')
    axs[0].set_ylim(0, 50)
    axs[0].legend()
    axs[0].set_title("f(xt, wt)")

    # plot g traj
    if res['g'] is not None:
        axs[1].plot(res['g'], label='g_gap', color='m')
        axs[1].legend()
    axs[1].set_yscale('log')
    axs[1].set_title("g_gap(xt, wt)")
    
    hull = ConvexHull(xcts)

    # plot x traj
    n = res['x'].shape[0]
    color_list = np.zeros((n,3))
    color_list[:,0] = 1.
    color_list[:,1] = np.linspace(0, 1, n)
    axs[2].scatter(res['x'][:,0], res['x'][:,1], color=color_list, s=3, zorder=4)

    axs[2].text(x=res['x'][0,0], y=res['x'][0,1], s="start", color='r')
    axs[2].text(x=xcts0[0,0], y=xcts0[0,1], s="goal", color='g')
    axs[2].scatter(res['x'][0,0], res['x'][0,1], s=15, color='r', zorder=15)
    axs[2].scatter(xcts0[:,0], xcts0[:,1], color='g', s=15, zorder=15)

    # plot X, the points and their convex hull
    axs[2].scatter(xcts[:,0], xcts[:,1], color='k', s=10, zorder=10)
    for simplex in hull.simplices:
        axs[2].plot(xcts[simplex, 0], xcts[simplex, 1], 'c')
    axs[2].plot(xcts[hull.vertices, 0],
                xcts[hull.vertices, 1], 'o', mec='r', color='none', lw=1,
                markersize=10)
    axs[2].set_title('x trajectory')
    axs[2].set_xlim(-4, 4)
    axs[2].set_ylim(-4, 4)

    fig.suptitle(f"{name}", x=0.5, y=0.04)
    plt.tight_layout()
    plt.savefig(f"imgs/{name}.png")
    plt.close()
    time.sleep(1)
    return


def control_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


### original f, g and their partial derivatives

# def f(x, w): 
#     return 0.5 * (w - x[1])**2 + 0.5 * (x[0] - 1)**2

# def g(x, w):
#     return 0.5 * x[0]**2 - w * x[0]

# def g_x(x, w):
#     grad = torch.autograd.grad(g(x,w), x, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(x)

# def g_w(x, w):
#     grad = torch.autograd.grad(g(x,w), w, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(w)

# def f_x(x, w):
#     grad = torch.autograd.grad(f(x,w), x, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(x)

# def f_w(x, w):
#     grad = torch.autograd.grad(f(x,w), w, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(w)

# def g_x_xhat_w(x, xhat, w):
#     loss = g(x, w) - g(xhat.detach(), w)
#     grad = torch.autograd.grad(loss, [x, w], allow_unused=True)
#     return loss.detach().cpu().item(), grad[0], grad[1]


### toy coreset
if __name__ == "__main__":
    seed = 0
    control_seed(seed)

    x = torch.FloatTensor([0,0]).requires_grad_()
    w = torch.FloatTensor([0]).requires_grad_()
    x_data = x.data.clone()
    w_data = w.data.clone()

    maxIter = 2000
    k = 10
    x_lr = xhat_lr = 0.5
    w_lr = 0.5


    if not os.path.exists("./imgs"):
        os.mkdir("./imgs")

    fn_maps = {
        "bome": bilevel_descent_bome,
        "BSG-1": BSG_1,
        "BVFSM": BVFSM,
        "penalty": penalty,
    }

    results = {
        'method': {},
        'iter': {},
        'eta': {},
    }

    # method = "bome"
    # eta = 0.5
    # k=1
    # x.data = x_data.clone()
    # w.data = w_data.clone()
    # res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
    # results['method'][method] = res
    # print(method, x, w)

    method = "BSG-1"
    x.data = x_data.clone()
    w.data = w_data.clone()
    res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter)
    results['method'][method] = res
    print(method, x, w)

    method = "BVFSM"
    l2reg = 0.1; lnreg = 0.001
    x.data = x_data.clone()
    w.data = w_data.clone()
    res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, l2reg, lnreg)
    results['method'][method] = res
    print(method, x, w)

    method = "penalty"
    lmbd_g = 0.1; eps = 0.1; gamma = 0.01
    x.data = x_data.clone()
    w.data = w_data.clone()
    res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, lmbd_g, eps, gamma)
    results['method'][method] = res
    print(method, x, w)

    method = "bome"
    eta = 0.5
    for k, maxIter in zip([1, 10, 100], [1000, 1000, 1000]):
        x.data = x_data.clone()
        w.data = w_data.clone()
        res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
        results['iter'][k] = res
        print(method, k, x, w)

    method = "bome"
    eta = 0.5
    maxIter =  5000
    k = 10
    for eta in [0.1, 0.5, 0.9]:
        x.data = x_data.clone()
        w.data = w_data.clone()
        res = fn_maps[method](x, w, x_lr, w_lr, xhat_lr, k, maxIter, eta)
        results['eta'][eta] = res
        print(method, eta, x, w)

    torch.save(results, "D:/BOME/Results 2025/toy_lls_result.pt")
