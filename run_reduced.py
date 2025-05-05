import torch, time
from common_problem import x_star_alpha, f, A

def phi_alpha(w, alpha):
    """
    Reduced upper-level objective:
      phi_alpha(w) = f_leader(x_star_alpha(w, alpha), w)
    """
    x = x_star_alpha(w, alpha)
    return f(x, w)

def grad_phi_alpha(w, alpha):
    w_ = w.detach().clone().requires_grad_(True)
    y = phi_alpha(w_, alpha)
    (g,) = torch.autograd.grad(y, w_)
    return g
L = 2 + 0.4 * (torch.linalg.svdvals(A).max()**2)
default_step = 0.9 / L       # safe choice



def run_reduced(alpha0=1.0, delta=0.5, step=default_step,
                max_inner=50, tol_inner=1e-8, tol_outer=1e-4):
    
    #Outer loop on alpha, inner (approximate) gradient descent on theta_alpha(w)=f(x_alpha(w),w).
      #alpha0:    initial regularization weight
      #delta:     factor to shrink alpha each outer iteration
      #step:  initial trial step‑size for w
      #max_inner: max inner GD iters on w per alpha
      #tol_inner: stop inner when ‖grad_phi_alpha‖<tol_inner
      #tol_outer: stop outer when alpha<tol_outer
    

    # initialize w and alpha
    # w = torch.tensor([6., -7.], requires_grad=True)
    w = torch.tensor([-2., 5.], requires_grad=True)

    alpha = alpha0

    # storage for plotting
    Ws, Phis, Alphas = [], [], []
    t0 = time.time() #to keep track of time

    # Outer loop: shrink alpha until it is small
    while alpha > tol_outer:
        # Inner loop: descend on phi_alpha(y) with back‑tracking line‑search
        for _ in range(max_inner):
            # compute gradient of the reduced objective phi_alpha at current y
            g = grad_phi_alpha(w, alpha)

            # record current state
            Ws.append(w.clone())
            Phis.append(phi_alpha(w, alpha).item())
            Alphas.append(alpha)

            # stopping check: if gradient is tiny, break
            if g.norm() < tol_inner:
                break

            # --------- Armijo back‑tracking line‑search on w ---------
            phi0 = phi_alpha(w, alpha)        # current phi_alpha,w
            dir  = -g                         # descent direction
            t    = step                       # initial trial step‑size
            beta, sigma = 0.5, 1e-4           # shrink factor & Armijo constant

            # precompute dot product once
            dg = g.dot(dir)                   # = −‖g‖²
            rhs = phi0 + sigma * t * dg       # Armijo right‑hand side

            # shrink t until sufficient decrease holds
            while True:
                w_new = (w + t * dir).detach().requires_grad_(True)
                if phi_alpha(w_new, alpha) <= rhs:
                    break
                t *= beta
                rhs = phi0 + sigma * t * dg
                # avoid infinite loop
                if t < 1e-8:
                    break

            # apply the accepted step
            w = w_new
            # --------------------------------------------------------

        # shrink α for the next outer iteration
        alpha *= delta

    # package results
    Ws = torch.stack(Ws)
    Phis = torch.tensor(Phis)
    Ts = torch.linspace(0, time.time() - t0, len(Ws))

    return {'w': Ws, 'f': Phis, 'alpha': torch.tensor(Alphas), 't': Ts}
