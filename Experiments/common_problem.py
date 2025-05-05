import torch

# ----- problem definition (same for all methods) -----------------
A = torch.tensor([[1, 10],
                  [0.0, 1.01]]).to(torch.float32)

def f(x, w):
    # leader objective
    return (w[0]-3.0)**2 + (w[1]+2.0)**2 + 0.2 * (x @ x)

def g(x, w):
    # follower objective (strongly convex in x)
    return 0.5 * (x - A @ w).T @ (x - A @ w)

# ----- closed‑form follower --------------------------------------


def x_star_alpha(w, alpha):
    return (A @ w) / (1.0 + 0.4 * alpha) #closed form minimizer of the regularized lower level problem x_alpha (using a quadratic objective function allows us to use direct closed form solution)


#since the problem is strongly convex on its own, we do not need to add another strongly convex function to it 

# ----- helper gradients expected by the BOME driver --------------
# def f_x(x, w):
#     grad = torch.autograd.grad(f(x,w), x, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(x)

# def f_w(x, w):
#     grad = torch.autograd.grad(f(x,w), w, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(w)

# def g_x(x, w):
#     grad = torch.autograd.grad(g(x,w), x, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(x)

# def g_w(x, w):
#     grad = torch.autograd.grad(g(x,w), w, allow_unused=True)[0]
#     return grad if grad is not None else torch.zeros_like(w)

# def g_x_xhat_w(x, xhat, w):
#     # exactly the signature toy_lls expects
#     loss = g(x,w) - g(xhat.detach(), w)
#     grad_x, grad_w = torch.autograd.grad(loss, (x,w), allow_unused=True)
#     if grad_x is None:  grad_x = torch.zeros_like(x)
#     if grad_w is None:  grad_w = torch.zeros_like(w)
#     return loss.item(), grad_x, grad_w



def g_x(x, w):
    grad = torch.autograd.grad(g(x,w), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)

def g_w(x, w):
    grad = torch.autograd.grad(g(x,w), w, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(w)

def f_x(x, w):
    grad = torch.autograd.grad(f(x,w), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)

def f_w(x, w):
    grad = torch.autograd.grad(f(x,w), w, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(w)

def g_x_xhat_w(x, xhat, w):
    loss = g(x, w) - g(xhat.detach(), w)
    grad = torch.autograd.grad(loss, [x, w], allow_unused=True)
    return loss.detach().cpu().item(), grad[0], grad[1]

