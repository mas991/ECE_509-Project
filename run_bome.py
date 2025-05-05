import torch
import toy_lls
from toy_lls import bilevel_descent_bome
import common_problem as P

# override toy_lls’ locals with these exact names
toy_lls.f          = P.f
toy_lls.g          = P.g
toy_lls.f_x        = P.f_x
toy_lls.f_w        = P.f_w
toy_lls.g_x        = P.g_x
toy_lls.g_w        = P.g_w
toy_lls.g_x_xhat_w = P.g_x_xhat_w
toy_lls.x_star     = P.x_star_alpha   # if toy_lls uses x_star



def run_bome(k=50, max_iter=50, eta=0.5,
             lr_x=0.2, lr_w=0.2, lr_xhat=1.1):

    x = torch.tensor([-5.,4.],requires_grad=True)
    w = torch.tensor([6.,-7.],requires_grad=True)

    res = bilevel_descent_bome(
        x, w,
        x_lr   = lr_x,
        w_lr   = lr_w,
        xhat_lr= lr_xhat,
        k      = k,
        maxIter= max_iter,
        eta    = eta
    )
    return res
