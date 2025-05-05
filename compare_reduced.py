

import time
import torch
import matplotlib.pyplot as plt

import common_problem    # this module defines A, f, g, x_star_alpha, etc.
from run_reduced import run_reduced, default_step

#--- define three A matrices with increasing condition number ---
A_list = {
    "well‑cond"   : torch.tensor([[1.0, 1.0],[0.0, 1.0]]),
    "med‑cond"    : torch.tensor([[1.0, 10.0],[0.0, 1.01]]),
    "ill‑cond"    : torch.tensor([[1.0, 100.0],[0.0, 1.001]]),
}

#--- three initial regularizers α₀ and shrink δ ---
alpha0_list = [1.0, 0.5, 0.1]
delta = 0.5

# inner‑loop and stopping parameters
max_inner = 50
tol_inner = 1e-8
tol_outer = 1e-4

# collect results
results = {}

plt.figure(figsize=(8,6))
for (name,A) in A_list.items():
    # monkey‑patch the follower matrix in common_problem and in run_reduced
    common_problem.A = A
    # recompute L and step for this A
    L = 2 + 0.4*(torch.linalg.svdvals(A).max()**2)
    step = 0.9 / L

    for alpha0 in alpha0_list:
        # run the reduced GD
        res = run_reduced(
            alpha0=alpha0,
            delta=delta,
            step=step,
            max_inner=max_inner,
            tol_inner=tol_inner,
            tol_outer=tol_outer,
        )

        key = (name, alpha0)
        results[key] = res

        # plot φ vs time
        plt.plot(res['t'], res['f'],
                 label=f"{name}, α₀={alpha0}")

        # print final w, φ
        w_final = res['w'][-1].tolist()
        phi_final = res['f'][-1].item()
        print(f"{name:>8s}  α₀={alpha0:<4.1f}  → w={w_final},  φ={phi_final:.4f}")

# finalize plot
plt.yscale('log')
plt.xlabel('time (s)')
plt.ylabel('φ(w)')
plt.title("Reduced two‑loop GD: φ vs time for varying A & α₀")
plt.legend()
plt.tight_layout()
plt.savefig("compare_reduced.png")
plt.show()
