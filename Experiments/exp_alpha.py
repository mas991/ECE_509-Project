import torch
import numpy as np
import matplotlib.pyplot as plt
from run_reduced import run_reduced, phi_alpha, default_step

# 1) Run the solver once to get the trajectory
res = run_reduced(alpha0=1.0, delta=0.5, step=default_step,
                  max_inner=50, tol_inner=1e-8, tol_outer=1e-4)

# 2) Build a grid of (w0,w1) points
w0_vals = np.linspace(-3, 5, 100)
w1_vals = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)

# 3) Compute Φₐ(w) on the grid (use alpha=final alpha or initial alpha)
alpha = 1.0  # or res['alpha'][-1]
Phi = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w = torch.tensor([W0[i,j], W1[i,j]], dtype=torch.float32)   
        Phi[i,j] = phi_alpha(w, alpha).item()

# 4) Contour‑plot
plt.figure(figsize=(6,5))
cs = plt.contour(W0, W1, Phi, levels=30, cmap='viridis')
plt.clabel(cs, inline=1, fontsize=8)
plt.title("Contour of Φₐ(w) with GD trajectory overlay")
plt.xlabel("w₀"); plt.ylabel("w₁")

# 5) Overlay the trajectory
traj = res['w'].detach().numpy()
plt.plot(traj[:,0], traj[:,1], '-o', color='red', markersize=3, label='GD path')

plt.legend()
plt.show()
