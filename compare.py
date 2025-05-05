import matplotlib.pyplot as plt
from run_bome   import run_bome

from run_reduced import run_reduced

# ---------- run both methods ----------
# bome_res = run_bome(k=3)
gd_res   = run_reduced()

# ---------- final numbers ----------
print("\nFinal leader variables / objective")
# print("BOME    w =", bome_res['w'][-1].tolist(),
#       "  φ =", bome_res['f'][-1].item())
print("Reduced w =", gd_res['w'][-1].tolist(),
      "  φ =", gd_res['f'][-1].item())

# ---------- convergence plot ----------
plt.figure(figsize=(6,4))
# plt.plot(bome_res['t'], bome_res['f'], label='BOME (k=1)')
plt.plot(gd_res['t'],   gd_res['f'],   label='Exact‑follower GD')
plt.yscale('log'); plt.xlabel('time (s)'); plt.ylabel('φ(w)')
# plt.semilogy(bome_res['g'], label='BOME gap g(x,w)-g(x*,w)')

plt.legend(); plt.tight_layout(); plt.show()
