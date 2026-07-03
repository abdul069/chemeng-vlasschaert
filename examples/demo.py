"""Genereert de demonstratiefiguren van het pakket."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chemeng.distillation import mccabe_thiele, eq_y
from chemeng.vle import txy, azeotrope
from chemeng.fvm import convection_diffusion, analytical
from chemeng.heat import transient_conduction

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1 — McCabe-Thiele
ax = axes[0, 0]
res = mccabe_thiele(alpha=2.5, xF=0.5, xD=0.95, xB=0.05, R=2.0, q=1.0)
xx = np.linspace(0, 1, 200)
ax.plot(xx, xx, "k--", lw=0.8)
ax.plot(xx, eq_y(xx, 2.5), "b", label="evenwicht (α=2.5)")
ax.plot([0.95, res["x_int"]], [0.95, res["y_int"]], "g", label="rectificatie")
ax.plot([res["x_int"], 0.05], [res["y_int"], 0.05], "orange", label="stripping")
ax.plot([0.5, res["x_int"]], [0.5, res["y_int"]], "m", label="q-lijn (q=1)")
sx, sy = zip(*res["stages"])
ax.step(sx, sy, "r", where="post", lw=1)
ax.set(title=f"McCabe-Thiele — N={res['N']}, voeding op {res['feed_stage']}",
       xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1))
ax.legend(fontsize=8)

# 2 — Txy met azeotroop (ethanol/water, Margules-2)
ax = axes[0, 1]
antE = (8.04494, 1554.3, 222.65); antW = (7.96681, 1668.21, 228.0)
x, y, T = txy(760, antE, antW, A12=1.60, A21=0.85, n=61, Tlo=50, Thi=110)
ax.plot(x, T, "b", label="kooklijn T(x)")
ax.plot(y, T, "r", label="dauwlijn T(y)")
az = azeotrope(760, antE, antW, 1.60, 0.85, 0.02, 0.98, Tlo=50, Thi=110)
ax.plot(az[0], az[1], "k*", ms=14, label=f"azeotroop ({az[0]:.3f}, {az[1]:.1f} °C)")
ax.set(title="Txy ethanol/water — Margules-2, 760 mmHg",
       xlabel="x, y ethanol", ylabel="T (°C)")
ax.legend(fontsize=8)

# 3 — FVM-schema's vs exact
ax = axes[1, 0]
pars = dict(rho=1.0, u=2.5, gamma=0.1, L=1.0, phi0=1.0, phiL=0.0)
xf = np.linspace(0, 1, 300)
ax.plot(xf, analytical(x=xf, **pars), "k", lw=2, label="exact (Pe=25)")
for s, mk in (("cds", "o"), ("upwind", "s"), ("powerlaw", "^")):
    xg, phi = convection_diffusion(n=15, scheme=s, **pars)
    ax.plot(xg, phi, mk, ms=4, ls=":", label=s)
ax.set(title="FVM 1D convectie-diffusie — schemavergelijking (n=15)",
       xlabel="x", ylabel="φ")
ax.legend(fontsize=8)

# 4 — transiënte geleiding: plaat vs cilinder vs bol
ax = axes[1, 1]
for g, col in (("plaat", "b"), ("cilinder", "g"), ("bol", "r")):
    r, T_ = transient_conduction(g, 1e-5, 0.1, 201, 0.05, 60.0, 500.0, 100.0)
    ax.plot(r * 100, T_, col, label=g)
ax.set(title="Transiënte geleiding t=60 s — r=0-singulariteit geregulariseerd",
       xlabel="r (cm)", ylabel="T (K)")
ax.legend(fontsize=8)

fig.suptitle("chemeng — herbouw archief P. Vlasschaert (geverifieerd, 27/27)",
             fontsize=13)
fig.tight_layout()
fig.savefig("examples/demo.png", dpi=130)
print("figuur weggeschreven: examples/demo.png")
