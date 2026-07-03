"""Verificatiesuite — elke module wordt getest tegen een analytische
oplossing of een onafhankelijke referentie. Dit is de laag die het
originele archief miste."""
import numpy as np
import sympy as sp

from chemeng.numerics import tdma, newton_symbolic, sor
from chemeng.heat import (steady_conduction_1d, transient_conduction,
                          slab_analytical)
from chemeng.eos import VanDerWaals, RedlichKwong, PengRobinson, R
from chemeng.vle import bubble_T, azeotrope, txy
from chemeng.flash import rachford_rice
from chemeng.distillation import mccabe_thiele, smoker, rayleigh, eq_y
from chemeng.reactor import axial_dispersion, axial_dispersion_analytical
from chemeng.fvm import convection_diffusion, analytical, symbolic_scheme

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    status = "PASS" if cond else "FAIL"
    if cond: ok += 1
    else: fail += 1
    print(f"[{status}] {name}  {detail}")

print("=" * 72)
print("1. NUMERIEKE KERN")
print("=" * 72)
# TDMA vs numpy
n = 50
rng = np.random.default_rng(1)
a = rng.uniform(0.5, 1, n); c = rng.uniform(0.5, 1, n)
b = a + c + rng.uniform(1, 2, n)          # diagonaal dominant
d = rng.uniform(-1, 1, n)
A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
err = np.max(np.abs(tdma(a, b, c, d) - np.linalg.solve(A, d)))
check("TDMA vs np.linalg.solve", err < 1e-10, f"max|Δ|={err:.2e}")

# Newton met symbolische Jacobiaan: snijpunt cirkel & hyperbool
x, y = sp.symbols("x y")
sol, it, conv = newton_symbolic([x**2 + y**2 - 4, x*y - 1], [x, y], [2.0, 0.3])
res = max(abs(sol[0]**2 + sol[1]**2 - 4), abs(sol[0]*sol[1] - 1))
check("Newton+sym.Jacobiaan", conv and res < 1e-10,
      f"iters={it}, residu={res:.1e}")

# SOR: Laplace met u=y op alle randen → exacte oplossing u=y
u2d, its = sor(0.0, 31, 31, bc={"left": np.linspace(0,1,31),
                                "right": np.linspace(0,1,31),
                                "bottom": 0.0, "top": 1.0})
Y = np.linspace(0, 1, 31)[:, None] * np.ones((1, 31))
err = np.max(np.abs(u2d - Y))
check("SOR 2D-Laplace (u=y exact)", err < 1e-5, f"iters={its}, max|Δ|={err:.1e}")

print()
print("=" * 72)
print("2. WARMTETRANSPORT (TDMA-reeks)")
print("=" * 72)
# steady met bron: analytisch T = T0 + q/(2k)·x(L−x) bij T0=TL
xg, T = steady_conduction_1d(k=2.0, L=1.0, n=101, T0=100, TL=100, q_gen=500)
Tan = 100 + 500/(2*2.0) * xg * (1 - xg)
err = np.max(np.abs(T - Tan))
check("1D geleiding + bron vs analytisch", err < 1e-9, f"max|Δ|={err:.1e}")

# transiënte plaat vs reeksoplossing
alpha, L = 1e-5, 0.1
r, Tp = transient_conduction("plaat", alpha, L, 201, 0.05, 60.0, 500.0, 100.0)
Ta = slab_analytical(alpha, L, r, 60.0, 500.0, 100.0)
err = np.max(np.abs(Tp - Ta))
check("transiënte plaat vs reeksoplossing", err < 0.5,
      f"max|ΔT|={err:.3f} K op ΔT=400 K")

# cilinder/bol: r=0-singulariteit → eindige, symmetrische, monotone oplossing
for geom in ("cilinder", "bol"):
    r, Tc = transient_conduction(geom, alpha, L, 201, 0.05, 60.0, 500.0, 100.0)
    check(f"transiënt {geom}: r=0 regulier & fysisch",
          np.isfinite(Tc).all() and Tc[0] == Tc.max()
          and np.all(np.diff(Tc) < 1e-9),
          f"T(0)={Tc[0]:.2f} K, T(R)={Tc[-1]:.2f} K")
# fysische ordening: bol koelt sneller dan cilinder, cilinder sneller dan plaat
_, Tpl = transient_conduction("plaat", alpha, L, 201, 0.05, 60.0, 500.0, 100.0)
_, Tcy = transient_conduction("cilinder", alpha, L, 201, 0.05, 60.0, 500.0, 100.0)
_, Tsp = transient_conduction("bol", alpha, L, 201, 0.05, 60.0, 500.0, 100.0)
check("fysica: T_bol < T_cil < T_plaat in centrum",
      Tsp[0] < Tcy[0] < Tpl[0],
      f"{Tsp[0]:.1f} < {Tcy[0]:.1f} < {Tpl[0]:.1f}")

print()
print("=" * 72)
print("3. TOESTANDSVERGELIJKINGEN (Elliott & Lira-lijn)")
print("=" * 72)
# CO2: Tc=304.13 K, Pc=73.77 bar, ω=0.2239
for name, eos in (("VdW", VanDerWaals(304.13, 73.77e5)),
                  ("RK", RedlichKwong(304.13, 73.77e5)),
                  ("PR", PengRobinson(304.13, 73.77e5, 0.2239))):
    Vl, Vv = eos.volumes(280.0, 30e5)           # subkritisch: 2 fasen mogelijk
    Pl, Pv = eos.pressure(280.0, Vl), eos.pressure(280.0, Vv)
    check(f"{name}: P(V_wortels) reproduceert P", 
          abs(Pl-30e5)<1 and abs(Pv-30e5)<1,
          f"Vl={Vl*1e6:.1f}, Vv={Vv*1e6:.0f} cm3/mol")
# ideale-gaslimiet
eos = PengRobinson(304.13, 73.77e5, 0.2239)
_, Vv = eos.volumes(600.0, 1e5)
check("PR → ideale-gaslimiet (600 K, 1 bar)",
      abs(Vv / (R*600/1e5) - 1) < 0.01, f"Z={Vv/(R*600/1e5):.4f}")

print()
print("=" * 72)
print("4. VLE & AZEOTROOP (Margules-2, ethanol/water-achtig)")
print("=" * 72)
# Antoine (mmHg, °C): ethanol & water — klassieke constanten
antE = (8.04494, 1554.3, 222.65)
antW = (7.96681, 1668.21, 228.0)
P = 760.0
T, y = bubble_T(0.5, P, antE, antW, 0, 0, Tlo=50, Thi=110)   # ideaal
check("ideale bubble-T tussen kookpunten", 78.3 < T < 100.0,
      f"T={T:.2f} °C, y_EtOH={y:.3f}")
az = azeotrope(P, antE, antW, A12=1.60, A21=0.85, x_lo=0.02, x_hi=0.98, Tlo=50, Thi=110)
check("azeotroop gevonden (Margules-2)",
      az is not None and 0.85 < az[0] < 0.95 and 77.5 < az[1] < 78.5,
      f"x_az={az[0]:.3f}, T_az={az[1]:.2f} °C  (lit.: ±0.894, 78.2 °C)")

print()
print("=" * 72)
print("5. FLASH (Rachford-Rice)")
print("=" * 72)
z = [0.30, 0.40, 0.30]; K = [3.0, 1.2, 0.4]
psi, xL, yV = rachford_rice(z, K)
mb = np.max(np.abs(psi*yV + (1-psi)*xL - np.asarray(z)))
check("massabalans per component", mb < 1e-12, f"ψ={psi:.4f}, max|Δ|={mb:.1e}")
check("Σx=Σy=1", abs(xL.sum()-1) < 1e-10 and abs(yV.sum()-1) < 1e-10,
      f"Σx={xL.sum():.12f}, Σy={yV.sum():.12f}")

print()
print("=" * 72)
print("6. DISTILLATIE")
print("=" * 72)
res = mccabe_thiele(alpha=2.5, xF=0.5, xD=0.95, xB=0.05, R=2.0, q=1.0)
check("McCabe-Thiele: plausibel N & voedingsschotel",
      8 <= res["N"] <= 12 and res["feed_stage"] is not None,
      f"N={res['N']}, feed={res['feed_stage']}, Rmin={res['Rmin']:.3f}")
# Smoker vs trapsgewijze telling op ÉÉN lijn (totale reflux: y=x)
N_smoker = smoker(alpha=2.5, m=1.0, b=0.0, x_from=0.95, x_to=0.05)
N_fenske = np.log((0.95/0.05)*(0.95/0.05)) / np.log(2.5)   # Fenske
check("Smoker (tot. reflux) = Fenske", abs(N_smoker - N_fenske) < 1e-9,
      f"Smoker={N_smoker:.4f}, Fenske={N_fenske:.4f}")
frac, xDavg = rayleigh(alpha=2.5, x0=0.5, x_end=0.2)
check("Rayleigh: massabalans & verrijking",
      0 < frac < 1 and xDavg > 0.5,
      f"W/W0={frac:.4f}, x̄_D={xDavg:.4f}")

print()
print("=" * 72)
print("7. BUISREACTOR (Danckwerts-BC's)")
print("=" * 72)
u_, D_, k_, L_ = 0.5, 0.01, 1.0, 1.0
z, C = axial_dispersion(u_, D_, k_, L_, 1.0, n=2001)
Ca = axial_dispersion_analytical(u_, D_, k_, L_, 1.0, z)
err = np.max(np.abs(C - Ca))
check("FD vs analytisch (Wehner-Wilhelm)", err < 5e-3,
      f"Pe={u_*L_/D_:.0f}, max|Δ|={err:.2e}, C_uit={C[-1]:.4f}")
check("Danckwerts-sprong aan inlaat: C(0) < C_in", C[0] < 1.0,
      f"C(0)={C[0]:.4f} (< 1 = correct)")

print()
print("=" * 72)
print("8. FVM: 1D CONVECTIE-DIFFUSIE")
print("=" * 72)
pars = dict(rho=1.0, u=2.5, gamma=0.1, L=1.0, phi0=1.0, phiL=0.0)
for scheme in ("cds", "upwind", "hybrid", "powerlaw"):
    e = {}
    for n_ in (40, 160):
        xg, phi = convection_diffusion(n=n_, scheme=scheme, **pars)
        e[n_] = np.max(np.abs(phi - analytical(x=xg, **pars)))
    tol = 0.12 if scheme == "upwind" else 0.05     # upwind = 1e orde
    check(f"schema '{scheme}': nauwkeurig én convergent", 
          e[40] < tol and e[160] < e[40] / 2.5,
          f"err(40)={e[40]:.4f} → err(160)={e[160]:.4f}")
# grensgeval: hoge Pe → CDS mag hier nog net; upwind blijft begrensd
xg, phi = convection_diffusion(n=10, scheme="upwind", rho=1.0, u=25.0,
                               gamma=0.1, L=1.0, phi0=1.0, phiL=0.0)
check("upwind begrensd bij Pe_cel≫2", (phi >= -1e-9).all() and (phi <= 1+1e-9).all(),
      f"min={phi.min():.3f}, max={phi.max():.3f}")
# symbolische afleiding reproduceert de leerboekcoëfficiënten
aW, aE, aP = symbolic_scheme()
F, D = sp.symbols("F D", positive=True)
check("symbolisch: aW=D+F/2, aE=D−F/2, aP=aW+aE",
      sp.simplify(aW-(D+F/2)) == 0 and sp.simplify(aE-(D-F/2)) == 0
      and sp.simplify(aP-(aW+aE)) == 0,
      f"aW={aW}, aE={aE}")

print()
print("=" * 72)
print("9. RIGOUREUZE KOLOM (Bubble Point / Wang-Henke, CMO)")
print("=" * 72)
from chemeng.rigorous import bubble_point_column
# benzeen / tolueen / p-xyleen (Antoine mmHg, °C)
antB = (6.90565, 1211.033, 220.79)
antT = (6.95464, 1344.8, 219.48)
antX = (6.99052, 1453.43, 215.31)
ants = [antB, antT, antX]
zf = [0.40, 0.35, 0.25]
res9 = bubble_point_column(ants, zf, F=100.0, q=1.0, N=12, feed_stage=6,
                           R=2.5, D=40.0, P=760.0, Tlo=40, Thi=180)
check("BP-kolom convergeert", res9["converged"], f"iters={res9['iters']}")
# componentmassabalans: D·x_D + B·x_B = F·z
mb = np.max(np.abs(40.0*res9["x_D"] + res9["B"]*res9["x_B"]
                   - 100.0*np.asarray(zf)))
check("componentmassabalans kolom", mb < 1e-6, f"max|Δ|={mb:.2e} kmol/h")
# sommatie & evenwicht per trap
sx = np.max(np.abs(res9["x"].sum(axis=1) - 1))
from chemeng.vle import antoine as _ant
KX = np.array([[ _ant(*a, res9["T"][j])/760.0 for a in ants]
               for j in range(12)])
bp = np.max(np.abs((KX*res9["x"]).sum(axis=1) - 1))
check("Σx=1 en ΣKx=1 op elke trap", sx < 1e-10 and bp < 1e-6,
      f"max|Σx−1|={sx:.1e}, max|ΣKx−1|={bp:.1e}")
# fysica: T stijgt van condensor naar reboiler; benzeen boven, xyleen onder
check("T-profiel monotoon stijgend", np.all(np.diff(res9["T"]) > 0),
      f"T: {res9['T'][0]:.1f} → {res9['T'][-1]:.1f} °C")
check("scheiding: benzeen→top, xyleen→bodem",
      res9["x_D"][0] > 0.85 and res9["x_B"][2] > 0.38,
      f"x_D(benz)={res9['x_D'][0]:.4f}, x_B(xyl)={res9['x_B'][2]:.4f}")
# meer trappen ⇒ scherpere scheiding (zelfde specificaties)
res9b = bubble_point_column(ants, zf, F=100.0, q=1.0, N=20, feed_stage=10,
                            R=2.5, D=40.0, P=760.0, Tlo=40, Thi=180)
check("N=12 → N=20 verhoogt topzuiverheid",
      res9b["converged"] and res9b["x_D"][0] > res9["x_D"][0],
      f"x_D(benz): {res9['x_D'][0]:.4f} → {res9b['x_D'][0]:.4f}")

print()
print("=" * 72)
print(f"RESULTAAT: {ok} PASS / {fail} FAIL")
print("=" * 72)
raise SystemExit(0 if fail == 0 else 1)
