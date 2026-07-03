"""
rigorous.py — Rigoureuze multicomponent-distillatie (Bubble Point)
==================================================================
Herimplementatie van het Bubble Point-kolomprogramma uit Module 1
(Wang-Henke-formulering, ref. Seader/Henley; origineel TrueBasic).

Stage-by-stage MESH-vergelijkingen:
  M — componentbalansen per trap → tridiagonaal stelsel per component
      (opgelost met de eigen tdma uit numerics.py)
  E — evenwicht via K_i = Psat_i(T)/P (Raoult, ideaal)
  S — sommatie: Σx = 1 per trap (normalisatie)
  H — hier: constant molal overflow (CMO); energiebalans-update van V
      staat op de roadmap.

Conventie: trap 1 = totale condensor, trap N = reboiler.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from .numerics import tdma
from .vle import antoine


def _bubble_T(x, ants, P, Tlo, Thi):
    """Kooktemperatuur van een trap: Σ xᵢ·Ksatᵢ(T)/P = 1."""
    def f(T):
        return sum(xi * antoine(*a, T) for xi, a in zip(x, ants)) / P - 1.0
    return brentq(f, Tlo, Thi)


def bubble_point_column(ants, z, F, q, N, feed_stage, R, D, P,
                        Tlo=0.0, Thi=200.0, tol=1e-8, maxit=1000):
    """Wang-Henke Bubble Point-methode, CMO-variant.

    ants       : lijst Antoine-tupels (A,B,C) per component
    z          : voedingssamenstelling
    F, q       : voedingsdebiet en thermische kwaliteit
    N          : aantal evenwichtstrappen (incl. condensor + reboiler)
    feed_stage : voedingstrap (1-gebaseerd)
    R, D, P    : refluxverhouding, distillaatdebiet, kolomdruk

    Retourneert dict met T, x, y, L, V, x_D, x_B, iters, converged.
    """
    nc = len(z)
    z = np.asarray(z, dtype=float)
    if not np.isclose(z.sum(), 1.0):
        raise ValueError("z moet sommeren tot 1")
    B = F - D
    if B <= 0 or D <= 0:
        raise ValueError("0 < D < F vereist")
    fs = feed_stage - 1                      # 0-gebaseerd

    # CMO-debietprofielen -------------------------------------------------
    L = np.empty(N)                          # L[j] = vloeistof die trap j verlaat
    V = np.empty(N)                          # V[j] = damp die trap j verlaat
    for j in range(N):
        L[j] = R * D if j < fs else R * D + q * F
    L[N - 1] = B                             # reboiler: bodemproduct
    V[0] = 0.0                               # totale condensor
    for j in range(1, N):
        V[j] = (R + 1.0) * D if j <= fs else (R + 1.0) * D - (1.0 - q) * F
    if np.any(V[1:] <= 0) or np.any(L[:-1] < 0):
        raise ValueError("negatieve interne debieten: specificatie onhaalbaar")
    Fv = np.zeros(N); Fv[fs] = F
    U = np.zeros(N); U[0] = D                # zijafname = distillaat

    # initialisatie T-profiel: lineair tussen kookpunten licht/zwaar ------
    Tb = [_bubble_T(np.eye(nc)[i], ants, P, Tlo, Thi) for i in range(nc)]
    T = np.linspace(min(Tb), max(Tb), N)

    x = np.tile(z, (N, 1))
    for it in range(1, maxit + 1):
        K = np.array([[antoine(*ants[i], T[j]) / P for i in range(nc)]
                      for j in range(N)])   # K[j,i]
        # M-vergelijkingen: tridiagonaal per component --------------------
        for i in range(nc):
            a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
            for j in range(N):
                if j > 0:
                    a[j] = L[j - 1]
                b[j] = -(L[j] + U[j] + V[j] * K[j, i])
                if j < N - 1:
                    c[j] = V[j + 1] * K[j + 1, i]
                d[j] = -Fv[j] * z[i]
            x[:, i] = tdma(a, b, c, d)
        x = np.clip(x, 1e-30, None)
        x /= x.sum(axis=1, keepdims=True)    # S-vergelijkingen
        # E-vergelijkingen: nieuwe kooktemperaturen -----------------------
        T_new = np.array([_bubble_T(x[j], ants, P, Tlo, Thi)
                          for j in range(N)])
        dT = np.max(np.abs(T_new - T))
        T = T_new
        if dT < tol:
            K = np.array([[antoine(*ants[i], T[j]) / P for i in range(nc)]
                          for j in range(N)])
            y = K * x
            y /= y.sum(axis=1, keepdims=True)
            return {"T": T, "x": x, "y": y, "L": L, "V": V,
                    "x_D": x[0], "x_B": x[-1], "B": B,
                    "iters": it, "converged": True}
    return {"T": T, "x": x, "y": None, "L": L, "V": V,
            "x_D": x[0], "x_B": x[-1], "B": B,
            "iters": maxit, "converged": False}
