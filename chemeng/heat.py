"""
heat.py — Warmte- en massatransport (FDM/TDMA-reeks)
====================================================
Herimplementatie van Module 4 uit het archief (ref. Majumdar,
"Computational Methods for Heat and Mass Transfer").

  - steady_conduction_1d()   : 1D geleiding met bron, Dirichlet-randen
  - transient_conduction()   : impliciet schema, geometrie = 'plaat',
                               'cilinder' of 'bol'; de r=0-singulariteit
                               wordt behandeld via L'Hôpital + symmetrie
                               (ghost node), zoals in het origineel.
"""
from __future__ import annotations
import numpy as np
from .numerics import tdma

_GEOM = {"plaat": 0, "cilinder": 1, "bol": 2}


def steady_conduction_1d(k, L, n, T0, TL, q_gen=0.0):
    """-k·d²T/dx² = q_gen op [0,L], T(0)=T0, T(L)=TL. Retourneert (x, T)."""
    x = np.linspace(0.0, L, n)
    dx = x[1] - x[0]
    a = np.full(n, 1.0); b = np.full(n, -2.0); c = np.full(n, 1.0)
    d = np.full(n, -q_gen * dx * dx / k)
    a[0] = c[0] = 0.0; b[0] = 1.0; d[0] = T0
    a[-1] = c[-1] = 0.0; b[-1] = 1.0; d[-1] = TL
    return x, tdma(a, b, c, d)


def transient_conduction(geometry, alpha, R, n, dt, t_end, T_init,
                         T_surface):
    """Volledig impliciete 1D transiënte geleiding.

    ∂T/∂t = α · (1/r^m) ∂/∂r ( r^m ∂T/∂r ),  m = 0 (plaat), 1 (cil.), 2 (bol)

    In r=0 is de term (m/r)·∂T/∂r singulier; met symmetrie (∂T/∂r|₀ = 0)
    geeft L'Hôpital:  ∂T/∂t|₀ = α·(1+m)·∂²T/∂r²|₀, gediscretiseerd met
    ghost node T₋₁ = T₁.

    Retourneert (r, T_eind).
    """
    m = _GEOM[geometry]
    r = np.linspace(0.0, R, n)
    dr = r[1] - r[0]
    lam = alpha * dt / dr**2
    T = np.full(n, float(T_init))
    nsteps = int(round(t_end / dt))
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    # centrum (i=0): (1 + 2λ(1+m))·T0 − 2λ(1+m)·T1 = T0_oud
    b[0] = 1.0 + 2.0 * lam * (1 + m)
    c[0] = -2.0 * lam * (1 + m)
    # inwendige nodes
    for i in range(1, n - 1):
        ri = r[i]
        a[i] = -lam * (1.0 - m * dr / (2.0 * ri))
        b[i] = 1.0 + 2.0 * lam
        c[i] = -lam * (1.0 + m * dr / (2.0 * ri))
    # rand r=R: Dirichlet
    b[-1] = 1.0
    for _ in range(nsteps):
        d = T.copy()
        d[-1] = T_surface
        T = tdma(a, b, c, d)
    return r, T


def slab_analytical(alpha, L, x, t, T_init, T_surface, nterms=400):
    """Analytische reeksoplossing voor de HALVE plaat: symmetrievlak in
    x=0 (∂T/∂x=0), oppervlak x=L op T_surface — identiek probleem als
    transient_conduction('plaat'). Verificatiereferentie.

    θ/θᵢ = Σₙ 4(−1)ⁿ/((2n+1)π) · cos(λₙx) · exp(−αλₙ²t),
    λₙ = (2n+1)π/(2L).
    """
    x = np.asarray(x, dtype=float)
    theta = np.zeros_like(x)
    for k in range(nterms):
        lam = (2 * k + 1) * np.pi / (2.0 * L)
        theta += (4.0 * (-1.0) ** k / ((2 * k + 1) * np.pi)
                  * np.cos(lam * x) * np.exp(-alpha * lam**2 * t))
    return T_surface + (T_init - T_surface) * theta
