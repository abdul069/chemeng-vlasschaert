"""
reactor.py — 1D buisreactor met axiale dispersie
================================================
Herimplementatie van het buisreactorprogramma uit Module 3
(origineel: TrueBasic, FDM):

    u·dC/dz = D·d²C/dz² − k·C     op 0 ≤ z ≤ L

met de DANCKWERTS-randvoorwaarden — de klassieke valkuil die in het
originele archief correct zat en hier behouden blijft:

    inlaat  : u·C_in = u·C(0) − D·dC/dz|₀      (gemengde/Robin-BC)
    uitlaat : dC/dz|_L = 0

Analytische referentieoplossing (eerste orde, gesloten vorm) is mee
geïmplementeerd voor verificatie.
"""
from __future__ import annotations
import numpy as np
from .numerics import tdma


def axial_dispersion(u, D, k, L, C_in, n=201):
    """FD-oplossing met Danckwerts-BC's. Retourneert (z, C)."""
    z = np.linspace(0.0, L, n)
    dz = z[1] - z[0]
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
    for i in range(1, n - 1):
        a[i] = D / dz**2 + u / (2.0 * dz)      # centraal in convectie
        b[i] = -2.0 * D / dz**2 - k
        c[i] = D / dz**2 - u / (2.0 * dz)
    # inlaat: u·C_in = u·C0 − D·(C1−C0)/dz  (voorwaartse differentie)
    b[0] = u + D / dz
    c[0] = -D / dz
    d[0] = u * C_in
    # uitlaat: (C_n − C_{n−1})/dz = 0
    a[-1] = -1.0
    b[-1] = 1.0
    d[-1] = 0.0
    return z, tdma(a, b, c, d)


def axial_dispersion_analytical(u, D, k, L, C_in, z):
    """Gesloten oplossing (Danckwerts 1953 / Wehner-Wilhelm) voor
    eerste-orde reactie met axiale dispersie."""
    Pe = u * L / D
    beta = np.sqrt(1.0 + 4.0 * k * D / u**2)
    lam1 = u / (2.0 * D) * (1.0 + beta)
    lam2 = u / (2.0 * D) * (1.0 - beta)
    # C = A·exp(λ1 z) + B·exp(λ2 z); BC's opleggen
    M = np.array([
        [u - D * lam1, u - D * lam2],
        [lam1 * np.exp(lam1 * L), lam2 * np.exp(lam2 * L)],
    ])
    rhs = np.array([u * C_in, 0.0])
    A, B = np.linalg.solve(M, rhs)
    return A * np.exp(lam1 * np.asarray(z)) + B * np.exp(lam2 * np.asarray(z))
