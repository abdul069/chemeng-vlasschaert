"""
fvm.py — Finite Volume Method: 1D convectie-diffusie
====================================================
Herimplementatie van de kern van Module 5 (ref. Versteeg & Malalasekera /
Patankar). Het standaardprobleem:

    d(ρuφ)/dx = d(Γ dφ/dx)/dx,   φ(0)=φ0, φ(L)=φL

gediscretiseerd met vier klassieke schema's: CDS, upwind, hybrid en
power-law. De analytische oplossing dient als referentie.

symbolic_scheme() reproduceert de signatuur-aanpak van het archief:
de discretisatiecoëfficiënten worden SYMBOLISCH afgeleid met sympy
(zoals origineel met Maxima) en pas daarna numeriek gebruikt.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from .numerics import tdma


def _coeffs(scheme, F, Dd):
    """aW en aE per schema (uniform rooster, constante F en D)."""
    if scheme == "cds":
        aW = Dd + F / 2.0
        aE = Dd - F / 2.0
    elif scheme == "upwind":
        aW = Dd + max(F, 0.0)
        aE = Dd + max(-F, 0.0)
    elif scheme == "hybrid":
        aW = max(F, Dd + F / 2.0, 0.0)
        aE = max(-F, Dd - F / 2.0, 0.0)
    elif scheme == "powerlaw":
        P = F / Dd
        f = max(0.0, (1.0 - 0.1 * abs(P)) ** 5)
        aW = Dd * f + max(F, 0.0)
        aE = Dd * f + max(-F, 0.0)
    else:
        raise ValueError(scheme)
    return aW, aE


def convection_diffusion(rho, u, gamma, L, phi0, phiL, n=21,
                         scheme="cds"):
    """FVM-oplossing. Retourneert (x_celcentra, φ)."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    F = rho * u
    Dd = gamma / dx
    aW, aE = _coeffs(scheme, F, Dd)
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
    for i in range(n):
        a[i] = -aW
        c[i] = -aE
        b[i] = aW + aE + (F - F)         # continuïteit: Fe−Fw = 0
    # randcellen: halve cel naar de wand → wandcoëfficiënt met D_w = Γ/(dx/2)
    Dw = gamma / (dx / 2.0)
    if scheme == "cds":
        aWb, aEb = Dw + F, Dw - F        # wandflux met F op de wand
    else:
        aWb = Dw + max(F, 0.0)
        aEb = Dw + max(-F, 0.0)
    a[0] = 0.0
    b[0] = aWb + (-c[0]) + (F - F)
    d[0] = aWb * phi0
    c[-1] = 0.0
    b[-1] = (-a[-1]) + aEb
    d[-1] = aEb * phiL
    return x, tdma(a, b, c, d)


def analytical(rho, u, gamma, L, phi0, phiL, x):
    """Exacte oplossing: (φ−φ0)/(φL−φ0) = (exp(Pe·x/L)−1)/(exp(Pe)−1)."""
    Pe = rho * u * L / gamma
    x = np.asarray(x, dtype=float)
    if abs(Pe) < 1e-12:
        frac = x / L
    else:
        frac = (np.exp(Pe * x / L) - 1.0) / (np.exp(Pe) - 1.0)
    return phi0 + (phiL - phi0) * frac


def symbolic_scheme():
    """Symbolische afleiding van de CDS-coëfficiënten (Maxima-stijl).

    Vertrekt van de flux-balans over een controlecel en retourneert
    (aW, aE, aP) als sympy-uitdrukkingen — de werkwijze van het
    origineel: eerst de algebra, dan pas getallen.
    """
    F, D = sp.symbols("F D", positive=True)
    phiW, phiP, phiE = sp.symbols("phi_W phi_P phi_E")
    # vlakwaarden via lineaire interpolatie (CDS)
    phie = (phiP + phiE) / 2
    phiw = (phiW + phiP) / 2
    conv = F * phie - F * phiw
    diff = D * (phiE - phiP) - D * (phiP - phiW)
    balance = sp.expand(conv - diff)          # = 0
    aW = -sp.simplify(balance.coeff(phiW))
    aE = -sp.simplify(balance.coeff(phiE))
    aP = sp.simplify(balance.coeff(phiP))
    return aW, aE, aP
