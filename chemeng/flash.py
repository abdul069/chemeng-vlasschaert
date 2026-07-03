"""
flash.py — Isothermal flash (Rachford-Rice)
===========================================
Herimplementatie van het multicomponent-flashprogramma uit Module 1
(origineel: Newton-Raphson in TrueBasic/Maxima).

Rachford-Rice:  Σ zᵢ(Kᵢ−1) / (1 + ψ(Kᵢ−1)) = 0, met ψ = V/F.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def rachford_rice(z, K):
    """Retourneert (psi, x, y).

    psi = dampfractie V/F; x, y = vloeistof- en dampsamenstelling.
    Werpt ValueError bij één-fasige voeding (geen oplossing in (0,1)).
    """
    z = np.asarray(z, dtype=float)
    K = np.asarray(K, dtype=float)
    if not np.isclose(z.sum(), 1.0):
        raise ValueError("voedingssamenstelling z moet sommeren tot 1")

    def f(psi):
        return np.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0)))

    # Bestaansvoorwaarden: f(0) > 0 (er is damp) en f(1) < 0 (er is vloeistof)
    if f(0.0) <= 0.0:
        raise ValueError("voeding is onderkoelde vloeistof (geen flash)")
    if f(1.0) >= 0.0:
        raise ValueError("voeding is oververhitte damp (geen flash)")

    lo = 1.0 / (1.0 - K.max()) + 1e-12   # asymptootgrenzen
    hi = 1.0 / (1.0 - K.min()) - 1e-12
    psi = brentq(f, max(lo, 1e-12), min(hi, 1.0 - 1e-12), xtol=1e-14)
    x = z / (1.0 + psi * (K - 1.0))
    y = K * x
    return psi, x, y
