"""
vle.py — Damp-vloeistofevenwicht
================================
Herimplementatie van de VLE-laag uit Module 1/2: Antoine-dampspanning,
Txy-diagram (Raoult), en het azeotroop-bubblepointprogramma met het
2-parameter Margules-activiteitsmodel (origineel: TrueBasic/Maxima,
ref. Elliott & Lira).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def antoine(A, B, C, T):
    """log10(Psat) = A − B/(T + C).  Eenheden bepaald door de constanten."""
    return 10.0 ** (A - B / (T + C))


def margules2(x1, A12, A21):
    """2-parameter Margules: retourneert (γ1, γ2)."""
    x2 = 1.0 - x1
    g1 = np.exp(x2**2 * (A12 + 2.0 * (A21 - A12) * x1))
    g2 = np.exp(x1**2 * (A21 + 2.0 * (A12 - A21) * x2))
    return g1, g2


def bubble_T(x1, P, ant1, ant2, A12=0.0, A21=0.0, Tlo=250.0, Thi=450.0):
    """Kooktemperatuur van een binaire vloeistof bij druk P.

    Ideaal (Raoult) indien A12=A21=0, anders modified Raoult met
    Margules-2. Retourneert (T, y1).
    """
    g1, g2 = margules2(x1, A12, A21)

    def resid(T):
        return (x1 * g1 * antoine(*ant1, T)
                + (1.0 - x1) * g2 * antoine(*ant2, T) - P)

    T = brentq(resid, Tlo, Thi)
    y1 = x1 * g1 * antoine(*ant1, T) / P
    return T, y1


def txy(P, ant1, ant2, A12=0.0, A21=0.0, n=51, Tlo=250.0, Thi=450.0):
    """Txy-diagramgegevens: (x1, y1, T)."""
    x = np.linspace(0.0, 1.0, n)
    T = np.empty(n); y = np.empty(n)
    for i, xi in enumerate(x):
        T[i], y[i] = bubble_T(xi, P, ant1, ant2, A12, A21, Tlo, Thi)
    return x, y, T


def azeotrope(P, ant1, ant2, A12, A21, x_lo=1e-4, x_hi=1.0 - 1e-4,
              Tlo=250.0, Thi=450.0):
    """Zoekt het homogene azeotroop (y1 = x1). Retourneert (x_az, T_az)
    of None als er geen teken-wissel is (geen azeotroop)."""
    def f(x1):
        _, y1 = bubble_T(x1, P, ant1, ant2, A12, A21, Tlo, Thi)
        return y1 - x1
    flo, fhi = f(x_lo), f(x_hi)
    if flo * fhi > 0:
        return None
    x_az = brentq(f, x_lo, x_hi)
    T_az, _ = bubble_T(x_az, P, ant1, ant2, A12, A21, Tlo, Thi)
    return x_az, T_az
