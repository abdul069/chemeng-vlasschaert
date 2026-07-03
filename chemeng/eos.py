"""
eos.py — Kubische toestandsvergelijkingen
=========================================
Herimplementatie van Module 2 (ref. Elliott & Lira). Van der Waals,
Redlich-Kwong en Peng-Robinson in één klassenhiërarchie, met de
fysische wortelselectie (grootste reële wortel = damp, kleinste > b
= vloeistof) zoals in de originele Maxima/TrueBasic-programma's.
"""
from __future__ import annotations
import numpy as np

R = 8.314462618  # J/(mol·K)


class CubicEOS:
    """P = RT/(V−b) − a(T)/((V+e1·b)(V+e2·b))  — generieke vorm."""
    e1 = 0.0
    e2 = 0.0

    def __init__(self, Tc, Pc, omega=0.0):
        self.Tc, self.Pc, self.omega = Tc, Pc, omega

    # -- door subklassen in te vullen -------------------------------------
    def a(self, T):  # noqa: D401
        raise NotImplementedError

    def b(self):
        raise NotImplementedError

    # -- kern ---------------------------------------------------------------
    def Z_roots(self, T, P):
        """Compressibiliteitsfactoren uit de kubische vergelijking in Z."""
        A = self.a(T) * P / (R * T) ** 2
        B = self.b() * P / (R * T)
        e1, e2 = self.e1, self.e2
        c2 = (e1 + e2 - 1.0) * B - 1.0
        c1 = A + e1 * e2 * B**2 - (e1 + e2) * B * (B + 1.0)
        c0 = -(A * B + e1 * e2 * B**2 * (B + 1.0))
        roots = np.roots([1.0, c2, c1, c0])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        return real[real > B]           # fysisch: V > b  ⇔  Z > B

    def volumes(self, T, P):
        """(V_vloeistof, V_damp) in m³/mol; gelijk indien één wortel."""
        Z = self.Z_roots(T, P)
        V = Z * R * T / P
        return V.min(), V.max()

    def pressure(self, T, V):
        return (R * T / (V - self.b())
                - self.a(T) / ((V + self.e1 * self.b())
                               * (V + self.e2 * self.b())))


class VanDerWaals(CubicEOS):
    e1 = e2 = 0.0

    def a(self, T):
        return 27.0 * R**2 * self.Tc**2 / (64.0 * self.Pc)

    def b(self):
        return R * self.Tc / (8.0 * self.Pc)


class RedlichKwong(CubicEOS):
    e1, e2 = 1.0, 0.0

    def a(self, T):
        return 0.42748 * R**2 * self.Tc**2.5 / (self.Pc * np.sqrt(T))

    def b(self):
        return 0.08664 * R * self.Tc / self.Pc


class PengRobinson(CubicEOS):
    e1, e2 = 1.0 + np.sqrt(2.0), 1.0 - np.sqrt(2.0)

    def a(self, T):
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        alpha = (1.0 + kappa * (1.0 - np.sqrt(T / self.Tc))) ** 2
        return 0.45724 * R**2 * self.Tc**2 / self.Pc * alpha

    def b(self):
        return 0.07780 * R * self.Tc / self.Pc
