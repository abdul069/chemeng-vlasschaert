"""
numerics.py — Numerieke kern
============================
Moderne herimplementatie (van nul) van de solverlaag uit het archief
P. Vlasschaert (petervlaschemeng.weebly.com), oorspronkelijk in TrueBasic/Maxima.

Bevat:
  - tdma()            : Thomas-algoritme (tridiagonale stelsels) — de ruggengraat
                        van de hele warmte/massa-transportreeks (Majumdar).
  - newton_symbolic() : Newton-Raphson met SYMBOLISCH gegenereerde Jacobiaan.
                        Dit is de signatuur-workflow van het archief: in het
                        origineel leidde wxMaxima de Jacobiaan af en rekende
                        TrueBasic; hier doet sympy de afleiding en numpy het
                        rekenwerk.
  - sor()             : Successive Over-Relaxation voor 2D Laplace/Poisson
                        (vervangt de Jacobi/Gauss-Seidel/SOR-reeks).
"""
from __future__ import annotations
import numpy as np
import sympy as sp


def tdma(a, b, c, d):
    """Thomas-algoritme voor  a_i·x_{i-1} + b_i·x_i + c_i·x_{i+1} = d_i.

    a[0] en c[-1] worden genegeerd. O(n), stabiel voor diagonaal-dominante
    stelsels zoals ze uit impliciete FD-discretisatie komen.
    """
    a, b, c, d = map(lambda v: np.asarray(v, dtype=float).copy(), (a, b, c, d))
    n = len(d)
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def newton_symbolic(eqs, symbols, x0, tol=1e-12, maxit=100):
    """Newton-Raphson voor een stelsel f(x)=0 met symbolische Jacobiaan.

    eqs      : lijst sympy-uitdrukkingen (== 0)
    symbols  : lijst sympy-symbolen, zelfde volgorde als x0
    x0       : startvector

    Retourneert (x, n_iter, geconvergeerd).
    """
    F = sp.Matrix(eqs)
    J = F.jacobian(symbols)                    # symbolische afleiding
    f_num = sp.lambdify(symbols, F, "numpy")
    J_num = sp.lambdify(symbols, J, "numpy")

    x = np.asarray(x0, dtype=float)
    for it in range(1, maxit + 1):
        fv = np.asarray(f_num(*x), dtype=float).ravel()
        Jv = np.asarray(J_num(*x), dtype=float)
        dx = np.linalg.solve(Jv, -fv)
        x = x + dx
        if np.linalg.norm(dx, np.inf) < tol:
            return x, it, True
    return x, maxit, False


def sor(f_rhs, nx, ny, lx=1.0, ly=1.0, bc=None, omega=1.7,
        tol=1e-8, maxit=50_000):
    """SOR voor  ∇²u = f  op een rechthoek met Dirichlet-randen.

    bc : dict met 'left','right','bottom','top' (scalars of arrays).
    Retourneert (u, n_iter).
    """
    dx, dy = lx / (nx - 1), ly / (ny - 1)
    u = np.zeros((ny, nx))
    bc = bc or {}
    u[:, 0] = bc.get("left", 0.0)
    u[:, -1] = bc.get("right", 0.0)
    u[0, :] = bc.get("bottom", 0.0)
    u[-1, :] = bc.get("top", 0.0)
    X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))
    f = f_rhs(X, Y) if callable(f_rhs) else np.full((ny, nx), float(f_rhs))
    beta2 = (dx / dy) ** 2
    denom = 2.0 * (1.0 + beta2)
    for it in range(1, maxit + 1):
        err = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                unew = ((u[j, i - 1] + u[j, i + 1]
                         + beta2 * (u[j - 1, i] + u[j + 1, i])
                         - dx * dx * f[j, i]) / denom)
                du = omega * (unew - u[j, i])
                u[j, i] += du
                err = max(err, abs(du))
        if err < tol:
            return u, it
    return u, maxit
