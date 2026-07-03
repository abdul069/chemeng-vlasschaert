"""
distillation.py — Binaire distillatie
=====================================
Herimplementatie van de kern van Module 1:

  - mccabe_thiele() : schotelbepaling met operating lines + q-lijn,
                      incl. optioneel diagram (origineel: TrueBasic
                      met eigen graphics-subs).
  - smoker()        : analytische Smoker-vergelijking (constante α).
  - rayleigh()      : differentiële batchdistillatie.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def eq_y(x, alpha):
    """Evenwichtslijn bij constante relatieve vluchtigheid α."""
    return alpha * x / (1.0 + (alpha - 1.0) * x)


def eq_x(y, alpha):
    """Inverse evenwichtslijn."""
    return y / (alpha - (alpha - 1.0) * y)


def mccabe_thiele(alpha, xF, xD, xB, R, q=1.0):
    """McCabe-Thiele-schotelconstructie.

    Retourneert dict met: N (theoretische trappen incl. reboiler),
    feed_stage, stages (lijst hoekpunten voor plot), snijpunt operating
    lines, en Rmin.
    """
    # q-lijn ∩ evenwichtslijn → Rmin
    if abs(q - 1.0) < 1e-12:
        x_q = xF
        y_q_eq = eq_y(xF, alpha)
    else:
        def g(x):
            return eq_y(x, alpha) - (q / (q - 1.0) * x - xF / (q - 1.0))
        x_q = brentq(g, 1e-9, 1.0 - 1e-9)
        y_q_eq = eq_y(x_q, alpha)
    Rmin = (xD - y_q_eq) / (y_q_eq - x_q)
    if R <= Rmin:
        raise ValueError(f"R={R} ≤ Rmin={Rmin:.4f}: onmogelijke specificatie")

    # snijpunt rectificatie- en q-lijn
    if abs(q - 1.0) < 1e-12:
        x_int = xF
        y_int = R / (R + 1.0) * xF + xD / (R + 1.0)
    else:
        x_int = ((xF / (q - 1.0) + xD / (R + 1.0))
                 / (q / (q - 1.0) - R / (R + 1.0)))
        y_int = R / (R + 1.0) * x_int + xD / (R + 1.0)

    def y_op(x):
        """Operating line: rectificatie boven het snijpunt, stripping eronder."""
        if x >= x_int:
            return R / (R + 1.0) * x + xD / (R + 1.0)
        m = (y_int - xB) / (x_int - xB)      # strippinglijn door (xB, xB)
        return xB + m * (x - xB)

    # trapconstructie vanaf (xD, xD) naar beneden
    stages = [(xD, xD)]
    x, y = xD, xD
    N = 0
    feed_stage = None
    while x > xB and N < 200:
        x_new = eq_x(y, alpha)               # horizontaal naar evenwichtslijn
        stages.append((x_new, y))
        N += 1
        if feed_stage is None and x_new < x_int:
            feed_stage = N
        if x_new <= xB:
            x = x_new
            break
        y_new = y_op(x_new)                  # verticaal naar operating line
        stages.append((x_new, y_new))
        x, y = x_new, y_new
    return {"N": N, "feed_stage": feed_stage, "Rmin": Rmin,
            "x_int": x_int, "y_int": y_int, "stages": stages}


def smoker(alpha, m, b, x_from, x_to):
    """Smoker-vergelijking: analytisch aantal trappen tussen x_from en x_to
    langs de operating line y = m·x + b, evenwicht y = αx/(1+(α−1)x).

    Volgt Smoker (1938): transformatie x = x* + k met k de wortel van
    m(α−1)k² + (m + b(α−1) − α)k + b = 0 in [0,1].
    """
    A_ = m * (alpha - 1.0)
    B_ = m + b * (alpha - 1.0) - alpha
    C_ = b
    disc = np.sqrt(B_**2 - 4.0 * A_ * C_)
    best = None
    for k in ((-B_ + disc) / (2 * A_), (-B_ - disc) / (2 * A_)):
        if not (-1e-12 <= k <= 1.0 + 1e-12):
            continue
        c = 1.0 + (alpha - 1.0) * k
        beta = m * c * (alpha - 1.0) / (alpha - m * c**2)   # Smoker 1938
        xa, xb = x_from - k, x_to - k
        arg = (xa / xb) * ((1.0 - beta * xb) / (1.0 - beta * xa))
        if arg > 0.0:
            N = np.log(arg) / np.log(alpha / (m * c**2))
            if np.isfinite(N) and N > 0 and (best is None or N < best):
                best = N
    if best is None:
        raise ValueError("geen geldige Smoker-wortel voor deze specificatie")
    return best


def rayleigh(alpha, x0, x_end):
    """Rayleigh-batchdistillatie bij constante α.

    ln(W/W0) = ∫_{x_end}^{x0} dx / (y*−x).
    Retourneert (W/W0, gemiddelde distillaatsamenstelling).
    """
    integrand = lambda x: 1.0 / (eq_y(x, alpha) - x)
    I, _ = quad(integrand, x_end, x0)
    frac_left = np.exp(-I)
    xD_avg = (x0 - frac_left * x_end) / (1.0 - frac_left)
    return frac_left, xD_avg
